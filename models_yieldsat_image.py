# --------------------------------------------------------
# YieldSAT image model (YI-02/YI-03; spec/yieldsat-image-training.md, plan v1).
#
# YI-02 (this part): the frozen optical backbone. Meta DINOv3 ViT-L/16
# pretrained on SAT-493M, pinned by Hugging Face revision, never trained. RGB
# (S2 B04/B03/B02) is converted explicitly: L2A digital number / 10000 ->
# reflectance -> true-colour scaling reflectance/0.3 clipped to [0,1] (clip
# fraction recorded) -> SAT-493M mean/std; missing pixels are set to the mean
# (0 after normalization) and carried as masks. Only spatial patch tokens are
# used (4x4 for a 64x64 tile), never only the class token. Because the backbone
# is frozen, its tokens are precomputed once per (tile, slot) by
# yieldsat_image_dino_cache.py; the cache key holds the revision and a hash of
# the preprocessing.
#
# YI-03: YieldSATImageModel. One encoder per modality, each emitting D-wide
# tokens on the tile's 4x4 patch grid (plus a 64/32/16/8 skip pyramid for the
# dense streams):
#   dino     cached frozen DINOv3 patch tokens -> LN -> Linear  (K x 16)
#   spec     nine extra S2 bands, masked conv pyramid per obs   (K x 16)
#   weather  point MaskedTemporalEncoder over the tile series   (1)
#   dem      masked conv pyramid                                (16)
#   terrain  masked conv pyramid (aspect as sin/cos)            (16)
#   soil     per-cell depth-aware MLP -> masked conv pyramid    (16)
#   crop     embedding                                          (1)
# Optical tokens add the shared observation-date encoding. The point model's
# LatentFusionTransformer (Perceiver) reads ALL tokens with per-token masks
# (missing observations, empty patches, absent or dropped streams). A
# Perceiver-IO decoder turns the latents into a 4x4 map via 16 learned spatial
# queries, then four x2 U-Net stages with the skip pyramids reach 64x64.
# --------------------------------------------------------

import hashlib
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dataset.yieldsat_schema import CROPS, SOIL_DEPTHS, SOIL_PROPERTIES
from models_latent_fusion import LatentFusionTransformer
from models_multimodal_encoder import MaskedTemporalEncoder
from models_yieldsat import DateEncoding

DINO_REPO = 'facebook/dinov3-vitl16-pretrain-sat493m'
DINO_REVISION = 'f692fa42da72c6797b67cd73494a168d1120d3ee'
SAT_MEAN = (0.430, 0.411, 0.296)
SAT_STD = (0.213, 0.156, 0.143)
REFLECTANCE_SCALE = 10000.0
TRUE_COLOUR_MAX = 0.3
PATCH = 16


def preprocessing_spec(input_size=64):
    return {'rgb_bands': ['B04', 'B03', 'B02'], 'reflectance_scale': REFLECTANCE_SCALE,
            'true_colour_max': TRUE_COLOUR_MAX, 'clip': [0.0, 1.0], 'mean': list(SAT_MEAN),
            'std': list(SAT_STD), 'missing_fill': 'mean', 'input_size': input_size,
            'tokens': 'patch tokens only (no class/register tokens)'}


def preprocessing_hash(spec=None):
    spec = spec or preprocessing_spec()
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def rgb_to_dino_input(rgb_dn, valid):
    """rgb_dn (N, H, W, 3) L2A digital numbers in B04/B03/B02 order, valid
    (N, H, W) bool -> (N, 3, H, W) float32 normalized input, clip fraction per
    image (share of valid RGB values outside [0, 1] before clipping)."""
    x = np.asarray(rgb_dn, dtype=np.float32) / REFLECTANCE_SCALE / TRUE_COLOUR_MAX
    v = np.asarray(valid, dtype=bool)[..., None] & np.isfinite(x)
    outside = ((x < 0) | (x > 1)) & v
    clip_frac = outside.sum(axis=(1, 2, 3)) / np.maximum(v.sum(axis=(1, 2, 3)), 1)
    x = np.clip(np.nan_to_num(x), 0.0, 1.0)
    mean = np.asarray(SAT_MEAN, np.float32)
    std = np.asarray(SAT_STD, np.float32)
    x = np.where(v, (x - mean) / std, 0.0)                     # missing -> mean -> 0
    return np.ascontiguousarray(np.moveaxis(x, -1, 1), dtype=np.float32), clip_frac


class FrozenDino(nn.Module):
    """DINOv3 SAT backbone; returns patch tokens (N, (H/16)*(W/16), hidden)."""

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = model.config
        self.hidden_size = cfg.hidden_size
        self.n_skip = 1 + getattr(cfg, 'num_register_tokens', 0)     # class + registers

    @classmethod
    def from_pretrained(cls, name_or_path=DINO_REPO, revision=DINO_REVISION, token=None,
                        cache_dir=None):
        from transformers import AutoModel
        model = AutoModel.from_pretrained(name_or_path, revision=revision, token=token,
                                          cache_dir=cache_dir)
        return cls(model)

    def train(self, mode=True):
        super().train(False)                                   # always eval: frozen
        return self

    @torch.no_grad()
    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        return out.last_hidden_state[:, self.n_skip:]


class StubDino(nn.Module):
    """Deterministic stand-in with the FrozenDino interface (tests, and
    development before checkpoint access): a fixed random patchify conv."""

    def __init__(self, hidden_size=1024, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.proj = nn.Conv2d(3, hidden_size, PATCH, PATCH)
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(self.proj.weight.shape, generator=g) * 0.02)
            self.proj.bias.zero_()
        self.hidden_size = hidden_size
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, pixel_values):
        return self.proj(pixel_values).flatten(2).transpose(1, 2)


# ---- YI-03: image model --------------------------------------------------------

GRID = 4                                   # 64 / 16 patch grid
STREAMS_S2 = ('dino', 'spec')
STREAMS_ADM = ('weather', 'dem', 'terrain', 'soil')
PYRAMID = (32, 48, 64, 96)                 # channels at 64, 32, 16, 8


def _block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.GroupNorm(8, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, 1, 1), nn.GroupNorm(8, cout), nn.GELU())


def patch_valid(mask):
    """(N, C, 64, 64) cell mask -> (N, 16) bool: patch has any valid value."""
    return F.adaptive_max_pool2d(mask.amax(1, keepdim=True), GRID).flatten(1) > 0


class MaskedConvPyramid(nn.Module):
    """[value*mask, mask] -> 64/32/16/8 skip features and 4x4 D-wide tokens.
    Each level is zeroed where no input cell under it is valid."""

    def __init__(self, in_ch, embed_dim, widths=PYRAMID, stem_in=None):
        super().__init__()
        self.stem = _block(stem_in or 2 * in_ch, widths[0])
        self.downs = nn.ModuleList(_block(widths[i - 1], widths[i], 2) for i in range(1, len(widths)))
        self.to_tokens = nn.Sequential(nn.Conv2d(widths[-1], embed_dim, 2, 2), nn.GroupNorm(1, embed_dim))

    def forward(self, x, mask):
        """x (N, Cin, 64, 64) already masked/packed, mask (N, 1, 64, 64)."""
        feats = [self.stem(x)]
        for d in self.downs:
            feats.append(d(feats[-1]))
        feats = [f * F.adaptive_max_pool2d(mask, f.shape[-1]) for f in feats]
        tok = self.to_tokens(feats[-1]).flatten(2).transpose(1, 2)         # (N, 16, D)
        return tok, feats


class DepthAwareSoil(nn.Module):
    """Per cell: a shared MLP per depth on [8 properties * mask, mask] plus a
    learned depth embedding, depth features concatenated -> 32 channels."""

    def __init__(self, hidden=16, out=32):
        super().__init__()
        self.P, self.Dp = len(SOIL_PROPERTIES), len(SOIL_DEPTHS)
        self.mlp = nn.Sequential(nn.Conv2d(2 * self.P, hidden, 1), nn.GELU(), nn.Conv2d(hidden, hidden, 1))
        self.depth = nn.Parameter(torch.zeros(1, self.Dp, hidden, 1, 1))
        self.out = nn.Sequential(nn.Conv2d(self.Dp * hidden, out, 1), nn.GELU())
        nn.init.trunc_normal_(self.depth, std=0.02)

    def forward(self, soil, mask):
        n, _, h, w = soil.shape
        v = (soil * mask).view(n, self.P, self.Dp, h, w).transpose(1, 2)    # (n, depth, prop, h, w)
        m = mask.view(n, self.P, self.Dp, h, w).transpose(1, 2)
        z = self.mlp(torch.cat([v, m], 2).flatten(0, 1)).view(n, self.Dp, -1, h, w) + self.depth
        z = z * (m.amax(2, keepdim=True) > 0)                                # absent depth -> 0
        return self.out(z.flatten(1, 2))


class SeriesEncoder(nn.Module):
    """v2 per-pixel temporal branch: every cell's 24-slot S2 series
    ([bands*mask, slot mask, days since seeding / 365, sin/cos day of year])
    -> two temporal convolutions -> masked attention pooling over observed
    slots -> ``out`` channels per cell (zero where nothing was observed)."""

    def __init__(self, bands=12, hidden=32, out=32):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(bands + 4, hidden, 3, padding=1), nn.GELU(),
                                  nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU())
        self.score = nn.Conv1d(hidden, 1, 1)
        self.proj = nn.Linear(hidden, out)

    def forward(self, series, mask, days, seeding_doy):
        """series (B,H,W,T,C), mask (B,H,W,T), days (B,H,W,T), seeding_doy (B,)
        -> (B, out, H, W)."""
        B, H, W, T, C = series.shape
        m = mask.float().unsqueeze(-1)                                        # (B,H,W,T,1)
        doy = 2 * math.pi * ((seeding_doy.view(B, 1, 1, 1) + days) % 365.2425) / 365.2425
        x = torch.cat([series * m, m, (days / 365.0).unsqueeze(-1),
                       torch.sin(doy).unsqueeze(-1) * m, torch.cos(doy).unsqueeze(-1) * m], -1)
        x = x.reshape(B * H * W, T, C + 4).transpose(1, 2)                    # (N, C+4, T)
        h = self.conv(x)                                                      # (N, hidden, T)
        valid = m.reshape(B * H * W, T) > 0
        logits = self.score(h).squeeze(1).masked_fill(~valid, float('-inf'))
        any_valid = valid.any(1, keepdim=True)
        w = torch.softmax(torch.where(any_valid, logits, torch.zeros_like(logits)), 1) * any_valid
        pooled = (h * w.unsqueeze(1)).sum(-1)                                # (N, hidden)
        out = self.proj(pooled) * any_valid
        return out.view(B, H, W, -1).permute(0, 3, 1, 2)


class SpatialQueryDecoder(nn.Module):
    """Perceiver-IO read-out: 16 learned 4x4 queries cross-attend to the latents."""

    def __init__(self, embed_dim, num_heads, layers=2):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(1, GRID * GRID, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.attn = nn.ModuleList(nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
                                  for _ in range(layers))
        self.qn = nn.ModuleList(nn.LayerNorm(embed_dim) for _ in range(layers))
        self.mlp = nn.ModuleList(nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 4 * embed_dim),
                                               nn.GELU(), nn.Linear(4 * embed_dim, embed_dim))
                                 for _ in range(layers))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, latents):
        q = self.queries.expand(latents.shape[0], -1, -1)
        for attn, qn, mlp in zip(self.attn, self.qn, self.mlp):
            q = q + attn(qn(q), latents, latents)[0]
            q = q + mlp(q)
        q = self.norm(q)
        return q.transpose(1, 2).reshape(q.shape[0], -1, GRID, GRID)        # (B, D, 4, 4)


class YieldSATImageModel(nn.Module):
    """Dense 64x64 yield from the YieldSAT image streams (see header)."""

    def __init__(self, inputs='s2_adm', embed_dim=192, k_obs=4, dino_dim=1024, num_latents=32,
                 depth=4, num_heads=4, cross_attn_layers=2, modality_embed=32,
                 modality_dropout=0.1, use_dino=True, use_crop=True, decoder_layers=2,
                 use_series=False, level_head=False, level_weight=1.0):
        super().__init__()
        if inputs not in ('s2', 's2_adm'):
            raise ValueError('inputs must be s2 or s2_adm')
        D = embed_dim
        self.streams = (('dino',) if use_dino else ()) + ('spec',) + (
            ('series',) if use_series else ()) + (STREAMS_ADM if inputs == 's2_adm' else ())
        self.dense = tuple(s for s in self.streams if s in ('spec', 'series', 'dem', 'terrain', 'soil'))
        self.k = k_obs
        self.modality_dropout = modality_dropout
        self.use_crop = use_crop
        self.date_enc = DateEncoding(D)
        if use_dino:
            self.dino_proj = nn.Sequential(nn.LayerNorm(dino_dim), nn.Linear(dino_dim, D))
        self.spec_enc = MaskedConvPyramid(9, D)
        if use_series:                    # v2: full per-pixel S2 time series
            self.series_cell = SeriesEncoder()
            self.series_enc = MaskedConvPyramid(0, D, stem_in=32)
        if inputs == 's2_adm':
            self.weather_enc = MaskedTemporalEncoder(4, D, time_dim=3, max_len=24)
            self.dem_enc = MaskedConvPyramid(1, D)
            self.terrain_enc = MaskedConvPyramid(5, D)
            self.soil_cell = DepthAwareSoil()
            self.soil_enc = MaskedConvPyramid(0, D, stem_in=32)
        if use_crop:
            self.crop_embed = nn.Embedding(len(CROPS), D)
        layout = {'dino': (GRID * GRID, k_obs), 'spec': (GRID * GRID, k_obs), 'weather': (1, 1),
                  'series': (GRID * GRID, 1),
                  'dem': (GRID * GRID, 1), 'terrain': (GRID * GRID, 1), 'soil': (GRID * GRID, 1)}
        mods = {s: {'spatial': layout[s][0], 'temporal': layout[s][1]} for s in self.streams}
        if use_crop:
            mods['crop'] = {'spatial': 1, 'temporal': 1}
        self.fusion = LatentFusionTransformer(
            embed_dim=D, num_latents=num_latents, depth=depth, num_heads=num_heads, modalities=mods,
            modality_embed=modality_embed, cross_attn_layers=cross_attn_layers, norm_first=True,
            input_norm=True, latent_init='trunc_normal')
        self.decoder = SpatialQueryDecoder(D, num_heads, decoder_layers)
        skip_ch = [w * len(self.dense) for w in PYRAMID]                    # 64, 32, 16, 8
        ups, cin = [], D
        for res, cout in zip((3, 2, 1, 0), (128, 96, 64, 32)):
            ups.append(_block(cin + skip_ch[res], cout))
            cin = cout
        self.ups = nn.ModuleList(ups)
        self.head = nn.Conv2d(32, 1, 1)
        self.level_weight = level_weight
        if level_head:                    # v2: tile level from season-wide latents
            self.level = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Linear(D, 1))
        self.config = dict(inputs=inputs, embed_dim=D, k_obs=k_obs, dino_dim=dino_dim,
                           num_latents=num_latents, depth=depth, num_heads=num_heads,
                           cross_attn_layers=cross_attn_layers, modality_embed=modality_embed,
                           modality_dropout=modality_dropout, use_dino=use_dino, use_crop=use_crop,
                           decoder_layers=decoder_layers, use_series=use_series,
                           level_head=level_head, level_weight=level_weight)

    def _obs_dates(self, b):
        return self.date_enc(b['obs_time'][..., 0] * 365.0, b['obs_valid'] > 0)   # (B, K, D)

    def encode(self, b, apply_dropout=True):
        """-> tokens {stream: (B, L, D)}, token masks {stream: (B, L) bool},
        skips {stream: [64, 32, 16, 8 maps]}."""
        B, K = b['obs_valid'].shape
        tok, tmask, skips = {}, {}, {}
        dates = self._obs_dates(b).unsqueeze(2)                               # (B, K, 1, D)
        obs = b['obs_valid'] > 0
        if 'dino' in self.streams:
            x = self.dino_proj(b['dino'].float()) + dates                     # (B, K, 16, D)
            rgb_patch = patch_valid(b['rgb_mask'].flatten(0, 1).unsqueeze(1)).view(B, K, -1)
            tok['dino'] = x.flatten(1, 2)
            tmask['dino'] = ((b['dino_valid'] > 0)[..., None] & rgb_patch).flatten(1)
        if 'spec' in self.streams:
            v, m = b['spec'].flatten(0, 1), b['spec_mask'].flatten(0, 1)      # (B*K, 9, 64, 64)
            t, f = self.spec_enc(torch.cat([v * m, m], 1), m.amax(1, keepdim=True))
            tok['spec'] = (t.view(B, K, -1, t.shape[-1]) + dates).flatten(1, 2)
            tmask['spec'] = (obs[..., None] & patch_valid(m).view(B, K, -1)).flatten(1)
            w = obs.float().view(B, K, 1, 1, 1)
            skips['spec'] = [(g.view(B, K, *g.shape[1:]) * w).sum(1) / w.sum(1).clamp(min=1) for g in f]
        if 'series' in self.streams:
            m = b['series_mask']                                              # (B, H, W, T)
            cell = self.series_cell(b['series'], m, b['series_days'], b['seeding_doy'])
            observed = m.amax(-1).unsqueeze(1)                                # (B, 1, H, W)
            tok['series'], skips['series'] = self.series_enc(cell, observed)
            tmask['series'] = patch_valid(observed)
        if 'weather' in self.streams:
            wm = b['weather_mask']
            x = torch.cat([b['weather'] * wm, wm, b['weather_time']], -1)
            tok['weather'] = self.weather_enc(x)
            tmask['weather'] = self.weather_enc.token_mask(x)
        for name, enc in (('dem', 'dem_enc'), ('terrain', 'terrain_enc')):
            if name in self.streams:
                v, m = b[name], b[name + '_mask']
                tok[name], skips[name] = getattr(self, enc)(torch.cat([v * m, m], 1), m.amax(1, keepdim=True))
                tmask[name] = patch_valid(m)
        if 'soil' in self.streams:
            m = b['soil_mask']
            cell = self.soil_cell(b['soil'], m)
            tok['soil'], skips['soil'] = self.soil_enc(cell, m.amax(1, keepdim=True))
            tmask['soil'] = patch_valid(m)
        if self.training and apply_dropout and self.modality_dropout > 0:
            self._drop_streams(tmask, skips, B)
        if self.use_crop:
            tok['crop'] = self.crop_embed(b['crop']).unsqueeze(1)
            tmask['crop'] = torch.ones(B, 1, dtype=torch.bool, device=b['crop'].device)
        return tok, tmask, skips

    def _drop_streams(self, tmask, skips, B):
        """Drop whole streams per sample (never all of a sample's streams)."""
        names = list(tmask)
        dev = tmask[names[0]].device
        drop = torch.rand(B, len(names), device=dev) < self.modality_dropout
        has = torch.stack([tmask[n].any(1) for n in names], 1)
        keep_one = (has & ~drop).any(1)
        drop[~keep_one] = False
        for j, n in enumerate(names):
            tmask[n] = tmask[n] & ~drop[:, j:j + 1]
            if n in skips:
                k = (~drop[:, j]).float().view(B, 1, 1, 1)
                skips[n] = [f * k for f in skips[n]]

    def forward(self, b, apply_dropout=True, return_level=False):
        """-> standardized yield (B, 64, 64) for every cell. With the level
        head (v2): tile level + residual map centered over present cells."""
        tok, tmask, skips = self.encode(b, apply_dropout)
        latents = self.fusion(tok, token_mask=tmask)
        x = self.decoder(latents)                                             # (B, D, 4, 4)
        for up, res in zip(self.ups, (3, 2, 1, 0)):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = up(torch.cat([x] + [skips[s][res] for s in self.dense], 1))
        dense = self.head(x).squeeze(1)
        if not hasattr(self, 'level'):
            return (dense, None) if return_level else dense
        level = self.level(latents.mean(1)).squeeze(1)                       # (B,)
        p = b['cell_present'].float()
        center = (dense * p).flatten(1).sum(1) / p.flatten(1).sum(1).clamp(min=1)
        pred = level.view(-1, 1, 1) + dense - center.view(-1, 1, 1)
        return (pred, level) if return_level else pred

    def loss(self, b, apply_dropout=True):
        """Tile-balanced MSE on valid cells of the standardized target (+ the
        level loss: tile level vs the tile's mean valid target, v2)."""
        pred, level = self.forward(b, apply_dropout, return_level=True)
        loss = tile_balanced_mse(pred, b['target'], b['target_valid'])
        if level is not None:
            v = b['target_valid'].float()
            n = v.flatten(1).sum(1)
            mean_t = (torch.nan_to_num(b['target']) * v).flatten(1).sum(1) / n.clamp(min=1)
            has = n > 0
            if has.any():
                loss = loss + self.level_weight * ((level - mean_t) ** 2)[has].mean()
        return pred, loss


def tile_balanced_mse(pred, target, valid):
    v = valid.float()
    n = v.flatten(1).sum(1)
    err = torch.where(valid, (pred - torch.nan_to_num(target)) ** 2, torch.zeros_like(pred))
    per_tile = err.flatten(1).sum(1) / n.clamp(min=1)
    has = n > 0
    return per_tile[has].mean() if has.any() else pred.sum() * 0.0
