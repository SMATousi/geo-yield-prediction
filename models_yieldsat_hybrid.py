# --------------------------------------------------------
# YieldSAT hybrid model (spec/yieldsat-improvement.md, solutions S4-S10).
#
# Predicts every cell of a 64x64 tile (full-coverage build) from:
#   * a per-cell point branch: a temporal transformer over each cell's own
#     24-slot history. With cell_fusion='early' (S4) every time step is the
#     concatenation of the cell's S2 bands, the field's weather and the cell's
#     static layers (DEM, terrain, soil; broadcast over time), each with its
#     validity mask, plus date features (input/early fusion, as the paper's
#     input-fusion LSTM). cell_fusion='s2' uses the S2 bands only.
#   * optional spatial context:
#       'local' (S6) three masked 3x3 conv blocks over the map of per-cell
#               embeddings (7x7-cell neighbourhood, all modalities);
#       'maps'  (S10) an image branch over S2 maps (cached DINO RGB tokens +
#               nine-band maps of K observations) and the DEM map, decoded to
#               64x64 features and gathered at each cell; dem_rgb_xattn (S9)
#               adds DEM<->RGB cross-attention inside that branch.
#   * optional level term (S5): prediction = tile level + cell residual; the
#     level head reads only season-wide inputs (weather series, crop, seeding
#     day of year) and is tied to the tile's mean target by an auxiliary loss.
# The loss is cell-weighted MSE over valid cells (like the point model).
# --------------------------------------------------------

import math

import torch
import torch.nn.functional as F
from torch import nn

from dataset.yieldsat_schema import CROPS
from models_multimodal_encoder import MaskedTemporalEncoder
from models_yieldsat_image import YieldSATImageModel

T = 24
N_STATIC = 1 + 5 + 48                           # DEM, terrain (aspect sin/cos), soil


def cell_weighted_mse(pred, target, valid):
    v = valid.float()
    err = torch.where(valid, (pred - torch.nan_to_num(target)) ** 2, torch.zeros_like(pred))
    return err.sum() / v.sum().clamp(min=1)


class CellEncoder(nn.Module):
    """Temporal transformer over each present cell's 24-slot sequence."""

    def __init__(self, fusion='early', adm=True, dim=96, depth=2, heads=4):
        super().__init__()
        self.fusion, self.adm = fusion, adm
        f = 12 + 1 + 3                                            # S2*m, slot mask, time
        if fusion == 'early' and adm:
            f += 4 + 4 + 2 * N_STATIC                             # weather*m, mask, static*m, mask
        self.inp = nn.Linear(f, dim)
        self.slot = nn.Parameter(torch.zeros(1, T, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.slot, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, dim_feedforward=2 * dim, dropout=0.0, activation='gelu',
                                       batch_first=True, norm_first=True),
            num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, b, present):
        """present (B, H, W) bool -> embeddings (N, dim) for present cells and
        the (b, h, w) index of each."""
        idx = present.nonzero(as_tuple=True)
        s2 = b['series'][idx]                                     # (N, T, 12)
        m = b['series_mask'][idx].float().unsqueeze(-1)           # (N, T, 1)
        days = b['series_days'][idx]                              # (N, T)
        doy = 2 * math.pi * ((b['seeding_doy'][idx[0]].unsqueeze(1) + days) % 365.2425) / 365.2425
        dated = (days != 0) | (m.squeeze(-1) > 0)
        parts = [s2 * m, m, (days / 365.0).unsqueeze(-1), torch.sin(doy).unsqueeze(-1),
                 torch.cos(doy).unsqueeze(-1)]
        valid = m.squeeze(-1) > 0
        if self.fusion == 'early' and self.adm:
            wm = b['weather_mask'][idx[0]]                        # (N, T, 4) field weather
            parts += [b['weather'][idx[0]] * wm, wm]
            stat = torch.cat([b['dem'], b['terrain'], b['soil']], 1)          # (B, 54, H, W)
            smask = torch.cat([b['dem_mask'], b['terrain_mask'], b['soil_mask']], 1)
            sv = stat.permute(0, 2, 3, 1)[idx]                    # (N, 54)
            sm = smask.permute(0, 2, 3, 1)[idx]
            st = torch.cat([sv * sm, sm], -1).unsqueeze(1).expand(-1, T, -1)
            parts.append(st)
            valid = valid | (wm.sum(-1) > 0)
        x = self.inp(torch.cat(parts, -1)) + self.slot
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1)
        pad = torch.cat([torch.zeros_like(valid[:, :1]), ~(valid & dated)], 1)
        h = self.blocks(x, src_key_padding_mask=pad)
        return self.norm(h[:, 0]), idx


class LocalContext(nn.Module):
    """S6: masked 3x3 conv blocks over the per-cell embedding map."""

    def __init__(self, dim, layers=3):
        super().__init__()
        self.convs = nn.ModuleList(nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim),
                                                 nn.GELU()) for _ in range(layers))

    def forward(self, emap, present):
        p = present.unsqueeze(1).to(emap.dtype)
        for conv in self.convs:
            emap = (emap + conv(emap)) * p
        return emap


class LevelHead(nn.Module):
    """S5: season-wide level from weather series, crop and seeding day."""

    def __init__(self, dim, use_weather=True):
        super().__init__()
        self.use_weather = use_weather
        if use_weather:
            self.weather = MaskedTemporalEncoder(4, dim, time_dim=3, max_len=T)
        self.crop = nn.Embedding(len(CROPS), dim)
        self.mlp = nn.Sequential(nn.Linear(dim + 2, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, b):
        h = self.crop(b['crop'])
        if self.use_weather:
            wm = b['weather_mask']
            h = h + self.weather(torch.cat([b['weather'] * wm, wm, b['weather_time']], -1))[:, 0]
        doy = 2 * math.pi * b['seeding_doy'] / 365.2425
        return self.mlp(torch.cat([h, torch.sin(doy)[:, None], torch.cos(doy)[:, None]], -1)).squeeze(1)


class YieldSATHybridModel(nn.Module):

    def __init__(self, inputs='s2_adm', cell_fusion='early', context='none', level_head=False,
                 level_weight=1.0, dem_rgb_xattn=False, dim=96, depth=2, heads=4, k_obs=4,
                 dino_dim=1024, map_dim=96):
        super().__init__()
        if context not in ('none', 'local', 'maps'):
            raise ValueError('context must be none, local or maps')
        adm = inputs == 's2_adm'
        self.cell = CellEncoder(cell_fusion, adm, dim, depth, heads)
        self.context = context
        ctx = 0
        if context == 'local':
            self.local = LocalContext(dim)
        elif context == 'maps':
            # S10: S2 maps (DINO RGB + nine bands) and the DEM map only
            self.maps = YieldSATImageModel(inputs='s2_adm' if adm else 's2', embed_dim=map_dim, k_obs=k_obs,
                                           dino_dim=dino_dim, num_latents=16, depth=2, num_heads=4,
                                           use_crop=False, adm_streams=('dem',) if adm else (),
                                           dem_rgb_xattn=dem_rgb_xattn and adm, modality_dropout=0.0)
            ctx = 32
        self.head = nn.Sequential(nn.LayerNorm(dim + ctx), nn.Linear(dim + ctx, dim), nn.GELU(),
                                  nn.Linear(dim, 1))
        self.level_weight = level_weight
        if level_head:
            self.level = LevelHead(dim, use_weather=adm)
        self.streams = ['series'] + (['weather', 'dem', 'terrain', 'soil'] if adm and cell_fusion == 'early'
                                     else []) + (['dino', 'spec'] + (['dem'] if adm else []) if context == 'maps'
                                                 else [])
        self.config = dict(arch='hybrid', inputs=inputs, cell_fusion=cell_fusion, context=context,
                           level_head=level_head, level_weight=level_weight, dem_rgb_xattn=dem_rgb_xattn,
                           dim=dim, depth=depth, heads=heads, k_obs=k_obs, map_dim=map_dim)

    def forward(self, b, apply_dropout=True, return_level=False):
        present = b['cell_present'] > 0
        e, idx = self.cell(b, present)                            # (N, dim)
        if self.context == 'local':
            B, H, W = present.shape
            emap = e.new_zeros(B, H, W, e.shape[1])
            emap[idx] = e
            e = self.local(emap.permute(0, 3, 1, 2), present).permute(0, 2, 3, 1)[idx]
        elif self.context == 'maps':
            fmap, _ = self.maps.features(b, apply_dropout)        # (B, 32, H, W)
            e = torch.cat([e, fmap.permute(0, 2, 3, 1)[idx]], -1)
        r = self.head(e).squeeze(-1)                              # (N,)
        level = None
        if hasattr(self, 'level'):
            level = self.level(b)
            r = r + level[idx[0]]
        pred = present.new_zeros(present.shape, dtype=r.dtype)
        pred[idx] = r
        return (pred, level) if return_level else pred

    def loss(self, b, apply_dropout=True):
        pred, level = self.forward(b, apply_dropout, return_level=True)
        loss = cell_weighted_mse(pred, b['target'], b['target_valid'])
        if level is not None:
            v = b['target_valid'].float()
            n = v.flatten(1).sum(1)
            mean_t = (torch.nan_to_num(b['target']) * v).flatten(1).sum(1) / n.clamp(min=1)
            has = n > 0
            if has.any():
                loss = loss + self.level_weight * ((level - mean_t) ** 2)[has].mean()
        return pred, loss
