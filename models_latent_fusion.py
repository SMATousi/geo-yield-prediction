# --------------------------------------------------------
# Perceiver-style latent geospatial-temporal fusion backbone.
#
# Adapted from danfenghong/IEEE_TPAMI_SpectralGPT (models_mae_spectral.py,
# sep_pos_embed) to the MMST-ViT plain-PyTorch layout. This is the unified
# latent fusion backbone (gap g4): it consumes the per-modality token sets
# produced by the dedicated native-resolution encoders, gives every token a
# separable spatial + temporal positional embedding plus a modality-identity
# embedding, and fuses them through a Perceiver-style latent bottleneck
# (cross-attention between learned query tokens and the concatenated modality
# tokens, followed by self-attention on the latents).
#
# The token sequence is laid out as (spatial_grid x temporal_frames): the
# spatial embedding is tiled across all frames and the temporal embedding is
# repeated across all spatial positions, then summed so every token carries
# both a spatial-coordinate signal and an acquisition-time signal. A
# modality-identity embedding is concatenated (as in the group-channels model)
# so tokens from DEM, soil, SAR, and optical are distinguishable in the shared
# transformer. The gather-by-ids_keep logic keeps positional embeddings aligned
# when tokens are masked/dropped, which is useful for the missing-modality
# case.
# --------------------------------------------------------

import numpy as np
import torch
from torch import nn

# Multi-scale feature-extraction layer indices for a given transformer depth.
# Adapted from opengeos/geoai (dinov3_finetune.py, ``_get_extraction_layers``)
# to the MMST-ViT naming conventions. The fusion backbone calls this on its
# depth to obtain the four evenly-spaced intermediate layer indices whose
# outputs become the multiscale patch-token set that is projected into the
# common latent dimension and fused (e.g. via the DPT head or cross-attention
# queries), letting high-resolution terrain tokens coexist with coarse
# soil/climate tokens.
_EXTRACTION_LAYERS = {
    12: [2, 5, 8, 11],   # ViT-S
    24: [5, 11, 17, 23],  # ViT-L
    32: [7, 15, 23, 31],  # ViT-H / 7B
    40: [9, 19, 29, 39],  # ViT-g
}


def get_extraction_layers(num_blocks):
    """Return the intermediate layer indices for a given transformer depth.

    Returns a list of four layer indices (0-indexed) used for multi-scale
    feature extraction from the fusion backbone. For depths in the known
    table the canonical ViT indices are returned; for any other depth the
    four evenly-spaced intermediate indices are computed so the fusion
    backbone still emits a multiscale token set.

    Raises:
        ValueError: If *num_blocks* is not a positive integer.
    """
    if num_blocks in _EXTRACTION_LAYERS:
        return _EXTRACTION_LAYERS[num_blocks]
    if not isinstance(num_blocks, int) or num_blocks < 1:
        raise ValueError(
            f"Unsupported number of transformer blocks: {num_blocks}. "
            f"Supported values: {sorted(_EXTRACTION_LAYERS.keys())}."
        )
    # fallback: four evenly-spaced intermediate layer indices
    return [int(round((num_blocks - 1) * i / 3)) for i in range(4)]


def get_1d_sincos_pos_embed(embed_dim, pos):
    """1D sincos positional embedding from a numpy grid (source helper)."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)[:, None]
    out = np.einsum('nd,d->nd', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class SeparablePosEmbed(nn.Module):
    """Separable spatial + temporal positional embedding.

    Adapted from SpectralGPT's ``sep_pos_embed``. The token sequence is laid
    out as (spatial_grid x temporal_frames): the spatial embedding is tiled
    across all frames and the temporal embedding is repeated across all spatial
    positions, then summed so every token carries both a spatial-coordinate
    signal and an acquisition-time signal.

    ``gather`` (the source's gather-by-ids_keep) keeps the positional
    embeddings aligned when a subset of tokens is kept after masking/dropping,
    which is useful for the missing-modality case.
    """

    def __init__(self, spatial_dim, temporal_dim, embed_dim):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.embed_dim = embed_dim
        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, spatial_dim, embed_dim))
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, temporal_dim, embed_dim))
        self._init_sincos()

    def _init_sincos(self):
        spatial = get_1d_sincos_pos_embed(
            self.embed_dim, np.arange(self.spatial_dim, dtype=np.float32))
        temporal = get_1d_sincos_pos_embed(
            self.embed_dim, np.arange(self.temporal_dim, dtype=np.float32))
        self.pos_embed_spatial.data.copy_(
            torch.from_numpy(spatial).float().unsqueeze(0))
        self.pos_embed_temporal.data.copy_(
            torch.from_numpy(temporal).float().unsqueeze(0))

    def forward(self, ids_keep=None):
        # spatial tiled across all frames + temporal repeated across all
        # spatial positions -> (1, spatial*temporal, embed_dim)
        pos_embed = self.pos_embed_spatial.repeat(1, self.temporal_dim, 1) + \
            torch.repeat_interleave(
                self.pos_embed_temporal, self.spatial_dim, dim=1)
        pos_embed = pos_embed.expand(1, -1, -1)
        if ids_keep is not None:
            pos_embed = pos_embed.expand(ids_keep.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed, dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]))
        return pos_embed


class LatentFusionTransformer(nn.Module):
    """Perceiver-style latent geospatial-temporal fusion backbone.

    Consumes a dict of per-modality token sets (each ``(B, L_m, embed_dim)``)
    produced by the dedicated native-resolution encoders. Each modality's
    tokens receive a separable spatial + temporal positional embedding
    (SpectralGPT ``sep_pos_embed``) plus a modality-identity embedding, so
    tokens from DEM, soil, SAR, and optical are distinguishable in the shared
    transformer and carry both a spatial coordinate and an acquisition
    timestamp. All token sets are concatenated and cross-attended by a fixed
    set of learned latent bottleneck queries (Perceiver), which then
    self-attend to produce the fused field representation.

    ``modalities`` maps a modality name to ``{'spatial': int, 'temporal': int}``
    describing the layout of that modality's token set (spatial grid size and
    number of temporal frames). Missing modalities are simply absent from the
    input dict; their tokens are dropped and the positional embeddings of the
    remaining tokens stay aligned.
    """

    def __init__(self, embed_dim=192, num_latents=64, depth=4, num_heads=3,
                 modalities=None, modality_embed=64, mlp_ratio=4., drop_rate=0.,
                 qkv_bias=True, cross_attn_layers=1, norm_first=False, input_norm=False,
                 latent_init='randn'):
        """``cross_attn_layers`` > 1 interleaves repeated cross-attention reads
        with the latent self-attention blocks (Perceiver-IO style);
        ``norm_first`` makes the latent blocks pre-norm; ``input_norm`` applies a
        LayerNorm to the modality tokens before they are read; ``latent_init``
        is 'randn' (original) or 'trunc_normal' (std 0.02). The defaults
        reproduce the original single-read, post-norm module exactly."""
        super().__init__()
        if cross_attn_layers < 1 or cross_attn_layers > max(1, depth):
            raise ValueError('cross_attn_layers must be in [1, max(1, depth)]')
        self.embed_dim = embed_dim
        self.num_latents = num_latents
        self.modalities = modalities or {}
        self.modality_embed = modality_embed

        # per-modality separable spatial + temporal positional embeddings
        self.pos_embeds = nn.ModuleDict()
        for name, cfg in self.modalities.items():
            self.pos_embeds[name] = SeparablePosEmbed(
                cfg['spatial'], cfg['temporal'], embed_dim - modality_embed)

        # modality-identity embedding (concatenated to the pos embed, as in the
        # group-channels model) so tokens from different sources are
        # distinguishable in the shared transformer.
        self.modality_embedding = nn.ParameterDict()
        for name in self.modalities:
            self.modality_embedding[name] = nn.Parameter(
                torch.zeros(1, 1, modality_embed))
        for i, name in enumerate(self.modalities):
            emb = get_1d_sincos_pos_embed(
                modality_embed, np.arange(1, dtype=np.float32) + i)
            self.modality_embedding[name].data.copy_(
                torch.from_numpy(emb).float().unsqueeze(0))

        # learned latent bottleneck queries (Perceiver)
        self.latent_tokens = nn.Parameter(torch.randn(1, num_latents, embed_dim))
        if latent_init == 'trunc_normal':
            nn.init.trunc_normal_(self.latent_tokens, std=0.02)
        elif latent_init != 'randn':
            raise ValueError('latent_init must be randn or trunc_normal')

        # cross-attention: latent queries attend to the concatenated modality
        # tokens, then a self-attention stack refines the latents.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=drop_rate)
        self.cross_norm = nn.LayerNorm(embed_dim)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio), dropout=drop_rate,
                activation='gelu', batch_first=True, norm_first=norm_first)
            for _ in range(depth)
        ])
        # additional cross-attention reads, placed before evenly spaced blocks
        self.cross_attn_layers = cross_attn_layers
        self.extra_cross_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=drop_rate)
            for _ in range(cross_attn_layers - 1)])
        self.extra_cross_norm = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(cross_attn_layers - 1)])
        self._read_before = {int(round(i * depth / cross_attn_layers)): i - 1
                             for i in range(1, cross_attn_layers)}
        self.input_norm = nn.LayerNorm(embed_dim) if input_norm else None
        self.norm = nn.LayerNorm(embed_dim)
        # multi-scale feature-extraction layer indices (finest first) whose
        # outputs become the multiscale patch-token set fed to the DPT head.
        self.extraction_layers = get_extraction_layers(depth)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _build_key_padding_mask(self, mask, tokens, token_mask=None):
        """Build a ``(B, sum L)`` key-padding mask for the Perceiver
        cross-attention from a per-modality availability dict.

        ``mask`` maps a modality name to a per-sample availability flag
        (scalar, or a ``(B,)`` tensor; 1 = present, 0 = absent). Tokens of an
        absent modality are masked out of the cross-attention (True = ignore)
        so a learned missing-modality token does not contribute to the fused
        representation. Returns ``None`` when no mask is supplied.
        """
        if mask is None and token_mask is None:
            return None
        mask = mask or {}
        token_mask = token_mask or {}
        parts = []
        for name, emb in tokens:
            avail = torch.as_tensor(mask.get(name, True), device=emb.device)
            if avail.dim() == 0:
                avail = avail.expand(emb.shape[0])
            if avail.shape != (emb.shape[0],):
                raise ValueError(f"Availability for {name} must have shape ({emb.shape[0]},)")
            # True = masked (missing), so invert the availability flag.
            missing = (avail < 0.5).unsqueeze(1).expand(-1, emb.shape[1])
            tm = token_mask.get(name)
            if tm is not None:
                tm = torch.as_tensor(tm, device=emb.device, dtype=torch.bool)
                if tm.shape != emb.shape[:2]:
                    raise ValueError(f"Token mask for {name} must have shape {tuple(emb.shape[:2])}")
                missing = missing | ~tm
            parts.append(missing)
        key_padding_mask = torch.cat(parts, dim=1)
        # An all-absent sample has no real key to attend to. Let its first
        # learned missing token act as a stable fallback instead of creating
        # an all-masked softmax row (which produces NaNs).
        all_absent = key_padding_mask.all(dim=1)
        if all_absent.any():
            key_padding_mask[all_absent, 0] = False
        return key_padding_mask

    def _assemble_tokens(self, embeddings, ids_keep=None, mask=None, token_mask=None):
        """Validate layouts, add position/identity, and build attention mask."""
        tokens = []
        batch_size = None
        for name, emb in embeddings.items():
            if name not in self.pos_embeds:
                continue
            if emb.ndim != 3 or emb.shape[2] != self.embed_dim:
                raise ValueError(f"{name} tokens must have shape (B, L, {self.embed_dim})")
            if batch_size is None:
                batch_size = emb.shape[0]
            elif emb.shape[0] != batch_size:
                raise ValueError(f"Batch size mismatch for {name}")
            declared = self.pos_embeds[name].spatial_dim * self.pos_embeds[name].temporal_dim
            if emb.shape[1] not in (1, declared):
                raise ValueError(
                    f"Token layout mismatch for {name}: declared {declared}, got {emb.shape[1]}"
                )
            keep = ids_keep.get(name) if ids_keep is not None else None
            if keep is not None:
                emb = torch.gather(emb, dim=1, index=keep.unsqueeze(-1).expand(-1, -1, emb.shape[2]))
                if token_mask is not None and name in token_mask:
                    raise ValueError('ids_keep and token_mask cannot be combined')
            pos = self.pos_embeds[name](ids_keep=keep)
            if emb.shape[1] == 1 and pos.shape[1] != 1:
                # Whole-modality absence uses one compact learned token.
                pos = pos[:, :1, :]
            elif pos.shape[1] != emb.shape[1]:
                raise ValueError(
                    f"Token layout mismatch for {name}: declared {pos.shape[1]}, got {emb.shape[1]}"
                )
            pos = pos.expand(emb.shape[0], -1, -1)
            mod = self.modality_embedding[name].expand(emb.shape[0], emb.shape[1], -1)
            tokens.append((name, emb + torch.cat([pos, mod], dim=-1)))
        if not tokens:
            raise ValueError('LatentFusionTransformer received no modality tokens')
        x = torch.cat([token for _, token in tokens], dim=1)
        if self.input_norm is not None:
            x = self.input_norm(x)
        return x, self._build_key_padding_mask(mask, tokens, token_mask)

    def _read(self, i, latents, x, key_padding_mask):
        """Cross-attention read ``i`` (0 = the original read)."""
        attn = self.cross_attn if i == 0 else self.extra_cross_attn[i - 1]
        norm = self.cross_norm if i == 0 else self.extra_cross_norm[i - 1]
        out, _ = attn(latents, x, x, key_padding_mask=key_padding_mask)
        return norm(latents + out)

    def _latent_stack(self, x, key_padding_mask, collect=False):
        latents = self.latent_tokens.expand(x.shape[0], -1, -1)
        latents = self._read(0, latents, x, key_padding_mask)
        multi_scale = []
        for i, blk in enumerate(self.blocks):
            if i in self._read_before:
                latents = self._read(self._read_before[i] + 1, latents, x, key_padding_mask)
            latents = blk(latents)
            if collect and i in self.extraction_layers:
                multi_scale.append(latents)
        return latents, multi_scale

    def forward(self, embeddings, ids_keep=None, mask=None, token_mask=None):
        """embeddings: dict {modality: (B, L_m, embed_dim)}.

        ``ids_keep`` optionally maps a modality name to a ``(B, K)`` index
        tensor selecting a kept subset of that modality's tokens (gather-by-
        ids_keep), keeping the positional embeddings aligned when tokens are
        masked/dropped. ``mask`` optionally maps a modality name to a per-sample
        availability flag (1 = present, 0 = absent); absent-modality tokens are
        masked out of the Perceiver cross-attention via a key-padding mask so
        they do not contribute to the fused representation. Returns the fused
        latent ``(B, num_latents, embed_dim)``. ``token_mask`` optionally maps a
        modality to a ``(B, L_m)`` bool validity of its individual tokens;
        invalid tokens are masked out of every cross-attention read.
        """
        x, key_padding_mask = self._assemble_tokens(embeddings, ids_keep, mask, token_mask)
        latents, _ = self._latent_stack(x, key_padding_mask)
        return self.norm(latents)

    def forward_multiscale(self, embeddings, ids_keep=None, mask=None, token_mask=None):
        """Emit multi-scale patch tokens from the fusion backbone.

        Runs the same Perceiver cross-attention + self-attention stack as
        :meth:`forward`, but collects the latent bottleneck states at the
        four evenly-spaced intermediate layer indices returned by
        :func:`get_extraction_layers` (finest first). These become the
        multiscale patch-token set that is projected into the common latent
        dimension and fused by the DPT head (or cross-attention queries),
        letting high-resolution terrain tokens coexist with coarse
        soil/climate tokens. ``mask`` optionally maps a modality name to a
        per-sample availability flag (1 = present, 0 = absent); absent-modality
        tokens are masked out of the cross-attention. Returns a list of four
        ``(B, num_latents, embed_dim)`` tensors, finest first.
        """
        x, key_padding_mask = self._assemble_tokens(embeddings, ids_keep, mask, token_mask)
        _, multi_scale = self._latent_stack(x, key_padding_mask, collect=True)
        # finest first, matching the DPT head's expected input order
        return list(reversed(multi_scale))



if __name__ == "__main__":
    # three modalities with different native token layouts: a 4x4 raster
    # (spatial=16, temporal=1), a 2-frame multitemporal raster (spatial=16,
    # temporal=2), and a 12-step time series (spatial=1, temporal=12).
    model = LatentFusionTransformer(
        embed_dim=192, num_latents=32, depth=2, num_heads=3,
        modalities={
            'dem': {'spatial': 16, 'temporal': 1},
            'sar': {'spatial': 16, 'temporal': 2},
            'weather': {'spatial': 1, 'temporal': 12},
        },
        modality_embed=64,
    )

    embeddings = {
        'dem': torch.randn(2, 16, 192),
        'sar': torch.randn(2, 32, 192),
        'weather': torch.randn(2, 12, 192),
    }
    out = model(embeddings)
    print('fused latent', tuple(out.shape))

    # missing-modality case: drop the SAR tokens entirely; the remaining
    # positional embeddings stay aligned.
    partial = {'dem': embeddings['dem'], 'weather': embeddings['weather']}
    out_partial = model(partial)
    print('fused latent (missing sar)', tuple(out_partial.shape))

    # gather-by-ids_keep: keep only a subset of the DEM tokens.
    keep = torch.randint(0, 16, (2, 8))
    out_keep = model({'dem': embeddings['dem']}, ids_keep={'dem': keep})
    print('fused latent (gathered dem)', tuple(out_keep.shape))

    # multi-scale patch tokens: four intermediate-layer latents (finest
    # first) that feed the DPT dense yield-map head.
    multi = model.forward_multiscale(embeddings)
    print('extraction layers', model.extraction_layers)
    print('multiscale tokens', [tuple(m.shape) for m in multi])

    # attention masking: mark SAR as absent (0) so its tokens are masked out
    # of the Perceiver cross-attention via a key-padding mask, while DEM and
    # weather remain present (1). The fused latent shape is unchanged.
    mask = {'dem': torch.ones(2), 'sar': torch.zeros(2), 'weather': torch.ones(2)}
    out_masked = model(embeddings, mask=mask)
    print('fused latent (masked sar)', tuple(out_masked.shape))
    multi_masked = model.forward_multiscale(embeddings, mask=mask)
    print('multiscale tokens (masked sar)', [tuple(m.shape) for m in multi_masked])
