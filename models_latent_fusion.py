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
                 qkv_bias=True):
        super().__init__()
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

        # cross-attention: latent queries attend to the concatenated modality
        # tokens, then a self-attention stack refines the latents.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=drop_rate)
        self.cross_norm = nn.LayerNorm(embed_dim)

        dpr = [drop_rate] * depth
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio), dropout=drop_rate,
                activation='gelu', batch_first=True)
            for _ in range(depth)
        ])
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

    def forward(self, embeddings, ids_keep=None):
        """embeddings: dict {modality: (B, L_m, embed_dim)}.

        ``ids_keep`` optionally maps a modality name to a ``(B, K)`` index
        tensor selecting a kept subset of that modality's tokens (gather-by-
        ids_keep), keeping the positional embeddings aligned when tokens are
        masked/dropped. Returns the fused latent ``(B, num_latents, embed_dim)``.
        """
        b = None
        tokens = []
        for name, emb in embeddings.items():
            if name not in self.pos_embeds:
                continue
            if b is None:
                b = emb.shape[0]
            keep = ids_keep.get(name) if ids_keep is not None else None
            if keep is not None:
                emb = torch.gather(
                    emb, dim=1,
                    index=keep.unsqueeze(-1).repeat(1, 1, emb.shape[2]))
            pos = self.pos_embeds[name].forward(ids_keep=keep)
            pos = pos.expand(emb.shape[0], -1, -1)
            mod = self.modality_embedding[name].expand(emb.shape[0], emb.shape[1], -1)
            pos = torch.cat([pos, mod], dim=-1)
            tokens.append(emb + pos)
        if not tokens:
            raise ValueError('LatentFusionTransformer received no modality tokens')
        x = torch.cat(tokens, dim=1)  # (B, sum L, embed_dim)

        # Perceiver cross-attention: latent queries attend to the modality tokens
        latents = self.latent_tokens.expand(b, -1, -1)
        attn_out, _ = self.cross_attn(latents, x, x)
        latents = self.cross_norm(latents + attn_out)
        for blk in self.blocks:
            latents = blk(latents)
        latents = self.norm(latents)
        return latents

    def forward_multiscale(self, embeddings, ids_keep=None):
        """Emit multi-scale patch tokens from the fusion backbone.

        Runs the same Perceiver cross-attention + self-attention stack as
        :meth:`forward`, but collects the latent bottleneck states at the
        four evenly-spaced intermediate layer indices returned by
        :func:`get_extraction_layers` (finest first). These become the
        multiscale patch-token set that is projected into the common latent
        dimension and fused by the DPT head (or cross-attention queries),
        letting high-resolution terrain tokens coexist with coarse
        soil/climate tokens. Returns a list of four ``(B, num_latents,
        embed_dim)`` tensors, finest first.
        """
        b = None
        tokens = []
        for name, emb in embeddings.items():
            if name not in self.pos_embeds:
                continue
            if b is None:
                b = emb.shape[0]
            keep = ids_keep.get(name) if ids_keep is not None else None
            if keep is not None:
                emb = torch.gather(
                    emb, dim=1,
                    index=keep.unsqueeze(-1).repeat(1, 1, emb.shape[2]))
            pos = self.pos_embeds[name].forward(ids_keep=keep)
            pos = pos.expand(emb.shape[0], -1, -1)
            mod = self.modality_embedding[name].expand(emb.shape[0], emb.shape[1], -1)
            pos = torch.cat([pos, mod], dim=-1)
            tokens.append(emb + pos)
        if not tokens:
            raise ValueError('LatentFusionTransformer received no modality tokens')
        x = torch.cat(tokens, dim=1)

        latents = self.latent_tokens.expand(b, -1, -1)
        attn_out, _ = self.cross_attn(latents, x, x)
        latents = self.cross_norm(latents + attn_out)
        multi_scale = []
        for i, blk in enumerate(self.blocks):
            latents = blk(latents)
            if i in self.extraction_layers:
                multi_scale.append(latents)
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
