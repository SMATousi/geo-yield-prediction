# --------------------------------------------------------
# Native-resolution heterogeneous modality encoders.
# Adapted from flyakon/AgriFM (models/encoders.py) to the MMST-ViT
# plain-PyTorch layout (no mmengine registry). MultiModalEncoder builds
# one dedicated encoder per input source from a config dict and returns a
# dict of per-modality embeddings keyed by source name. Each modality keeps
# its own native-resolution encoder (raster CNN for DEM/terrain/soil, a 3D
# conv for multitemporal SAR, a time-series encoder for weather/moisture, a
# tabular MLP for point soil attributes, a categorical embedding for
# crop-history), so nothing is forced onto a common resolution before
# feature extraction. A per-modality projection maps every encoder output
# into a shared latent dimension for the fusion transformer.
# --------------------------------------------------------

import numpy as np
import torch
from torch import nn

from util.norm_stats import normalize_modality

from models_group_channels_vit import (
    GroupChannelsVisionTransformer,
    get_2d_sincos_pos_embed_from_grid,
)


def _flatten_spatial(x):
    """Collapse trailing spatial dims into a token dim: (B, C, H, W) -> (B, H*W, C)."""
    b, c = x.shape[0], x.shape[1]
    return x.reshape(b, c, -1).transpose(1, 2)


class RasterCNNEncoder(nn.Module):
    """Convolutional encoder for high-resolution raster layers (DEM, terrain,
    soil rasters). Operates on (B, C, H, W) at the layer's native resolution
    and returns a set of spatial tokens (B, H*W, embed_dim)."""

    def __init__(self, in_channels, embed_dim, hidden_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(hidden_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        h = self.net(x)
        return _flatten_spatial(self.proj(h))


class PatchEmbeddingEncoder(nn.Module):
    """Native-resolution raster tokenizer with a learnable cls token and a
    spatial positional embedding.

    Adapted from srinadh99/VISION-TRANSFORMER-DRIVEN-LIDAR-DATA-FUSION-FOR-
    ENHANCED-HYPERSPECTRAL-IMAGE-CLASSIFICATION (UH_ViT_CA_LF_CLS_HSI_LiDAR.ipynb,
    PatchEmbedding) to the MMST-ViT plain-PyTorch layout. It tokenizes a raster
    patch (B, C, H, W) into a sequence of learned tokens, projects each spatial
    location to a shared latent dim, and prepends a learnable cls token plus a
    positional embedding. Unlike the source's 1x1 conv (a channel-wise no-op),
    the projection here is a real patchify conv (kernel=patch_size,
    stride=patch_size) so spatial patches are actually aggregated. Because it
    is parameterized by ``in_channels`` and ``patch_size``, it can be
    instantiated once per modality (DEM/terrain, SAR, soil, optical) at that
    modality's own native resolution without resampling. The cls token serves
    as a field-level summary token for the fusion transformer, and the
    positional embedding already encodes spatial position.

    Returns (B, num_patches + 1, embed_dim): the cls token followed by one
    token per spatial patch.
    """

    def __init__(self, in_channels, embed_dim, patch_size=16, dropout=0.0,
                 num_patches=None):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.dropout = dropout
        # real patchify conv: aggregates spatial patches (kernel=stride=patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.num_patches = num_patches
        # positional embedding sized for num_patches + 1 (cls) tokens; if
        # num_patches is not known up front it is inferred on the first forward.
        self.pos_embedding = None
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout_layer = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _init_pos_embedding(self, num_patches, device):
        self.num_patches = num_patches
        pos_embedding = torch.zeros(1, num_patches + 1, self.embed_dim)
        pos_embedding = pos_embedding + self._get_2d_sincos_pos_embed(
            self.embed_dim, int(num_patches ** 0.5), cls_token=True)
        self.pos_embedding = nn.Parameter(pos_embedding.to(device))

    @staticmethod
    def _get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
        pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token:
            pos_embed = np.concatenate(
                [np.zeros([1, embed_dim], dtype=np.float32), pos_embed], axis=0)
        return pos_embed

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)                       # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)       # (B, num_patches, embed_dim)
        x = self.norm(x)
        if self.pos_embedding is None or self.pos_embedding.shape[1] != x.shape[1] + 1:
            self._init_pos_embedding(x.shape[1], x.device)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout_layer(x)
        return x


class MultitemporalSAREncoder(nn.Module):
    """3D-convolutional encoder for multitemporal SAR backscatter stacks.
    Accepts (B, T, C, H, W) (e.g. T frames of VV/VH) and returns a set of
    spatial tokens (B, H*W, embed_dim) that retain the temporal context."""

    def __init__(self, in_channels, embed_dim, hidden_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(hidden_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        h = self.net(x)
        # temporal mean pooling -> (B, hidden, H, W)
        h = h.mean(dim=2)
        return _flatten_spatial(self.proj(h))


class TimeSeriesEncoder(nn.Module):
    """Temporal encoder for weather / precipitation / soil-moisture series.
    Accepts (B, T, D) and returns a single field-level token (B, 1, embed_dim)."""

    def __init__(self, in_dim, embed_dim, hidden_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        h = self.net(x)          # (B, T, hidden)
        h = h.mean(dim=1)        # temporal mean pooling -> (B, hidden)
        return self.proj(h).unsqueeze(1)


class TabularEncoder(nn.Module):
    """MLP encoder for point / tabular soil attributes. Accepts (B, D) and
    returns a single learned token (B, 1, embed_dim)."""

    def __init__(self, in_dim, embed_dim, hidden_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        h = self.net(x)
        return self.proj(h).unsqueeze(1)


class CategoricalEncoder(nn.Module):
    """Learned categorical / spatial representation for crop-history and
    land-cover layers. Accepts integer class indices (B, H, W) and returns a
    set of spatial tokens (B, H*W, embed_dim)."""

    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_classes, embed_dim)

    def forward(self, x):
        return self.embedding(x.long()).reshape(x.shape[0], -1, self.embed_dim)


class MultiModalEncoder(nn.Module):
    """Container that dispatches each input source to its own dedicated
    encoder and returns a dict of per-modality embeddings keyed by source
    name.

    ``encoders_cfg`` maps a modality name to a config dict::

        {
            'dem':    {'type': 'raster', 'in_channels': 1, 'embed_dim': 192},
            'sar':    {'type': 'sar', 'in_channels': 2, 'embed_dim': 192},
            'weather':{'type': 'timeseries', 'in_dim': 9, 'embed_dim': 192},
            'soil':   {'type': 'tabular', 'in_dim': 12, 'embed_dim': 192},
            'crop':   {'type': 'categorical', 'num_classes': 20, 'embed_dim': 192},
        }

    A prebuilt encoder instance may be supplied directly (e.g. an optical
    PVTSimCLR) via ``{'type': 'module', 'module': <nn.Module>}``. Every
    encoder output is projected to ``embed_dim`` so tokens from different
    sources share a common latent dimension for the fusion transformer.

    ``modality_dropout`` (0..1) enables training-time random removal of input
    sources so the backbone learns to operate on arbitrary subsets of
    modalities (missing-modality robustness). A learned missing-modality token
    is kept per source and substituted whenever a modality is absent or dropped,
    following the tessera convention that an absent modality still yields a
    well-formed embedding plus an explicit availability mask instead of failing
    the forward pass.
    """

    _TYPES = {
            'raster': RasterCNNEncoder,
            'patch': PatchEmbeddingEncoder,
            'sar': MultitemporalSAREncoder,
            'timeseries': TimeSeriesEncoder,
            'tabular': TabularEncoder,
            'categorical': CategoricalEncoder,
            'grouped_vit': GroupChannelsVisionTransformer,
        }

    def __init__(self, encoders_cfg, embed_dim=192, modality_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.modality_dropout = modality_dropout
        self.encoders = nn.ModuleDict()
        for name, cfg in encoders_cfg.items():
            self.encoders[name] = self._build_encoder(cfg)
        # Per-modality standardization: each source is normalized with its own
        # mean/std (see util/norm_stats.py) before its dedicated encoder runs,
        # so every modality is standardized in its own scientific units rather
        # than being forced onto a common raster scale. A config may override
        # the registry key via 'norm_source' (e.g. 's1a' vs 's1d' for SAR).
        self.norm_sources = {
            name: cfg.get('norm_source', name) for name, cfg in encoders_cfg.items()
        }
        # Learned missing-modality token per source (tessera convention: an
        # absent modality still yields a well-formed embedding instead of
        # erroring). Used whenever a modality is unavailable or dropped.
        self.missing_tokens = nn.ParameterDict()
        for name in self.encoders:
            self.missing_tokens[name] = nn.Parameter(torch.zeros(embed_dim))

    def _build_encoder(self, cfg):
        enc_type = cfg.get('type')
        if enc_type == 'module':
            return cfg['module']
        if enc_type not in self._TYPES:
            raise ValueError('Unknown modality encoder type: {}'.format(enc_type))
        cls = self._TYPES[enc_type]
        kwargs = {k: v for k, v in cfg.items() if k != 'type'}
        kwargs.setdefault('embed_dim', self.embed_dim)
        return cls(**kwargs)

    def forward(self, inputs):
        """inputs: dict {modality_name: tensor}. Returns dict of per-modality
        embeddings keyed by source name, each projected to the shared latent
        dimension. Missing modalities are simply absent from the output dict,
so the backbone can run on arbitrary subsets of sources."""
        outputs = {}
        for name, encoder in self.encoders.items():
            if name not in inputs:
                continue
            x = inputs[name]
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(normalize_modality(self.norm_sources[name], x))
            outputs[name] = encoder(x)
        return outputs

    def forward_with_missing(self, inputs, available=None, apply_dropout=True):
        """Missing-modality-robust forward pass.

        Returns ``(embeddings, mask)`` where ``embeddings`` contains an entry
        for *every* configured modality and ``mask`` is a dict of per-modality
        availability flags (1 = present, 0 = absent). Absent modalities are
        represented by a learned missing-modality token (the tessera
        zero-fill/mask convention adapted to this repo's dict-of-embeddings
        design) so the downstream fusion transformer always receives a
        well-formed input and can gate attention on ``mask``.

        ``available`` optionally overrides which modalities are present (e.g.
        from a data loader's availability mask). When ``apply_dropout`` is set
        and the module is in training mode, ``modality_dropout`` randomly
        removes input sources so the backbone learns to operate on arbitrary
        subsets of modalities.
        """
        if available is None:
            available = {name: name in inputs for name in self.encoders}
        else:
            available = {name: bool(available.get(name, False)) for name in self.encoders}

        if self.training and apply_dropout and self.modality_dropout > 0:
            for name in self.encoders:
                if available[name] and torch.rand(1).item() < self.modality_dropout:
                    available[name] = False

        # Determine the batch size from the first present modality so that
        # missing-modality tokens are emitted at the same batch size as the
        # present ones. This keeps every modality's token set batch-aligned for
        # the fusion transformer, which concatenates token sets across sources.
        batch_size = None
        for name in self.encoders:
            if available[name]:
                x = inputs[name]
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(normalize_modality(self.norm_sources[name], x))
                batch_size = x.shape[0]
                break

        embeddings = {}
        mask = {}
        for name, encoder in self.encoders.items():
            if available[name]:
                x = inputs[name]
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(normalize_modality(self.norm_sources[name], x))
                embeddings[name] = encoder(x)
                mask[name] = torch.ones(embeddings[name].shape[0], dtype=torch.float32,
                                        device=embeddings[name].device)
            else:
                # learned missing-modality token: a well-formed embedding with
                # an explicit availability flag of 0. When a batch size can be
                # inferred from a present modality, the token is expanded to
                # that batch size so all modalities stay batch-aligned; when no
# modality is present at all, a single token is returned.
                token = self.missing_tokens[name].unsqueeze(0).unsqueeze(0)
                if batch_size is not None:
                    token = token.expand(batch_size, 1, -1)
                embeddings[name] = token
                mask[name] = torch.zeros(embeddings[name].shape[0], dtype=torch.float32,
                                         device=embeddings[name].device)
        return embeddings, mask


if __name__ == "__main__":
    cfg = {
        'dem': {'type': 'raster', 'in_channels': 1},
        'sar': {'type': 'sar', 'in_channels': 2},
        'weather': {'type': 'timeseries', 'in_dim': 9},
        'soil': {'type': 'tabular', 'in_dim': 12},
        'crop': {'type': 'categorical', 'num_classes': 20},
    }
    model = MultiModalEncoder(cfg, embed_dim=192, modality_dropout=0.3)

    inputs = {
        'dem': torch.randn(2, 1, 64, 64),
        'sar': torch.randn(2, 6, 2, 32, 32),
        'weather': torch.randn(2, 12, 9),
        'soil': torch.randn(2, 12),
        'crop': torch.randint(0, 20, (2, 32, 32)),
    }
    out = model(inputs)
    for name, emb in out.items():
        print(name, tuple(emb.shape))

    # missing-modality-robust forward: every modality present, absent ones
    # substituted with a learned token and flagged 0 in the availability mask.
    model.train()
    partial = {'dem': inputs['dem'], 'weather': inputs['weather']}
    embeddings, mask = model.forward_with_missing(partial, apply_dropout=False)
    for name in model.encoders:
        print('fwm', name, tuple(embeddings[name].shape), 'mask', mask[name].tolist())
