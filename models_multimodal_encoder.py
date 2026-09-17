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
        'sar': MultitemporalSAREncoder,
        'timeseries': TimeSeriesEncoder,
        'tabular': TabularEncoder,
        'categorical': CategoricalEncoder,
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
                # learned missing-modality token: single well-formed embedding
                # with an explicit availability flag of 0.
                token = self.missing_tokens[name].unsqueeze(0).expand(
                    inputs[name].shape[0], -1) if name in inputs else \
                    self.missing_tokens[name].unsqueeze(0)
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
