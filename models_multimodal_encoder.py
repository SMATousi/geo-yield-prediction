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

import torch
from torch import nn


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
    """

    _TYPES = {
        'raster': RasterCNNEncoder,
        'sar': MultitemporalSAREncoder,
        'timeseries': TimeSeriesEncoder,
        'tabular': TabularEncoder,
        'categorical': CategoricalEncoder,
    }

    def __init__(self, encoders_cfg, embed_dim=192):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoders = nn.ModuleDict()
        for name, cfg in encoders_cfg.items():
            self.encoders[name] = self._build_encoder(cfg)

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
            outputs[name] = encoder(inputs[name])
        return outputs


if __name__ == "__main__":
    cfg = {
        'dem': {'type': 'raster', 'in_channels': 1},
        'sar': {'type': 'sar', 'in_channels': 2},
        'weather': {'type': 'timeseries', 'in_dim': 9},
        'soil': {'type': 'tabular', 'in_dim': 12},
        'crop': {'type': 'categorical', 'num_classes': 20},
    }
    model = MultiModalEncoder(cfg, embed_dim=192)

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
