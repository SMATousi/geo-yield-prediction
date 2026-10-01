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
# --------------------------------------------------------

import hashlib
import json

import numpy as np
import torch
from torch import nn

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
