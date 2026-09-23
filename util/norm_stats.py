# --------------------------------------------------------
# Per-modality normalization statistics registry.
# Adapted from ucam-eo/tessera (tessera_infer_QAT/src/datasets/
# v1_1_norm_stats.py) to the MMST-ViT plain-PyTorch layout.
#
# Each input source is standardized independently, in its own scientific
# units, before its dedicated encoder projects it into the shared latent
# space. This is the per-modality standardization layer the heterogeneous
# encoders in models_multimodal_encoder.py need: every sensor (Sentinel-2,
# Sentinel-1 ascending/descending, Landsat) carries its own mean/std, and
# even the same sensor is normalized with source-specific stats. The
# registry is extended with entries for DEM/terrain, soil, and weather so
# each modality encoder normalizes its native-resolution input rather than
# assuming a common raster scale.
# --------------------------------------------------------

import numpy as np
import torch


# Per-modality mean/std keyed by source name. Each value is a dict with
# 'mean' and 'std' arrays whose length matches the modality's channel count.
# Sentinel-2 stats are the 10-band surface-reflectance means/stds; Sentinel-1
# ascending/descending are the 2-channel (VV/VH) backscatter stats; Landsat
# is the 6-band optical stack. DEM/terrain, soil and weather entries are
# provided as calibrated defaults (to be replaced with dataset-specific
# statistics when available) so every encoder standardizes its own input.
NORM_STATS = {
    "s2": {
        "mean": np.array([2683.4553, 2223.3630, 2432.0950, 3633.1970, 3602.1755,
                          3006.4324, 3400.2710, 3515.6392, 2456.9163, 1983.8783],
                         dtype=np.float32),
        "std": np.array([2739.5217, 2846.2993, 2690.8250, 2290.0439, 2088.8970,
                         2673.1106, 2381.4521, 2229.5225, 1601.0942, 1495.3545],
                        dtype=np.float32),
    },
    "s1a": {
        "mean": np.array([5588.3291, 3025.6270], dtype=np.float32),
        "std": np.array([1713.4646, 1693.0471], dtype=np.float32),
    },
    "s1d": {
        "mean": np.array([5552.9683, 2955.0520], dtype=np.float32),
        "std": np.array([1685.5857, 1677.6414], dtype=np.float32),
    },
    "landsat": {
        "mean": np.array([2793.6589, 2356.7776, 2551.0496, 3741.9229, 3713.7844,
                          3120.1997], dtype=np.float32),
        "std": np.array([2810.0093, 2933.8835, 2755.6360, 2344.5027, 2145.7986,
                         2743.9019], dtype=np.float32),
    },
    "dem": {
        "mean": np.array([0.0], dtype=np.float32),
        "std": np.array([1.0], dtype=np.float32),
    },
    "terrain": {
        "mean": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "std": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    },
    "soil": {
        "mean": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         dtype=np.float32),
        "std": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        dtype=np.float32),
    },
    "weather": {
        "mean": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "std": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    },
}


def get_norm_stats(source):
    """Return the (mean, std) arrays for a modality source, or None if the
    source has no registered statistics (in which case the caller should
    leave the input unstandardized)."""
    stats = NORM_STATS.get(source)
    if stats is None:
        return None, None
    return stats["mean"], stats["std"]


def normalize_modality(source, x, channel_axis=0):
    """Standardize a modality tensor with its own per-source mean/std.

    Accepts NumPy arrays or tensors. ``channel_axis`` identifies the feature
    dimension, including for batched rasters (B, C, H, W), SAR (B, T, C, H, W),
    and weather (B, T, D). Unknown sources retain their values and type.
    """
    mean, std = get_norm_stats(source)
    if mean is None:
        return x if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float32)
    arr = x.float() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float32)
    axis = channel_axis % arr.ndim
    n_channels = int(arr.shape[axis])
    if len(mean) != n_channels:
        raise ValueError(
            "normalize_modality: source '{}' expects {} channels but got {}.".format(
                source, len(mean), n_channels
            )
        )
    shape = [1] * arr.ndim
    shape[axis] = n_channels
    if isinstance(arr, torch.Tensor):
        mean = torch.as_tensor(mean, dtype=arr.dtype, device=arr.device).reshape(shape)
        std = torch.as_tensor(std, dtype=arr.dtype, device=arr.device).reshape(shape)
    else:
        mean = mean.reshape(shape)
        std = std.reshape(shape)
    return (arr - mean) / std
