# --------------------------------------------------------
# Sentinel-1 SAR VV/VH backscatter normalization.
# Adapted from cybergis/rs-embed (providers/fetch.py) to feed
# the SAR modality encoder with a radar-appropriate [0,1] input.
# Layout-agnostic: operates on the trailing H,W axes so it also
# works on [T,2,H,W] multitemporal SAR stacks frame-by-frame.
# --------------------------------------------------------

import numpy as np


def normalize_s1_vvvh_chw(raw_chw):
    """Convert raw **linear-scale** S1 VV/VH to numerically stable [0,1] CHW.

    dB-scale input is rejected: its mostly-negative values would be zeroed by
    the ``max(x, 0)`` step and the output would be a constant-zero image with
    no warning, silently poisoning the SAR encoder.
    """
    arr = np.asarray(raw_chw, dtype=np.float32)
    if arr.ndim != 3 or int(arr.shape[0]) != 2:
        raise ValueError(
            "Expected raw S1 VV/VH CHW with C=2, got shape={}".format(
                getattr(arr, 'shape', None)
            )
        )
    finite_nonzero = arr[np.isfinite(arr) & (arr != 0.0)]
    if finite_nonzero.size and float(np.mean(finite_nonzero < 0)) > 0.5:
        raise ValueError(
            "normalize_s1_vvvh_chw expects linear-scale S1 backscatter but the "
            "input looks dB-scaled (mostly negative values). Convert dB to "
            "linear (10**(x/10)) first."
        )
    x = np.log1p(np.maximum(arr, 0.0))
    denom = np.percentile(x, 99) if np.isfinite(x).all() else 1.0
    denom = float(denom) if float(denom) > 0 else 1.0
    return np.clip(x / denom, 0.0, 1.0).astype(np.float32)
