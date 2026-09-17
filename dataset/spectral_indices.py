"""Vegetation spectral indices for optical imagery preprocessing.

Adapted from Momahmoses/agricultural-yield-forecasting (src/data_loader.py) to
the MMST-ViT field-level ingestion module. These are numerically-stabilized
(epsilon-guarded) NDVI and EVI computations that convert raw NIR/Red/Blue
surface-reflectance rasters into vegetation-index rasters before they feed the
optical-imagery encoder (gap g2) in the field-level geospatial data pipeline
(gap g6). They are self-contained and require only numpy, so they transplant
cleanly into the field-level data ingestion module.

Each function accepts either a single band array or a ``(C, H, W)`` raster stack
with the band selected by ``nir_idx`` / ``red_idx`` / ``blue_idx``, so they work
both on raw band arrays and on the channel-first rasters produced by the field
loaders (``field_patch_dataset``, ``field_roi``).
"""

import numpy as np

#: Small epsilon added to the denominator to avoid division by zero on
#: saturated / nodata pixels where the band difference vanishes.
_EPS = 1e-8


def _select_band(band, stack, idx):
    """Return ``band`` if given, else channel ``idx`` of a ``(C, H, W)`` stack."""
    if band is not None:
        return np.asarray(band, dtype=np.float32)
    if stack is None:
        raise ValueError("must provide either a band array or a (C, H, W) stack")
    return np.asarray(stack, dtype=np.float32)[idx]


def compute_ndvi(nir, red, stack=None, nir_idx=0, red_idx=1):
    """Normalized Difference Vegetation Index.

    ``(NIR - Red) / (NIR + Red)`` with an epsilon-guarded denominator so
    saturated or nodata pixels do not produce ``inf``/``nan``.

    Parameters
    ----------
    nir, red : np.ndarray, optional
        NIR and Red reflectance band arrays. If ``None``, they are read from
        ``stack`` at ``nir_idx`` / ``red_idx``.
    stack : np.ndarray, optional
        ``(C, H, W)`` channel-first raster stack to select bands from.
    nir_idx, red_idx : int
        Channel indices of the NIR / Red bands in ``stack``.

    Returns
    -------
    np.ndarray
        NDVI in ``[-1, 1]`` (float32).
    """
    nir = _select_band(nir, stack, nir_idx)
    red = _select_band(red, stack, red_idx)
    return (nir - red) / (nir + red + _EPS)


def compute_evi(nir, red, blue, stack=None, nir_idx=0, red_idx=1, blue_idx=2,
                G=2.5, C1=6.0, C2=7.5, L=1.0):
    """Enhanced Vegetation Index.

    ``G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)`` with an epsilon-guarded
    denominator. The ``L`` soil-adjustment and the ``C1``/``C2`` aerosol
    coefficients follow the standard EVI formulation.

    Parameters
    ----------
    nir, red, blue : np.ndarray, optional
        NIR, Red and Blue reflectance band arrays. If ``None``, they are read
        from ``stack`` at ``nir_idx`` / ``red_idx`` / ``blue_idx``.
    stack : np.ndarray, optional
        ``(C, H, W)`` channel-first raster stack to select bands from.
    nir_idx, red_idx, blue_idx : int
        Channel indices of the NIR / Red / Blue bands in ``stack``.
    G, C1, C2, L : float
        EVI gain, aerosol-resistance coefficients and canopy background term.

    Returns
    -------
    np.ndarray
        EVI (float32).
    """
    nir = _select_band(nir, stack, nir_idx)
    red = _select_band(red, stack, red_idx)
    blue = _select_band(blue, stack, blue_idx)
    return G * (nir - red) / (nir + C1 * red - C2 * blue + L + _EPS)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    nir = rng.random((8, 8)).astype(np.float32)
    red = rng.random((8, 8)).astype(np.float32)
    blue = rng.random((8, 8)).astype(np.float32)

    ndvi = compute_ndvi(nir, red)
    evi = compute_evi(nir, red, blue)
    print("ndvi:", ndvi.shape, ndvi.dtype, "range", float(ndvi.min()), float(ndvi.max()))
    print("evi:", evi.shape, evi.dtype, "range", float(evi.min()), float(evi.max()))

    stack = np.stack([nir, red, blue], axis=0)
    ndvi_s = compute_ndvi(None, None, stack=stack, nir_idx=0, red_idx=1)
    evi_s = compute_evi(None, None, None, stack=stack, nir_idx=0, red_idx=1, blue_idx=2)
    print("stack ndvi matches:", np.allclose(ndvi, ndvi_s))
    print("stack evi matches:", np.allclose(evi, evi_s))
