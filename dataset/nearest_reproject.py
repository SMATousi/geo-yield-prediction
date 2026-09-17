"""Nearest-neighbour reprojection for field-level geospatial registration.

Given a coarse native-resolution raster (e.g. PRISM at 4 km, LST at 1 km, soil
moisture at 9 km) and a finer reference grid (e.g. the field ROI template from
``field_roi.load_field_roi``), precompute a kd-tree nearest-neighbour index
mapping each target-grid cell to its nearest source cell once, then scatter
source values into the target grid while masking missing cells.

This geographically registers every layer to the same field geometry without
forcing a common raster resolution, so each modality can be encoded at its own
native scale before fusion (gap g6).
"""

from typing import Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree


def build_nearest_index(
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    tgt_lats: np.ndarray,
    tgt_lons: np.ndarray,
    *,
    max_dist_deg: Optional[float] = None,
) -> np.ndarray:
    """Precompute the nearest-source-cell index for every target cell.

    Builds a kd-tree over the source grid's (lat, lon) cell centres and, for
    each target cell centre, finds the nearest source cell. The returned flat
    index can be reused across many layers sharing the same source grid, so the
    expensive nearest-neighbour search is done once per (source, target) pair.

    Parameters
    ----------
    src_lats, src_lons : np.ndarray
        1-D arrays of source grid cell-centre latitudes / longitudes.
    tgt_lats, tgt_lons : np.ndarray
        1-D arrays of target grid cell-centre latitudes / longitudes.
    max_dist_deg : float, optional
        Maximum search radius in degrees. Target cells with no source cell
        within this radius map to ``-1`` (missing).

    Returns
    -------
    projected_indices : np.ndarray
        Shape ``(n_tgt,)`` int array; entry ``i`` is the flat index into the
        ``(n_src_lat, n_src_lon)`` source grid of the nearest source cell, or
        ``-1`` when no source cell is within ``max_dist_deg``.
    """
    src_lats = np.asarray(src_lats, dtype=np.float64)
    src_lons = np.asarray(src_lons, dtype=np.float64)
    tgt_lats = np.asarray(tgt_lats, dtype=np.float64)
    tgt_lons = np.asarray(tgt_lons, dtype=np.float64)

    n_src_lat, n_src_lon = len(src_lats), len(src_lons)
    src_pts = np.stack(
        np.meshgrid(src_lons, src_lats, indexing="xy"), axis=-1
    ).reshape(-1, 2)
    tgt_pts = np.stack(
        np.meshgrid(tgt_lons, tgt_lats, indexing="xy"), axis=-1
    ).reshape(-1, 2)

    tree = cKDTree(src_pts)
    dist, idx = tree.query(tgt_pts, k=1)
    projected_indices = idx.astype(np.int64)
    if max_dist_deg is not None:
        projected_indices[dist > max_dist_deg] = -1
    return projected_indices


def reproject_nearest(
    source: np.ndarray,
    projected_indices: np.ndarray,
    n_tgt_lat: int,
    n_tgt_lon: int,
    *,
    nodata: float = -9999.9,
    src_nodata: Optional[float] = None,
) -> np.ndarray:
    """Scatter a source raster into a target grid via a nearest-neighbour index.

    Each target cell takes the value of its nearest source cell (from
    ``build_nearest_index``); target cells whose index is ``-1`` or whose
    nearest source cell is masked / nodata are filled with ``nodata`` so they
    can be masked downstream. The source raster keeps its own native resolution
    while being geographically registered to the target grid.

    Parameters
    ----------
    source : np.ndarray
        Source raster of shape ``(n_src_lat, n_src_lon)`` (optionally with a
        leading ``(n_src_lat, n_src_lon)``-indexed mask attribute, e.g. a
        ``numpy.ma.MaskedArray``).
    projected_indices : np.ndarray
        Flat nearest-source-cell index from :func:`build_nearest_index`.
    n_tgt_lat, n_tgt_lon : int
        Target grid dimensions.
    nodata : float
        Fill value for missing target cells (default ``-9999.9``).
    src_nodata : float, optional
        Source nodata value to treat as missing. If ``None``, masked entries of
        a ``numpy.ma.MaskedArray`` are used instead.

    Returns
    -------
    projected : np.ndarray
        Target grid of shape ``(n_tgt_lat, n_tgt_lon)`` with source values
        scattered by nearest neighbour and ``nodata`` where missing.
    """
    src = np.asarray(source)
    n_src_lat, n_src_lon = src.shape
    flat = src.reshape(-1)

    valid = projected_indices >= 0
    proj_i = projected_indices[valid] // n_src_lon
    proj_j = projected_indices[valid] % n_src_lon

    projected = np.full((n_tgt_lat, n_tgt_lon), nodata, dtype=src.dtype)
    tgt_flat = projected.reshape(-1)
    tgt_flat[valid] = flat[proj_i * n_src_lon + proj_j]

    if src_nodata is not None:
        tgt_flat[valid] = np.where(
            flat[proj_i * n_src_lon + proj_j] == src_nodata,
            nodata,
            tgt_flat[valid],
        )
    elif isinstance(source, np.ma.MaskedArray):
        src_mask = np.ma.getmaskarray(source).reshape(-1)
        tgt_flat[valid] = np.where(
            src_mask[proj_i * n_src_lon + proj_j], nodata, tgt_flat[valid]
        )

    return projected


def reproject_layers(
    layers: Sequence[np.ndarray],
    projected_indices: np.ndarray,
    n_tgt_lat: int,
    n_tgt_lon: int,
    *,
    nodata: float = -9999.9,
) -> dict:
    """Reproject several source layers sharing one grid onto a target grid.

    Convenience wrapper around :func:`reproject_nearest` for the common case
    where multiple variables (e.g. PRISM precipitation, tmax, tmin) live on the
    same coarse source grid and share one precomputed index.

    Returns
    -------
    dict
        ``name -> target-grid array`` for each input layer.
    """
    return {
        name: reproject_nearest(
            layer, projected_indices, n_tgt_lat, n_tgt_lon, nodata=nodata
        )
        for name, layer in layers
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # coarse 4 km PRISM-like source grid
    src_lats = np.linspace(40.0, 42.0, 5)
    src_lons = np.linspace(-95.0, -93.0, 6)
    src = rng.random((5, 6)).astype(np.float32)
    src = np.ma.masked_where(src < 0.2, src)

    # finer 1 km reference grid (subset of the source extent)
    tgt_lats = np.linspace(40.2, 41.8, 9)
    tgt_lons = np.linspace(-94.8, -93.2, 9)

    idx = build_nearest_index(src_lats, src_lons, tgt_lats, tgt_lons)
    proj = reproject_nearest(src, idx, len(tgt_lats), len(tgt_lons))
    print("index shape:", idx.shape, "projected shape:", proj.shape)
    print("missing cells:", int((proj == -9999.9).sum()))

    layers = reproject_layers(
        [("prcp", src), ("tmax", src * 2.0)], idx, len(tgt_lats), len(tgt_lons)
    )
    print("layers:", {k: v.shape for k, v in layers.items()})
