"""Field-level ROI registration primitive.

Given a field boundary / mask TIFF, build a canonical output template grid at a
fixed ground resolution (``out_resolution_m``) and resample the ROI mask into
that grid. Every modality (S1/S2/terrain/soil/...) can then be geographically
registered to the same field geometry without forcing a shared native
resolution: the returned template (crs/transform/width/height) defines the
common extent, while each layer is still encoded at its own native scale.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds


def load_field_roi(
    tiff: Path, partition_id: str, out_resolution_m: float
) -> Tuple[Dict, BoundingBox, BoundingBox, np.ndarray]:
    """Load ROI bounds + CRS from a field boundary TIFF and build an output
    template grid at ``out_resolution_m``.

    The ROI mask is resampled (nearest) into that output grid so all downstream
    mosaics use the same extent but a fixed resolution (e.g. 10 m).

    Returns
    -------
    tpl : dict
        Output template grid: ``crs``, ``transform``, ``width``, ``height``.
    bbox_proj : BoundingBox
        Field extent in the source (projected) CRS.
    bbox_ll : BoundingBox
        Field extent in EPSG:4326 (lon/lat).
    mask_np : np.ndarray
        Binary ROI mask resampled into the output grid (uint8, H x W).
    """
    with rasterio.open(tiff) as src:
        if src.crs is None:
            raise ValueError(f"ROI has no CRS: {tiff}")
        bbox_src = src.bounds
        bbox_ll = transform_bounds(
            src.crs,
            "EPSG:4326",
            bbox_src.left,
            bbox_src.bottom,
            bbox_src.right,
            bbox_src.top,
            densify_pts=21,
        )
        mask_src = (src.read(1) > 0).astype(np.uint8)
        left, bottom, right, top = (
            bbox_src.left,
            bbox_src.bottom,
            bbox_src.right,
            bbox_src.top,
        )
        res = float(out_resolution_m)
        width = int(np.ceil((right - left) / res))
        height = int(np.ceil((top - bottom) / res))
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid output grid computed from bounds/res: "
                f"width={width} height={height}"
            )
        transform = from_origin(left, top, res, res)
        tpl = dict(crs=src.crs, transform=transform, width=width, height=height)
        bbox_proj = BoundingBox(
            left=left, bottom=bottom, right=right, top=top
        )
        mask_np = np.zeros((height, width), dtype=np.uint8)
        reproject(
            source=mask_src,
            destination=mask_np,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=src.crs,
            resampling=Resampling.nearest,
            src_nodata=0,
            dst_nodata=0,
        )
    return tpl, bbox_proj, bbox_ll, mask_np


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tiff_path = Path(tmp) / "field_boundary.tif"
        with rasterio.open(
            tiff_path,
            "w",
            driver="GTiff",
            height=20,
            width=20,
            count=1,
            dtype="uint8",
            crs="EPSG:32615",
            transform=from_origin(500000.0, 4000000.0, 10.0, 10.0),
        ) as dst:
            mask = np.zeros((20, 20), dtype=np.uint8)
            mask[5:15, 5:15] = 1
            dst.write(mask, 1)

        tpl, bbox_proj, bbox_ll, mask_np = load_field_roi(
            tiff_path, "field_2021", 10.0
        )
        print("template:", tpl)
        print("bbox_proj:", bbox_proj)
        print("bbox_ll:", bbox_ll)
        print("mask:", mask_np.shape, mask_np.dtype, int(mask_np.sum()))
