"""Field-level yield dataset: boundary alignment + yield-monitor registration.

This is the field-level geospatial data ingestion entry point (gap g6). The
county-FIPS / 9x9 km grid loaders (``usda_loader``, ``sentinel_loader``,
``hrrr_loader``) operate on coarse county partitions; this dataset instead
treats each field-year pair as one multimodal sample and performs the two
registration steps the gap calls out:

* **Field boundary alignment** -- the field boundary mask TIFF is loaded via
  ``field_roi.load_field_roi``, which builds a canonical output template grid
  (crs / transform / width / height) at a fixed ground resolution. Every
  modality is then geographically registered to that same field geometry
  without forcing a shared native resolution.
* **Harvest yield-monitor raster registration** -- the yield-monitor raster
  (which may be at a different CRS, resolution, or extent than the boundary)
  is reprojected / resampled into the field template grid so the dense
  per-pixel yield target lines up with the input layers.

Each modality layer is registered to the field grid with its own resampling
mode (nearest for categorical crop-history / land-cover, bilinear for
continuous DEM / soil / optical rasters) and is returned as a separate tensor
so it can be encoded at its own scale before fusion. Modalities are optional:
a layer file that is absent for a field-year is skipped and flagged 0 in the
per-sample availability mask, so the loader supports the missing-modality
robustness requirement (a field-year may lack SAR, weather, soil, ...).
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from torch.utils.data import Dataset

from dataset.field_roi import load_field_roi

#: Default nodata fill used when registering a raster into the field grid.
DEFAULT_NODATA: float = -9999.0

#: Resampling mode per layer kind: categorical layers use nearest neighbour,
#: continuous layers use bilinear interpolation.
CATEGORICAL_LAYERS: Tuple[str, ...] = ("crop", "landcover", "crop_history")


def register_raster_to_grid(
    src_path: str,
    tpl: Dict,
    *,
    resampling: Resampling = Resampling.bilinear,
    nodata: float = DEFAULT_NODATA,
) -> np.ndarray:
    """Register a georeferenced raster to a field template grid.

    Reprojects / resamples the source raster (which may be in a different CRS,
    resolution, or extent) into the field grid defined by ``tpl`` (the output
    of ``field_roi.load_field_roi``). This is the harvest yield-monitor raster
    registration primitive: the yield map is aligned to the field boundary grid
    so per-pixel targets line up with the input layers.

    Parameters
    ----------
    src_path : str
        Path to the source raster (e.g. the yield-monitor TIFF).
    tpl : dict
        Field template grid with ``crs``, ``transform``, ``width``, ``height``.
    resampling : rasterio.warp.Resampling
        Resampling method used to warp the source into the target grid.
    nodata : float
        Fill value for target cells outside the source extent.

    Returns
    -------
    np.ndarray
        ``(height, width)`` float32 array registered to the field grid.
    """
    with rasterio.open(src_path) as src:
        out = np.full((tpl["height"], tpl["width"]), nodata, dtype=np.float32)
        reproject(
            source=src.read(1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=tpl["transform"],
            dst_crs=tpl["crs"],
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=nodata,
        )
    return out


def _resampling_for(layer: str) -> Resampling:
    """Pick a resampling mode for a layer: nearest for categorical, else bilinear."""
    if layer in CATEGORICAL_LAYERS:
        return Resampling.nearest
    return Resampling.bilinear


class FieldYieldDataset(Dataset):
    """Field-year multimodal dataset with boundary + yield registration.

    Each field-year sample is identified by a name in the split file
    (``train.txt`` / ``test.txt``, one field name per line). For every field
    the dataset:

    1. loads the field boundary mask (``<field>_boundary.tif``) and builds the
       field template grid via :func:`field_roi.load_field_roi`;
    2. registers the harvest yield-monitor raster (``<field>_yield.tif``) into
       that grid as the dense per-pixel target;
    3. registers each configured modality layer (``<field>_<layer>.tif``) into
       the same grid, skipping any layer that is absent for that field-year.

    The returned sample is a dict of ``layer -> (C, H, W)`` tensors plus the
    dense ``yield`` target, a per-layer ``available`` mask, and the field name.
    Layers keep their own native resolution before fusion; only their
    geographic registration to the field geometry is performed here.

    Parameters
    ----------
    root_dir : str
        Directory containing the field rasters and the split files.
    split_file : str
        Name of the split file (``train.txt`` or ``test.txt``) listing one
        field name per line.
    layers : sequence of str
        Modality layer names, e.g. ``("DEM", "S2", "SOIL", "crop")``. Each is
        loaded as ``<root>/<field>_<layer>.tif``.
    boundary_suffix : str
        Suffix of the field boundary mask file (default ``"_boundary.tif"``).
    yield_suffix : str
        Suffix of the harvest yield-monitor raster (default ``"_yield.tif"``).
    out_resolution_m : float
        Ground resolution (metres) of the field template grid.
    """

    def __init__(
        self,
        root_dir: str,
        split_file: str,
        layers: Sequence[str] = ("DEM", "S2", "SOIL", "crop"),
        *,
        boundary_suffix: str = "_boundary.tif",
        yield_suffix: str = "_yield.tif",
        out_resolution_m: float = 10.0,
    ):
        self.root_dir = Path(root_dir)
        self.layers = tuple(layers)
        self.boundary_suffix = boundary_suffix
        self.yield_suffix = yield_suffix
        self.out_resolution_m = out_resolution_m

        with open(self.root_dir / split_file) as f:
            self.names = [line.strip() for line in f if line.strip()]

        # Pre-register every field once so __getitem__ is a pure lookup.
        self._samples: List[Dict] = []
        for field in self.names:
            self._samples.append(self._build_sample(field))

    def _build_sample(self, field: str) -> Dict:
        boundary_path = self.root_dir / f"{field}{self.boundary_suffix}"
        yield_path = self.root_dir / f"{field}{self.yield_suffix}"

        if not boundary_path.exists():
            raise FileNotFoundError(f"Field boundary missing: {boundary_path}")
        if not yield_path.exists():
            raise FileNotFoundError(f"Yield raster missing: {yield_path}")

        # 1. Field boundary alignment -> canonical template grid.
        tpl, _, _, _ = load_field_roi(
            boundary_path, field, self.out_resolution_m
        )

        # 2. Harvest yield-monitor raster registration -> dense per-pixel target.
        yield_map = register_raster_to_grid(
            str(yield_path), tpl, resampling=Resampling.bilinear
        )

        # 3. Multi-modal ingestion at native resolution, registered to the grid.
        layers: Dict[str, np.ndarray] = {}
        available: Dict[str, bool] = {}
        for layer in self.layers:
            layer_path = self.root_dir / f"{field}_{layer}.tif"
            if not layer_path.exists():
                # missing modality -> skip the layer, flag it unavailable
                available[layer] = False
                continue
            arr = register_raster_to_grid(
                str(layer_path), tpl, resampling=_resampling_for(layer)
            )
            layers[layer] = arr[None, ...]  # (1, H, W) channel-first
            available[layer] = True

        return {
            "field": field,
            "layers": layers,
            "yield": yield_map,
            "available": available,
        }

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> Dict:
        sample = self._samples[index]
        out: Dict = {}
        for layer, arr in sample["layers"].items():
            out[layer] = np.asarray(arr, dtype=np.float32)
        out["yield"] = np.asarray(sample["yield"], dtype=np.float32)
        out["available"] = np.asarray(
            [sample["available"].get(layer, False) for layer in self.layers],
            dtype=np.float32,
        )
        out["field"] = sample["field"]
        return out


if __name__ == "__main__":
    import tempfile

    from rasterio.transform import from_origin

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Build a small field: boundary mask, yield raster, and two layers.
        for field in ("field_2021", "field_2022"):
            with rasterio.open(
                tmp / f"{field}_boundary.tif",
                "w", driver="GTiff", height=20, width=20, count=1,
                dtype="uint8", crs="EPSG:32615",
                transform=from_origin(500000.0, 4000000.0, 10.0, 10.0),
            ) as dst:
                mask = np.zeros((20, 20), dtype=np.uint8)
                mask[5:15, 5:15] = 1
                dst.write(mask, 1)
            with rasterio.open(
                tmp / f"{field}_yield.tif",
                "w", driver="GTiff", height=20, width=20, count=1,
                dtype="float32", crs="EPSG:32615",
                transform=from_origin(500000.0, 4000000.0, 10.0, 10.0),
            ) as dst:
                dst.write(np.random.rand(1, 20, 20).astype(np.float32))
            with rasterio.open(
                tmp / f"{field}_DEM.tif",
                "w", driver="GTiff", height=20, width=20, count=1,
                dtype="float32", crs="EPSG:32615",
                transform=from_origin(500000.0, 4000000.0, 10.0, 10.0),
            ) as dst:
                dst.write(np.random.rand(1, 20, 20).astype(np.float32))
            # field_2022 deliberately lacks the SOIL layer (missing modality).
            if field == "field_2021":
                with rasterio.open(
                    tmp / f"{field}_SOIL.tif",
                    "w", driver="GTiff", height=20, width=20, count=1,
                    dtype="float32", crs="EPSG:32615",
                    transform=from_origin(500000.0, 4000000.0, 10.0, 10.0),
                ) as dst:
                    dst.write(np.random.rand(1, 20, 20).astype(np.float32))
        with open(tmp / "train.txt", "w") as f:
            f.write("field_2021\nfield_2022\n")

        ds = FieldYieldDataset(
            str(tmp), "train.txt", layers=("DEM", "SOIL"), out_resolution_m=10.0
        )
        print("n_fields:", len(ds))
        for i in range(len(ds)):
            s = ds[i]
            print(
                s["field"],
                "keys:", sorted(k for k in s if k not in ("field",)),
                "yield:", tuple(s["yield"].shape),
                "available:", s["available"].tolist(),
            )
