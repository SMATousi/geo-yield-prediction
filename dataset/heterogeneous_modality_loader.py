"""Native-resolution heterogeneous modality ingestion.

The dedicated per-modality encoders (``models_multimodal_encoder.py``) each
expect a tensor shaped to that source's scientific structure: a single-band
raster for DEM/terrain/soil rasters, a ``(T, C, H, W)`` stack for multitemporal
SAR backscatter, a ``(D,)`` vector for point/tabular soil attributes, and an
integer class map for categorical crop-history / land-cover layers. The generic
``field_yield_dataset`` registers every layer as a single-band float raster, so
it cannot produce those native-resolution shapes.

This loader fills that gap (gap g2): it ingests each modality at its own native
resolution and returns the exact tensor shape the corresponding encoder
expects, while still geographically registering every raster to the same field
template grid (via ``field_roi.load_field_roi``) so layers from different
sources line up without being forced onto a common raster resolution before
feature extraction. Modalities are optional: a layer file that is absent for a
field-year is skipped and flagged 0 in the per-sample availability mask, so the
loader supports the missing-modality robustness requirement.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from dataset.field_roi import load_field_roi

#: Default nodata fill used when registering a raster into the field grid.
DEFAULT_NODATA: float = -9999.0

#: Modality kinds understood by the loader. Each kind maps to the tensor shape
#: the corresponding dedicated encoder expects.
RASTER_KINDS: Tuple[str, ...] = ("raster", "dem", "terrain", "soil_raster")
SAR_KIND: str = "sar"
TABULAR_KIND: str = "tabular"
CATEGORICAL_KIND: str = "categorical"


def _resampling_for(kind: str) -> Resampling:
    """Pick a resampling mode for a layer kind: nearest for categorical, else
    bilinear (continuous DEM / terrain / soil / SAR backscatter)."""
    if kind == CATEGORICAL_KIND:
        return Resampling.nearest
    return Resampling.bilinear


def register_raster_to_grid(
    src_path: str,
    tpl: Dict,
    *,
    resampling: Resampling = Resampling.bilinear,
    nodata: float = DEFAULT_NODATA,
) -> np.ndarray:
    """Reproject / resample a georeferenced raster into a field template grid.

    ``tpl`` is the output of ``field_roi.load_field_roi`` (crs / transform /
    width / height). The source raster may be in a different CRS, resolution,
    or extent; it is warped into the field grid so per-pixel values line up
    with the other layers. Returns a ``(height, width)`` float32 array.
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


class HeterogeneousModalityLoader:
    """Ingest heterogeneous field-level modalities at their native resolutions.

    Each field-year sample is identified by a name in the split file (one field
    name per line). For every field the loader builds the canonical template
    grid from the boundary mask and registers each configured modality layer
    into that grid, shaping it according to its kind:

    * ``raster`` / ``dem`` / ``terrain`` / ``soil_raster`` -> ``(1, H, W)``
      single-band float raster (RasterCNNEncoder).
    * ``sar`` -> ``(T, C, H, W)`` multitemporal backscatter stack, where ``T``
      is the number of acquisition frames and ``C`` the polarisation channels
      (e.g. VV/VH). A SAR layer is stored as a multi-band TIFF with
      ``T * C`` bands laid out frame-major (MultitemporalSAREncoder).
    * ``tabular`` -> ``(D,)`` point / tabular attribute vector, read from a
      ``.npy`` file (TabularEncoder).
    * ``categorical`` -> ``(H, W)`` integer class-index map (CategoricalEncoder).

    Layers are optional: a file that is absent for a field-year is skipped and
    flagged 0 in the per-sample availability mask, so the loader supports the
    missing-modality robustness requirement.

    Parameters
    ----------
    root_dir : str
        Directory containing the field rasters and the split files.
    split_file : str
        Name of the split file (``train.txt`` / ``test.txt``) listing one
        field name per line.
    layers : sequence of (name, kind) tuples
        Modality layers to load, e.g. ``(("DEM", "raster"), ("SAR", "sar"),
        ("SOIL", "tabular"), ("crop", "categorical"))``.
    boundary_suffix : str
        Suffix of the field boundary mask file (default ``"_boundary.tif"``).
    out_resolution_m : float
        Ground resolution (metres) of the field template grid.
    """

    def __init__(
        self,
        root_dir: str,
        split_file: str,
        layers: Sequence[Tuple[str, str]] = (
            ("DEM", "raster"),
            ("SAR", "sar"),
            ("SOIL", "tabular"),
            ("crop", "categorical"),
        ),
        *,
        boundary_suffix: str = "_boundary.tif",
        out_resolution_m: float = 10.0,
    ):
        self.root_dir = Path(root_dir)
        self.layers = tuple(layers)
        self.boundary_suffix = boundary_suffix
        self.out_resolution_m = out_resolution_m

        with open(self.root_dir / split_file) as f:
            self.names = [line.strip() for line in f if line.strip()]

        # Pre-register every field once so __getitem__ is a pure lookup.
        self._samples: List[Dict] = []
        for field in self.names:
            self._samples.append(self._build_sample(field))

    def _build_sample(self, field: str) -> Dict:
        boundary_path = self.root_dir / f"{field}{self.boundary_suffix}"
        if not boundary_path.exists():
            raise FileNotFoundError(f"Field boundary missing: {boundary_path}")

        # Field boundary alignment -> canonical template grid.
        tpl, _, _, _ = load_field_roi(
            boundary_path, field, self.out_resolution_m
        )

        layers: Dict[str, np.ndarray] = {}
        available: Dict[str, bool] = {}
        for name, kind in self.layers:
            if kind == TABULAR_KIND:
                arr = self._load_tabular(field, name)
            else:
                arr = self._load_raster(field, name, kind, tpl)
            if arr is None:
                # missing modality -> skip the layer, flag it unavailable
                available[name] = False
                continue
            layers[name] = arr
            available[name] = True

        return {
            "field": field,
            "layers": layers,
            "available": available,
        }

    def _load_tabular(self, field: str, name: str) -> Optional[np.ndarray]:
        """Load a point / tabular attribute vector from ``<field>_<name>.npy``.

        Returns a ``(D,)`` float32 vector, or ``None`` if the file is absent.
        """
        path = self.root_dir / f"{field}_{name}.npy"
        if not path.exists():
            return None
        arr = np.load(path).astype(np.float32)
        return arr.reshape(-1)

    def _load_raster(
        self, field: str, name: str, kind: str, tpl: Dict
    ) -> Optional[np.ndarray]:
        """Load and register a raster layer into the field grid.

        Returns the layer shaped for its kind, or ``None`` if the file is
        absent. ``sar`` layers are read as multi-band TIFFs with ``T * C``
        bands laid out frame-major and reshaped to ``(T, C, H, W)``.
        """
        path = self.root_dir / f"{field}_{name}.tif"
        if not path.exists():
            return None
        resampling = _resampling_for(kind)
        with rasterio.open(path) as src:
            count = src.count
            h, w = tpl["height"], tpl["width"]
            if kind == SAR_KIND:
                # multitemporal SAR: bands are T*C, frame-major (T frames of C
                # polarisation channels). Infer T from the band count and a
                # known channel count, defaulting to C=2 (VV/VH).
                channels = 2
                if count % channels != 0:
                    channels = 1
                t = count // channels
                out = np.full((t, channels, h, w), DEFAULT_NODATA, dtype=np.float32)
                for b in range(count):
                    band = np.full((h, w), DEFAULT_NODATA, dtype=np.float32)
                    reproject(
                        source=src.read(b + 1),
                        destination=band,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=tpl["transform"],
                        dst_crs=tpl["crs"],
                        resampling=resampling,
                        src_nodata=src.nodata,
                        dst_nodata=DEFAULT_NODATA,
                    )
                    out[b // channels, b % channels] = band
                return out
            # single-band continuous / categorical raster -> (1, H, W) or (H, W)
            band = register_raster_to_grid(
                str(path), tpl, resampling=resampling
            )
            if kind == CATEGORICAL_KIND:
                return band.astype(np.int64)  # (H, W) integer class indices
            return band[None, ...]  # (1, H, W) channel-first

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> Dict:
        sample = self._samples[index]
        out: Dict = {}
        for name, arr in sample["layers"].items():
            out[name] = np.asarray(arr)
        out["available"] = np.asarray(
            [sample["available"].get(name, False) for name, _ in self.layers],
            dtype=np.float32,
        )
        out["field"] = sample["field"]
        return out


if __name__ == "__main__":
    import tempfile

    from rasterio.transform import from_origin

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Build a small field: boundary mask, DEM raster, SAR stack, tabular
        # soil vector, and categorical crop-history layer.
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
                tmp / f"{field}_DEM.tif",
                "w", driver="GTiff", height=20, width=20, count=1,
                dtype="float32", crs="EPSG:32615",
                transform=from_origin(500000.0, 4000000.0, 10.0, 10.0),
            ) as dst:
                dst.write(np.random.rand(1, 20, 20).astype(np.float32))
            # SAR: 2 frames x 2 channels (VV/VH) = 4 bands, frame-major.
            with rasterio.open(
                tmp / f"{field}_SAR.tif",
                "w", driver="GTiff", height=20, width=20, count=4,
                dtype="float32", crs="EPSG:32615",
                transform=from_origin(500000.0, 4000000.0, 10.0, 10.0),
            ) as dst:
                dst.write(np.random.rand(4, 20, 20).astype(np.float32))
            # categorical crop-history: integer class indices.
            with rasterio.open(
                tmp / f"{field}_crop.tif",
                "w", driver="GTiff", height=20, width=20, count=1,
                dtype="uint8", crs="EPSG:32615",
                transform=from_origin(500000.0, 4000000.0, 10.0, 10.0),
            ) as dst:
                dst.write(np.random.randint(0, 5, (1, 20, 20)).astype(np.uint8))
            # tabular soil attributes.
            np.save(tmp / f"{field}_SOIL.npy", np.random.rand(12).astype(np.float32))
            # field_2022 deliberately lacks the SAR layer (missing modality).
            if field == "field_2021":
                pass
            else:
                (tmp / f"{field}_SAR.tif").unlink()
        with open(tmp / "train.txt", "w") as f:
            f.write("field_2021\nfield_2022\n")

        loader = HeterogeneousModalityLoader(
            str(tmp), "train.txt",
            layers=(("DEM", "raster"), ("SAR", "sar"),
                    ("SOIL", "tabular"), ("crop", "categorical")),
            out_resolution_m=10.0,
        )
        print("n_fields:", len(loader))
        for i in range(len(loader)):
            s = loader[i]
            print(
                s["field"],
                "keys:", sorted(k for k in s if k != "field"),
                "shapes:",
                {k: tuple(v.shape) for k, v in s.items() if k not in ("field", "available")},
                "available:", s["available"].tolist(),
            )
