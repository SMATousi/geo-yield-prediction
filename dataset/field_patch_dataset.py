"""Field-level geospatial patch dataset with per-pixel dense label alignment.

Adapted from the SpectralGPT change-detection loader: it ingests georeferenced
rasters (via rasterio), registers a dense per-pixel label map (the harvest
yield-monitor raster) to the same grid, and extracts co-located patches with a
configurable stride while recording each patch's coordinates. This is the
field-level geospatial data pipeline (gap g6): every field-year sample keeps
its input layers geographically aligned to the yield map without forcing a
shared native resolution, and each patch carries a dense per-pixel yield
target so within-field spatial variability is preserved.

The ``sample`` dict pattern (multiple aligned tensors + dense label) maps
directly onto the winner's field-year multimodal sample; per-channel min/max
normalization and per-patch coordinate bookkeeping are portable from the
source loader.
"""

from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def _read_raster_chw(path: str) -> np.ndarray:
    """Read a georeferenced raster as a float32 ``(C, H, W)`` array."""
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def _read_label_hw(path: str) -> np.ndarray:
    """Read a dense per-pixel label raster (e.g. yield map) as ``(H, W)``."""
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


class FieldPatchDataset(Dataset):
    """Stride-based patch extraction over field-level geospatial rasters.

    Each field-year sample is a set of georeferenced input layers (``layers``)
    plus a dense per-pixel label raster (``label``). All rasters are assumed to
    be geographically registered to the same field; they need not share native
    resolution. Patches of side ``patch_side`` are extracted with a configurable
    ``stride`` and every patch is paired with the co-located dense yield target.

    Parameters
    ----------
    path : str
        Directory containing the field rasters and the ``train.txt`` /
        ``test.txt`` split files (one field name per line).
    train : bool
        Whether to use the ``train.txt`` (True) or ``test.txt`` (False) split.
    layers : sequence of str
        Names of the input layer rasters, e.g. ``("DEM", "S2", "SOIL")``. Each
        is loaded as ``<path>/<field>_<layer>.tif``.
    label : str
        Name of the dense label raster, loaded as ``<path>/<field>_<label>.tif``.
    patch_side : int
        Side length of the square patches to extract.
    stride : int, optional
        Patch stride. Defaults to 1 (dense sliding window).
    transform : callable, optional
        Optional per-sample transform applied to the returned ``sample`` dict.
    """

    def __init__(
        self,
        path: str,
        train: bool = True,
        layers: Sequence[str] = ("DEM", "S2", "SOIL"),
        label: str = "yield",
        patch_side: int = 96,
        stride: Optional[int] = None,
        transform=None,
    ):
        self.path = path
        self.layers = tuple(layers)
        self.label = label
        self.patch_side = patch_side
        self.stride = stride if stride else 1
        self.transform = transform

        fname = "train.txt" if train else "test.txt"
        with open(Path(path) / fname) as f:
            self.names = [line.strip() for line in f if line.strip()]
        self.n_imgs = len(self.names)

        self.layer_rasters: Dict[str, Dict[str, np.ndarray]] = {}
        self.label_rasters: Dict[str, np.ndarray] = {}
        self.n_patches_per_field: Dict[str, int] = {}
        self.n_patches = 0
        self.patch_coords: List[Tuple[str, List[int], List[int]]] = []

        for field in tqdm(self.names):
            layer_data = {}
            for layer in self.layers:
                layer_path = Path(path) / f"{field}_{layer}.tif"
                if layer_path.exists():
                    layer_data[layer] = _read_raster_chw(str(layer_path))
            self.layer_rasters[field] = layer_data
            self.label_rasters[field] = _read_label_hw(str(Path(path) / f"{field}_{label}.tif"))

            h, w = self.label_rasters[field].shape
            n1 = ceil((h - self.patch_side + 1) / self.stride)
            n2 = ceil((w - self.patch_side + 1) / self.stride)
            n_patches_i = n1 * n2
            self.n_patches_per_field[field] = n_patches_i
            self.n_patches += n_patches_i
            for i in range(n1):
                for j in range(n2):
                    self.patch_coords.append(
                        (
                            field,
                            [
                                self.stride * i,
                                self.stride * i + self.patch_side,
                                self.stride * j,
                                self.stride * j + self.patch_side,
                            ],
                            [self.stride * (i + 1), self.stride * (j + 1)],
                        )
                    )

    def __len__(self) -> int:
        return self.n_patches

    def __getitem__(self, idx: int):
        field, limits, _ = self.patch_coords[idx]
        y0, y1, x0, x1 = limits

        sample: Dict[str, torch.Tensor] = {}
        for layer, arr in self.layer_rasters[field].items():
            patch = arr[:, y0:y1, x0:x1]
            patch = (patch - patch.min(axis=(1, 2), keepdims=True)) / (
                patch.max(axis=(1, 2), keepdims=True) - patch.min(axis=(1, 2), keepdims=True)
            )
            sample[layer] = torch.from_numpy(patch)

        label = self.label_rasters[field][y0:y1, x0:x1]
        sample["label"] = torch.from_numpy(label).float()
        sample["field"] = field

        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for field in ("field_2021", "field_2022"):
            with rasterio.open(
                tmp / f"{field}_DEM.tif",
                "w",
                driver="GTiff",
                height=120,
                width=120,
                count=1,
                dtype="float32",
            ) as dst:
                dst.write(np.random.rand(1, 120, 120).astype(np.float32))
            with rasterio.open(
                tmp / f"{field}_S2.tif",
                "w",
                driver="GTiff",
                height=120,
                width=120,
                count=4,
                dtype="float32",
            ) as dst:
                dst.write(np.random.rand(4, 120, 120).astype(np.float32))
            with rasterio.open(
                tmp / f"{field}_yield.tif",
                "w",
                driver="GTiff",
                height=120,
                width=120,
                count=1,
                dtype="float32",
            ) as dst:
                dst.write(np.random.rand(1, 120, 120).astype(np.float32))
        with open(tmp / "train.txt", "w") as f:
            f.write("field_2021\nfield_2022\n")

        ds = FieldPatchDataset(str(tmp), train=True, layers=("DEM", "S2"), label="yield", patch_side=32, stride=16)
        print("n_patches:", len(ds))
        sample = ds[0]
        print("keys:", sorted(sample.keys()))
        print("DEM:", tuple(sample["DEM"].shape), "S2:", tuple(sample["S2"].shape))
        print("label:", tuple(sample["label"].shape), "field:", sample["field"])
