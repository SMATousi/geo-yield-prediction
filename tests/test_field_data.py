import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from dataset.collate import collate_field_samples
from dataset.field_roi import load_field_roi
from dataset.field_yield_dataset import FieldYieldDataset
from dataset.heterogeneous_modality_loader import HeterogeneousModalityLoader


def write_raster(path, values, *, resolution=10, crs="EPSG:32615", nodata=None):
    values = np.asarray(values)
    if values.ndim == 2:
        values = values[None]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[2],
        height=values.shape[1],
        count=values.shape[0],
        dtype=values.dtype,
        crs=crs,
        transform=from_origin(500000, 4000000, resolution, resolution),
        nodata=nodata,
    ) as dst:
        dst.write(values)


@pytest.fixture
def field_root(tmp_path):
    (tmp_path / "train.txt").write_text("alpha_2021\nbeta_2022\n", encoding="utf-8")
    for field in ("alpha_2021", "beta_2022"):
        write_raster(tmp_path / f"{field}_boundary.tif", np.ones((4, 4), dtype=np.uint8))
        write_raster(
            tmp_path / f"{field}_yield.tif",
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            resolution=20,
        )
        write_raster(tmp_path / f"{field}_DEM.tif", np.full((4, 4), 7, dtype=np.float32))
    write_raster(
        tmp_path / "alpha_2021_crop.tif",
        np.array([[1, 2], [3, 4]], dtype=np.uint8),
        resolution=20,
    )
    sar = np.stack([np.full((4, 4), i, dtype=np.float32) for i in (10, 20, 30, 40)])
    write_raster(tmp_path / "alpha_2021_SAR.tif", sar)
    np.save(tmp_path / "alpha_2021_SOIL.npy", np.array([0.25, 0.75], dtype=np.float32))
    return tmp_path


def test_field_roi_uses_projected_boundary_grid(field_root):
    template, projected, lonlat, mask = load_field_roi(
        field_root / "alpha_2021_boundary.tif", "alpha_2021", 10
    )
    assert (template["height"], template["width"]) == (4, 4)
    assert template["crs"].to_epsg() == 32615
    assert mask.shape == (4, 4)
    assert mask.sum() == 16
    assert projected.left == 500000
    assert -180 <= lonlat[0] <= 180


def test_field_yield_dataset_registers_yield_and_categorical_without_interpolation(field_root):
    dataset = FieldYieldDataset(
        field_root, "train.txt", layers=("DEM", "crop"), out_resolution_m=10
    )
    first, second = dataset[0], dataset[1]
    assert len(dataset) == 2
    assert first["field"] == "alpha_2021"
    assert first["yield"].shape == (4, 4)
    assert np.isfinite(first["yield"]).all()
    assert first["DEM"].shape == (1, 4, 4)
    np.testing.assert_allclose(first["DEM"], 7)
    np.testing.assert_array_equal(
        first["crop"][0],
        np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]),
    )
    assert first["available"].tolist() == [1.0, 1.0]
    assert "crop" not in second
    assert second["available"].tolist() == [1.0, 0.0]


def test_heterogeneous_loader_preserves_sar_frame_major_order(field_root):
    dataset = HeterogeneousModalityLoader(
        field_root,
        "train.txt",
        layers=(("DEM", "raster"), ("SAR", "sar"),
                ("SOIL", "tabular"), ("crop", "categorical")),
    )
    first, second = dataset[0], dataset[1]
    assert first["SAR"].shape == (2, 2, 4, 4)
    np.testing.assert_array_equal(first["SAR"][:, :, 0, 0], [[10, 20], [30, 40]])
    np.testing.assert_array_equal(first["SOIL"], [0.25, 0.75])
    assert first["crop"].shape == (4, 4)
    assert np.issubdtype(first["crop"].dtype, np.integer)
    assert first["available"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert second["available"].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_collate_preserves_availability_and_zero_fills_missing():
    sample_a = {
        "dem": torch.ones(1, 2, 2),
        "sar": torch.full((2, 2, 2), 3.0),
        "yield": torch.ones(1, 2, 2),
        "field": "a",
    }
    sample_b = {
        "dem": torch.full((1, 2, 2), 2.0),
        "yield": torch.zeros(1, 2, 2),
        "field": "b",
    }
    batch = collate_field_samples([sample_a, sample_b])
    assert batch["inputs"]["sar"].shape == (2, 2, 2, 2)
    assert batch["available"]["sar"].tolist() == [1.0, 0.0]
    assert batch["available"]["dem"].tolist() == [1.0, 1.0]
    assert torch.count_nonzero(batch["inputs"]["sar"][1]) == 0
    assert batch["yield"].shape == (2, 1, 2, 2)
    assert batch["field"] == ["a", "b"]
