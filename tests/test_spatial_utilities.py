import json

import numpy as np
import pytest

from dataset.field_patch_dataset import FieldPatchDataset
from dataset.spectral_indices import compute_evi, compute_ndvi
from util.metrics import SpatialYieldMetric, evaluate_regression
from util.spatial_split import cross_location_split

from test_field_data import write_raster


def test_spectral_indices_match_band_and_stack_inputs():
    nir = np.array([[0.8, 0], [0.4, 0.2]], dtype=np.float32)
    red = np.array([[0.2, 0], [0.2, 0.1]], dtype=np.float32)
    blue = np.full((2, 2), 0.05, dtype=np.float32)
    stack = np.stack([nir, red, blue])
    ndvi = compute_ndvi(nir, red)
    evi = compute_evi(nir, red, blue)
    np.testing.assert_allclose(ndvi, compute_ndvi(None, None, stack=stack))
    np.testing.assert_allclose(evi, compute_evi(None, None, None, stack=stack))
    assert ndvi[0, 0] == pytest.approx(0.6)
    assert np.isfinite(ndvi).all() and np.isfinite(evi).all()


def test_patch_dataset_keeps_label_and_input_coordinates_aligned(tmp_path):
    field = "alpha_2021"
    (tmp_path / "train.txt").write_text(field + "\n", encoding="utf-8")
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    write_raster(tmp_path / f"{field}_DEM.tif", values)
    write_raster(tmp_path / f"{field}_yield.tif", values + 10)
    dataset = FieldPatchDataset(tmp_path, layers=("DEM",), patch_side=2, stride=2)
    assert len(dataset) == 4
    first, last = dataset[0], dataset[3]
    assert first["DEM"].shape == (1, 2, 2)
    np.testing.assert_array_equal(first["label"].numpy(), values[:2, :2] + 10)
    np.testing.assert_array_equal(last["label"].numpy(), values[2:, 2:] + 10)
    assert first["field"] == field


def test_dense_metrics_respect_valid_pixel_mask():
    truth = np.array([[1, 2], [3, 4]], dtype=np.float32)
    prediction = np.array([[1, 2], [30, 40]], dtype=np.float32)
    metrics = evaluate_regression(truth, prediction, mask=np.array([[1, 1], [0, 0]], dtype=bool))
    assert metrics["rmse"] == 0
    assert metrics["mae"] == 0
    spatial = SpatialYieldMetric()
    spatial.process(truth[None], truth[None])
    summary = spatial.compute()
    assert summary["spatial_corr"] == pytest.approx(1)
    assert summary["zone_preservation"] == pytest.approx(1)


def test_cross_location_split_keeps_state_groups_disjoint(tmp_path):
    records = [
        {"state_ansi": 10, "year": 2021},
        {"state_ansi": 10, "year": 2022},
        {"state_ansi": 20, "year": 2021},
        {"state_ansi": 20, "year": 2020},
    ]
    path = tmp_path / "fields.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    train, valid, test = cross_location_split(
        path, group_indices={"held": (10,), "train": (20,)},
        test_group="held", test_year=2022, train_years=1,
    )
    assert train == [2]
    assert valid == [0]
    assert test == [1]
