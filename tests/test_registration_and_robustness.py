import numpy as np
import pytest
import torch

from dataset.block_retiler import load_block_data, retile_block
from dataset.nearest_reproject import build_nearest_index, reproject_layers, reproject_nearest
from util.modality_robustness import evaluate_modality_robustness, summarize_robustness


def test_nearest_registration_preserves_source_cells_and_missing_values():
    index = build_nearest_index(
        np.array([0.0, 1.0]), np.array([0.0, 1.0]),
        np.array([0.0, 1.0, 5.0]), np.array([0.0, 1.0]),
        max_dist_deg=0.1,
    )
    np.testing.assert_array_equal(index, [0, 1, 2, 3, -1, -1])
    source = np.ma.array([[1.0, 2.0], [3.0, 4.0]], mask=[[False, True], [False, False]])
    result = reproject_nearest(source, index, 3, 2, nodata=-99)
    np.testing.assert_array_equal(result, [[1, -99], [3, 4], [-99, -99]])
    layers = reproject_layers([("a", source.filled(-99)), ("b", np.ones((2, 2)))], index, 3, 2, nodata=-99)
    assert set(layers) == {"a", "b"}
    np.testing.assert_array_equal(layers["b"][-1], [-99, -99])


def test_block_retiling_preserves_spatial_alignment_across_arrays(tmp_path):
    bands = np.arange(2 * 4 * 4 * 2, dtype=np.float32).reshape(2, 4, 4, 2)
    masks = np.arange(2 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4)
    doys = np.array([20, 40], dtype=np.int32)
    files = {}
    for name, data in {"bands": bands, "masks": masks, "doys": doys}.items():
        path = tmp_path / f"{name}.npy"
        np.save(path, data)
        files[name] = {"path": str(path)}
    block = load_block_data(files, 0, 4, 0, 4)
    patches = retile_block(block, patch_size=2)
    assert len(patches["bands"]) == len(patches["masks"]) == 4
    np.testing.assert_array_equal(patches["bands"][3], bands[:, 2:4, 2:4, :])
    np.testing.assert_array_equal(patches["masks"][3], masks[:, 2:4, 2:4])
    np.testing.assert_array_equal(patches["doys"][0], doys)


def test_robustness_sweep_enumerates_subsets_and_tracks_metric_changes():
    class AddAvailable(torch.nn.Module):
        def forward(self, inputs, *, available, mode):
            assert mode == "tensor"
            return sum(value for name, value in inputs.items() if available[name])

    model = AddAvailable()
    inputs = {
        "dem": torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4),
        "sar": torch.linspace(1, 2, 16).reshape(1, 1, 4, 4),
    }
    targets = inputs["dem"] + inputs["sar"]
    results = evaluate_modality_robustness(model, inputs, targets)
    assert {tuple(row["modalities"]) for row in results} == {("dem", "sar"), ("dem",), ("sar",)}
    by_subset = {tuple(row["modalities"]): row for row in results}
    assert by_subset[("dem", "sar")]["rmse"] == pytest.approx(0)
    assert by_subset[("dem",)]["rmse"] == pytest.approx(
        np.sqrt(np.mean(inputs["sar"].numpy() ** 2))
    )
    assert by_subset[("sar",)]["rmse"] > 1
    summary = summarize_robustness(results)
    assert [row["num_subsets"] for row in summary] == [1, 2]
    assert summary[1]["best_subset"] == ["dem"]
