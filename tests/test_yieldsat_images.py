"""Grid-aligned, non-overlapping YieldSAT image planning contracts."""

import numpy as np
import pytest

from yieldsat_build_images import MIN_VALID, plan_field_windows, shift_affine
from yieldsat_export_tif_samples import pick_rgb


def test_windows_are_disjoint_and_coverage_is_per_field():
    # One full tile and one half tile are accepted; the last is below threshold.
    rows = np.repeat(np.arange(64), 64)
    cols = np.tile(np.arange(64), 64)
    rows = np.concatenate([rows, np.repeat(np.arange(32), 64), np.arange(10)])
    cols = np.concatenate([cols, np.tile(np.arange(64, 128), 32), np.arange(128, 138)])
    target = np.ones(len(rows), np.float32)
    windows = list(plan_field_windows(rows, cols, target, width=192, height=64))
    assert [(r, c, len(ids)) for r, c, ids in windows] == [(0, 0, 4096), (0, 64, MIN_VALID)]
    assert set(windows[0][2]).isdisjoint(windows[1][2])


def test_target_validity_and_duplicate_grid_points():
    rows = np.repeat(np.arange(64), 64)
    cols = np.tile(np.arange(64), 64)
    target = np.ones(len(rows), np.float32)
    target[2047:] = np.nan
    assert list(plan_field_windows(rows, cols, target, 64, 64)) == []
    with pytest.raises(ValueError, match='duplicate'):
        list(plan_field_windows([0, 0], [0, 0], [1, 2], 64, 64))
    with pytest.raises(ValueError, match='outside'):
        list(plan_field_windows([64], [0], [1], 64, 64))


def test_tile_affine_is_shifted_on_source_grid():
    assert shift_affine([10, 0, 594140, 0, -10, 6551630], 64, 128) == [
        10, 0, 595420, 0, -10, 6550990]


def test_rgb_preview_selects_best_dated_slot_and_transparent_invalid_pixels():
    values = np.full((64, 64, 24, 16), np.nan, np.float32)
    dates = np.full((64, 64, 24), np.nan, np.float32)
    valid = np.zeros((64, 64), dtype=bool)
    valid[0, :3] = True
    values[0, 0, 0, :3] = [1, 2, 3]
    dates[0, 0, 0] = 20000
    values[0, :3, 1, :3] = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    dates[0, :3, 1] = 20001
    slot, count, rgba = pick_rgb(values, dates, valid, [0, 1, 2])
    assert (slot, count, rgba.shape) == (1, 3, (4, 64, 64))
    assert np.count_nonzero(rgba[3]) == 3
    assert rgba[3, 1, 0] == 0
