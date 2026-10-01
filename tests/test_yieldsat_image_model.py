"""YieldSAT image pipeline contracts (YI-01..YI-05) on a synthetic image dataset
with the real HDF5/patch-table layout."""

import json

import h5py
import numpy as np
import pytest
import torch

from dataset.yieldsat_image_dataset import (
    RGB_IDX, WEATHER_IDX, ImageNormalizer, TileReader, YieldSATImageDataset, collate_tiles,
    load_tile_table, select_observations, temporal_valid, tiles_for_split,
)
from dataset.yieldsat_schema import STATIC_CHANNELS, TEMPORAL_CHANNELS

SEEDING = 18000.0


def _write_country(root, country, seasons, tiles_per_season=2, seed=0):
    rng = np.random.default_rng(seed)
    n = len(seasons) * tiles_per_season
    d = root / country
    d.mkdir(parents=True)
    temporal = rng.normal(1000, 100, (n, 64, 64, 24, len(TEMPORAL_CHANNELS))).astype(np.float32)
    static = rng.normal(50, 5, (n, 64, 64, len(STATIC_CHANNELS))).astype(np.float32)
    times = np.full((n, 64, 64, 24), np.nan, np.float32)
    times[..., 4:20] = SEEDING + np.arange(16) * 20.0             # dated slots 4..19
    temporal[~np.isfinite(times)] = np.nan
    temporal[:, :, :, 9, :3] = np.nan                              # a cloudy slot (no RGB)
    target = rng.uniform(1, 9, (n, 64, 64)).astype(np.float32)
    source = np.arange(n * 4096, dtype=np.int32).reshape(n, 64, 64)
    source[:, 48:, :] = -1                                          # padded bottom rows
    for a in (temporal, static, times):
        a[:, 48:] = np.nan
    target[:, 48:] = np.nan
    valid = np.isfinite(target)
    recs = []
    with h5py.File(d / 'images.h5', 'w') as f:
        for k, v in (('temporal', temporal), ('static', static), ('times', times),
                     ('target', target), ('valid_pixel', valid), ('source_row', source)):
            f[k] = v
    for i in range(n):
        name, crop = seasons[i // tiles_per_season]
        recs.append({'patch_index': i, 'country': country, 'field_shared_name': name, 'crop': crop,
                     'year': 2020, 'row0': 64 * (i % tiles_per_season), 'col0': 0,
                     'physical_field_id': '{}/pf{}'.format(country, i // tiles_per_season)})
    (d / 'patches.jsonl').write_text('\n'.join(json.dumps(r) for r in recs))
    return {'temporal': temporal, 'target': target}


@pytest.fixture(scope='module')
def image_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('img')
    seasons = [('Germany_DUP3_farm1_field{}_rapeseed_2020'.format(i), 'rapeseed') for i in range(4)]
    seasons += [('Germany_DUP3_farm2_field9_wheat_2020', 'wheat')]
    truth = _write_country(root, 'Germany', seasons)
    return root, truth, seasons


def _split(seasons):
    sid = ['Germany/' + s for s, _ in seasons]
    return {'countries': ['Germany'], 'crops': ['rapeseed'],
            'partitions': {'train': sid[:2], 'val': [sid[2]], 'test': [sid[3]], 'excluded': [sid[4]]}}


def _days(seasons):
    return {'Germany/' + s: (SEEDING, SEEDING + 300) for s, _ in seasons}


def test_tiles_inherit_point_fold_partitions(image_root):
    root, _, seasons = image_root
    parts = tiles_for_split(load_tile_table(root, ['Germany']), _split(seasons))
    assert {k: len(v) for k, v in parts.items()} == {'train': 4, 'val': 2, 'test': 2}
    assert all(t['crop'] == 'rapeseed' for v in parts.values() for t in v)   # wheat dropped


def test_first_dated_weather_slot_is_invalid():
    temporal = np.ones((2, 2, 24, 16), np.float32)
    times = np.full((2, 2, 24), np.nan, np.float32)
    times[..., 3:] = 1.0
    v = temporal_valid(temporal, times)
    assert not v[..., 3, WEATHER_IDX].any() and v[..., 4, WEATHER_IDX].all()
    assert v[..., 3, RGB_IDX].all()


def test_observation_selection_is_deterministic_for_eval_and_binned_for_training():
    dates = np.arange(24, dtype=float) * 10
    coverage = np.linspace(0.3, 1.0, 24)
    slots = np.arange(2, 22)
    ev = select_observations(slots, dates, coverage, 4, 20, 220)
    assert ev == select_observations(slots, dates, coverage, 4, 20, 220) and len(ev) == 4
    tr = {tuple(select_observations(slots, dates, coverage, 4, 20, 220, np.random.default_rng(s)))
          for s in range(10)}
    assert len(tr) > 1
    assert select_observations([], dates, coverage, 4, 20, 220) == []


def test_normalizer_and_item_masks(image_root):
    root, truth, seasons = image_root
    parts = tiles_for_split(load_tile_table(root, ['Germany']), _split(seasons))
    norm = ImageNormalizer.fit(TileReader(root), parts['train'])
    train_idx = [t['patch_index'] for t in parts['train']]
    y = truth['target'][train_idx]
    assert abs(norm.stats['target']['mean'][0] - np.nanmean(y)) < 1e-3
    ds = YieldSATImageDataset(root, parts['test'], norm, _days(seasons))
    it = ds[0]
    assert it['spec'].shape == (4, 9, 64, 64) and it['soil'].shape == (48, 64, 64)
    assert it['terrain'].shape == (5, 64, 64)                     # aspect sin/cos
    assert it['cell_present'][48:].sum() == 0 and it['dem_mask'][0, 48:].sum() == 0
    assert not it['target_valid'][48:].any()
    assert 9 not in it['obs_slot'].tolist()                     # cloudy slot never chosen
    assert it['weather_mask'][4].sum() == 0                      # first dated weather slot
    cut = YieldSATImageDataset(root, parts['test'], norm, _days(seasons), cutoff_mode='before_harvest',
                               cutoff_days=300 - 100)            # cutoff = seeding + 100
    it2 = cut[0]
    assert all(s == -1 or SEEDING + (s - 4) * 20 <= SEEDING + 100 for s in it2['obs_slot'].tolist())
    batch = collate_tiles([ds[0], ds[1]])
    assert batch['spec'].shape == (2, 4, 9, 64, 64) and batch['country'] == ['Germany', 'Germany']
