"""TabM point model and flat feature builder (spec/yieldsat-tabm.md TM-01/TM-02)."""
import json

import numpy as np
import pytest
import torch

from dataset.yieldsat_schema import STATIC_CHANNELS, TEMPORAL_CHANNELS
from dataset.yieldsat_tabular import build_pair, fold_rows, standardize, standardize_stats
from models_yieldsat_tabm import TabMRegressor


@pytest.mark.parametrize('arch,k', [('tabm', 4), ('tabm-mini', 4), ('mlp', 1)])
def test_tabm_members_shapes_and_gradients(arch, k):
    torch.manual_seed(0)
    m = TabMRegressor(10, 3, k=4, n_blocks=2, d_block=16, d_embedding=4, arch_type=arch)
    v, mk, y = torch.randn(32, 10), torch.rand(32, 3) > 0.5, torch.randn(32)
    m.eval()                                                     # dropout off: passes are comparable
    mem = m.members(v, mk)
    assert mem.shape == (32, k)
    torch.testing.assert_close(m(v, mk), mem.mean(1))
    m.train()
    m.loss(v, mk, y).backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    if k > 1:
        assert not torch.allclose(mem[:, 0], mem[:, 1])          # members differ


def _synthetic_artifacts(tmp_path, n_fields=3, rows_per=5):
    country = 'Germany'
    n = n_fields * rows_per
    rng = np.random.default_rng(0)
    cache = tmp_path / 'cache' / country
    index = tmp_path / 'index' / country
    cache.mkdir(parents=True)
    index.mkdir(parents=True)
    T = rng.normal(size=(n, 24, len(TEMPORAL_CHANNELS))).astype(np.float32) + 100
    TM = np.full((n, 24), np.nan, np.float32)
    TM[:, 10:15] = 18000 + np.arange(5) * 30
    T[:, 12, :12] = np.nan                                        # one cloudy slot
    S = rng.normal(size=(n, len(STATIC_CHANNELS))).astype(np.float32)
    S[0, STATIC_CHANNELS.index('curvature')] = 5e9                # corrupt (validity v3)
    np.save(cache / 'temporal.npy', T)
    np.save(cache / 'times.npy', TM)
    np.save(cache / 'static.npy', S)
    fields = [{'season_id': 's{}'.format(i), 'field_shared_name': 'f{}'.format(i), 'crop': 'rapeseed',
               'row_start': i * rows_per, 'row_end': (i + 1) * rows_per} for i in range(n_fields)]
    (index / 'fields.json').write_text(json.dumps(fields))
    np.savez(index / 'rows.npz', target=rng.normal(size=n).astype(np.float32), seeding_day=np.full(n, 17990),
             grid_row=np.arange(n), grid_col=np.zeros(n))
    return country


def test_flat_builder_schema_validity_and_folds(tmp_path):
    country = _synthetic_artifacts(tmp_path)
    d = build_pair(tmp_path, country, 'rapeseed')
    v, m = d['values'], d['masks']
    assert v.shape[1] == len(d['value_names']) and m.shape[1] == len(d['mask_names'])
    assert not any('target' in c or 'yield' in c for c in d['value_names'] + d['mask_names'])
    vi = {c: i for i, c in enumerate(d['value_names'])}
    mi = {c: i for i, c in enumerate(d['mask_names'])}
    # first dated slot's weather is invalid; the cloudy slot's S2 is invalid
    assert np.isnan(v[:, vi['temp_mean@10']]).all() and m[0, mi['valid:weather@10']] == 0
    assert np.isnan(v[:, vi['B04@12']]).all() and m[0, mi['valid:s2@12']] == 0 and m[0, mi['valid:s2@13']] == 1
    assert np.isnan(v[0, vi['curvature']]) and m[0, mi['valid:curvature']] == 0
    assert m[0, mi['dated@11']] == 1 and m[0, mi['dated@3']] == 0
    split = {'partitions': {'train': ['s0', 's1'], 'val': [], 'test': ['s2']}}
    r = fold_rows(d, split)
    assert set(d['season'][r['test']]) == {2} and len(r['train']) == 10
    mean, std = standardize_stats(v, r['train'])
    z = standardize(v, mean, std)
    assert np.isfinite(z).all() and (z[np.isnan(v)] == 0).all()
    np.testing.assert_allclose(np.nanmean(z[r['train']][:, vi['B04@11']]), 0, atol=1e-5)
