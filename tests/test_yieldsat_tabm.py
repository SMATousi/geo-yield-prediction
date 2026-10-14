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


def test_chunked_standardize_matches_nan_reference():
    """standardize_stats / standardize are chunked (dense-series matrices, 2026-10-14); same result as nanmean/nanstd."""
    rng = np.random.default_rng(0)
    v = rng.normal(3, 2, (1000, 7)).astype(np.float32)
    v[rng.random(v.shape) < 0.2] = np.nan
    v[:, 6] = np.nan                                                  # an all-invalid column
    rows = np.sort(rng.choice(1000, 700, replace=False))
    mean, std = standardize_stats(v, rows, chunk=97)
    with np.errstate(invalid='ignore'):
        rm, rs = np.nanmean(v[rows], 0), np.nanstd(v[rows], 0)
    np.testing.assert_allclose(mean[:6], rm[:6], rtol=1e-5)
    np.testing.assert_allclose(std[:6], rs[:6], rtol=1e-5)
    assert mean[6] == 0 and std[6] == 1
    z = standardize(v, mean, std, chunk=97)
    ref = np.where(np.isfinite((v - mean) / std), (v - mean) / std, 0)
    assert z.dtype == np.float32 and np.isfinite(z).all()
    np.testing.assert_allclose(z, ref, rtol=1e-6, atol=1e-6)


def test_lgbm_rows_on_demand_matches_full_matrix():
    """train_lgbm builds feature rows on demand (2026-10-14); predictions equal the full-matrix version."""
    lgb = pytest.importorskip('lightgbm')
    import argparse
    from main_yieldsat_tabm import cap_rows, train_lgbm
    rng = np.random.default_rng(0)
    n = 3000
    values = rng.normal(size=(n, 6)).astype(np.float32)
    values[rng.random(values.shape) < 0.1] = np.nan
    masks = (rng.random((n, 3)) < 0.8).astype(np.uint8)
    y = np.nan_to_num(values[:, 0]) * 2 + rng.normal(size=n)
    pair = {'values': values, 'masks': masks, 'target': y, 'season': np.repeat(np.arange(30), 100)}
    rows = {'train': np.arange(0, 2400), 'val': np.zeros(0, np.int64), 'test': np.arange(2400, n)}
    args = argparse.Namespace(lgbm_rows_per_field=60, val_rows_per_field=0, seed=0, lgbm_threads=2, lgbm_rounds=50)
    pt, _ = train_lgbm(args, pair, rows)
    X = np.concatenate([values, masks.astype(np.float32)], 1)                 # the previous full-matrix path
    tr = cap_rows(pair['season'], rows['train'][np.isfinite(y[rows['train']])], 60, 0)
    params = {'objective': 'regression', 'learning_rate': 0.05, 'num_leaves': 63, 'min_data_in_leaf': 100,
              'feature_fraction': 0.5, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l2': 1.0,
              'num_threads': 2, 'verbose': -1, 'seed': 0}
    ref = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=50).predict(X[rows['test']])
    np.testing.assert_allclose(pt, ref, rtol=1e-6)
