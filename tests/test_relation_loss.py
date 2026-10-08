"""Relation-matching loss, clustered sampler and within-field metrics (spec/yieldsat-relational-loss.md)."""
import numpy as np
import torch

from yieldsat_relation_loss import relation_loss, variance_loss, within_field_metrics


def _batch(n_fields=4, k=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    season = torch.arange(n_fields).repeat_interleave(k)
    row = torch.randint(0, 6, (n_fields * k,), generator=g)
    col = torch.randint(0, 6, (n_fields * k,), generator=g)
    target = torch.randn(n_fields * k, generator=g)
    valid = torch.ones(n_fields * k, dtype=torch.bool)
    return target, valid, season, row, col


def test_zero_for_perfect_predictions():
    t, v, s, r, c = _batch()
    assert float(relation_loss(t.clone(), t, v, s, r, c)) == 0.0


def test_invariant_to_per_field_constant_shift():
    t, v, s, r, c = _batch()
    pred = t + 0.3 * torch.randn_like(t)
    shifted = pred + torch.tensor([5.0, -2.0, 0.7, 11.0])[s]      # a different constant per field
    assert torch.allclose(relation_loss(pred, t, v, s, r, c), relation_loss(shifted, t, v, s, r, c), atol=1e-6)


def test_penalizes_flattened_fields():
    t, v, s, r, c = _batch()
    flat = torch.stack([t[s == f].mean() for f in range(4)])[s]  # right level, no within-field pattern
    assert float(relation_loss(flat, t, v, s, r, c)) > 0.1


def test_pairs_across_fields_ignored():
    t, v, s, r, c = _batch()
    pred = t.clone()
    pred[s == 0] += 3.0                                          # only a level change in field 0
    assert float(relation_loss(pred, t, v, s, r, c)) < 1e-10      # float32 rounding only


def test_variance_loss_zero_when_std_matches():
    t, v, s, _, _ = _batch()
    assert float(variance_loss(t + 1.0, t, v, s)) < 1e-10
    assert float(variance_loss(0.5 * t, t, v, s)) > 0


def test_within_field_metrics_extremes():
    rng = np.random.default_rng(0)
    season = np.repeat(np.arange(5), 100)
    target = rng.normal(size=500) + season
    m = within_field_metrics(target + 2.0, target, season)      # perfect pattern, wrong level
    assert abs(m['within_r2'] - 1) < 1e-9 and abs(m['median_var_ratio'] - 1) < 1e-9
    half = np.concatenate([0.5 * (target[season == f] - target[season == f].mean()) for f in range(5)])
    m = within_field_metrics(half, target, season)               # right pattern, half amplitude
    assert abs(m['median_field_r'] - 1) < 1e-9 and abs(m['median_var_ratio'] - 0.5) < 1e-9
    assert 0.7 < m['within_r2'] < 0.8                            # 1 - 0.5^2


def test_cluster_sampler_stays_within_radius():
    from dataset.yieldsat_dataset import FieldClusterBatchSampler

    class DS:
        pass
    ds = DS()
    n = 400
    ds.season_ranges = [(0, 200), (200, 400)]
    ds.countries = ['X']
    ds.country_of = np.zeros(n, dtype=np.int16)
    ds.row = np.arange(n)
    grid = np.stack(np.meshgrid(np.arange(20), np.arange(10), indexing='ij'), -1).reshape(-1, 2)
    ds.index = {'X': {'rows': {'grid_row': np.concatenate([grid[:, 0], grid[:, 0]]),
                               'grid_col': np.concatenate([grid[:, 1], grid[:, 1]])}}}
    s = FieldClusterBatchSampler(ds, batch_size=64, batches_per_epoch=5, cluster=8, radius=2)
    for batch in s:
        b = np.array(batch).reshape(-1, 8)
        for cl in b:
            assert len(set((cl >= 200).tolist())) == 1               # one season per cluster
            gr = ds.index['X']['rows']['grid_row'][cl]
            gc = ds.index['X']['rows']['grid_col'][cl]
            assert (np.abs(gr - gr[0]) <= 2).all() and (np.abs(gc - gc[0]) <= 2).all()
