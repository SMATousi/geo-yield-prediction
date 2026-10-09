"""Window dataset and the paper's best models (spec/yieldsat-paper-models.md, PM-01/PM-02)."""
import numpy as np
import pytest
import torch

from dataset.yieldsat_window import CENTRE, K, ROT, YieldSATWindowDataset
from dataset.yieldsat_dataset import YieldSATNormalizer
from dataset.yieldsat_source import load_country_index
from dataset.yieldsat_splits import load_field_table
from models_yieldsat_paper import PAPER_MODELS
from tests.test_yieldsat import corpus  # noqa: F401  (pytest fixture)


def _window_ds(corpus, **kw):  # noqa: F811
    root, art, _ = corpus
    fields = load_field_table(art, root, ['Germany', 'Uruguay'])['fields']
    tf = [(f['country'], f['field_code'],
           load_country_index(art, root, f['country'])['rows']['target'][f['row_start']:f['row_end']])
          for f in fields[:2]]
    norm = YieldSATNormalizer.fit(art, tf)
    streams = ['yieldsat_s2', 'yieldsat_weather', 'yieldsat_dem', 'yieldsat_soil', 'yieldsat_coords']
    return YieldSATWindowDataset(root, art, fields, norm, streams=streams, fill_value=-1.0, **kw)


def test_window_index_matches_grid(corpus):  # noqa: F811
    ds = _window_ds(corpus)
    assert (ds.win_items[:, CENTRE] == np.arange(len(ds))).all()
    for i in np.random.default_rng(0).choice(len(ds), 30, replace=False):
        c = ds.countries[ds.country_of[i]]
        rows = ds.index[c]['rows']
        r0, c0 = rows['grid_row'][ds.row[i]], rows['grid_col'][ds.row[i]]
        for k, j in enumerate(ds.win_items[i]):
            dr, dc = k // 5 - 2, k % 5 - 2
            if j >= 0:
                assert ds.season_of[j] == ds.season_of[i]
                assert rows['grid_row'][ds.row[j]] == r0 + dr and rows['grid_col'][ds.row[j]] == c0 + dc
            else:      # no cell of this field season at that position
                s = ds.seasons[ds.season_of[i]]
                a, b = s['row_start'], s['row_end']
                assert not ((rows['grid_row'][a:b] == r0 + dr) & (rows['grid_col'][a:b] == c0 + dc)).any()


def test_window_batch_centre_equals_point_item_and_padding(corpus):  # noqa: F811
    ds = _window_ds(corpus, return_masks=True)
    idx = np.arange(min(len(ds), 48))
    b = ds.get_batch(idx)
    point = super(YieldSATWindowDataset, ds).get_batch(idx)
    for n in point['inputs']:
        assert b['win_inputs'][n].dtype == torch.float16
        assert torch.allclose(b['win_inputs'][n][:, CENTRE].float(), point['inputs'][n].half().float())
        assert torch.equal(b['inputs'][n], point['inputs'][n])
    assert torch.equal(b['target'], point['target'])
    absent = ~b['win_valid']
    for n, v in b['win_inputs'].items():
        if absent.any():
            assert (v[absent] == -1.0).all() and not b['win_masks'][n][absent].any()


def test_rotation_permutes_consistently():
    W = np.arange(K)[None]
    for k in range(4):
        g = np.arange(K).reshape(5, 5)
        assert (W[0, ROT[k]] == np.rot90(g, k).ravel()).all()
    assert (ROT[:, CENTRE] == CENTRE).all()


def test_augmentation_drops_slots_and_keeps_targets(corpus):  # noqa: F811
    ds = _window_ds(corpus, augment=True, temporal_dropout=0.5, return_masks=True)
    idx = np.arange(min(len(ds), 48))
    b = ds.get_batch(idx)
    plain = _window_ds(corpus, return_masks=True).get_batch(idx)
    assert torch.equal(b['target'], plain['target'])
    assert b['win_masks']['yieldsat_s2'].sum() < plain['win_masks']['yieldsat_s2'].sum()


def test_centre_only_streams(corpus):  # noqa: F811
    idx = np.arange(32)
    full = _window_ds(corpus).get_batch(idx)
    part = _window_ds(corpus, window='s2_static').get_batch(idx)
    assert part['win_inputs']['yieldsat_weather'].shape[1] == 1
    assert torch.equal(part['win_inputs']['yieldsat_weather'][:, 0], full['win_inputs']['yieldsat_weather'][:, CENTRE])
    assert torch.equal(part['win_inputs']['yieldsat_s2'], full['win_inputs']['yieldsat_s2'])
    none = _window_ds(corpus, window='none').get_batch(idx)
    assert all(v.shape[1] == 1 for v in none['win_inputs'].values())


@pytest.mark.parametrize('name', list(PAPER_MODELS))
def test_paper_models_train_step(corpus, name):  # noqa: F811
    ds = _window_ds(corpus, augment=True, window=PAPER_MODELS[name].window)
    b = ds.get_batch(np.arange(min(len(ds), 32)))
    model = PAPER_MODELS[name](ds.layout)
    model.train()
    pred, loss = model.loss(b)
    loss.backward()
    assert pred.shape == (b['target'].shape[0],) and torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(b, apply_dropout=False)).all()
