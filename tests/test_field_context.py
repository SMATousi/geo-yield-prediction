"""Field-context builder and dataset streams (spec/yieldsat-field-context.md, FC-02/FC-03)."""
import numpy as np
import torch

from dataset.yieldsat_schema import NBR_NAME
from models_yieldsat import YieldSATPointModel
from tests.test_yieldsat import _dataset, corpus  # noqa: F401  (pytest fixture)
from yieldsat_build_field_context import build_country, edge_distance, field_rel, field_s2_stats, REL_POS


def test_field_s2_stats_masks_and_minimum():
    v = np.array([[[1.0]], [[3.0]], [[np.nan]], [[5.0]]])           # 4 cells, 1 slot, 1 band
    m, s, n = field_s2_stats(v)
    assert n[0, 0] == 3 and np.isclose(m[0, 0], 3.0) and np.isclose(s[0, 0], np.std([1, 3, 5]))
    m, _, _ = field_s2_stats(v[:2])                                  # 2 valid cells < MIN_CELLS
    assert np.isnan(m[0, 0])


def test_edge_distance_zero_at_border_and_grows_inside():
    rr, cc = np.meshgrid(np.arange(7), np.arange(7), indexing='ij')
    d = edge_distance(rr.ravel(), cc.ravel()).reshape(7, 7)
    assert d[0].max() == 0 and d[:, 0].max() == 0 and d[3, 3] == 3


def test_field_rel_z_scores():
    rng = np.random.default_rng(0)
    rr, cc = np.meshgrid(np.arange(10), np.arange(10), indexing='ij')
    static = np.zeros((100, 104))
    static[:, REL_POS] = rng.normal(5, 2, size=(100, len(REL_POS)))
    out = field_rel(static, rr.ravel(), cc.ravel())
    assert np.allclose(out[:, :4].mean(0), 0, atol=1e-9) and np.allclose(out[:, :4].std(0), 1, atol=1e-2)
    assert out[:, 4].min() == 0 and out[:, 4].max() <= 1


def test_dataset_streams_match_definition(corpus):  # noqa: F811
    _, art, _ = corpus
    for c in ('Germany', 'Uruguay'):
        build_country(art, c)
    ds, norm, _ = _dataset(corpus, field_context_root=str(art / NBR_NAME))
    plain, _, _ = _dataset(corpus)
    idx = np.arange(min(len(ds), 40))
    b, b0 = ds.get_batch(idx), plain.get_batch(idx)
    for k in b0['inputs']:                                           # existing streams unchanged
        assert torch.equal(b['inputs'][k], b0['inputs'][k])
    assert torch.equal(b['target'], b0['target'])
    # deviation stream = (cell - field mean) / field std, from the raw S2 values of the field's cells
    i = 0
    country = ds.countries[ds.country_of[idx[i]]]
    row = ds.row[idx[i]]
    ctx = ds.field_context[country]
    code = int(ds.index[country]['rows']['field_code'][row])
    f = ctx['pos'][code]
    raw = ds.readers[country].read(np.array([row]))[0][0][:, ds.layout['yieldsat_s2_dev']['positions']]
    m = b['masks']['yieldsat_s2_dev'][i].numpy()
    want = np.clip((raw - ctx['mean'][f]) / ctx['std'][f], -5, 5)
    assert m.any() and np.allclose(b['inputs']['yieldsat_s2_dev'][i].numpy()[m], want[m], atol=1e-4)
    assert b['inputs']['yieldsat_field_rel'].shape == (len(idx), 5)
    model = YieldSATPointModel(ds.layout, embed_dim=32, num_latents=4, depth=1, num_heads=2, modality_embed=8)
    pred, loss = model.loss(b)
    loss.backward()
    assert torch.isfinite(loss) and pred.shape == (len(idx),)
