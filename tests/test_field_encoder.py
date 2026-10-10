"""Field encoder: context batches, field-grouped sampler and model (spec/yieldsat-field-encoder.md, FE-01/FE-02)."""
import numpy as np
import torch

from dataset.yieldsat_dataset import FieldClusterBatchSampler, YieldSATNormalizer
from dataset.yieldsat_field_context_batch import FieldContextDataset
from dataset.yieldsat_source import load_country_index
from dataset.yieldsat_splits import load_field_table
from models_yieldsat_field import FieldEncoderModel
from tests.test_yieldsat import corpus  # noqa: F401  (pytest fixture)


def _ds(corpus, **kw):  # noqa: F811
    root, art, _ = corpus
    fields = load_field_table(art, root, ['Germany', 'Uruguay'])['fields']
    tf = [(f['country'], f['field_code'],
           load_country_index(art, root, f['country'])['rows']['target'][f['row_start']:f['row_end']])
          for f in fields[:2]]
    norm = YieldSATNormalizer.fit(art, tf)
    return FieldContextDataset(root, art, fields, norm, context_k=8, **kw)


def test_context_items_come_from_the_target_field(corpus):  # noqa: F811
    ds = _ds(corpus)
    idx = np.arange(min(len(ds), 40))
    b = ds.get_batch(idx)
    G = len(np.unique(ds.season_of[idx]))
    assert b['ctx_group'].shape == (G * 8,) and b['tgt_group'].shape == (len(idx),)
    seasons = np.unique(ds.season_of[idx])
    for g, s in enumerate(seasons):
        items = ds.context_items(int(s))
        assert (ds.season_of[items] == s).all()
        assert (b['tgt_group'].numpy()[ds.season_of[idx] == s] == g).all()
    assert 'ctx_target' not in b and not any(k.startswith('ctx_') and 'target' in k for k in b)


def test_eval_context_fixed_train_context_random(corpus):  # noqa: F811
    ev = _ds(corpus)
    s = int(ev.season_of[0])
    assert (ev.context_items(s) == ev.context_items(s)).all()
    assert (ev.context_items(s) == _ds(corpus).context_items(s)).all()
    tr = _ds(corpus, train=True)
    draws = {tuple(tr.context_items(s)) for _ in range(5)}
    assert len(draws) > 1


def test_sampler_groups_clusters_by_field(corpus):  # noqa: F811
    ds = _ds(corpus, train=True)
    sm = FieldClusterBatchSampler(ds, batch_size=32, batches_per_epoch=3, cluster=4, radius=3, clusters_per_field=2)
    for batch in sm:
        seasons = ds.season_of[np.array(batch)]
        assert len(np.unique(seasons)) <= 4                       # 32 / (4 x 2) fields at most


def _model(ds):
    return FieldEncoderModel(ds.layout, seeds=2, embed_dim=32, num_latents=4, depth=1, num_heads=2,
                             modality_embed=8)


def test_model_trains_and_decomposes(corpus):  # noqa: F811
    ds = _ds(corpus, train=True)
    b = ds.get_batch(np.arange(min(len(ds), 24)))
    m = _model(ds)
    m.train()
    pred, loss = m.loss(b)
    loss.backward()
    assert torch.isfinite(loss) and pred.shape == (b['target'].shape[0],)
    assert m.seeds.grad is not None and m.cross.attn.in_proj_weight.grad.abs().sum() > 0
    m.eval()
    with torch.no_grad():
        p, level = m(b, apply_dropout=False, return_level=True)
        g = b['tgt_group']
        for k in g.unique():                                     # one level per field season
            assert torch.allclose(level[g == k], level[g == k][0].expand_as(level[g == k]))


def test_context_order_does_not_matter(corpus):  # noqa: F811
    ds = _ds(corpus)
    idx = np.arange(min(len(ds), 24))
    b = ds.get_batch(idx)
    m = _model(ds).eval()
    perm = torch.cat([torch.randperm(8) + 8 * g for g in range(int(b['ctx_group'].max()) + 1)])
    b2 = dict(b)
    for k in ('ctx_time_features', 'ctx_time_valid', 'ctx_crop', 'ctx_grid_row', 'ctx_grid_col'):
        b2[k] = b[k][perm]
    for k in ('ctx_inputs', 'ctx_masks', 'ctx_available'):
        b2[k] = {n: t[perm] for n, t in b[k].items()}
    with torch.no_grad():
        assert torch.allclose(m(b, apply_dropout=False), m(b2, apply_dropout=False), atol=1e-5)
