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
                     'physical_field_id': '{}/pf{}'.format(country, i // tiles_per_season),
                     'valid_pixels': int(valid[i].sum())})
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


def test_rgb_preprocessing_and_stub_dino_tokens():
    from models_yieldsat_image import SAT_MEAN, SAT_STD, StubDino, rgb_to_dino_input
    rgb = np.full((2, 64, 64, 3), 1500.0, np.float32)            # reflectance 0.15 -> 0.5
    rgb[1, 0, 0] = 6000.0                                          # 0.6 / 0.3 = 2 -> clipped
    valid = np.ones((2, 64, 64), bool)
    valid[0, :8] = False
    x, clip = rgb_to_dino_input(rgb, valid)
    assert x.shape == (2, 3, 64, 64)
    np.testing.assert_allclose(x[0, :, 20, 20], (0.5 - np.array(SAT_MEAN)) / np.array(SAT_STD), rtol=1e-5)
    assert np.all(x[0, :, :8] == 0)                               # missing -> mean -> 0
    assert clip[0] == 0 and clip[1] > 0
    tok = StubDino()(torch.from_numpy(x))
    assert tok.shape == (2, 16, 1024)                             # 4x4 patch tokens


def test_dino_cache_build_and_lookup(image_root, tmp_path, monkeypatch):
    import sys
    import yieldsat_image_dino_cache as dc
    from dataset.yieldsat_image_dataset import DinoFeatureCache
    root, _, seasons = image_root
    monkeypatch.setattr(sys, 'argv', ['x', '--image-root', str(root), '--out', str(tmp_path),
                                      '--countries', 'Germany', '--device', 'cpu', '--stub'])
    dc.main()
    cache_dir = next(p for p in tmp_path.iterdir() if p.is_dir())
    cache = DinoFeatureCache(cache_dir, expected_revision='stub0000')
    feats, valid = cache.get('Germany', 0, [4, 9, 12, -1])
    assert feats.shape == (4, 16, 1024)
    assert valid.tolist() == [1, 0, 1, 0]                         # slot 9 has no RGB
    with pytest.raises(ValueError):
        DinoFeatureCache(cache_dir, expected_revision='f692fa42')
    parts = tiles_for_split(load_tile_table(root, ['Germany']), _split(seasons))
    norm = ImageNormalizer.fit(TileReader(root), parts['train'])
    it = YieldSATImageDataset(root, parts['test'], norm, _days(seasons), dino_cache=cache)[0]
    assert it['dino'].shape == (4, 16, 1024) and it['dino_valid'].sum() == it['obs_valid'].sum()


def _batch(B=2, K=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    ones = lambda *s: torch.ones(*s)
    b = {'dino': r(B, K, 16, 1024), 'dino_valid': ones(B, K), 'obs_valid': ones(B, K),
         'obs_time': torch.rand(B, K, 3, generator=g), 'rgb_mask': ones(B, K, 64, 64),
         'spec': r(B, K, 9, 64, 64), 'spec_mask': ones(B, K, 9, 64, 64),
         'weather': r(B, 24, 4), 'weather_mask': ones(B, 24, 4), 'weather_time': torch.rand(B, 24, 3, generator=g),
         'dem': r(B, 1, 64, 64), 'dem_mask': ones(B, 1, 64, 64),
         'terrain': r(B, 5, 64, 64), 'terrain_mask': ones(B, 5, 64, 64),
         'soil': r(B, 48, 64, 64), 'soil_mask': ones(B, 48, 64, 64),
         'crop': torch.zeros(B, dtype=torch.long), 'target': r(B, 64, 64),
         'target_valid': torch.ones(B, 64, 64, dtype=torch.bool)}
    b['weather_mask'][:, :5] = 0
    return b


def _small(**kw):
    from models_yieldsat_image import YieldSATImageModel
    torch.manual_seed(0)
    return YieldSATImageModel(embed_dim=64, num_latents=8, depth=2, num_heads=4, **kw)


def test_image_model_shapes_and_gradients_reach_every_encoder():
    m = _small()
    b = _batch()
    pred, loss = m.loss(b)
    assert pred.shape == (2, 64, 64) and torch.isfinite(loss)
    loss.backward()
    for name in ('dino_proj', 'spec_enc', 'weather_enc', 'dem_enc', 'terrain_enc', 'soil_cell',
                 'soil_enc', 'crop_embed', 'fusion', 'decoder', 'ups', 'head'):
        grads = [p.grad for p in getattr(m, name).parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), name


def test_image_model_s2_only_has_no_adm_encoders():
    m = _small(inputs='s2')
    assert m.streams == ('dino', 'spec') and not hasattr(m, 'soil_enc')
    b = {k: v for k, v in _batch().items() if not k.startswith(('weather', 'dem', 'terrain', 'soil'))}
    assert m(b).shape == (2, 64, 64)


def test_image_model_handles_missing_observations_and_padded_cells():
    m = _small().eval()
    b = _batch()
    for k in ('obs_valid', 'dino_valid'):
        b[k][0] = 0                                    # tile 0: no optical observation at all
    b['spec_mask'][0] = 0
    b['rgb_mask'][0] = 0
    for k in ('dem_mask', 'terrain_mask', 'soil_mask'):
        b[k][1, :, 48:] = 0                            # tile 1: padded bottom rows
    b['target_valid'][1, 48:] = False
    b['target'][1, 48:] = float('nan')
    with torch.no_grad():
        pred, loss = m.loss(b)
        assert torch.isfinite(pred).all() and torch.isfinite(loss)  # output covers padded cells too
        # all streams of a sample masked -> fallback token, still finite
        empty = {k: (torch.zeros_like(v) if k.endswith(('mask', 'valid')) and k != 'target_valid' else v)
                 for k, v in b.items()}
        assert torch.isfinite(m(empty)).all()


def test_image_model_masked_tokens_do_not_matter_and_valid_ones_do():
    m = _small().eval()
    b = _batch()
    b['obs_valid'][:, 3] = 0
    b['dino_valid'][:, 3] = 0
    b['spec_mask'][:, 3] = 0
    with torch.no_grad():
        base = m(b)
        c = {k: v.clone() for k, v in b.items()}
        c['dino'][:, 3] += torch.randn(2, 16, 1024)                        # masked observation changes
        assert torch.allclose(m(c), base, atol=1e-5)
        c['dino'][:, 0] += torch.randn(2, 16, 1024)                        # valid observation changes
        assert not torch.allclose(m(c), base, atol=1e-4)


def test_tile_balanced_mse_weights_tiles_equally():
    from models_yieldsat_image import tile_balanced_mse
    pred = torch.zeros(2, 64, 64)
    target = torch.zeros(2, 64, 64)
    target[0] = 1.0
    valid = torch.zeros(2, 64, 64, dtype=torch.bool)
    valid[0, :1, :1] = True                            # 1 cell, error 1
    valid[1] = True                                    # 4096 cells, error 0
    assert abs(tile_balanced_mse(pred, target, valid).item() - 0.5) < 1e-6


def test_image_model_on_dataset_batch_with_stub_cache(image_root, tmp_path, monkeypatch):
    import sys
    import yieldsat_image_dino_cache as dc
    from dataset.yieldsat_image_dataset import DinoFeatureCache
    root, _, seasons = image_root
    monkeypatch.setattr(sys, 'argv', ['x', '--image-root', str(root), '--out', str(tmp_path),
                                      '--countries', 'Germany', '--device', 'cpu', '--stub'])
    dc.main()
    cache = DinoFeatureCache(next(p for p in tmp_path.iterdir() if p.is_dir()))
    parts = tiles_for_split(load_tile_table(root, ['Germany']), _split(seasons))
    norm = ImageNormalizer.fit(TileReader(root), parts['train'])
    ds = YieldSATImageDataset(root, parts['train'], norm, _days(seasons), train=True, dino_cache=cache)
    from dataset.yieldsat_image_dataset import cast_batch
    b = collate_tiles([ds[0], ds[1]])
    assert b['soil'].dtype == torch.float16 and b['soil_mask'].dtype == torch.bool
    b = cast_batch(b)
    assert b['soil'].dtype == torch.float32 and b['target_valid'].dtype == torch.bool
    pred, loss = _small().loss(b)
    loss.backward()
    assert pred.shape == (2, 64, 64) and torch.isfinite(loss)


def _artifacts(tmp_path, seasons, split_name='fold_test'):
    art = tmp_path / 'art'
    (art / 'index' / 'Germany').mkdir(parents=True)
    (art / 'splits').mkdir()
    fields = [{'season_id': 'Germany/' + s, 'country': 'Germany', 'crop': c, 'field_shared_name': s,
               'seeding_day': SEEDING, 'harvest_day': SEEDING + 300} for s, c in seasons]
    (art / 'index' / 'Germany' / 'fields.json').write_text(json.dumps(fields))
    split = dict(_split(seasons), label='synthetic fold', scheme='test', partition_hash='h')
    (art / 'splits' / '{}.json'.format(split_name)).write_text(json.dumps(split))
    return art


def test_image_training_entry_end_to_end_with_donor_warm_start(image_root, tmp_path, monkeypatch):
    import sys
    import main_yieldsat_image as mi
    import yieldsat_image_dino_cache as dc
    root, _, seasons = image_root
    art = _artifacts(tmp_path, seasons)
    cache = tmp_path / 'cache'
    monkeypatch.setattr(sys, 'argv', ['x', '--image-root', str(root), '--out', str(cache),
                                      '--countries', 'Germany', '--device', 'cpu', '--stub'])
    dc.main()
    cache_dir = str(next(p for p in cache.iterdir() if p.is_dir()))
    small = ['--embed_dim', '32', '--num_latents', '4', '--depth', '2', '--num_heads', '4',
             '--epochs', '2', '--batch_size', '2', '--min_steps_per_epoch', '2', '--num_workers', '0',
             '--device', 'cpu', '--artifact_root', str(art), '--image_root', str(root),
             '--dino_cache', cache_dir, '--dino_revision', 'stub0000', '--countries', 'Germany']
    donor = mi.main(mi.get_args_parser().parse_args(
        small + ['--donor', '--output_dir', str(tmp_path / 'donor'), '--streams', *mi.S2_ADM]))
    assert donor['mode'] == 'donor' and 'test' not in donor and donor['tiles']['val'] > 0
    rep = mi.main(mi.get_args_parser().parse_args(
        small + ['--crops', 'rapeseed', '--split', 'fold_test', '--output_dir', str(tmp_path / 'run'),
                 '--streams', *mi.S2_ADM, '--init_ckpt', str(tmp_path / 'donor' / 'checkpoint_best.pth'),
                 '--data_contract', 'yieldsat_preprocessed_v1', '--save_maps', '1']))
    assert rep['transfer']['loaded'] > 0 and all('crop_embed' not in k for k in rep['transfer']['skipped'] if k)
    assert rep['tiles'] == {'train': 4, 'val': 2, 'test': 2}
    npz = np.load(tmp_path / 'run' / 'test_predictions.npz')
    assert set(npz.files) == {'pred', 'target', 'season', 'grid_row', 'grid_col', 'season_names'}
    assert len(npz['pred']) == rep['test']['rows_evaluated'] == 2 * 48 * 64   # valid cells only
    assert np.isfinite(npz['pred']).all() and npz['grid_row'].max() == 64 + 47
    assert rep['test']['overall']['pixel']['rmse'] > 0
    s2 = mi.main(mi.get_args_parser().parse_args(
        small + ['--crops', 'rapeseed', '--split', 'fold_test', '--output_dir', str(tmp_path / 's2'),
                 '--streams', 'yieldsat_s2', '--epochs', '1']))
    assert s2['inputs'] == 's2' and s2['streams'] == ['dino', 'spec']


def test_sampler_carries_epoch_so_persistent_workers_resample(image_root):
    from dataset.yieldsat_image_dataset import SeasonBalancedSampler
    root, _, seasons = image_root
    parts = tiles_for_split(load_tile_table(root, ['Germany']), _split(seasons))
    s = SeasonBalancedSampler(parts['train'], 6, seed=0)
    s.set_epoch(3)
    idx = list(s)
    assert all(e == 3 for _, e in idx)
    norm = ImageNormalizer.fit(TileReader(root), parts['train'])
    ds = YieldSATImageDataset(root, parts['train'], norm, _days(seasons), train=True)
    picks = {tuple(ds[(0, e)]['obs_slot'].tolist()) for e in range(8)}
    assert len(picks) > 1 and ds[(0, 2)]['obs_slot'].tolist() == ds[(0, 2)]['obs_slot'].tolist()


def test_image_vs_point_comparison_matches_identical_cells(tmp_path):
    import yieldsat_image_compare as cmp
    names = np.array(['GER_a', 'GER_b', 'GER_c'])
    rng = np.random.default_rng(0)
    # point: 3 seasons x 30 cells; image tiles cover 20 cells of each season
    ps = np.repeat([0, 1, 2], 30)
    pr = np.tile(np.arange(30), 3)
    y = rng.uniform(2, 8, 90)
    point = dict(pred=y + 0.5, target=y, season=ps, grid_row=pr, grid_col=np.zeros(90, int), season_names=names)
    sel = pr < 20
    # image lists seasons in another order
    order = np.array([2, 0, 1])
    inv = np.argsort(order)
    image = dict(pred=y[sel] + 0.1, target=y[sel], season=inv[ps[sel]], grid_row=pr[sel],
                 grid_col=np.zeros(sel.sum(), int), season_names=names[order])
    rel = 'paper/cv10_season_paper_s0/s2_adm/{}_seed0/GER-R/fold00'
    for root, tag, d in ((tmp_path / 'img', 'image', image), (tmp_path / 'pt', 'ours', point)):
        f = root / rel.format(tag)
        f.mkdir(parents=True)
        np.savez(f / 'test_predictions.npz', **d)
    rows, missing = cmp.collect(tmp_path / 'img', tmp_path / 'pt')
    assert not missing and len(rows) == 1
    r = rows[0]
    assert r['n_matched'] == 60 and abs(r['point_coverage'] - 60 / 90) < 1e-9 and r['image_coverage'] == 1
    assert abs(r['image']['pixel_rmse'] - 0.1) < 1e-9 and abs(r['point']['pixel_rmse'] - 0.5) < 1e-9
    s = cmp.summarize(rows)[0]
    assert s['folds'] == 1 and abs(s['delta_pixel_rmse'] + 0.4) < 1e-9
    image['target'] = image['target'] + 1                    # different cells -> refuse
    np.savez(tmp_path / 'img' / rel.format('image') / 'test_predictions.npz', **image)
    with pytest.raises(ValueError):
        cmp.collect(tmp_path / 'img', tmp_path / 'pt')


def test_cluster_image_runs_donors_and_warm_start():
    import yieldsat_cluster as yc
    suite = {'suite': 's', 'data': {'donor_root': '/pvc/donors'}}
    exp = {'name': 'image', 'warm_start': {'GER-R': 'pooled'}, 'budget': {'epochs': 3}}
    split = {'summary': {'train': {'rows': 1}, 'test': {'rows': 1}}}
    r = yc._make_run(suite, exp, 'image', 'cv', 'paper', 'season', 'paper', 10, 's2', 1, 'GER-R',
                     'Germany', 'rapeseed', 0, 'split0', split)
    assert r['init_ckpt'] == '/pvc/donors/donors/pooled/s2/seed1/checkpoint_best.pth'
    a = r['args']
    assert a[a.index('--init_ckpt') + 1] == r['init_ckpt'] and a[a.index('--lr') + 1] == '0.00015'
    other = yc._make_run(suite, exp, 'image', 'cv', 'paper', 'season', 'paper', 10, 's2', 1, 'GER-W',
                         'Germany', 'wheat', 0, 'split0', split)
    assert other['init_ckpt'] is None and '--init_ckpt' not in other['args']
    tiles = [{'country': 'Brazil', 'crop': 'wheat'}] * 30 + [{'country': 'Brazil', 'crop': 'soybean'}] * 10
    d = yc._make_donor_run(suite, {'name': 'donor'}, {'name': 'bra-w', 'countries': ['Brazil'],
                                                      'crops': ['wheat']}, 's2_adm', 0, tiles)
    assert d['rel_path'] == 'donors/bra-w/s2_adm/seed0' and d['keep_files'] == ['checkpoint_best.pth']
    assert d['n_train'] == 27 and '--donor' in d['args'] and d['pair'] == 'donor-bra-w'
    d['est_seconds'] = yc.estimate_seconds(d, {'image': 1.0})
    # 60 epochs x 20 steps x 16 tiles at the reference rate (+ eval, start-up)
    assert abs(d['est_seconds'] - (30 + 60 * (320 / yc.REF_THROUGHPUT['image'] + 2))) < 1e-6


def test_pool_defers_jobs_until_donor_checkpoints_exist(tmp_path):
    import yieldsat_cluster as yc
    ck = tmp_path / 'donor.pth'
    runs = {'a': {'init_ckpt': str(ck)}, 'b': {'init_ckpt': None}}
    assert yc._job_ready({'run_ids': ['b']}, runs)
    assert not yc._job_ready({'run_ids': ['a', 'b']}, runs)
    ck.write_bytes(b'x')
    assert yc._job_ready({'run_ids': ['a', 'b']}, runs)


def test_plan_refuses_warm_starts_without_donor_runs():
    import yieldsat_cluster as yc
    donor = {'protocol': 'donor', 'pair': 'donor-pooled', 'inputs': 's2_adm', 'seed': 0}
    target = {'protocol': 'cv10', 'donor': 'pooled', 'inputs': 's2', 'seed': 0}
    with pytest.raises(SystemExit):
        yc.check_warm_starts({}, [donor, target])
    yc.check_warm_starts({}, [donor, dict(target, inputs='s2_adm')])


# ---- v2 (plan v2: YI-09 series branch, YI-10 level head, YI-11 coverage) ----------

def test_series_stream_masks_undated_and_post_cutoff_slots(image_root):
    root, _, seasons = image_root
    parts = tiles_for_split(load_tile_table(root, ['Germany']), _split(seasons))
    norm = ImageNormalizer.fit(TileReader(root), parts['train'])
    it = YieldSATImageDataset(root, parts['test'], norm, _days(seasons), with_series=True)[0]
    assert it['series'].shape == (64, 64, 24, 12) and it['series'].dtype == np.float16
    m = it['series_mask']
    assert m.shape == (64, 64, 24) and not m[..., :4].any() and not m[..., 20:].any()  # undated
    assert not m[48:].any() and m[:48, :, 4:20].all()                         # padded rows
    assert float(it['series_days'][0, 0, 5]) == 20.0                          # slot 5 = seeding + 20
    cut = YieldSATImageDataset(root, parts['test'], norm, _days(seasons), with_series=True,
                               cutoff_mode='before_harvest', cutoff_days=200)[0]   # cutoff = +100
    assert cut['series_mask'][..., :10].any() and not cut['series_mask'][..., 10:].any()
    assert 'series' not in YieldSATImageDataset(root, parts['test'], norm, _days(seasons))[0]


def test_slot_coverage_relative_to_present_cells():
    from dataset.yieldsat_image_dataset import slot_dates
    times = np.full((64, 64, 24), 100.0)
    rgb_ok = np.zeros((64, 64, 24), bool)
    rgb_ok[:8, :8, 3] = True                          # 64 of 4096 cells, all present cells
    present = np.zeros((64, 64), bool)
    present[:8, :8] = True
    assert slot_dates(times, rgb_ok)[2][3] == 64 / 4096
    assert slot_dates(times, rgb_ok, present)[2][3] == 1.0


def test_series_encoder_ignores_masked_slots_and_empty_cells():
    from models_yieldsat_image import SeriesEncoder
    torch.manual_seed(0)
    enc = SeriesEncoder().eval()
    s = torch.randn(1, 4, 4, 24, 12)
    m = torch.zeros(1, 4, 4, 24)
    m[..., 5:15] = 1
    m[:, 0, 0] = 0                                     # a cell never observed
    d = torch.arange(24.0).view(1, 1, 1, 24).expand(1, 4, 4, 24) * 10
    doy = torch.tensor([100.0])
    with torch.no_grad():
        a = enc(s, m, d, doy)
        s2 = s.clone()
        s2[..., 15:, :] += 3.0                         # change unobserved slots only
        assert torch.allclose(a, enc(s2, m, d, doy), atol=1e-6)       # unobserved values ignored
        s3 = s.clone()
        s3[..., 7, :] += 3.0                                                # observed slot changes
        assert not torch.allclose(a, enc(s3, m, d, doy), atol=1e-4)
    assert a.shape == (1, 32, 4, 4) and torch.all(a[0, :, 0, 0] == 0)


def test_level_head_decomposes_prediction_and_trains():
    m = _small(use_series=True, level_head=True).train()
    b = _batch()
    B = 2
    b['series'] = torch.randn(B, 64, 64, 24, 12)
    b['series_mask'] = torch.rand(B, 64, 64, 24) > 0.3
    b['series_days'] = torch.arange(24.0).view(1, 1, 1, 24).expand(B, 64, 64, 24) * 10
    b['seeding_doy'] = torch.tensor([100.0, 300.0])
    b['cell_present'] = torch.ones(B, 64, 64)
    b['cell_present'][1, 48:] = 0
    b = {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.bool and k != 'target_valid' else v)
         for k, v in b.items()}
    pred, level = m(b, return_level=True)
    p = b['cell_present']
    centered = (pred * p).flatten(1).sum(1) / p.flatten(1).sum(1)
    assert torch.allclose(centered, level, atol=1e-4)                   # mean over present = level
    _, loss = m.loss(b)
    loss.backward()
    for name in ('series_cell', 'series_enc', 'level'):
        assert any(q.grad is not None and q.grad.abs().sum() > 0 for q in getattr(m, name).parameters()), name
    assert 'series' in m.streams and 'series' in m.dense


def test_image_entry_v2_flags_end_to_end(image_root, tmp_path, monkeypatch):
    import sys
    import main_yieldsat_image as mi
    import yieldsat_image_dino_cache as dc
    root, _, seasons = image_root
    art = _artifacts(tmp_path, seasons)
    cache = tmp_path / 'cache'
    monkeypatch.setattr(sys, 'argv', ['x', '--image-root', str(root), '--out', str(cache),
                                      '--countries', 'Germany', '--device', 'cpu', '--stub'])
    dc.main()
    args = ['--embed_dim', '32', '--num_latents', '4', '--depth', '2', '--num_heads', '4',
            '--epochs', '1', '--batch_size', '2', '--min_steps_per_epoch', '2', '--num_workers', '0',
            '--device', 'cpu', '--artifact_root', str(art), '--image_root', str(root),
            '--dino_cache', str(next(p for p in cache.iterdir() if p.is_dir())), '--dino_revision',
            'stub0000', '--countries', 'Germany', '--crops', 'rapeseed', '--split', 'fold_test',
            '--output_dir', str(tmp_path / 'v2'), '--series', '--level_head', '--slot_coverage', 'present',
            '--train_min_valid', '1000000']
    fb = mi.main(mi.get_args_parser().parse_args(args))      # no tile that full: keep all tiles
    assert fb['train_tile_filter'] == 'fallback: all tiles' and fb['tiles']['train'] == 4
    args[-1] = '1'
    args[args.index('--output_dir') + 1] = str(tmp_path / 'v2b')
    rep = mi.main(mi.get_args_parser().parse_args(args))
    assert rep['train_tile_filter'] == 'min_valid 1'
    assert rep['model']['descriptor']['use_series'] and rep['model']['descriptor']['level_head']
    assert 'series' in rep['streams'] and np.isfinite(rep['test']['overall']['pixel']['rmse'])
