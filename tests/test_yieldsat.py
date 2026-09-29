"""Contract tests for the YieldSAT preprocessed branch (YS-01..YS-09).

A synthetic HDF5 file with the real NetCDF layout is generated per country:
shuffled band order, country-local time origin, crop code 0, a non-identity
row dictionary (as in Brazil), NaN slots and static repetition.
"""

import json
import os

import h5py
import numpy as np
import pytest
import torch

from dataset.yieldsat_cache import (
    FLAG_STATIC_CONFLICT, build_country_cache, canonicalize_block,
    load_cache_manifest,
)
from dataset.yieldsat_dataset import (
    FieldBalancedBatchSampler, YieldSATNormalizer, YieldSATPointDataset, stream_layout,
)
from dataset.yieldsat_schema import (
    ALL_CHANNELS, OPTICAL, SOURCE_FILENAME, STATIC_CHANNELS, TEMPORAL_CHANNELS, decode_codes,
    integer_lookup, parse_field_shared_name, parse_time_origin, resolve_channel_indices,
)
from dataset.yieldsat_source import SnapshotError, build_country_index, load_country_index
from dataset.yieldsat_splits import check_split, load_field_table, make_split
from models_yieldsat import YieldSATPointModel
from util.yieldsat_eval import evaluate_predictions, scatter_to_grid
from yieldsat_objectives import (
    contrastive_negative_mask, forecast_split, mask_observations, require_objective,
)

ORIGIN = {'Germany': '2019-09-22', 'Uruguay': '2018-11-25'}
FIELDS = {
    'Germany': [('Germany_DUP3_farm1_field10_rapeseed_2019', 0, 0),
                ('Germany_DUP3_farm1_field11_wheat_2019', 1, 0),
                ('Germany_DUP3_farm2_field12_wheat_2020', 1, 1),
                ('Germany_DUP3_farm3_field13_rapeseed_2020', 0, 1)],
    'Uruguay': [('Uruguay_DUP4_farm1_field1_soybean_2019', 0, 0),
                ('Uruguay_DUP4_farm2_field2_soybean_2019', 0, 0),
                ('Uruguay_DUP4_farm3_field3_soybean_2020', 0, 1)],
}
CELLS = 12  # rows per field (3 x 4 grid)


def _write_country(root, country, seed):
    rng = np.random.default_rng(seed)
    order = list(ALL_CHANNELS)
    rng.shuffle(order)
    fields = FIELDS[country]
    n = len(fields) * CELLS
    origin = np.datetime64(ORIGIN[country])
    canon_t = rng.normal(1000, 200, size=(n, 24, len(TEMPORAL_CHANNELS))).astype(np.float32)
    canon_s = rng.normal(50, 10, size=(n, len(STATIC_CHANNELS))).astype(np.float32)
    times = np.full((n, 24), np.nan)
    seeding, harvest = [], []
    for fi in range(len(fields)):
        s = slice(fi * CELLS, (fi + 1) * CELLS)
        base = 30 * fi
        times[s, 6:20] = base + np.arange(14) * 20.0          # dated slots 6..19
        seeding.append(str(origin + np.timedelta64(base + 100, 'D')))
        harvest.append(str(origin + np.timedelta64(base + 200, 'D')))
    canon_t[~np.isfinite(times)] = np.nan
    canon_t[:, 8, :len(OPTICAL)] = np.nan                     # cloudy dated slot
    sample = np.empty((n, 24, 120), dtype=np.float32)
    pos = {c: i for i, c in enumerate(order)}
    for j, c in enumerate(TEMPORAL_CHANNELS):
        sample[:, :, pos[c]] = canon_t[:, :, j]
    for j, c in enumerate(STATIC_CHANNELS):
        sample[:, :, pos[c]] = canon_s[:, None, j]
    sample[:, 0, pos['dem']] = np.nan                         # missing slot, static still valid
    target = rng.uniform(1, 9, size=n).astype(np.float32)
    path = root / country / SOURCE_FILENAME
    path.parent.mkdir(parents=True)
    str_dt = h5py.string_dtype()
    with h5py.File(path, 'w') as f:
        f['sample'] = sample
        f['target'] = target
        f['times'] = times
        f['times'].attrs['units'] = 'days since {}'.format(ORIGIN[country])
        f['times'].attrs['calendar'] = 'proleptic_gregorian'
        f.create_dataset('band', data=np.array(order, dtype=object), dtype=str_dt)
        f.create_dataset('index', data=np.array(['{}-{}'.format(country, i) for i in range(n)],
                                                dtype=object), dtype=str_dt)
        codes = np.repeat(np.arange(len(fields)), CELLS)
        f['field_shared_name'] = codes.astype(np.uint16)
        for i, (name, _, _) in enumerate(fields):
            f['field_shared_name'].attrs[str(i)] = name
        farms = sorted({parse_field_shared_name(n_)['farm'] for n_, _, _ in fields})
        f['farm_identifier'] = np.array([farms.index(parse_field_shared_name(fields[c][0])['farm'])
                                         for c in codes], dtype=np.uint8)
        for i, fm in enumerate(farms):
            f['farm_identifier'].attrs[str(i)] = fm
        crops = sorted({parse_field_shared_name(n_)['crop'] for n_, _, _ in fields})
        f['crop'] = np.array([crops.index(parse_field_shared_name(fields[c][0])['crop'])
                              for c in codes], dtype=np.uint8)
        for i, cr in enumerate(crops):
            f['crop'].attrs[str(i)] = cr
        years = sorted({parse_field_shared_name(n_)['year'] for n_, _, _ in fields})
        f['year'] = np.array([years.index(parse_field_shared_name(fields[c][0])['year'])
                              for c in codes], dtype=np.uint8)
        for i, y in enumerate(years):
            f['year'].attrs[str(i)] = y
        f['country'] = np.zeros(n, dtype=np.uint8)
        f['country'].attrs['0'] = country
        f['seeding_date'] = codes.astype(np.uint16)
        f['harvesting_date'] = codes.astype(np.uint16)
        for i in range(len(fields)):
            f['seeding_date'].attrs[str(i)] = seeding[i]
            f['harvesting_date'].attrs[str(i)] = harvest[i]
        f['seeding_date_type'] = np.zeros(n, dtype=np.uint8)
        f['seeding_date_type'].attrs['0'] = 'provided_by_farmer'
        # non-identity row dictionary: code k -> row 2k
        rr = np.tile(np.repeat(np.arange(3), 4), len(fields))
        cc = np.tile(np.tile(np.arange(4), 3), len(fields))
        f['row'] = rr.astype(np.uint16)
        for k in range(3):
            f['row'].attrs[str(k)] = 2 * k
        f['col'] = cc.astype(np.uint16)
        for k in range(4):
            f['col'].attrs[str(k)] = k
        for name, _, _ in fields:
            f.attrs[name + '_<>_yield_ground_truth'] = 'UNKNOWN'
    return {'canon_t': canon_t, 'canon_s': canon_s, 'target': target, 'order': order}


@pytest.fixture(scope='module')
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp('src')
    art = tmp_path_factory.mktemp('art')
    truth = {c: _write_country(root, c, i) for i, c in enumerate(FIELDS)}
    for c in FIELDS:
        build_country_index(root, c, art / 'index' / c)
        build_country_cache(root, c, art, block_rows=7, log_every=0)
    return root, art, truth


# ---- schema / decoding -------------------------------------------------------

def test_channel_resolution_by_name_and_schema_failures():
    names = list(reversed(ALL_CHANNELS))
    idx = resolve_channel_indices(names, ALL_CHANNELS)
    assert [names[i] for i in idx] == list(ALL_CHANNELS)
    with pytest.raises(ValueError, match='lacks'):
        resolve_channel_indices(names[1:], ALL_CHANNELS)
    with pytest.raises(ValueError, match='duplicate'):
        resolve_channel_indices(names + [names[0]], ALL_CHANNELS)
    with pytest.raises(ValueError, match='unexpected'):
        resolve_channel_indices(names + ['B10'], ALL_CHANNELS)


def test_code_zero_is_valid_and_unknown_codes_fail():
    assert list(decode_codes(np.array([0, 1, 0]), {0: 'rapeseed', 1: 'wheat'})) == \
        ['rapeseed', 'wheat', 'rapeseed']
    with pytest.raises(ValueError, match='without dictionary'):
        decode_codes(np.array([2]), {0: 'a', 1: 'b'})
    assert list(integer_lookup({0: 0, 1: 2, 2: 4})[[2, 0]]) == [4, 0]
    assert str(parse_time_origin('days since 2021-10-28', 'proleptic_gregorian')) == '2021-10-28'
    with pytest.raises(ValueError):
        parse_time_origin('hours since 2021-10-28', 'proleptic_gregorian')


# ---- index / cache ------------------------------------------------------------

def test_index_decodes_rows_dates_and_keeps_target_side_separate(corpus):
    root, art, truth = corpus
    idx = load_country_index(art, root, 'Germany')
    rows = idx['rows']
    assert set(np.unique(rows['grid_row'])) == {0, 2, 4}          # decoded, not codes
    assert idx['manifest']['audit']['fields_contiguous']
    assert idx['manifest']['audit']['duplicate_field_row_col'] == 0
    assert idx['fields'][0]['crop'] == 'rapeseed'                  # crop code 0
    assert idx['fields'][0]['farm_id'] == 'Germany/DUP3/farm1'
    side = json.loads((idx['dir'] / 'target_side.json').read_text())
    assert 'never model input' in side['note']
    np.testing.assert_array_equal(rows['target'], truth['Germany']['target'])


def test_cache_reorders_per_file_and_collapses_static(corpus):
    root, art, truth = corpus
    for c in FIELDS:
        t = np.load(art / 'cache' / c / 'temporal.npy')
        s = np.load(art / 'cache' / c / 'static.npy')
        np.testing.assert_array_equal(t, truth[c]['canon_t'])
        np.testing.assert_array_equal(s, truth[c]['canon_s'])
        m = load_cache_manifest(art, root, c)
        assert m['audit']['rows_static_conflict'] == 0
        assert m['audit']['dateless_optical_values'] == 0


def test_static_conflicts_are_flagged_not_averaged():
    order = list(ALL_CHANNELS)

    class L:
        temporal_idx = resolve_channel_indices(order, TEMPORAL_CHANNELS)
        static_idx = resolve_channel_indices(order, STATIC_CHANNELS)
        origin_offset = 0

    sample = np.ones((2, 24, 120), dtype=np.float32)
    sample[1, 5, order.index('clay_0-5')] = 2.0
    _, static, _, flags, conflicts = canonicalize_block(sample, np.zeros((2, 24)), L)
    assert flags[0] & FLAG_STATIC_CONFLICT == 0 and flags[1] & FLAG_STATIC_CONFLICT
    assert static[1, STATIC_CHANNELS.index('clay_0-5')] == 1.0
    assert conflicts[STATIC_CHANNELS.index('clay_0-5')] == 1


def test_stale_cache_and_index_are_rejected(corpus, tmp_path):
    root, art, _ = corpus
    path = root / 'Uruguay' / SOURCE_FILENAME
    st = os.stat(path)
    try:
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        with pytest.raises(SnapshotError):
            load_cache_manifest(art, root, 'Uruguay')
        with pytest.raises(SnapshotError):
            load_country_index(art, root, 'Uruguay')
    finally:
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


# ---- dataset -----------------------------------------------------------------

def _dataset(corpus, countries=('Germany', 'Uruguay'), **kw):
    root, art, _ = corpus
    table = load_field_table(art, root, list(countries))
    fields = table['fields']
    tf = [(f['country'], f['field_code'],
           load_country_index(art, root, f['country'])['rows']['target'][f['row_start']:f['row_end']])
          for f in fields[:2]]
    norm = YieldSATNormalizer.fit(art, tf)
    return YieldSATPointDataset(root, art, fields, norm, **kw), norm, fields


def test_cache_and_h5_backends_agree(corpus):
    a, _, _ = _dataset(corpus, backend='cache')
    b, _, _ = _dataset(corpus, backend='h5')
    idx = np.array([0, 5, 13, 40, len(a) - 1])
    ba, bb = a.get_batch(idx), b.get_batch(idx)
    for k in ba['inputs']:
        assert torch.equal(ba['inputs'][k], bb['inputs'][k])
        assert torch.equal(ba['masks'][k], bb['masks'][k])


def test_masks_cutoff_and_static_validity(corpus):
    ds, _, _ = _dataset(corpus, cutoff_mode='before_harvest', cutoff_days=0)
    b = ds.get_batch(np.arange(len(ds)))
    s2 = b['masks']['yieldsat_s2']
    assert not s2[:, 8].any()                                 # cloudy dated slot
    assert not b['masks']['yieldsat_weather'][:, :6].any()    # undated slots
    assert not b['masks']['yieldsat_weather'][:, 6].any()     # first dated slot: no interval start
    assert b['masks']['yieldsat_s2'][:, 6].all()              # ... but optical there is valid
    assert b['masks']['yieldsat_dem'].all()                   # static survives slot-0 NaN
    # field 0: dates at 0,20,..; harvest at day 200 -> slots with t > 200 excluded
    tv = b['time_valid'][:CELLS]
    assert tv[:, 6:17].all() and not tv[:, 17:].any()
    assert torch.all(b['inputs']['yieldsat_s2'][~s2] == 0)
    assert not ds.retrospective
    full, _, _ = _dataset(corpus, cutoff_mode='all_slots')
    assert full.retrospective
    assert full.get_batch(np.arange(CELLS))['time_valid'][:, 6:20].all()


def test_normalizer_uses_training_fields_only(corpus):
    root, art, truth = corpus
    ds, norm, fields = _dataset(corpus)
    st = norm.get('Germany')
    t = np.concatenate([truth['Germany']['target'][:2 * CELLS]])
    assert abs(st['target_mean'] - t.mean()) < 1e-4            # first two Germany fields
    dem = STATIC_CHANNELS.index('dem')
    assert abs(st['static_mean'][dem] - truth['Germany']['canon_s'][:2 * CELLS, dem].mean()) < 1e-3
    per = YieldSATNormalizer.fit(art, [('Germany', fields[0]['field_code'], t)], pooling='per_country')
    with pytest.raises(KeyError):
        per.get('Uruguay')


def test_field_balanced_sampler_blocks_stay_in_one_season(corpus):
    ds, _, _ = _dataset(corpus)
    sampler = FieldBalancedBatchSampler(ds.season_ranges, 8, 5, alpha=0.0, block_size=4)
    for batch in sampler:
        seasons = ds.season_of[np.array(batch)]
        for k in range(0, len(batch), 4):
            assert len(set(seasons[k:k + 4])) == 1


# ---- splits --------------------------------------------------------------------

def test_splits_group_farms_and_refuse_leakage(corpus):
    root, art, _ = corpus
    table = load_field_table(art, root, ['Germany', 'Uruguay'])
    split = make_split(table, 'farm', seed=0, val_frac=0.25, test_frac=0.25)
    by = {f['season_id']: f for f in table['fields']}
    farms = {p: {by[s]['farm_group_id'] for s in split['partitions'][p]} for p in ('train', 'val', 'test')}
    assert not (farms['train'] & farms['test']) and not (farms['train'] & farms['val'])
    assert split['label'] == 'farm-held-out'
    loco = make_split(table, 'country', holdout_country='Uruguay')
    assert {by[s]['country'] for s in loco['partitions']['test']} == {'Uruguay'}
    # aliasing: the same physical field under two farms joins the farm groups
    for f in table['fields']:
        if f['field_shared_name'].endswith('field11_wheat_2019'):
            f['physical_field_id'] = 'X'
        if f['field_shared_name'].endswith('field12_wheat_2020'):
            f['physical_field_id'] = 'X'
    from dataset.yieldsat_splits import _attach_farm_groups
    _attach_farm_groups(table['fields'])
    g = {f['farm_group_id'] for f in table['fields'] if f['country'] == 'Germany'}
    assert 'Germany/DUP3/farm1+Germany/DUP3/farm2' in g
    bad = make_split(table, 'farm', seed=0)
    s = bad['partitions']['train'][0]
    bad['partitions']['test'].append(s)
    with pytest.raises(ValueError):
        check_split(bad, table['fields'])


# ---- model / objectives ----------------------------------------------------------

def test_point_model_handles_missing_streams_and_empty_histories(corpus):
    ds, _, _ = _dataset(corpus, soil_uncertainty='ancillary', aspect_encoding='cyclic')
    b = ds.get_batch(np.arange(16))
    b['available']['yieldsat_s2'][:4] = 0
    b['masks']['yieldsat_weather'][4:8] = False
    b['available']['yieldsat_weather'][4:8] = 0
    model = YieldSATPointModel(ds.layout, embed_dim=32, num_latents=4, depth=1, num_heads=2,
                               modality_embed=8, modality_dropout=0.5)
    model.train()
    pred, loss = model.loss(b)
    loss.backward()
    assert pred.shape == (16,) and torch.isfinite(pred).all() and torch.isfinite(loss)
    assert model.layout['yieldsat_terrain']['out_channels'][:2] == ['aspect_sin', 'aspect_cos']
    grads = [p.grad for n, p in model.named_parameters() if 'encoders.encoders.yieldsat_soil' in n]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_sensor_checkpoint_transfer_is_checked(corpus):
    ds, _, _ = _dataset(corpus)
    kw = dict(embed_dim=32, num_latents=4, depth=1, num_heads=2, modality_embed=8)
    a = YieldSATPointModel(ds.layout, **kw)
    payload = a.sensor_state_dict()
    assert not any(k.startswith('head.') or k.startswith('crop_embed') for k in payload['state_dict'])
    b = YieldSATPointModel(ds.layout, **kw)
    info = b.load_sensor_state_dict(payload)
    assert info['loaded'] == len(payload['state_dict']) and not info['missing_sensor_keys']
    other = stream_layout(['yieldsat_s2', 'yieldsat_soil'], soil_uncertainty='ancillary')
    with pytest.raises(ValueError):
        YieldSATPointModel(other, **kw).load_sensor_state_dict(payload)


def test_forecast_hides_all_temporal_streams_from_target_slot(corpus):
    ds, _, _ = _dataset(corpus, cutoff_mode='all_slots')
    b = ds.get_batch(np.arange(8))
    ctx, target, tmask, eligible = forecast_split(b, ds.layout)
    assert eligible.all()
    # last valid optical slot is 19 -> nothing at or after it in any temporal stream
    assert not ctx['masks']['yieldsat_s2'][:, 19:].any()
    assert not ctx['masks']['yieldsat_weather'][:, 19:].any()
    assert torch.equal(target, b['inputs']['yieldsat_s2'][:, 19])
    masked, hidden = mask_observations(b, ds.layout, ratio=0.5, generator=torch.Generator().manual_seed(0))
    for k in hidden:
        assert not (masked['masks'][k] & hidden[k]).any()
        assert (hidden[k] <= b['masks'][k]).all()
    with pytest.raises(ValueError):
        require_objective('masked_spatial_reconstruction')
    neg = contrastive_negative_mask(['a', 'a', 'b'])
    assert not neg[0, 1] and neg[0, 2] and not neg[2, 2]


def test_grid_reconstruction_and_metrics():
    grid = scatter_to_grid([0, 1], [2, 0], [1.0, 2.0], 2, 3)
    assert grid[0, 2] == 1.0 and np.isnan(grid[1, 1])
    with pytest.raises(ValueError):
        scatter_to_grid([2], [0], [1.0], 2, 3)
    with pytest.raises(ValueError):
        scatter_to_grid([0, 0], [1, 1], [1.0, 2.0], 2, 3)
    meta = [{'country': 'A', 'crop': 'wheat'}, {'country': 'B', 'crop': 'soybean'}]
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = evaluate_predictions(y, y, np.array([0, 0, 1, 1]), meta)
    assert m['overall']['pixel']['rmse'] == 0 and m['overall']['pixel']['r2'] == 1
    assert set(m['per_country']) == {'A', 'B'} and m['macro_country']['pixel_rmse'] == 0


def test_entry_point_contract_selector():
    import main_yieldsat_finetune as entry
    args = entry.get_args_parser().parse_args([
        '--data_contract', 'downloader_native_layers_v1', '--countries', 'Germany',
        '--split', 'x', '--source_root', '/nonexistent', '--artifact_root', '/nonexistent'])
    with pytest.raises(SystemExit, match='not implemented'):
        entry.main(args)


# ---- YS-11 fusion variants ---------------------------------------------------

def _fake_batch(layout, B=6, T=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    b = {'inputs': {}, 'masks': {}, 'available': {}}
    for n, lay in layout.items():
        C = len(lay['out_channels'])
        shape = (B, T, C) if lay['temporal'] else (B, C)
        b['inputs'][n] = torch.randn(shape, generator=g)
        b['masks'][n] = torch.rand(shape, generator=g) > 0.3
        b['available'][n] = torch.ones(B)
    for n, i in (('yieldsat_s2', 0), ('yieldsat_weather', 1)):   # empty histories
        b['masks'][n][i] = False
        b['available'][n][i] = 0
    b['time_features'] = torch.randn(B, T, 3, generator=g)
    b['time_valid'] = torch.rand(B, T, generator=g) > 0.3
    b['crop'] = torch.randint(0, 4, (B,), generator=g)
    b['target'] = torch.randn(B, generator=g)
    b['target_valid'] = torch.ones(B, dtype=torch.bool)
    return b


def test_encoders_emit_tokens_with_masks():
    from models_multimodal_encoder import MaskedTemporalEncoder, SoilProfileEncoder
    enc = MaskedTemporalEncoder(in_dim=3, embed_dim=16, hidden_dim=16, output='tokens')
    x = torch.zeros(2, 24, 3 * 2 + 3)
    x[0, 5, 3] = 1.0                                            # one valid slot in sample 0
    assert enc(x).shape == (2, 24, 16)
    tm = enc.token_mask(x)
    assert tm.shape == (2, 24) and tm[0, 5] and tm.sum() == 1
    soil = SoilProfileEncoder(embed_dim=16, hidden_dim=16, output='tokens')
    xs = torch.zeros(2, 96)
    xs[1, 48] = 1.0                                             # mask of property 0, depth 0
    assert soil(xs).shape == (2, 6, 16)
    assert soil.token_mask(xs)[1].tolist() == [True] + [False] * 5
    summary = MaskedTemporalEncoder(in_dim=3, embed_dim=16, hidden_dim=16)
    assert summary(x).shape == (2, 1, 16) and summary.token_mask(x).tolist() == [[True], [False]]


def test_fusion_token_mask_and_all_invalid_rows():
    from models_latent_fusion import LatentFusionTransformer
    f = LatentFusionTransformer(embed_dim=32, num_latents=4, depth=2, num_heads=2,
                                modalities={'a': {'spatial': 1, 'temporal': 5},
                                            'b': {'spatial': 1, 'temporal': 1}},
                                modality_embed=8, cross_attn_layers=2, norm_first=True,
                                input_norm=True, latent_init='trunc_normal')
    emb = {'a': torch.randn(3, 5, 32), 'b': torch.randn(3, 1, 32)}
    tm = {'a': torch.tensor([[1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)}
    mask = {'a': torch.ones(3), 'b': torch.tensor([1.0, 1.0, 0.0])}
    out = f(emb, mask=mask, token_mask=tm)                      # sample 2: every key invalid
    assert out.shape == (3, 4, 32) and torch.isfinite(out).all()
    # masked tokens must not influence the output
    emb2 = {'a': emb['a'].clone(), 'b': emb['b']}
    emb2['a'][0, 2:] = 100.0
    out2 = f(emb2, mask=mask, token_mask=tm)
    assert torch.allclose(out[0], out2[0], atol=1e-5)
    with pytest.raises(ValueError):
        f(emb, token_mask={'a': torch.ones(3, 4, dtype=torch.bool)})


@pytest.mark.parametrize('fusion', ['perceiver_summary', 'perceiver_tokens', 'concat_mlp',
                                    'token_transformer'])
def test_every_fusion_variant_trains_with_missing_streams(fusion):
    from dataset.yieldsat_schema import STREAMS
    layout = stream_layout(list(STREAMS))
    model = YieldSATPointModel(layout, embed_dim=32, num_latents=4, depth=2, num_heads=2,
                               modality_embed=8, modality_dropout=0.3, fusion=fusion)
    model.train()
    pred, loss = model.loss(_fake_batch(layout))
    loss.backward()
    assert pred.shape == (6,) and torch.isfinite(pred).all()
    assert model.descriptor()['config']['fusion'] == fusion


def test_summary_perceiver_reproduces_original_parameter_count():
    from dataset.yieldsat_schema import STREAMS
    model = YieldSATPointModel(stream_layout(list(STREAMS)))
    assert sum(p.numel() for p in model.parameters()) == 1127937
    assert model.config['cross_attn_layers'] == 1 and model.encoder_output == 'summary'


def test_date_encoding_aligns_optical_and_weather_tokens():
    from dataset.yieldsat_schema import STREAMS
    from models_yieldsat import DateEncoding
    enc = DateEncoding(16)
    days = torch.tensor([[0.0, 30.0, 30.0]])
    dated = torch.tensor([[True, True, False]])
    e = enc(days, dated)
    assert not torch.allclose(e[0, 0], e[0, 1])
    assert torch.equal(e[0, 2], enc.undated)
    # the same module (one set of weights) serves both temporal streams
    model = YieldSATPointModel(stream_layout(list(STREAMS)), embed_dim=32, num_latents=4,
                               num_heads=2, modality_embed=8, fusion='perceiver_tokens')
    assert sum(1 for n, _ in model.named_modules() if n.endswith('date_enc')) == 1


def test_fusion_layout_is_checked_on_transfer():
    from dataset.yieldsat_schema import STREAMS
    layout = stream_layout(list(STREAMS))
    kw = dict(embed_dim=32, num_latents=4, depth=2, num_heads=2, modality_embed=8)
    tokens = YieldSATPointModel(layout, fusion='perceiver_tokens', **kw).sensor_state_dict()
    summary = YieldSATPointModel(layout, **kw)
    with pytest.raises(ValueError, match='fusion layout differs'):
        summary.load_sensor_state_dict(tokens)
    info = summary.load_sensor_state_dict(tokens, encoders_only=True)
    assert info['loaded'] > 0 and all(k.startswith('encoders.') for k in info['missing_sensor_keys'] or ['encoders.'])
    # an old descriptor without fusion keys is read as perceiver_summary
    legacy = summary.sensor_state_dict()
    for k in ('fusion', 'encoder_output', 'cross_attn_layers'):
        legacy['descriptor']['config'].pop(k)
    assert YieldSATPointModel(layout, **kw).load_sensor_state_dict(legacy)['loaded'] > 0
