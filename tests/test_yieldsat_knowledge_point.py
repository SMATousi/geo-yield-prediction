"""Point-model knowledge pretraining: raw concept indices (PK-02)."""
import numpy as np
import pytest

from dataset.yieldsat_schema import STATIC_CHANNELS, TEMPORAL_CHANNELS
import yieldsat_build_knowledge as kb

T = {c: i for i, c in enumerate(TEMPORAL_CHANNELS)}
S = {c: i for i, c in enumerate(STATIC_CHANNELS)}


def _weather_row(times, tmean_c, prec_mm_day, tmax_c=None):
    """One row whose weather is a constant daily value, stored as inclusive interval sums."""
    temporal = np.full((1, 24, len(TEMPORAL_CHANNELS)), np.nan, dtype=np.float32)
    t = np.full((1, 24), np.nan, dtype=np.float32)
    t[0, :len(times)] = times
    prev = None
    for k, tk in enumerate(times):
        days = (tk - prev + 1) if prev is not None else 7   # first slot: arbitrary span
        temporal[0, k, T['temp_mean']] = (tmean_c + 273.15) * days
        temporal[0, k, T['temp_max']] = ((tmax_c if tmax_c is not None else tmean_c + 5) + 273.15) * days
        temporal[0, k, T['temp_min']] = (tmean_c - 5 + 273.15) * days
        temporal[0, k, T['total_prec']] = prec_mm_day / 1000.0 * days
        prev = tk
    return temporal, t


def test_weather_daily_means_and_window_total():
    times = np.arange(0, 241, 20, dtype=np.float32)
    temporal, t = _weather_row(times, 22.0, 3.0, tmax_c=30.0)
    p, tm, tx, cov = kb.weather_window(temporal, t, np.array([0.0]), np.array([200.0]), kb.MID)
    assert tm[0] == pytest.approx(22.0, abs=1e-3)
    assert tx[0] == pytest.approx(30.0, abs=1e-3)
    # mid-season = days 60-160 = 100 days at 3 mm/day
    assert p[0] == pytest.approx(300.0, rel=1e-3)
    assert cov[0] >= 1.0


def test_weather_first_slot_never_used():
    times = np.array([100, 120, 140, 160, 180], dtype=np.float32)
    temporal, t = _weather_row(times, 10.0, 1.0)
    temporal[0, 0, T['temp_mean']] = 1e9                      # garbage at the first dated slot
    _, tm, _, _ = kb.weather_window(temporal, t, np.array([0.0]), np.array([250.0]), kb.MID)
    assert tm[0] == pytest.approx(10.0, abs=1e-3)


def test_weather_low_coverage_is_unknown():
    times = np.array([0, 10, 20], dtype=np.float32)           # ends before mid-season
    temporal, t = _weather_row(times, 10.0, 1.0)
    p, tm, _, cov = kb.weather_window(temporal, t, np.array([0.0]), np.array([200.0]), kb.MID)
    assert np.isnan(p[0]) and np.isnan(tm[0]) and cov[0] < kb.MIN_WEATHER_COVERAGE


def _s2_row(times, ndvi):
    temporal = np.full((1, 24, len(TEMPORAL_CHANNELS)), np.nan, dtype=np.float32)
    t = np.full((1, 24), np.nan, dtype=np.float32)
    t[0, :len(times)] = times
    red = 1000.0
    for k, v in enumerate(ndvi):
        nir = red * (1 + v) / (1 - v)
        temporal[0, k, T['B04']] = red
        temporal[0, k, T['B08']] = nir
        temporal[0, k, T['B05']] = red
        temporal[0, k, T['B8A']] = nir
        temporal[0, k, T['B11']] = red
    return temporal, t


def test_s2_indices():
    times = np.array([0, 20, 40, 60, 80, 100], dtype=np.float32)
    ndvi = [0.2, 0.3, 0.6, 0.8, 0.7, 0.4]
    temporal, t = _s2_row(times, ndvi)
    rise, ndre, persist, ndmi = kb.s2_indices(temporal, t, np.array([0.0]), np.array([100.0]))
    assert rise[0] == pytest.approx(0.8 - 0.2, abs=1e-4)            # max - early min (days 0-30)
    assert ndre[0] == pytest.approx(np.mean([0.6, 0.8, 0.7]), abs=1e-4)  # days 30-80
    assert persist[0] == pytest.approx(np.mean([0.8, 0.7, 0.4]) / 0.8, abs=1e-4)  # days 60-100
    assert ndmi[0] == pytest.approx(ndre[0], abs=1e-4)


def test_s2_outside_season_ignored_and_unobserved_unknown():
    temporal, t = _s2_row(np.array([-50, 150], dtype=np.float32), [0.9, 0.9])
    rise, ndre, persist, ndmi = kb.s2_indices(temporal, t, np.array([0.0]), np.array([100.0]))
    assert all(np.isnan(x[0]) for x in (rise, ndre, persist, ndmi))


def test_soil_units_and_depth_weighting():
    static = np.full((1, len(STATIC_CHANNELS)), np.nan, dtype=np.float32)
    for d, clay in zip(('0-5', '5-15', '15-30'), (100, 200, 400)):
        static[0, S['clay_' + d]] = clay
        static[0, S['silt_' + d]] = 300
        static[0, S['sand_' + d]] = 1000 - clay - 300
        static[0, S['soc_' + d]] = 300                          # dg/kg
    clay, fine, soc = kb.soil_indices(static)
    assert clay[0] == pytest.approx((5 * 100 + 10 * 200 + 15 * 400) / 30)
    assert fine[0] == pytest.approx((clay[0] + 300) / 1000)
    assert soc[0] == pytest.approx(30.0)                        # g/kg


def test_dem_within_field_and_flat_field():
    dem = np.array([10, 11, 12, 13, 50, 50.2, 50.4], dtype=np.float32)
    rel, relief, low = kb.dem_indices(dem, [(0, 4), (4, 7)])
    assert rel[:4] == pytest.approx([-1.5, -0.5, 0.5, 1.5])
    assert low[0] == 1.0 and low[3] == 0.0
    assert relief[0] > 1.0 and relief[5] < 1.0                   # second field is flat (abstains later)


def test_pole_facing_hemisphere():
    static = np.full((2, len(STATIC_CHANNELS)), np.nan, dtype=np.float32)
    static[:, S['aspect']] = [180.0, 0.0]
    pole_s, _ = kb.terrain_indices(static, 'Argentina')
    pole_n, _ = kb.terrain_indices(static, 'Germany')
    assert pole_s == pytest.approx([1.0, -1.0]) and pole_n == pytest.approx([-1.0, 1.0])


def test_estimators_respect_input_cutoff():
    times = np.array([0, 20, 40, 60, 80, 100], dtype=np.float32)
    temporal, t = _s2_row(times, [0.2, 0.3, 0.6, 0.8, 0.7, 0.1])
    vis = kb.visible_times(t, np.array([100.0]), cutoff_days=30)     # keeps slots <= day 70
    assert np.isfinite(vis[0, :4]).all() and np.isnan(vis[0, 4:]).all()
    _, _, persist, _ = kb.s2_indices(temporal, vis, np.array([0.0]), np.array([100.0]))
    assert persist[0] == pytest.approx(0.8 / 0.8)                    # late window sees only day 60


# ---- PK-03: reference, gates, pretrainer -------------------------------------
import torch  # noqa: E402

import yieldsat_knowledge_point as kp  # noqa: E402
from dataset.yieldsat_dataset import stream_layout  # noqa: E402
from dataset.yieldsat_schema import CROPS  # noqa: E402
from models_yieldsat import YieldSATPointModel  # noqa: E402

STREAMS = ['yieldsat_s2', 'yieldsat_weather', 'yieldsat_dem', 'yieldsat_terrain', 'yieldsat_soil']


def _raw(n_seasons=40, rows_per=5, seed=0, crop='corn'):
    rng = np.random.default_rng(seed)
    n = n_seasons * rows_per
    season = np.repeat(np.arange(n_seasons), rows_per)
    per_season = lambda: np.repeat(rng.normal(size=n_seasons), rows_per)  # noqa: E731
    raw = {k: rng.normal(size=n).astype(np.float32) for k in kb.RAW_FIELDS}
    raw.update(season=season, crop=np.full(n, CROPS.index(crop), np.int8),
               precip_mid_mm=(200 + 50 * per_season()).astype(np.float32),
               tmean_mid_c=(20 + 2 * per_season()).astype(np.float32),
               tmax_mid_c=(28 + 2 * per_season()).astype(np.float32),
               field_relief=np.full(n, 3.0, np.float32), elev_low_rank=rng.uniform(size=n).astype(np.float32),
               soc030_gkg=np.full(n, 20.0, np.float32), slope=rng.uniform(1000, 5000, n).astype(np.float32))
    return raw


def test_reference_ranks_are_fold_train_percentiles_and_season_weighted():
    raw = _raw()
    train = np.arange(0, 100)                                      # seasons 0-19 only
    ref = kp.KnowledgeReference.fit({'Argentina': (raw, train)}, kp.load_library())
    assert ref.meta['reference_seasons']['precip_mid_mm|Argentina/corn'] == 20
    t, g = ref.transform(raw, 'Argentina', train)
    rain = t[:, ref.concepts.index('rain_supported')]
    # one value per season, 20 seasons: ranks spread uniformly over (0, 1)
    assert 0.0 < rain.min() < 0.1 and 0.9 < rain.max() < 1.0
    cool = t[:, ref.concepts.index('cool_regime')]
    warm = t[:, ref.concepts.index('warm_regime')]
    np.testing.assert_allclose(cool, 1 - warm, atol=1e-6)        # no row is heat-stressed here
    # reference survives serialization
    ref2 = kp.KnowledgeReference.from_json(ref.to_json())
    np.testing.assert_allclose(ref2.transform(raw, 'Argentina', train)[0], t, equal_nan=True)


def test_small_strata_and_unseen_crops_are_unknown():
    raw = _raw(n_seasons=10)                                        # below MIN_REFERENCE_SEASONS
    ref = kp.KnowledgeReference.fit({'Argentina': (raw, np.arange(len(raw['season'])))}, kp.load_library())
    t, _ = ref.transform(raw, 'Argentina', np.arange(5))
    assert np.isnan(t[:, ref.concepts.index('rain_supported')]).all()


def test_gate_semantics():
    raw = _raw()
    ref = kp.KnowledgeReference.fit({'Argentina': (raw, np.arange(200))}, kp.load_library())
    rows = np.arange(200)
    raw['tmax_mid_c'][:5] = 40.0                                   # heat-stressed season
    raw['field_relief'][5:10] = 0.5                                # flat field
    raw['soc030_gkg'][10:15] = 200.0                               # organic soil
    raw['slope'][15:20] = 0.0                                      # flat cells
    t, g = ref.transform(raw, 'Argentina', rows)
    r = {rid['id']: j for j, rid in enumerate(ref.rules)}
    c = {cid: i for i, cid in enumerate(ref.concepts)}
    assert (g[:, r['ys_r01']] == 1).all()                          # corn: rainfed summer crop
    assert (g[:5, r['ys_r02']] == 0).all() and (t[:5, c['warm_regime']] == 0).all()
    assert (g[5:10, r['ys_r04']] == 0).all() and np.isnan(t[5:10, c['low_elevation_position']]).all()
    assert (g[10:15, r['ys_r06']] == 0).all()
    assert np.isnan(t[15:20, c['pole_facing_aspect']]).all()
    assert (g[:, r['ys_r05']] == 0).all()                          # always abstains
    rain = t[:, c['rain_supported']]
    np.testing.assert_array_equal(g[:, r['ys_r03']] == 1, rain < 0.5)
    raw['crop'][:] = CROPS.index('wheat')
    ref_w = kp.KnowledgeReference.fit({'Argentina': (raw, rows)}, kp.load_library())
    assert (ref_w.transform(raw, 'Argentina', rows)[1][:, r['ys_r01']] == 0).all()


def test_derangement_has_no_fixed_point():
    for seed in range(20):
        p = kp.derangement(12, seed)
        assert sorted(p) == list(range(12)) and not (p == np.arange(12)).any()


def test_season_mean_counts_each_season_once():
    loss = torch.tensor([1.0, 1.0, 1.0, 5.0])
    w = torch.ones(4)
    season = torch.tensor([0, 0, 0, 1])
    assert float(kp.season_mean(loss, w, season)) == pytest.approx(3.0)
    assert kp.season_mean(loss, torch.zeros(4), season) is None


def _kbatch(layout, lib, B=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    b = {'inputs': {}, 'masks': {}, 'available': {}}
    for n, l in layout.items():
        C = len(l['out_channels'])
        sh = (B, 24, C) if l['temporal'] else (B, C)
        b['inputs'][n] = torch.randn(sh, generator=g)
        b['masks'][n] = torch.rand(sh, generator=g) > 0.2
        b['available'][n] = torch.ones(B)
    b['time_features'] = torch.randn(B, 24, 3, generator=g)
    b['time_valid'] = torch.ones(B, 24, dtype=torch.bool)
    b['crop'] = torch.zeros(B, dtype=torch.long)
    b['season'] = torch.arange(B) // 4
    b['concept_target'] = torch.rand(B, len(lib['concepts']), generator=g) * 0.5 + 0.5
    b['rule_gate'] = torch.ones(B, len(lib['rules']))
    return b


def _pretrainer(control=None):
    lib = kp.load_library()
    layout = stream_layout(STREAMS)
    torch.manual_seed(0)
    model = YieldSATPointModel(layout, embed_dim=32, num_latents=4, depth=1, modality_dropout=0.0)
    text = np.random.default_rng(1).normal(size=(len(lib['concepts']), 16)).astype(np.float32)
    return kp.KnowledgePointPretrainer(model, text, lib, control=control), layout, lib


def test_pretrainer_gradients_reach_every_stream_encoder():
    pt, layout, lib = _pretrainer()
    b = _kbatch(layout, lib)
    total, losses = pt(b)
    assert losses['ground_terms'] > 0 and losses['relation_terms'] > 0
    lg, lr, _ = pt.knowledge_losses(b)
    (lg + lr).backward()
    for s in STREAMS:
        grads = [p.grad for p in pt.model.encoders.encoders[s].parameters() if p.grad is not None]
        assert grads and sum(float(x.abs().sum()) for x in grads) > 0, s


def test_unknown_targets_and_closed_gates_contribute_nothing():
    pt, layout, lib = _pretrainer()
    b = _kbatch(layout, lib)
    b['concept_target'][:] = float('nan')
    lg, lr, n = pt.knowledge_losses(b)
    assert n == {'ground_terms': 0, 'relation_terms': 0} and float(lg) == 0 and float(lr) == 0
    b = _kbatch(layout, lib)
    b['rule_gate'][:] = 0
    _, lr, n = pt.knowledge_losses(b)
    assert n['relation_terms'] == 0 and float(lr) == 0
    b = _kbatch(layout, lib)
    b['concept_target'][:] = 0.2                                   # below minimum_presence
    _, _, n = pt.knowledge_losses(b)
    assert n['relation_terms'] == 0 and n['ground_terms'] > 0
    b = _kbatch(layout, lib)
    b['available']['yieldsat_soil'][:] = 0                         # absent stream: its concepts masked
    _, _, n_abs = pt.knowledge_losses(b)
    _, _, n_all = pt.knowledge_losses(_kbatch(layout, lib))
    assert n_abs['ground_terms'] == n_all['ground_terms'] - 3 * 32


def test_controls():
    base, _, lib = _pretrainer()
    sh, _, _ = _pretrainer('shuffled')
    p = sh.permutation
    torch.testing.assert_close(sh.prototypes(), base.prototypes()[torch.as_tensor(p)])
    assert not torch.allclose(sh.relations(), base.relations())
    nt, layout, _ = _pretrainer('notext')
    assert nt.learned_prototypes.requires_grad
    b = _kbatch(layout, lib)
    lg, lr, _ = nt.knowledge_losses(b)
    (lg + lr).backward()
    assert nt.learned_prototypes.grad.abs().sum() > 0
    sl, _, _ = _pretrainer('ssl_long')
    _, losses = sl(b)
    assert 'ground' not in losses and 'masked_observation' in losses


def test_random_targets_moves_knowledge_between_seasons_of_a_stratum():
    class DS:
        pass
    ds = DS()
    ds.seasons = [{'country': 'Argentina', 'crop': 'corn'}] * 3 + [{'country': 'Brazil', 'crop': 'soybean'}]
    ds.season_ranges = [(0, 2), (2, 5), (5, 6), (6, 8)]
    n = 8
    ds.__len__ = None
    season_of = np.repeat(np.arange(4), [2, 3, 1, 2])
    ds.knowledge = {'concept_target': season_of[:, None].astype(np.float32).repeat(12, 1),
                    'rule_gate': season_of[:, None].astype(np.float32).repeat(6, 1)}
    type(ds).__len__ = lambda self: n
    kp.randomize_targets(ds, seed=0)
    got = ds.knowledge['concept_target'][:, 0]
    for s in range(3):                                            # Argentina/corn: all from another season
        rows = season_of == s
        assert len(set(got[rows])) == 1 and got[rows][0] != s and got[rows][0] in (0, 1, 2)
    assert (got[season_of == 3] == 3).all()                       # a singleton stratum keeps its own
    np.testing.assert_array_equal(ds.knowledge['rule_gate'][:, 0], got)


def test_check_unit_detects_leaks(tmp_path):
    import json
    import yieldsat_pretrain_units as pu
    from dataset.yieldsat_schema import CONTRACT_KEY
    fields = [{'season_id': 'S{}'.format(i), 'country': 'Germany', 'year': 2018 + i % 3,
               'physical_field_id': 'P{}'.format(i), 'farm_id': 'Germany/f{}'.format(i % 2)} for i in range(12)]
    (tmp_path / 'splits').mkdir()
    fold = {'contract': CONTRACT_KEY, 'scheme': 'loyo', 'fold': 0, 'countries': ['Germany'], 'holdout': [2018],
            'group_key': 'year', 'partitions': {'test': ['S0', 'S3'], 'train': [], 'val': []}}
    (tmp_path / 'splits' / 'f0.json').write_text(json.dumps(fold))
    unit = {'excluded': {'S0', 'S3', 'S6', 'S9'}, 'folds': ['f0'], 'protocol': 'loyo',
            'rule': {'key': 'year', 'values': [2018]}}
    split = pu.make_unit('loyo_2018', unit, fields, {})
    assert pu.check_unit(split, tmp_path, fields)['excluded'] == 4
    leaked = json.loads(json.dumps(split))
    leaked['partitions']['excluded'].remove('S6')                  # same year, not a fold test season
    leaked['partitions']['train'].append('S6')
    with pytest.raises(ValueError, match='held-out year'):
        pu.check_unit(leaked, tmp_path, fields)
    leaked = json.loads(json.dumps(split))
    leaked['partitions']['excluded'].remove('S0')                  # a fold test season
    leaked['partitions']['val'].append('S0')
    with pytest.raises(ValueError, match='test seasons'):
        pu.check_unit(leaked, tmp_path, fields)


def test_static_validity_rejects_corrupt_curvature():
    from dataset.yieldsat_cache import static_valid_mask
    st = np.zeros((3, len(STATIC_CHANNELS)), dtype=np.float32)
    st[:, S['curvature']] = [0.5, -2.0e9, np.nan]
    st[2, S['dem']] = np.nan
    v = static_valid_mask(st)
    assert v[0, S['curvature']] and not v[1, S['curvature']] and not v[2, S['curvature']]
    assert v[1, S['dem']] and not v[2, S['dem']]
