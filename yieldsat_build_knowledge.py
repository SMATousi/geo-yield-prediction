"""Raw rule-based concept indices per YieldSAT point-cache row (PK-02).

Inputs only (no yield). Writes ``<artifact_root>/knowledge/<Country>/concept_raw.npz``
aligned with ``cache/<Country>/temporal.npy`` rows. Soft targets are NOT made
here: they are percentile ranks in a fold-train reference
(``yieldsat_knowledge_point.KnowledgeReference``). See
spec/yieldsat-point-knowledge-pretraining.md §3.

Units (verified 2026-10-04 on the cache, spec §9):
  S2         L2A DN / 10000 = reflectance
  weather    sums over the inclusive interval [t_(k-1), t_k] (dt+1 days);
             temp_* Kelvin-days, total_prec metres; first dated slot invalid
  soil       SoilGrids mapped units: clay/silt/sand g/kg, soc dg/kg (/10 -> g/kg)
  dem        metres; aspect degrees; slope unresolved scale (used as a rank only)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from dataset.yieldsat_schema import COUNTRIES, CROPS, STATIC_CHANNELS, TEMPORAL_CHANNELS

FORMAT = 'yieldsat_concept_raw_v2'
MID = (0.3, 0.8)          # fraction of the seeding -> harvest window
LATE = (0.6, 1.0)
EARLY = (0.0, 0.3)
MIN_WEATHER_COVERAGE = 0.5
# Estimators read only what the point model sees: slots dated <= harvest -
# cutoff_days. 0 matches --cutoff_mode all_slots / harvest (the paper
# protocols; estimator windows end at harvest anyway), 30 matches the
# pre-harvest default (--cutoff_mode before_harvest --cutoff_days 30).
CUTOFF_DAYS = 0
POLE_AZIMUTH = {'Argentina': 180.0, 'Brazil': 180.0, 'Uruguay': 180.0, 'Germany': 0.0}
RAW_FIELDS = ('precip_mid_mm', 'tmean_mid_c', 'tmax_mid_c', 'ndvi_rise', 'ndre_mid',
              'persist_ratio', 'ndmi_mid', 'clay030', 'fine030', 'soc030_gkg',
              'rel_elev', 'field_relief', 'elev_low_rank', 'pole_cos', 'slope')

_T = {c: i for i, c in enumerate(TEMPORAL_CHANNELS)}
_S = {c: i for i, c in enumerate(STATIC_CHANNELS)}
_DEPTHS = (('0-5', 5.0), ('5-15', 10.0), ('15-30', 15.0))


def weather_window(temporal, times, seeding, harvest, window):
    """Mid-season weather from interval sums by day overlap.

    Returns (precip_mm, tmean_c, tmax_c, coverage); NaN where the dated
    intervals cover less than MIN_WEATHER_COVERAGE of the window.
    """
    n = temporal.shape[0]
    length = (harvest - seeding).astype(np.float64)
    w0 = seeding + window[0] * length
    w1 = seeding + window[1] * length
    prec = np.zeros(n)
    tmean = np.zeros(n)
    tmax = np.zeros(n)
    covered = np.zeros(n)
    prev = np.full(n, np.nan)
    for k in range(times.shape[1]):
        t = times[:, k].astype(np.float64)
        dated = np.isfinite(t)
        ok = dated & np.isfinite(prev)        # first dated slot: unknown start
        if ok.any():
            days = t - prev + 1.0                  # inclusive interval
            lo = np.maximum(prev, w0)
            hi = np.minimum(t + 1.0, w1)
            ov = np.where(ok, np.clip(hi - lo, 0.0, None), 0.0)
            vals = temporal[:, k]
            good = ok & np.isfinite(vals[:, _T['total_prec']]) & np.isfinite(vals[:, _T['temp_mean']]) \
                & np.isfinite(vals[:, _T['temp_max']]) & (days > 0)
            ov = np.where(good, ov, 0.0)
            with np.errstate(invalid='ignore', divide='ignore'):
                prec += np.where(good, vals[:, _T['total_prec']] / days * ov, 0.0)
                tmean += np.where(good, vals[:, _T['temp_mean']] / days * ov, 0.0)
                tmax += np.where(good, vals[:, _T['temp_max']] / days * ov, 0.0)
            covered += ov
        prev = np.where(dated, t, prev)
    span = np.maximum(w1 - w0, 1e-9)
    coverage = covered / span
    keep = (coverage >= MIN_WEATHER_COVERAGE) & (length > 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        precip_mm = np.where(keep, prec * 1000.0 * span / np.maximum(covered, 1e-9), np.nan)
        tmean_c = np.where(keep, tmean / np.maximum(covered, 1e-9) - 273.15, np.nan)
        tmax_c = np.where(keep, tmax / np.maximum(covered, 1e-9) - 273.15, np.nan)
    return precip_mm, tmean_c, tmax_c, coverage


def _ratio(a, b):
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (a - b) / (a + b)
    return np.where(np.isfinite(r) & (a > 0) & (b > 0), r, np.nan)


def _masked(values, mask, how):
    v = np.where(mask & np.isfinite(values), values, np.nan)
    has = np.isfinite(v).any(axis=1)
    out = np.full(v.shape[0], np.nan)
    if has.any():
        with np.errstate(all='ignore'):
            out[has] = {'mean': np.nanmean, 'max': np.nanmax, 'min': np.nanmin}[how](v[has], axis=1)
    return out


def s2_indices(temporal, times, seeding, harvest):
    """(ndvi_rise, ndre_mid, persist_ratio, ndmi_mid) per row; NaN when unobserved."""
    refl = lambda b: temporal[:, :, _T[b]].astype(np.float64) / 10000.0  # noqa: E731
    ndvi = _ratio(refl('B08'), refl('B04'))
    ndre = _ratio(refl('B8A'), refl('B05'))
    ndmi = _ratio(refl('B08'), refl('B11'))
    length = (harvest - seeding).astype(np.float64)[:, None]
    with np.errstate(invalid='ignore', divide='ignore'):
        frac = (times.astype(np.float64) - seeding[:, None]) / length
    dated = np.isfinite(frac) & (length > 0)
    win = lambda w: dated & (frac >= w[0]) & (frac <= w[1])  # noqa: E731
    season = win((0.0, 1.0))
    vmax = _masked(ndvi, season, 'max')
    rise = vmax - _masked(ndvi, win(EARLY), 'min')
    late = _masked(ndvi, win(LATE), 'mean')
    with np.errstate(invalid='ignore', divide='ignore'):
        persist = np.where(vmax > 0.1, late / vmax, np.nan)
    return rise, _masked(ndre, win(MID), 'mean'), persist, _masked(ndmi, win(MID), 'mean')


def soil_indices(static):
    """(clay030 g/kg, fine030 fraction, soc030 g/kg), depth-weighted 0-30 cm."""
    def dw(prop):
        tot = sum(w for _, w in _DEPTHS)
        return sum(static[:, _S['{}_{}'.format(prop, d)]].astype(np.float64) * w for d, w in _DEPTHS) / tot
    clay, silt, sand = dw('clay'), dw('silt'), dw('sand')
    with np.errstate(invalid='ignore', divide='ignore'):
        fine = (clay + silt) / (clay + silt + sand)
    return clay, np.where(np.isfinite(fine), fine, np.nan), dw('soc') / 10.0


def dem_indices(dem, field_slices):
    """(rel_elev m, field_relief m, elev_low_rank in [0,1]) using within-field statistics."""
    rel = np.full(dem.shape, np.nan)
    relief = np.full(dem.shape, np.nan)
    low = np.full(dem.shape, np.nan)
    for a, b in field_slices:
        d = dem[a:b].astype(np.float64)
        ok = np.isfinite(d)
        if ok.sum() < 2:
            continue
        v = d[ok]
        rel[a:b][ok] = v - v.mean()
        relief[a:b] = np.percentile(v, 95) - np.percentile(v, 5)
        order = np.argsort(np.argsort(v, kind='stable'), kind='stable')
        low[a:b][ok] = 1.0 - order / (len(v) - 1)
    return rel, relief, low


def terrain_indices(static, country):
    aspect = static[:, _S['aspect']].astype(np.float64)
    pole = np.cos(np.deg2rad(aspect - POLE_AZIMUTH[country]))
    return np.where(np.isfinite(pole), pole, np.nan), static[:, _S['slope']].astype(np.float64)


def visible_times(times, harvest, cutoff_days=CUTOFF_DAYS):
    """Slot dates after the model's input cutoff become undated (NaN)."""
    if cutoff_days is None:
        return times
    return np.where(times <= (harvest - cutoff_days)[:, None], times, np.nan)


def compute_block(temporal, static, times, seeding, harvest, field_slices, country,
                  cutoff_days=CUTOFF_DAYS):
    """All raw indices for a contiguous block of whole fields; dict of float32 arrays."""
    seeding = seeding.astype(np.float64)
    harvest = harvest.astype(np.float64)
    times = visible_times(times, harvest, cutoff_days)
    p, tm, tx, _ = weather_window(temporal, times, seeding, harvest, MID)
    rise, ndre, persist, ndmi = s2_indices(temporal, times, seeding, harvest)
    clay, fine, soc = soil_indices(static)
    rel, relief, low = dem_indices(static[:, _S['dem']], field_slices)
    pole, slope = terrain_indices(static, country)
    out = dict(precip_mid_mm=p, tmean_mid_c=tm, tmax_mid_c=tx, ndvi_rise=rise, ndre_mid=ndre,
               persist_ratio=persist, ndmi_mid=ndmi, clay030=clay, fine030=fine, soc030_gkg=soc,
               rel_elev=rel, field_relief=relief, elev_low_rank=low, pole_cos=pole, slope=slope)
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def build_country(artifact_root, country, block_rows=262144, overwrite=False, cutoff_days=CUTOFF_DAYS):
    root = Path(artifact_root)
    out = raw_path(root, country, cutoff_days)
    if out.exists() and not overwrite:
        raise FileExistsError(out)
    cache = root / 'cache' / country
    temporal = np.load(cache / 'temporal.npy', mmap_mode='r')
    static = np.load(cache / 'static.npy', mmap_mode='r')
    times = np.load(cache / 'times.npy', mmap_mode='r')
    rows = np.load(root / 'index' / country / 'rows.npz')
    fields = json.loads((root / 'index' / country / 'fields.json').read_text())
    n = temporal.shape[0]
    if len(rows['field_code']) != n:
        raise ValueError('index and cache row counts differ')
    res = {k: np.full(n, np.nan, dtype=np.float32) for k in RAW_FIELDS}
    crop = np.full(n, -1, dtype=np.int8)
    season = np.full(n, -1, dtype=np.int32)
    fields = sorted(fields, key=lambda f: f['row_start'])
    for i, f in enumerate(fields):
        crop[f['row_start']:f['row_end']] = CROPS.index(f['crop'])
        season[f['row_start']:f['row_end']] = i
    t0 = time.time()
    i = 0
    while i < len(fields):
        j, a = i, fields[i]['row_start']
        while j < len(fields) and (fields[j]['row_end'] - a <= block_rows or j == i):
            j += 1
        b = fields[j - 1]['row_end']
        sl = [(f['row_start'] - a, f['row_end'] - a) for f in fields[i:j]]
        blk = compute_block(np.asarray(temporal[a:b]), np.asarray(static[a:b]), np.asarray(times[a:b]),
                            rows['seeding_day'][a:b], rows['harvest_day'][a:b], sl, country, cutoff_days)
        for k, v in blk.items():
            res[k][a:b] = v
        print('{} rows {}-{} / {} ({:.0f}s)'.format(country, a, b, n, time.time() - t0), flush=True)
        i = j
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, format=FORMAT, country=country, cutoff_days=cutoff_days, crop=crop, season=season,
             season_id=np.array([f['season_id'] for f in fields]), **res)
    summary = {k: {'missing': float(np.isnan(v).mean()),
                   'p5': float(np.nanpercentile(v, 5)) if np.isfinite(v).any() else None,
                   'p50': float(np.nanmedian(v)) if np.isfinite(v).any() else None,
                   'p95': float(np.nanpercentile(v, 95)) if np.isfinite(v).any() else None}
               for k, v in res.items()}
    (out.parent / 'concept_raw_c{}_summary.json'.format(cutoff_days)).write_text(json.dumps(
        {'format': FORMAT, 'country': country, 'cutoff_days': cutoff_days, 'rows': int(n), 'fields': summary}, indent=1))
    return out


def raw_path(artifact_root, country, cutoff_days):
    return Path(artifact_root) / 'knowledge' / country / 'concept_raw_c{}.npz'.format(cutoff_days)


def estimator_cutoff(cutoff_mode, cutoff_days):
    """The estimator cutoff that matches a run's input cutoff."""
    if cutoff_mode in ('all_slots', 'harvest'):
        return 0
    if cutoff_mode == 'before_harvest':
        return int(cutoff_days)
    raise ValueError('no knowledge estimators for cutoff_mode {}'.format(cutoff_mode))


def load_raw(artifact_root, country, cutoff_days=CUTOFF_DAYS):
    with np.load(raw_path(artifact_root, country, cutoff_days)) as z:
        if str(z['format']) != FORMAT or int(z['cutoff_days']) != int(cutoff_days):
            raise ValueError('unexpected concept_raw format or cutoff')
        return {k: z[k] for k in z.files}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--artifact_root', required=True)
    p.add_argument('--countries', default=','.join(COUNTRIES))
    p.add_argument('--block_rows', type=int, default=262144)
    p.add_argument('--cutoff_days', type=int, default=CUTOFF_DAYS,
                   help='use only slots dated <= harvest - this (match the fine-tuning input cutoff)')
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    for c in a.countries.split(','):
        print(build_country(a.artifact_root, c, a.block_rows, a.overwrite, a.cutoff_days))


if __name__ == '__main__':
    main()
