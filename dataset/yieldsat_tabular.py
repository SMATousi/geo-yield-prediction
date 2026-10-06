"""Flat tabular view of a YieldSAT country-crop pair for TabM / LightGBM (TM-02).

One row per cache cell of the pair (all field seasons of that crop in the
country). Columns follow spec/yieldsat-tabm.md §2 (feature set F0, optionally
F1 with the S6 neighbourhood stream):

  values  temporal  24 slots x 16 channels (S2 bands + weather), then (F1) 24 x 12
                    neighbourhood S2
          time      24 x 3 (days since seeding / 365, day-of-year sin, cos)
          static    dem, aspect sin, aspect cos, curvature, slope, twi, soil (48)
  masks   one validity flag per slot and stream (S2, weather, neighbourhood; cloud
          masking invalidates a slot's bands together), per static value (aspect
          sin/cos share the aspect flag), and per slot (dated)

Validity is the point contract's: temporal_valid_mask (first dated slot's
weather invalid), static_valid_mask (v3), cutoff ``all_slots`` (paper
protocols: every dated slot). Values stay raw here; ``standardize`` applies
train-only per-column statistics per fold, and invalid values become 0.
"""
import json
from pathlib import Path

import numpy as np

from dataset.yieldsat_cache import static_valid_mask, temporal_valid_mask
from dataset.yieldsat_schema import CACHE_NAME, NBR_NAME, NUM_TIME_SLOTS, SOIL, STATIC_CHANNELS, TEMPORAL_CHANNELS

_S = {c: i for i, c in enumerate(STATIC_CHANNELS)}


def build_pair(artifact_root, country, crop, neighbourhood=False):
    """Returns dict with values (N, V) float32 (raw, NaN invalid), masks (N, M) uint8,
    target (N,), season (N,) index into ``seasons``, grid_row/col, column names."""
    root = Path(artifact_root)
    fields = sorted((f for f in json.loads((root / 'index' / country / 'fields.json').read_text())
                     if f['crop'] == crop), key=lambda f: f['row_start'])
    rows = np.load(root / 'index' / country / 'rows.npz')
    cache = root / CACHE_NAME / country
    T = np.load(cache / 'temporal.npy', mmap_mode='r')
    S = np.load(cache / 'static.npy', mmap_mode='r')
    TM = np.load(cache / 'times.npy', mmap_mode='r')
    nbr = np.load(root / NBR_NAME / country / 's2_nbr5.npy', mmap_mode='r') if neighbourhood else None
    idx = np.concatenate([np.arange(f['row_start'], f['row_end']) for f in fields])
    season = np.concatenate([np.full(f['row_end'] - f['row_start'], i, np.int32) for i, f in enumerate(fields)])
    n = len(idx)
    vals, masks, vnames, mnames = [], [], [], []
    for a in range(0, n, 200000):
        sl = idx[a:a + 200000]
        lo, hi = sl[0], sl[-1] + 1                       # fields are contiguous row ranges
        t = np.asarray(T[lo:hi])[sl - lo].astype(np.float32)
        tm = np.asarray(TM[lo:hi])[sl - lo].astype(np.float64)
        s = np.asarray(S[lo:hi])[sl - lo].astype(np.float32)
        sd = rows['seeding_day'][sl].astype(np.float64)
        tv = temporal_valid_mask(t, tm) & np.isfinite(tm)[:, :, None]
        sv = static_valid_mask(s)
        dated = np.isfinite(tm)
        with np.errstate(invalid='ignore'):
            rel = np.where(dated, (tm - sd[:, None]) / 365.0, np.nan)
            doy = np.mod(tm, 365.2425) / 365.2425 * 2 * np.pi
        parts_v = [np.where(tv, t, np.nan).reshape(len(sl), -1),
                   np.stack([rel, np.sin(doy), np.cos(doy)], -1).reshape(len(sl), -1)]
        parts_m = [tv[:, :, :12].any(-1), tv[:, :, 12:].any(-1)]
        if nbr is not None:
            nb = np.asarray(nbr[lo:hi])[sl - lo].astype(np.float32)
            nv = np.isfinite(nb) & dated[:, :, None]
            parts_v.insert(1, np.where(nv, nb, np.nan).reshape(len(sl), -1))
            parts_m.append(nv.any(-1))
        asp = np.deg2rad(s[:, _S['aspect']])
        stat_cols = [s[:, _S['dem']], np.sin(asp), np.cos(asp), s[:, _S['curvature']], s[:, _S['slope']],
                     s[:, _S['twi']]] + [s[:, _S[c]] for c in SOIL]
        stat_valid = [sv[:, _S['dem']], sv[:, _S['aspect']], sv[:, _S['aspect']], sv[:, _S['curvature']],
                      sv[:, _S['slope']], sv[:, _S['twi']]] + [sv[:, _S[c]] for c in SOIL]
        st = np.stack(stat_cols, 1)
        stv = np.stack(stat_valid, 1)
        parts_v.append(np.where(stv, st, np.nan))
        parts_m += [stv[:, [0, 1, 3, 4, 5] + list(range(6, stv.shape[1]))], dated]
        vals.append(np.concatenate(parts_v, 1).astype(np.float32))
        masks.append(np.concatenate(parts_m, 1).astype(np.uint8))
    tnames = ['{}@{}'.format(c, k) for k in range(NUM_TIME_SLOTS) for c in TEMPORAL_CHANNELS]
    vnames = list(tnames)
    if neighbourhood:
        vnames += ['nbr5_{}@{}'.format(c, k) for k in range(NUM_TIME_SLOTS) for c in TEMPORAL_CHANNELS[:12]]
    vnames += ['{}@{}'.format(c, k) for k in range(NUM_TIME_SLOTS) for c in ('days_since_seeding', 'doy_sin', 'doy_cos')]
    snames = ['dem', 'aspect_sin', 'aspect_cos', 'curvature', 'slope', 'twi'] + list(SOIL)
    vnames += snames
    mnames = ['valid:s2@{}'.format(k) for k in range(NUM_TIME_SLOTS)] + ['valid:weather@{}'.format(k) for k in range(NUM_TIME_SLOTS)]
    if neighbourhood:
        mnames += ['valid:nbr5@{}'.format(k) for k in range(NUM_TIME_SLOTS)]
    mnames += ['valid:' + c for c in ['dem', 'aspect', 'curvature', 'slope', 'twi'] + list(SOIL)]
    mnames += ['dated@{}'.format(k) for k in range(NUM_TIME_SLOTS)]
    out = {'values': np.concatenate(vals), 'masks': np.concatenate(masks), 'value_names': vnames,
           'mask_names': mnames, 'target': rows['target'][idx].astype(np.float32), 'season': season,
           'grid_row': rows['grid_row'][idx], 'grid_col': rows['grid_col'][idx], 'seasons': fields,
           'cache_rows': idx}
    assert out['values'].shape[1] == len(vnames) and out['masks'].shape[1] == len(mnames)
    return out


def fold_rows(pair, split):
    """Row indices of the pair for each partition of a fold manifest (season-based)."""
    sid = {f['season_id']: i for i, f in enumerate(pair['seasons'])}
    out = {}
    for p in ('train', 'val', 'test'):
        keep = np.array(sorted(sid[s] for s in split['partitions'][p] if s in sid), dtype=np.int64)
        out[p] = np.flatnonzero(np.isin(pair['season'], keep))
    return out


def standardize_stats(values, train_rows):
    """Per-column mean/std over valid (finite) training values."""
    v = values[train_rows]
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(v, 0)
        std = np.nanstd(v, 0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(values, mean, std):
    z = (values - mean) / std
    return np.where(np.isfinite(z), z, 0.0).astype(np.float32)
