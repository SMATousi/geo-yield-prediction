# --------------------------------------------------------
# YieldSAT point-mode dataset adapter (YS-04, YS-05; spec YS-R02..R04).
#
# One item is one grid-cell history (``point_timeseries`` mode):
#   inputs[stream]  - values in the stream's canonical channel order,
#                     standardized with train-only statistics, 0 where invalid
#   masks[stream]   - validity per value ((T, C) temporal, (C,) static)
#   available       - 1 if the stream has any valid value for this cell
#   time_features   - (T, 3): days since seeding / 365, doy sin, doy cos
#   time_valid      - (T,) slot has a date and passes the cutoff
#   target / target_raw / target_valid, crop id, field/row/col metadata
# Masks are independent: an observed date does not make optical or weather
# usable, static groups stay valid whatever the time mask says, and uint code
# zero is never treated as missing. Two backends return identical items:
# 'cache' (local memmaps from yieldsat_cache) and 'h5' (bounded reads of the
# source NetCDF, file handle opened per worker).
# --------------------------------------------------------

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from dataset.yieldsat_cache import (
    FIELD_STATS_VERSION, N_S, N_T, SourceLayout, canonicalize_block, load_cache_manifest,
    temporal_valid_mask,
)
from dataset.yieldsat_schema import (
    CROPS, NUM_TIME_SLOTS, STATIC_CHANNELS, STREAMS, TEMPORAL_CHANNELS, SOIL_UNCERTAINTY,
)
from dataset.yieldsat_source import load_country_index, open_source, source_path

CUTOFF_MODES = ('before_harvest', 'after_seeding', 'harvest', 'all_slots')
RETROSPECTIVE_MODES = ('harvest', 'all_slots')

_T_POS = {c: i for i, c in enumerate(TEMPORAL_CHANNELS)}
_S_POS = {c: i for i, c in enumerate(STATIC_CHANNELS)}


def stream_layout(streams, soil_uncertainty='none', aspect_encoding='raw'):
    """Channel names and positions (in the canonical temporal/static arrays)
    for each selected stream."""
    layout = {}
    for name in streams:
        spec = STREAMS[name]
        channels = list(spec['channels'])
        if spec['temporal']:
            pos = [_T_POS[c] for c in channels]
        else:
            if name == 'yieldsat_soil' and soil_uncertainty == 'ancillary':
                channels = channels + list(SOIL_UNCERTAINTY)
            pos = [_S_POS[c] for c in channels]
        out_channels = list(channels)
        if name == 'yieldsat_terrain' and aspect_encoding == 'cyclic':
            out_channels = ['aspect_sin', 'aspect_cos'] + [c for c in channels if c != 'aspect']
        layout[name] = {'temporal': spec['temporal'], 'channels': channels,
                        'positions': np.array(pos, dtype=np.int64),
                        'out_channels': out_channels}
    return layout


# ---- train-only normalization ---------------------------------------------

class YieldSATNormalizer:
    """Per-channel mean/std fitted from per-field sufficient statistics of
    *training* field seasons only (supplied ``stats-*`` are never used).
    ``pooling='pooled'`` fits one set over all training countries;
    ``'per_country'`` fits one set per country (a held-out country then has
    no statistics and is refused)."""

    def __init__(self, stats, pooling, fitted_on):
        self.stats = stats
        self.pooling = pooling
        self.fitted_on = fitted_on

    @staticmethod
    def _moments(count, s, ss):
        count = np.maximum(count, 1)
        mean = s / count
        var = np.maximum(ss / count - mean * mean, 0.0)
        return mean, np.sqrt(var)

    @classmethod
    def fit(cls, artifact_root, train_fields, pooling='pooled'):
        """train_fields: list of (country, field_code, target_values ndarray)."""
        by_country = {}
        for country, code, tvals in train_fields:
            by_country.setdefault(country, []).append((code, tvals))
        acc = {}
        for country, items in by_country.items():
            cache_dir = Path(artifact_root) / 'cache' / country
            version = json.loads((cache_dir / 'cache_manifest.json').read_text()).get('field_stats_version')
            if version != FIELD_STATS_VERSION:
                raise ValueError('{}: field statistics use validity rule v{}, need v{}; run '
                                 'yieldsat_prepare.py stats'.format(country, version, FIELD_STATS_VERSION))
            fs = np.load(cache_dir / 'field_stats.npz')
            slot = {int(c): i for i, c in enumerate(fs['field_codes'])}
            sel = np.array([slot[int(code)] for code, _ in items])
            key = 'all' if pooling == 'pooled' else country
            a = acc.setdefault(key, {k: 0.0 for k in ('tc', 'ts', 'tss', 'sc', 'ss', 'sss', 'yc', 'ys', 'yss')})
            a['tc'] = a['tc'] + fs['temporal_count'][sel].sum(0)
            a['ts'] = a['ts'] + fs['temporal_sum'][sel].sum(0)
            a['tss'] = a['tss'] + fs['temporal_sumsq'][sel].sum(0)
            a['sc'] = a['sc'] + fs['static_count'][sel].sum(0)
            a['ss'] = a['ss'] + fs['static_sum'][sel].sum(0)
            a['sss'] = a['sss'] + fs['static_sumsq'][sel].sum(0)
            y = np.concatenate([t[np.isfinite(t)] for _, t in items]).astype(np.float64)
            a['yc'] += y.size
            a['ys'] += y.sum()
            a['yss'] += (y * y).sum()
        stats = {}
        for key, a in acc.items():
            tm, tsd = cls._moments(a['tc'], a['ts'], a['tss'])
            sm, ssd = cls._moments(a['sc'], a['ss'], a['sss'])
            ym, ysd = cls._moments(np.array([a['yc']]), np.array([a['ys']]), np.array([a['yss']]))
            stats[key] = {
                'temporal_mean': tm, 'temporal_std': np.where(tsd > 0, tsd, 1.0),
                'temporal_count': np.asarray(a['tc']),
                'static_mean': sm, 'static_std': np.where(ssd > 0, ssd, 1.0),
                'static_count': np.asarray(a['sc']),
                'target_mean': float(ym[0]), 'target_std': float(ysd[0]) if ysd[0] > 0 else 1.0,
            }
        fitted_on = sorted({c for c, _, _ in train_fields})
        return cls(stats, pooling, fitted_on)

    def get(self, country):
        key = 'all' if self.pooling == 'pooled' else country
        if key not in self.stats:
            raise KeyError('no training statistics for {} (pooling={})'.format(country, self.pooling))
        return self.stats[key]

    def to_json(self):
        return {'pooling': self.pooling, 'fitted_on': self.fitted_on,
                'temporal_channels': list(TEMPORAL_CHANNELS),
                'static_channels': list(STATIC_CHANNELS),
                'stats': {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                              for kk, vv in v.items()} for k, v in self.stats.items()}}

    @classmethod
    def from_json(cls, data):
        stats = {k: {kk: (np.asarray(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()}
                 for k, v in data['stats'].items()}
        return cls(stats, data['pooling'], data['fitted_on'])


def load_supplied_stats(source_root, country):
    """The file's own ``stats-mean``/``stats-std`` in canonical channel order
    (PC-03 ablation only: their fitting population is unknown and may include
    test fields, so they are never the default)."""
    from dataset.yieldsat_schema import read_band_names, resolve_channel_indices
    with open_source(source_path(source_root, country)) as f:
        names = read_band_names(f)
        mean = f['stats-mean'][:].astype(np.float64)
        std = f['stats-std'][:].astype(np.float64)
    ti = resolve_channel_indices(names, TEMPORAL_CHANNELS)
    si = resolve_channel_indices(names, STATIC_CHANNELS)
    fix = lambda v: np.where(np.isfinite(v) & (v > 0), v, 1.0)
    return {'temporal_mean': np.nan_to_num(mean[ti]), 'temporal_std': fix(std[ti]),
            'static_mean': np.nan_to_num(mean[si]), 'static_std': fix(std[si])}


class NormalizerView:
    """Feature/target normalization policy on top of a fitted normalizer.

    ``features``: 'train' (train-fold statistics, default), 'supplied'
    (the file's ``stats-*``, per country) or 'none' (raw values).
    ``target``: 'train' (standardized with train statistics) or 'none' (raw
    t/ha). Used to mirror the paper's protocol variants (PC-03).
    """

    def __init__(self, base, features='train', target='train', supplied=None):
        if features not in ('train', 'supplied', 'none') or target not in ('train', 'none'):
            raise ValueError('bad normalization policy')
        if features == 'supplied' and not supplied:
            raise ValueError('supplied statistics required')
        self.base, self.features, self.target, self.supplied = base, features, target, supplied

    def get(self, country):
        st = dict(self.base.get(country))
        if self.features == 'supplied':
            st.update(self.supplied[country])
        elif self.features == 'none':
            for k in ('temporal', 'static'):
                st[k + '_mean'] = np.zeros_like(st[k + '_mean'])
                st[k + '_std'] = np.ones_like(st[k + '_std'])
        if self.target == 'none':
            st['target_mean'], st['target_std'] = 0.0, 1.0
        return st

    def to_json(self):
        out = self.base.to_json()
        out['policy'] = {'features': self.features, 'target': self.target}
        return out


# ---- row readers -------------------------------------------------------------

class _CacheReader:
    def __init__(self, artifact_root, source_root, country, check_source=True):
        load_cache_manifest(artifact_root, source_root, country, check_source=check_source)
        self.dir = Path(artifact_root) / 'cache' / country
        self._arrays = None

    def _open(self):
        if self._arrays is None:
            self._arrays = tuple(np.load(self.dir / n, mmap_mode='r') for n in
                                 ('temporal.npy', 'static.npy', 'times.npy'))
        return self._arrays

    def read(self, rows):
        temporal, static, times = self._open()
        return (np.asarray(temporal[rows]), np.asarray(static[rows]), np.asarray(times[rows]))


class _H5Reader:
    """Bounded direct reads. Sorted row requests are grouped into contiguous
    runs; runs are read as slices (never ``sample[:]``)."""

    def __init__(self, source_root, country, max_run=8192):
        self.path = source_path(source_root, country)
        self.max_run = max_run
        self._file = None
        self._pid = None
        self._layout = None

    def _open(self):
        if self._file is None or self._pid != os.getpid():
            self._file = open_source(self.path)
            self._pid = os.getpid()
            self._layout = SourceLayout(self._file)
        return self._file

    def read(self, rows):
        f = self._open()
        rows = np.asarray(rows)
        order = np.argsort(rows)
        srt = rows[order]
        breaks = np.flatnonzero(np.diff(srt) != 1) + 1
        samples, times = [], []
        for run in np.split(srt, breaks):
            for s in range(0, len(run), self.max_run):
                a, b = int(run[s]), int(run[min(len(run), s + self.max_run) - 1]) + 1
                samples.append(f['sample'][a:b])
                times.append(f['times'][a:b])
        temporal, static, t, _, _ = canonicalize_block(np.concatenate(samples), np.concatenate(times),
                                                      self._layout)
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        return temporal[inv], static[inv], t[inv]

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


# ---- dataset -------------------------------------------------------------------

class YieldSATPointDataset(Dataset):
    """Point-mode items for the field seasons of one split partition.

    ``seasons``: list of field-season dicts (from ``load_field_table``) to
    include. Rows are addressed by a global position over the concatenated
    contiguous row ranges of those seasons.
    """

    def __init__(self, source_root, artifact_root, seasons, normalizer, streams=None,
                 backend='cache', cutoff_mode='before_harvest', cutoff_days=30,
                 soil_uncertainty='none', aspect_encoding='raw', max_rows_per_field=None,
                 seed=0, check_source=True, fill_value=0.0, neighbourhood_root=None):
        if cutoff_mode not in CUTOFF_MODES:
            raise ValueError('cutoff_mode must be one of {}'.format(CUTOFF_MODES))
        if backend not in ('cache', 'h5'):
            raise ValueError('backend must be cache or h5')
        self.streams = tuple(streams or STREAMS)
        self.layout = stream_layout(self.streams, soil_uncertainty, aspect_encoding)
        self.normalizer = normalizer
        self.cutoff_mode = cutoff_mode
        self.cutoff_days = cutoff_days
        self.aspect_encoding = aspect_encoding
        self.fill_value = float(fill_value)
        self.retrospective = cutoff_mode in RETROSPECTIVE_MODES
        self.countries = sorted({s['country'] for s in seasons})
        self.index = {c: load_country_index(artifact_root, source_root, c, check_source)
                      for c in self.countries}
        self.readers = {}
        for c in self.countries:
            self.readers[c] = (_CacheReader(artifact_root, source_root, c, check_source)
                               if backend == 'cache' else _H5Reader(source_root, c))
        rng = np.random.default_rng(seed)
        seg_country, seg_rows, seg_season = [], [], []
        for i, s in enumerate(sorted(seasons, key=lambda s: (s['country'], s['row_start']))):
            n = s['n_rows']
            if max_rows_per_field is not None and n > max_rows_per_field:
                rows = np.sort(rng.choice(n, max_rows_per_field, replace=False)) + s['row_start']
            else:
                rows = np.arange(s['row_start'], s['row_end'])
            seg_rows.append(rows)
            seg_country.append(np.full(len(rows), self.countries.index(s['country']), dtype=np.int16))
            seg_season.append(np.full(len(rows), i, dtype=np.int32))
        self.seasons = sorted(seasons, key=lambda s: (s['country'], s['row_start']))
        self.row = np.concatenate(seg_rows) if seg_rows else np.zeros(0, dtype=np.int64)
        self.country_of = np.concatenate(seg_country) if seg_country else np.zeros(0, dtype=np.int16)
        self.season_of = np.concatenate(seg_season) if seg_season else np.zeros(0, dtype=np.int32)
        # item ranges per season (for samplers and field-level aggregation)
        bounds = np.flatnonzero(np.diff(self.season_of) != 0) + 1
        self.season_ranges = list(zip(np.concatenate([[0], bounds]).tolist(),
                                      np.concatenate([bounds, [len(self.row)]]).tolist()))
        self._crop_ids = np.array([CROPS.index(s['crop']) for s in self.seasons], dtype=np.int64)
        # field-season mean target (raw t/ha), for the S5 season-level loss only
        self._season_target = np.array([s.get('target_mean', np.nan) for s in self.seasons], dtype=np.float32)
        # S6: per-cell 5x5 neighbourhood mean of the S2 bands (yieldsat_build_neighbourhood.py),
        # aligned with the cache rows; one more temporal stream
        self.neighbourhood = None
        if neighbourhood_root:
            self.neighbourhood = {c: np.load(Path(neighbourhood_root) / c / 's2_nbr5.npy', mmap_mode='r')
                                  for c in self.countries}
            s2 = list(STREAMS['yieldsat_s2']['channels'])
            self.layout['yieldsat_s2_nbr'] = {'temporal': True, 'channels': s2,
                                              'positions': np.array([_T_POS[c] for c in s2], dtype=np.int64),
                                              'out_channels': ['nbr5_' + c for c in s2]}

    def __len__(self):
        return len(self.row)

    # -- batch path (DataLoader calls __getitems__ with a list of indices) --
    def __getitems__(self, indices):
        batch = self.get_batch(np.asarray(indices, dtype=np.int64))
        return [batch]

    def __getitem__(self, i):
        return self.get_batch(np.array([i], dtype=np.int64))

    def cutoff_day(self, seeding_day, harvest_day):
        if self.cutoff_mode == 'before_harvest':
            return harvest_day - self.cutoff_days
        if self.cutoff_mode == 'after_seeding':
            return seeding_day + self.cutoff_days
        if self.cutoff_mode == 'harvest':
            return harvest_day.astype(np.float64)
        return np.full(seeding_day.shape, np.inf)

    def get_batch(self, idx):
        B = len(idx)
        temporal = np.empty((B, NUM_TIME_SLOTS, N_T), dtype=np.float32)
        static = np.empty((B, N_S), dtype=np.float32)
        times = np.empty((B, NUM_TIME_SLOTS), dtype=np.float32)
        meta = {k: np.empty(B, dtype=np.int64) for k in ('seeding_day', 'harvest_day', 'grid_row',
                                                         'grid_col', 'field_code')}
        target_raw = np.empty(B, dtype=np.float32)
        nbr = (np.full((B, NUM_TIME_SLOTS, len(STREAMS['yieldsat_s2']['channels'])), np.nan, np.float32)
               if self.neighbourhood is not None else None)
        t_mean = np.empty((B, N_T), dtype=np.float32)
        t_std = np.empty((B, N_T), dtype=np.float32)
        s_mean = np.empty((B, N_S), dtype=np.float32)
        s_std = np.empty((B, N_S), dtype=np.float32)
        y_mean = np.empty(B, dtype=np.float32)
        y_std = np.empty(B, dtype=np.float32)
        for ci in np.unique(self.country_of[idx]):
            sel = np.flatnonzero(self.country_of[idx] == ci)
            country = self.countries[ci]
            rows = self.row[idx[sel]]
            tt, ss, tm = self.readers[country].read(rows)
            temporal[sel], static[sel], times[sel] = tt, ss, tm
            r = self.index[country]['rows']
            meta['seeding_day'][sel] = r['seeding_day'][rows]
            meta['harvest_day'][sel] = r['harvest_day'][rows]
            meta['grid_row'][sel] = r['grid_row'][rows]
            meta['grid_col'][sel] = r['grid_col'][rows]
            meta['field_code'][sel] = r['field_code'][rows]
            target_raw[sel] = r['target'][rows]
            if nbr is not None:
                order = np.argsort(rows)
                nbr[sel[order]] = np.asarray(self.neighbourhood[country][rows[order]], dtype=np.float32)
            st = self.normalizer.get(country)
            t_mean[sel], t_std[sel] = st['temporal_mean'], st['temporal_std']
            s_mean[sel], s_std[sel] = st['static_mean'], st['static_std']
            y_mean[sel], y_std[sel] = st['target_mean'], st['target_std']

        cutoff = self.cutoff_day(meta['seeding_day'], meta['harvest_day'])
        time_valid = np.isfinite(times) & (times <= cutoff[:, None])
        # temporal values: valid value (see temporal_valid_mask) AND eligible
        # slot; a date alone is not validity
        t_valid = temporal_valid_mask(temporal, times) & time_valid[:, :, None]
        t_norm = np.where(t_valid, (temporal - t_mean[:, None, :]) / t_std[:, None, :],
                          self.fill_value)
        s_valid = np.isfinite(static)
        s_norm = np.where(s_valid, (static - s_mean) / s_std, self.fill_value)

        with np.errstate(invalid='ignore'):
            rel = (times - meta['seeding_day'][:, None]) / 365.0
            doy = np.mod(times, 365.2425) / 365.2425 * 2 * np.pi
            time_features = np.stack([rel, np.sin(doy), np.cos(doy)], axis=-1)
        time_features = np.where(time_valid[:, :, None], time_features, 0.0).astype(np.float32)

        inputs, masks, available = {}, {}, {}
        for name, lay in self.layout.items():
            pos = lay['positions']
            if name == 'yieldsat_s2_nbr':
                nv = np.isfinite(nbr) & time_valid[:, :, None]
                v = np.where(nv, (nbr - t_mean[:, None, pos]) / t_std[:, None, pos], self.fill_value)
                m = nv
            elif lay['temporal']:
                v, m = t_norm[:, :, pos], t_valid[:, :, pos]
            else:
                v, m = s_norm[:, pos], s_valid[:, pos]
                if name == 'yieldsat_terrain' and self.aspect_encoding == 'cyclic':
                    # aspect is channel 0 of TERRAIN; degrees per RichDEM
                    a = np.deg2rad(static[:, pos[0]])
                    am = m[:, 0]
                    sin_cos = np.stack([np.where(am, np.sin(a), 0.0),
                                        np.where(am, np.cos(a), 0.0)], axis=1)
                    v = np.concatenate([sin_cos, v[:, 1:]], axis=1)
                    m = np.concatenate([am[:, None], am[:, None], m[:, 1:]], axis=1)
            inputs[name] = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
            masks[name] = torch.from_numpy(np.ascontiguousarray(m))
            available[name] = torch.from_numpy(m.reshape(B, -1).any(axis=1).astype(np.float32))

        target_valid = np.isfinite(target_raw)
        target = np.where(target_valid, (target_raw - y_mean) / y_std, 0.0).astype(np.float32)
        season = self.season_of[idx]
        out = {
            'inputs': inputs,
            'masks': masks,
            'available': available,
            'time_features': torch.from_numpy(time_features),
            'time_valid': torch.from_numpy(time_valid),
            'target': torch.from_numpy(target),
            'target_raw': torch.from_numpy(target_raw),
            'target_valid': torch.from_numpy(target_valid),
            'target_mean': torch.from_numpy(y_mean),
            'target_std': torch.from_numpy(y_std),
            'crop': torch.from_numpy(self._crop_ids[season]),
            'season_target': torch.from_numpy(((self._season_target[season] - y_mean) / y_std).astype(np.float32)),
            'season': torch.from_numpy(season.astype(np.int64)),
            'item': torch.from_numpy(idx),
            'grid_row': torch.from_numpy(meta['grid_row']),
            'grid_col': torch.from_numpy(meta['grid_col']),
        }
        # knowledge pretraining (yieldsat_knowledge_point.attach_knowledge): per-item
        # concept soft targets (NaN = unknown) and rule gates
        knowledge = getattr(self, 'knowledge', None)
        if knowledge is not None:
            out['concept_target'] = torch.from_numpy(knowledge['concept_target'][idx].astype(np.float32))
            out['rule_gate'] = torch.from_numpy(knowledge['rule_gate'][idx].astype(np.float32))
        return out


def collate_point_batch(items):
    """``__getitems__`` already returns one assembled batch."""
    if len(items) == 1:
        return items[0]
    raise ValueError('use a batch sampler with YieldSATPointDataset')


class FieldBalancedBatchSampler(Sampler):
    """Draw training batches so millions of pixels from large fields do not
    dominate: a season is drawn with probability proportional to
    ``n_rows ** alpha`` (0 = field-balanced, 1 = pixel-weighted), then a
    contiguous block of ``block_size`` items from it (block reads are what
    make the direct HDF5 backend affordable; use 1 with the cache)."""

    def __init__(self, season_ranges, batch_size, batches_per_epoch, alpha=0.5, block_size=1,
                 seed=0):
        self.ranges = np.asarray(season_ranges, dtype=np.int64)
        sizes = (self.ranges[:, 1] - self.ranges[:, 0]).astype(np.float64)
        w = sizes ** alpha
        self.p = w / w.sum()
        self.batch_size = batch_size
        self.batches = batches_per_epoch
        self.block = max(1, block_size)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.batches

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        per_batch = max(1, self.batch_size // self.block)
        for _ in range(self.batches):
            seasons = rng.choice(len(self.ranges), size=per_batch, p=self.p)
            idx = []
            for s in seasons:
                a, b = self.ranges[s]
                n = min(self.block, b - a)
                start = rng.integers(a, b - n + 1)
                if self.block == 1:
                    idx.append(np.array([start]))
                else:
                    idx.append(np.arange(start, start + n))
            yield np.concatenate(idx)[: self.batch_size].tolist()


class SequentialBatchSampler(Sampler):
    def __init__(self, n, batch_size):
        self.n, self.batch_size = n, batch_size

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for s in range(0, self.n, self.batch_size):
            yield list(range(s, min(self.n, s + self.batch_size)))
