# --------------------------------------------------------
# YieldSAT canonicalization, full-corpus audit pass and compact cache (YS-01,
# YS-04, YS-05).
#
# ``canonicalize_block`` is the single place where a block of source rows
# (N, 24, 120) in a file's own band order becomes the canonical layout:
#   temporal (N, 24, 16)  = 12 S2 bands + 4 weather interval features
#   static   (N, 104)     = dem, 4 terrain, 48 soil, 48 soil uncertainty, 3 coords
#   times    (N, 24)      = days since 1970-01-01 (NaN where the slot has no date)
# Static channels repeat through the 24 slots in the source; one verified
# representation is kept (first finite slot) and repetitions that disagree
# beyond tolerance are flagged per row, never averaged away. Both the direct
# HDF5 backend and the cache builder use it, so they produce identical rows.
#
# The cache is an optional, reproducible, bounded artifact outside the source
# root (~2 KB/row instead of 11.5 KB) that turns random row access into cheap
# local memmap reads. The same pass accumulates per-field sufficient
# statistics so train-only normalization never needs a second source read.
# --------------------------------------------------------

import json
import os
import time
from pathlib import Path

import numpy as np

from dataset.yieldsat_schema import (
    ALL_CHANNELS,
    CONTRACT_KEY,
    NUM_TIME_SLOTS,
    OPTICAL,
    STATIC_CHANNELS,
    TEMPORAL_CHANNELS,
    WEATHER,
    origin_offset_days,
    parse_time_origin,
    read_band_names,
    resolve_channel_indices,
)
from dataset.yieldsat_source import fingerprint, open_source, same_snapshot, source_path

CACHE_VERSION = 1
# Version of the per-field statistics' validity rule (see temporal_valid_mask).
FIELD_STATS_VERSION = 2
STATIC_RTOL = 1e-5
STATIC_ATOL = 1e-6
FLAG_STATIC_CONFLICT = 1
FLAG_STATIC_MISSING = 2
FLAG_NO_VALID_TIME = 4

N_T = len(TEMPORAL_CHANNELS)
N_S = len(STATIC_CHANNELS)
OPT_SLICE = slice(0, len(OPTICAL))
WEA_SLICE = slice(len(OPTICAL), len(OPTICAL) + len(WEATHER))


class SourceLayout:
    """Per-file channel positions and time origin."""

    def __init__(self, h5file):
        names = read_band_names(h5file)
        resolve_channel_indices(names, ALL_CHANNELS)  # full-schema validation
        self.temporal_idx = resolve_channel_indices(names, TEMPORAL_CHANNELS)
        self.static_idx = resolve_channel_indices(names, STATIC_CHANNELS)
        attrs = h5file['times'].attrs
        units = attrs['units']
        calendar = attrs['calendar']
        units = units.decode() if isinstance(units, bytes) else units
        calendar = calendar.decode() if isinstance(calendar, bytes) else calendar
        self.origin_offset = origin_offset_days(parse_time_origin(units, calendar))


def canonicalize_block(sample, times, layout):
    """Source rows -> (temporal, static, times_1970, flags, conflict_counts)."""
    sample = np.asarray(sample, dtype=np.float32)
    temporal = sample[:, :, layout.temporal_idx]
    stat_all = sample[:, :, layout.static_idx]                 # (N, 24, S)
    finite = np.isfinite(stat_all)
    any_finite = finite.any(axis=1)                            # (N, S)
    first = np.argmax(finite, axis=1)                          # first finite slot
    static = np.take_along_axis(stat_all, first[:, None, :], axis=1)[:, 0, :]
    static = np.where(any_finite, static, np.nan).astype(np.float32)
    dev = np.abs(np.where(finite, stat_all - static[:, None, :], 0.0))
    tol = STATIC_ATOL + STATIC_RTOL * np.abs(static)
    conflict = (dev > np.nan_to_num(tol, nan=0.0)[:, None, :]).any(axis=1)  # (N, S)
    t = np.asarray(times, dtype=np.float64) + layout.origin_offset
    flags = np.zeros(sample.shape[0], dtype=np.uint8)
    flags |= np.where(conflict.any(axis=1), FLAG_STATIC_CONFLICT, 0).astype(np.uint8)
    flags |= np.where((~any_finite).any(axis=1), FLAG_STATIC_MISSING, 0).astype(np.uint8)
    flags |= np.where(~np.isfinite(t).any(axis=1), FLAG_NO_VALID_TIME, 0).astype(np.uint8)
    return temporal, static, t.astype(np.float32), flags, conflict.sum(axis=0)


def temporal_valid_mask(temporal, times):
    """Validity of canonical temporal values before any cutoff.

    A value is valid when it is finite on a dated slot, except weather at a
    row's first dated slot: its interval start is not recorded (Argentina and
    Brazil store 0 there, Germany and Uruguay a sum over an unknown span), so
    it is not a usable observation.
    """
    tvalid = np.isfinite(times)
    valid = np.isfinite(temporal) & tvalid[:, :, None]
    has = np.flatnonzero(tvalid.any(axis=1))
    first = np.argmax(tvalid, axis=1)
    valid[has, first[has], WEA_SLICE] = False
    return valid


class _Accumulator:
    """Per-field sums for count/sum/sumsq/min/max of every canonical channel."""

    def __init__(self, n_fields, n_channels):
        self.count = np.zeros((n_fields, n_channels), dtype=np.int64)
        self.sum = np.zeros((n_fields, n_channels), dtype=np.float64)
        self.sumsq = np.zeros((n_fields, n_channels), dtype=np.float64)
        self.min = np.full((n_fields, n_channels), np.inf)
        self.max = np.full((n_fields, n_channels), -np.inf)

    def add(self, slot, values):
        """values: (M, C) with NaN = invalid, all belonging to field ``slot``."""
        valid = np.isfinite(values)
        v = np.where(valid, values, 0.0).astype(np.float64)
        self.count[slot] += valid.sum(axis=0)
        self.sum[slot] += v.sum(axis=0)
        self.sumsq[slot] += (v * v).sum(axis=0)
        if valid.any():
            self.min[slot] = np.minimum(self.min[slot], np.where(valid, values, np.inf).min(axis=0))
            self.max[slot] = np.maximum(self.max[slot], np.where(valid, values, -np.inf).max(axis=0))


def build_country_cache(source_root, country, artifact_root, block_rows=4096,
                        write_cache=True, max_rows=None, log_every=200):
    """One sequential pass over a country's ``sample``/``times``.

    Writes (when ``write_cache``) ``temporal.npy``, ``static.npy``,
    ``times.npy``, ``flags.npy`` under ``<artifact_root>/cache/<country>/``
    plus ``field_stats.npz`` and ``cache_manifest.json`` (written last; a
    cache without a completed manifest is ignored). Requires the index.
    """
    from dataset.yieldsat_source import load_country_index

    index = load_country_index(artifact_root, source_root, country)
    fields = index['fields']
    code_to_slot = {f['field_code']: i for i, f in enumerate(fields)}
    out = Path(artifact_root) / 'cache' / country
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'cache_manifest.json'
    if manifest_path.exists():
        manifest_path.unlink()
    path = source_path(source_root, country)
    fp = fingerprint(path)
    t0 = time.time()
    with open_source(path) as f:
        layout = SourceLayout(f)
        n = f['sample'].shape[0] if max_rows is None else min(max_rows, f['sample'].shape[0])
        if write_cache:
            mm = np.lib.format.open_memmap
            temporal_mm = mm(out / 'temporal.npy', 'w+', np.float32, (n, NUM_TIME_SLOTS, N_T))
            static_mm = mm(out / 'static.npy', 'w+', np.float32, (n, N_S))
            times_mm = mm(out / 'times.npy', 'w+', np.float32, (n, NUM_TIME_SLOTS))
            flags_mm = mm(out / 'flags.npy', 'w+', np.uint8, (n,))
        acc_t = _Accumulator(len(fields), N_T)
        acc_s = _Accumulator(len(fields), N_S)
        field_code = index['rows']['field_code']
        audit = {
            'rows': int(n),
            'static_conflicts_per_channel': np.zeros(N_S, dtype=np.int64),
            'rows_static_conflict': 0,
            'rows_static_missing_channel': 0,
            'static_missing_per_channel': np.zeros(N_S, dtype=np.int64),
            'rows_no_valid_time': 0,
            'valid_time_slots': 0,
            'valid_time_slots_per_slot': np.zeros(NUM_TIME_SLOTS, dtype=np.int64),
            'valid_time_but_optical_all_nan': 0,
            'valid_time_but_weather_all_nan': 0,
            'dateless_optical_values': 0,
            'dateless_weather_values': 0,
            'rows_no_optical': 0,
            'optical_zero_values': 0,
            'optical_negative_values': 0,
            'weather_zero_at_first_valid_slot': 0,
            'first_valid_slots': 0,
            'nonfinite_inf_values': 0,
        }
        for bi, s in enumerate(range(0, n, block_rows)):
            e = min(n, s + block_rows)
            sample = f['sample'][s:e]
            times = f['times'][s:e]
            audit['nonfinite_inf_values'] += int(np.isinf(sample).sum())
            temporal, static, t, flags, conflicts = canonicalize_block(sample, times, layout)
            if write_cache:
                temporal_mm[s:e] = temporal
                static_mm[s:e] = static
                times_mm[s:e] = t
                flags_mm[s:e] = flags
            tvalid = np.isfinite(t)                             # (B, 24)
            opt = temporal[:, :, OPT_SLICE]
            wea = temporal[:, :, WEA_SLICE]
            opt_any = np.isfinite(opt).any(axis=2)
            wea_any = np.isfinite(wea).any(axis=2)
            audit['static_conflicts_per_channel'] += conflicts
            audit['rows_static_conflict'] += int((flags & FLAG_STATIC_CONFLICT > 0).sum())
            audit['rows_static_missing_channel'] += int((flags & FLAG_STATIC_MISSING > 0).sum())
            audit['static_missing_per_channel'] += (~np.isfinite(static)).sum(axis=0)
            audit['rows_no_valid_time'] += int((flags & FLAG_NO_VALID_TIME > 0).sum())
            audit['valid_time_slots'] += int(tvalid.sum())
            audit['valid_time_slots_per_slot'] += tvalid.sum(axis=0)
            audit['valid_time_but_optical_all_nan'] += int((tvalid & ~opt_any).sum())
            audit['valid_time_but_weather_all_nan'] += int((tvalid & ~wea_any).sum())
            audit['dateless_optical_values'] += int((np.isfinite(opt) & ~tvalid[:, :, None]).sum())
            audit['dateless_weather_values'] += int((np.isfinite(wea) & ~tvalid[:, :, None]).sum())
            audit['rows_no_optical'] += int((~opt_any).all(axis=1).sum())
            audit['optical_zero_values'] += int((opt == 0).sum())
            audit['optical_negative_values'] += int((opt < 0).sum())
            has = tvalid.any(axis=1)
            first = np.argmax(tvalid, axis=1)
            fw = wea[np.arange(len(first)), first]
            audit['first_valid_slots'] += int(has.sum())
            audit['weather_zero_at_first_valid_slot'] += int(((fw == 0).all(axis=1) & has).sum())
            # per-field sufficient statistics (fields are contiguous row ranges)
            codes = field_code[s:e]
            tmask = np.where(temporal_valid_mask(temporal, t), temporal, np.nan)
            for code in np.unique(codes):
                sel = codes == code
                slot = code_to_slot[int(code)]
                acc_t.add(slot, tmask[sel].reshape(-1, N_T))
                acc_s.add(slot, static[sel])
            if log_every and bi % log_every == 0:
                rate = e * sample[0].nbytes / 1e6 / max(1e-6, time.time() - t0)
                print('[{}] {}/{} rows, {:.0f} MB/s'.format(country, e, n, rate), flush=True)
        if write_cache:
            for arr in (temporal_mm, static_mm, times_mm, flags_mm):
                arr.flush()
    np.savez(out / 'field_stats.npz',
             field_codes=np.array([fl['field_code'] for fl in fields]),
             temporal_count=acc_t.count, temporal_sum=acc_t.sum, temporal_sumsq=acc_t.sumsq,
             temporal_min=acc_t.min, temporal_max=acc_t.max,
             static_count=acc_s.count, static_sum=acc_s.sum, static_sumsq=acc_s.sumsq,
             static_min=acc_s.min, static_max=acc_s.max)
    audit = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in audit.items()}
    audit['static_conflicts_by_name'] = {
        STATIC_CHANNELS[i]: c for i, c in enumerate(audit['static_conflicts_per_channel']) if c}
    audit['static_missing_by_name'] = {
        STATIC_CHANNELS[i]: c for i, c in enumerate(audit['static_missing_per_channel']) if c}
    manifest = {
        'contract': CONTRACT_KEY,
        'cache_version': CACHE_VERSION,
        'country': country,
        'fingerprint': fp,
        'index_fingerprint': index['manifest']['fingerprint'],
        'rows': int(n),
        'complete': max_rows is None,
        'cache_written': bool(write_cache),
        'temporal_channels': list(TEMPORAL_CHANNELS),
        'static_channels': list(STATIC_CHANNELS),
        'time_units': 'days since 1970-01-01',
        'field_stats_version': FIELD_STATS_VERSION,
        'static_policy': 'first finite slot; rtol={} atol={}; conflicts flagged'.format(
            STATIC_RTOL, STATIC_ATOL),
        'audit': audit,
        'seconds': round(time.time() - t0, 1),
    }
    tmp = out / 'cache_manifest.json.tmp'
    tmp.write_text(json.dumps(manifest, indent=1))
    os.replace(tmp, manifest_path)
    return manifest


def recompute_field_stats(artifact_root, source_root, country, block_rows=65536):
    """Rebuild ``field_stats.npz`` from a completed cache (no source read),
    applying the current validity rule, and record the version."""
    from dataset.yieldsat_source import load_country_index

    manifest = load_cache_manifest(artifact_root, source_root, country)
    d = Path(artifact_root) / 'cache' / country
    fields = load_country_index(artifact_root, source_root, country)['fields']
    temporal = np.load(d / 'temporal.npy', mmap_mode='r')
    static = np.load(d / 'static.npy', mmap_mode='r')
    times = np.load(d / 'times.npy', mmap_mode='r')
    acc_t = _Accumulator(len(fields), N_T)
    acc_s = _Accumulator(len(fields), N_S)
    for slot, fl in enumerate(fields):
        for s in range(fl['row_start'], fl['row_end'], block_rows):
            e = min(fl['row_end'], s + block_rows)
            t = np.asarray(temporal[s:e])
            valid = temporal_valid_mask(t, np.asarray(times[s:e]))
            acc_t.add(slot, np.where(valid, t, np.nan).reshape(-1, N_T))
            acc_s.add(slot, np.asarray(static[s:e]))
    np.savez(d / 'field_stats.npz',
             field_codes=np.array([fl['field_code'] for fl in fields]),
             temporal_count=acc_t.count, temporal_sum=acc_t.sum, temporal_sumsq=acc_t.sumsq,
             temporal_min=acc_t.min, temporal_max=acc_t.max,
             static_count=acc_s.count, static_sum=acc_s.sum, static_sumsq=acc_s.sumsq,
             static_min=acc_s.min, static_max=acc_s.max)
    manifest['field_stats_version'] = FIELD_STATS_VERSION
    tmp = d / 'cache_manifest.json.tmp'
    tmp.write_text(json.dumps(manifest, indent=1))
    os.replace(tmp, d / 'cache_manifest.json')
    return manifest


def load_cache_manifest(artifact_root, source_root, country, check_source=True):
    """Return the manifest of a completed cache, rejecting stale caches."""
    from dataset.yieldsat_source import SnapshotError

    path = Path(artifact_root) / 'cache' / country / 'cache_manifest.json'
    if not path.exists():
        raise SnapshotError('{}: no completed cache at {}'.format(country, path.parent))
    manifest = json.loads(path.read_text())
    if manifest.get('contract') != CONTRACT_KEY or manifest.get('cache_version') != CACHE_VERSION:
        raise SnapshotError('{}: cache built for another contract/version'.format(country))
    if not manifest.get('complete'):
        raise SnapshotError('{}: cache is partial'.format(country))
    if check_source and not same_snapshot(fingerprint(source_path(source_root, country)),
                                          manifest['fingerprint']):
        raise SnapshotError('{}: cache was built from a different source snapshot'.format(country))
    return manifest
