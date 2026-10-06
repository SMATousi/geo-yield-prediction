"""Dense raw Sentinel-2 series cache (spec/yieldsat-dense-series.md, DS-01).

From ``Raw.zip`` (every S2 acquisition of every field season, per-pixel SCL,
daily weather) build ``<artifact_root>/cache_dense/<Country>/`` with the same
rows and row order as ``cache/<Country>/`` but 72 ten-day slots:

  slot k  = month k // 3 (from January of harvest year - 1), third k % 3
            (days 1-10, 11-20, 21-end); slots outside [seeding, harvest] stay NaN
  S2      = per cell, the clear acquisition (SCL 4/5/6) closest to the slot
            centre; raw DN, identical to the release's values (verified)
  weather = sum of the daily values over the slot's days inside the season
            (the release's interval-sum operator at slot resolution)
  times   = date of the selected acquisition; slot centre (clamped to the
            season) when the slot has weather but no clear S2

temporal.npy is float16 (N, 72, 16) in the canonical channel order, times.npy
float32 (N, 72). static.npy and flags.npy are links to the monthly cache;
field_stats.npz is recomputed from the dense values; cache_manifest.json is
the monthly manifest plus the dense description. Inputs only: no labels, no
split.

    python yieldsat_build_dense.py --raw_zip /data/YieldSAT/Raw.zip \\
        --artifact_root /data/YieldSAT/yieldsat_artifacts --workers 16
"""
import argparse
import calendar
import datetime
import io
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np

# GDAL's raster block cache defaults to 5% of the host's RAM per process; with one GDAL per worker
# that alone exceeded the pod's memory on large nodes (2026-10-06). Every raster is read once.
os.environ.setdefault('GDAL_CACHEMAX', '64')

N_SLOTS = 72
CLEAR_SCL = (4, 5, 6)
EPOCH = datetime.date(1970, 1, 1)
# canonical weather channels (dataset/yieldsat_schema.py WEATHER) <- raw CSV columns
WEATHER_COLS = {'temp_max': 'Temp_max', 'temp_mean': 'Temp_mean', 'temp_min': 'Temp_min', 'total_prec': 'Total_prec'}


def day(d):
    return (d - EPOCH).days


def slot_bounds(harvest_year):
    """(start_day, end_day) inclusive, days since 1970, for the 72 slots."""
    out = []
    for k in range(N_SLOTS):
        y = harvest_year - 1 + k // 36
        m = (k % 36) // 3 + 1
        third = k % 3
        last = calendar.monthrange(y, m)[1]
        a, b = (1, 10) if third == 0 else (11, 20) if third == 1 else (21, last)
        out.append((day(datetime.date(y, m, a)), day(datetime.date(y, m, b))))
    return out


def _read_band_values(raw_zip, info, rows, cols, n_bands):
    from rasterio.io import MemoryFile
    with MemoryFile(_member(raw_zip, info)) as m, m.open() as d:
        a = d.read()
    if a.shape[0] != n_bands:
        raise ValueError('{}: {} bands, expected {}'.format(info, a.shape[0], n_bands))
    inside = (rows >= 0) & (cols >= 0) & (rows < a.shape[1]) & (cols < a.shape[2])
    out = np.zeros((len(rows), n_bands), dtype=a.dtype)
    out[inside] = a[:, rows[inside], cols[inside]].T
    return out, inside


_FD = {}


def _read_raster_pair(raw_zip, s2_info, scl_info, rows, cols):
    """S2 bands and SCL at the cells, one retry on a read error."""
    for attempt in (0, 1):
        try:
            v, inside = _read_band_values(raw_zip, s2_info, rows, cols, 12)
            c, _ = _read_band_values(raw_zip, scl_info, rows, cols, 1)
            return (v, c), inside
        except Exception:
            _FD.clear()                         # reopen the file on retry
            if attempt:
                raise


def _member(raw_zip, info):
    """Bytes of one zip member from (header_offset, compress_type, compress_size): a pread of the
    local header and data, inflated when deflated. Workers never parse the 266k-entry central
    directory (a forked or re-opened ZipFile cost ~1.9 GB per worker, 2026-10-06)."""
    import struct
    import zlib
    off, ctype, csize = info
    fd = _FD.get(raw_zip)
    if fd is None:
        _FD.clear()
        fd = _FD[raw_zip] = os.open(raw_zip, os.O_RDONLY)
    head = os.pread(fd, 30, off)
    if head[:4] != b'PK\x03\x04':
        raise ValueError('bad local header at {}'.format(off))
    n_name, n_extra = struct.unpack('<HH', head[26:30])
    start = off + 30 + n_name + n_extra
    chunks, got = [], 0
    while got < csize:                      # pread may return fewer bytes than asked
        c = os.pread(fd, csize - got, start + got)
        if not c:
            raise IOError('short read at {} ({} of {} bytes)'.format(start, got, csize))
        chunks.append(c)
        got += len(c)
    data = b''.join(chunks)
    if ctype == 0:
        return data
    if ctype == 8:
        return zlib.decompressobj(-15).decompress(data)
    raise ValueError('unsupported compression {}'.format(ctype))


def _npy_header_len(path):
    with open(path, 'rb') as f:
        np.lib.format.read_magic(f)
        np.lib.format.read_array_header_1_0(f)
        return f.tell()


def _read_rows(path, a, b):
    """Rows [a, b) of a C-order .npy file via pread, pages dropped afterwards (no memory map)."""
    with open(path, 'rb') as f:
        version = np.lib.format.read_magic(f)
        shape, _, dtype = np.lib.format._read_array_header(f, version)
        head = f.tell()
    row = int(np.prod(shape[1:])) * dtype.itemsize
    fd = os.open(path, os.O_RDONLY)
    try:
        buf = os.pread(fd, (b - a) * row, head + a * row)
        os.posix_fadvise(fd, head + a * row, (b - a) * row, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return np.frombuffer(buf, dtype=dtype).reshape((b - a,) + tuple(shape[1:]))


def _write_rows(path, row_start, block):
    """Write rows [row_start, row_start + len(block)) of a C-order .npy file with pwrite, then flush
    and drop those pages from the page cache: no long-lived memory map, so a pod's memory (which
    counts file cache) stays at one field's arrays per worker (2026-10-06: memmap writes reached
    26 GiB PSS for Argentina and were OOM-killed on the cluster)."""
    data = np.ascontiguousarray(block).tobytes()
    off = _npy_header_len(path) + row_start * (len(data) // max(1, len(block)))
    fd = os.open(path, os.O_WRONLY)
    try:
        os.pwrite(fd, data, off)
        os.fdatasync(fd)
        os.posix_fadvise(fd, off, len(data), os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def build_field(task):
    """Fill one field season's rows of the dense memmaps. Returns statistics."""
    (raw_zip, out_dir, members, row_start, row_end, grid_row, grid_col, seeding, harvest, year,
     temporal_channels) = task
    n = row_end - row_start
    temporal = np.full((n, N_SLOTS, len(temporal_channels)), np.nan, dtype=np.float32)
    times = np.full((n, N_SLOTS), np.nan, dtype=np.float32)
    bounds = slot_bounds(year)
    s2 = {int(day(datetime.datetime.strptime(m.rsplit('_', 1)[1][:8], '%Y%m%d').date())): m
          for m in members if '/s2_images/' in m}
    scl = {int(day(datetime.datetime.strptime(m.rsplit('_', 1)[1][:8], '%Y%m%d').date())): m
           for m in members if '/scl_masks/' in m}
    dates = sorted(d for d in s2 if d in scl and seeding <= d <= harvest)
    rows = np.asarray(grid_row, dtype=np.int64)
    cols = np.asarray(grid_col, dtype=np.int64)
    # daily weather
    wcsv = [m for m in members if '/weather/' in m and m.endswith('.csv')]
    weather = {}
    if wcsv:
        lines = _member(raw_zip, members[wcsv[0]]).decode().strip().split('\n')
        head = lines[0].split(',')
        idx = {c: head.index(v) for c, v in WEATHER_COLS.items() if v in head}
        for line in lines[1:]:
            p = line.split(',')
            d = day(datetime.date.fromisoformat(p[0]))
            weather[d] = [float(p[idx[c]]) if c in idx and p[idx[c]] != '' else np.nan for c in WEATHER_COLS]
    w_pos = [temporal_channels.index(c) for c in WEATHER_COLS]
    s2_pos = list(range(12))
    stats = {'acquisitions': len(dates), 'clear_cells': 0, 'season_slots': 0, 'unreadable': 0}
    for k, (a, b) in enumerate(bounds):
        lo, hi = max(a, seeding), min(b, harvest)
        if lo > hi:
            continue
        stats['season_slots'] += 1
        days = [d for d in range(lo, hi + 1) if d in weather]
        if days:
            temporal[:, k, w_pos] = np.nansum(np.array([weather[d] for d in days]), axis=0)
        centre = (a + b) / 2.0
        cand = sorted((d for d in dates if a <= d <= b), key=lambda d: (abs(d - centre), d))
        todo = np.ones(n, dtype=bool)
        cache = {}                      # a date belongs to one slot only: nothing to keep across slots
        for d in cand:
            if d not in cache:
                try:
                    v, inside = _read_raster_pair(raw_zip, members[s2[d]], members[scl[d]], rows, cols)
                except Exception:                       # unreadable acquisition: skipped and counted
                    stats['unreadable'] += 1
                    continue
                c = v[1]
                v = v[0]
                ok = inside & np.isin(c[:, 0], CLEAR_SCL) & (v > 0).all(1)
                cache[d] = (v, ok)
            v, ok = cache[d]
            take = todo & ok
            if take.any():
                temporal[take, k, :12] = v[take][:, s2_pos]
                times[take, k] = d
                todo &= ~take
            if not todo.any():
                break
        stats['clear_cells'] += int((~todo).sum())
        if days:
            times[todo, k] = min(max(round(centre), lo), hi)
    # saturated S2 DN above float16's range (~1e-4 of B01 values in Argentina) are not observations
    s2v = temporal[:, :, :12]
    stats['saturated'] = int((s2v > 65504).sum())
    s2v[s2v > 65504] = np.nan
    _write_rows(Path(out_dir) / 'temporal.npy', row_start, temporal.astype(np.float16))
    _write_rows(Path(out_dir) / 'times.npy', row_start, times)
    stats['cells'] = n
    return stats


def dense_field_stats(out, mono, fields, block_rows=65536):
    """field_stats.npz for the dense cache: temporal sums from the dense values (every
    in-season slot is a full interval, so first-slot weather is valid); static sums copied from
    the monthly cache (same static rows)."""
    from dataset.yieldsat_cache import _Accumulator, temporal_valid_mask
    acc = None
    for slot, fl in enumerate(fields):
        for s in range(fl['row_start'], fl['row_end'], block_rows):
            e = min(fl['row_end'], s + block_rows)
            t = _read_rows(out / 'temporal.npy', s, e).astype(np.float32)
            if acc is None:
                acc = _Accumulator(len(fields), t.shape[2])
            valid = temporal_valid_mask(t, _read_rows(out / 'times.npy', s, e), mask_first_weather=False)
            acc.add(slot, np.where(valid, t, np.nan).reshape(-1, t.shape[2]))
    mono_stats = np.load(mono / 'field_stats.npz')
    np.savez(out / 'field_stats.npz', field_codes=np.array([fl['field_code'] for fl in fields]),
             temporal_count=acc.count, temporal_sum=acc.sum, temporal_sumsq=acc.sumsq,
             temporal_min=acc.min, temporal_max=acc.max,
             **{k: mono_stats[k] for k in mono_stats.files if k.startswith('static_')})


def build_country(raw_zip, artifact_root, country, workers, limit=None, out_name='cache_dense'):
    from dataset.yieldsat_schema import TEMPORAL_CHANNELS
    root = Path(artifact_root)
    fields = json.loads((root / 'index' / country / 'fields.json').read_text())
    with np.load(root / 'index' / country / 'rows.npz') as z:     # an NpzFile re-reads a member on every access
        grid_row, grid_col = z['grid_row'], z['grid_col']
    mono = root / 'cache' / country
    n = np.load(mono / 'times.npy', mmap_mode='r').shape[0]
    out = root / out_name / country
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'cache_manifest.json'
    if manifest_path.exists():
        manifest_path.unlink()
    t0 = time.time()
    folders = {}
    with zipfile.ZipFile(raw_zip) as zf:
        for zi in zf.infolist():
            parts = zi.filename.split('/')
            if len(parts) >= 3 and parts[0] == country:
                folders.setdefault(parts[1], {})[zi.filename] = (zi.header_offset, zi.compress_type,
                                                                 zi.compress_size)
    print('{}: {} raw folders, {} index seasons, listing {:.0f} s'.format(
        country, len(folders), len(fields), time.time() - t0), flush=True)
    # NaN-filled outputs written in chunks (no full-size memory map)
    for name, dtype, shape in (('temporal.npy', np.float16, (n, N_SLOTS, len(TEMPORAL_CHANNELS))),
                               ('times.npy', np.float32, (n, N_SLOTS))):
        mm = np.lib.format.open_memmap(out / name, 'w+', dtype, shape)
        del mm
        step = 65536
        for s in range(0, n, step):
            e = min(n, s + step)
            _write_rows(out / name, s, np.full((e - s,) + shape[1:], np.nan, dtype=dtype))
    todo = fields[:limit] if limit else fields
    missing = [f['field_shared_name'] for f in todo if f['field_shared_name'] not in folders]
    tasks = [(raw_zip, str(out), folders[f['field_shared_name']], f['row_start'], f['row_end'],
              grid_row[f['row_start']:f['row_end']].copy(), grid_col[f['row_start']:f['row_end']].copy(),
              int(f['seeding_day']), int(f['harvest_day']), int(f['year']), list(TEMPORAL_CHANNELS))
             for f in todo if f['field_shared_name'] in folders]
    agg = {'acquisitions': 0, 'clear_cells': 0, 'season_slots': 0, 'cells': 0, 'fields': 0, 'saturated': 0,
           'unreadable': 0}
    import multiprocessing
    with multiprocessing.get_context('spawn').Pool(workers) as pool:      # workers do not inherit the parent heap
        for i, st in enumerate(pool.imap_unordered(build_field, tasks, chunksize=1)):
            for k in ('acquisitions', 'clear_cells', 'season_slots', 'cells', 'saturated', 'unreadable'):
                agg[k] += st[k]
            agg['fields'] += 1
            if (i + 1) % 100 == 0:
                print('{}: {}/{} fields, {:.0f} s'.format(country, i + 1, len(tasks), time.time() - t0), flush=True)
    # static / flags shared with the monthly cache (same rows)
    for name in ('static.npy', 'flags.npy'):
        link = out / name
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(os.path.relpath(mono / name, out), link)
    dense_field_stats(out, mono, fields)
    manifest = json.loads((mono / 'cache_manifest.json').read_text())
    manifest['dense'] = {
        'n_slots': N_SLOTS, 'slot_rule': '3 per month (1-10, 11-20, 21-end) from January of harvest year - 1',
        'clear_scl': list(CLEAR_SCL), 'selection': 'clear acquisition closest to the slot centre',
        'weather': 'sum of daily values over the slot days inside [seeding, harvest]',
        'source': str(raw_zip), 'dtype': 'float16', 'fields_built': agg['fields'],
        'fields_missing_raw': missing, 'limit': limit,
        'acquisitions_in_season': agg['acquisitions'], 'cells': agg['cells'],
        'clear_cell_slots': agg['clear_cells'], 'season_slots_total': agg['season_slots'],
        'saturated_s2_values_dropped': agg['saturated'], 'unreadable_acquisitions_skipped': agg['unreadable'],
        'seconds': round(time.time() - t0, 1)}
    manifest['complete'] = manifest.get('complete', True) and not limit and not missing
    tmp = out / 'cache_manifest.json.tmp'
    tmp.write_text(json.dumps(manifest, indent=1))
    os.replace(tmp, manifest_path)
    return manifest['dense']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw_zip', required=True)
    p.add_argument('--artifact_root', required=True)
    p.add_argument('--countries', nargs='+', default=['Argentina', 'Brazil', 'Germany', 'Uruguay'])
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=0, help='first N field seasons only (tests)')
    p.add_argument('--out_name', default='cache_dense')
    a = p.parse_args()
    for c in a.countries:
        d = build_country(a.raw_zip, a.artifact_root, c, a.workers, a.limit or None, a.out_name)
        print(json.dumps({'country': c, **{k: v for k, v in d.items() if k != 'fields_missing_raw'},
                          'fields_missing_raw': len(d['fields_missing_raw'])}), flush=True)
    print('COMPLETE')


if __name__ == '__main__':
    main()
