"""Build non-overlapping 64x64 YieldSAT image tensors from prepared point caches.

See spec/yieldsat-image-training.md. The source NetCDF and prepared artifacts are
read-only; only --output-root is written.
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from dataset.yieldsat_cache import load_cache_manifest, temporal_valid_mask
from dataset.yieldsat_schema import COUNTRIES, STATIC_CHANNELS, TEMPORAL_CHANNELS
from dataset.yieldsat_source import fingerprint, load_country_index, same_snapshot, source_path

TILE = 64
MIN_VALID = TILE * TILE // 2
VERSION = 1
GEOMETRY_FILE = 'fields_geometry_Argentina-Brazil-Germany-Uruguay.json'


def plan_field_windows(rows, cols, target, width, height, tile=TILE, min_valid=MIN_VALID):
    """Return (row0, col0, source-relative row positions) for eligible windows."""
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    target = np.asarray(target)
    if not (len(rows) == len(cols) == len(target)):
        raise ValueError('row, col and target lengths differ')
    if ((rows < 0) | (rows >= height) | (cols < 0) | (cols >= width)).any():
        raise ValueError('source row/col outside field grid')
    if np.unique(rows.astype(np.int64) * width + cols).size != len(rows):
        raise ValueError('duplicate grid point within a field-season')
    tiles_w = (width + tile - 1) // tile
    tile_id = (rows // tile) * tiles_w + cols // tile
    order = np.argsort(tile_id, kind='stable')
    ids = tile_id[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1]
    ends = np.r_[starts[1:], len(order)]
    for start, end in zip(starts, ends):
        if start == end:
            continue
        selected = order[start:end]
        if np.isfinite(target[selected]).sum() >= min_valid:
            tid = int(ids[start])
            yield (tid // tiles_w) * tile, (tid % tiles_w) * tile, selected


def shift_affine(affine, row0, col0):
    a, b, c, d, e, f = affine
    return [a, b, c + a * col0 + b * row0, d, e, f + d * col0 + e * row0]


def load_geometry(path):
    records = json.loads(Path(path).read_text())
    return {(r['country'], r['field_shared_name']): r for r in records}


def make_datasets(h5, n, min_valid=MIN_VALID):
    shape = (n, TILE, TILE)
    opts = dict(compression='lzf', shuffle=True, maxshape=(None, TILE, TILE))
    h5.create_dataset('target', shape=shape, chunks=(1, TILE, TILE), dtype='f4', **opts)
    h5.create_dataset('valid_pixel', shape=shape, chunks=(1, TILE, TILE), dtype='bool', **opts)
    h5.create_dataset('source_row', shape=shape, chunks=(1, TILE, TILE), dtype='i4', **opts)
    for name, tail, dtype in [('temporal', (24, len(TEMPORAL_CHANNELS)), 'f4'),
                              ('static', (len(STATIC_CHANNELS),), 'f4'),
                              ('times', (24,), 'f4')]:
        h5.create_dataset(name, shape=shape + tail, maxshape=(None,) + shape[1:] + tail,
                          chunks=(1, TILE, TILE) + tail, dtype=dtype,
                          compression='lzf', shuffle=True)
    h5.attrs['schema_version'] = VERSION
    h5.attrs['tile_size'] = TILE
    h5.attrs['min_valid_pixels'] = min_valid
    h5.attrs['temporal_channels_json'] = json.dumps(TEMPORAL_CHANNELS)
    h5.attrs['static_channels_json'] = json.dumps(STATIC_CHANNELS)
    h5.attrs['time_units'] = 'days since 1970-01-01'
    h5.attrs['target_units'] = 't/ha'


def build_country(country, source_root, artifact_root, output_root, geom, min_valid=MIN_VALID):
    index = load_country_index(artifact_root, source_root, country)
    cache_manifest = load_cache_manifest(artifact_root, source_root, country)
    before = fingerprint(source_path(source_root, country))
    if not same_snapshot(before, cache_manifest['fingerprint']):
        raise RuntimeError('{} cache/source snapshot mismatch'.format(country))
    if not index['manifest']['audit']['uuid_unique'] or not index['manifest']['audit']['fields_contiguous']:
        raise ValueError('{} index has duplicate UUIDs or noncontiguous field seasons'.format(country))
    if index['manifest']['audit']['duplicate_field_row_col']:
        raise ValueError('{} has duplicate field-season grid points'.format(country))
    rows = index['rows']
    cache_dir = Path(artifact_root) / 'cache' / country
    temporal = np.load(cache_dir / 'temporal.npy', mmap_mode='r')
    static = np.load(cache_dir / 'static.npy', mmap_mode='r')
    times = np.load(cache_dir / 'times.npy', mmap_mode='r')
    if not (len(temporal) == len(static) == len(times) == len(rows['target'])):
        raise ValueError('{} cache/index row-count mismatch'.format(country))
    plans = []
    excluded_windows = 0
    for field in index['fields']:
        g = geom.get((country, field['field_shared_name']))
        if g is None:
            raise ValueError('{} has no verified geometry'.format(field['field_shared_name']))
        if (g['year'], g['crop']) != (field['year'], field['crop']):
            raise ValueError('{} geometry year/crop mismatch'.format(field['field_shared_name']))
        start, end = field['row_start'], field['row_end']
        field_rows = rows['grid_row'][start:end]
        field_cols = rows['grid_col'][start:end]
        accepted = list(plan_field_windows(field_rows, field_cols,
                                           rows['target'][start:end], g['width'], g['height'],
                                           min_valid=min_valid))
        excluded_windows += ((g['width'] + 63) // 64) * ((g['height'] + 63) // 64) - len(accepted)
        for row0, col0, relative in accepted:
            plans.append((field, g, row0, col0, start + relative))
    output_root = Path(output_root)
    country_dir = output_root / country
    country_dir.mkdir(parents=True, exist_ok=True)
    h5_final = country_dir / 'images.h5'
    json_final = country_dir / 'patches.jsonl'
    h5_tmp = country_dir / 'images.h5.partial'
    json_tmp = country_dir / 'patches.jsonl.partial'
    records = []
    with h5py.File(h5_tmp, 'w') as out, json_tmp.open('w') as meta:
        make_datasets(out, len(plans), min_valid)
        written = 0
        for field, g, row0, col0, ids in plans:
            local_r = rows['grid_row'][ids] - row0
            local_c = rows['grid_col'][ids] - col0
            t = np.asarray(temporal[ids])
            s = np.asarray(static[ids])
            tm = np.asarray(times[ids])
            y = rows['target'][ids]
            sensor_valid = (np.isfinite(s[:, :-3]).any(axis=1)
                            | temporal_valid_mask(t, tm).any(axis=(1, 2)))
            valid_rows = np.isfinite(y) & sensor_valid
            if int(valid_rows.sum()) < min_valid:
                continue
            dense_t = np.full((TILE, TILE, 24, len(TEMPORAL_CHANNELS)), np.nan, np.float32)
            dense_s = np.full((TILE, TILE, len(STATIC_CHANNELS)), np.nan, np.float32)
            dense_tm = np.full((TILE, TILE, 24), np.nan, np.float32)
            dense_y = np.full((TILE, TILE), np.nan, np.float32)
            dense_valid = np.zeros((TILE, TILE), dtype=np.bool_)
            source_map = np.full((TILE, TILE), -1, np.int32)
            dense_t[local_r, local_c] = t
            dense_s[local_r, local_c] = s
            dense_tm[local_r, local_c] = tm
            dense_y[local_r, local_c] = y
            dense_valid[local_r, local_c] = valid_rows
            source_map[local_r, local_c] = ids
            for name, value in [('temporal', dense_t), ('static', dense_s),
                                ('times', dense_tm), ('target', dense_y),
                                ('valid_pixel', dense_valid), ('source_row', source_map)]:
                out[name][written] = value
            rec = {'patch_index': written, 'country': country,
                   'field_shared_name': field['field_shared_name'],
                   'physical_field_id': g['physical_field_id'],
                   'crop': field['crop'], 'year': field['year'],
                   'row0': row0, 'col0': col0,
                   'actual_height': min(TILE, g['height'] - row0),
                   'actual_width': min(TILE, g['width'] - col0),
                   'occupied_pixels': int(len(ids)), 'valid_pixels': int(valid_rows.sum()),
                   'source_row_min': int(ids.min()), 'source_row_max': int(ids.max()),
                   'field_row_start': field['row_start'], 'field_row_end': field['row_end'],
                   'epsg': g['epsg'], 'transform': shift_affine(g['transform'], row0, col0),
                   'source_head_tail_sha256': before['head_tail_sha256'],
                   'cache_version': cache_manifest['cache_version']}
            meta.write(json.dumps(rec, separators=(',', ':')) + '\n')
            records.append(rec)
            written += 1
            if written % 100 == 0:
                print(country, 'wrote', written, 'of', len(plans), flush=True)
        if written != len(plans):
            for ds in out.values():
                ds.resize((written,) + ds.shape[1:])
        out.attrs['country'] = country
        out.attrs['n_patches'] = written
        out.flush()
    after = fingerprint(source_path(source_root, country))
    if not same_snapshot(before, after):
        raise RuntimeError('{} source changed during build'.format(country))
    verify_country(h5_tmp, records, rows, temporal, static, times, min_valid)
    os.replace(h5_tmp, h5_final)
    os.replace(json_tmp, json_final)
    report = {'country': country, 'patches': len(records),
              'candidate_windows': len(plans), 'excluded_windows': excluded_windows,
              'valid_pixels': sum(r['valid_pixels'] for r in records),
              'source_fingerprint': before, 'cache_version': cache_manifest['cache_version']}
    print(json.dumps(report), flush=True)
    return report


def verify_country(path, records, rows, temporal, static, times, min_valid=MIN_VALID):
    """Read every mask/row map and bounded full-tensor probes before publication."""
    seen = set()
    with h5py.File(path, 'r') as f:
        if len(f['target']) != len(records):
            raise ValueError('stored patch count mismatch')
        for i, rec in enumerate(records):
            src = f['source_row'][i]
            valid = f['valid_pixel'][i]
            present = src >= 0
            if (int(valid.sum()) != rec['valid_pixels'] or valid.sum() < min_valid
                    or int(present.sum()) != rec['occupied_pixels'] or np.any(valid & ~present)):
                raise ValueError('patch validity/count mismatch: {}'.format(i))
            used = src[present]
            if np.unique(used).size != used.size:
                raise ValueError('duplicate rows within patch')
            if np.any((used < rec['field_row_start']) | (used >= rec['field_row_end'])):
                raise ValueError('cross-field rows in patch')
            rr, cc = np.where(present)
            if not (np.array_equal(rows['grid_row'][used] - rec['row0'], rr)
                    and np.array_equal(rows['grid_col'][used] - rec['col0'], cc)):
                raise ValueError('row/col mismatch')
            key = (rec['field_shared_name'], rec['row0'], rec['col0'])
            if key in seen:
                raise ValueError('overlapping/duplicate tile')
            seen.add(key)
            if i in (0, len(records) // 2, len(records) - 1):
                if not np.array_equal(f['target'][i][present], rows['target'][used], equal_nan=True):
                    raise ValueError('target readback mismatch')
                if not np.array_equal(f['temporal'][i][present], temporal[used], equal_nan=True):
                    raise ValueError('temporal readback mismatch')
                if not np.array_equal(f['static'][i][present], static[used], equal_nan=True):
                    raise ValueError('static readback mismatch')
                if not np.array_equal(f['times'][i][present], times[used], equal_nan=True):
                    raise ValueError('times readback mismatch')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', required=True)
    p.add_argument('--artifact-root', required=True)
    p.add_argument('--output-root', default='/home1/pupil/SMATousi/YieldSAT-Image')
    p.add_argument('--countries', nargs='+', default=list(COUNTRIES), choices=COUNTRIES)
    p.add_argument('--min-valid', type=int, default=MIN_VALID,
                   help='minimum valid cells per 64x64 window (1 = every cell with a target is '
                        'covered; the grid is fixed, so smaller values give a superset)')
    args = p.parse_args()
    root = Path(args.output_root)
    if (root / 'manifest.json').exists():
        raise SystemExit('completed dataset already exists; choose a new output root')
    geometry = load_geometry(Path(args.artifact_root) / 'geometry' / GEOMETRY_FILE)
    root.mkdir(parents=True, exist_ok=True)
    reports = [build_country(c, args.source_root, args.artifact_root, root, geometry, args.min_valid)
               for c in args.countries]
    manifest = {'schema_version': VERSION, 'tile_size': TILE,
                'min_valid_pixels': args.min_valid, 'countries': args.countries,
                'temporal_channels': TEMPORAL_CHANNELS,
                'static_channels': STATIC_CHANNELS,
                'time_units': 'days since 1970-01-01', 'target_units': 't/ha',
                'reports': reports, 'total_patches': sum(r['patches'] for r in reports),
                'total_valid_pixels': sum(r['valid_pixels'] for r in reports)}
    temp = root / 'manifest.json.partial'
    temp.write_text(json.dumps(manifest, indent=2))
    os.replace(temp, root / 'manifest.json')
    print('COMPLETE', manifest['total_patches'], 'patches', flush=True)


if __name__ == '__main__':
    main()
