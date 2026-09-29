# --------------------------------------------------------
# YieldSAT source snapshot checks and chunked row index (YS-01, YS-03).
#
# Source NetCDF files are opened read-only with h5py (HDF5 backend; no
# xarray/netCDF4 dependency). Nothing here evaluates ``sample[:]``: the index
# reads only the 1-D per-row variables. Every artifact written under the
# artifact root records the source fingerprint it was built from, and loading
# an artifact whose fingerprint no longer matches the source fails (stale
# cache rejection).
# --------------------------------------------------------

import hashlib
import json
import time
import zlib
from pathlib import Path

import numpy as np

from dataset.yieldsat_schema import (
    ALL_CHANNELS,
    CONTRACT_KEY,
    COUNTRIES,
    EXPECTED_SNAPSHOT,
    NUM_TIME_SLOTS,
    SOURCE_FILENAME,
    decode_codes,
    integer_lookup,
    origin_offset_days,
    parse_field_shared_name,
    parse_time_origin,
    read_band_names,
    read_dictionary,
    resolve_channel_indices,
    date_strings_to_days,
)

HDF5_SIGNATURE = b'\x89HDF\r\n\x1a\n'
INDEX_VERSION = 1


class SnapshotError(RuntimeError):
    pass


def source_path(root, country):
    if country not in COUNTRIES:
        raise ValueError('unknown YieldSAT country: {}'.format(country))
    return Path(root) / country / SOURCE_FILENAME


def open_source(path):
    import h5py
    return h5py.File(str(path), 'r')


def fingerprint(path, probe_bytes=4 << 20):
    """Cheap snapshot fingerprint: size, mtime and hashes of the head/tail."""
    path = Path(path)
    st = path.stat()
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read(probe_bytes))
        f.seek(max(0, st.st_size - probe_bytes))
        h.update(f.read(probe_bytes))
    return {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns, 'head_tail_sha256': h.hexdigest()}


def compute_crc32(path, block=64 << 20):
    crc = 0
    with open(path, 'rb') as f:
        while True:
            b = f.read(block)
            if not b:
                break
            crc = zlib.crc32(b, crc)
    return '{:08x}'.format(crc)


def check_snapshot(root, countries, crc_evidence_dir=None, full_crc=False,
                   stability_wait=2.0):
    """Verify that each selected source file is a complete, stable snapshot.

    Checks: file present, HDF5 signature, byte size equal to the recorded
    snapshot, readable schema with the expected row count and all 120
    channels, unchanged size/mtime across ``stability_wait`` seconds, and a
    whole-file CRC32 equal to the recorded archive CRC (either computed now
    with ``full_crc`` or read from an evidence JSON produced earlier for the
    same size/mtime). Raises SnapshotError listing every failure.
    """
    report, failures = {}, []
    for country in countries:
        path = source_path(root, country)
        entry = {'path': str(path)}
        report[country] = entry
        if not path.exists():
            failures.append('{}: missing {}'.format(country, path))
            continue
        with open(path, 'rb') as f:
            entry['hdf5_signature'] = f.read(8) == HDF5_SIGNATURE
        fp1 = fingerprint(path)
        time.sleep(stability_wait)
        fp2 = fingerprint(path)
        entry['fingerprint'] = fp2
        entry['stable'] = fp1 == fp2
        expected = EXPECTED_SNAPSHOT[country]
        entry['size_matches_record'] = fp2['bytes'] == expected['bytes']
        with open_source(path) as f:
            rows, slots, bands = f['sample'].shape
            entry['shape'] = [rows, slots, bands]
            entry['schema_ok'] = (rows == expected['rows'] and slots == NUM_TIME_SLOTS
                                  and bands == len(ALL_CHANNELS))
            resolve_channel_indices(read_band_names(f), ALL_CHANNELS)
            probe = f['sample'][rows - 1]
            entry['last_row_readable'] = probe.shape == (NUM_TIME_SLOTS, len(ALL_CHANNELS))
        crc = None
        if full_crc:
            crc = {'crc32': compute_crc32(path), 'mtime_ns': fp2['mtime_ns'],
                   'bytes': fp2['bytes'], 'source': 'computed'}
        elif crc_evidence_dir is not None:
            ev = Path(crc_evidence_dir) / 'crc32_{}.json'.format(country)
            if ev.exists():
                data = json.loads(ev.read_text())
                if (data.get('bytes') == fp2['bytes']
                        and int(round(data.get('mtime_after', -1))) == fp2['mtime_ns'] // 10**9):
                    crc = {'crc32': data['crc32'], 'bytes': data['bytes'],
                           'source': str(ev), 'finished': data.get('finished')}
        entry['crc32'] = crc
        entry['crc32_matches_record'] = crc is not None and crc['crc32'] == expected['crc32']
        for key in ('hdf5_signature', 'stable', 'size_matches_record', 'schema_ok',
                    'last_row_readable', 'crc32_matches_record'):
            if not entry.get(key):
                failures.append('{}: {} failed'.format(country, key))
    if failures:
        raise SnapshotError('; '.join(failures))
    return report


# ---- chunked row index -----------------------------------------------------

def _read_1d(ds, chunk=1 << 20):
    n = ds.shape[0]
    out = np.empty(n, dtype=ds.dtype)
    for s in range(0, n, chunk):
        out[s:s + chunk] = ds[s:s + chunk]
    return out


def build_country_index(root, country, out_dir):
    """Build the per-row index and per-field table for one country.

    Writes ``rows.npz`` (decoded per-row metadata), ``uuids.npy``,
    ``fields.json`` (field-season table with contiguous row ranges and
    country-qualified identities), ``target_side.json`` (field-level ground
    truth attributes: diagnostics only) and ``index_manifest.json`` (audit
    results and source fingerprint).
    """
    path = source_path(root, country)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(path)
    t0 = time.time()
    with open_source(path) as f:
        n = f['target'].shape[0]
        dicts = {k: read_dictionary(f[k]) for k in (
            'field_shared_name', 'farm_identifier', 'country', 'crop', 'year',
            'seeding_date', 'harvesting_date', 'seeding_date_type', 'row', 'col')}
        raw = {k: _read_1d(f[k]) for k in (
            'field_shared_name', 'farm_identifier', 'country', 'crop', 'year',
            'seeding_date', 'harvesting_date', 'seeding_date_type', 'row', 'col', 'target')}
        uuids = _read_1d(f['index'])
        times_attrs = dict(f['times'].attrs)
        origin = parse_time_origin(times_attrs['units'], times_attrs['calendar'])
        band_names = read_band_names(f)
        target_side = {k: (v.item() if isinstance(v, np.generic) else
                           v.decode() if isinstance(v, bytes) else v)
                       for k, v in f.attrs.items() if k.endswith('_yield_ground_truth')}

    # decode every categorical; unknown codes raise
    for k in ('field_shared_name', 'farm_identifier', 'country', 'crop', 'year',
              'seeding_date', 'harvesting_date', 'seeding_date_type'):
        decode_codes(np.unique(raw[k]), dicts[k], k)
    countries_in_file = set(dicts['country'][c] for c in np.unique(raw['country']))
    if countries_in_file != {country}:
        raise ValueError('{}: file declares countries {}'.format(country, countries_in_file))
    grid_row = integer_lookup(dicts['row'])[raw['row']]
    grid_col = integer_lookup(dicts['col'])[raw['col']]
    if (grid_row < 0).any() or (grid_col < 0).any():
        raise ValueError('{}: undecodable row/col codes'.format(country))
    def day_table(mapping):
        # dictionary codes are validated above, so unset entries are never used
        table = np.zeros(max(mapping) + 1, dtype=np.int64)
        codes = sorted(mapping)
        table[codes] = date_strings_to_days([mapping[c] for c in codes])
        return table

    seeding_day = day_table(dicts['seeding_date'])[raw['seeding_date']].astype(np.int32)
    harvest_day = day_table(dicts['harvesting_date'])[raw['harvesting_date']].astype(np.int32)

    uuid_strings = np.array([u.decode() if isinstance(u, bytes) else u for u in uuids], dtype='S36')
    target = raw['target'].astype(np.float32)
    field_code = raw['field_shared_name'].astype(np.int32)

    # contiguity and duplicates
    change = np.flatnonzero(np.diff(field_code) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n]])
    run_codes = field_code[starts]
    contiguous = len(np.unique(run_codes)) == len(run_codes)
    cell_key = (field_code.astype(np.int64) << 40) | (grid_row.astype(np.int64) << 20) | grid_col
    dup_cells = int(n - np.unique(cell_key).size)
    uuid_unique = bool(np.unique(uuid_strings).size == n)

    fields = []
    for code, s, e in zip(run_codes.tolist(), starts.tolist(), ends.tolist()):
        name = dicts['field_shared_name'][code]
        parts = parse_field_shared_name(name)
        t = target[s:e]
        seasons = {
            'seeding_days': sorted(set(seeding_day[s:e].tolist())),
            'harvest_days': sorted(set(harvest_day[s:e].tolist())),
        }
        fields.append({
            'country': country,
            'field_code': int(code),
            'field_shared_name': name,
            'provider': parts['provider'],
            'farm': parts['farm'],
            'farm_id': '{}/{}/{}'.format(country, parts['provider'], parts['farm']),
            'season_id': '{}/{}'.format(country, name),
            'crop': dicts['crop'][int(raw['crop'][s])],
            'year': int(dicts['year'][int(raw['year'][s])]),
            'row_start': int(s),
            'row_end': int(e),
            'n_rows': int(e - s),
            'grid_rows': [int(grid_row[s:e].min()), int(grid_row[s:e].max())],
            'grid_cols': [int(grid_col[s:e].min()), int(grid_col[s:e].max())],
            'target_mean': float(np.nanmean(t)),
            'target_std': float(np.nanstd(t)),
            'single_season_dates': len(seasons['seeding_days']) == 1 and len(seasons['harvest_days']) == 1,
            'seeding_day': seasons['seeding_days'][0],
            'harvest_day': seasons['harvest_days'][0],
        })
        if parts['crop'] != fields[-1]['crop'] or parts['year'] != fields[-1]['year']:
            raise ValueError('{}: name/crop/year disagreement'.format(name))

    np.savez(out_dir / 'rows.npz', field_code=field_code, grid_row=grid_row.astype(np.int32),
             grid_col=grid_col.astype(np.int32), target=target, seeding_day=seeding_day,
             harvest_day=harvest_day,
             seeding_date_type=raw['seeding_date_type'].astype(np.int16))
    np.save(out_dir / 'uuids.npy', uuid_strings)
    (out_dir / 'fields.json').write_text(json.dumps(fields, indent=1))
    unknown = sum(1 for v in target_side.values() if isinstance(v, str) and v.strip().upper() == 'UNKNOWN')
    (out_dir / 'target_side.json').write_text(json.dumps(
        {'note': 'field-level ground truth; diagnostics only, never model input',
         'values': target_side}, indent=1))

    finite = np.isfinite(target)
    manifest = {
        'contract': CONTRACT_KEY,
        'index_version': INDEX_VERSION,
        'country': country,
        'source': str(path),
        'fingerprint': fp,
        'rows': int(n),
        'band_order_hash': hashlib.sha256('|'.join(band_names).encode()).hexdigest()[:16],
        'time_origin': origin.isoformat(),
        'time_origin_offset_days': origin_offset_days(origin),
        'dictionaries': {k: {str(c): v for c, v in d.items()} for k, d in dicts.items()
                         if k not in ('row', 'col', 'field_shared_name')},
        'row_col_identity_maps': {k: all(int(v) == c for c, v in dicts[k].items())
                                  for k in ('row', 'col')},
        'audit': {
            'uuid_unique': uuid_unique,
            'fields_contiguous': bool(contiguous),
            'field_seasons': len(fields),
            'field_dictionary_entries': len(dicts['field_shared_name']),
            'duplicate_field_row_col': dup_cells,
            'target_nonfinite': int((~finite).sum()),
            'target_eq_minus_one': int((target == -1).sum()),
            'target_le_zero': int((target[finite] <= 0).sum()),
            'target_quantiles_p0_p1_p50_p99_p100': np.percentile(target[finite], [0, 1, 50, 99, 100]).round(4).tolist(),
            'target_side_attrs': len(target_side),
            'target_side_unknown': unknown,
            'fields_with_multiple_dates': sum(not fl['single_season_dates'] for fl in fields),
        },
        'seconds': round(time.time() - t0, 1),
    }
    (out_dir / 'index_manifest.json').write_text(json.dumps(manifest, indent=1))
    return manifest


def load_country_index(artifact_root, source_root, country, check_source=True):
    """Load an index, refusing it if the source snapshot has changed."""
    d = Path(artifact_root) / 'index' / country
    manifest = json.loads((d / 'index_manifest.json').read_text())
    if manifest.get('contract') != CONTRACT_KEY or manifest.get('index_version') != INDEX_VERSION:
        raise SnapshotError('{}: index built for another contract/version'.format(country))
    if check_source:
        current = fingerprint(source_path(source_root, country))
        if current != manifest['fingerprint']:
            raise SnapshotError('{}: index was built from a different source snapshot'.format(country))
    rows = dict(np.load(d / 'rows.npz'))
    fields = json.loads((d / 'fields.json').read_text())
    return {'manifest': manifest, 'rows': rows, 'fields': fields, 'dir': d}
