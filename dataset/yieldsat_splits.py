# --------------------------------------------------------
# YieldSAT grouped split manifests (YS-03, YS-R05).
#
# Splits are made over field seasons grouped by country-qualified identity
# before any row is sampled. Schemes:
#   farm    - farm-held-out within each selected country (baseline)
#   field   - physical-field-held-out (geometry-derived identity when present)
#   block   - geographic block holdout from verified WGS84 centroids
#   country - leave-one-country-out; validation farms from training countries
#   year    - leave-one-year-out; physical fields seen in the test year are
#             removed from train/validation
# Every manifest records its scheme, seed, source fingerprints and overlap
# checks; overlapping physical fields (or farms, for farm/country schemes)
# across partitions are an error.
# --------------------------------------------------------

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from dataset.yieldsat_schema import CONTRACT_KEY

PARTITIONS = ('train', 'val', 'test')


def load_field_table(artifact_root, source_root, countries, require_geometry=False,
                     check_source=True):
    """Field-season records for the selected countries with geometry joined
    (physical field id, centroid) when a geometry table is available."""
    from dataset.yieldsat_source import load_country_index

    fields, fingerprints = [], {}
    geo = {}
    geo_dir = Path(artifact_root) / 'geometry'
    if geo_dir.exists():
        for p in sorted(geo_dir.glob('fields_geometry_*.json')):
            for r in json.loads(p.read_text()):
                geo[r['field_shared_name']] = r
    for c in countries:
        idx = load_country_index(artifact_root, source_root, c, check_source=check_source)
        fingerprints[c] = idx['manifest']['fingerprint']
        for fl in idx['fields']:
            fl = dict(fl)
            g = geo.get(fl['field_shared_name'])
            if g is not None:
                fl['physical_field_id'] = g['physical_field_id']
                fl['centroid_lat'] = g['centroid_lat']
                fl['centroid_lon'] = g['centroid_lon']
                fl['geometry_verified'] = True
            else:
                if require_geometry:
                    raise ValueError('{}: no geometry; run the geometry step'.format(
                        fl['field_shared_name']))
                # without geometry each season is its own physical field: only
                # farm-grouped results are leakage-safe in that case
                fl['physical_field_id'] = fl['season_id']
                fl['geometry_verified'] = False
            fields.append(fl)
    _attach_farm_groups(fields)
    return {'fields': fields, 'fingerprints': fingerprints, 'countries': list(countries)}


def _attach_farm_groups(fields):
    """Farm clusters: farms linked through a shared physical field (the same
    ground recorded under two farm aliases) are one group, so farm-held-out
    splits cannot leak aliased ground."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    farms_by_phys = {}
    for fl in fields:
        find(fl['farm_id'])
        farms_by_phys.setdefault(fl['physical_field_id'], set()).add(fl['farm_id'])
    for farms in farms_by_phys.values():
        farms = sorted(farms)
        for f in farms[1:]:
            parent[find(f)] = find(farms[0])
    members = {}
    for fl in fields:
        members.setdefault(find(fl['farm_id']), set()).add(fl['farm_id'])
    for fl in fields:
        fl['farm_group_id'] = '+'.join(sorted(members[find(fl['farm_id'])]))


def _assign_groups(groups, rows, rng, val_frac, test_frac):
    """Assign whole groups to partitions to approach row-fraction targets.

    Groups are visited largest-first (sizes jittered by the seed so different
    seeds give different splits) and each goes to the partition with the
    largest remaining row deficit. Afterwards every partition with a positive
    target gets at least one group, taken as the smallest group of the
    partition with the most groups, when that donor would keep one.
    """
    targets = {'train': 1.0 - val_frac - test_frac, 'val': val_frac, 'test': test_frac}
    total = float(sum(rows[g] for g in groups))
    jitter = {g: rows[g] * rng.uniform(0.5, 1.5) for g in groups}
    order = sorted(groups, key=lambda g: -jitter[g])
    acc = {p: 0.0 for p in targets}
    out = {}
    for g in order:
        part = max(targets, key=lambda p: targets[p] * total - acc[p])
        out[g] = part
        acc[part] += rows[g]
    for part, frac in targets.items():
        if frac <= 0 or any(v == part for v in out.values()):
            continue
        counts = {p: sum(1 for v in out.values() if v == p) for p in targets}
        donor = max(counts, key=counts.get)
        if counts[donor] < 2:
            continue
        g = min((g for g, v in out.items() if v == donor), key=lambda g: rows[g])
        out[g] = part
    return out


def _block_id(lat, lon, block_km):
    dlat = block_km / 111.0
    dlon = block_km / (111.0 * max(0.1, math.cos(math.radians(lat))))
    return '{}:{}'.format(int(math.floor(lat / dlat)), int(math.floor(lon / dlon)))


def make_split(table, scheme, seed=0, val_frac=0.15, test_frac=0.2, holdout_country=None,
               holdout_year=None, block_km=20.0):
    fields = table['fields']
    rng = np.random.default_rng(seed)
    assign = {}

    def group_key(fl):
        if scheme in ('farm', 'country'):
            return fl['farm_group_id']
        if scheme == 'field':
            return fl['physical_field_id']
        if scheme == 'block':
            if fl.get('centroid_lat') is None:
                raise ValueError('block scheme needs verified centroids')
            return '{}/{}'.format(fl['country'], _block_id(fl['centroid_lat'], fl['centroid_lon'], block_km))
        raise ValueError(scheme)

    if scheme in ('farm', 'field', 'block'):
        for c in table['countries']:
            cf = [fl for fl in fields if fl['country'] == c]
            rows = {}
            for fl in cf:
                rows[group_key(fl)] = rows.get(group_key(fl), 0) + fl['n_rows']
            ga = _assign_groups(sorted(rows), rows, rng, val_frac, test_frac)
            for fl in cf:
                assign[fl['season_id']] = ga[group_key(fl)]
        if scheme == 'block':
            # a physical field straddling two blocks must stay whole
            _merge_physical(fields, assign)
    elif scheme == 'country':
        if holdout_country not in table['countries']:
            raise ValueError('holdout_country must be one of the selected countries')
        for c in table['countries']:
            cf = [fl for fl in fields if fl['country'] == c]
            if c == holdout_country:
                for fl in cf:
                    assign[fl['season_id']] = 'test'
                continue
            rows = {}
            for fl in cf:
                rows[fl['farm_group_id']] = rows.get(fl['farm_group_id'], 0) + fl['n_rows']
            ga = _assign_groups(sorted(rows), rows, rng, val_frac, 0.0)
            for fl in cf:
                assign[fl['season_id']] = ga[fl['farm_group_id']]
    elif scheme == 'year':
        if holdout_year is None:
            raise ValueError('year scheme needs holdout_year')
        rest = {}
        for fl in fields:
            if fl['year'] == holdout_year:
                assign[fl['season_id']] = 'test'
            else:
                rest.setdefault(fl['country'], []).append(fl)
        for c, cf in rest.items():
            rows = {}
            for fl in cf:
                rows[fl['farm_group_id']] = rows.get(fl['farm_group_id'], 0) + fl['n_rows']
            ga = _assign_groups(sorted(rows), rows, rng, val_frac, 0.0)
            for fl in cf:
                assign[fl['season_id']] = ga[fl['farm_group_id']]
        test_phys = {fl['physical_field_id'] for fl in fields if assign[fl['season_id']] == 'test'}
        for fl in fields:
            if assign[fl['season_id']] != 'test' and fl['physical_field_id'] in test_phys:
                assign[fl['season_id']] = 'excluded'
    else:
        raise ValueError('unknown split scheme: {}'.format(scheme))

    split = {'contract': CONTRACT_KEY, 'scheme': scheme, 'seed': seed,
             'val_frac': val_frac, 'test_frac': test_frac,
             'holdout_country': holdout_country, 'holdout_year': holdout_year,
             'block_km': block_km if scheme == 'block' else None,
             'countries': table['countries'], 'fingerprints': table['fingerprints'],
             'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
             'geometry_verified': all(fl['geometry_verified'] for fl in fields),
             'partitions': {p: sorted(s for s, a in assign.items() if a == p)
                            for p in PARTITIONS + ('excluded',)}}
    split['summary'] = summarize_split(split, fields)
    check_split(split, fields)
    blob = json.dumps(split['partitions'], sort_keys=True).encode()
    split['partition_hash'] = hashlib.sha256(blob).hexdigest()[:16]
    split['label'] = split_label(split)
    return split


def _merge_physical(fields, assign):
    rank = {'test': 0, 'val': 1, 'train': 2}
    by_phys = {}
    for fl in fields:
        by_phys.setdefault(fl['physical_field_id'], []).append(fl['season_id'])
    for seasons in by_phys.values():
        parts = {assign[s] for s in seasons}
        if len(parts) > 1:
            target = min(parts, key=rank.get)
            for s in seasons:
                assign[s] = target


def split_label(split):
    """How results on this split may be described (spec YS-R05)."""
    s = split['scheme']
    if s == 'farm':
        return 'farm-held-out'
    if s == 'field':
        return 'physical-field-held-out' if split['geometry_verified'] else 'field-season-held-out'
    if s == 'block':
        return 'geographic-block-held-out ({} km)'.format(split['block_km'])
    if s == 'country':
        return 'leave-one-country-out ({})'.format(split['holdout_country'])
    return 'leave-one-year-out ({})'.format(split['holdout_year'])


def summarize_split(split, fields):
    by_season = {fl['season_id']: fl for fl in fields}
    out = {}
    for p, seasons in split['partitions'].items():
        fl = [by_season[s] for s in seasons]
        per_country = {}
        for f in fl:
            d = per_country.setdefault(f['country'], {'seasons': 0, 'rows': 0, 'farms': set(), 'crops': {}})
            d['seasons'] += 1
            d['rows'] += f['n_rows']
            d['farms'].add(f['farm_group_id'])
            d['crops'][f['crop']] = d['crops'].get(f['crop'], 0) + 1
        for d in per_country.values():
            d['farms'] = len(d['farms'])
        out[p] = {'seasons': len(fl), 'rows': sum(f['n_rows'] for f in fl),
                  'per_country': per_country}
    return out


def check_split(split, fields):
    """Raise if physical fields (always) or farms (farm/country schemes)
    appear in more than one of train/val/test."""
    by_season = {fl['season_id']: fl for fl in fields}
    for key in ('physical_field_id',) + (('farm_group_id',) if split['scheme'] in ('farm', 'country') else ()):
        seen = {}
        for p in PARTITIONS:
            for s in split['partitions'][p]:
                k = by_season[s][key]
                if seen.setdefault(k, p) != p:
                    raise ValueError('split leakage: {} {} in {} and {}'.format(key, k, seen[k], p))
    assigned = sum(len(v) for v in split['partitions'].values())
    if assigned != len(fields):
        raise ValueError('split does not assign every field season exactly once')


def save_split(split, artifact_root, name):
    d = Path(artifact_root) / 'splits'
    d.mkdir(parents=True, exist_ok=True)
    path = d / '{}.json'.format(name)
    path.write_text(json.dumps(split, indent=1))
    return path


def load_split(artifact_root, name, fingerprints=None):
    split = json.loads((Path(artifact_root) / 'splits' / '{}.json'.format(name)).read_text())
    if split.get('contract') != CONTRACT_KEY:
        raise ValueError('split manifest was made for another contract')
    if fingerprints is not None:
        for c, fp in split['fingerprints'].items():
            if c in fingerprints and fingerprints[c] != fp:
                raise ValueError('split {} was built from another snapshot of {}'.format(name, c))
    return split
