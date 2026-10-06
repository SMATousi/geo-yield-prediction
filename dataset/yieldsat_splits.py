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
from dataset.yieldsat_source import same_snapshot

PARTITIONS = ('train', 'val', 'test')
PAPER_PROTOCOLS = ('cv', 'loro', 'loyo')
COUNTRY_CODES = {'Argentina': 'ARG', 'Brazil': 'BRA', 'Germany': 'GER', 'Uruguay': 'URG'}
CROP_CODES = {'corn': 'C', 'rapeseed': 'R', 'soybean': 'S', 'wheat': 'W'}
# the nine country-crop subsets reported in the YieldSAT paper
PAPER_PAIRS = ('ARG-C', 'ARG-S', 'ARG-W', 'BRA-C', 'BRA-S', 'BRA-W', 'GER-R', 'GER-W', 'URG-S')


def pair_code(country, crop):
    return '{}-{}'.format(COUNTRY_CODES[country], CROP_CODES[crop])


def parse_pair(code):
    c, k = code.split('-')
    country = {v: n for n, v in COUNTRY_CODES.items()}[c]
    crop = {v: n for n, v in CROP_CODES.items()}[k]
    return country, crop


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
                # first-level administrative unit (province/state) from Raw.zip
                # metadata; candidate LORO region (Argentina: 8 provinces for
                # soybean, matching the paper's 8 regions)
                adm = g.get('adm_units') or {}
                fl['province'] = '{}/{}'.format(fl['country'], adm.get('adm_unit0', adm.get('adm_1', 'unknown')))
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
    if s in PAPER_PROTOCOLS:
        return '{} fold {} of {} ({}, {} grouping, {} policy)'.format(
            s.upper(), split['fold'] + 1, split['n_folds'], split['pair'], split['group'],
            split['leakage_policy'])
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
    appear in more than one of train/val/test. Paper-compatible folds with
    ``leakage_policy='paper'`` check their own grouping key instead (season,
    or farm for LORO, or year for LOYO), because the paper's protocol allows
    other seasons of the same ground in training; their cross-partition
    physical overlap is reported, not refused."""
    by_season = {fl['season_id']: fl for fl in fields}
    if split['scheme'] in PAPER_PROTOCOLS:
        # the fold's grouping key (season / farm / year) separates test from
        # train+val; the validation carve-out may share it with training
        gk = split['group_key']
        test_keys = {by_season[s][gk] for s in split['partitions']['test']}
        for p in ('train', 'val'):
            clash = test_keys & {by_season[s][gk] for s in split['partitions'][p]}
            if clash:
                raise ValueError('split leakage: {} {} in test and {}'.format(gk, sorted(clash)[:3], p))
    if split.get('leakage_policy') == 'paper':
        keys = ()
    else:
        keys = ('physical_field_id',) + (('farm_group_id',) if split['scheme'] in ('farm', 'country')
                                         else ())
    for key in keys:
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
            if c in fingerprints and not same_snapshot(fingerprints[c], fp):
                raise ValueError('split {} was built from another snapshot of {}'.format(name, c))
    return split


# ---- paper-compatible fold manifests (PC-01) -------------------------------

def make_paper_folds(table, pair, protocol, k=10, group='season', policy='paper', seed=0,
                     val_frac=0.1, region='farm'):
    """Fold manifests for one country-crop pair under the YieldSAT paper's
    protocols (Pathak et al., CVPR 2026, §5):

    ``cv``   stratified grouped k-fold: seasons grouped by field season
             (``group='season'``, paper) or physical field (``'physical'``,
             leakage-safe), stratified by region (farm), balanced in pixels;
    ``loro`` one fold per region; a region is the fields of one farmer/local
             provider, i.e. the farm (paper) or the farm cluster (strict);
    ``loyo`` one fold per harvest year.
    ``policy='strict'`` additionally moves train/val seasons that share a
    physical field with the test fold to ``excluded``. A validation subset
    (``val_frac`` of the training groups, 0 = none) is carved from training
    for model selection; the paper does not specify one. Returns a list of
    split dicts (one per fold).
    """
    if protocol not in PAPER_PROTOCOLS:
        raise ValueError('protocol must be one of {}'.format(PAPER_PROTOCOLS))
    if policy not in ('paper', 'strict') or group not in ('season', 'physical'):
        raise ValueError('bad policy/group')
    country, crop = parse_pair(pair)
    fields = [f for f in table['fields'] if f['country'] == country and f['crop'] == crop]
    if not fields:
        raise ValueError('no field seasons for {}'.format(pair))
    rng = np.random.default_rng(seed)
    region_key = 'farm_id' if policy == 'paper' else 'farm_group_id'
    if region not in ('farm', 'province'):
        raise ValueError('region must be farm or province')
    if region == 'province':
        if any('province' not in f for f in fields):
            raise ValueError('province regions need the geometry table (run the geometry step)')
        loro_key = 'province'
    else:
        loro_key = region_key
    if protocol == 'cv':
        group_key = 'season_id' if group == 'season' else 'physical_field_id'
        test_folds = _stratified_group_folds(fields, group_key, region_key, k, seed)
    elif protocol == 'loro':
        group_key = loro_key
        regions = sorted({f[loro_key] for f in fields})
        test_folds = [[f['season_id'] for f in fields if f[loro_key] == r] for r in regions]
    else:
        group_key = 'year'
        years = sorted({f['year'] for f in fields})
        test_folds = [[f['season_id'] for f in fields if f['year'] == y] for y in years]
    by_season = {f['season_id']: f for f in fields}
    splits = []
    for i, test in enumerate(test_folds):
        test = set(test)
        assign = {s: 'test' for s in test}
        excluded = set()
        if policy == 'strict':
            test_phys = {by_season[s]['physical_field_id'] for s in test}
            excluded = {f['season_id'] for f in fields
                        if f['season_id'] not in test and f['physical_field_id'] in test_phys}
        rest = [f for f in fields if f['season_id'] not in test and f['season_id'] not in excluded]
        # validation for model selection only: carved by the CV grouping key,
        # by season for LORO/LOYO (strict merges shared ground below)
        vkey = 'physical_field_id' if protocol == 'cv' and group == 'physical' else 'season_id'
        groups = sorted({f[vkey] for f in rest}, key=str)
        rng.shuffle(groups)
        n_val = int(round(val_frac * len(groups))) if val_frac > 0 else 0
        val_groups = set(groups[:n_val]) if len(groups) > 1 else set()
        for f in rest:
            assign[f['season_id']] = 'val' if f[vkey] in val_groups else 'train'
        for s in excluded:
            assign[s] = 'excluded'
        if policy == 'strict':
            # the validation carve-out must not share ground with training either
            _merge_physical([f for f in fields if assign[f['season_id']] in ('train', 'val')], assign)
        train_phys = {by_season[s]['physical_field_id'] for s, a in assign.items() if a == 'train'}
        split = {
            'contract': CONTRACT_KEY, 'scheme': protocol, 'pair': pair, 'fold': i,
            'n_folds': len(test_folds), 'group': group if protocol == 'cv' else group_key,
            'group_key': group_key, 'leakage_policy': policy, 'seed': seed, 'val_frac': val_frac,
            'countries': [country], 'crops': [crop], 'fingerprints': table['fingerprints'],
            'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'geometry_verified': all(f['geometry_verified'] for f in fields),
            'loro_region': region if protocol == 'loro' else None,
            'holdout': (sorted({by_season[s][loro_key] for s in test}) if protocol == 'loro'
                        else sorted({by_season[s]['year'] for s in test}) if protocol == 'loyo'
                        else None),
            'partitions': {p: sorted(s for s, a in assign.items() if a == p)
                           for p in PARTITIONS + ('excluded',)},
            'physical_overlap_test_seasons': sum(1 for s in test
                                                 if by_season[s]['physical_field_id'] in train_phys),
        }
        split['summary'] = summarize_split(split, fields)
        check_split(split, fields)
        split['partition_hash'] = hashlib.sha256(
            json.dumps(split['partitions'], sort_keys=True).encode()).hexdigest()[:16]
        split['label'] = split_label(split)
        splits.append(split)
    covered = sorted(s for sp in splits for s in sp['partitions']['test'])
    if protocol != 'loyo' or policy == 'paper':
        if covered != sorted(by_season):
            raise ValueError('folds do not cover every season exactly once as test')
    return splits


def _stratified_group_folds(fields, group_key, region_key, k, seed):
    """Stratified grouped k-fold over pixels (as in the paper): each season
    contributes its pixel count, groups never straddle folds, and the region
    distribution is kept per fold."""
    from sklearn.model_selection import StratifiedGroupKFold

    groups = sorted({f[group_key] for f in fields}, key=str)
    if len(groups) < k:
        raise ValueError('only {} groups for {}-fold CV'.format(len(groups), k))
    gid = {g: i for i, g in enumerate(groups)}
    regions = sorted({f[region_key] for f in fields})
    rid = {r: i for i, r in enumerate(regions)}
    # one sample per 100 pixels keeps the pixel weighting at a tractable size
    reps = np.array([max(1, f['n_rows'] // 100) for f in fields])
    y = np.repeat([rid[f[region_key]] for f in fields], reps)
    g = np.repeat([gid[f[group_key]] for f in fields], reps)
    season = np.repeat(np.arange(len(fields)), reps)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)   # regions smaller than k
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        folds = []
        for _, test_idx in sgkf.split(np.zeros(len(y)), y, g):
            folds.append(sorted({fields[i]['season_id'] for i in np.unique(season[test_idx])}))
    return folds


def save_folds(splits, artifact_root, prefix):
    """Save fold manifests as ``<prefix>_fold<ii>`` plus an index file."""
    names = []
    for sp in splits:
        name = '{}_fold{:02d}'.format(prefix, sp['fold'])
        save_split(sp, artifact_root, name)
        names.append(name)
    index = {'prefix': prefix, 'pair': splits[0]['pair'], 'protocol': splits[0]['scheme'],
             'group': splits[0]['group'], 'leakage_policy': splits[0]['leakage_policy'],
             'n_folds': len(splits), 'folds': names,
             'test_seasons': [len(sp['partitions']['test']) for sp in splits],
             'physical_overlap_test_seasons': [sp['physical_overlap_test_seasons'] for sp in splits]}
    path = Path(artifact_root) / 'splits' / '{}.folds.json'.format(prefix)
    path.write_text(json.dumps(index, indent=1))
    return names


# ---- the paper's protocol: no validation set (spec/yieldsat-paper-protocol.md) ----

PROTOCOL_ROWS = {'cv': 'paper_cv10_{}_season_paper_s0',
                 'loyo': 'paper_loyo_{}_na_paper_s0',
                 # province regions for Argentina, farms elsewhere (point-suite convention)
                 'loro': 'paper_loro_{}_na_paper_s0'}


def paper_row_prefix(pair, protocol):
    if protocol == 'loro' and pair.startswith('ARG'):
        return 'paper_loro_{}_province_paper_s0'.format(pair)
    return PROTOCOL_ROWS[protocol].format(pair)


def _finish(split, label):
    split['partition_hash'] = hashlib.sha256(
        json.dumps(split['partitions'], sort_keys=True).encode()).hexdigest()[:16]
    split['label'] = label
    return split


def noval_split(split, source_name):
    """Train/test only, as in the paper: training absorbs the validation
    carve-out and model selection (the ``val`` partition every trainer reads)
    is the held-out test fold itself. ``selection: test_fold`` marks it."""
    p = split['partitions']
    s = dict(split)
    s['partitions'] = {'train': sorted(p['train'] + p['val']), 'val': list(p['test']), 'test': list(p['test']),
                       'excluded': list(p.get('excluded', []))}
    s['selection'] = 'test_fold'
    s['val_frac'] = 0.0
    s['source_split'] = source_name
    s['created'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    return _finish(s, split['label'] + '; no validation set, selection on the test fold')


def _norm_region(r):
    # the source metadata spells some Argentine provinces two ways ('Buenos Aires' / 'Buenos_Aires')
    return str(r).replace('_', ' ')


def pooled_noval_splits(artifact_root, protocol, pairs=PAPER_PAIRS):
    """One model for all country-crop pairs, reported per pair. Fold k holds out
    every pair's fold k (CV10), every season of one harvest year (LOYO) or of
    one region (LORO, provinces merged across spellings); each season is still
    tested exactly once and the per-pair test folds are the per-pair protocol's.
    No validation set: selection on the test fold."""
    root = Path(artifact_root) / 'splits'
    per_pair = {}
    for pair in pairs:
        prefix = paper_row_prefix(pair, protocol)
        names = json.loads((root / '{}.folds.json'.format(prefix)).read_text())['folds']
        per_pair[pair] = [(n, json.loads((root / '{}.json'.format(n)).read_text())) for n in names]
    if protocol == 'cv':
        keys = sorted({sp['fold'] for v in per_pair.values() for _, sp in v})
        key_of = lambda sp: sp['fold']
    else:
        keys = sorted({_norm_region(h) for v in per_pair.values() for _, sp in v for h in sp['holdout']}, key=str)
        key_of = lambda sp: _norm_region(sp['holdout'][0])
    fingerprints = {}
    for v in per_pair.values():
        fingerprints.update(v[0][1]['fingerprints'])
    splits = []
    for i, key in enumerate(keys):
        train, test, excluded, sources = [], [], [], []
        for pair, folds in per_pair.items():
            p0 = folds[0][1]['partitions']
            every = sorted(set(p0['train'] + p0['val'] + p0['test']))
            held = set()
            for name, sp in folds:
                if key_of(sp) == key:
                    held |= set(sp['partitions']['test'])
                    excluded += sp['partitions'].get('excluded', [])
                    sources.append(name)
            test += sorted(held)
            train += [s for s in every if s not in held]
        s = {'contract': CONTRACT_KEY, 'scheme': protocol, 'pair': 'ALL', 'fold': i, 'n_folds': len(keys),
             'group': 'season' if protocol == 'cv' else protocol, 'leakage_policy': 'paper', 'seed': 0,
             'val_frac': 0.0, 'selection': 'test_fold', 'pooled_pairs': list(pairs),
             'countries': sorted({parse_pair(p)[0] for p in pairs}),
             'crops': sorted({parse_pair(p)[1] for p in pairs}), 'fingerprints': fingerprints,
             'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'holdout': None if protocol == 'cv' else [key],
             'source_splits': sources,
             'partitions': {'train': sorted(train), 'val': sorted(test), 'test': sorted(test),
                            'excluded': sorted(set(excluded))},
             'physical_overlap_test_seasons': None}
        splits.append(_finish(s, 'pooled {} fold {} of {} (all pairs; no validation set, selection on the '
                                 'test fold)'.format(protocol.upper(), i + 1, len(keys))))
    tested = sorted(s for sp in splits for s in sp['partitions']['test'])
    if len(tested) != len(set(tested)):
        raise ValueError('a season is tested twice')
    return splits
