"""Pretraining-unit split manifests for knowledge pretraining (PK-05).

A unit is one pretraining corpus over all YieldSAT countries (inputs only)
that excludes every season a group of DEV evaluation folds tests on:

  CV10 fold k   -> unit ``cv_kNN``: the fold-k test seasons of every DEV pair
  LOYO year Y   -> unit ``loyo_Y``: every season harvested in Y, all countries/crops
  LORO region R -> unit ``loro_<code>``: every season of R (all crops of that country)

Each unit manifest has the split-manifest contract (``scheme='pretrain_unit'``)
with train / val (10% of physical fields) / test (empty) / excluded, and a
``units.json`` index maps every DEV fold manifest to its unit.
``check_unit`` proves no excluded season reaches train or val. See
spec/yieldsat-point-knowledge-pretraining.md §5.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from dataset.yieldsat_schema import CONTRACT_KEY, COUNTRIES
from dataset.yieldsat_splits import load_field_table, load_split, save_split

DEV_PAIRS = ('ARG-W', 'BRA-C', 'GER-R', 'URG-S')
# DEV fold manifests (paper policy); ARG-W LORO uses provinces
DEV_FOLDS = {'cv': 'paper_cv10_{pair}_season_paper_s0',
             'loyo': 'paper_loyo_{pair}_na_paper_s0',
             'loro': 'paper_loro_{pair}_na_paper_s0'}
LORO_PROVINCE_PAIRS = ('ARG-W',)
INDEX_NAME = 'pretrain_units_dev_s0'


def dev_fold_names(artifact_root, pairs=DEV_PAIRS):
    out = []
    for pair in pairs:
        for proto, pattern in DEV_FOLDS.items():
            prefix = pattern.format(pair=pair)
            if proto == 'loro' and pair in LORO_PROVINCE_PAIRS:
                prefix = 'paper_loro_{}_province_paper_s0'.format(pair)
            index = json.loads((Path(artifact_root) / 'splits' / '{}.folds.json'.format(prefix)).read_text())
            out += [(proto, name) for name in index['folds']]
    return out


def _slug(text):
    """Region slug; spelling variants in the source metadata ('Buenos Aires' /
    'Buenos_Aires') map to one slug, i.e. one physical region."""
    return re.sub(r'[^A-Za-z0-9]+', '-', text).strip('-')


def unit_of(proto, split):
    """(unit name, exclusion rule) for one DEV fold manifest."""
    if proto == 'cv':
        return 'cv_k{:02d}'.format(split['fold']), ('seasons', None)
    if proto == 'loyo':
        if len(split['holdout']) != 1:
            raise ValueError('LOYO fold with several years: {}'.format(split['holdout']))
        y = int(split['holdout'][0])
        return 'loyo_{}'.format(y), ('year', y)
    if len(split['holdout']) != 1:
        raise ValueError('LORO fold with several regions: {}'.format(split['holdout']))
    region = split['holdout'][0]
    return 'loro_{}_{}'.format(split['group_key'], _slug(region)), (split['group_key'], region)


def plan_units(artifact_root, fields, pairs=DEV_PAIRS):
    """{unit: {'excluded': set(season_id), 'folds': [fold names], 'rule': ...}}"""
    units = {}
    for proto, name in dev_fold_names(artifact_root, pairs):
        split = load_split(artifact_root, name)
        unit, (key, value) = unit_of(proto, split)
        u = units.setdefault(unit, {'excluded': set(), 'folds': [], 'protocol': proto,
                                    'rule': None if key == 'seasons' else {'key': key, 'values': []}})
        u['folds'].append(name)
        if key != 'seasons' and value not in u['rule']['values']:
            u['rule']['values'].append(value)   # spelling variants of one region share a unit
        test = set(split['partitions']['test'])
        u['excluded'] |= test
        if key != 'seasons':
            country = split['countries'][0]
            u['excluded'] |= {f['season_id'] for f in fields
                              if f.get(key) == value and (key == 'year' or f['country'] == country)}
    return units


def make_unit(name, unit, fields, fingerprints, val_frac=0.1, seed=0):
    eligible = [f for f in fields if f['season_id'] not in unit['excluded']]
    phys = sorted({f['physical_field_id'] for f in eligible}, key=str)
    rng = np.random.default_rng((seed, int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)))
    rng.shuffle(phys)
    val_phys = set(phys[:int(round(val_frac * len(phys)))])
    parts = {'train': sorted(f['season_id'] for f in eligible if f['physical_field_id'] not in val_phys),
             'val': sorted(f['season_id'] for f in eligible if f['physical_field_id'] in val_phys),
             'test': [], 'excluded': sorted(unit['excluded'])}
    split = {'contract': CONTRACT_KEY, 'scheme': 'pretrain_unit', 'unit': name, 'protocol': unit['protocol'],
             'rule': unit['rule'], 'dev_folds': sorted(unit['folds']), 'seed': seed, 'val_frac': val_frac,
             'countries': sorted({f['country'] for f in fields}), 'crops': None,
             'fingerprints': fingerprints, 'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
             'partitions': parts, 'holdout': unit['rule'],
             'label': 'pretraining unit {} (excludes {} seasons of {} DEV folds)'.format(
                 name, len(unit['excluded']), len(unit['folds']))}
    split['partition_hash'] = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]
    return split


def check_unit(unit_split, artifact_root, fields):
    """Raise if any season tested by a mapped DEV fold, or any season of a
    held-out year/region, is in the unit's train or val partition."""
    used = set(unit_split['partitions']['train']) | set(unit_split['partitions']['val'])
    if set(unit_split['partitions']['test']):
        raise ValueError('pretraining units have no test partition')
    for name in unit_split['dev_folds']:
        clash = used & set(load_split(artifact_root, name)['partitions']['test'])
        if clash:
            raise ValueError('unit {} trains on {} test seasons of {} (e.g. {})'.format(
                unit_split['unit'], len(clash), name, sorted(clash)[:2]))
    rule = unit_split['rule']
    if rule is not None:
        by_season = {f['season_id']: f for f in fields}
        countries = {load_split(artifact_root, n)['countries'][0] for n in unit_split['dev_folds']}
        bad = [s for s in used if by_season[s].get(rule['key']) in rule['values']
               and (rule['key'] == 'year' or by_season[s]['country'] in countries)]
        if bad:
            raise ValueError('unit {} trains on held-out {} {} (e.g. {})'.format(
                unit_split['unit'], rule['key'], rule['values'], bad[:2]))
    all_ids = {f['season_id'] for f in fields}
    parts = unit_split['partitions']
    assigned = parts['train'] + parts['val'] + parts['excluded']
    if len(assigned) != len(set(assigned)) or set(assigned) != all_ids:
        raise ValueError('unit {} does not assign every season exactly once'.format(unit_split['unit']))
    return {'unit': unit_split['unit'], 'train': len(parts['train']), 'val': len(parts['val']),
            'excluded': len(parts['excluded'])}


def build(artifact_root, source_root, pairs=DEV_PAIRS, check_source=True, val_frac=0.1, seed=0):
    table = load_field_table(artifact_root, source_root, list(COUNTRIES), check_source=check_source)
    fields = table['fields']
    units = plan_units(artifact_root, fields, pairs)
    index = {'name': INDEX_NAME, 'pairs': list(pairs), 'units': {}, 'fold_to_unit': {}}
    for name in sorted(units):
        split = make_unit(name, units[name], fields, table['fingerprints'], val_frac, seed)
        audit = check_unit(split, artifact_root, fields)
        manifest = 'pretrain_unit_{}_s{}'.format(name, seed)
        save_split(split, artifact_root, manifest)
        index['units'][name] = dict(audit, manifest=manifest, partition_hash=split['partition_hash'])
        for f in units[name]['folds']:
            index['fold_to_unit'][f] = manifest
    path = Path(artifact_root) / 'splits' / '{}.json'.format(INDEX_NAME)
    path.write_text(json.dumps(index, indent=1))
    return index


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--artifact_root', required=True)
    p.add_argument('--source_root', default=None)
    p.add_argument('--pairs', default=','.join(DEV_PAIRS))
    p.add_argument('--no_check_source', action='store_true')
    a = p.parse_args()
    idx = build(a.artifact_root, a.source_root, a.pairs.split(','), not a.no_check_source)
    for name, u in idx['units'].items():
        print('{:40s} train {:5d} val {:4d} excluded {:4d}'.format(name, u['train'], u['val'], u['excluded']))
    print('{} units for {} DEV folds; audit passed'.format(len(idx['units']), len(idx['fold_to_unit'])))


if __name__ == '__main__':
    main()
