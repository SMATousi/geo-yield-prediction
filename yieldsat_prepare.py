# --------------------------------------------------------
# YieldSAT (yieldsat_preprocessed_v1) preparation CLI.
#
#   python yieldsat_prepare.py snapshot --countries Germany
#   python yieldsat_prepare.py index    --countries Germany
#   python yieldsat_prepare.py geometry --countries Germany --raw_zip .../Raw.zip
#   python yieldsat_prepare.py cache    --countries Germany
#   python yieldsat_prepare.py splits   --countries Germany --scheme farm
#
# Source files are only read. Everything is written under --artifact_root
# (default $YIELDSAT_ARTIFACT_ROOT), never under --source_root.
# --------------------------------------------------------

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

from dataset.yieldsat_schema import COUNTRIES


def _common(parser):
    parser.add_argument('--source_root', default=os.environ.get('YIELDSAT_SOURCE_ROOT'),
                        help='directory containing <Country>/merge_s2-soil-dem-weather-coords.nc')
    parser.add_argument('--artifact_root', default=os.environ.get('YIELDSAT_ARTIFACT_ROOT'),
                        help='writable directory for indexes, caches, splits and audits')
    parser.add_argument('--countries', nargs='+', required=True, choices=COUNTRIES)


def _check_roots(args):
    if not args.source_root or not args.artifact_root:
        sys.exit('--source_root and --artifact_root (or YIELDSAT_* env vars) are required')
    src = Path(args.source_root).resolve()
    art = Path(args.artifact_root).resolve()
    if art == src or src in art.parents:
        sys.exit('artifact_root must be outside source_root')
    art.mkdir(parents=True, exist_ok=True)


def cmd_snapshot(args):
    from dataset.yieldsat_source import check_snapshot
    report = check_snapshot(args.source_root, args.countries,
                            crc_evidence_dir=Path(args.artifact_root) / 'audit',
                            full_crc=args.full_crc)
    out = Path(args.artifact_root) / 'audit' / 'snapshot_{}.json'.format('-'.join(args.countries))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def cmd_index(args):
    from dataset.yieldsat_source import build_country_index
    for c in args.countries:
        m = build_country_index(args.source_root, c, Path(args.artifact_root) / 'index' / c)
        print(c, json.dumps(m['audit']))


def cmd_geometry(args):
    from dataset.yieldsat_geometry import (
        extract_field_geometry, physical_field_groups, verify_cell_correspondence)
    from dataset.yieldsat_source import load_country_index

    records = extract_field_geometry(args.raw_zip, countries=set(args.countries))
    physical, cross = physical_field_groups(records, min_iou=args.min_iou)
    for r in records:
        r['physical_field_id'] = physical[r['field_shared_name']]
    checks = []
    rng = np.random.default_rng(args.seed)
    by_name = {r['field_shared_name']: r for r in records}
    for c in args.countries:
        idx = load_country_index(args.artifact_root, args.source_root, c)
        missing = [f['field_shared_name'] for f in idx['fields'] if f['field_shared_name'] not in by_name]
        if missing:
            print('{}: {} fields without raw geometry'.format(c, len(missing)))
        chosen = rng.choice(len(idx['fields']), size=min(args.verify_fields, len(idx['fields'])),
                            replace=False)
        for i in chosen:
            fl = idx['fields'][int(i)]
            rec = by_name.get(fl['field_shared_name'])
            if rec is None:
                continue
            s, e = fl['row_start'], fl['row_end']
            rows = idx['rows']
            checks.append(verify_cell_correspondence(
                args.raw_zip, rec, rows['grid_row'][s:e], rows['grid_col'][s:e], rows['target'][s:e]))
            print(json.dumps(checks[-1]))
    out = Path(args.artifact_root) / 'geometry'
    out.mkdir(parents=True, exist_ok=True)
    tag = '-'.join(args.countries)
    (out / 'fields_geometry_{}.json'.format(tag)).write_text(json.dumps(records, indent=1))
    n_phys = len(set(physical.values()))
    summary = {'field_seasons': len(records), 'physical_fields': n_phys,
               'min_iou': args.min_iou, 'cross_farm_overlaps': cross,
               'cell_checks': checks}
    (out / 'geometry_summary_{}.json'.format(tag)).write_text(json.dumps(summary, indent=1))
    print('seasons', len(records), 'physical fields', n_phys, 'cross-farm overlaps', len(cross))


def cmd_cache(args):
    from dataset.yieldsat_cache import build_country_cache
    for c in args.countries:
        m = build_country_cache(args.source_root, c, args.artifact_root,
                                block_rows=args.block_rows, write_cache=not args.stats_only,
                                max_rows=args.max_rows)
        print(c, 'done in', m['seconds'], 's')


def cmd_stats(args):
    from dataset.yieldsat_cache import recompute_field_stats
    for c in args.countries:
        recompute_field_stats(args.artifact_root, args.source_root, c)
        print(c, 'field statistics rebuilt')


def cmd_semantics(args):
    """YS-02 evidence from the cache: weather interval semantics (value at a
    slot divided by the days since the previous dated slot), within-field
    weather constancy, and value ranges per channel on sampled rows."""
    from dataset.yieldsat_cache import load_cache_manifest
    from dataset.yieldsat_schema import STATIC_CHANNELS, TEMPORAL_CHANNELS
    from dataset.yieldsat_source import load_country_index

    rng = np.random.default_rng(args.seed)
    for c in args.countries:
        load_cache_manifest(args.artifact_root, args.source_root, c)
        d = Path(args.artifact_root) / 'cache' / c
        temporal = np.load(d / 'temporal.npy', mmap_mode='r')
        static = np.load(d / 'static.npy', mmap_mode='r')
        times = np.load(d / 'times.npy', mmap_mode='r')
        rows = np.sort(rng.choice(temporal.shape[0], min(args.rows, temporal.shape[0]), replace=False))
        t, s, tm = np.asarray(temporal[rows]), np.asarray(static[rows]), np.asarray(times[rows])
        dt = np.diff(tm, axis=1)                                   # (N, 23), NaN when undated
        w = {name: t[:, 1:, TEMPORAL_CHANNELS.index(name)] for name in
             ('temp_max', 'temp_mean', 'temp_min', 'total_prec')}
        ok = np.isfinite(dt) & (dt > 0) & np.isfinite(w['temp_mean'])
        # inclusive-interval hypothesis: the value at slot k sums daily values
        # over [t_{k-1}, t_k], i.e. dt + 1 days; it holds when the per-day
        # ratio is flat across interval lengths (offset 0 is shown as control)
        per_day = {k: v[ok] / (dt[ok] + 1) for k, v in w.items()}
        flatness = {}
        for off in (0, 1):
            r = w['temp_mean'][ok] / (dt[ok] + off)
            flatness['offset_{}'.format(off)] = {
                int(k): round(float(np.median(r[dt[ok] == k])), 1)
                for k in (5, 10, 20, 30, 40, 50) if (dt[ok] == k).sum() > 100}
        q = lambda x: np.percentile(x, [1, 25, 50, 75, 99]).round(4).tolist()
        tv = np.isfinite(tm)
        order_ok = (t[..., TEMPORAL_CHANNELS.index('temp_max')] + 1e-3 >= t[..., TEMPORAL_CHANNELS.index('temp_mean')]) \
            & (t[..., TEMPORAL_CHANNELS.index('temp_mean')] + 1e-3 >= t[..., TEMPORAL_CHANNELS.index('temp_min')])
        # within-field constancy: for a few fields, spread of weather across cells per slot
        idx = load_country_index(args.artifact_root, args.source_root, c)
        spreads = []
        for fl in [idx['fields'][i] for i in rng.choice(len(idx['fields']), 10, replace=False)]:
            a, b = fl['row_start'], min(fl['row_end'], fl['row_start'] + 500)
            wf = np.asarray(temporal[a:b, :, TEMPORAL_CHANNELS.index('temp_mean')])
            with np.errstate(invalid='ignore'), warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                spreads.append(float(np.nanmax(np.nanstd(wf, axis=0)) if np.isfinite(wf).any() else 0.0))
        ranges = {}
        for j, name in enumerate(TEMPORAL_CHANNELS):
            v = t[..., j][np.isfinite(t[..., j])]
            ranges[name] = q(v) if v.size else None
        for j, name in enumerate(STATIC_CHANNELS):
            v = s[:, j][np.isfinite(s[:, j])]
            ranges[name] = q(v) if v.size else None
        report = {
            'country': c, 'rows_sampled': int(len(rows)),
            'weather_per_day_quantiles_p1_p25_p50_p75_p99': {k: q(v) for k, v in per_day.items()},
            'weather_intervals_checked': int(ok.sum()),
            'temp_mean_per_day_median_by_interval_days': flatness,
            'temp_order_max_ge_mean_ge_min_fraction': float(order_ok[tv].mean()),
            'max_within_field_temp_mean_std_per_slot': max(spreads),
            'interval_days_quantiles': q(dt[np.isfinite(dt)]),
            'channel_quantiles_p1_p25_p50_p75_p99': ranges,
        }
        out = Path(args.artifact_root) / 'audit' / 'semantics_{}.json'.format(c)
        out.write_text(json.dumps(report, indent=1))
        print(c, json.dumps({k: report[k] for k in ('weather_per_day_quantiles_p1_p25_p50_p75_p99',
                                                     'temp_mean_per_day_median_by_interval_days',
                                                     'temp_order_max_ge_mean_ge_min_fraction',
                                                     'max_within_field_temp_mean_std_per_slot',
                                                     'interval_days_quantiles')}))


def cmd_splits(args):
    from dataset.yieldsat_splits import load_field_table, make_split, save_split
    fields = load_field_table(args.artifact_root, args.source_root, args.countries,
                              require_geometry=args.scheme == 'block')
    split = make_split(fields, scheme=args.scheme, seed=args.seed, val_frac=args.val_frac,
                       test_frac=args.test_frac, holdout_country=args.holdout_country,
                       holdout_year=args.holdout_year, block_km=args.block_km)
    path = save_split(split, args.artifact_root, args.name)
    print(json.dumps(split['summary'], indent=1))
    print('wrote', path)


def cmd_folds(args):
    from dataset.yieldsat_splits import (
        PAPER_PAIRS, load_field_table, make_paper_folds, parse_pair, save_folds)
    pairs = args.pairs or [p for p in PAPER_PAIRS if parse_pair(p)[0] in args.countries]
    table = load_field_table(args.artifact_root, args.source_root,
                             sorted({parse_pair(p)[0] for p in pairs}),
                             require_geometry=args.policy == 'strict' or args.group == 'physical')
    for pair in pairs:
        for protocol in args.protocols:
            splits = make_paper_folds(table, pair, protocol, k=args.k, group=args.group,
                                      policy=args.policy, seed=args.seed, val_frac=args.val_frac)
            prefix = fold_prefix(pair, protocol, args.group, args.policy, args.seed, args.k)
            save_folds(splits, args.artifact_root, prefix)
            print(prefix, len(splits), 'folds; physical overlap (test seasons):',
                  sum(sp['physical_overlap_test_seasons'] for sp in splits))


def fold_prefix(pair, protocol, group, policy, seed, k=10, region='farm'):
    name = {'cv': 'cv{}'.format(k), 'loro': 'loro', 'loyo': 'loyo'}[protocol]
    tag = group if protocol == 'cv' else ('province' if protocol == 'loro' and region == 'province' else 'na')
    return 'paper_{}_{}_{}_{}_s{}'.format(name, pair, tag, policy, seed)


def main():
    parser = argparse.ArgumentParser('YieldSAT preparation')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('snapshot', help='YS-01 snapshot/integrity check')
    _common(p)
    p.add_argument('--full_crc', action='store_true', help='recompute whole-file CRC32')
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser('index', help='YS-01/03 per-row index and field table')
    _common(p)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser('geometry', help='YS-02 per-field geometry from Raw.zip')
    _common(p)
    p.add_argument('--raw_zip', required=True)
    p.add_argument('--min_iou', type=float, default=0.3)
    p.add_argument('--verify_fields', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.set_defaults(func=cmd_geometry)

    p = sub.add_parser('cache', help='YS-04 audit pass, compact cache, field statistics')
    _common(p)
    p.add_argument('--block_rows', type=int, default=4096)
    p.add_argument('--max_rows', type=int, default=None, help='partial pass (not a valid cache)')
    p.add_argument('--stats_only', action='store_true', help='audit + statistics without cache')
    p.set_defaults(func=cmd_cache)

    p = sub.add_parser('stats', help='rebuild per-field statistics from an existing cache')
    _common(p)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser('folds', help='PC-01 paper-compatible fold manifests per country-crop pair')
    _common(p)
    p.add_argument('--pairs', nargs='+', default=None,
                   help='e.g. GER-R ARG-S (default: all paper pairs of --countries)')
    p.add_argument('--protocols', nargs='+', default=['cv', 'loro', 'loyo'],
                   choices=['cv', 'loro', 'loyo'])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--group', default='season', choices=['season', 'physical'])
    p.add_argument('--policy', default='paper', choices=['paper', 'strict'])
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=0)
    p.set_defaults(func=cmd_folds)

    p = sub.add_parser('semantics', help='YS-02 unit/aggregation evidence from the cache')
    _common(p)
    p.add_argument('--rows', type=int, default=200000)
    p.add_argument('--seed', type=int, default=0)
    p.set_defaults(func=cmd_semantics)

    p = sub.add_parser('splits', help='YS-03 grouped split manifest')
    _common(p)
    p.add_argument('--scheme', required=True, choices=['farm', 'field', 'block', 'country', 'year'])
    p.add_argument('--name', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--val_frac', type=float, default=0.15)
    p.add_argument('--test_frac', type=float, default=0.2)
    p.add_argument('--holdout_country', default=None)
    p.add_argument('--holdout_year', type=int, default=None)
    p.add_argument('--block_km', type=float, default=20.0)
    p.set_defaults(func=cmd_splits)

    args = parser.parse_args()
    _check_roots(args)
    args.func(args)


if __name__ == '__main__':
    main()
