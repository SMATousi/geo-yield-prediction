"""Score improvement-plan configurations on the DEV subset (spec/yieldsat-improvement.md).

Inputs: pooled metrics of a DEV round (``yieldsat_pooled_metrics.py`` on its
results root), the round's plan ``runs.jsonl``, the point suite's pooled
metrics (baseline) and the paper benchmark. For every configuration
(experiment name) and DEV row (pair x protocol), pooled pixel and field R²
next to the point model and the paper's best (S2+ADM). Then the DEV means and
the success criterion: mean ≥ paper-best mean + margin at pixel and field
level, and no protocol mean below the paper's best. Only complete experiments
(all folds finished) count.

    python yieldsat_dev_eval.py --pooled pooled_dev_r1.json --runs runs.jsonl \
        --point pooled_before_full.json --out results/dev_r1.md
"""

import argparse
import collections
import csv
import json
import time
from pathlib import Path

import numpy as np

PROTO = {'cv10': 'CV10', 'loyo': 'LOYO', 'loro': 'LORO'}


def paper_best(path, inputs='S2+ADM'):
    best = {}
    for p in csv.DictReader(open(path)):
        if p['modalities'] != inputs:
            continue
        k = (p['protocol'], p['level'], p['pair'])
        v = float(p['r2_mean'])
        if k not in best or v > best[k][0]:
            best[k] = (v, p['model'])
    return best


def row_key(exp_dir):
    """paper/<group>/<inputs>/<tag>_seed<s>/<pair> -> (protocol, pair, inputs, policy, seed)."""
    _, group, inputs, tagseed, pair = exp_dir.split('/')
    return PROTO[group.split('_')[0]], pair, inputs, group.split('_')[-2], int(tagseed.rsplit('_seed', 1)[1]), group


def evaluate(pooled, runs, point, best, margin=0.03):
    expected = collections.Counter()
    exp_name = {}
    for r in runs:
        d = r['rel_path'].rsplit('/', 1)[0]
        expected[d] += 1
        exp_name[d] = r['experiment']
    configs = collections.defaultdict(dict)       # name -> {(proto, pair): [metrics over seeds]}
    incomplete = collections.Counter()
    for d, n in expected.items():
        got = pooled.get(d)
        name = exp_name[d]
        if not got or len(got['folds']) < n:
            incomplete[name] += 1
            continue
        proto, pair, inputs, policy, seed, group = row_key(d)
        configs[name].setdefault((proto, pair, group, inputs, policy), []).append(got)
    rows = sorted({k for c in configs.values() for k in c})
    base = {}
    for k in rows:
        proto, pair, group, inputs, policy = k
        vals = [v for d, v in point.items() if d.startswith('paper/{}/{}/ours_seed'.format(group, inputs))
                and d.endswith('/' + pair)]
        if vals:
            base[k] = (np.mean([v['pixel_r2'] for v in vals]), np.mean([v['field_r2'] for v in vals]))
    out = {}
    for name, c in configs.items():
        res = {}
        for k, vs in c.items():
            proto, pair = k[0], k[1]
            res[k] = {'pixel': float(np.mean([v['pixel_r2'] for v in vs])),
                      'field': float(np.mean([v['field_r2'] for v in vs])),
                      'best_pixel': best.get((proto, 'pixel', pair), (np.nan, ''))[0],
                      'best_field': best.get((proto, 'field', pair), (np.nan, ''))[0],
                      'point': base.get(k, (np.nan, np.nan))}
        complete = len(res) == len(rows)
        mean = lambda key, sel=None: float(np.nanmean([r[key] for k, r in res.items() if sel is None or k[0] == sel]))
        summary = {'rows': len(res), 'complete': complete, 'incomplete_experiments': incomplete.get(name, 0),
                   'pixel': mean('pixel'), 'field': mean('field'), 'best_pixel': mean('best_pixel'),
                   'best_field': mean('best_field'),
                   'protocols': {p: (mean('pixel', p), mean('best_pixel', p)) for p in ('CV10', 'LOYO', 'LORO')
                                 if any(k[0] == p for k in res)}}
        summary['pass'] = bool(complete and summary['pixel'] >= summary['best_pixel'] + margin
                               and summary['field'] >= summary['best_field'] + margin
                               and all(a >= b for a, b in summary['protocols'].values()))
        out[name] = {'rows': res, 'summary': summary}
    return out, rows, base


def write_md(out, rows, base, path, title, margin):
    point_pix = np.nanmean([base[k][0] for k in rows if k in base]) if base else float('nan')
    point_fld = np.nanmean([base[k][1] for k in rows if k in base]) if base else float('nan')
    lines = ['# {}'.format(title), '',
             'Generated {} by `yieldsat_dev_eval.py`. Pooled out-of-fold R² (the paper\'s computation) on the '
             'DEV subset (spec/yieldsat-improvement.md); paper = best model per row, S2+ADM. Success: DEV mean ≥ '
             'paper-best mean + {:.2f} at pixel and field level, and no protocol mean below the paper\'s best.'.format(
                 time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()), margin), '',
             '## Summary', '',
             '| Configuration | Rows | Pixel R² | Field R² | CV10 / LOYO / LORO pixel (paper best) | Pass |',
             '|---|---|---|---|---|---|']
    for name in sorted(out):
        s = out[name]['summary']
        prot = ' / '.join('{:.2f} ({:.2f})'.format(*s['protocols'][p]) for p in ('CV10', 'LOYO', 'LORO')
                          if p in s['protocols'])
        lines.append('| {} | {}/{}{} | {:.3f} | {:.3f} | {} | {} |'.format(
            name, s['rows'], len(rows), '' if s['complete'] else ' (incomplete)', s['pixel'], s['field'], prot,
            '**yes**' if s['pass'] else 'no'))
    any_s = next(iter(out.values()))['summary'] if out else None
    if any_s:
        lines.append('| Point model (before_full) | {} | {:.3f} | {:.3f} | | |'.format(len(rows), point_pix, point_fld))
        lines.append('| **Paper best** | | **{:.3f}** | **{:.3f}** | | |'.format(any_s['best_pixel'], any_s['best_field']))
    lines += ['', '## Per row (pixel R² / field R²)', '',
              '| Configuration | ' + ' | '.join('{} {}'.format(k[1], k[0]) for k in rows) + ' |',
              '|---|' + '---|' * len(rows)]
    for name in sorted(out):
        r = out[name]['rows']
        lines.append('| {} | {} |'.format(name, ' | '.join(
            '{:.2f} / {:.2f}'.format(r[k]['pixel'], r[k]['field']) if k in r else '–' for k in rows)))
    lines.append('| Point model | {} |'.format(' | '.join(
        '{:.2f} / {:.2f}'.format(*base[k]) if k in base else '–' for k in rows)))
    any_r = next(iter(out.values()))['rows'] if out else {}
    lines.append('| Paper best | {} |'.format(' | '.join(
        '{:.2f} / {:.2f}'.format(any_r[k]['best_pixel'], any_r[k]['best_field']) if k in any_r else '–'
        for k in rows)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(lines) + '\n')


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--pooled', required=True)
    a.add_argument('--runs', required=True)
    a.add_argument('--point', required=True)
    a.add_argument('--paper', default='results/yieldsat/paper_benchmark.csv')
    a.add_argument('--margin', type=float, default=0.03)
    a.add_argument('--title', default='DEV round results')
    a.add_argument('--out', required=True)
    args = a.parse_args()
    runs = [json.loads(l) for l in open(args.runs)]
    out, rows, base = evaluate(json.loads(Path(args.pooled).read_text()), runs,
                               json.loads(Path(args.point).read_text()), paper_best(args.paper), args.margin)
    write_md(out, rows, base, args.out, args.title, args.margin)
    for name in sorted(out):
        s = out[name]['summary']
        print('{:<22} rows {:>2}/{} pixel {:.3f} field {:.3f} (paper best {:.3f} / {:.3f}) pass={}'.format(
            name, s['rows'], len(rows), s['pixel'], s['field'], s['best_pixel'], s['best_field'], s['pass']))


if __name__ == '__main__':
    main()
