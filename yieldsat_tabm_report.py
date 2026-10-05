"""TabM DEV report (spec/yieldsat-tabm.md, TM-07): results/tabm_dev.md.

Inputs are pooled metrics with per-fold statistics (yieldsat_pooled_metrics.py
--fold_stats) for the TabM round, the point model A0 (before_full, ours_seed0)
and p3-nbr (dev_r2). Reports DEV means, protocol means, per-row values and
paired fold-bootstrap Δ with 95% CIs (yieldsat_pretrain_report.compare).

    python yieldsat_tabm_report.py --tm pooled_tm_fs.json --a0 pooled_a0_fs.json \
        --p3 pooled_p3_fs.json --out results/tabm_dev.md
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from yieldsat_dev_eval import PROTO, paper_best
from yieldsat_pretrain_report import _r2_from, compare

LABELS = {'tm-tabm-f0': 'TabM F0', 'tm-tabm-f1': 'TabM F1 (+ S6 neighbourhood)', 'tm-lgbm-f0': 'LightGBM F0',
          'tm-mlp-f0': 'MLP F0 (k = 1)', 'A0': 'Point model A0 (before_full seed 0)',
          'p3-nbr': 'Point + S6 (p3-nbr, dev_r2)'}
EXPECTED_FOLDS = None


def arms_from(pooled, tag_of):
    arms = {}
    for d, v in pooled.items():
        _, group, inputs, tagseed, pair = d.split('/')
        tag = tag_of(tagseed)
        if tag is None or 'fold_stats' not in v:
            continue
        arms.setdefault(tag, {})[(PROTO[group.split('_')[0]], pair, group)] = v['fold_stats']
    return arms


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tm', required=True)
    p.add_argument('--a0', required=True)
    p.add_argument('--p3', required=True)
    p.add_argument('--paper', default='results/yieldsat/paper_benchmark.csv')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    arms = arms_from(json.loads(Path(a.tm).read_text()), lambda t: t.rsplit('_seed', 1)[0] if t.startswith('tm-') else None)
    arms.update(arms_from(json.loads(Path(a.a0).read_text()), lambda t: 'A0' if t == 'ours_seed0' else None))
    arms.update(arms_from(json.loads(Path(a.p3).read_text()), lambda t: 'p3-nbr' if t == 'p3-nbr_seed0' else None))
    rows = sorted(arms['tm-tabm-f0'])
    for k in arms:
        arms[k] = {r: v for r, v in arms[k].items() if r in rows}
    best = paper_best(a.paper)
    order = ['tm-tabm-f1', 'tm-tabm-f0', 'tm-lgbm-f0', 'tm-mlp-f0', 'p3-nbr', 'A0']
    order = [o for o in order if o in arms]
    means = {}
    for k in order:
        vals = {r: _r2_from(v, list(v)) for r, v in arms[k].items()}
        means[k] = {'rows': len(vals), 'pixel': float(np.mean([x[0] for x in vals.values()])),
                    'field': float(np.mean([x[1] for x in vals.values()])), 'per_row': vals,
                    'proto': {pr: float(np.mean([x[0] for r, x in vals.items() if r[0] == pr]))
                              for pr in ('CV10', 'LOYO', 'LORO') if any(r[0] == pr for r in vals)}}
    bp = float(np.nanmean([best.get((r[0], 'pixel', r[1]), (np.nan,))[0] for r in rows]))
    bf = float(np.nanmean([best.get((r[0], 'field', r[1]), (np.nan,))[0] for r in rows]))
    bproto = {pr: float(np.nanmean([best.get((r[0], 'pixel', r[1]), (np.nan,))[0] for r in rows if r[0] == pr]))
              for pr in ('CV10', 'LOYO', 'LORO')}
    L = ['# TabM DEV results (round TM-1)', '',
         'Generated {} by `yieldsat_tabm_report.py`. spec/yieldsat-tabm.md. Pooled out-of-fold R² (the paper\'s '
         'computation) on the DEV subset (12 rows: ARG-W, BRA-C, GER-R, URG-S × CV10/LOYO/LORO, paper policy, S2+ADM, '
         'seed 0). Δ: paired fold bootstrap (2,000 draws), 95% CI. Success bar (improvement plan): paper best + 0.03 '
         'at pixel and field level.'.format(time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())), '',
         'Feature sets: F0 = month-aligned flat S2 (12 bands) + weather (4 sums) per slot, time features, DEM/terrain/'
         'soil, validity masks; F1 = F0 + 5×5 field-masked neighbourhood mean of the S2 bands. TabM: k = 32, '
         '3 × 256, dropout 0.2, periodic embeddings d = 8, early stopping on validation seasons (TM-06 sweep).', '',
         '## Summary', '', '| Model | Rows | Pixel R² | Field R² | CV10 / LOYO / LORO pixel |', '|---|---|---|---|---|']
    for k in order:
        m = means[k]
        L.append('| {} | {} | {:.3f} | {:.3f} | {} |'.format(LABELS.get(k, k), m['rows'], m['pixel'], m['field'],
                 ' / '.join('{:.2f}'.format(m['proto'][pr]) for pr in ('CV10', 'LOYO', 'LORO'))))
    L += ['| **Paper best** | 12 | **{:.3f}** | **{:.3f}** | {} |'.format(
        bp, bf, ' / '.join('{:.2f}'.format(bproto[pr]) for pr in ('CV10', 'LOYO', 'LORO'))),
        '| Success bar | | {:.3f} | {:.3f} | |'.format(bp + 0.03, bf + 0.03), '']
    pairs = [('tm-tabm-f0', 'A0'), ('tm-tabm-f1', 'A0'), ('tm-tabm-f0', 'p3-nbr'), ('tm-tabm-f1', 'p3-nbr'),
             ('tm-tabm-f1', 'tm-tabm-f0'), ('tm-tabm-f0', 'tm-lgbm-f0'), ('tm-tabm-f0', 'tm-mlp-f0'),
             ('tm-lgbm-f0', 'A0')]
    L += ['## Paired comparisons', '', '| Comparison | Rows | Δ pixel [95% CI] | Δ field [95% CI] | Δ pixel CV10 / LOYO / LORO |',
          '|---|---|---|---|---|']
    verdict = {}
    for x, y in pairs:
        if x not in arms or y not in arms:
            continue
        c = compare(arms, rows, x, y)
        verdict[(x, y)] = c
        L.append('| {} − {} | {} | {:+.3f} [{:+.3f}, {:+.3f}] | {:+.3f} [{:+.3f}, {:+.3f}] | {} |'.format(
            LABELS.get(x, x), LABELS.get(y, y), c['rows'], c['pixel'], *c['pixel_ci'], c['field'], *c['field_ci'],
            ' / '.join('{:+.3f}'.format(c['protocols'][pr][0]) for pr in ('CV10', 'LOYO', 'LORO'))))
    L += ['', '## Per row (pixel / field R²)', '',
          '| Model | ' + ' | '.join('{} {}'.format(r[1], r[0]) for r in rows) + ' |', '|---|' + '---|' * len(rows)]
    for k in order:
        L.append('| {} | {} |'.format(LABELS.get(k, k), ' | '.join(
            '{:.2f} / {:.2f}'.format(*means[k]['per_row'][r]) if r in means[k]['per_row'] else '–' for r in rows)))
    L.append('| Paper best | {} |'.format(' | '.join('{:.2f} / {:.2f}'.format(
        best.get((r[0], 'pixel', r[1]), (np.nan,))[0], best.get((r[0], 'field', r[1]), (np.nan,))[0]) for r in rows)))
    adopt = {k: (verdict.get((k, 'A0')) or {}) for k in ('tm-tabm-f0', 'tm-tabm-f1')}
    L += ['', '## Decisions (spec/yieldsat-tabm.md §4)', '']
    for k, c in adopt.items():
        if c:
            ok = c['pixel'] >= 0.01 and c['pixel_ci'][0] > 0 and c['field'] >= 0
            L.append('- **{} adopted as point backbone** (Δ pixel ≥ +0.01 vs A0 with CI > 0, field not worse): {}'.format(
                LABELS[k], 'yes' if ok else 'no'))
    for k in ('tm-tabm-f0', 'tm-tabm-f1'):
        m = means.get(k)
        if m:
            L.append('- {} passes the paper bar: {}'.format(LABELS[k], 'yes' if m['pixel'] >= bp + 0.03 and
                                                            m['field'] >= bf + 0.03 else 'no'))
    Path(a.out).write_text('\n'.join(L) + '\n')
    print('\n'.join(L[:30]))


if __name__ == '__main__':
    main()
