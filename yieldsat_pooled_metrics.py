"""Pooled out-of-fold metrics per experiment (the paper's computation).

For every experiment directory ``paper/<group>/<inputs>/<tag>_seed<s>/<pair>``
under a results root, all folds' ``test_predictions.npz`` are pooled (every
held-out cell appears in exactly one fold) and metrics are computed once:
pixel R²/RMSE/MAE/bias over all cells, and field-level R²/RMSE over field
seasons (mean prediction vs mean target per season). This matches the YieldSAT
paper's reported numbers (see results/negative_r2_investigation.md).

    python yieldsat_pooled_metrics.py --root /data/YieldSAT/yieldsat_results/before_full --out pooled.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from util.yieldsat_eval import regression_metrics


def pooled(exp_dir, fold_stats=False):
    ys, ps, seasons, folds, fold_of = [], [], [], [], []
    names = {}
    for d in sorted(exp_dir.glob('fold*')):
        f = d / 'test_predictions.npz'
        if not f.exists():
            continue
        z = np.load(f)
        ok = np.isfinite(z['target']) & np.isfinite(z['pred'])
        sn = z['season_names'].astype(str)
        code = np.array([names.setdefault(n, len(names)) for n in sn], np.int64)
        ys.append(z['target'][ok].astype(np.float64))
        ps.append(z['pred'][ok].astype(np.float64))
        seasons.append(code[z['season'][ok]])
        folds.append(d.name)
        fold_of.append(np.full(int(ok.sum()), len(folds) - 1, np.int16))
    if not folds:
        return None
    y, p, s = np.concatenate(ys), np.concatenate(ps), np.concatenate(seasons)
    px = regression_metrics(y, p)
    order = np.argsort(s, kind='stable')
    s_sorted = s[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s_sorted)) + 1]
    ym = np.add.reduceat(y[order], starts) / np.diff(np.r_[starts, len(s)])
    pm = np.add.reduceat(p[order], starts) / np.diff(np.r_[starts, len(s)])
    fl = regression_metrics(ym, pm)
    out = {'folds': folds, 'n_cells': int(len(y)), 'n_seasons': int(len(starts)),
           'pixel_r2': px.get('r2'), 'pixel_rmse': px.get('rmse'), 'pixel_mae': px.get('mae'),
           'pixel_bias': px.get('bias'), 'field_r2': fl.get('r2'), 'field_rmse': fl.get('rmse')}
    if fold_stats:
        # per-fold sufficient statistics: pooled R2 of any multiset of folds can be
        # recomputed (paired fold bootstrap in yieldsat_pretrain_report.py)
        f = np.concatenate(fold_of)
        f_season = f[order][starts]
        stats = {}
        for i, name in enumerate(folds):
            m = f == i
            fm = f_season == i
            stats[name] = {'n': int(m.sum()), 'sy': float(y[m].sum()), 'syy': float((y[m] ** 2).sum()),
                           'sse': float(((y[m] - p[m]) ** 2).sum()),
                           'field': np.stack([ym[fm], pm[fm]], 1).round(5).tolist()}
        out['fold_stats'] = stats
    return out


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--root', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--fold_stats', action='store_true', help='add per-fold sufficient statistics')
    a.add_argument('--match', default='*', help='glob on the experiment tag directory (e.g. "pk-*")')
    args = a.parse_args()
    root = Path(args.root)
    out = {}
    for exp in sorted(root.glob('paper/*/*/{}/*'.format(args.match))):
        if exp.is_dir():
            r = pooled(exp, args.fold_stats)
            if r:
                out[exp.relative_to(root).as_posix()] = r
                if len(out) % 50 == 0:
                    print('experiments', len(out), flush=True)
    Path(args.out).write_text(json.dumps(out))
    print('wrote', args.out, len(out), 'experiments')


if __name__ == '__main__':
    main()
