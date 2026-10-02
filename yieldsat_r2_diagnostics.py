"""R² diagnostics for cluster results (investigation of negative R², 2026-10-02).

For every experiment directory ``paper/<group>/<inputs>/<tag>_seed<s>/<pair>``
under a results root, from each fold's ``test_predictions.npz`` and
``normalizer.json``:

- ``fold_r2``          per-fold pixel R² (1 − SSE/SST on the test fold), as
                       reported (the paper's text: mean over folds);
- ``pooled_r2``        R² of all folds' out-of-fold predictions pooled (the
                       release tutorial's computation);
- ``fold_pearson2``    squared correlation per fold (insensitive to bias/scale);
- ``fold_r2_debiased`` per-fold R² after removing that fold's mean error;
- ``fold_bias_share``  share of the fold's MSE that is mean bias (bias² / MSE);
- ``trainmean_r2``     per-fold R² of a constant prediction = training-fold
                       target mean (how hard a fold is for R²);
- ``test_std`` / ``pooled_std``  target std within a test fold vs over all folds.

    python yieldsat_r2_diagnostics.py --root /data/YieldSAT/yieldsat_results/before_full --out diag.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def _r2(y, p):
    sst = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / sst) if sst > 0 else float('nan')


def _train_mean(norm):
    if 'stats' in norm:                        # point normalizer
        return norm['stats']['all']['target_mean']
    return norm['target']['mean'][0]           # image normalizer


def experiment(exp_dir):
    folds, ys, ps = [], [], []
    for d in sorted(exp_dir.glob('fold*')):
        f = d / 'test_predictions.npz'
        if not f.exists():
            continue
        z = np.load(f)
        ok = np.isfinite(z['target']) & np.isfinite(z['pred'])
        y, p = z['target'][ok].astype(np.float64), z['pred'][ok].astype(np.float64)
        if y.size < 2:
            continue
        tm = _train_mean(json.loads((d / 'normalizer.json').read_text()))
        e = p - y
        mse = float((e ** 2).mean())
        folds.append({'fold': d.name, 'n': int(y.size), 'r2': _r2(y, p), 'r2_debiased': _r2(y, p - e.mean()),
                      'pearson2': float(np.corrcoef(y, p)[0, 1] ** 2) if y.std() > 0 and p.std() > 0 else float('nan'),
                      'bias': float(e.mean()), 'bias_share': float(e.mean() ** 2 / mse) if mse > 0 else 0.0,
                      'rmse': mse ** 0.5, 'test_mean': float(y.mean()), 'test_std': float(y.std()),
                      'train_mean': float(tm), 'trainmean_r2': _r2(y, np.full_like(y, tm)),
                      'pred_std': float(p.std())})
        ys.append(y)
        ps.append(p)
    if not folds:
        return None
    y, p = np.concatenate(ys), np.concatenate(ps)
    mean = lambda k: float(np.nanmean([f[k] for f in folds]))
    return {'folds': len(folds), 'fold_r2': mean('r2'), 'pooled_r2': _r2(y, p),
            'fold_pearson2': mean('pearson2'), 'fold_r2_debiased': mean('r2_debiased'),
            'fold_bias_share': mean('bias_share'), 'fold_abs_bias': float(np.mean([abs(f['bias']) for f in folds])),
            'trainmean_r2': mean('trainmean_r2'), 'fold_rmse': mean('rmse'),
            'test_std': mean('test_std'), 'pooled_std': float(y.std()),
            'pred_std_ratio': float(np.nanmean([f['pred_std'] / f['test_std'] for f in folds if f['test_std'] > 0])),
            'per_fold': folds}


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--root', required=True)
    a.add_argument('--out', required=True)
    args = a.parse_args()
    root = Path(args.root)
    out = {}
    for exp in sorted(root.glob('paper/*/*/*/*')):
        if not exp.is_dir():
            continue
        r = experiment(exp)
        if r:
            out[exp.relative_to(root).as_posix()] = r
            if len(out) % 50 == 0:
                print('experiments', len(out), flush=True)
    Path(args.out).write_text(json.dumps(out))
    print('wrote', args.out, len(out), 'experiments')


if __name__ == '__main__':
    main()
