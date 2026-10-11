"""IC-A (spec/yieldsat-input-information.md §2): within-field oracle, an input-side bound on within-field R².

For every field season with >= MIN_CELLS valid cells, models are fitted on that field's own cells (inputs -> yield
minus the training-cell mean) and evaluated by spatial block cross-validation inside the field: square blocks of B
cells are assigned at random to 5 folds; each cell is predicted by a model that never saw its block.

Methods per field and block size B:
  gbm    LightGBM on the cell features (monthly S2 24x12, 5x5 neighbourhood S2 mean, DEM, terrain, soil)
  ridge  ridge regression on the same features (standardised, missing -> training mean)
  coord  LightGBM on grid row/col only: pure spatial interpolation, the leakage control

Per field the out-of-block predictions and targets are centred on the field's means; the within-field sums of squares
are written per field (so any subset of fields can be pooled later).

    python yieldsat_input_oracle.py --artifact_root /root/yieldsat_artifacts --out results/input_oracle_fields.jsonl
"""
import argparse
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import warnings

import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")

from dataset.yieldsat_cache import static_valid_mask
from dataset.yieldsat_schema import OPTICAL, STATIC_CHANNELS, TEMPORAL_CHANNELS

CROP = {'C': 'corn', 'S': 'soybean', 'W': 'wheat', 'R': 'rapeseed'}
COUNTRY = {'ARG': 'Argentina', 'BRA': 'Brazil', 'GER': 'Germany', 'URG': 'Uruguay'}
PAIRS = ('ARG-C', 'ARG-S', 'ARG-W', 'BRA-C', 'BRA-S', 'BRA-W', 'GER-R', 'GER-W', 'URG-S')
S2_POS = [TEMPORAL_CHANNELS.index(c) for c in OPTICAL]
STATIC_SEL = [i for i, c in enumerate(STATIC_CHANNELS)
              if not c.startswith('coord') and not c.endswith(('_unc', 'uncertainty')) and 'uncert' not in c]
MIN_CELLS = 200
BLOCKS = (4, 8, 16)
FOLDS = 5
GBM = dict(objective='l2', n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20,
           colsample_bytree=0.5, subsample=0.8, subsample_freq=1, n_jobs=1, verbose=-1)

_A = {}


def _init(artifact_root, country):
    root = Path(artifact_root)
    with np.load(root / 'index' / country / 'rows.npz') as z:
        _A['rows'] = {k: z[k] for k in ('target', 'grid_row', 'grid_col')}
    _A['temporal'] = np.load(root / 'cache' / country / 'temporal.npy', mmap_mode='r')
    _A['static'] = np.load(root / 'cache' / country / 'static.npy', mmap_mode='r')
    _A['nbr'] = np.load(root / 'neighbourhood' / country / 's2_nbr5.npy', mmap_mode='r')


def _features(a, b):
    s2 = np.asarray(_A['temporal'][a:b])[:, :, S2_POS].reshape(b - a, -1).astype(np.float32)
    nbr = np.asarray(_A['nbr'][a:b], dtype=np.float32).reshape(b - a, -1)
    st = np.asarray(_A['static'][a:b]).astype(np.float32)
    st = np.where(static_valid_mask(st), st, np.nan)[:, STATIC_SEL]
    return np.concatenate([s2, nbr, st], axis=1)


def _ridge_fit_predict(Xtr, ytr, Xte, alpha=None):
    mu = np.nanmean(Xtr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, mu)
    Xte = np.where(np.isfinite(Xte), Xte, mu)
    sd = Xtr.std(0)
    keep = sd > 0
    Xtr, Xte = (Xtr[:, keep] - mu[keep]) / sd[keep], (Xte[:, keep] - mu[keep]) / sd[keep]
    if alpha is None:                     # ~600 features vs a few hundred cells: scale the penalty with p / n
        alpha = 100.0 * max(1.0, Xtr.shape[1] / max(1, Xtr.shape[0]))
    A = Xtr.T @ Xtr + alpha * len(ytr) / 100.0 * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ ytr)
    return Xte @ w


def _within_ss(pred, y):
    pc, yc = pred - pred.mean(), y - y.mean()
    return float(np.sum((pc - yc) ** 2)), float(np.sum(yc ** 2)), float(np.dot(pc, yc)), float(np.sum(pc * pc))


def run_field(task):
    import lightgbm as lgb
    pair, f = task
    a, b = f['row_start'], f['row_end']
    y = _A['rows']['target'][a:b].astype(np.float64)
    ok = np.isfinite(y)
    if ok.sum() < MIN_CELLS:
        return None
    X = _features(a, b)[ok]
    y = y[ok]
    gr, gc = _A['rows']['grid_row'][a:b][ok], _A['rows']['grid_col'][a:b][ok]
    coords = np.stack([gr, gc], 1).astype(np.float32)
    out = {'pair': pair, 'field': f['field_shared_name'], 'n': int(len(y))}
    rng = np.random.default_rng(abs(hash(f['field_shared_name'])) % (2 ** 32))
    for B in BLOCKS:
        blk = (gr // B).astype(np.int64) * 100000 + (gc // B)
        ub, inv = np.unique(blk, return_inverse=True)
        if len(ub) < FOLDS:
            continue
        fold = rng.permutation(np.arange(len(ub)) % FOLDS)[inv]
        preds = {m: np.empty(len(y)) for m in ('gbm', 'ridge', 'coord')}
        for k in range(FOLDS):
            te, tr = fold == k, fold != k
            if te.sum() == 0 or tr.sum() < 50:
                for m in preds:
                    preds[m][te] = np.nan
                continue
            ytr = y[tr] - y[tr].mean()
            preds['gbm'][te] = lgb.LGBMRegressor(**GBM).fit(X[tr], ytr).predict(X[te])
            preds['ridge'][te] = _ridge_fit_predict(X[tr], ytr, X[te])
            preds['coord'][te] = lgb.LGBMRegressor(**GBM).fit(coords[tr], ytr).predict(coords[te])
        good = np.all([np.isfinite(p) for p in preds.values()], axis=0)
        if good.sum() < MIN_CELLS // 2:
            continue
        out['B%d' % B] = {m: _within_ss(p[good], y[good]) for m, p in preds.items()}
        out['B%d' % B]['n'] = int(good.sum())
    return out


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--artifact_root', required=True)
    pa.add_argument('--out', required=True)
    pa.add_argument('--workers', type=int, default=20)
    pa.add_argument('--pairs', nargs='+', default=list(PAIRS))
    a = pa.parse_args()
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)['field'] for l in open(a.out)}
    t0 = time.time()
    for country in sorted({COUNTRY[p[:3]] for p in a.pairs}):
        fields = json.loads((Path(a.artifact_root) / 'index' / country / 'fields.json').read_text())
        tasks = []
        for f in fields:
            pair = next((p for p in a.pairs if COUNTRY[p[:3]] == country and CROP[p[4]] == f['crop']), None)
            if pair and f['field_shared_name'] not in done:
                tasks.append((pair, f))
        tasks.sort(key=lambda t: -t[1]['n_rows'])                      # big fields first
        with get_context('fork').Pool(a.workers, initializer=_init, initargs=(a.artifact_root, country)) as pool, \
                open(a.out, 'a') as fh:
            for i, r in enumerate(pool.imap_unordered(run_field, tasks)):
                if r is not None:
                    fh.write(json.dumps(r) + '\n')
                    fh.flush()
                if (i + 1) % 100 == 0:
                    print(country, i + 1, 'of', len(tasks), '%.0f s' % (time.time() - t0), flush=True)
        print(country, 'done', '%.0f s' % (time.time() - t0), flush=True)
    print('ORACLE_DONE')


if __name__ == '__main__':
    main()
