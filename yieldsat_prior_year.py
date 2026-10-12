"""RX-3 (spec/yieldsat-review-experiments.md): prior-year field calibration under LOYO.

For every LOYO test field season (field F, year t) whose physical field has an earlier season t' < t, last year's
observed yield map y_t' (the farmer's data) and the model's out-of-sample prediction for that earlier season are
aligned to this year's cells by geographic cell centre. Predictors (within-field anomaly space):
  M0  model                      yhat'_t
  M1  persistence                a * s * z(y_t')
  M2  blend                      b1 yhat'_t + b2 s z(y_t')
  M3  residual transfer          yhat'_t + c s z(y'_t' - yhat'_t')
  M4  blend + residual           b1 yhat'_t + b2 s z(y_t') + b3 s z(y'_t' - yhat'_t')
  M5  per-field amplitude        beta_F yhat'_t, beta_F = regression slope of y'_t' on yhat'_t' (clipped to [0, 3])
with s = std(yhat'_t) (no labels needed) and z() the within-field z-score. a, b, c are least-squares coefficients
fitted, for test year t, only on target seasons of the same pair whose current year != t (training years), so no
label of year t is used. Scored with within-field R2 on the seasons that have an earlier season.

    python yieldsat_prior_year.py --results_root /root/yieldsat_backups/protocol_results \\
        --artifact_root /root/yieldsat_artifacts --out results/prior_year.json
"""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

TAG = 'ours-p3nbr-dense-fc-rel10'
GROUP = 'loyo_na_noval_s0'
PAIRS = ('ARG-C', 'ARG-S', 'ARG-W', 'BRA-C', 'BRA-S', 'BRA-W', 'GER-R', 'GER-W', 'URG-S')
MIN_CELLS = 200
METHODS = ('M0', 'M1', 'M2', 'M3', 'M4', 'M5')


def centres(rows, cols, t):
    a, b, c, d, e, f = t
    x = c + a * (cols + 0.5) + b * (rows + 0.5)
    y = f + d * (cols + 0.5) + e * (rows + 0.5)
    return np.stack([x, y], 1)


def zscore(v):
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def load_predictions(root):
    """-> {season name: (row, col, target, pred)} from every pair's LOYO test folds (each season is out of sample)."""
    out = {}
    for f in glob.glob('%s/paper/%s/s2_adm/%s_seed0/*/fold*/test_predictions.npz' % (root, GROUP, TAG)):
        z = np.load(f)
        names = z['season_names'].astype(str)
        for si in np.unique(z['season']):
            sel = (z['season'] == si) & np.isfinite(z['target']) & np.isfinite(z['pred'])
            out[names[si]] = (z['grid_row'][sel], z['grid_col'][sel], z['target'][sel].astype(float), z['pred'][sel].astype(float))
    return out


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--results_root', required=True)
    pa.add_argument('--artifact_root', required=True)
    pa.add_argument('--out', required=True)
    a = pa.parse_args()
    geo = {x['field_shared_name']: x for g in glob.glob('%s/geometry/fields_geometry_*.json' % a.artifact_root)
           for x in json.load(open(g))}
    by_field = defaultdict(list)
    for x in geo.values():
        by_field[x['physical_field_id']].append(x)
    pred = load_predictions(a.results_root)
    crop_code = {'corn': 'C', 'soybean': 'S', 'wheat': 'W', 'rapeseed': 'R'}
    cc = {'Argentina': 'ARG', 'Brazil': 'BRA', 'Germany': 'GER', 'Uruguay': 'URG'}

    # 1. assemble aligned (current, prior) cases
    cases = []
    for name, (r, c, y, p) in pred.items():
        g = geo.get(name)
        if g is None:
            continue
        earlier = [x for x in by_field[g['physical_field_id']] if x['year'] < g['year'] and x['field_shared_name'] in pred]
        if not earlier:
            continue
        prev = max(earlier, key=lambda x: x['year'])
        pr, pc_, py, pp = pred[prev['field_shared_name']]
        tree = cKDTree(centres(pr, pc_, prev['transform']))
        dist, j = tree.query(centres(r, c, g['transform']), distance_upper_bound=5.0)
        ok = np.isfinite(dist)
        if ok.sum() < MIN_CELLS:
            continue
        j = j[ok]
        y_t, p_t = y[ok], p[ok]
        y_prev, p_prev = py[j], pp[j]
        cases.append({'name': name, 'pair': '%s-%s' % (cc[g['country']], crop_code[g['crop']]), 'year': g['year'],
                      'same_crop': prev['crop'] == g['crop'], 'gap': g['year'] - prev['year'],
                      'y': y_t - y_t.mean(), 'm': p_t - p_t.mean(),
                      'zp': zscore(y_prev), 'zr': zscore((y_prev - y_prev.mean()) - (p_prev - p_prev.mean())),
                      'beta': float(np.clip(np.cov(y_prev - y_prev.mean(), p_prev - p_prev.mean())[0, 1]
                                            / max(np.var(p_prev), 1e-12), 0, 3))})
    print('cases', len(cases), 'cells', sum(len(x['y']) for x in cases), flush=True)

    # 2. features, coefficients fitted on training years only, within-field R2
    def feats(x, method):
        s = x['m'].std()
        if method == 'M1':
            return np.stack([s * x['zp']], 1)
        if method == 'M2':
            return np.stack([x['m'], s * x['zp']], 1)
        if method == 'M3':
            return np.stack([s * x['zr']], 1)
        if method == 'M4':
            return np.stack([x['m'], s * x['zp'], s * x['zr']], 1)

    res = {}
    for pair in PAIRS:
        pc = [x for x in cases if x['pair'] == pair]
        if not pc:
            continue
        sums = {m: defaultdict(lambda: np.zeros(2)) for m in METHODS}            # subset -> [sse, sst]
        per_case = []                                                           # (sse per method, sst) for the bootstrap
        coefs = defaultdict(list)
        for x in pc:
            train = [o for o in pc if o['year'] != x['year']]
            preds = {'M0': x['m'], 'M5': x['beta'] * x['m']}
            for m in ('M1', 'M2', 'M3', 'M4'):
                if not train:
                    preds[m] = x['m']
                    continue
                X = np.concatenate([feats(o, m) for o in train])
                target = np.concatenate([o['y'] - (o['m'] if m == 'M3' else 0) for o in train])
                w = np.linalg.lstsq(X, target, rcond=None)[0]
                coefs[m].append(w.tolist())
                preds[m] = (x['m'] if m == 'M3' else 0) + feats(x, m) @ w
            subsets = ['all', 'same_crop' if x['same_crop'] else 'rotation'] + (['previous_year'] if x['gap'] == 1 else [])
            per_case.append(([float(np.sum(((preds[m] - preds[m].mean()) - x['y']) ** 2)) for m in METHODS],
                             float(np.sum(x['y'] ** 2))))
            for m, v in preds.items():
                v = v - v.mean()
                for sub in subsets:
                    sums[m][sub] += [np.sum((v - x['y']) ** 2), np.sum(x['y'] ** 2)]
        res[pair] = {'seasons': len(pc), 'same_crop': sum(x['same_crop'] for x in pc),
                     'previous_year': sum(x['gap'] == 1 for x in pc),
                     'within_r2': {m: {sub: float(1 - v[0] / v[1]) for sub, v in sums[m].items()} for m in METHODS},
                     'sums': {m: {sub: v.tolist() for sub, v in sums[m].items()} for m in METHODS},
                     'coef_mean': {m: np.mean(v, 0).tolist() for m, v in coefs.items()},
                     'per_case': per_case}
        print(pair, res[pair]['seasons'], {m: round(res[pair]['within_r2'][m]['all'], 3) for m in METHODS}, flush=True)
    tot = {m: {} for m in METHODS}
    for m in METHODS:
        for sub in ('all', 'same_crop', 'rotation', 'previous_year'):
            s = np.sum([res[p]['sums'][m][sub] for p in res if sub in res[p]['sums'][m]], 0)
            if np.ndim(s) and s[1] > 0:
                tot[m][sub] = float(1 - s[0] / s[1])
    res['POOLED'] = tot
    res['MEAN_PAIRS'] = {m: float(np.mean([res[p]['within_r2'][m]['all'] for p in PAIRS if p in res])) for m in METHODS}
    # paired field-season bootstrap of each method vs M0: mean over pairs of the within-field R2 difference
    rng = np.random.default_rng(0)
    present = [p for p in PAIRS if p in res]
    draws = {m: [] for m in METHODS[1:]}
    for _ in range(2000):
        d = {m: [] for m in METHODS[1:]}
        for p in present:
            pc_ = res[p]['per_case']
            idx = rng.integers(0, len(pc_), len(pc_))
            sse = np.sum([pc_[i][0] for i in idx], 0)
            sst = np.sum([pc_[i][1] for i in idx])
            for k, m in enumerate(METHODS[1:], start=1):
                d[m].append((1 - sse[k] / sst) - (1 - sse[0] / sst))
        for m in METHODS[1:]:
            draws[m].append(np.mean(d[m]))
    res['DELTA_VS_M0'] = {m: [float(res['MEAN_PAIRS'][m] - res['MEAN_PAIRS']['M0']), float(np.percentile(v, 2.5)),
                              float(np.percentile(v, 97.5))] for m, v in draws.items()}
    print('DELTA_VS_M0', json.dumps({m: [round(x, 3) for x in v] for m, v in res['DELTA_VS_M0'].items()}))
    print('POOLED', json.dumps({m: {k: round(v, 3) for k, v in d.items()} for m, d in tot.items()}))
    print('MEAN_PAIRS', json.dumps({m: round(v, 3) for m, v in res['MEAN_PAIRS'].items()}))
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
