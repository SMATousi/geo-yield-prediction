"""IC-D (spec/yieldsat-input-information.md §5): mutual information between the within-field input deviation and the
within-field yield, by the Kraskov-Stoegbauer-Grassberger estimator (KSG, algorithm 1), per pair.

Per cell: x = PCA (d components, fitted per pair) of the field-centred S2 series (monthly cache, 24 slots x 12 bands;
missing values -> the field mean, i.e. 0 after centring) plus the field-centred DEM, slope and curvature; y = yield minus
the field mean. All dimensions are standardised; the joint space uses the max-norm.
  pooled     one sample of N cells across all fields of the pair: information that holds across fields
  per_field  each field with >= 500 cells (up to 2,000 sampled): information within one field; the median over fields
R2-equivalent bound: 1 - exp(-2 I) (exact for jointly Gaussian variables). Sensitivity over k and d.

    python yieldsat_input_mi.py --artifact_root /root/yieldsat_artifacts --out results/input_mi.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import digamma
from sklearn.neighbors import KDTree

from dataset.yieldsat_cache import static_valid_mask
from dataset.yieldsat_schema import OPTICAL, STATIC_CHANNELS, TEMPORAL_CHANNELS

CROP = {'C': 'corn', 'S': 'soybean', 'W': 'wheat', 'R': 'rapeseed'}
COUNTRY = {'ARG': 'Argentina', 'BRA': 'Brazil', 'GER': 'Germany', 'URG': 'Uruguay'}
PAIRS = ('ARG-C', 'ARG-S', 'ARG-W', 'BRA-C', 'BRA-S', 'BRA-W', 'GER-R', 'GER-W', 'URG-S')
S2_POS = [TEMPORAL_CHANNELS.index(c) for c in OPTICAL]
TERRAIN_POS = [STATIC_CHANNELS.index(c) for c in ('dem', 'slope', 'curvature')]


def ksg(x, y, k=5):
    """I(X; Y) in nats, KSG algorithm 1; x (n, dx), y (n, 1), both standardised."""
    n = len(y)
    xy = np.hstack([x, y])
    eps = KDTree(xy, metric='chebyshev').query(xy, k=k + 1)[0][:, -1]
    r = np.maximum(eps - 1e-12, 0)
    nx = KDTree(x, metric='chebyshev').query_radius(x, r, count_only=True) - 1
    ny = KDTree(y, metric='chebyshev').query_radius(y, r, count_only=True) - 1
    return float(digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1)))


def field_features(temporal, static, target, a, b):
    """-> (X_s2 centred (n, 288), terrain centred (n, 3), y centred (n,)) over the field's valid cells."""
    y = target[a:b].astype(np.float64)
    ok = np.isfinite(y)
    s2 = np.asarray(temporal[a:b])[:, :, S2_POS].astype(np.float64)[ok].reshape(ok.sum(), -1)
    mu = np.nanmean(s2, axis=0)
    s2 = np.where(np.isfinite(s2), s2 - np.where(np.isfinite(mu), mu, 0.0), 0.0)
    st = np.asarray(static[a:b]).astype(np.float64)
    st = np.where(static_valid_mask(st), st, np.nan)[ok][:, TERRAIN_POS]
    tm = np.nanmean(st, axis=0)
    st = np.where(np.isfinite(st), st - np.where(np.isfinite(tm), tm, 0.0), 0.0)
    yy = y[ok]
    return s2, st, yy - yy.mean()


def standardise(a):
    sd = a.std(0)
    return (a - a.mean(0)) / np.where(sd > 0, sd, 1.0)


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--artifact_root', required=True)
    pa.add_argument('--out', required=True)
    pa.add_argument('--n_pooled', type=int, default=20000)
    pa.add_argument('--ks', type=int, nargs='+', default=[3, 5, 10])
    pa.add_argument('--dims', type=int, nargs='+', default=[4, 8, 16])
    a = pa.parse_args()
    root = Path(a.artifact_root)
    rng = np.random.default_rng(0)
    res = {}
    for pair in PAIRS:
        country = COUNTRY[pair[:3]]
        fields = [f for f in json.loads((root / 'index' / country / 'fields.json').read_text())
                  if f['crop'] == CROP[pair[4]] and f['n_rows'] >= 200]
        temporal = np.load(root / 'cache' / country / 'temporal.npy', mmap_mode='r')
        static = np.load(root / 'cache' / country / 'static.npy', mmap_mode='r')
        with np.load(root / 'index' / country / 'rows.npz') as z:
            target = z['target']
        per_field_cap = max(50, a.n_pooled // len(fields) + 1)
        S2, TR, Y, FID = [], [], [], []
        for i, f in enumerate(fields):
            s2, tr, y = field_features(temporal, static, target, f['row_start'], f['row_end'])
            if len(y) < 50:
                continue
            take = rng.choice(len(y), size=min(len(y), max(per_field_cap, 2000)), replace=False)
            S2.append(s2[take]); TR.append(tr[take]); Y.append(y[take]); FID.append(np.full(len(take), i))
        S2, TR, Y, FID = np.concatenate(S2), np.concatenate(TR), np.concatenate(Y), np.concatenate(FID)
        Y = Y + 1e-9 * rng.standard_normal(len(Y))                  # break ties of gridded yields
        mu, sd = S2.mean(0), S2.std(0)
        Z = (S2 - mu) / np.where(sd > 0, sd, 1.0)
        _, _, vt = np.linalg.svd(Z[rng.choice(len(Z), size=min(len(Z), 20000), replace=False)], full_matrices=False)
        row = {'fields': len(fields), 'pooled': {}, 'per_field': {}}
        pooled_idx = rng.choice(len(Y), size=min(len(Y), a.n_pooled), replace=False)
        for d in a.dims:
            X = np.hstack([Z @ vt[:d].T, TR])
            for k in a.ks:
                i_p = ksg(standardise(X[pooled_idx]), standardise(Y[pooled_idx])[:, None], k)
                row['pooled']['d%d_k%d' % (d, k)] = {'mi': i_p, 'r2_eq': 1 - np.exp(-2 * max(i_p, 0))}
            vals = []
            for fi in np.unique(FID):
                sel = FID == fi
                if sel.sum() < 500:
                    continue
                vals.append(ksg(standardise(X[sel]), standardise(Y[sel])[:, None], 5))
            if vals:
                v = np.array(vals)
                row['per_field']['d%d_k5' % d] = {'fields': len(v), 'mi_median': float(np.median(v)),
                                                  'r2_eq_median': float(1 - np.exp(-2 * max(np.median(v), 0)))}
        res[pair] = row
        print(pair, json.dumps({'pooled_d8_k5': row['pooled'].get('d8_k5'), 'per_field_d8': row['per_field'].get('d8_k5')}),
              flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print('MI_DONE')


if __name__ == '__main__':
    main()
