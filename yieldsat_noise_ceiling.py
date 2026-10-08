"""Within-field noise ceiling from the yield maps alone (spec/yieldsat-field-context.md §3).

Per field season the targets are centred on their mean and placed on the 10 m grid. The empirical
semivariogram along grid rows and columns, gamma(h) = 0.5 * mean((y(x) - y(x + h))^2) for h = 1..30 cells,
is extrapolated linearly from h = 1..3 to h = 0: the nugget, the part of the within-field variance that
is uncorrelated even between adjacent cells (harvester noise, sub-cell variation). No input can predict
it, so 1 - nugget / within-field variance bounds the within-field R2 of any model.

    python yieldsat_noise_ceiling.py --artifact_root /root/yieldsat_artifacts --out results/noise_ceiling.json
"""
import argparse
import collections
import json

import numpy as np

from dataset.yieldsat_source import load_country_index

CROP = {'C': 'corn', 'S': 'soybean', 'W': 'wheat', 'R': 'rapeseed'}
COUNTRY = {'ARG': 'Argentina', 'BRA': 'Brazil', 'GER': 'Germany', 'URG': 'Uruguay'}
PAIRS = ('ARG-C', 'ARG-S', 'ARG-W', 'BRA-C', 'BRA-S', 'BRA-W', 'GER-R', 'GER-W', 'URG-S')
LAGS = tuple(range(1, 31))          # fit uses h = 1..3; longer lags for the plots (to 300 m)


def field_variogram(r, c, y):
    """-> (n, sum of centred y^2, {lag: (sum of half squared differences, pair count)})."""
    y = y - y.mean()
    g = np.full((r.max() - r.min() + 1, c.max() - c.min() + 1), np.nan)
    g[r - r.min(), c - c.min()] = y
    out = {}
    for h in LAGS:
        d = np.concatenate([(g[h:, :] - g[:-h, :]).ravel(), (g[:, h:] - g[:, :-h]).ravel()])
        d = d[np.isfinite(d)]
        out[h] = (0.5 * float(np.sum(d * d)), len(d))
    return len(y), float(np.sum(y * y)), out


def ceiling(acc):
    var = acc['ss'] / acc['n']
    gam = np.array([acc['g'][h] / acc['k'][h] if acc['k'][h] else np.nan for h in LAGS])
    slope, nugget = np.polyfit(np.array(LAGS[:3], float), gam[:3], 1)
    nugget = max(float(nugget), 0.0)
    return {'within_var': var, 'gamma': dict(zip(map(str, LAGS), (gam / var).round(4).tolist())),
            'pairs': dict(zip(map(str, LAGS), [int(acc['k'][h]) for h in LAGS])),
            'nugget_frac': nugget / var, 'fit_slope_frac': float(slope) / var,
            'ceiling_within_r2': 1 - nugget / var, 'fields': acc['fields']}


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--artifact_root', required=True)
    pa.add_argument('--out', required=True)
    a = pa.parse_args()
    res = {}
    for country in sorted(set(COUNTRY.values())):
        idx = load_country_index(a.artifact_root, None, country, False)
        rows = idx['rows']
        accs = collections.defaultdict(lambda: {'n': 0, 'ss': 0.0, 'g': collections.Counter(),
                                                'k': collections.Counter(), 'fields': 0})
        per_field = collections.defaultdict(list)
        for f in idx['fields']:
            pair = next((p for p in PAIRS if COUNTRY[p[:3]] == country and CROP[p[4]] == f['crop']), None)
            if pair is None:
                continue
            s = slice(f['row_start'], f['row_end'])
            y = rows['target'][s].astype(np.float64)
            ok = np.isfinite(y)
            if ok.sum() < 50:
                continue
            n, ss, gv = field_variogram(rows['grid_row'][s][ok], rows['grid_col'][s][ok], y[ok])
            acc = accs[pair]
            acc['n'] += n; acc['ss'] += ss; acc['fields'] += 1
            for h, (gs, k) in gv.items():
                acc['g'][h] += gs; acc['k'][h] += k
            single = {'n': n, 'ss': ss, 'g': {h: v[0] for h, v in gv.items()}, 'k': {h: v[1] for h, v in gv.items()},
                      'fields': 1}
            if all(single['k'][h] > 0 for h in LAGS[:3]) and ss > 0:
                per_field[pair].append(ceiling(single)['nugget_frac'])
        for pair, acc in sorted(accs.items()):
            res[pair] = ceiling(acc)
            res[pair]['median_field_nugget_frac'] = float(np.median(per_field[pair]))
            print(pair, json.dumps(res[pair]), flush=True)
    m = {k: float(np.mean([res[p][k] for p in PAIRS])) for k in ('nugget_frac', 'ceiling_within_r2')}
    res['MEAN'] = m
    print('MEAN', json.dumps(m))
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
