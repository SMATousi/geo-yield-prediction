"""Per-field within-field R2 of the paper's main comparisons (the typical field, each field season weighted equally).

The paper's metric pools cells over all test fields (eq. within). This scores each field season on its own, on
identical test pixels: median per-field R2 and its IQR for both models, the median of the paired per-field gain,
and the share of field seasons that improve, with a field-season bootstrap (resampled within each pair) for the
last two. Seeds pooled per field (1 - sum_s wsse_s / sum_s wsst_s, i.e. the mean over seeds). Field seasons with
fewer than MIN_CELLS labelled test cells are left out (per-field R2 of tiny fields is unstable); the counts at
other thresholds are kept in the JSON.

    python results/field_cmp.py <results root containing paper/ and own/> <out.json>
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else '/data/YieldSAT/yieldsat_results/protocol'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'field_cmp.json'
OURS = 'ours-p3nbr-dense-fc-rel10'
PAIRS = ['ARG-C', 'ARG-S', 'ARG-W', 'BRA-C', 'BRA-S', 'BRA-W', 'GER-R', 'GER-W', 'URG-S']
OWN_SUB = os.environ.get('FC_OWN_SUB', 'own')                  # own_v2 once the v2 runs are done
#: (name, results sub-root, baseline, variant, pairs, protocols, seeds) -- as in the paper's tables
COMPARISONS = [
    ('vs_mmgf', 'paper', 'paper_mmgf', OURS, PAIRS, ('cv10', 'loyo', 'loro'), (0,)),
    ('vs_tabm', 'paper', 'tabm-f0-dense-bs2048', OURS, PAIRS, ('cv10', 'loyo', 'loro'), (0,)),
    ('vs_base', 'paper', 'ours-p3nbr-dense', OURS, PAIRS, ('cv10', 'loyo', 'loro'), (0,)),
    ('ps_yieldsat', 'paper', OURS, OURS + '-ps3x3-pswin', ['ARG-S', 'ARG-C', 'BRA-C'], ('cv10', 'loyo'), (0, 1, 2)),
    ('ps_own', OWN_SUB, OURS, OURS + '-ps3x3-pswin', ['MO-S', 'MO-C'], ('cv10', 'loyo'), (0, 1, 2)),
]
MIN_CELLS, THRESHOLDS, B = 1000, (50, 1000, 3000), 2000
SEL = os.environ.get('FC_SEL', 'noval')


def load(grp, tag, pair, seed):
    fs = sorted(glob.glob('%s/s2_adm/%s_seed%d/%s/fold*/test_predictions.npz' % (grp, tag, seed, pair)))
    if not fs:
        return None, 0
    P, Y, S = [], [], []
    for f in fs:
        z = np.load(f)
        ok = np.isfinite(z['pred']) & np.isfinite(z['target'])
        names = (z['season_names'].astype(str)[z['season'][ok]] if 'season_names' in z.files
                 else np.asarray(z['season'][ok]).astype(str))
        P.append(z['pred'][ok].astype(np.float64)); Y.append(z['target'][ok].astype(np.float64))
        S.append(np.char.add(os.path.basename(os.path.dirname(f)) + ':', names))      # ids are per fold
    return (np.concatenate(P), np.concatenate(Y), np.concatenate(S)), len(fs)


def field_stats(p, y, s):
    keys, inv = np.unique(s, return_inverse=True)
    n = np.bincount(inv).astype(float)
    yc = y - (np.bincount(inv, y) / n)[inv]
    pc = p - (np.bincount(inv, p) / n)[inv]
    return keys, n, np.bincount(inv, (pc - yc) ** 2), np.bincount(inv, yc ** 2)


def per_field_r2(grp, tag, pair, seeds):
    """{field key: (test cells, R2 with seeds pooled)}, or None if a seed is missing."""
    acc = {}
    for sd in seeds:
        d, _ = load(grp, tag, pair, sd)
        if d is None:
            return None
        for k, n, wsse, wsst in zip(*field_stats(*d)):
            a = acc.setdefault(k, [n, 0.0, 0.0, 0])
            a[1] += wsse; a[2] += wsst; a[3] += 1
    return {k: (a[0], 1 - a[1] / a[2]) for k, a in acc.items() if a[3] == len(seeds) and a[2] > 0}


def summary(rb, rv, pair_of, rng):
    gain = rv - rb
    out = {'fields': int(len(rb)),
           'base_median': float(np.median(rb)), 'base_iqr': [float(x) for x in np.percentile(rb, [25, 75])],
           'var_median': float(np.median(rv)), 'var_iqr': [float(x) for x in np.percentile(rv, [25, 75])],
           'median_gain': float(np.median(gain)), 'share_improved': float(np.mean(gain > 0))}
    strata = [np.flatnonzero(pair_of == p) for p in np.unique(pair_of)]
    bm, bs = np.empty(B), np.empty(B)
    for b in range(B):
        g = gain[np.concatenate([rng.choice(ix, len(ix)) for ix in strata])]
        bm[b], bs[b] = np.median(g), np.mean(g > 0)
    out['ci_median_gain'] = [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
    out['ci_share_improved'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


rng = np.random.default_rng(0)
res = {}
for name, sub, base, var, pairs, protos, seeds in COMPARISONS:
    for g in protos:
        # LORO is farm-level for most pairs and province-level for the Argentine ones (first group holding the pair)
        groups = sorted(glob.glob('%s/%s/%s_*_%s_s0' % (ROOT, sub, g, SEL)), key=lambda d: ('province' in d, d))
        rows, per_pair = [], {}
        for pair in pairs:
            grp = next((d for d in groups if glob.glob('%s/s2_adm/%s_seed0/%s' % (d, base, pair))
                        and glob.glob('%s/s2_adm/%s_seed0/%s' % (d, var, pair))), None)
            fb = grp and per_field_r2(grp, base, pair, seeds)
            fv = grp and per_field_r2(grp, var, pair, seeds)
            if not fb or not fv or set(fb) != set(fv):
                per_pair[pair] = {'incomplete': True}
                continue
            ks = sorted(fb)
            rows += [(pair, fb[k][0], fb[k][1], fv[k][1]) for k in ks]
            n = np.array([fb[k][0] for k in ks]); m = n >= MIN_CELLS
            per_pair[pair] = {'fields': int(m.sum()),
                              'base_median': float(np.median(np.array([fb[k][1] for k in ks])[m])),
                              'var_median': float(np.median(np.array([fv[k][1] for k in ks])[m])),
                              'share_improved': float(np.mean((np.array([fv[k][1] - fb[k][1] for k in ks]))[m] > 0))}
        if any(v.get('incomplete') for v in per_pair.values()):
            res['%s|%s' % (name, g)] = {'incomplete': [p for p, v in per_pair.items() if v.get('incomplete')]}
            print(name, g, 'incomplete', flush=True)
            continue
        pr = np.array([r[0] for r in rows]); n = np.array([r[1] for r in rows])
        rb = np.array([r[2] for r in rows]); rv = np.array([r[3] for r in rows])
        m = n >= MIN_CELLS
        res['%s|%s' % (name, g)] = {'base': base, 'variant': var, 'seeds': list(seeds), 'min_cells': MIN_CELLS,
                                    **summary(rb[m], rv[m], pr[m], rng), 'per_pair': per_pair,
                                    'by_threshold': {t: {'fields': int((n >= t).sum()),
                                                         'median_gain': float(np.median((rv - rb)[n >= t])),
                                                         'share_improved': float(np.mean((rv - rb)[n >= t] > 0))}
                                                     for t in THRESHOLDS}}
        print(name, g, json.dumps({k: v for k, v in res['%s|%s' % (name, g)].items()
                                   if k not in ('per_pair', 'by_threshold')}), flush=True)
json.dump(res, open(OUT, 'w'), indent=1, default=float)
print('FIELDCMPDONE', OUT)
