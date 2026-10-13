"""Seed variance (spec/yieldsat-ablations-seeds.md §5): our best model and MMGF, seeds 0-2, per pair x protocol.

Per model, cell (pair x protocol) and seed: pooled and within-field R2 on identical test pixels. A cell is used only
when all seeds of both models are complete. Reported: seed mean and sd per cell; per seed, ours - MMGF; and a paired
field-season cluster bootstrap of the seed-averaged difference (each resample scores all seeds of both models on the
same field seasons, the per-seed differences are averaged).

    python results/seed_table.py <results root containing paper/> <out.json>
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.join(sys.argv[1] if len(sys.argv) > 1 else '/data/YieldSAT/yieldsat_results/protocol', 'paper')
OUT = sys.argv[2] if len(sys.argv) > 2 else 'seed_table.json'
OURS, MMGF = 'ours-p3nbr-dense-fc-rel10', 'paper_mmgf'
SEEDS = (0, 1, 2)
PAIRS = ['ARG-C', 'ARG-S', 'ARG-W', 'BRA-C', 'BRA-S', 'BRA-W', 'GER-R', 'GER-W', 'URG-S']
PROTOS = {'cv10': 'CV10', 'loyo': 'LOYO', 'loro': 'LORO'}
B = 2000
METRICS = ('pooled_r2', 'within_r2')


def load(grp, tag, seed, pair):
    fs = sorted(glob.glob('%s/s2_adm/%s_seed%d/%s/fold*/test_predictions.npz' % (grp, tag, seed, pair)))
    if not fs:
        return None, 0
    P, Y, S = [], [], []
    for f in fs:
        z = np.load(f)
        ok = np.isfinite(z['pred']) & np.isfinite(z['target'])
        names = z['season_names'].astype(str)[z['season'][ok]]
        fold = os.path.basename(os.path.dirname(f))
        P.append(z['pred'][ok].astype(np.float64)); Y.append(z['target'][ok].astype(np.float64))
        S.append(np.char.add(fold + ':', names))
    return (np.concatenate(P), np.concatenate(Y), np.concatenate(S)), len(fs)


def field_stats(p, y, s, keys):
    k_of = {k: i for i, k in enumerate(np.unique(s))}
    _, inv = np.unique(s, return_inverse=True)
    idx = np.array([k_of[k] for k in keys])
    n = np.bincount(inv).astype(float)
    sy, sp = np.bincount(inv, y), np.bincount(inv, p)
    yc, pc = y - (sy / n)[inv], p - (sp / n)[inv]
    st = dict(n=n, sy=sy, syy=np.bincount(inv, y * y), sse=np.bincount(inv, (p - y) ** 2),
              wsse=np.bincount(inv, (pc - yc) ** 2), wsst=np.bincount(inv, yc ** 2))
    return {k: v[idx] for k, v in st.items()}


def summarize(st, W):
    """W (R, F) field-season weights -> {metric: (R,)}."""
    n, sy, syy = W @ st['n'], W @ st['sy'], W @ st['syy']
    return {'pooled_r2': 1 - (W @ st['sse']) / (syy - sy * sy / n), 'within_r2': 1 - (W @ st['wsse']) / (W @ st['wsst'])}


rng = np.random.default_rng(0)
res, boot = {}, {}
for g, proto in PROTOS.items():
    groups = sorted(glob.glob('%s/%s_*noval_s0' % (ROOT, g)), key=lambda d: ('province' in d, d))
    for pair in PAIRS:
        grp = next((d for d in groups if os.path.isdir('%s/s2_adm/%s_seed0/%s' % (d, OURS, pair))), None)
        if grp is None:
            continue
        ref, nb = load(grp, OURS, 0, pair)
        keys = np.unique(ref[2])
        stats, row = {}, {'group': os.path.basename(grp), 'folds': nb}
        for tag in (OURS, MMGF):
            for seed in SEEDS:
                d, nf = (ref, nb) if (tag, seed) == (OURS, 0) else load(grp, tag, seed, pair)
                if d is None or nf < nb or set(np.unique(d[2])) != set(keys):
                    stats = None
                    break
                stats[(tag, seed)] = field_stats(*d, keys)
            if stats is None:
                break
        if stats is None:
            res['%s|%s' % (proto, pair)] = dict(row, incomplete=True)
            continue
        one = np.ones((1, len(keys)))
        W = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=B).astype(float)
        for tag in (OURS, MMGF):
            per_seed = {m: [float(summarize(stats[(tag, s)], one)[m][0]) for s in SEEDS] for m in METRICS}
            row[tag] = {m: {'seeds': v, 'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1))}
                        for m, v in per_seed.items()}
        bs = {m: np.mean([summarize(stats[(OURS, s)], W)[m] - summarize(stats[(MMGF, s)], W)[m] for s in SEEDS], 0)
              for m in METRICS}
        row['diff'] = {m: {'per_seed': [a - b for a, b in zip(row[OURS][m]['seeds'], row[MMGF][m]['seeds'])],
                           'mean': row[OURS][m]['mean'] - row[MMGF][m]['mean'],
                           'ci': [float(np.percentile(bs[m], 2.5)), float(np.percentile(bs[m], 97.5))]}
                       for m in METRICS}
        boot['%s|%s' % (proto, pair)] = bs
        res['%s|%s' % (proto, pair)] = row
        print(proto, pair, json.dumps({t: {m: [round(x, 3) for x in row[t][m]['seeds']] for m in METRICS}
                                       for t in (OURS, MMGF)}), flush=True)
# 9-pair means per seed (protocols with every pair complete)
for proto in PROTOS.values():
    rows = [res.get('%s|%s' % (proto, p), {}) for p in PAIRS]
    if not all(OURS in r for r in rows):
        continue
    out = {}
    for tag in (OURS, MMGF):
        out[tag] = {}
        for m in METRICS:
            v = np.mean([r[tag][m]['seeds'] for r in rows], 0)
            out[tag][m] = {'seeds': v.tolist(), 'mean': float(v.mean()), 'sd': float(v.std(ddof=1))}
    mb = {m: np.mean([boot['%s|%s' % (proto, p)][m] for p in PAIRS], 0) for m in METRICS}
    out['diff'] = {m: {'per_seed': [a - b for a, b in zip(out[OURS][m]['seeds'], out[MMGF][m]['seeds'])],
                       'mean': out[OURS][m]['mean'] - out[MMGF][m]['mean'],
                       'ci': [float(np.percentile(mb[m], 2.5)), float(np.percentile(mb[m], 97.5))]} for m in METRICS}
    res['%s|MEAN' % proto] = out
    print(proto, 'MEAN', json.dumps(out['diff']), flush=True)
json.dump(res, open(OUT, 'w'), indent=1, default=float)
print('SEEDDONE', OUT)
