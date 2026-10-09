"""PM-05 (spec/yieldsat-paper-models.md): pooled and within-field R2 of the paper's best models and ours, per
pair x protocol, on the same test pixels; paired field-season bootstrap of our best model vs the best paper model
of each row (chosen on the same test data, which favours the paper models).

    python results/paper_models_table.py <results root containing paper/> <out.json>
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.join(sys.argv[1] if len(sys.argv) > 1 else '/data/YieldSAT/yieldsat_results/protocol', 'paper')
OUT = sys.argv[2] if len(sys.argv) > 2 else 'paper_models_table.json'
PAPER = ['paper_3dlstm', 'paper_3dconvlstm', 'paper_aff', 'paper_mmgf']
OURS = ['ours-p3nbr-dense', 'ours-p3nbr-dense-fc-rel10']
BEST_OURS = 'ours-p3nbr-dense-fc-rel10'
PAIRS = ['ARG-C', 'ARG-S', 'ARG-W', 'BRA-C', 'BRA-S', 'BRA-W', 'GER-R', 'GER-W', 'URG-S']
PROTOS = {'cv10': 'CV10', 'loyo': 'LOYO', 'loro': 'LORO'}
MIN_PIX, B = 50, 2000


def load(grp, tag, pair):
    fs = sorted(glob.glob('%s/s2_adm/%s_seed0/%s/fold*/test_predictions.npz' % (grp, tag, pair)))
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
              wsse=np.bincount(inv, (pc - yc) ** 2), wsst=np.bincount(inv, yc ** 2),
              spy=np.bincount(inv, pc * yc), spp=np.bincount(inv, pc * pc))
    return {k: v[idx] for k, v in st.items()}


def summarize(st, w=None):
    w = np.ones_like(st['n']) if w is None else w
    n, sy, syy = (w * st['n']).sum(), (w * st['sy']).sum(), (w * st['syy']).sum()
    return {'pooled_r2': 1 - (w * st['sse']).sum() / (syy - sy * sy / n),
            'within_r2': 1 - (w * st['wsse']).sum() / (w * st['wsst']).sum()}


def per_field(st):
    m = (st['n'] >= MIN_PIX) & (st['wsst'] > 0) & (st['spp'] > 0)
    return {'median_field_r': float(np.median(st['spy'][m] / np.sqrt(st['spp'][m] * st['wsst'][m]))),
            'median_var_ratio': float(np.median(np.sqrt(st['spp'][m] / st['wsst'][m]))), 'fields': int(m.sum())}


rng = np.random.default_rng(0)
res = {}
for g, proto in PROTOS.items():
    groups = sorted(glob.glob('%s/%s_*noval_s0' % (ROOT, g)), key=lambda d: ('province' in d, d))
    for pair in PAIRS:
        grp = next((d for d in groups if os.path.isdir('%s/s2_adm/%s_seed0/%s' % (d, OURS[0], pair))), None)
        if grp is None:
            continue
        base, nb = load(grp, OURS[0], pair)
        keys = np.unique(base[2])
        row, stats = {'group': os.path.basename(grp), 'folds': nb}, {}
        for tag in OURS + PAPER:
            d, nf = load(grp, tag, pair)
            if d is None or nf < nb or set(np.unique(d[2])) != set(keys):
                row[tag] = {'folds': nf, 'incomplete': True}
                continue
            stats[tag] = field_stats(*d, keys)
            row[tag] = {**summarize(stats[tag]), **per_field(stats[tag]), 'cells': int(len(d[0]))}
        done = [t for t in PAPER if t in stats]
        if BEST_OURS in stats and done:
            W = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=B).astype(float)
            for crit in ('pooled_r2', 'within_r2'):
                bp = max(done, key=lambda t: row[t][crit])
                dl = {m: np.array([summarize(stats[BEST_OURS], w)[m] - summarize(stats[bp], w)[m] for w in W])
                      for m in ('pooled_r2', 'within_r2')}
                row['vs_best_paper_by_' + crit] = {
                    'paper_model': bp,
                    **{'d_' + m: row[BEST_OURS][m] - row[bp][m] for m in dl},
                    **{'ci_' + m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in dl.items()}}
        res['%s|%s' % (proto, pair)] = row
        print(proto, pair, json.dumps({t: {k: round(v, 3) for k, v in row[t].items() if k in ('pooled_r2', 'within_r2')}
                                      for t in OURS + PAPER if isinstance(row.get(t), dict)}), flush=True)
json.dump(res, open(OUT, 'w'), indent=1, default=float)
print('TABLEDONE', OUT)
