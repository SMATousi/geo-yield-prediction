"""RL-04/RL-05 comparison: relation-loss runs vs the dense p3-nbr baseline (same splits, same seed).
Pooled R2 and within-field metrics per pair x protocol, plus a paired field-season cluster bootstrap of the
deltas (per pair and for the mean over pairs). All statistics are additive per field season, so a bootstrap
replicate is a weighted sum."""
import glob, os, json, sys, numpy as np
root = '/data/YieldSAT/yieldsat_results/protocol/paper'
BASE = os.environ.get('REL_BASE', 'ours-p3nbr-dense')
VARS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['ours-p3nbr-dense-rel05', 'ours-p3nbr-dense-rel10']
PAIRS = sys.argv[2].split(',') if len(sys.argv) > 2 else ['ARG-W', 'BRA-C', 'GER-R', 'URG-S']
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
        P.append(z['pred'][ok].astype(np.float64)); Y.append(z['target'][ok].astype(np.float64))
        S.append(np.char.add(os.path.basename(os.path.dirname(f)) + ':', np.asarray(z['season'][ok]).astype(str)))  # ids are per fold
    return (np.concatenate(P), np.concatenate(Y), np.concatenate(S)), len(fs)


def field_stats(p, y, s, keys):
    """Per field season (in `keys` order): n, sum y, sum y^2, sse, within sse, within sst, and per-field r2/r/ratio."""
    _, inv = np.unique(s, return_inverse=True)
    k_of = {k: i for i, k in enumerate(np.unique(s))}
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
    sst = syy - sy * sy / n
    out = {'pooled_r2': 1 - (w * st['sse']).sum() / sst, 'within_r2': 1 - (w * st['wsse']).sum() / (w * st['wsst']).sum()}
    return out


def per_field(st):
    m = (st['n'] >= MIN_PIX) & (st['wsst'] > 0) & (st['spp'] > 0)
    r2 = 1 - st['wsse'][m] / st['wsst'][m]
    r = st['spy'][m] / np.sqrt(st['spp'][m] * st['wsst'][m])
    ratio = np.sqrt(st['spp'][m] / st['wsst'][m])
    return {'median_field_r2': float(np.median(r2)), 'median_field_r': float(np.median(r)),
            'median_var_ratio': float(np.median(ratio)), 'fields': int(m.sum())}


rng = np.random.default_rng(0)
res = {}
for g, proto in PROTOS.items():
    # per pair, the first group holding its baseline: LORO is farm-level (loro_na_noval_s0) for most pairs and
    # province-level (loro_na_province_noval_s0) for the Argentine pairs
    groups = sorted(glob.glob('%s/%s_*noval_s0' % (root, g)), key=lambda d: ('province' in d, d))
    if not groups:
        continue
    boots = {v: [] for v in VARS}
    for pair in PAIRS:
        grp = next((d for d in groups if os.path.isdir('%s/s2_adm/%s_seed0/%s' % (d, BASE, pair))
                    and glob.glob('%s/s2_adm/%s_seed0/%s' % (d, VARS[0], pair))), None)
        if grp is None:
            continue
        base, nb = load(grp, BASE, pair)
        if base is None:
            continue
        keys = np.unique(base[2])
        sb = field_stats(*base, keys)
        W = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=B).astype(float)
        row = {'baseline': {**summarize(sb), **per_field(sb), 'folds': nb}}
        for v in VARS:
            d, nv = load(grp, v, pair)
            if d is None or nv < nb or set(np.unique(d[2])) != set(keys):
                row[v] = {'folds': nv, 'incomplete': True}
                continue
            sv = field_stats(*d, keys)
            dl = {m: np.array([summarize(sv, w)[m] - summarize(sb, w)[m] for w in W]) for m in ('pooled_r2', 'within_r2')}
            boots[v].append(dl)
            m = {**summarize(sv), **per_field(sv), 'folds': nv}
            for k in ('pooled_r2', 'within_r2'):
                m['d_' + k] = m[k] - row['baseline'][k]
                m['ci_' + k] = [float(np.percentile(dl[k], 2.5)), float(np.percentile(dl[k], 97.5))]
            row[v] = m
        res['%s|%s' % (proto, pair)] = row
        print(proto, pair, json.dumps(row, default=float), flush=True)
    for v in VARS:
        if boots[v] and len(boots[v]) == len(PAIRS):
            mean = {k: np.mean([b[k] for b in boots[v]], axis=0) for k in ('pooled_r2', 'within_r2')}
            pt = {k: float(np.mean([res['%s|%s' % (proto, pr)][v]['d_' + k] for pr in PAIRS])) for k in mean}
            res['%s|MEAN|%s' % (proto, v)] = {k: [pt[k], float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
                                              for k, x in mean.items()}
            print(proto, 'MEAN', v, json.dumps(res['%s|MEAN|%s' % (proto, v)]), flush=True)
out = '/data/YieldSAT/yieldsat_results/protocol/%s.json' % (sys.argv[3] if len(sys.argv) > 3 else
                                                          'rel_cmp_%s' % ('dev' if len(PAIRS) == 4 else 'all'))
json.dump(res, open(out, 'w'), default=float)
print('RESJSON ' + json.dumps(res, default=float))
print('RELDONE')
