"""RL-04/RL-05 comparison: relation-loss runs vs the dense p3-nbr baseline (same splits, same seed).
Pooled R2 and within-field metrics per pair x protocol, plus a paired field-season cluster bootstrap of the
deltas (per pair and for the mean over pairs). All statistics are additive per field season, so a bootstrap
replicate is a weighted sum. REL_SEEDS=0,1,2 pools seeds: each seed's statistics are computed on its own (its own
within-field centring) and summed per field season; the per-seed within-field R2 are reported as well."""
import glob, os, json, sys, numpy as np
root = (os.environ.get('REL_ROOT', '/data/YieldSAT/yieldsat_results/protocol')   # REL_ROOT: e.g. the local backup
        + '/' + os.environ.get('REL_SUB', 'paper'))                               # REL_SUB=own: our dataset
BASE = os.environ.get('REL_BASE', 'ours-p3nbr-dense')
VARS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['ours-p3nbr-dense-rel05', 'ours-p3nbr-dense-rel10']
PAIRS = sys.argv[2].split(',') if len(sys.argv) > 2 else ['ARG-W', 'BRA-C', 'GER-R', 'URG-S']
PROTOS = {'cv10': 'CV10', 'loyo': 'LOYO', 'loro': 'LORO'}
SEEDS = [int(x) for x in os.environ.get('REL_SEEDS', '0').split(',')]
MIN_PIX, B = 50, 2000


def load(grp, tag, pair, seed=0):
    fs = sorted(glob.glob('%s/s2_adm/%s_seed%d/%s/fold*/test_predictions.npz' % (grp, tag, seed, pair)))
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


def pooled_stats(runs, keys):
    """Per field-season statistics summed over seeds (one (pred, target, season) run per seed), and each
    seed's own within-field R2."""
    per = [field_stats(*r, keys) for r in runs]
    return ({k: sum(p[k] for p in per) for k in per[0]},
            [float(summarize(p)['within_r2']) for p in per])


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
        base = [load(grp, BASE, pair, sd) for sd in SEEDS]
        if any(b[0] is None for b in base) or len({b[1] for b in base}) > 1:
            continue
        nb = base[0][1]
        keys = np.unique(base[0][0][2])
        sb, seed_b = pooled_stats([b[0] for b in base], keys)
        W = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=B).astype(float)
        row = {'baseline': {**summarize(sb), **per_field(sb), 'folds': nb, 'seed_within_r2': seed_b}}
        for v in VARS:
            ds = [load(grp, v, pair, sd) for sd in SEEDS]
            nv = min(d[1] for d in ds)
            if any(d[0] is None or set(np.unique(d[0][2])) != set(keys) for d in ds) or nv < nb:
                row[v] = {'folds': nv, 'incomplete': True}
                continue
            sv, seed_v = pooled_stats([d[0] for d in ds], keys)
            dl = {m: np.array([summarize(sv, w)[m] - summarize(sb, w)[m] for w in W]) for m in ('pooled_r2', 'within_r2')}
            boots[v].append(dl)
            m = {**summarize(sv), **per_field(sv), 'folds': nv, 'seed_within_r2': seed_v}
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
out = os.environ.get('REL_OUT') or '/data/YieldSAT/yieldsat_results/protocol/%s.json' % (
    sys.argv[3] if len(sys.argv) > 3 else 'rel_cmp_%s' % ('dev' if len(PAIRS) == 4 else 'all'))
json.dump(res, open(out, 'w'), default=float)
print('RESJSON ' + json.dumps(res, default=float))
print('RELDONE')
