"""Knowledge-pretraining round report: success criteria P, I, E and verdicts (PK-08).

Inputs:
  --pooled         pooled metrics of the round (yieldsat_pooled_metrics.py --fold_stats --match 'pk-*')
  --a0             pooled metrics of the from-scratch point model (before_full, --fold_stats)
  --runs           the round's plan runs.jsonl
  --pretrain_root  results root holding pretrain/<arm>/<unit>/{report,diagnostics}.json
Criteria and decision rules: spec/pretraining/success-criteria.md. Confidence
intervals: paired bootstrap over DEV folds (E) or held-out field seasons (I2).

    python yieldsat_pretrain_report.py --pooled pooled_pk.json --a0 pooled_before_full_fs.json \
        --runs runs.jsonl --pretrain_root /data/.../pk_dev1 --out results/pretrain_dev.md
"""
import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np

from yieldsat_dev_eval import PROTO, paper_best
from yieldsat_pretrain_diagnostics import p2_pass

B = 2000
ACTIVE_RULE_CONCEPTS = ('rain_supported', 'optical_growth', 'warm_regime', 'active_spectral_state',
                        'clay_rich_surface', 'persistent_canopy', 'low_elevation_position',
                        'fine_surface_texture', 'organic_surface', 'spectral_moisture_state')
ABSTAIN_CONCEPTS = ('pole_facing_aspect', 'cool_regime')        # only in ys_r05 (always abstains)


def arm_of(experiment):
    return 'A' + experiment.split('-a', 1)[1] if experiment.startswith('pk-a') else experiment


# ---- E: pooled DEV metrics with a paired fold bootstrap -------------------------

def _r2_from(stats, folds):
    n = sum(stats[f]['n'] for f in folds)
    sy = sum(stats[f]['sy'] for f in folds)
    syy = sum(stats[f]['syy'] for f in folds)
    sse = sum(stats[f]['sse'] for f in folds)
    fl = np.concatenate([np.asarray(stats[f]['field'], float).reshape(-1, 2) for f in folds])
    sst = syy - sy * sy / n
    fsst = ((fl[:, 0] - fl[:, 0].mean()) ** 2).sum()
    return 1 - sse / sst, 1 - ((fl[:, 0] - fl[:, 1]) ** 2).sum() / fsst


def load_rows(pooled, runs, a0):
    """{arm: {row: fold_stats}} for complete rows; row = (protocol, pair, group)."""
    expected, exp_of = collections.Counter(), {}
    for r in runs:
        if r['protocol'] == 'pretrain':
            continue
        d = r['rel_path'].rsplit('/', 1)[0]
        expected[d] += 1
        exp_of[d] = r['experiment']
    arms = collections.defaultdict(dict)
    incomplete = collections.Counter()
    for d, n in expected.items():
        got = pooled.get(d)
        if not got or len(got['folds']) < n or 'fold_stats' not in got:
            incomplete[arm_of(exp_of[d])] += 1
            continue
        _, group, inputs, _, pair = d.split('/')
        arms[arm_of(exp_of[d])][(PROTO[group.split('_')[0]], pair, group)] = got['fold_stats']
    rows = sorted({k for a in arms.values() for k in a})
    for k in rows:
        key = 'paper/{}/s2_adm/ours_seed0/{}'.format(k[2], k[1])
        if key in a0 and 'fold_stats' in a0[key]:
            arms['A0'][k] = a0[key]['fold_stats']
    return arms, rows, incomplete


def compare(arms, rows, x, y, seed=0):
    """Paired DEV-mean differences x - y with fold-bootstrap CIs, per protocol and per row."""
    common = [k for k in rows if k in arms.get(x, {}) and k in arms.get(y, {})]
    if not common:
        return None
    rng = np.random.default_rng(seed)
    point = {k: (np.subtract(_r2_from(arms[x][k], list(arms[x][k])), _r2_from(arms[y][k], list(arms[y][k]))))
             for k in common}
    boots = np.zeros((B, 2))
    for b in range(B):
        acc = []
        for k in common:
            folds = list(arms[x][k])
            draw = [folds[i] for i in rng.integers(0, len(folds), len(folds))]
            try:
                acc.append(np.subtract(_r2_from(arms[x][k], draw), _r2_from(arms[y][k], draw)))
            except (ZeroDivisionError, ValueError, FloatingPointError):
                continue
        boots[b] = np.mean(acc, axis=0)
    mean = np.mean([point[k] for k in common], axis=0)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    proto = {p: np.mean([point[k] for k in common if k[0] == p], axis=0)
             for p in ('CV10', 'LOYO', 'LORO') if any(k[0] == p for k in common)}
    return {'rows': len(common), 'pixel': float(mean[0]), 'field': float(mean[1]),
            'pixel_ci': (float(lo[0]), float(hi[0])), 'field_ci': (float(lo[1]), float(hi[1])),
            'protocols': {p: (float(v[0]), float(v[1])) for p, v in proto.items()},
            'per_row': {k: (float(v[0]), float(v[1])) for k, v in point.items()}}


def arm_means(arms, rows, best):
    out = {}
    for a, rr in arms.items():
        vals = {k: _r2_from(v, list(v)) for k, v in rr.items()}
        out[a] = {'rows': len(vals), 'pixel': float(np.mean([v[0] for v in vals.values()])),
                  'field': float(np.mean([v[1] for v in vals.values()])), 'per_row': vals,
                  'protocols': {p: float(np.mean([v[0] for k, v in vals.items() if k[0] == p]))
                                for p in ('CV10', 'LOYO', 'LORO') if any(k[0] == p for k in vals)}}
    bp = [best.get((k[0], 'pixel', k[1]), (np.nan,))[0] for k in rows]
    bf = [best.get((k[0], 'field', k[1]), (np.nan,))[0] for k in rows]
    return out, float(np.nanmean(bp)), float(np.nanmean(bf))


# ---- P / I: pretraining reports and diagnostics -------------------------------

def load_pretrain(root):
    out = collections.defaultdict(dict)        # arm -> unit -> {'report', 'diag', 'ref'}
    for rep in sorted(Path(root).glob('pretrain/*/*/report.json')):
        arm, unit = rep.parent.parent.name.upper(), rep.parent.name
        rec = {'report': json.loads(rep.read_text())}
        d = rep.parent / 'diagnostics.json'
        rec['diag'] = json.loads(d.read_text()) if d.exists() else None
        ref = rep.parent / 'knowledge_reference.json'
        rec['ref'] = json.loads(ref.read_text()) if ref.exists() else None
        out[arm][unit] = rec
    return out


def p_criteria(units):
    res = {'P1': [], 'P2': [], 'P3': [], 'P4': []}
    for unit, rec in units.items():
        h = rec['report']['history']
        val = [e['val_loss'] for e in h if 'val_loss' in e]
        comps = [k for k in h[0] if k.startswith('train_') and not k.endswith('_terms') and k != 'train_loss']
        dec = all(h[-1][k] < h[0][k] for k in comps + ['train_loss'])
        res['P1'].append(bool(val) and val[-1] <= 1.05 * min(val) and dec)
        d = rec['diag']
        res['P2'].append(d is not None and all(
            p2_pass(v['variance'], v['variance_init'], v['effective_rank'], v['effective_rank_init'], s)
            for s, v in d['P2_collapse'].items()))
        k = rec['report'].get('knowledge')
        if k:
            cov = all(v >= 0.2 for v in k['concept_season_coverage'].values())
            rules = all(v['applicable_seasons'] >= 200 for r, v in k['rules'].items() if r != 'ys_r05')
            res['P3'].append(cov and rules)
            tables = rec['ref']['tables'] if rec['ref'] else {}
            res['P4'].append(bool(tables) and all(np.asarray(t)[250] < np.asarray(t)[750] for t in tables.values()))
    return {c: (sum(v), len(v)) for c, v in res.items() if v}


MIN_HELDOUT_SEASONS = 10


def mean_diag(units, path):
    """Median over units with >= MIN_HELDOUT_SEASONS held-out seasons (tiny held-out
    sets, e.g. one 3-season farm, give unstable probe R2 and AUROC)."""
    vals = collections.defaultdict(list)
    for rec in units.values():
        d = rec['diag']
        if not d or d.get('heldout_seasons', 0) < MIN_HELDOUT_SEASONS:
            continue
        block = d
        for p in path:
            block = block.get(p, {}) if isinstance(block, dict) else {}
        for k, v in block.items():
            v = v.get('auroc') if isinstance(v, dict) and 'auroc' in v else v
            if isinstance(v, (int, float)) and v is not None:
                vals[k].append(v)
    return {k: float(np.median(v)) for k, v in vals.items()}


def i2_paired(a3, a7, seed=0):
    """Per rule: A3 - A7 alignment, paired by (unit, season), bootstrap over seasons."""
    rng = np.random.default_rng(seed)
    out = {}
    rules = set()
    for rec in a3.values():
        if rec['diag'] and 'I2_relational' in rec['diag']:
            rules |= set(rec['diag']['I2_relational'])
    for r in sorted(rules):
        diffs = []
        for unit, rec in a3.items():
            other = a7.get(unit)
            if not (rec['diag'] and other and other['diag']):
                continue
            x = rec['diag']['I2_relational'].get(r, {}).get('per_season', {})
            y = other['diag']['I2_relational'].get(r, {}).get('per_season', {})
            diffs += [x[s] - y[s] for s in x if s in y]
        if not diffs:
            out[r] = None
            continue
        diffs = np.asarray(diffs)
        bs = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(B)]
        out[r] = {'n': len(diffs), 'mean': float(diffs.mean()), 'ci': tuple(np.percentile(bs, [2.5, 97.5]).tolist())}
    return out


# ---- report --------------------------------------------------------------------

def fmt_ci(d, key):
    lo, hi = d[key + '_ci']
    return '{:+.3f} [{:+.3f}, {:+.3f}]'.format(d[key], lo, hi)


def build(args):
    pooled = json.loads(Path(args.pooled).read_text())
    a0 = json.loads(Path(args.a0).read_text())
    runs = [json.loads(l) for l in open(args.runs)]
    best = paper_best(args.paper)
    arms, rows, incomplete = load_rows(pooled, runs, a0)
    means, best_pix, best_fld = arm_means(arms, rows, best)
    pre = load_pretrain(args.pretrain_root)
    L = ['# Knowledge pretraining, DEV round ({})'.format(args.round), '',
         'Generated {} by `yieldsat_pretrain_report.py`.'.format(time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())),
         'Criteria and decision rules: spec/pretraining/success-criteria.md. Arms: spec/yieldsat-point-knowledge-pretraining.md §6.',
         'Pooled out-of-fold R² on the DEV subset. Δ CIs: paired bootstrap over DEV folds ({} draws). '.format(B) +
         'Failed criteria are reported as failures.', '']
    # E table
    L += ['## Yield prediction (E)', '', '| Arm | Rows | Pixel R² | Field R² | CV10 / LOYO / LORO pixel |', '|---|---|---|---|---|']
    for a in sorted(means):
        m = means[a]
        L.append('| {} | {}/{}{} | {:.3f} | {:.3f} | {} |'.format(
            a, m['rows'], len(rows), ' (incomplete)' if m['rows'] < len(rows) else '', m['pixel'], m['field'],
            ' / '.join('{:.2f}'.format(m['protocols'][p]) for p in ('CV10', 'LOYO', 'LORO') if p in m['protocols'])))
    L += ['| **Paper best** | | **{:.3f}** | **{:.3f}** | |'.format(best_pix, best_fld), '']
    if incomplete:
        L += ['Incomplete experiments (excluded): {}'.format(dict(incomplete)), '']
    cmp_ = {}
    pairs = [('A2', 'A0'), ('A3', 'A0'), ('A3', 'A2'), ('A3', 'A6'), ('A3', 'A7'), ('A3', 'A4'), ('A3', 'A5')]
    L += ['| Comparison | Rows | Δ pixel [95% CI] | Δ field [95% CI] | Δ pixel CV10 / LOYO / LORO |', '|---|---|---|---|---|']
    for x, y in pairs:
        c = compare(arms, rows, x, y)
        if c is None:
            continue
        cmp_[(x, y)] = c
        L.append('| {} − {} | {} | {} | {} | {} |'.format(x, y, c['rows'], fmt_ci(c, 'pixel'), fmt_ci(c, 'field'),
                 ' / '.join('{:+.3f}'.format(c['protocols'][p][0]) for p in ('CV10', 'LOYO', 'LORO') if p in c['protocols'])))
    verdict = {}
    c = cmp_.get(('A2', 'A0'))
    if c:
        verdict['E1'] = c['pixel'] >= 0 and c['field'] >= 0 and all(v[0] >= -0.02 for v in c['protocols'].values())
    e2 = [cmp_.get(('A3', y)) for y in ('A2', 'A6', 'A7')]
    if all(e2):
        verdict['E2'] = all(c['pixel'] >= 0.01 and c['pixel_ci'][0] > 0 and c['field'] >= 0 for c in e2)
    e2b = [cmp_.get(('A3', y)) for y in ('A4', 'A5')]
    if all(e2b):
        verdict['E2b'] = all(c['pixel'] >= 0.01 and c['pixel_ci'][0] > 0 and c['field'] >= 0 for c in e2b)
    c = cmp_.get(('A3', 'A2'))
    if c and {'CV10', 'LOYO', 'LORO'} <= set(c['protocols']):
        cv = c['protocols']['CV10'][0]
        verdict['E4'] = c['protocols']['LOYO'][0] >= cv and c['protocols']['LORO'][0] >= cv
        verdict['E5'] = all(v[0] >= -0.05 for v in c['per_row'].values())
    if 'A3' in means:
        verdict['E6 (A3 ≥ paper best + 0.03)'] = (means['A3']['pixel'] >= best_pix + 0.03
                                                  and means['A3']['field'] >= best_fld + 0.03)
    L += ['', '## Pretraining health (P)', '', '| Arm | Units | P1 convergence | P2 no collapse | P3 coverage | P4 estimator sanity |',
          '|---|---|---|---|---|---|']
    health = {}
    for a in sorted(pre):
        p = p_criteria(pre[a])
        health[a] = all(v[0] == v[1] for v in p.values())
        L.append('| {} | {} | {} |'.format(a, len(pre[a]), ' | '.join(
            '{}/{}'.format(*p[c]) if c in p else 'n/a' for c in ('P1', 'P2', 'P3', 'P4'))))
    L += ['', '## Knowledge learned (I), held-out seasons', '']
    i1 = {a: mean_diag(pre[a], ['I1_grounding']) for a in pre}
    if 'A3' in i1 and i1['A3']:
        concepts = sorted(i1['A3'])
        L += ['| Concept | ' + ' | '.join('{} AUROC'.format(a) for a in sorted(i1) if i1[a]) + ' | I1 pass (≥0.70 and ≥ A7+0.10) |',
              '|---|' + '---|' * (sum(1 for a in i1 if i1[a]) + 1)]
        passes = {}
        for cpt in concepts:
            v3, v7 = i1['A3'].get(cpt), i1.get('A7', {}).get(cpt)
            ok = v3 is not None and v3 >= 0.70 and (v7 is None or v3 >= v7 + 0.10)
            passes[cpt] = ok
            L.append('| {} | {} | {} |'.format(cpt, ' | '.join(
                '{:.3f}'.format(i1[a][cpt]) if cpt in i1[a] else '–' for a in sorted(i1) if i1[a]), 'yes' if ok else 'no'))
        verdict['I1'] = all(passes.values())
        if 'A7' in i1 and i1['A7']:
            g_act = [i1['A3'][c] - i1['A7'][c] for c in ACTIVE_RULE_CONCEPTS if c in i1['A3'] and c in i1['A7']]
            g_abs = [i1['A3'][c] - i1['A7'][c] for c in ABSTAIN_CONCEPTS if c in i1['A3'] and c in i1['A7']]
            if g_act and g_abs:
                verdict['I5'] = float(np.mean(g_act)) > float(np.mean(g_abs))
                L += ['', 'I5: mean I1 gain over A7, active-rule concepts {:+.3f} vs abstaining-rule concepts {:+.3f}.'.format(
                    np.mean(g_act), np.mean(g_abs))]
    if 'A3' in pre and 'A7' in pre:
        i2 = i2_paired(pre['A3'], pre['A7'])
        L += ['', '| Rule | Pairs (unit × season) | A3 − A7 alignment [95% CI] |', '|---|---|---|']
        for r, v in i2.items():
            L.append('| {} | {} | {} |'.format(r, v['n'] if v else 0,
                     '{:+.3f} [{:+.3f}, {:+.3f}]'.format(v['mean'], *v['ci']) if v else 'n/a (abstains)'))
        act = [v for v in i2.values() if v]
        if act:
            verdict['I2'] = all(v['ci'][0] > 0 for v in act)
    # I3/I4 compare arms on the units every arm has finished (different unit subsets
    # differ by held-out population, which dominates probe R2 and SSL loss)
    common = set.intersection(*[set(u) for u in pre.values()]) if pre else set()
    sub = {a: {u: pre[a][u] for u in common} for a in pre}
    i3 = {a: mean_diag(sub[a], ['I3_probe_r2']) for a in sub}
    i4 = {a: mean_diag(sub[a], ['I4_ssl_heldout']) for a in sub}
    L += ['', 'I3/I4 use the {} units finished by every arm.'.format(len(common))]
    if i3.get('A3') and i3.get('A2'):
        L += ['', '| Probe (R², held-out) | ' + ' | '.join(sorted(a for a in i3 if i3[a])) + ' |', '|---|' + '---|' * sum(1 for a in i3 if i3[a])]
        for k in sorted(i3['A2']):
            L.append('| {} | {} |'.format(k, ' | '.join('{:.3f}'.format(i3[a][k]) if k in i3[a] else '–'
                                                         for a in sorted(i3) if i3[a])))
        verdict['I3'] = all(i3['A3'].get(k, -9) >= v - 0.02 for k, v in i3['A2'].items())
    if i4.get('A3') and i4.get('A2'):
        keys = [k for k in ('masked_observation', 'forecast_last_valid') if k in i4['A2']]
        verdict['I4'] = all(i4['A3'][k] <= 1.05 * i4['A2'][k] for k in keys)
        L += ['', 'I4 held-out SSL losses: ' + '; '.join('{} {}'.format(a, ', '.join('{} {:.4f}'.format(k, i4[a][k]) for k in keys if k in i4[a]))
                                                         for a in sorted(i4) if i4[a])]
    L += ['', '## Criteria and decisions', '', '| Criterion | Result |', '|---|---|']
    for k, v in verdict.items():
        L.append('| {} | {} |'.format(k, 'pass' if v else '**fail**'))
    healthy = health.get('A3', False) and health.get('A2', False)
    doing = healthy and all(verdict.get(k, False) for k in ('I1', 'I2', 'I4')) and verdict.get('E2', False)
    helping = verdict.get('E1', False) and verdict.get('E2', False) and (verdict.get('E3', False) or verdict.get('E4', False)) \
        and verdict.get('E5', False)
    L += ['', '- **Pretraining is healthy** (A2, A3: P1–P4 on every unit): {}'.format('yes' if healthy else 'no'),
          '- **The knowledge is doing its job** (I1, I2, I4, E2 vs A7): {}'.format('yes' if doing else 'no'),
          '- **Pretraining is helping** (E1, E2, E3 or E4, E5): {}'.format('yes' if helping else 'no')]
    if 'E2b' in verdict:
        L.append('- **Language matters** (E2b): {}'.format('yes' if verdict['E2b'] else 'no'))
    comp = collections.defaultdict(float)
    for a, units in pre.items():
        for rec in units.values():
            comp[a] += rec['report']['resources']['wall_seconds'] / 3600
    L += ['', '## Compute (pretraining wall-hours per arm, concurrent runs share GPUs)', '',
          ' · '.join('{} {:.1f} h'.format(a, h) for a, h in sorted(comp.items()))]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text('\n'.join(L) + '\n')
    return verdict


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pooled', required=True)
    p.add_argument('--a0', required=True)
    p.add_argument('--runs', required=True)
    p.add_argument('--pretrain_root', required=True)
    p.add_argument('--paper', default='results/yieldsat/paper_benchmark.csv')
    p.add_argument('--round', default='phase 1')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    print(json.dumps(build(a)))


if __name__ == '__main__':
    main()
