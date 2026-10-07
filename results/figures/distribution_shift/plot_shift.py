import json, sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
d = json.load(open(sys.argv[1])); out = sys.argv[2]
def cat_colors(vals):
    u = sorted(set(vals), key=str); cmap = plt.get_cmap('tab20' if len(u) > 10 else 'tab10')
    return {v: cmap(i % cmap.N) for i, v in enumerate(u)}, u
for pair, r in d.items():
    pts = r['points']; x = np.array([p['x'] for p in pts]); y = np.array([p['y'] for p in pts]); yl = np.array([p['yield'] for p in pts])
    years = [p['year'] for p in pts]; regs = [p['region'].split('/')[-1] for p in pts]
    fig, ax = plt.subplots(2, 3, figsize=(18, 10.5))
    for a, labels, title, key in ((ax[0, 0], years, 'Harvest year (LOYO groups)', 'loyo'), (ax[0, 1], regs, 'Region (LORO groups)', 'loro')):
        cm, u = cat_colors(labels)
        for v in u:
            m = np.array([l == v for l in labels]); a.scatter(x[m], y[m], s=18, color=cm[v], label=str(v), alpha=.85)
        a.set_title('%s: t-SNE of field-season inputs, by %s' % (pair, title.split(' (')[0].lower()))
        a.legend(fontsize=7, ncol=2 if len(u) > 8 else 1, loc='best', markerscale=1.2)
    sc = ax[0, 2].scatter(x, y, c=yl, s=18, cmap='viridis'); fig.colorbar(sc, ax=ax[0, 2], label='field mean yield (t/ha)')
    ax[0, 2].set_title('%s: t-SNE coloured by yield' % pair)
    for a in ax[0]: a.set_xticks([]); a.set_yticks([])
    for a, labels, key, name in ((ax[1, 0], years, 'loyo', 'year'), (ax[1, 1], regs, 'loro', 'region')):
        u = sorted(set(labels), key=str)
        data = [yl[np.array([l == v for l in labels])] for v in u]
        a.boxplot(data, tick_labels=[str(v) for v in u], showfliers=False)
        for i, v in enumerate(data): a.scatter(np.full(len(v), i + 1) + np.random.default_rng(0).uniform(-.15, .15, len(v)), v, s=6, alpha=.4)
        a.axhline(yl.mean(), color='grey', ls='--', lw=.8)
        a.set_ylabel('field mean yield (t/ha)'); a.set_title('%s: yield by %s' % (pair, name)); a.tick_params(axis='x', rotation=45, labelsize=7)
    a = ax[1, 2]
    for key, mk, name in (('loyo', 'o', 'held-out year'), ('loro', '^', 'held-out region')):
        for g, s in r[key].items():
            if s['auroc'] is None: continue
            a.scatter(s['auroc_s2'], s['yield_shift_sd'], marker=mk, s=30 + 3 * s['n'], alpha=.7, label=name)
            a.annotate(g.split('/')[-1], (s['auroc_s2'], s['yield_shift_sd']), fontsize=7)
    h, l = a.get_legend_handles_labels(); seen = {}
    for hh, ll in zip(h, l): seen.setdefault(ll, hh)
    a.legend(seen.values(), seen.keys(), fontsize=8)
    a.axvline(.5, color='grey', ls=':'); a.axhline(0, color='grey', ls=':')
    a.set_xlabel('S2 covariate shift: AUROC of telling the group from the rest (S2 spectral features only)'); a.set_ylabel('label shift: group mean yield - rest (SD)')
    a.set_title('%s: shift per held-out group (marker size = fields)' % pair)
    fig.tight_layout(); fig.savefig('%s/%s.png' % (out, pair), dpi=110); plt.close(fig)
for key, name in (('loyo', 'year'), ('loro', 'region')):
    fig, a = plt.subplots(figsize=(11, 8)); cmap = plt.get_cmap('tab10')
    for i, (pair, r) in enumerate(d.items()):
        pts = [(s['auroc_s2'], s['yield_shift_sd'], g) for g, s in r[key].items() if s['auroc'] is not None]
        if not pts: continue
        a.scatter([p[0] for p in pts], [p[1] for p in pts], color=cmap(i), s=40, label=pair, alpha=.8)
        for p in pts: a.annotate(str(p[2]).split('/')[-1], (p[0], p[1]), fontsize=6, color=cmap(i))
    a.axvline(.5, color='grey', ls=':'); a.axhline(0, color='grey', ls=':'); a.legend()
    a.set_xlabel('S2 covariate shift: AUROC of telling the held-out %s from the rest (S2 spectral features only; 0.5 = indistinguishable)' % name)
    a.set_ylabel('label shift: held-out %s mean yield - rest (SD of field yields)' % name)
    a.set_title('Distribution shift of every held-out %s (%s), all pairs' % (name, key.upper()))
    fig.tight_layout(); fig.savefig('%s/summary_%s.png' % (out, key), dpi=110); plt.close(fig)
lines = ['| Pair | Protocol | Group | Fields | AUROC, all inputs | AUROC, S2 only | Yield mean | Rest mean | Label shift (SD) |', '|---|---|---|---|---|---|---|---|---|']
for pair, r in d.items():
    for key in ('loyo', 'loro'):
        for g, s in r[key].items():
            if s['auroc'] is None: lines.append('| %s | %s | %s | %d | – | – | – | – | – |' % (pair, key.upper(), g, s['n'])); continue
            lines.append('| %s | %s | %s | %d | %.2f | %.2f | %.2f | %.2f | %+.2f |' % (pair, key.upper(), g, s['n'], s['auroc'], s['auroc_s2'], s['yield_mean'], s['rest_mean'], s['yield_shift_sd']))
open('%s/shift_scores.md' % out, 'w').write('\n'.join(lines) + '\n')
print('PLOTSDONE')
