"""Qualitative yield maps: ground truth vs models on CV10 test fields (paper figures).

Field selection is fixed in advance, not by eye: per pair, among CV10 test field seasons with >= MIN_CELLS cells,
the field whose within-field R2 under our best model is the median (a typical field, neither best nor worst).
Each row shares one colour scale (2-98th percentile of the ground truth); anomaly maps show each map minus its own
field mean with a symmetric diverging scale.

    python results/figures/qualitative/plot_yield_maps.py <results root with paper/> <out dir>
    python results/figures/qualitative/plot_yield_maps.py <results root with own/> <out dir> own

`own`: our Missouri dataset (spec/own-dataset-training.md), the best arm (S2 + PlanetScope 3x3 + 30 m window)
against the 3x3 alone and S2 only; main figure the fields at the 75th and 90th percentile of the best arm's
within-field R2 per crop (good fields, not the best; corn, with 12 test fields, uses the next field below its 90th
percentile, the 82nd), supplementary the fields at the 25th, 50th and 75th percentile.
"""
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else '/root/yieldsat_backups/protocol_results')
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parent)
OWN = len(sys.argv) > 3 and sys.argv[3] == 'own'
SUB, GROUP = (os.environ.get('MAPS_OWN_SUB', 'own') if OWN else 'paper'), 'cv10_season_noval_s0'   # own_v2: v2 data
MODELS = [('ours-p3nbr-dense-fc-rel10', 'Ours'), ('ours-p3nbr-dense', 'Ours (base)'), ('paper_mmgf', 'MMGF')]
PAIRS_ALL = ['ARG-C', 'ARG-S', 'ARG-W', 'BRA-C', 'BRA-S', 'BRA-W', 'GER-R', 'GER-W', 'URG-S']
PAIRS_MAIN = ['ARG-S', 'BRA-C', 'GER-R', 'URG-S']
MIN_CELLS = 2000
if OWN:
    MODELS = [('ours-p3nbr-dense-fc-rel10-ps3x3-pswin', 'Ours + PS'), ('ours-p3nbr-dense-fc-rel10-ps3x3', 'Ours + PS 3x3'),
              ('ours-p3nbr-dense-fc-rel10', 'Ours (S2)')]
    PAIRS_ALL = PAIRS_MAIN = ['MO-S', 'MO-C']
RANK_SHIFT = {('MO-C', 0.9): -1} if OWN and SUB == 'own' else {}               # main figure: corn's 90th percentile -> the next below


def load(tag, pair):
    """-> {field name: (row, col, target, pred)} over the CV10 test folds."""
    out = {}
    for f in sorted(glob.glob('%s/%s/%s/s2_adm/%s_seed0/%s/fold*/test_predictions.npz' % (ROOT, SUB, GROUP, tag, pair))):
        z = np.load(f)
        names = z['season_names'].astype(str)
        for si in np.unique(z['season']):
            sel = (z['season'] == si) & np.isfinite(z['target']) & np.isfinite(z['pred'])
            out[names[si]] = (z['grid_row'][sel], z['grid_col'][sel], z['target'][sel].astype(float), z['pred'][sel].astype(float))
    return out


def within_r2(y, p):
    yc, pc = y - y.mean(), p - p.mean()
    return 1 - np.sum((pc - yc) ** 2) / np.sum(yc ** 2), np.corrcoef(pc, yc)[0, 1]


def grid(r, c, v):
    g = np.full((r.max() - r.min() + 1, c.max() - c.min() + 1), np.nan)
    g[r - r.min(), c - c.min()] = v
    return g


_cache = {}


def pick(pair, q=0.5):
    """The field at quantile q of the best model's within-field R2 (q=0.5: the median field); returns its quantile."""
    if pair not in _cache:
        _cache[pair] = {tag: load(tag, pair) for tag, _ in MODELS}
    data = _cache[pair]
    ours = data[MODELS[0][0]]
    cands = [(within_r2(v[2], v[3])[0], k) for k, v in ours.items()
             if len(v[2]) >= MIN_CELLS and all(k in data[t] for t, _ in MODELS)]
    cands.sort()
    i = int(round(q * (len(cands) - 1))) if q != 0.5 else len(cands) // 2
    i += RANK_SHIFT.get((pair, q), 0)
    name = cands[i][1]
    return name, {tag: data[tag][name] for tag, _ in MODELS}, i / (len(cands) - 1)


def figure(pairs, models, path, panel=1.55, qs=(0.5,)):
    """One row per field: absolute maps (shared scale per row) | within-field anomaly maps (symmetric scale)."""
    nm = 1 + len(models)
    rows = [(p, q) for p in pairs for q in qs]
    fig, axes = plt.subplots(len(rows), 2 * nm, figsize=(panel * 2 * nm + 1.2, panel * len(rows) + 0.5), squeeze=False)
    meta = {}
    for i, (pair, q) in enumerate(rows):
        name, d, qq = pick(pair, q)
        row_label = pair if qs == (0.5,) else '%s, p%d' % (pair, round(100 * qq))
        r, c, y, _ = d[MODELS[0][0]]
        lo, hi = np.percentile(y, [2, 98])
        a = np.percentile(np.abs(y - y.mean()), 98)
        maps = [('Ground truth', y, None)] + [(lab, d[tag][3], within_r2(d[tag][2], d[tag][3])) for tag, lab in models]
        meta[row_label] = {'field': name, 'cells': int(len(y)), 'within_r2': {lab: float(sc[0]) for lab, _, sc in maps[1:]}}
        for j, (lab, v, sc) in enumerate(maps):
            for k, (vals, cmap, vmin, vmax) in enumerate(((v, 'RdYlGn', lo, hi), (v - v.mean(), 'RdBu', -a, a))):
                ax = axes[i, j + k * nm]
                im = ax.imshow(grid(r, c, vals), cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if i == 0:
                    ax.set_title(lab + ('' if k == 0 else ' (anomaly)'), fontsize=8)
                if sc is not None and k == 1:
                    ax.set_xlabel(r'$R^2_w$ %.2f, $r$ %.2f' % sc, fontsize=7, labelpad=2)
                if j == 0 and k == 0:
                    ax.set_ylabel(row_label, fontsize=9)
                if j == nm - 1:
                    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
                    cb.ax.tick_params(labelsize=6)
                    cb.set_label('t/ha', fontsize=6)
    fig.tight_layout(w_pad=0.3, h_pad=0.6)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if OWN:
        meta = {'main': figure(PAIRS_MAIN, [MODELS[0], MODELS[2]], OUT / 'yield_maps_own_main.png', qs=(0.75, 0.9)),
                'supp': figure(PAIRS_ALL, MODELS, OUT / 'yield_maps_own_supp.png', qs=(0.25, 0.5, 0.75))}
        (OUT / 'yield_maps_own_fields.json').write_text(json.dumps(meta, indent=1))
        print(json.dumps({k: {p: v['field'] for p, v in m.items()} for k, m in meta.items()}, indent=1))
        return
    main_models = [m for m in MODELS if m[0] != 'ours-p3nbr-dense']
    meta = {'main': figure(PAIRS_MAIN, main_models, OUT / 'yield_maps_main.png'),
            'supp_a': figure(PAIRS_ALL[:5], MODELS, OUT / 'yield_maps_supp_a.png'),
            'supp_b': figure(PAIRS_ALL[5:], MODELS, OUT / 'yield_maps_supp_b.png')}
    (OUT / 'yield_maps_fields.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: {p: v['field'] for p, v in m.items()} for k, m in meta.items()}, indent=1))


if __name__ == '__main__':
    main()
