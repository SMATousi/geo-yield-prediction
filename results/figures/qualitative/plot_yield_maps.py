"""Qualitative yield maps: ground truth vs models on CV10 test fields (paper figures).

Field selection is fixed in advance, not by eye: per pair, among CV10 test field seasons with >= MIN_CELLS cells,
the field whose within-field R2 under our best model is the median (a typical field, neither best nor worst).
Each row shares one colour scale (2-98th percentile of the ground truth); anomaly maps show each map minus its own
field mean with a symmetric diverging scale.

    python results/figures/qualitative/plot_yield_maps.py <results root with paper/> <out dir>
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else '/root/yieldsat_backups/protocol_results')
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parent)
GROUP = 'cv10_season_noval_s0'
MODELS = [('ours-p3nbr-dense-fc-rel10', 'Ours'), ('ours-p3nbr-dense', 'Ours (base)'), ('paper_mmgf', 'MMGF')]
PAIRS_ALL = ['ARG-C', 'ARG-S', 'ARG-W', 'BRA-C', 'BRA-S', 'BRA-W', 'GER-R', 'GER-W', 'URG-S']
PAIRS_MAIN = ['ARG-S', 'BRA-C', 'GER-R', 'URG-S']
MIN_CELLS = 2000


def load(tag, pair):
    """-> {field name: (row, col, target, pred)} over the CV10 test folds."""
    out = {}
    for f in sorted(glob.glob('%s/paper/%s/s2_adm/%s_seed0/%s/fold*/test_predictions.npz' % (ROOT, GROUP, tag, pair))):
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


def pick(pair):
    data = {tag: load(tag, pair) for tag, _ in MODELS}
    ours = data[MODELS[0][0]]
    cands = [(within_r2(v[2], v[3])[0], k) for k, v in ours.items()
             if len(v[2]) >= MIN_CELLS and all(k in data[t] for t, _ in MODELS)]
    cands.sort()
    name = cands[len(cands) // 2][1]
    return name, {tag: data[tag][name] for tag, _ in MODELS}


def figure(pairs, models, path, panel=1.55):
    """One row per field: absolute maps (shared scale per row) | within-field anomaly maps (symmetric scale)."""
    nm = 1 + len(models)
    fig, axes = plt.subplots(len(pairs), 2 * nm, figsize=(panel * 2 * nm + 1.2, panel * len(pairs) + 0.5), squeeze=False)
    meta = {}
    for i, pair in enumerate(pairs):
        name, d = pick(pair)
        r, c, y, _ = d[MODELS[0][0]]
        lo, hi = np.percentile(y, [2, 98])
        a = np.percentile(np.abs(y - y.mean()), 98)
        maps = [('Ground truth', y, None)] + [(lab, d[tag][3], within_r2(d[tag][2], d[tag][3])) for tag, lab in models]
        meta[pair] = {'field': name, 'cells': int(len(y)), 'within_r2': {lab: float(sc[0]) for lab, _, sc in maps[1:]}}
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
                    ax.set_ylabel(pair, fontsize=9)
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
    main_models = [m for m in MODELS if m[0] != 'ours-p3nbr-dense']
    meta = {'main': figure(PAIRS_MAIN, main_models, OUT / 'yield_maps_main.png'),
            'supp_a': figure(PAIRS_ALL[:5], MODELS, OUT / 'yield_maps_supp_a.png'),
            'supp_b': figure(PAIRS_ALL[5:], MODELS, OUT / 'yield_maps_supp_b.png')}
    (OUT / 'yield_maps_fields.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: {p: v['field'] for p, v in m.items()} for k, m in meta.items()}, indent=1))


if __name__ == '__main__':
    main()
