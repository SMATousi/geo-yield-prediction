"""Missouri soybean showcase: ground truth vs our best model on the N_FIELDS fields it predicts best (no scores in the image).

Best model = S2 + PlanetScope 3x3 + 30 m window, validation-selected (paper setting), CV10 test predictions.
The fields are the MO-S test field seasons with the highest within-field R2 averaged over seeds 0-2, one per
row; the maps show seed 0. Absolute maps share one scale (2-98th percentile of the ground truth); anomaly maps show each
map minus its own field mean on a symmetric diverging scale.

    python results/figures/qualitative/plot_best_soy_field.py <results root with own_v2/> <out png>
"""
import glob
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else '/root/owndata_results_v2'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'yield_maps_own_best_soy.png'
TAG = 'ours-p3nbr-dense-fc-rel10-ps3x3-pswin'
GROUP = 'own_v2/cv10_season_val_s0/s2_adm'
N_FIELDS = 2


def load(seed):
    out = {}
    for f in sorted(glob.glob('%s/%s/%s_seed%d/MO-S/fold*/test_predictions.npz' % (ROOT, GROUP, TAG, seed))):
        z = np.load(f)
        names = z['season_names'].astype(str)
        for si in np.unique(z['season']):
            sel = (z['season'] == si) & np.isfinite(z['target']) & np.isfinite(z['pred'])
            out[names[si]] = (z['grid_row'][sel], z['grid_col'][sel], z['target'][sel].astype(float), z['pred'][sel].astype(float))
    return out


def within_r2(y, p):
    yc, pc = y - y.mean(), p - p.mean()
    return 1 - np.sum((pc - yc) ** 2) / np.sum(yc ** 2)


def grid(r, c, v):
    g = np.full((r.max() - r.min() + 1, c.max() - c.min() + 1), np.nan)
    g[r - r.min(), c - c.min()] = v
    return g


def main():
    seeds = [load(s) for s in range(3)]
    names = set.intersection(*(set(d) for d in seeds))
    score = {k: np.mean([within_r2(d[k][2], d[k][3]) for d in seeds]) for k in names}
    best = sorted(score, key=score.get, reverse=True)[:N_FIELDS]
    fig, axes = plt.subplots(len(best), 4, figsize=(11, 3.2 * len(best)), squeeze=False)
    for i, name in enumerate(best):
        r, c, y, p = seeds[0][name]
        print(name, 'mean within R2 %.3f, seed0 %.3f, %d cells' % (score[name], within_r2(y, p), len(y)))
        lo, hi = np.percentile(y, [2, 98])
        a = np.percentile(np.abs(y - y.mean()), 98)
        panels = [('Ground truth', y, 'RdYlGn', lo, hi), ('Prediction (ours)', p, 'RdYlGn', lo, hi),
                  ('Ground truth anomaly', y - y.mean(), 'RdBu', -a, a), ('Prediction anomaly', p - p.mean(), 'RdBu', -a, a)]
        for j, (ax, (lab, v, cmap, vmin, vmax)) in enumerate(zip(axes[i], panels)):
            im = ax.imshow(grid(r, c, v), cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
            if i == 0:
                ax.set_title(lab, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if j in (1, 3):
                cb = fig.colorbar(im, ax=list(axes[i, j - 1:j + 1]), fraction=0.04, pad=0.02)
                cb.ax.tick_params(labelsize=8)
                cb.set_label('Yield (t/ha)' if j == 1 else 'Deviation from field mean (t/ha)', fontsize=8)
    fig.savefig(OUT, dpi=250, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
