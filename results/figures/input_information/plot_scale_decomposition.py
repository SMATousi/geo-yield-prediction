"""Scale-decomposition figure (paper supplement, fig:scale; spec/yieldsat-input-information.md §3) from
results/scale_decomposition.json: (a) share of the within-field yield variance per difference-of-Gaussians band,
(b) correlation of predictions and peak NDVI with the target per band, (c) variance explained per band.
CV10, nine pairs pooled, seed 0. Same style as plot_information_budget.py.

    python results/figures/input_information/plot_scale_decomposition.py <out dir>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
D = json.loads((HERE.parents[1] / 'scale_decomposition.json').read_text())
B, R = D['bands'], D['results']
PURPLE, MID, LIGHT = '#4E4460', '#8A7EA3', '#C4BBD4'
INK, MUTED, GREY = '#222222', '#6B6B6B', '#A7AFB4'
# (source, label, colour, line style, marker); lines differ in style and marker, not only colour
SRC = [('ours-p3nbr-dense-fc-rel10', 'Yield-ReFiNe', PURPLE, '-', 'o'),
       ('ours-p3nbr-dense', 'Yield-ReFiNe (base)', MID, '--', 's'),
       ('paper_mmgf', 'MMGF', GREY, '-', 'D'),
       ('input:peak_ndvi', 'Peak-season NDVI (input)', MUTED, ':', '^')]


def totals(src):
    ks = [k for k in R if k.endswith('|' + src)]
    return {x: np.sum([R[k]['sums'][x] for k in ks], 0) for x in ('yy', 'pp', 'py', 'ee')}


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
                         'xtick.color': MUTED, 'ytick.color': MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.85), gridspec_kw={'wspace': 0.28})
    x = np.arange(len(B))
    ticks = [b.replace(' m', '').replace('-', '–') for b in B]

    t = totals(SRC[0][0])
    share = t['yy'] / t['yy'].sum()
    axes[0].bar(x, share, width=0.62, color=LIGHT, lw=0)
    for i, v in enumerate(share):
        axes[0].text(i, v + 0.01, '%.2f' % v, ha='center', va='bottom', fontsize=6, color=INK)
    axes[0].set_ylim(0, 0.56)
    axes[0].set_ylabel('Share of variance')

    for src, lab, col, ls, mk in SRC:
        s = totals(src)
        kw = dict(color=col, ls=ls, marker=mk, ms=3.2, lw=1.4 if src == SRC[0][0] else 1.0, label=lab,
                  zorder=3 if src == SRC[0][0] else 2)
        axes[1].plot(x, s['py'] / np.sqrt(s['pp'] * s['yy']), **kw)
        if not src.startswith('input'):
            axes[2].plot(x, 1 - s['ee'] / s['yy'], **kw)
    axes[1].set_ylabel('Correlation')
    axes[2].set_ylabel('Variance explained')

    for ax, title in zip(axes, ['(a) Within-field yield variance', '(b) Correlation with the target',
                                '(c) Variance explained']):
        ax.set_title(title, fontsize=7, loc='left', color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=6, rotation=35, ha='right', rotation_mode='anchor')
        ax.set_xlabel('Wavelength band (m)')
        ax.tick_params(length=2, width=0.6)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.grid(axis='y', color='#E4E4E4', lw=0.5)
        ax.set_axisbelow(True)
    for ax in axes[1:]:
        ax.set_ylim(0, 0.8)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False, fontsize=6.5,
               handlelength=2.6, columnspacing=1.6)

    fig.savefig(out / 'scale_decomposition.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(out / 'scale_decomposition.png', bbox_inches='tight', pad_inches=0.02, dpi=300)


if __name__ == '__main__':
    main()
