"""IC-B figure (spec/yieldsat-input-information.md §3) from results/scale_decomposition.json.

    python results/figures/input_information/plot_scale_decomposition.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
D = json.loads((HERE.parents[1] / 'scale_decomposition.json').read_text())
B, R = D['bands'], D['results']
SRC = [('ours-p3nbr-dense-fc-rel10', 'ours: dense p3-nbr + field context + rel. loss', 'tab:blue', '-'),
       ('ours-p3nbr-dense', 'ours: dense p3-nbr', 'tab:cyan', '--'),
       ('paper_mmgf', 'paper best (MMGF)', 'tab:orange', '-'),
       ('input:peak_ndvi', 'input: peak-season NDVI', 'tab:green', ':')]


def totals(src):
    ks = [k for k in R if k.endswith('|' + src)]
    return {x: np.sum([R[k]['sums'][x] for k in ks], 0) for x in ('yy', 'pp', 'py', 'ee')}


def main():
    x = np.arange(len(B))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    t = totals(SRC[0][0])
    axes[0].bar(x, t['yy'] / t['yy'].sum(), color='0.6')
    axes[0].set_title('Share of within-field yield variance per spatial scale', fontsize=10)
    axes[0].set_ylabel('share')
    for src, label, col, ls in SRC:
        s = totals(src)
        axes[1].plot(x, s['py'] / np.sqrt(s['pp'] * s['yy']), ls, color=col, marker='o', label=label)
        if not src.startswith('input'):
            axes[2].plot(x, 1 - s['ee'] / s['yy'], ls, color=col, marker='o', label=label)
    axes[1].set_title('Correlation with the target, per scale band', fontsize=10)
    axes[2].set_title('Variance explained, per scale band', fontsize=10)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(B, rotation=20, fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlabel('spatial scale (difference-of-Gaussians band)')
    axes[1].set_ylim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[1].legend(fontsize=8, loc='lower right')
    fig.suptitle('YieldSAT CV10, 9 pairs pooled: within-field variability by spatial scale', fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / 'scale_decomposition.png', dpi=110)


if __name__ == '__main__':
    main()
