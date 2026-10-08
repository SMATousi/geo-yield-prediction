"""Semivariogram figures for spec/yieldsat-field-context.md §3 from results/noise_ceiling.json.

    python results/figures/semivariogram/plot_semivariogram.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = json.loads((HERE.parents[1] / 'noise_ceiling.json').read_text())
PAIRS = ['ARG-C', 'ARG-S', 'ARG-W', 'BRA-C', 'BRA-S', 'BRA-W', 'GER-R', 'GER-W', 'URG-S']
CELL_M = 10
# dense p3-nbr CV10 within-field R2 (spec/yieldsat-relational-loss.md §4)
MODEL = {'ARG-C': 0.45, 'ARG-S': 0.52, 'ARG-W': 0.37, 'BRA-C': 0.28, 'BRA-S': 0.23, 'BRA-W': 0.10,
         'GER-R': 0.23, 'GER-W': 0.22, 'URG-S': 0.10}


def curve(pair):
    g = RES[pair]['gamma']
    h = np.array(sorted(int(k) for k in g))
    return h, np.array([g[str(k)] for k in h], dtype=float)


def panel(ax, pair, max_lag=30, legend=False):
    h, g = curve(pair)
    keep = h <= max_lag
    r = RES[pair]
    nug, slope = r['nugget_frac'], r['fit_slope_frac']
    ax.axhspan(0, nug, color='tab:red', alpha=0.12, lw=0, label='noise (nugget)')
    ax.axhspan(nug, 1, color='tab:green', alpha=0.07, lw=0, label='predictable within-field variance')
    ax.axhline(1, color='k', lw=0.8, ls=':', label='within-field variance')
    x = np.array([0, 3])
    ax.plot(x * CELL_M, nug + slope * x, color='tab:red', lw=1.3, ls='--', label='linear fit, lags 1-3')
    ax.plot(h[keep] * CELL_M, g[keep], 'o-', ms=3, lw=1.2, color='tab:blue', label='semivariance γ(h) / variance')
    ax.plot([0], [nug], 's', color='tab:red', ms=6)
    ax.set_title('{}  nugget {:.2f} -> ceiling {:.2f}\n(dense p3-nbr CV10 within-field R² {:.2f})'.format(
        pair, nug, 1 - nug, MODEL[pair]), fontsize=9)
    ax.set_xlim(0, max_lag * CELL_M)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)
    if legend:
        ax.legend(fontsize=7, loc='lower right')


def main():
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True, sharey=True)
    for ax, pair in zip(axes.ravel(), PAIRS):
        panel(ax, pair, legend=(pair == 'ARG-C'))
    for ax in axes[-1]:
        ax.set_xlabel('lag h (m)')
    for ax in axes[:, 0]:
        ax.set_ylabel('γ(h) / within-field variance')
    fig.suptitle('Within-field semivariograms of the yield targets (rows and columns, pooled over field seasons)', fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / 'semivariogram_pairs.png', dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for pair in PAIRS:
        h, g = curve(pair)
        axes[0].plot(h * CELL_M, g, 'o-', ms=2.5, lw=1.1, label=pair)
        k = h <= 4
        axes[1].plot(h[k] * CELL_M, g[k], 'o-', ms=3.5, lw=1.1, label=pair)
        axes[1].plot([0, 10], [RES[pair]['nugget_frac'], g[0]], ':', lw=0.9, color=axes[1].lines[-1].get_color())
    axes[0].axhline(1, color='k', lw=0.8, ls=':')
    axes[0].axvspan(0, 20, color='0.85', alpha=0.6, lw=0)
    axes[0].text(22, 0.08, '5x5 neighbourhood\n(±20 m)', fontsize=8)
    axes[0].set_title('All pairs, lags to 300 m', fontsize=10)
    axes[1].set_title('Short lags and extrapolation to 0 (nugget)', fontsize=10)
    for ax in axes:
        ax.set_xlabel('lag h (m)')
        ax.set_ylabel('γ(h) / within-field variance')
        ax.set_ylim(0, 1.1)
        ax.set_xlim(0, None)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=3, loc='lower right')
    fig.tight_layout()
    fig.savefig(HERE / 'semivariogram_overview.png', dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(PAIRS))
    ceil = np.array([1 - RES[p]['nugget_frac'] for p in PAIRS])
    mod = np.array([MODEL[p] for p in PAIRS])
    ax.bar(x, ceil, color='tab:green', alpha=0.35, label='noise ceiling (1 - nugget share)')
    ax.bar(x, mod, color='tab:blue', alpha=0.85, width=0.5, label='dense p3-nbr CV10 within-field R²')
    for i in range(len(PAIRS)):
        ax.text(i, ceil[i] + 0.01, '{:.0%}'.format(mod[i] / ceil[i]), ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(PAIRS)
    ax.set_ylim(0, 1)
    ax.set_ylabel('within-field R²')
    ax.set_title('Within-field R² reached vs noise ceiling (label: share of ceiling)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / 'ceiling_vs_model.png', dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
