"""IC-C figure (spec/yieldsat-input-information.md §4) from results/coarse_dev_icc.json: within-field and pooled R2
of the best model vs Sentinel-2 input resolution (10 m = the uncoarsened runs), CV10, four DEV pairs.

    python results/figures/input_information/plot_resolution.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
D = json.loads((HERE.parents[1] / 'coarse_dev_icc.json').read_text())
OURS = 'ours-p3nbr-dense-fc-rel10'
DEV = ['ARG-W', 'BRA-C', 'GER-R', 'URG-S']
RES = [(1, 10), (2, 20), (3, 30), (6, 60)]


def series(pair, metric):
    v = D['CV10|%s' % pair]
    base = v['baseline'][metric]
    ys, lo, hi = [base], [base], [base]
    for k, _ in RES[1:]:
        m = v['%s-c%d' % (OURS, k)]
        ys.append(base + m['d_' + metric]); lo.append(base + m['ci_' + metric][0]); hi.append(base + m['ci_' + metric][1])
    return np.array(ys), np.array(lo), np.array(hi)


def main():
    x = [r for _, r in RES]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, metric, title in ((axes[0], 'within_r2', 'Within-field $R^2$'), (axes[1], 'pooled_r2', 'Pooled $R^2$')):
        for pair in DEV:
            y, _, _ = series(pair, metric)
            ax.plot(x, y, 'o-', lw=1, ms=3, alpha=0.6, label=pair)
        base = np.mean([D['CV10|%s' % p]['baseline'][metric] for p in DEV])
        mean = [base] + [base + D['CV10|MEAN|%s-c%d' % (OURS, k)][metric][0] for k, _ in RES[1:]]
        err = np.array([[0, 0]] + [[D['CV10|MEAN|%s-c%d' % (OURS, k)][metric][0] - D['CV10|MEAN|%s-c%d' % (OURS, k)][metric][1],
                                    D['CV10|MEAN|%s-c%d' % (OURS, k)][metric][2] - D['CV10|MEAN|%s-c%d' % (OURS, k)][metric][0]]
                                   for k, _ in RES[1:]]).T
        ax.errorbar(x, mean, yerr=err, fmt='s-', color='k', lw=2, ms=5, capsize=3, label='mean (95% CI)')
        ax.set_xticks(x); ax.set_xlabel('Sentinel-2 input resolution (m)'); ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / 'resolution.png', dpi=150)


if __name__ == '__main__':
    main()
