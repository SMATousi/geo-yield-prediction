"""Information budget figure (paper Q3, sec:information): how the within-field yield variance splits into
transferable input information, field-specific input information, signal not in the inputs and measurement noise,
and how far every model gets. CV10, mean over the nine pairs, on the oracle's field seasons (>= 200 cells), so the
models and the three levels of tab:levels are on the same fields.

    python results/figures/input_information/plot_information_budget.py <out dir>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
ORC, MI, NC = (json.loads((R / f).read_text()) for f in ('input_oracle.json', 'input_mi.json', 'noise_ceiling.json'))
PAIRS = ['ARG-C', 'ARG-S', 'ARG-W', 'BRA-C', 'BRA-S', 'BRA-W', 'GER-R', 'GER-W', 'URG-S']
MODELS = [('paper_3dconvlstm', '3D-ConvLSTM'), ('paper_3dlstm', '3D-LSTM'), ('paper_aff', 'AFF'),
          ('paper_mmgf', 'MMGF'), ('lgbm-f0-dense', 'LightGBM'), ('ours-p3nbr-dense', 'Yield-ReFiNe (base)'),
          ('tabm-f0-dense-bs2048', 'TabM'), ('ours-p3nbr-dense-fc-rel10', 'Yield-ReFiNe')]
OURS = 'ours-p3nbr-dense-fc-rel10'
# one hue (the paper's purple), dark = most accessible information; ink and baseline bars neutral
PURPLE, MID, LIGHT, PALE = '#4E4460', '#8A7EA3', '#C4BBD4', '#E6E1EE'
INK, MUTED, GREY = '#222222', '#6B6B6B', '#A7AFB4'

mean = lambda f: float(np.mean([f(p) for p in PAIRS]))
transfer = mean(lambda p: MI[p]['pooled']['d8_k5']['r2_eq'])          # cross-field information (KSG, pooled)
oracle = mean(lambda p: ORC[p]['gbm_B8'])                             # per-field information (per-field GBM)
mi_field = mean(lambda p: MI[p]['per_field']['d8_k5']['r2_eq_median'])
ceiling = mean(lambda p: NC[p]['ceiling_within_r2'])                  # 1 - nugget share


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
                         'xtick.color': MUTED, 'ytick.color': INK, 'hatch.linewidth': 0.5})
    fig, (top, ax) = plt.subplots(2, 1, figsize=(3.35, 2.9), sharex=True,
                                  gridspec_kw={'height_ratios': [1, 3.3], 'hspace': 0.08})

    # budget bar: the target's within-field variance = 1
    segs = [(0, transfer, PURPLE, 'transferable\nacross fields', 'w'), (transfer, oracle, MID, 'field-\nspecific', 'w'),
            (oracle, ceiling, LIGHT, 'not in\ninputs', INK), (ceiling, 1, PALE, 'noise', INK)]
    for a, b, c, lab, tc in segs:
        top.barh(0, b - a - 0.004, left=a + 0.002, height=0.62, color=c, lw=0,
                 hatch='////' if lab == 'noise' else None, edgecolor=MUTED if lab == 'noise' else None)
        top.text((a + b) / 2, 0, lab, ha='center', va='center', color=tc, fontsize=6, linespacing=0.95,
                 bbox=dict(boxstyle='square,pad=0.15', fc=PALE, ec='none') if lab == 'noise' else None)
    top.set_ylim(-0.5, 0.5); top.set_yticks([0]); top.set_yticklabels(['Within-field\nvariance'])
    top.tick_params(axis='y', length=0)
    for s in ('top', 'right', 'left'):
        top.spines[s].set_visible(False)
    top.tick_params(axis='x', length=0)

    # models on the same fields
    vals = [(lab, mean(lambda p: ORC[p][t]), t == OURS) for t, lab in MODELS]
    vals.sort(key=lambda v: v[1])
    y = np.arange(len(vals))
    for i, (lab, v, ours) in enumerate(vals):
        ax.barh(i, v, height=0.62, color=PURPLE if ours else GREY, lw=0)
        ax.text(v + 0.008, i, '%.2f' % v, va='center', fontsize=6.5, color=INK, weight='bold' if ours else 'normal',
                bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'), zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([v[0] for v in vals])
    for t, (_, _, ours) in zip(ax.get_yticklabels(), vals):
        t.set_fontweight('bold' if ours else 'normal')
    ax.tick_params(axis='y', length=0)

    # the levels, carried through both panels
    for x, ls in ((transfer, '-'), (oracle, '--'), (ceiling, ':')):
        for a in (top, ax):
            a.axvline(x, color=INK if ls == '-' else MUTED, lw=0.8, ls=ls, zorder=0 if a is ax else 3)
    for x in (transfer, oracle, ceiling):          # named in the caption: cross-field MI, per-field oracle, noise ceiling
        top.text(x, 0.4, '%.2f' % x, ha='center', va='bottom', fontsize=6.5, color=INK if x == transfer else MUTED)

    ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(vals) - 0.4)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel('Within-field $R^2$ (CV10, nine pairs)')
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='x', color='#E4E4E4', lw=0.5, zorder=-1); ax.set_axisbelow(True)

    fig.savefig(out / 'information_budget.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(out / 'information_budget.png', bbox_inches='tight', pad_inches=0.02, dpi=300)
    print(json.dumps({'transferable_mi': transfer, 'oracle': oracle, 'mi_field': mi_field, 'ceiling': ceiling,
                      'models': {l: round(v, 4) for l, v, _ in vals}}))


if __name__ == '__main__':
    main()
