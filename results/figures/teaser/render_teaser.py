"""Page-1 teaser (paper figures/teaser.tikz): pooled R2 rewards field means, within-field R2 scores the pattern.

Three ARG-S CV10 test field seasons (seed 0) on one colour scale: ground truth, the field-mean map (each field
constant at its true mean), MMGF and our model. Field selection is fixed by rule, not by eye: among the test field
seasons with >= MIN_CELLS cells and predictions from both models, split into terciles of the true field-mean
yield; in each tercile the field at the 75th percentile of our model's within-field R2 (good fields, not the best,
as in the Missouri figure). Writes one transparent PNG per row of the figure (the three fields side by side at a
common scale, nearest-neighbour upsampled) and teaser_samples.json.

    python results/figures/teaser/render_teaser.py <results root with paper/> <out dir>
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
PAIR, GROUP = 'ARG-S', 'cv10_season_noval_s0'
OURS, MMGF = 'ours-p3nbr-dense-fc-rel10', 'paper_mmgf'
MIN_CELLS, Q, UP, GAP = 2000, 0.75, 4, 6                  # GAP: empty cells between fields


def load(tag):
    out = {}
    for f in sorted(glob.glob('%s/paper/%s/s2_adm/%s_seed0/%s/fold*/test_predictions.npz' % (ROOT, GROUP, tag, PAIR))):
        z = np.load(f)
        names = z['season_names'].astype(str)
        for si in np.unique(z['season']):
            sel = (z['season'] == si) & np.isfinite(z['target']) & np.isfinite(z['pred'])
            out[names[si]] = (z['grid_row'][sel].astype(int), z['grid_col'][sel].astype(int),
                              z['target'][sel].astype(float), z['pred'][sel].astype(float))
    return out


def within_r2(y, p):
    yc, pc = y - y.mean(), p - p.mean()
    return float(1 - np.sum((pc - yc) ** 2) / np.sum(yc ** 2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ours, mmgf = load(OURS), load(MMGF)
    cands = [k for k, v in ours.items() if len(v[2]) >= MIN_CELLS and k in mmgf]
    means = np.array([ours[k][2].mean() for k in cands])
    t1, t2 = np.quantile(means, [1 / 3, 2 / 3])
    picks = []
    for lo, hi in ((-np.inf, t1), (t1, t2), (t2, np.inf)):
        grp = sorted((within_r2(ours[k][2], ours[k][3]), k) for k, m in zip(cands, means) if lo <= m < hi)
        picks.append(grp[int(round(Q * (len(grp) - 1)))][1])
    picks.sort(key=lambda k: ours[k][2].mean())                    # low to high mean yield, left to right
    meta = {'pair': PAIR, 'candidates': len(cands), 'fields': []}
    for k in picks:
        r, c, y, p = ours[k]
        m = mmgf[k]
        assert len(m[2]) == len(y) and (m[0] == r).all() and (m[1] == c).all(), k
        meta['fields'].append({'name': k, 'cells': len(y), 'mean_t_ha': float(y.mean()),
                               'shape': [int(r.max() - r.min() + 1), int(c.max() - c.min() + 1)],
                               'within_r2_ours': within_r2(y, p), 'within_r2_mmgf': within_r2(y, m[3])})
    ally = np.concatenate([ours[k][2] for k in picks])
    vlo, vhi = np.percentile(ally, [2, 98])
    meta['scale_t_ha'] = [float(vlo), float(vhi)]
    cmap = plt.get_cmap('RdYlGn')
    # the three fields side by side, bottom-aligned, on one canvas per row
    shapes = [f['shape'] for f in meta['fields']]
    H, W = max(s[0] for s in shapes), sum(s[1] for s in shapes) + GAP * (len(shapes) - 1)
    rows = {'truth': lambda k: ours[k][2], 'fieldmean': lambda k: np.full(len(ours[k][2]), ours[k][2].mean()),
            'mmgf': lambda k: mmgf[k][3], 'ours': lambda k: ours[k][3]}
    for name, val in rows.items():
        img = np.zeros((H, W, 4))
        x0 = 0
        for k, (h, w) in zip(picks, shapes):
            r, c = ours[k][0], ours[k][1]
            rr, cc = r - r.min() + (H - h), c - c.min() + x0
            img[rr, cc, :3] = cmap(np.clip((val(k) - vlo) / (vhi - vlo), 0, 1))[:, :3]
            img[rr, cc, 3] = 1
            x0 += w + GAP
        plt.imsave(OUT / 'teaser_{}.png'.format(name), np.kron(img, np.ones((UP, UP, 1))))
    bar = cmap(np.linspace(0, 1, 256))[None, :, :3].repeat(12, 0)
    plt.imsave(OUT / 'teaser_cbar.png', bar)
    meta['canvas_cells'] = [H, W]
    (OUT / 'teaser_samples.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
