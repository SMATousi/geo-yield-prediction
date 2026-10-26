"""Real-data panels for the method overview figure (paper figures/crop_yield_overview.tikz).

One YieldSAT field season (the ARG-S field of the qualitative figure): three in-season Sentinel-2 true-colour
acquisitions (monthly cache: first and last in-season slot, then peak NDVI in front), the front one with the 5x5 neighbourhood and cell i marked,
a magnified copy of
that window, and the field's CV10 test prediction of our model. Transparent outside the field; pixels upsampled by nearest
neighbour so the 10 m cells stay visible.

    python results/figures/method/render_overview_samples.py <artifact root> <results root with paper/> <out dir>
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ART, RES, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
COUNTRY, PAIR, FIELD = 'Argentina', 'ARG-S', 'Argentina_DUP1_farm52_field691_soybean_2020'
TAG, GROUP = 'ours-p3nbr-dense-fc-rel10', 'cv10_season_noval_s0'
B02, B03, B04, B08 = 1, 2, 3, 7                     # cache temporal channels B01..B12, weather
UP = 8                                              # output pixels per 10 m cell


def rgba(img, mask):
    out = np.zeros(mask.shape + (4,))
    out[..., :3] = np.nan_to_num(img)
    out[..., 3] = mask
    return np.kron(out, np.ones((UP, UP, 1)))


def save(a, path):
    plt.imsave(path, np.clip(a, 0, 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = [r for r in json.loads((ART / 'index' / COUNTRY / 'fields.json').read_text()) if r['field_shared_name'] == FIELD][0]
    s, e = rec['row_start'], rec['row_end']
    idx = np.load(ART / 'index' / COUNTRY / 'rows.npz')
    rows, cols = idx['grid_row'][s:e].astype(int), idx['grid_col'][s:e].astype(int)
    r0, c0 = rows.min(), cols.min()
    H, W = rows.max() - r0 + 1, cols.max() - c0 + 1
    mask = np.zeros((H, W)); mask[rows - r0, cols - c0] = 1
    t = np.load(ART / 'cache' / COUNTRY / 'temporal.npy', mmap_mode='r')[s:e].astype(np.float32)
    times = np.load(ART / 'cache' / COUNTRY / 'times.npy', mmap_mode='r')[s:e]

    # in-season slots observed on >= 90% of the cells; early / mid / peak by field-median NDVI
    ndvi = (t[:, :, B08] - t[:, :, B04]) / (t[:, :, B08] + t[:, :, B04])
    ok = [k for k in range(t.shape[1]) if np.isfinite(ndvi[:, k]).mean() >= 0.9
          and rec['seeding_day'] <= np.nanmedian(times[:, k]) <= rec['harvest_day']]
    med = np.array([np.nanmedian(ndvi[:, k]) for k in ok])
    peak = int(np.argmax(med))
    picks = [ok[0], ok[-1], ok[peak]]                  # back to front: first and last in-season slot, peak
    rgb_all = t[:, :, [B04, B03, B02]]
    meta = {'field': FIELD, 'slots': picks, 'median_ndvi': [float(np.nanmedian(ndvi[:, k])) for k in picks]}
    from scipy.ndimage import binary_erosion
    inner = np.argwhere(binary_erosion(mask > 0, np.ones((9, 9))))
    ci, cj = inner[np.argmin(((inner - inner.mean(0)) ** 2).sum(1))]       # cell i: central, 5x5 window inside
    meta['cell'] = [int(ci), int(cj)]
    interior = binary_erosion(mask > 0, np.ones((5, 5)))[rows - r0, cols - c0]
    for j, k in enumerate(picks):
        v = rgb_all[:, k]                               # per date one soft stretch over interior cells (edges mixed)
        lo, hi = np.nanpercentile(v[interior], 1), 1.25 * np.nanpercentile(v[interior], 99.5)
        x = np.clip((v - lo) / (hi - lo), 0, 1)
        g = x.mean(1, keepdims=True)
        img = np.full((H, W, 3), np.nan)
        img[rows - r0, cols - c0] = np.clip(g + 0.7 * (x - g), 0, 1) ** 0.9      # slightly desaturated
        a = rgba(img, mask)
        if j == len(picks) - 1:                       # front image: the 5x5 window, white outline
            y0, y1, x0, x1 = (ci - 2) * UP, (ci + 3) * UP, (cj - 2) * UP, (cj + 3) * UP
            for w in range(4):
                a[y0 + w, x0:x1, :3] = 1; a[y1 - 1 - w, x0:x1, :3] = 1
                a[y0:y1, x0 + w, :3] = 1; a[y0:y1, x1 - 1 - w, :3] = 1
            z = np.kron(img[ci - 2:ci + 3, cj - 2:cj + 3], np.ones((40, 40, 1)))   # magnified window
            z[::40, :] = 1; z[:, ::40] = 1; z[-1, :] = 1; z[:, -1] = 1
            for w in range(4):                          # cell i
                z[80 + w, 80:120] = z[119 - w, 80:120] = z[80:120, 80 + w] = z[80:120, 119 - w] = (0.26, 0.33, 0.36)
            save(z, OUT / 'overview_s2_zoom.png')
        save(a, OUT / 'overview_s2_{}.png'.format(j))

    # prediction of the CV10 test fold that holds the field
    for f in sorted(glob.glob('%s/paper/%s/s2_adm/%s_seed0/%s/fold*/test_predictions.npz' % (RES, GROUP, TAG, PAIR))):
        z = np.load(f)
        names = z['season_names'].astype(str)
        if FIELD in names:
            sel = (z['season'] == list(names).index(FIELD)) & np.isfinite(z['pred'])
            pr, pc, p, y = z['grid_row'][sel], z['grid_col'][sel], z['pred'][sel], z['target'][sel]
            break
    vlo, vhi = np.percentile(y[np.isfinite(y)], [2, 98])
    cmap = plt.get_cmap('RdYlGn')
    img = np.full((H, W, 3), np.nan); m2 = np.zeros((H, W))
    img[pr - r0, pc - c0] = cmap(np.clip((p - vlo) / (vhi - vlo), 0, 1))[:, :3]
    m2[pr - r0, pc - c0] = 1
    save(rgba(img, m2), OUT / 'overview_pred.png')
    meta['pred_cells'] = int(sel.sum()); meta['yield_range_t_ha'] = [float(vlo), float(vhi)]
    (OUT / 'overview_samples.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
