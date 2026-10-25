"""PlanetScope vs Sentinel-2 registration on the YieldSAT subset (spec/yieldsat-planetscope.md §2).

Per field season and dense slot where both are observed on >= 500 cells: the Pearson correlation across cells of
S2 NDVI (dense cache) with PlanetScope NDVI, the cell mean of the 3x3 sub-pixels from the window stack, shifted
by (dy, dx) sub-pixels of 3.33 m (|dy|, |dx| <= 3, i.e. up to one 10 m cell). Per field the mean correlation over
its slots; the best shift tells how far PlanetScope sits from the yield / S2 grid.

    python results/ps_registration_qc.py /data/YieldSAT/yieldsat_artifacts results/ps_registration_qc.json
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

root, out = Path(sys.argv[1]), sys.argv[2]
SH = range(-3, 4)
B04, B08 = 3, 7                                   # canonical optical order B01..B12
RED, NIR = 5, 7                                   # PlanetScope band order (8 bands; 4-band scenes fill these too)
res = {}
for country in ('Argentina', 'Brazil'):
    d = root / 'neighbourhood_dense' / country
    man = json.loads((root / 'neighbourhood_dense' / 'ps_subset_manifest.json').read_text())['fields']
    idx = np.load(root / 'index' / country / 'rows.npz')
    recs = {r['field_shared_name']: r for r in json.loads((root / 'index' / country / 'fields.json').read_text())}
    temporal = np.load(root / 'cache_dense' / country / 'temporal.npy', mmap_mode='r')
    scene_all = np.load(d / 'ps_win_scene.npy', mmap_mode='r')
    for name, m in man.items():
        if name not in recs or not m['scenes']:
            continue
        r = recs[name]
        s, e = r['row_start'], r['row_end']
        rows, cols = idx['grid_row'][s:e].astype(np.int64), idx['grid_col'][s:e].astype(np.int64)
        scene = np.asarray(scene_all[s:e])
        t = np.asarray(temporal[s:e], np.float32)
        with np.errstate(invalid='ignore', divide='ignore'):
            s2 = (t[:, :, B08] - t[:, :, B04]) / (t[:, :, B08] + t[:, :, B04])
        stack = np.load(d / 'ps_win' / '{}.npy'.format(m['field_code']), mmap_mode='r')
        corr = np.zeros((len(SH), len(SH))); n_slots = 0
        for k in range(scene.shape[1]):
            ok = (scene[:, k] >= 0) & np.isfinite(s2[:, k])
            if ok.sum() < 500:
                continue
            sc = np.unique(scene[ok, k])
            if len(sc) != 1:                           # one scene per slot keeps the comparison simple
                ok &= scene[:, k] == sc[0]
                if ok.sum() < 500:
                    continue
            img = np.asarray(stack[int(scene[ok, k][0])], np.float32)
            with np.errstate(invalid='ignore', divide='ignore'):
                nd = (img[:, :, NIR] - img[:, :, RED]) / (img[:, :, NIR] + img[:, :, RED])
            n_slots += 1
            for a, dy in enumerate(SH):
                for b, dx in enumerate(SH):
                    R = (3 * (rows[ok] + 1) + dy)[:, None] + np.arange(3)
                    C = (3 * (cols[ok] + 1) + dx)[:, None] + np.arange(3)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        p = np.nanmean(nd[R[:, :, None], C[:, None, :]].reshape(len(R), 9), axis=1)
                    q = np.isfinite(p)
                    corr[a, b] += np.corrcoef(p[q], s2[ok, k][q])[0, 1] if q.sum() > 100 else np.nan
        if n_slots:
            corr /= n_slots
            a, b = np.unravel_index(np.nanargmax(corr), corr.shape)
            res[name] = {'pair': {'soybean': 'ARG-S', 'corn': 'ARG-C' if country == 'Argentina' else 'BRA-C'}[r['crop']],
                         'slots': n_slots, 'r_at_0': float(corr[3, 3]), 'r_best': float(corr[a, b]),
                         'best_dy_m': SH[a] * 10 / 3, 'best_dx_m': SH[b] * 10 / 3}
Path(out).write_text(json.dumps(res, indent=1))
v = list(res.values())
off = np.array([np.hypot(x['best_dy_m'], x['best_dx_m']) for x in v])
print('fields', len(v), 'median r at 0 %.3f best %.3f' % (np.median([x['r_at_0'] for x in v]), np.median([x['r_best'] for x in v])))
print('best shift 0: %.1f%%, <= 3.3 m: %.1f%%, <= 4.7 m: %.1f%%, median %.1f m' % (
    100 * (off == 0).mean(), 100 * (off <= 3.34).mean(), 100 * (off <= 4.72).mean(), np.median(off)))
