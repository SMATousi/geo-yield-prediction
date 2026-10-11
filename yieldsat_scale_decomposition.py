"""IC-B (spec/yieldsat-input-information.md §3): at which spatial scales do models (and the inputs) capture within-field
yield variability?

Per test field season, the centred target y and centred prediction yhat are placed on the 10 m grid and decomposed into
bands with differences of Gaussians (normalised convolution, so missing cells are handled):
    band 0 = x - G1(x), band k = G_sk(x) - G_s(k+1)(x), last band = G16(x); sigma in cells = 1, 2, 4, 8, 16
Per band, pooled over fields: target energy share sum(y_b^2) / sum(y^2), explained variance 1 - sum((yhat_b - y_b)^2) /
sum(y_b^2) and correlation corr(yhat_b, y_b). The same correlation for an input summary (peak-season NDVI from the
monthly cache) measures whether the inputs themselves carry the band's signal.

    python yieldsat_scale_decomposition.py --results_root /root/yieldsat_backups/protocol_results \\
        --artifact_root /root/yieldsat_artifacts --models ours-p3nbr-dense-fc-rel10 ours-p3nbr-dense --out results/scale_decomposition.json
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from dataset.yieldsat_schema import TEMPORAL_CHANNELS

SIGMAS = (1, 2, 4, 8, 16)
# band labels = wavelength range, from the Gaussians' half-amplitude wavelengths (pi * sigma / sqrt(ln2 / 2) = 5.3 sigma
# cells: 53, 107, 213, 427, 854 m) and the 10 m grid's shortest representable wavelength (20 m, two cells).
# Uncorrelated cell-to-cell noise (the nugget) falls 94% into the first band.
BANDS = ('20-50 m', '50-110 m', '110-210 m', '210-430 m', '430-850 m', '>850 m')
COUNTRY = {'ARG': 'Argentina', 'BRA': 'Brazil', 'GER': 'Germany', 'URG': 'Uruguay'}
PAIRS = ('ARG-C', 'ARG-S', 'ARG-W', 'BRA-C', 'BRA-S', 'BRA-W', 'GER-R', 'GER-W', 'URG-S')
B04, B08 = TEMPORAL_CHANNELS.index('B04'), TEMPORAL_CHANNELS.index('B08')
MIN_CELLS = 50


def smooth(grid, mask, sigma):
    num = gaussian_filter(np.where(mask, grid, 0.0), sigma, mode='constant')
    den = gaussian_filter(mask.astype(np.float64), sigma, mode='constant')
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 1e-6, num / den, 0.0)


def bands(values, rr, cc):
    """values (n,) on cells (rr, cc) -> (len(BANDS), n) band components at the cells."""
    H, W = rr.max() + 1, cc.max() + 1
    g = np.zeros((H, W))
    m = np.zeros((H, W), bool)
    g[rr, cc], m[rr, cc] = values, True
    levels = [g] + [smooth(g, m, s) for s in SIGMAS]
    out = [levels[k] - levels[k + 1] for k in range(len(SIGMAS))] + [levels[-1]]
    return np.stack([b[rr, cc] for b in out])


def peak_ndvi(cache, rows):
    t = np.asarray(cache[rows])
    with np.errstate(invalid='ignore', divide='ignore'):
        nd = (t[:, :, B08] - t[:, :, B04]) / (t[:, :, B08] + t[:, :, B04])
    nd = np.where(np.isfinite(nd), nd, np.nan)
    return np.nanmax(np.where(np.isnan(nd), -np.inf, nd), axis=1)


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--results_root', required=True, help='directory containing paper/<group>/s2_adm/<tag>_seed0')
    pa.add_argument('--artifact_root', required=True)
    pa.add_argument('--models', nargs='+', required=True)
    pa.add_argument('--group', default='cv10_season_noval_s0')
    pa.add_argument('--out', required=True)
    a = pa.parse_args()
    nb = len(BANDS)
    acc = {}                                      # (pair, source) -> sums per band

    def add(key, yb, pb):
        s = acc.setdefault(key, {'yy': np.zeros(nb), 'pp': np.zeros(nb), 'py': np.zeros(nb), 'ee': np.zeros(nb),
                                 'tot': 0.0, 'fields': 0})
        s['yy'] += (yb ** 2).sum(1); s['pp'] += (pb ** 2).sum(1); s['py'] += (pb * yb).sum(1)
        s['ee'] += ((pb - yb) ** 2).sum(1); s['fields'] += 1

    index, caches = {}, {}
    for pair in PAIRS:
        country = COUNTRY[pair[:3]]
        if country not in index:
            fl = json.loads((Path(a.artifact_root) / 'index' / country / 'fields.json').read_text())
            with np.load(Path(a.artifact_root) / 'index' / country / 'rows.npz') as z:
                rows = {k: z[k] for k in ('grid_row', 'grid_col')}
            index[country] = ({f['field_shared_name']: f for f in fl}, rows)
            caches[country] = np.load(Path(a.artifact_root) / 'cache' / country / 'temporal.npy', mmap_mode='r')
        fmap, rows = index[country]
        ndvi_done = set()
        for tag in a.models:
            for f in sorted(glob.glob('%s/paper/%s/s2_adm/%s_seed0/%s/fold*/test_predictions.npz'
                                      % (a.results_root, a.group, tag, pair))):
                z = np.load(f)
                names = z['season_names'].astype(str)
                for si in np.unique(z['season']):
                    sel = (z['season'] == si) & np.isfinite(z['pred']) & np.isfinite(z['target'])
                    if sel.sum() < MIN_CELLS:
                        continue
                    y = z['target'][sel].astype(np.float64)
                    p = z['pred'][sel].astype(np.float64)
                    rr, cc = z['grid_row'][sel].astype(np.int64), z['grid_col'][sel].astype(np.int64)
                    rr, cc = rr - rr.min(), cc - cc.min()
                    yb = bands(y - y.mean(), rr, cc)
                    add((pair, tag), yb, bands(p - p.mean(), rr, cc))
                    name = names[si]
                    if name not in ndvi_done and name in fmap:          # input side, once per field season
                        fr = fmap[name]
                        a0, b0 = fr['row_start'], fr['row_end']
                        key = dict(zip(zip(rows['grid_row'][a0:b0], rows['grid_col'][a0:b0]), range(a0, b0)))
                        gr0, gc0 = z['grid_row'][sel], z['grid_col'][sel]
                        ridx = np.array([key.get((r, c), -1) for r, c in zip(gr0, gc0)])
                        if (ridx >= 0).all():
                            nd = peak_ndvi(caches[country], ridx)
                            ok = np.isfinite(nd)
                            if ok.sum() >= MIN_CELLS:
                                nd = np.where(ok, nd, np.nanmean(nd))
                                add((pair, 'input:peak_ndvi'), yb, bands(nd - nd.mean(), rr, cc))
                        ndvi_done.add(name)
        print(pair, 'done', flush=True)
    res = {}
    for (pair, src), s in acc.items():
        res['%s|%s' % (pair, src)] = {
            'fields': s['fields'],
            'target_share': (s['yy'] / s['yy'].sum()).tolist(),
            'explained': (1 - s['ee'] / s['yy']).tolist() if not src.startswith('input:') else None,
            'corr': (s['py'] / np.sqrt(s['pp'] * s['yy'])).tolist(),
            'sums': {k: s[k].tolist() for k in ('yy', 'pp', 'py', 'ee')}}
    json.dump({'bands': BANDS, 'sigmas_cells': SIGMAS, 'results': res}, open(a.out, 'w'), indent=1)
    print('SCALE_DONE')


if __name__ == '__main__':
    main()
