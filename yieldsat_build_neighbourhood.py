"""S6 neighbourhood stream for the point model (spec/yieldsat-improvement.md, round 2).

For every cache row (one cell of one field season) and every time slot, the
masked mean of the 12 S2 bands over the surrounding 5x5 cells of the same
field season, centre excluded, from the raw (un-normalized) cache values.
Missing neighbours (outside the field, no observation) are ignored; a cell with
no valid neighbour gets NaN. Output ``<artifact_root>/neighbourhood/<Country>/
s2_nbr5.npy`` (rows, 24, 12) float16, aligned with ``cache/<Country>/
temporal.npy``. Features use inputs only (no labels, no split), like the cache.

    python yieldsat_build_neighbourhood.py --artifact_root /data/YieldSAT/yieldsat_artifacts
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from dataset.yieldsat_schema import COUNTRIES, STREAMS, TEMPORAL_CHANNELS

S2_POS = [TEMPORAL_CHANNELS.index(c) for c in STREAMS['yieldsat_s2']['channels']]
RADIUS = 2                                                   # 5x5 window


def box_sum(a, r=RADIUS):
    """Sum over a (2r+1)^2 window on axes 0/1 of (H, W, ...) via 2-D cumsums."""
    p = np.pad(a, ((r + 1, r), (r + 1, r)) + ((0, 0),) * (a.ndim - 2))
    c = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]


def field_neighbourhood(values, rows, cols):
    """values (n, T, C) raw S2 of one field season's cells at grid (rows, cols)
    -> (n, T, C) masked mean of the 5x5 neighbours, centre excluded."""
    r0, c0 = rows.min(), cols.min()
    H, W = rows.max() - r0 + 1, cols.max() - c0 + 1
    grid = np.zeros((H, W) + values.shape[1:], np.float64)
    cnt = np.zeros_like(grid)
    ok = np.isfinite(values)
    rr, cc = rows - r0, cols - c0
    grid[rr, cc] = np.where(ok, values, 0.0)
    cnt[rr, cc] = ok
    s = box_sum(grid)[rr, cc] - grid[rr, cc]
    n = box_sum(cnt)[rr, cc] - cnt[rr, cc]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def build_country(artifact_root, country):
    root = Path(artifact_root)
    temporal = np.load(root / 'cache' / country / 'temporal.npy', mmap_mode='r')
    rows = np.load(root / 'index' / country / 'rows.npz')
    fields = json.loads((root / 'index' / country / 'fields.json').read_text())
    out_dir = root / 'neighbourhood' / country
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / 's2_nbr5.npy'
    tmp = out_dir / 's2_nbr5.npy.partial'
    out = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float16,
                                    shape=(temporal.shape[0], temporal.shape[1], len(S2_POS)))
    out[:] = np.nan
    t0 = time.time()
    for i, f in enumerate(fields):
        a, b = f['row_start'], f['row_end']
        vals = np.asarray(temporal[a:b])[..., S2_POS].astype(np.float64)
        out[a:b] = field_neighbourhood(vals, rows['grid_row'][a:b], rows['grid_col'][a:b]).astype(np.float16)
        if (i + 1) % 200 == 0:
            print(country, i + 1, 'of', len(fields), 'fields, %.0f s' % (time.time() - t0), flush=True)
    out.flush()
    del out
    os.replace(tmp, final)
    manifest = {'country': country, 'rows': int(temporal.shape[0]), 'window': 2 * RADIUS + 1,
                'centre_excluded': True, 'channels': list(STREAMS['yieldsat_s2']['channels']),
                'dtype': 'float16', 'seconds': round(time.time() - t0, 1)}
    (out_dir / 'neighbourhood_manifest.json').write_text(json.dumps(manifest, indent=1))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--artifact_root', required=True)
    p.add_argument('--countries', nargs='+', default=list(COUNTRIES), choices=COUNTRIES)
    a = p.parse_args()
    for c in a.countries:
        print(json.dumps(build_country(a.artifact_root, c)), flush=True)
    print('COMPLETE')


if __name__ == '__main__':
    main()
