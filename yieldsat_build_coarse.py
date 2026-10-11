"""IC-C (spec/yieldsat-input-information.md §4): dense cache variants with spatially coarsened Sentinel-2.

For factor k, every cell's S2 values (12 bands x 72 slots) are replaced by the mean of its aligned k x k block of the
same field season (blocks (grid_row // k, grid_col // k); per slot and band over the block's valid cells; a block with
no valid value stays missing). Weather, terrain and soil keep their own resolution; targets stay at 10 m.
Writes <artifact_root>/cache_dense_c<k>/<Country>/: temporal.npy (new) and links to the unchanged dense-cache files
(times, static, flags) plus copies of the manifest and field statistics (the normaliser keeps the 10 m statistics).
The neighbourhood and field-context arrays are then rebuilt from it with the existing builders:

    YIELDSAT_NUM_SLOTS=72 python yieldsat_build_coarse.py --artifact_root A --factor 2
    YIELDSAT_NUM_SLOTS=72 YIELDSAT_CACHE_NAME=cache_dense_c2 YIELDSAT_NBR_NAME=neighbourhood_dense_c2 \\
        python yieldsat_build_neighbourhood.py --artifact_root A    (then yieldsat_build_field_context.py)
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from dataset.yieldsat_schema import COUNTRIES, OPTICAL, TEMPORAL_CHANNELS

S2_POS = np.array([TEMPORAL_CHANNELS.index(c) for c in OPTICAL])


def coarsen_field(block, grid_row, grid_col, k):
    """block (n, T, C) float16 dense rows of one field season -> same with S2 replaced by k x k block means."""
    n, T, _ = block.shape
    key = (grid_row // k).astype(np.int64) * 1000003 + (grid_col // k)
    _, inv = np.unique(key, return_inverse=True)
    nb = inv.max() + 1
    M = csr_matrix((np.ones(n, np.float32), (inv, np.arange(n))), shape=(nb, n))
    v = block[:, :, S2_POS].astype(np.float32).reshape(n, -1)
    ok = np.isfinite(v)
    s = M @ np.where(ok, v, 0.0)
    c = M @ ok.astype(np.float32)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(c > 0, s / c, np.nan)
    out = block.copy()
    out[:, :, S2_POS] = mean[inv].reshape(n, T, len(S2_POS)).astype(block.dtype)
    return out


def build_country(root, country, k):
    from yieldsat_build_dense import _read_rows, _write_rows
    src_dir, dst_dir = root / 'cache_dense' / country, root / 'cache_dense_c{}'.format(k) / country
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ('times.npy', 'static.npy', 'flags.npy'):                 # unchanged: link to the dense cache
        link = dst_dir / name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(os.path.relpath(os.path.realpath(src_dir / name), dst_dir), link)
    shutil.copy2(src_dir / 'field_stats.npz', dst_dir / 'field_stats.npz')
    src = src_dir / 'temporal.npy'
    shape, dtype = np.load(src, mmap_mode='r').shape, np.load(src, mmap_mode='r').dtype
    tmp = dst_dir / 'temporal.npy.partial'
    m = np.lib.format.open_memmap(tmp, mode='w+', dtype=dtype, shape=shape)
    del m
    with np.load(root / 'index' / country / 'rows.npz') as z:
        gr, gc = z['grid_row'], z['grid_col']
    fields = json.loads((root / 'index' / country / 'fields.json').read_text())
    t0, done = time.time(), 0
    for i, f in enumerate(fields):
        a, b = f['row_start'], f['row_end']
        if a > done:                                                       # rows outside any field: copy as is
            _write_rows(tmp, done, _read_rows(src, done, a))
        _write_rows(tmp, a, coarsen_field(_read_rows(src, a, b), gr[a:b], gc[a:b], k))
        done = b
        if (i + 1) % 200 == 0:
            print(country, 'k', k, i + 1, 'of', len(fields), '%.0f s' % (time.time() - t0), flush=True)
    if done < shape[0]:
        _write_rows(tmp, done, _read_rows(src, done, shape[0]))
    os.replace(tmp, dst_dir / 'temporal.npy')
    man = json.loads((src_dir / 'cache_manifest.json').read_text())
    man['coarsened'] = {'factor': k, 'resolution_m': 10 * k, 'channels': list(OPTICAL),
                        'note': 'S2 = mean of the aligned k x k block within the field season; field_stats from 10 m'}
    (dst_dir / 'cache_manifest.json').write_text(json.dumps(man, indent=1))
    return {'country': country, 'factor': k, 'seconds': round(time.time() - t0, 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--artifact_root', required=True)
    p.add_argument('--factor', type=int, required=True)
    p.add_argument('--countries', nargs='+', default=list(COUNTRIES), choices=COUNTRIES)
    a = p.parse_args()
    for c in a.countries:
        print(json.dumps(build_country(Path(a.artifact_root), c, a.factor)), flush=True)
    print('COARSE_COMPLETE')


if __name__ == '__main__':
    main()
