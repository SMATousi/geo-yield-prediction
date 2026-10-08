"""Field-context arrays for the point model (spec/yieldsat-field-context.md §4, FC-02).

Per country, next to the neighbourhood stream (``<artifact_root>/<NBR_NAME>/<Country>/``):

  field_s2.npz   field_codes (F,), mean (F, T, 12), std (F, T, 12), count (F, T, 12): per field season
                 and slot, the masked mean / std of each raw S2 band over the field's cells (NaN where
                 fewer than MIN_CELLS cells are valid)
  field_rel.npy  (rows, 5) float32, aligned with the cache rows: within-field z-scores of elevation,
                 slope, curvature and TWI, and log(1 + chessboard distance to the field edge in cells) / log(51)

Inputs only (no labels, no split), like the cache and the neighbourhood stream.

    YIELDSAT_NUM_SLOTS=72 YIELDSAT_CACHE_NAME=cache_dense YIELDSAT_NBR_NAME=neighbourhood_dense \\
        python yieldsat_build_field_context.py --artifact_root /data/YieldSAT/yieldsat_artifacts
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from dataset.yieldsat_cache import static_valid_mask
from dataset.yieldsat_schema import CACHE_NAME, COUNTRIES, NBR_NAME, STATIC_CHANNELS, STREAMS, TEMPORAL_CHANNELS

S2_POS = [TEMPORAL_CHANNELS.index(c) for c in STREAMS['yieldsat_s2']['channels']]
REL_CHANNELS = ('dem', 'slope', 'curvature', 'twi')
REL_POS = [STATIC_CHANNELS.index(c) for c in REL_CHANNELS]
REL_OUT = tuple('rel_' + c for c in REL_CHANNELS) + ('edge_dist',)
MIN_CELLS = 3
EDGE_CAP = 50
CHUNK = 8192                                                 # cells per read (2026-10-08: whole-field
                                                             # float64 reads OOM-killed a 1.2 Gi pod on a 67k-cell field)


def s2_sums(values):
    """values (n, T, C) raw S2 -> count, sum, sum of squares over cells (T, C), float64."""
    ok = np.isfinite(values)
    v = np.where(ok, values, 0.0).astype(np.float64)
    return ok.sum(0), v.sum(0), (v * v).sum(0)


def s2_finish(cnt, s, ss):
    """-> mean, std (NaN below MIN_CELLS), count."""
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = s / cnt
        var = np.maximum(ss / cnt - mean * mean, 0.0)
    keep = cnt >= MIN_CELLS
    return np.where(keep, mean, np.nan), np.where(keep, np.sqrt(var), np.nan), cnt


def field_s2_stats(values):
    """values (n, T, C) raw S2 of one field season -> mean, std, count (T, C); NaN below MIN_CELLS."""
    return s2_finish(*s2_sums(values))


def edge_distance(rows, cols, cap=EDGE_CAP):
    """Chessboard distance (cells) from each cell to the nearest grid position outside the field, capped:
    repeated 8-neighbour erosion of the field mask (numpy only)."""
    r0, c0 = rows.min(), cols.min()
    mask = np.zeros((rows.max() - r0 + 3, cols.max() - c0 + 3), bool)       # 1-cell outside border
    mask[rows - r0 + 1, cols - c0 + 1] = True
    dist = np.zeros(mask.shape, np.float64)
    cur = mask
    for _ in range(cap):
        p = np.pad(cur, 1)
        er = cur.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                er &= p[1 + dr:p.shape[0] - 1 + dr, 1 + dc:p.shape[1] - 1 + dc]
        if not er.any():
            break
        dist += er
        cur = er
    return dist[rows - r0 + 1, cols - c0 + 1]                                 # border cells: 0


def field_rel(static, rows, cols):
    """static (n, N_S) raw statics of one field season -> (n, 5) relative features (NaN invalid)."""
    out = np.full((len(rows), len(REL_OUT)), np.nan, np.float64)
    valid = static_valid_mask(static)
    for j, p in enumerate(REL_POS):
        ok = valid[:, p] & np.isfinite(static[:, p])
        if ok.sum() >= MIN_CELLS:
            x = static[ok, p].astype(np.float64)
            sd = x.std()
            if sd > 0:
                out[ok, j] = np.clip((x - x.mean()) / sd, -5, 5)
    out[:, -1] = np.log1p(np.minimum(edge_distance(rows, cols), EDGE_CAP)) / np.log1p(EDGE_CAP)
    return out


def build_country(artifact_root, country):
    root = Path(artifact_root)
    from yieldsat_build_dense import _read_rows, _write_rows
    src_t = root / CACHE_NAME / country / 'temporal.npy'
    src_s = root / CACHE_NAME / country / 'static.npy'
    shape = np.load(src_t, mmap_mode='r').shape
    with np.load(root / 'index' / country / 'rows.npz') as z:
        grid_row, grid_col = z['grid_row'], z['grid_col']
    fields = json.loads((root / 'index' / country / 'fields.json').read_text())
    out_dir = root / NBR_NAME / country
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / 'field_rel.npy.partial'
    m = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float32, shape=(shape[0], len(REL_OUT)))
    m[:] = np.nan
    m.flush()
    del m
    F, T, C = len(fields), shape[1], len(S2_POS)
    means, stds = np.full((F, T, C), np.nan, np.float32), np.full((F, T, C), np.nan, np.float32)
    counts = np.zeros((F, T, C), np.int32)
    t0 = time.time()
    for i, f in enumerate(fields):
        a, b = f['row_start'], f['row_end']
        acc = [np.zeros((T, C)), np.zeros((T, C)), np.zeros((T, C))]
        for k in range(a, b, CHUNK):                 # row chunks: peak memory independent of field size
            for j, x in enumerate(s2_sums(_read_rows(src_t, k, min(b, k + CHUNK))[..., S2_POS])):
                acc[j] += x
        means[i], stds[i], counts[i] = s2_finish(*acc)
        _write_rows(tmp, a, field_rel(_read_rows(src_s, a, b), grid_row[a:b], grid_col[a:b]).astype(np.float32))
        if (i + 1) % 200 == 0:
            print(country, i + 1, 'of', F, 'fields, %.0f s' % (time.time() - t0), flush=True)
    np.savez(out_dir / 'field_s2.npz.partial.npz', field_codes=np.array([f['field_code'] for f in fields], np.int64),
             mean=means, std=stds, count=counts)
    os.replace(out_dir / 'field_s2.npz.partial.npz', out_dir / 'field_s2.npz')
    os.replace(tmp, out_dir / 'field_rel.npy')
    manifest = {'country': country, 'rows': int(shape[0]), 'fields': F, 'slots': T,
                's2_channels': list(STREAMS['yieldsat_s2']['channels']), 'rel_channels': list(REL_OUT),
                'min_cells': MIN_CELLS, 'edge_cap': EDGE_CAP, 'seconds': round(time.time() - t0, 1)}
    (out_dir / 'field_context_manifest.json').write_text(json.dumps(manifest, indent=1))
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
