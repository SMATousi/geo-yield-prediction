"""Pilot sampling frame (NP-04b): CDL cell statistics, cluster selection and
point sampling, per harvest year.

Cluster cells are a fixed 10,230 m grid (341 x 341 CDL 30 m pixels) in the
CDL's EPSG:5070 frame. Clusters are the primary sampling units (D15),
allocated per NOAA region x crop stratum (D17); every CDL cropland class is
eligible for points (D18). Within a cluster, one 10 m cell centre is drawn per
100 m block and points are stratified by CDL class.
"""
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from yieldsat_us.regions import (CROPLAND_CODES, GROUPS, PILOT_ALLOCATION, REGIONS, STATE_REGION,
                                 group_lut, largest_remainder, split_years, stratum_area)

CELL_PX = 341                     # 341 x 30 m = 10,230 m cluster cell
CELL_M = CELL_PX * 30
BLOCK_M = 100                     # thinning block
POINTS_PER_CLUSTER = 2000
MIN_CROPLAND_FRAC = 0.30
MIN_STRATUM_SHARE = (0.25, 0.10, 0.02)   # relaxed in order if a stratum has no candidates
CDL_URL = 'https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/{y}_30m_cdls.zip'


def seed_of(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:12], 16)


def cdl_tif_path(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        tifs = [n for n in z.namelist() if n.lower().endswith('.tif')]
    if len(tifs) != 1:
        raise ValueError('expected one tif in {}, got {}'.format(zip_path, tifs))
    return tifs[0]


def extract_cdl(zip_path, out_dir):
    name = cdl_tif_path(zip_path)
    out = Path(out_dir) / Path(name).name
    if not out.exists():
        with zipfile.ZipFile(zip_path) as z, open(str(out) + '.partial', 'wb') as f:
            with z.open(name) as src:
                while True:
                    b = src.read(1 << 24)
                    if not b:
                        break
                    f.write(b)
        Path(str(out) + '.partial').rename(out)
    return out


def state_raster(states_shp_zip, transform, shape, crs='EPSG:5070'):
    """State postal code per cluster cell (by cell centre), via rasterization of the
    Census cartographic boundaries."""
    import shapefile
    from pyproj import Transformer
    from rasterio.features import rasterize
    with zipfile.ZipFile(states_shp_zip) as z:
        base = [n[:-4] for n in z.namelist() if n.endswith('.shp')][0]
        rd = shapefile.Reader(shp=io.BytesIO(z.read(base + '.shp')), dbf=io.BytesIO(z.read(base + '.dbf')),
                              shx=io.BytesIO(z.read(base + '.shx')))
    tr = Transformer.from_crs('EPSG:4269', crs, always_xy=True)
    fields = [f[0] for f in rd.fields[1:]]
    codes, shapes = [], []
    for sr in rd.shapeRecords():
        rec = dict(zip(fields, sr.record))
        st = rec['STUSPS']
        if st not in STATE_REGION:
            continue
        geo = sr.shape.__geo_interface__

        def proj(ring):
            xs, ys = tr.transform([p[0] for p in ring], [p[1] for p in ring])
            return list(zip(xs, ys))
        if geo['type'] == 'Polygon':
            g = {'type': 'Polygon', 'coordinates': [proj(r) for r in geo['coordinates']]}
        else:
            g = {'type': 'MultiPolygon', 'coordinates': [[proj(r) for r in poly] for poly in geo['coordinates']]}
        codes.append(st)
        shapes.append((g, len(codes)))
    ras = rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype='int16')
    return ras, ['--'] + codes


def cell_stats(cdl_tif, log_every=20):
    """Per cluster cell: pixel counts per crop group (+ non-cropland) and the cell grid."""
    import rasterio
    from rasterio.transform import Affine
    lut = group_lut()
    with rasterio.open(cdl_tif) as src:
        H, W = src.height, src.width
        nr, nc = H // CELL_PX, W // CELL_PX
        counts = np.zeros((nr, nc, len(GROUPS) + 1), dtype=np.int32)
        for r in range(nr):
            band = src.read(1, window=((r * CELL_PX, (r + 1) * CELL_PX), (0, nc * CELL_PX)))
            g = lut[band].reshape(CELL_PX, nc, CELL_PX)
            for k in range(len(GROUPS) + 1):
                counts[r, :, k] = (g == k).sum(axis=(0, 2))
            if log_every and r % log_every == 0:
                print('  cell rows {}/{}'.format(r, nr), flush=True)
        t = src.transform
        cell_transform = Affine(t.a * CELL_PX, t.b, t.c, t.d, t.e * CELL_PX, t.f)
        crs = src.crs.to_string()
    return counts, cell_transform, crs


def select_clusters(year, counts, region_of_cell, taken, region_offsets):
    """Clusters of one harvest year: per region, its year quota split across strata
    proportional to stratum cropland area; cells drawn with probability
    proportional to stratum area among qualifying cells (cropland >= 30%, stratum
    share >= 25%, relaxed if needed). ``taken`` = cells used by earlier years."""
    cells = counts.shape[0] * counts.shape[1]
    flat = counts.reshape(cells, -1)
    total = flat.sum(1)
    crop = flat[:, :len(GROUPS)].sum(1)
    frac = np.where(total > 0, crop / np.maximum(total, 1), 0)
    reg = region_of_cell.reshape(cells)
    out = []
    for region in REGIONS:
        n_reg, strata = PILOT_ALLOCATION[region]
        n_year = split_years(n_reg, offset=region_offsets[region])[year]
        in_reg = reg == REGIONS.index(region)
        weights = {s: float(stratum_area(flat[in_reg], s).sum()) for s in strata}
        alloc = largest_remainder(n_year, weights)
        for stratum, k in alloc.items():
            if k == 0:
                continue
            area = stratum_area(flat, stratum).astype(float)
            share = np.where(crop > 0, area / np.maximum(crop, 1), 0)
            rng = np.random.default_rng(seed_of('cluster', year, region, stratum))
            for min_share in MIN_STRATUM_SHARE:
                cand = np.flatnonzero(in_reg & (frac >= MIN_CROPLAND_FRAC) & (share >= min_share)
                                      & ~np.isin(np.arange(cells), list(taken)))
                if len(cand) >= k:
                    break
            if len(cand) == 0:
                continue
            p = area[cand] / area[cand].sum()
            pick = rng.choice(cand, size=min(k, len(cand)), replace=False, p=p)
            for c in pick:
                taken.add(int(c))
                out.append({'year': year, 'region': region, 'stratum': stratum, 'cell': int(c),
                            'cell_row': int(c // counts.shape[1]), 'cell_col': int(c % counts.shape[1]),
                            'cropland_frac': float(frac[c]), 'stratum_share': float(share[c]),
                            'min_share_used': min_share, 'selection_weight': float(p[cand.tolist().index(c)]),
                            'group_counts': {g: int(flat[c, i]) for i, g in enumerate(GROUPS)}})
    return out


def sample_points(cdl_tif, cluster, cell_transform, n=POINTS_PER_CLUSTER):
    """~n cropland points in the cluster cell: one random 10 m cell centre per
    100 m block, then stratified by CDL class (largest remainder)."""
    import rasterio
    r0, c0 = cluster['cell_row'] * CELL_PX, cluster['cell_col'] * CELL_PX
    with rasterio.open(cdl_tif) as src:
        win = src.read(1, window=((r0, r0 + CELL_PX), (c0, c0 + CELL_PX)))
        t = src.transform
    x0, y0 = t.c + c0 * t.a, t.f + r0 * t.e            # top-left of the cell (y decreasing)
    nb = CELL_M // BLOCK_M
    rng = np.random.default_rng(seed_of('points', cluster['year'], cluster['cell']))
    bi, bj = np.meshgrid(np.arange(nb), np.arange(nb), indexing='ij')
    sub = rng.integers(0, BLOCK_M // 10, size=(2, nb, nb))
    x = x0 + bj * BLOCK_M + sub[1] * 10 + 5.0
    y = y0 - (bi * BLOCK_M + sub[0] * 10 + 5.0)
    pc = ((x - x0) // 30).astype(int)
    pr = ((y0 - y) // 30).astype(int)
    code = win[pr, pc]
    ok = np.isin(code, CROPLAND_CODES)
    x, y, code = x[ok], y[ok], code[ok]
    classes, cnt = np.unique(code, return_counts=True)
    alloc = largest_remainder(min(n, len(code)), dict(zip(classes.tolist(), cnt.tolist())), floor=1)
    keep = []
    for cls, k in alloc.items():
        idx = np.flatnonzero(code == cls)
        keep += rng.choice(idx, size=min(k, len(idx)), replace=False).tolist()
    keep = np.sort(np.array(keep, dtype=int))
    return x[keep], y[keep], code[keep]


def build_year(year, cdl_tif, states_zip, out_dir, taken, region_offsets):
    from pyproj import Transformer
    out_dir = Path(out_dir)
    counts, cell_transform, crs = cell_stats(cdl_tif)
    st, codes = state_raster(states_zip, cell_transform, counts.shape[:2], crs)
    # region by cell centre (state raster is sampled at centres by rasterize's default)
    region_of_cell = np.full(st.shape, -1, dtype=np.int16)
    for i, s in enumerate(codes[1:], start=1):
        region_of_cell[st == i] = REGIONS.index(STATE_REGION[s])
    clusters = select_clusters(year, counts, region_of_cell, taken, region_offsets)
    tr = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    pts = {k: [] for k in ('cluster', 'x', 'y', 'lon', 'lat', 'cdl')}
    for ci, cl in enumerate(clusters):
        cl['cluster_id'] = '{}_{}_{:05d}'.format(year, cl['region'], cl['cell'])
        cl['state'] = codes[st[cl['cell_row'], cl['cell_col']]]
        x, y, c = sample_points(cdl_tif, cl, cell_transform)
        lon, lat = tr.transform(x, y)
        cl['n_points'] = int(len(x))
        x0 = cell_transform.c + cl['cell_col'] * CELL_M
        y1 = cell_transform.f - cl['cell_row'] * CELL_M
        cl['bbox_5070'] = [x0, y1 - CELL_M, x0 + CELL_M, y1]
        for k, v in (('cluster', np.full(len(x), ci)), ('x', x), ('y', y), ('lon', lon), ('lat', lat), ('cdl', c)):
            pts[k].append(np.asarray(v))
    np.savez(out_dir / 'frame_{}_points.npz'.format(year),
             **{k: np.concatenate(v) for k, v in pts.items()})
    (out_dir / 'frame_{}_clusters.json'.format(year)).write_text(json.dumps(
        {'year': year, 'crs': crs, 'cell_m': CELL_M, 'cell_transform': list(cell_transform)[:6],
         'clusters': clusters}, indent=1))
    np.savez_compressed(out_dir / 'frame_{}_cellstats.npz'.format(year), counts=counts,
                        region=region_of_cell, groups=np.array(GROUPS + ('non_cropland',)))
    return clusters
