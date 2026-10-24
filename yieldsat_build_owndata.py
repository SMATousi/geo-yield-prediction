"""Our own yield dataset (Missouri) in the YieldSAT artifact contract (spec/own-dataset-training.md).

One "country" ``Missouri``: every clean field-season (portal ground truth, QC passed, corn or soybean)
of the geo-data-downloader workspace. Rows are the 10 m cells of each field's Sentinel-2 grid with a
yield value (``layers/<id>/yield/<id>_yield_s2_10m.tif``, band 1). Written, as for the dense YieldSAT
cache (yieldsat_build_dense.py):

  cache_dense/Missouri/   temporal.npy (N, 72, 16) f16, static.npy (N, 104) f32, times.npy (N, 72) f32,
                          flags.npy, field_stats.npz, cache_manifest.json
  index/Missouri/         rows.npz, fields.json, index_manifest.json, target_side.json
  geometry/fields_geometry_Missouri.json   physical field (shared ground across years), centroid, county
  neighbourhood_dense/Missouri/            ps10.npy (N, 72, 8) f16, ps3x3.npy (N, 72, 72) f16,
                          ps_times.npy (N, 72) f32, ps_field_sums.npz (PlanetScope streams, D6),
                          dem10x10.npy (N, 100) f16 + dem_patch_sums.npz: the 1 m DEM on each cell's
                          10x10 sub-grid minus the cell mean, m (``--dem_patch_only`` adds it to a built root),
                          ps_win/<field_code>.npy (scenes, 3(H+2), 3(W+2), 8) f16 + ps_win_scene.npy (N, 72) i16:
                          the PlanetScope scene chosen per cell and slot on the field's 3.33 m grid padded by one
                          cell, from which the dataset cuts each cell's 30 m (9x9) window (``--ps_window_only``)

Slots, S2 and weather follow the dense YieldSAT rules (10-day bins from January of harvest year - 1, masked
outside [seeding, harvest]; per cell the clear S2 acquisition nearest the bin centre; weather summed over
the bin's in-season days, K·days and m). B01 and B09 were not downloaded: NaN.

    python yieldsat_build_owndata.py --workspace /home1/pupil/SMATousi/gdd-workspace \\
        --artifact_root /root/owndata_artifacts --workers 12
"""

import argparse
import calendar
import csv
import datetime
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject, transform as warp_transform

from dataset.yieldsat_schema import (CONTRACT_KEY, SOIL_DEPTHS, SOIL_PROPERTIES, STATIC_CHANNELS,
                                     TEMPORAL_CHANNELS)

COUNTRY = 'Missouri'
N_SLOTS = 72
CLEAR_SCL = (4, 5, 6)
BOA_OFFSET = 1000.0
BOA_FROM = '20220125'
EPOCH = datetime.date(1970, 1, 1)
CROP_OF = {'SoyBean': 'soybean', 'Corn': 'corn'}
S2_10M = ('B02', 'B03', 'B04', 'B08')
S2_20M = ('B05', 'B06', 'B07', 'B8A', 'B11', 'B12')
OPTICAL = TEMPORAL_CHANNELS[:12]
# ERA5-Land daily (Open-Meteo, degC and mm) -> YieldSAT weather channels (K and m)
WEATHER_COLS = {'temp_max': ('temperature_2m_max', 273.15, 1.0), 'temp_mean': ('temperature_2m_mean', 273.15, 1.0),
                'temp_min': ('temperature_2m_min', 273.15, 1.0), 'total_prec': ('precipitation_sum', 0.0, 1e-3)}
# soil: POLARIS 30 m where it has the property (sampled per cell), SoilGrids 250 m at the field centroid otherwise
POLARIS = {'clay': 'clay', 'sand': 'sand', 'silt': 'silt', 'phh2o': 'ph', 'soc': 'om'}
SOILGRIDS = {'cec': 'cec', 'cfvo': 'cfvo', 'nitrogen': 'nitrogen'}
PS_BANDS = ('coastal_blue', 'blue', 'green_i', 'green', 'yellow', 'red', 'rededge', 'nir')
PS4_POS = (1, 3, 5, 7)                  # 4-band SR (blue, green, red, nir) in the 8-band order
SUB = 3                                 # PlanetScope sub-cells per 10 m cell side (3.33 m)
PS_CELL_CLEAR = 0.9                     # a cell's PlanetScope observation: >= 90% of its area clear
PS_SUB_CLEAR = 0.5                      # a sub-cell's: >= 50% clear
DEM_SUB = 10                            # 1 m DEM sub-cells per 10 m cell side


def day(d):
    return (d - EPOCH).days


def slot_bounds(harvest_year):
    """(start_day, end_day) inclusive, days since 1970, for the 72 slots (as yieldsat_build_dense.py)."""
    out = []
    for k in range(N_SLOTS):
        y = harvest_year - 1 + k // 36
        m = (k % 36) // 3 + 1
        third = k % 3
        last = calendar.monthrange(y, m)[1]
        a, b = (1, 10) if third == 0 else (11, 20) if third == 1 else (21, last)
        out.append((day(datetime.date(y, m, a)), day(datetime.date(y, m, b))))
    return out


def _date_day(s):
    return day(datetime.date.fromisoformat(s[:10]))


def _planting_day(s, year):
    """Catalog planting date -> first planting day, or None. Besides ISO dates the catalog has
    '4/6,7,9,10,11/2023', '5/21/2024, 5/22/2024' and 'Early May' (-> the 5th)."""
    import re
    s = (s or '').strip()
    if not s:
        return None
    if s[:2] in ('19', '20'):
        return _date_day(s)
    m = re.match(r'(\d{1,2})/(\d{1,2})', s)
    if m:
        return day(datetime.date(year, int(m.group(1)), int(m.group(2))))
    m = re.match(r'(early|mid|late)\s+([a-z]+)', s.lower())
    if m:
        month = datetime.datetime.strptime(m.group(2)[:3], '%b').month
        return day(datetime.date(year, month, {'early': 5, 'mid': 15, 'late': 25}[m.group(1)]))
    return None


# ---- field table -------------------------------------------------------------------------------

def field_table(ws):
    """Clean corn/soybean field-seasons with planting and harvest days (imputed planting: crop x year
    median of the reported dates, else crop median day of year)."""
    gt = Path(ws) / 'ground_truth'
    rows = [r for r in csv.DictReader(open(gt / '_summary.csv')) if not r['qc_flags'] and r['crop'] in CROP_OF]
    out = []
    for r in sorted(rows, key=lambda r: r['vgir_id']):
        meta = json.loads((gt / r['vgir_id'] / 'metadata.json').read_text())
        out.append({'vgir_id': r['vgir_id'], 'crop': CROP_OF[r['crop']], 'year': int(r['year']),
                    'county': meta.get('County') or 'unknown', 'harvest_day': _date_day(r['harvest_start']),
                    'planting_day': _planting_day(meta.get('PlantingDate'), int(r['year']))})
    doy = {}
    for f in out:
        if f['planting_day'] is not None:
            d = EPOCH + datetime.timedelta(days=f['planting_day'])
            doy.setdefault((f['crop'], f['year']), []).append(f['planting_day'])
            doy.setdefault(f['crop'], []).append(d.timetuple().tm_yday)
    for f in out:
        if f['planting_day'] is not None:
            f['seeding_date_type'] = 0
            continue
        same = doy.get((f['crop'], f['year']))
        if same:
            f['planting_day'] = int(np.median(same))
            f['seeding_date_type'] = 1
        else:
            f['planting_day'] = day(datetime.date(f['year'], 1, 1)) + int(np.median(doy[f['crop']])) - 1
            f['seeding_date_type'] = 2
    return out


# ---- per-field readers -------------------------------------------------------------------------

def _warp(src_path, bands, grid, resampling, src_nodata=None, scale=1):
    """Read ``bands`` of a raster onto the yield grid (or its ``scale``-times finer sub-grid)."""
    tr, shape, crs = grid
    if scale > 1:
        tr = tr * rasterio.Affine.scale(1.0 / scale)
        shape = (shape[0] * scale, shape[1] * scale)
    with rasterio.open(src_path) as src:
        out = np.full((len(bands),) + shape, np.nan, dtype=np.float32)
        for i, b in enumerate(bands):
            reproject(rasterio.band(src, b), out[i], src_transform=src.transform, src_crs=src.crs,
                      src_nodata=src.nodata if src_nodata is None else src_nodata, dst_transform=tr, dst_crs=crs,
                      dst_nodata=np.nan, resampling=resampling)
    return out


def _s2_date(ld, rec, grid, rows, cols):
    """(values (n, 12) with NaN for B01/B09, clear (n,)) of one S2 acquisition at the cells."""
    tr, shape, crs = grid
    n = len(rows)
    v = np.full((n, 12), np.nan, dtype=np.float32)
    with rasterio.open(ld / rec['file_10m']) as r10:
        aligned = r10.crs == crs and r10.transform.almost_equals(tr) and r10.shape == shape
    if aligned:
        with rasterio.open(ld / rec['file_10m']) as r:
            a = r.read().astype(np.float32)
        b10 = rec['bands_10m'].split()
        for i, b in enumerate(b10):
            v[:, OPTICAL.index(b)] = a[i, rows, cols]
        with rasterio.open(ld / rec['file_20m']) as r:
            a = r.read().astype(np.float32)
        for i, b in enumerate(rec['bands_20m'].split()):
            v[:, OPTICAL.index(b)] = a[i, rows // 2, cols // 2]        # nearest upsampling, as YieldSAT
        with rasterio.open(ld / rec['file_scl']) as r:
            scl = r.read(1)[rows // 2, cols // 2]
    else:                                                              # another UTM grid (one date of one field)
        b10, b20 = rec['bands_10m'].split(), rec['bands_20m'].split()
        a = _warp(ld / rec['file_10m'], range(1, len(b10) + 1), grid, Resampling.nearest, 0)
        for i, b in enumerate(b10):
            v[:, OPTICAL.index(b)] = a[i, rows, cols]
        a = _warp(ld / rec['file_20m'], range(1, len(b20) + 1), grid, Resampling.nearest, 0)
        for i, b in enumerate(b20):
            v[:, OPTICAL.index(b)] = a[i, rows, cols]
        scl = _warp(ld / rec['file_scl'], [1], grid, Resampling.nearest, 0)[0, rows, cols]
    present = [OPTICAL.index(b) for b in S2_10M + S2_20M]
    clear = np.isin(scl, CLEAR_SCL) & (v[:, present] > 0).all(1) & np.isfinite(v[:, present]).all(1)
    if boa_offset(rec['item_id']):
        v = np.maximum(v - BOA_OFFSET, 0.0)
    return v, clear


def boa_offset(item_id):
    """Planetary Computer serves L2A unharmonized: products of processing baseline >= 04.00 (processed
    from 2022-01-25 on, the reprocessed archive included) carry +1000 DN. The product's processing time is
    the item id's last token (``..._20221012T045449``)."""
    return item_id.rsplit('_', 1)[-1][:8] >= BOA_FROM


def _weather(ld):
    with open(ld / 'weather_era5land' / 'era5land_daily.csv') as fh:
        rd = csv.DictReader(fh)
        out = {}
        for r in rd:
            vals = []
            for c, (col, add, mul) in WEATHER_COLS.items():
                s = r.get(col, '')
                vals.append((float(s) + add) * mul if s not in ('', None) else np.nan)
            out[_date_day(r['date'])] = vals
    return out


def _terrain(ld, grid):
    """dem, aspect, curvature, slope, twi on the 10 m grid: the 1 m DEM averaged to 10 m, derivatives
    of that 10 m DEM (Horn slope in degrees, aspect = downslope azimuth in degrees, Zevenbergen-Thorne
    profile curvature x 100), TWI = mean of the 5 m TWI."""
    tr, shape, crs = grid
    dem = _warp(ld / 'dem' / 'dem_1m.tif', [1], grid, Resampling.average)[0]
    g = 10.0
    z = np.pad(dem, 1, mode='edge')
    z1, z2, z3 = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    z4, z5, z6 = z[1:-1, :-2], z[1:-1, 1:-1], z[1:-1, 2:]
    z7, z8, z9 = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    dzdx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8 * g)          # east
    dzdy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8 * g)          # south (rows grow southwards)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = np.mod(np.degrees(np.arctan2(dzdx, -dzdy)) + 180.0, 360.0)   # downslope azimuth from north
    D = ((z4 + z6) / 2 - z5) / g ** 2
    E = ((z2 + z8) / 2 - z5) / g ** 2
    F = (-z1 + z3 + z7 - z9) / (4 * g ** 2)
    G = (-z4 + z6) / (2 * g)
    H = (z2 - z8) / (2 * g)
    with np.errstate(invalid='ignore', divide='ignore'):
        prof = np.where(G * G + H * H > 0, -2 * (D * G * G + E * H * H + F * G * H) / (G * G + H * H), 0.0)
    twi = _warp(ld / 'terrain' / 'twi_5m.tif', [1], grid, Resampling.average)[0]
    return np.stack([dem, aspect, prof * 100.0, slope, twi])


def _soilgrids_centroid(ld):
    p = ld / 'soil_soilgrids' / 'soilgrids_centroid.json'
    out = {}
    if not p.exists():
        return out
    for layer in json.loads(p.read_text())['properties']['layers']:
        for d in layer['depths']:
            key = (layer['name'], d['label'].replace('cm', ''))
            mean = d['values'].get('mean')
            out[key] = np.nan if mean is None else float(mean)
    return out


def _static(ld, grid, rows, cols):
    n = len(rows)
    st = np.full((n, len(STATIC_CHANNELS)), np.nan, dtype=np.float32)
    ter = _terrain(ld, grid)
    for i, c in enumerate(('dem', 'aspect', 'curvature', 'slope', 'twi')):
        st[:, STATIC_CHANNELS.index(c)] = ter[i, rows, cols]
    # cell centres -> lon/lat (POLARIS sampling, coordinates)
    tr, shape, crs = grid
    x = tr.c + (cols + 0.5) * tr.a
    y = tr.f + (rows + 0.5) * tr.e
    lon, lat = warp_transform(crs, 'EPSG:4326', x.tolist(), y.tolist())
    lon, lat = np.asarray(lon), np.asarray(lat)
    sg = _soilgrids_centroid(ld)
    for prop in SOIL_PROPERTIES:
        for d in SOIL_DEPTHS:
            j = STATIC_CHANNELS.index('{}_{}'.format(prop, d))
            if prop in POLARIS:
                f = ld / 'soil_polaris' / 'polaris_{}_{}.tif'.format(POLARIS[prop], d.replace('-', '_'))
                if f.exists():
                    with rasterio.open(f) as r:
                        a = r.read(1).astype(np.float32)
                        a[a == r.nodata] = np.nan
                        rr, cc = rasterio.transform.rowcol(r.transform, lon, lat)
                        rr, cc = np.clip(rr, 0, a.shape[0] - 1), np.clip(cc, 0, a.shape[1] - 1)
                        st[:, j] = a[rr, cc]
            else:
                st[:, j] = sg.get((SOILGRIDS[prop], d), np.nan)
    lam, phi = np.radians(lon), np.radians(lat)
    for c, v in zip(('coord_x', 'coord_y', 'coord_z'),
                    (np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi))):
        st[:, STATIC_CHANNELS.index(c)] = v
    return st, lon, lat


def _ps_scenes(ld):
    idx = ld / 'planetscope' / 'planetscope_index.csv'
    if not idx.exists():
        return []
    out = []
    for r in csv.DictReader(open(idx)):
        if r['delivered'] != 'True':
            continue
        stem = '{}_{}'.format(r['date'], r['item_id'])
        sr, udm = ld / 'planetscope' / (stem + '_sr.tif'), ld / 'planetscope' / (stem + '_udm2.tif')
        if sr.exists() and udm.exists():
            out.append((_date_day(r['acquired']), sr, udm, int(r['bands'] or 0)))
    return out


def _ps_read(sr, udm, nb, grid, rows, cols):
    """One scene at the cells: (cell values (n, 8), cell ok (n,), sub values (n, 9, 8))."""
    n = len(rows)
    cell = np.full((n, 8), np.nan, np.float32)
    sub = np.full((n, SUB * SUB, 8), np.nan, np.float32)
    pos = list(range(8)) if nb == 8 else list(PS4_POS)
    with rasterio.open(sr) as r:
        k = r.count
    pos = pos[:k]
    a10 = _warp(sr, range(1, k + 1), grid, Resampling.average, 0)
    c10 = _warp(udm, [1], grid, Resampling.average, 255)[0]          # clear fraction (UDM2 band 1)
    a3 = _warp(sr, range(1, k + 1), grid, Resampling.average, 0, scale=SUB)
    c3 = _warp(udm, [1], grid, Resampling.average, 255, scale=SUB)[0]
    ok = (np.nan_to_num(c10[rows, cols], nan=0.0) >= PS_CELL_CLEAR) & np.isfinite(a10[:, rows, cols]).all(0)
    cell[:, pos] = a10[:, rows, cols].T
    for i in range(SUB):
        for j in range(SUB):
            rr, cc = rows * SUB + i, cols * SUB + j
            v = a3[:, rr, cc].T
            sok = (np.nan_to_num(c3[rr, cc], nan=0.0) >= PS_SUB_CLEAR) & np.isfinite(v).all(1)
            sub[:, i * SUB + j, pos] = np.where(sok[:, None], v, np.nan)
    cell[~ok] = np.nan
    return cell, ok, sub


def build_field(task):
    ws, out, f, row_start, rows, cols, grid = task
    ld = Path(ws) / 'layers' / f['vgir_id']
    n = len(rows)
    seeding, harvest = f['planting_day'], f['harvest_day']
    bounds = slot_bounds(f['year'])
    temporal = np.full((n, N_SLOTS, 16), np.nan, np.float32)
    times = np.full((n, N_SLOTS), np.nan, np.float32)
    ps10 = np.full((n, N_SLOTS, 8), np.nan, np.float32)
    ps3 = np.full((n, N_SLOTS, SUB * SUB * 8), np.nan, np.float32)
    ps_times = np.full((n, N_SLOTS), np.nan, np.float32)
    s2 = {}
    for r in csv.DictReader(open(ld / 'sentinel2' / 'sentinel2_index.csv')):
        d = _date_day(r['date'])
        if seeding <= d <= harvest and r.get('file_10m') and r.get('file_20m') and r.get('file_scl'):
            s2.setdefault(d, r)
    weather = _weather(ld)
    ps = [s for s in _ps_scenes(ld) if seeding <= s[0] <= harvest]
    w_pos = [TEMPORAL_CHANNELS.index(c) for c in WEATHER_COLS]
    st = {'s2_dates': len(s2), 'ps_scenes': len(ps), 'season_slots': 0, 'clear_cells': 0, 'ps_clear_cells': 0,
          'unreadable': 0}
    for k, (a, b) in enumerate(bounds):
        lo, hi = max(a, seeding), min(b, harvest)
        if lo > hi:
            continue
        st['season_slots'] += 1
        days = [d for d in range(lo, hi + 1) if d in weather]
        if days:
            temporal[:, k, w_pos] = np.nansum(np.array([weather[d] for d in days]), axis=0)
        centre = (a + b) / 2.0
        todo = np.ones(n, bool)
        for d in sorted((d for d in s2 if a <= d <= b), key=lambda d: (abs(d - centre), d)):
            try:
                v, ok = _s2_date(ld / 'sentinel2', s2[d], grid, rows, cols)
            except Exception:
                st['unreadable'] += 1
                continue
            take = todo & ok
            if take.any():
                temporal[take, k, :12] = v[take]
                times[take, k] = d
                todo &= ~take
            if not todo.any():
                break
        st['clear_cells'] += int((~todo).sum())
        if days:
            times[todo, k] = min(max(round(centre), lo), hi)
        # PlanetScope: per cell the clear scene nearest the bin centre (same rule as S2)
        todo = np.ones(n, bool)
        for d, sr, udm, nb in sorted((s for s in ps if a <= s[0] <= b), key=lambda s: (abs(s[0] - centre), s[0])):
            try:
                cv, ok, sv = _ps_read(sr, udm, nb, grid, rows, cols)
            except Exception:
                st['unreadable'] += 1
                continue
            take = todo & ok
            if take.any():
                ps10[take, k] = cv[take]
                ps3[take, k] = sv[take].reshape(int(take.sum()), -1)
                ps_times[take, k] = d
                todo &= ~take
            if not todo.any():
                break
        st['ps_clear_cells'] += int((~todo).sum())
    s2v = temporal[:, :, :12]
    s2v[s2v > 65504] = np.nan
    static, lon, lat = _static(ld, grid, rows, cols)
    o = Path(out)
    for name, arr in (('cache_dense/{}/temporal.npy', temporal.astype(np.float16)),
                      ('cache_dense/{}/times.npy', times), ('cache_dense/{}/static.npy', static),
                      ('neighbourhood_dense/{}/ps10.npy', ps10.astype(np.float16)),
                      ('neighbourhood_dense/{}/ps3x3.npy', ps3.astype(np.float16)),
                      ('neighbourhood_dense/{}/ps_times.npy', ps_times)):
        mm = np.load(o / name.format(COUNTRY), mmap_mode='r+')
        mm[row_start:row_start + n] = arr
        mm.flush()
        del mm
    st['cells'] = n
    st['centroid'] = (float(lon.mean()), float(lat.mean()))
    return f['vgir_id'], st


# ---- driver ------------------------------------------------------------------------------------

def _grid_and_cells(ws, vgir):
    p = Path(ws) / 'layers' / vgir / 'yield' / '{}_yield_s2_10m.tif'.format(vgir)
    with rasterio.open(p) as r:
        y = r.read(1)
        grid = (r.transform, r.shape, r.crs)
    rows, cols = np.nonzero(np.isfinite(y))
    return grid, rows.astype(np.int32), cols.astype(np.int32), y[rows, cols].astype(np.float32)


def physical_fields(cells):
    """Field-seasons on shared ground (>= 30% of the smaller one's cells at the same 10 m position in
    UTM 15N) are one physical field: union-find over the pairs."""
    keys = list(cells)
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            inter = len(cells[a] & cells[b])
            if inter and inter >= 0.3 * min(len(cells[a]), len(cells[b])):
                parent[find(b)] = find(a)
    return {k: find(k) for k in keys}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workspace', default='/home1/pupil/SMATousi/gdd-workspace')
    p.add_argument('--artifact_root', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=0, help='first N field-seasons only (tests)')
    a = p.parse_args()
    t0 = time.time()
    fields = field_table(a.workspace)
    if a.limit:
        fields = fields[:a.limit]
    out = Path(a.artifact_root)
    for d in ('cache_dense', 'index', 'neighbourhood_dense'):
        (out / d / COUNTRY).mkdir(parents=True, exist_ok=True)
    (out / 'geometry').mkdir(exist_ok=True)
    tasks, recs, row = [], [], 0
    all_rows = {k: [] for k in ('field_code', 'grid_row', 'grid_col', 'target', 'seeding_day', 'harvest_day',
                                'seeding_date_type')}
    cellsets = {}
    for code, f in enumerate(fields):
        grid, rows, cols, y = _grid_and_cells(a.workspace, f['vgir_id'])
        n = len(rows)
        tr = grid[0]
        cellsets[f['vgir_id']] = set(zip(np.round((tr.c + (cols + 0.5) * tr.a) / 10).astype(int).tolist(),
                                         np.round((tr.f + (rows + 0.5) * tr.e) / 10).astype(int).tolist()))
        tasks.append((a.workspace, str(out), f, row, rows, cols, grid))
        recs.append({'country': COUNTRY, 'field_code': code, 'field_shared_name': f['vgir_id'], 'provider': 'TG',
                     'farm': f['county'], 'farm_id': '{}/{}'.format(COUNTRY, f['county']),
                     'season_id': '{}/{}'.format(COUNTRY, f['vgir_id']), 'crop': f['crop'], 'year': f['year'],
                     'row_start': row, 'row_end': row + n, 'n_rows': n,
                     'grid_rows': [int(rows.min()), int(rows.max())], 'grid_cols': [int(cols.min()), int(cols.max())],
                     'target_mean': float(y.mean()), 'target_std': float(y.std()), 'single_season_dates': True,
                     'seeding_day': int(f['planting_day']), 'harvest_day': int(f['harvest_day']),
                     'seeding_date_type': f['seeding_date_type']})
        for k, v in (('field_code', code), ('seeding_day', f['planting_day']), ('harvest_day', f['harvest_day']),
                     ('seeding_date_type', f['seeding_date_type'])):
            all_rows[k].append(np.full(n, v))
        all_rows['grid_row'].append(rows)
        all_rows['grid_col'].append(cols)
        all_rows['target'].append(y)
        row += n
    N = row
    shapes = {'cache_dense/{}/temporal.npy': ((N, N_SLOTS, 16), np.float16),
              'cache_dense/{}/times.npy': ((N, N_SLOTS), np.float32),
              'cache_dense/{}/static.npy': ((N, len(STATIC_CHANNELS)), np.float32),
              'neighbourhood_dense/{}/ps10.npy': ((N, N_SLOTS, 8), np.float16),
              'neighbourhood_dense/{}/ps3x3.npy': ((N, N_SLOTS, SUB * SUB * 8), np.float16),
              'neighbourhood_dense/{}/ps_times.npy': ((N, N_SLOTS), np.float32)}
    for name, (shape, dt) in shapes.items():
        mm = np.lib.format.open_memmap(out / name.format(COUNTRY), mode='w+', dtype=dt, shape=shape)
        mm[:] = np.nan
        mm.flush()
        del mm
    print('{} field-seasons, {} cells; building with {} workers'.format(len(fields), N, a.workers), flush=True)
    stats = {}
    with Pool(a.workers) as pool:
        for vgir, st in pool.imap_unordered(build_field, tasks):
            stats[vgir] = st
            print('  {} {} cells, {} S2 dates, {} PS scenes, clear S2 {:.1f} / PS {:.1f} slots per cell'.format(
                vgir, st['cells'], st['s2_dates'], st['ps_scenes'], st['clear_cells'] / st['cells'],
                st['ps_clear_cells'] / st['cells']), flush=True)
    write_index(out, recs, all_rows, stats, cellsets, a.workspace)
    write_cache_meta(out, recs, N)
    print('done in {:.0f} s'.format(time.time() - t0))


def write_index(out, recs, all_rows, stats, cellsets, ws):
    d = out / 'index' / COUNTRY
    rows = {k: np.concatenate(v).astype(np.float32 if k == 'target' else np.int16 if k == 'seeding_date_type'
                                        else np.int32) for k, v in all_rows.items()}
    np.savez(d / 'rows.npz', **rows)
    (d / 'fields.json').write_text(json.dumps(recs, indent=1))
    (d / 'target_side.json').write_text(json.dumps(
        {'note': 'field-level mean of the 10 m target; diagnostics only, never model input',
         'values': {r['field_shared_name']: r['target_mean'] for r in recs}}, indent=1))
    (d / 'index_manifest.json').write_text(json.dumps(
        {'contract': CONTRACT_KEY, 'index_version': 1, 'country': COUNTRY,
         'source': str(Path(ws) / 'layers'), 'fingerprint': {'own_dataset': True},
         'rows': int(len(rows['target'])), 'built': datetime.datetime.now().isoformat(timespec='seconds')}, indent=1))
    phys = physical_fields(cellsets)
    geo = [{'field_shared_name': r['field_shared_name'], 'physical_field_id': '{}/{}'.format(COUNTRY, phys[r['field_shared_name']]),
            'centroid_lon': stats[r['field_shared_name']]['centroid'][0],
            'centroid_lat': stats[r['field_shared_name']]['centroid'][1],
            'adm_units': {'adm_unit0': r['farm']}} for r in recs]
    (out / 'geometry' / 'fields_geometry_{}.json'.format(COUNTRY)).write_text(json.dumps(geo, indent=1))
    (out / 'cache_dense' / COUNTRY / 'build_stats.json').write_text(json.dumps(stats, indent=1))


def write_cache_meta(out, recs, N):
    """field_stats.npz (dense rule: first-slot weather valid), flags, cache manifest; PlanetScope per-field
    sums of ps10 per band (train-only normalization of both PlanetScope streams)."""
    from dataset.yieldsat_cache import (CACHE_VERSION, FIELD_STATS_VERSION, N_S, N_T, _Accumulator,
                                        static_valid_mask, temporal_valid_mask)
    c = out / 'cache_dense' / COUNTRY
    temporal = np.load(c / 'temporal.npy', mmap_mode='r')
    times = np.load(c / 'times.npy', mmap_mode='r')
    static = np.load(c / 'static.npy', mmap_mode='r')
    ps10 = np.load(out / 'neighbourhood_dense' / COUNTRY / 'ps10.npy', mmap_mode='r')
    acc_t, acc_s, acc_p = _Accumulator(len(recs), N_T), _Accumulator(len(recs), N_S), _Accumulator(len(recs), 8)
    for slot, r in enumerate(recs):
        s, e = r['row_start'], r['row_end']
        t = np.asarray(temporal[s:e]).astype(np.float32)
        valid = temporal_valid_mask(t, np.asarray(times[s:e]), mask_first_weather=False)
        acc_t.add(slot, np.where(valid, t, np.nan).reshape(-1, N_T))
        st = np.asarray(static[s:e])
        acc_s.add(slot, np.where(static_valid_mask(st), st, np.nan))
        acc_p.add(slot, np.asarray(ps10[s:e]).astype(np.float32).reshape(-1, 8))
    codes = np.array([r['field_code'] for r in recs])
    np.savez(c / 'field_stats.npz', field_codes=codes,
             temporal_count=acc_t.count, temporal_sum=acc_t.sum, temporal_sumsq=acc_t.sumsq,
             temporal_min=acc_t.min, temporal_max=acc_t.max,
             static_count=acc_s.count, static_sum=acc_s.sum, static_sumsq=acc_s.sumsq,
             static_min=acc_s.min, static_max=acc_s.max)
    np.savez(out / 'neighbourhood_dense' / COUNTRY / 'ps_field_sums.npz', field_codes=codes, bands=np.array(PS_BANDS),
             count=acc_p.count, sum=acc_p.sum, sumsq=acc_p.sumsq)
    np.save(c / 'flags.npy', np.zeros(N, np.uint8))
    (c / 'cache_manifest.json').write_text(json.dumps(
        {'contract': CONTRACT_KEY, 'cache_version': CACHE_VERSION, 'country': COUNTRY,
         'fingerprint': {'own_dataset': True}, 'index_fingerprint': {'own_dataset': True}, 'rows': int(N),
         'complete': True, 'cache_written': True, 'temporal_channels': list(TEMPORAL_CHANNELS),
         'static_channels': list(STATIC_CHANNELS), 'n_slots': N_SLOTS, 'field_stats_version': FIELD_STATS_VERSION,
         'slot_rule': '10-day bins from January of harvest year - 1; clear S2 (SCL 4/5/6) nearest the bin centre',
         'missing_channels': ['B01', 'B09', 'soil uncertainty'],
         'planetscope': {'bands': list(PS_BANDS), 'sub_cells': SUB * SUB, 'cell_clear': PS_CELL_CLEAR,
                         'sub_clear': PS_SUB_CLEAR, '4band_positions': list(PS4_POS)}}, indent=1))


def _dem_patch(task):
    ws, f, rows, cols, grid = task
    z = _warp(Path(ws) / 'layers' / f['vgir_id'] / 'dem' / 'dem_1m.tif', [1], grid, Resampling.average,
              scale=DEM_SUB)[0]
    H, W = grid[1]
    z = z.reshape(H, DEM_SUB, W, DEM_SUB).transpose(0, 2, 1, 3).reshape(H, W, DEM_SUB * DEM_SUB)[rows, cols]
    with np.errstate(invalid='ignore'):
        rel = z - np.nanmean(z, axis=1, keepdims=True)
    return f['vgir_id'], rel.astype(np.float32)


def _ps_window(task):
    """One field: the round-1 PlanetScope selection (per cell and slot the scene nearest the bin centre whose
    cell is >= 90% clear) and the selected scenes on the padded 3.33 m grid (sub-pixels < 50% clear are NaN)."""
    ws, f, rows, cols, grid = task
    ld = Path(ws) / 'layers' / f['vgir_id']
    tr, (H, W), crs = grid
    pgrid = (tr * rasterio.Affine.translation(-1, -1), (H + 2, W + 2), crs)
    n = len(rows)
    seeding, harvest = f['planting_day'], f['harvest_day']
    ps = [s for s in _ps_scenes(ld) if seeding <= s[0] <= harvest]
    scene = np.full((n, N_SLOTS), -1, np.int16)
    days = np.full((n, N_SLOTS), np.nan, np.float32)
    used, imgs, cache = {}, [], {}

    def read(i):
        if i not in cache:
            d, sr, udm, nb = ps[i]
            with rasterio.open(sr) as r:
                k = r.count
            pos = (list(range(8)) if nb == 8 else list(PS4_POS))[:k]
            a10 = _warp(sr, range(1, k + 1), pgrid, Resampling.average, 0)
            c10 = _warp(udm, [1], pgrid, Resampling.average, 255)[0]
            ok = (np.nan_to_num(c10[rows + 1, cols + 1], nan=0.0) >= PS_CELL_CLEAR) & \
                np.isfinite(a10[:, rows + 1, cols + 1]).all(0)
            cache[i] = (ok, pos)
        return cache[i]

    for k, (a, b) in enumerate(slot_bounds(f['year'])):
        if max(a, seeding) > min(b, harvest):
            continue
        centre = (a + b) / 2.0
        todo = np.ones(n, bool)
        for i in sorted((i for i, s in enumerate(ps) if a <= s[0] <= b), key=lambda i: (abs(ps[i][0] - centre), ps[i][0])):
            try:
                ok, _ = read(i)
            except Exception:
                continue
            take = todo & ok
            if take.any():
                if i not in used:
                    used[i] = len(used)
                scene[take, k] = used[i]
                days[take, k] = ps[i][0]
                todo &= ~take
            if not todo.any():
                break
    for i in sorted(used, key=used.get):
        d, sr, udm, nb = ps[i]
        _, pos = read(i)
        a3 = _warp(sr, range(1, len(pos) + 1), pgrid, Resampling.average, 0, scale=SUB)
        c3 = _warp(udm, [1], pgrid, Resampling.average, 255, scale=SUB)[0]
        img = np.full((8,) + a3.shape[1:], np.nan, np.float32)
        img[pos] = np.where((np.nan_to_num(c3, nan=0.0) >= PS_SUB_CLEAR) & np.isfinite(a3).all(0), a3, np.nan)
        imgs.append(img.transpose(1, 2, 0).astype(np.float16))
    stack = (np.stack(imgs) if imgs else np.full((0, 3 * (H + 2), 3 * (W + 2), 8), np.nan, np.float16))
    return f['vgir_id'], scene, days, stack


def write_ps_window(ws, artifact_root, workers):
    """ps_win/<field_code>.npy and ps_win_scene.npy (row order of the built index); the selection is checked
    against the round-1 ps_times.npy."""
    out = Path(artifact_root)
    d = out / 'neighbourhood_dense' / COUNTRY
    (d / 'ps_win').mkdir(exist_ok=True)
    recs, tasks = _field_tasks(ws, out)
    ps_times = np.load(d / 'ps_times.npy', mmap_mode='r')
    scene = np.full((sum(r['n_rows'] for r in recs), N_SLOTS), -1, np.int16)
    pos = {r['field_shared_name']: r for r in recs}
    with Pool(workers) as pool:
        for vgir, sc, days, stack in pool.imap_unordered(_ps_window, tasks):
            r = pos[vgir]
            ref = np.asarray(ps_times[r['row_start']:r['row_end']])
            assert np.array_equal(np.isfinite(ref), sc >= 0) and np.allclose(ref[sc >= 0], days[sc >= 0]), vgir
            scene[r['row_start']:r['row_end']] = sc
            np.save(d / 'ps_win' / '{}.npy'.format(r['field_code']), stack)
            print('  {} {} scenes {} MB'.format(vgir, len(stack), stack.nbytes >> 20), flush=True)
    np.save(d / 'ps_win_scene.npy', scene)


def _field_tasks(ws, out):
    idx = np.load(out / 'index' / COUNTRY / 'rows.npz')
    recs = json.loads((out / 'index' / COUNTRY / 'fields.json').read_text())
    by_id = {f['vgir_id']: f for f in field_table(ws)}
    tasks = []
    for r in recs:
        grid, rows, cols, _ = _grid_and_cells(ws, r['field_shared_name'])
        s, e = r['row_start'], r['row_end']
        assert (idx['grid_row'][s:e] == rows).all() and (idx['grid_col'][s:e] == cols).all(), r['field_shared_name']
        tasks.append((ws, by_id[r['field_shared_name']], rows, cols, grid))
    return recs, tasks


def write_dem_patch(ws, artifact_root, workers):
    """dem10x10.npy (N, 100) f16, row order of the built index: per cell the 1 m DEM on its 10x10 sub-grid
    (row-major, north first) minus the cell's mean elevation (the mean is the ``dem`` static channel), and
    dem_patch_sums.npz (per field season count / sum / sum of squares, train-only normalization)."""
    out = Path(artifact_root)
    recs, tasks = _field_tasks(ws, out)
    N = sum(r['n_rows'] for r in recs)
    arr = np.full((N, DEM_SUB * DEM_SUB), np.nan, np.float16)
    cnt, sm, ssq = np.zeros(len(recs)), np.zeros(len(recs)), np.zeros(len(recs))
    pos = {r['field_shared_name']: i for i, r in enumerate(recs)}
    with Pool(workers) as pool:
        for vgir, rel in pool.imap_unordered(_dem_patch, tasks):
            i = pos[vgir]
            arr[recs[i]['row_start']:recs[i]['row_end']] = rel
            ok = np.isfinite(rel)
            cnt[i], sm[i], ssq[i] = ok.sum(), rel[ok].sum(), (rel[ok] ** 2).sum()
            print('  {} {} cells, {:.1%} valid, within-cell std {:.3f} m'.format(
                vgir, len(rel), ok.mean(), np.sqrt(ssq[i] / max(cnt[i], 1))), flush=True)
    d = out / 'neighbourhood_dense' / COUNTRY
    np.save(d / 'dem10x10.npy', arr)
    np.savez(d / 'dem_patch_sums.npz', field_codes=np.array([r['field_code'] for r in recs]),
             count=cnt, sum=sm, sumsq=ssq)
    print('dem10x10: {} cells, {:.1%} valid'.format(N, np.isfinite(arr).mean()))


PAIRS = ('MO-S', 'MO-C')
PROTOCOLS = ('cv', 'loyo')


def write_splits(artifact_root):
    """The paper's protocol on our pairs (no validation set, selection on the test fold): CV10 grouped by
    field season and stratified by county, and leave-one-year-out; ``noval_paper_<cv10|loyo>_<pair>_...``."""
    from dataset.yieldsat_splits import (load_field_table, make_paper_folds, noval_split, paper_row_prefix,
                                         save_folds)
    table = load_field_table(artifact_root, None, [COUNTRY], check_source=False)
    for pair in PAIRS:
        for protocol in PROTOCOLS:
            prefix = 'noval_' + paper_row_prefix(pair, protocol)
            splits = [noval_split(s, paper_row_prefix(pair, protocol) + '_fold{:02d}'.format(s['fold']))
                      for s in make_paper_folds(table, pair, protocol, val_frac=0.0)]
            save_folds(splits, artifact_root, prefix)
            print(prefix, len(splits), [len(s['partitions']['test']) for s in splits])


if __name__ == '__main__':
    import sys
    if '--splits_only' in sys.argv:
        write_splits(sys.argv[sys.argv.index('--artifact_root') + 1])
    elif '--dem_patch_only' in sys.argv:
        arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
        write_dem_patch(arg('--workspace', '/home1/pupil/SMATousi/gdd-workspace'), arg('--artifact_root', None),
                        int(arg('--workers', 8)))
    elif '--ps_window_only' in sys.argv:
        arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
        write_ps_window(arg('--workspace', '/home1/pupil/SMATousi/gdd-workspace'), arg('--artifact_root', None),
                        int(arg('--workers', 8)))
    else:
        main()
