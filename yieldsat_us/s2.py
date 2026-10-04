"""Pixel-only Sentinel-2 L2A reader (NP-04c, decision D14).

Per cluster: STAC search (AWS Earth Search ``sentinel-2-l2a``, public COGs);
for every acquisition, read the SCL window covering the cluster (HTTP range
reads of the COG blocks only) and record the class at each point; per point and
calendar month of its season, select one acquisition (clear at the point, SCL
4/5/6, lowest scene cloud cover, then latest, then item ID; if none is clear,
the least-cloudy valid acquisition with optical values masked); read the 12
bands only for selected acquisitions, at the native pixel containing each point.
Harmonization: subtract 1,000 (clamped at 0) when processing baseline >= 04.00
and the source has not applied the BOA offset (Earth Search reports
``earthsearch:boa_offset_applied``). Nothing is written to disk but the values.
"""
import datetime
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STAC_URL = 'https://earth-search.aws.element84.com/v1'
COLLECTION = 'sentinel-2-l2a'
# YieldSAT band order -> Earth Search asset keys
BANDS = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12')
ASSET = {'B01': 'coastal', 'B02': 'blue', 'B03': 'green', 'B04': 'red', 'B05': 'rededge1', 'B06': 'rededge2',
         'B07': 'rededge3', 'B08': 'nir', 'B8A': 'nir08', 'B09': 'nir09', 'B11': 'swir16', 'B12': 'swir22'}
CLEAR_SCL = (4, 5, 6)
GDAL_ENV = {'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR', 'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
            'GDAL_HTTP_MULTIRANGE': 'YES', 'GDAL_HTTP_MERGE_CONSECUTIVE_RANGES': 'YES',
            'GDAL_HTTP_MAX_RETRY': '5', 'GDAL_HTTP_RETRY_DELAY': '2', 'VSI_CACHE': 'TRUE',
            'AWS_NO_SIGN_REQUEST': 'YES'}


def search_items(bbox_lonlat, start, end):
    import pystac_client
    cat = pystac_client.Client.open(STAC_URL)
    items = cat.search(collections=[COLLECTION], bbox=bbox_lonlat,
                       datetime='{}/{}'.format(start.isoformat(), end.isoformat())).item_collection()
    out = []
    for it in items:
        p = it.properties
        out.append({'id': it.id, 'date': it.datetime.date(), 'cloud': float(p.get('eo:cloud_cover', 100.0)),
                    'baseline': p.get('s2:processing_baseline'), 'offset_applied': p.get('earthsearch:boa_offset_applied'),
                    'epsg': int(p.get('proj:epsg') or it.properties.get('proj:code', 'EPSG:0').split(':')[-1]),
                    'hrefs': {k: it.assets[v].href for k, v in list(ASSET.items()) + [('SCL', 'scl')]}})
    return out


def needs_offset(item):
    try:
        b = float(item['baseline'])
    except (TypeError, ValueError):
        b = 0.0
    return b >= 4.0 and not item['offset_applied']


def sample_asset(href, epsg, lon, lat):
    """Value of the native pixel containing each point (exact transform, no warping);
    reads only the window spanning the points. 0 (nodata) -> -1."""
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window
    xs, ys = Transformer.from_crs('EPSG:4326', 'EPSG:{}'.format(epsg), always_xy=True).transform(lon, lat)
    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as d:
        inv = ~d.transform
        cols, rows = inv * (np.asarray(xs), np.asarray(ys))
        cols, rows = np.floor(cols).astype(int), np.floor(rows).astype(int)
        inside = (cols >= 0) & (rows >= 0) & (cols < d.width) & (rows < d.height)
        out = np.full(len(xs), -1, dtype=np.int32)
        if inside.any():
            c0, c1 = cols[inside].min(), cols[inside].max() + 1
            r0, r1 = rows[inside].min(), rows[inside].max() + 1
            win = d.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
            v = win[rows[inside] - r0, cols[inside] - c0].astype(np.int32)
            v[v == (d.nodata if d.nodata is not None else 0)] = -1
            out[inside] = v
    return out


def month_iter(start, end):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def slot_of(year_month, harvest_year):
    """YieldSAT slot index: calendar month counted from January of harvest_year - 1."""
    y, m = year_month
    return (y - (harvest_year - 1)) * 12 + (m - 1)


def extract_cluster(lon, lat, seeding, harvest, harvest_year, workers=16, log=print):
    """Returns per point: values (N,24,12) float32 (NaN = masked/undated), dates
    (N,24) as days since 1970 (NaN = undated), item ids (N,24), scl (N,24), and
    read statistics. seeding/harvest: per-point datetime.date arrays."""
    n = len(lon)
    start = min(seeding).replace(day=1)
    end_m = max(harvest)
    end = (end_m.replace(day=28) + datetime.timedelta(days=4)).replace(day=1) - datetime.timedelta(days=1)
    pad = 0.002
    bbox = [float(np.min(lon)) - pad, float(np.min(lat)) - pad, float(np.max(lon)) + pad, float(np.max(lat)) + pad]
    items = search_items(bbox, start, end)
    stats = {'items': len(items), 'scl_reads': 0, 'band_reads': 0}
    # SCL per item at every point (-1 = outside footprint / nodata)
    with ThreadPoolExecutor(workers) as ex:
        scl = list(ex.map(lambda it: sample_asset(it['hrefs']['SCL'], it['epsg'], lon, lat), items))
    stats['scl_reads'] = len(items)
    scl = np.stack(scl) if items else np.zeros((0, n), np.int32)          # (I, N)
    vals = np.full((n, 24, len(BANDS)), np.nan, dtype=np.float32)
    dates = np.full((n, 24), np.nan, dtype=np.float32)
    chosen = np.full((n, 24), -1, dtype=np.int32)
    clear_sel = np.zeros((n, 24), dtype=bool)
    ym = np.array([(it['date'].year, it['date'].month) for it in items]).reshape(-1, 2)
    order_key = [(it['cloud'], -it['date'].toordinal(), it['id']) for it in items]
    for p in range(n):
        for (y, m) in month_iter(seeding[p], harvest[p]):
            k = slot_of((y, m), harvest_year)
            if not 0 <= k < 24:
                continue
            cand = [i for i in range(len(items)) if ym[i, 0] == y and ym[i, 1] == m and scl[i, p] > 0]
            if not cand:
                continue
            clear = [i for i in cand if scl[i, p] in CLEAR_SCL]
            pool = clear or cand
            i = min(pool, key=lambda j: order_key[j])
            chosen[p, k] = i
            clear_sel[p, k] = bool(clear)
            dates[p, k] = items[i]['date'].toordinal() - datetime.date(1970, 1, 1).toordinal()
    # bands only for acquisitions some point selected with a clear view
    need = sorted({int(i) for i in chosen[clear_sel]})
    jobs = [(i, b) for i in need for b in BANDS]

    def read(job):
        i, b = job
        sel = np.flatnonzero((chosen == i).any(1))
        return job, sel, sample_asset(items[i]['hrefs'][b], items[i]['epsg'], lon[sel], lat[sel])
    with ThreadPoolExecutor(workers) as ex:
        for (i, b), sel, v in ex.map(read, jobs):
            bi = BANDS.index(b)
            for jj, p in enumerate(sel):
                for k in np.flatnonzero((chosen[p] == i) & clear_sel[p]):
                    x = float(v[jj])
                    if x < 0:
                        continue
                    if needs_offset(items[i]):
                        x = max(0.0, x - 1000.0)
                    vals[p, k, bi] = x
    stats['band_reads'] = len(jobs)
    scl_sel = np.where(chosen >= 0, np.take_along_axis(scl.T, np.maximum(chosen, 0), 1) if items else -1, -1)
    ids = np.array([it['id'] for it in items] + [''])
    item_id = np.where(chosen >= 0, ids[chosen], '')
    prov = [{'id': it['id'], 'date': it['date'].isoformat(), 'cloud': it['cloud'], 'baseline': it['baseline'],
             'offset_applied': it['offset_applied'], 'offset_subtracted': needs_offset(it)} for it in items]
    return {'values': vals, 'dates': dates, 'item_id': item_id, 'scl': scl_sel.astype(np.int16),
            'clear': clear_sel, 'stats': stats, 'items': prov}
