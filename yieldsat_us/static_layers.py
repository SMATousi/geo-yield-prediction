"""Static layers for US points in YieldSAT conventions (NP-04d).

Soil: SoilGrids 2.0 (ISRIC, 250 m, Homolosine VRTs), 8 properties x 6 depths,
``mean`` (mapped units: clay/silt/sand g/kg, soc dg/kg, phh2o pH*10, ...) and
``uncertainty``; cubic interpolation at the point, as YieldSAT's cubic upsampling.

DEM/terrain: NASADEM (reprocessed SRTM, 1 arcsec, EPSG:4326) from Planetary
Computer. Derivatives on the native lat/lon grid, as recovered from YieldSAT's
Raw.zip (spec D3): slope = Horn rise/run with horizontal spacing in degrees
(-> YieldSAT's ~84,000 x rise/run scale), aspect = downslope azimuth clockwise
from north (degrees), curvature = Zevenbergen-Thorne in 1/100 m (scale to be
calibrated against YieldSAT, NP-04a); TWI is NaN, as in most of YieldSAT.
Values are cubic-interpolated at the point.
"""
import numpy as np

SOIL_PROPS = ('cec', 'cfvo', 'clay', 'nitrogen', 'phh2o', 'sand', 'silt', 'soc')
SOIL_DEPTHS = ('0-5', '5-15', '15-30', '30-60', '60-100', '100-200')
SOIL_URL = '/vsicurl/https://files.isric.org/soilgrids/latest/data/{p}/{p}_{d}cm_{s}.vrt'
HOMOLOSINE = '+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs'
GDAL_ENV = {'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR', 'GDAL_HTTP_MAX_RETRY': '5',
            'GDAL_HTTP_RETRY_DELAY': '3', 'VSI_CACHE': 'TRUE'}


def _cubic_sample(arr, cols, rows, nodata=None):
    from scipy.ndimage import map_coordinates
    a = arr.astype(np.float64)
    bad = ~np.isfinite(a) if nodata is None else (a == nodata) | ~np.isfinite(a)
    if bad.all():
        return np.full(len(cols), np.nan)
    fill = np.where(bad, np.nanmean(np.where(bad, np.nan, a)), a)
    v = map_coordinates(fill, [rows, cols], order=3, mode='nearest')
    near_bad = map_coordinates(bad.astype(float), [rows, cols], order=1, mode='nearest') > 0.01
    v[near_bad] = np.nan
    return v


def _window_sample(src, xs, ys, halo=3):
    """Read the window spanning the points (+halo) and cubic-sample at pixel centres."""
    from rasterio.windows import Window
    inv = ~src.transform
    c, r = inv * (np.asarray(xs), np.asarray(ys))
    c0, r0 = int(np.floor(c.min())) - halo, int(np.floor(r.min())) - halo
    c1, r1 = int(np.ceil(c.max())) + halo, int(np.ceil(r.max())) + halo
    win = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0), boundless=True,
                   fill_value=src.nodata if src.nodata is not None else 0)
    return win, c - c0 - 0.5, r - r0 - 0.5


def soil_for_points(lon, lat, workers=8, uncertainty=False):
    """(N, 96): 48 means then 48 uncertainties, property-major and depth-ordered
    (YieldSAT SOIL then SOIL_UNCERTAINTY order). Uncertainty layers are NaN unless
    requested (YieldSAT excludes them by default; halves the remote reads)."""
    import rasterio
    from concurrent.futures import ThreadPoolExecutor
    from pyproj import Transformer
    xs, ys = Transformer.from_crs('EPSG:4326', HOMOLOSINE, always_xy=True).transform(lon, lat)
    stats = ('mean', 'uncertainty') if uncertainty else ('mean',)
    jobs = [(s, p, d) for s in stats for p in SOIL_PROPS for d in SOIL_DEPTHS]

    def one(job):
        s, p, d = job
        with rasterio.Env(**GDAL_ENV), rasterio.open(SOIL_URL.format(p=p, d=d, s=s)) as src:
            win, cc, rr = _window_sample(src, xs, ys)
            return _cubic_sample(win, cc, rr, nodata=src.nodata)
    with ThreadPoolExecutor(workers) as ex:
        cols = list(ex.map(one, jobs))
    out = np.full((len(xs), 96), np.nan, dtype=np.float32)
    out[:, :len(cols)] = np.stack(cols, 1)
    return out


def nasadem_items(bbox):
    import planetary_computer
    import pystac_client
    cat = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',
                                    modifier=planetary_computer.sign_inplace)
    return list(cat.search(collections=['nasadem'], bbox=bbox).items())


def terrain_for_points(lon, lat, halo_deg=0.01):
    """(N, 5): dem (m), aspect (deg), curvature, slope (degree-unit rise/run), twi (NaN)."""
    import rasterio
    from rasterio.merge import merge
    lon, lat = np.asarray(lon), np.asarray(lat)
    bbox = [lon.min() - halo_deg, lat.min() - halo_deg, lon.max() + halo_deg, lat.max() + halo_deg]
    items = nasadem_items(bbox)
    srcs = [rasterio.open(it.assets['elevation'].href) for it in items]
    with rasterio.Env(**GDAL_ENV):
        mosaic, tr = merge(srcs, bounds=bbox)
    for s in srcs:
        s.close()
    z = mosaic[0].astype(np.float64)
    z[z <= -32767] = np.nan
    dx, dy = abs(tr.a), abs(tr.e)                              # degrees (1/3600)
    p = np.pad(z, 1, mode='edge')
    a, b, c = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    d, e, f = p[1:-1, :-2], p[1:-1, 1:-1], p[1:-1, 2:]
    g, h, i = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * dx)      # eastward, per degree
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * dy)      # southward (row) direction, per degree
    slope = np.hypot(dzdx, dzdy)
    aspect = (np.degrees(np.arctan2(-dzdx, dzdy)) + 360.0) % 360.0
    # Zevenbergen-Thorne curvature in 1/100 m with metric spacing at the window latitude
    lat0 = np.radians(np.mean(bbox[1::2]))
    lx, ly = dx * 111320.0 * np.cos(lat0), dy * 110574.0
    D = ((d + f) / 2 - e) / lx ** 2
    E = ((b + h) / 2 - e) / ly ** 2
    curv = -2.0 * (D + E) * 100.0
    inv = ~tr
    cc, rr = inv * (lon, lat)
    cc, rr = cc - 0.5, rr - 0.5
    out = np.stack([_cubic_sample(z, cc, rr), _circular_sample(aspect, cc, rr), _cubic_sample(curv, cc, rr),
                    _cubic_sample(slope, cc, rr), np.full(len(lon), np.nan)], 1)
    return out.astype(np.float32)


def _circular_sample(deg, cols, rows):
    s = _cubic_sample(np.sin(np.radians(deg)), cols, rows)
    c = _cubic_sample(np.cos(np.radians(deg)), cols, rows)
    return (np.degrees(np.arctan2(s, c)) + 360.0) % 360.0
