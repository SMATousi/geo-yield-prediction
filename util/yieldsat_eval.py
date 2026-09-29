# --------------------------------------------------------
# YieldSAT evaluation and spatial reconstruction (YS-08, YS-09).
#
# Metrics are reported in target units (t/ha) on valid targets only:
#   pixel-weighted  - every held-out cell counts once
#   field-balanced  - per-field-season errors averaged with equal weight
#   field-level     - per-season mean prediction vs mean target
# each overall, per country, per crop, and macro-averaged over countries so
# the largest country cannot dominate a pooled number. ``r2`` is the
# coefficient of determination (1 - SSE/SST); ``pearson_r2`` is the squared
# correlation used by util.metrics.R2_Score, reported separately.
#
# Point predictions are scattered back onto each field's verified grid
# (geometry from Raw.zip): unobserved cells stay NaN, out-of-bounds or
# duplicate cells raise, and GeoTIFFs are written only with a verified
# transform/CRS.
# --------------------------------------------------------

import numpy as np


def regression_metrics(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    n = int(y.size)
    if n == 0:
        return {'n': 0}
    err = p - y
    sst = float(((y - y.mean()) ** 2).sum())
    out = {'n': n, 'rmse': float(np.sqrt(np.mean(err ** 2))), 'mae': float(np.mean(np.abs(err))),
           'bias': float(err.mean()),
           'r2': float(1 - (err ** 2).sum() / sst) if sst > 0 else float('nan')}
    out['pearson_r2'] = float(np.corrcoef(y, p)[0, 1] ** 2) if n > 1 and y.std() > 0 and p.std() > 0 else float('nan')
    return out


def _field_balanced(y, p, season):
    per = []
    for s in np.unique(season):
        sel = season == s
        e = p[sel] - y[sel]
        per.append((np.sqrt(np.mean(e ** 2)), np.mean(np.abs(e))))
    per = np.array(per)
    return {'fields': int(len(per)), 'rmse': float(per[:, 0].mean()), 'mae': float(per[:, 1].mean())}


def _field_level(y, p, season):
    seasons = np.unique(season)
    ym = np.array([y[season == s].mean() for s in seasons])
    pm = np.array([p[season == s].mean() for s in seasons])
    return regression_metrics(ym, pm)


def evaluate_predictions(pred, target, season, season_meta):
    """season_meta: list indexed by season id with 'country' and 'crop'."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    season = np.asarray(season)
    ok = np.isfinite(target)
    pred, target, season = pred[ok], target[ok], season[ok]
    country = np.array([season_meta[s]['country'] for s in season])
    crop = np.array([season_meta[s]['crop'] for s in season])

    def block(sel):
        return {'pixel': regression_metrics(target[sel], pred[sel]),
                'field_balanced': _field_balanced(target[sel], pred[sel], season[sel]),
                'field_level': _field_level(target[sel], pred[sel], season[sel])}

    out = {'overall': block(np.ones_like(target, dtype=bool)), 'per_country': {}, 'per_crop': {},
           'per_country_crop': {}}
    for c in np.unique(country):
        out['per_country'][str(c)] = block(country == c)
        for k in np.unique(crop[country == c]):
            out['per_country_crop']['{}/{}'.format(c, k)] = block((country == c) & (crop == k))
    for k in np.unique(crop):
        out['per_crop'][str(k)] = block(crop == k)
    pc = out['per_country'].values()
    out['macro_country'] = {
        'pixel_rmse': float(np.mean([v['pixel']['rmse'] for v in pc])),
        'pixel_r2': float(np.mean([v['pixel']['r2'] for v in pc])),
        'field_level_rmse': float(np.mean([v['field_level']['rmse'] for v in pc])),
    }
    return out


def scatter_to_grid(rows, cols, values, height, width):
    """Scatter per-cell values onto an (height, width) grid; NaN elsewhere.
    Raises on out-of-bounds or duplicate cells."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if rows.size and (rows.min() < 0 or cols.min() < 0 or rows.max() >= height or cols.max() >= width):
        raise ValueError('cells fall outside the {}x{} field grid'.format(height, width))
    flat = rows * width + cols
    if np.unique(flat).size != flat.size:
        raise ValueError('duplicate cells cannot be scattered')
    grid = np.full((height, width), np.nan, dtype=np.float32)
    grid[rows, cols] = values
    return grid


def reconstruct_field(season_record, rows, cols, values, geometry=None):
    """Return (grid, georef). With verified geometry the grid is the field's
    raster grid and georef carries the transform/EPSG; without it the grid is
    the rows/cols bounding box from the origin and georef is None."""
    if geometry is not None:
        grid = scatter_to_grid(rows, cols, values, geometry['height'], geometry['width'])
        return grid, {'transform': geometry['transform'], 'epsg': geometry['epsg']}
    h = int(np.max(rows)) + 1
    w = int(np.max(cols)) + 1
    return scatter_to_grid(rows, cols, values, h, w), None


def write_geotiff(path, grid, georef):
    if georef is None:
        raise ValueError('refusing to write a GeoTIFF without a verified transform')
    import rasterio
    from rasterio.transform import Affine

    with rasterio.open(path, 'w', driver='GTiff', height=grid.shape[0], width=grid.shape[1],
                       count=1, dtype='float32', crs='EPSG:{}'.format(georef['epsg']),
                       transform=Affine(*georef['transform']), nodata=np.nan) as dst:
        dst.write(grid.astype(np.float32), 1)
