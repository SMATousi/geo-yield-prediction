# --------------------------------------------------------
# YieldSAT per-field geometry and provenance recovery (YS-02).
#
# Reads the companion raw release archive (``Raw.zip``) strictly read-only,
# without extraction: each field's metadata JSON (projected CRS, WGS84
# centroid, provider, quality, dates) and the header of its DEM raster (grid
# affine/shape; the DEM, S2 and yield-mask rasters share one 10 m grid). The
# result is a field geometry table used for
#   * physical-field identity: field numbers are unique per *season*, so
#     seasons of the same ground are grouped by footprint overlap;
#   * geographic (block) splits from verified centroids;
#   * validated georeferencing of reconstructed point predictions.
# Scalar ``yield_ground_truth`` is kept as a target-side diagnostic only.
# --------------------------------------------------------

import json
import zipfile

import numpy as np

from dataset.yieldsat_schema import parse_field_shared_name

YIELD_MASK = 'yield_masks/mean_scaled_yield_masked_regional_statistical_outlier.tif'
TARGET_SIDE_KEYS = ('yield_ground_truth', 'area_ground_truth', 'min_yield_per_hectare',
                    'max_yield_per_hectare', 'standard_moisture', 'yieldmap_quality',
                    'quality_density')


def _vsizip(raw_zip, member):
    return '/vsizip/{}/{}'.format(raw_zip, member)


def extract_field_geometry(raw_zip, countries=None, fields=None):
    """Return one record per field season found in ``raw_zip``.

    ``fields`` optionally restricts extraction to a set of field_shared_names.
    """
    import rasterio

    records = []
    with zipfile.ZipFile(raw_zip) as zf:
        members = [n for n in zf.namelist()
                   if n.rsplit('/', 1)[-1].startswith('metadata-') and n.endswith('.json')]
        for member in sorted(members):
            country, name = member.split('/')[:2]
            if countries is not None and country not in countries:
                continue
            if fields is not None and name not in fields:
                continue
            meta = json.loads(zf.read(member))
            parts = parse_field_shared_name(name)
            dem = '{}/{}/dem/dem-{}.tif'.format(country, name, name)
            with rasterio.open(_vsizip(raw_zip, dem)) as src:
                transform = tuple(float(v) for v in tuple(src.transform)[:6])
                width, height = int(src.width), int(src.height)
                crs = src.crs.to_epsg() if src.crs is not None else None
            if meta.get('projected_crs_epsg') not in (None, crs):
                raise ValueError('{}: metadata EPSG {} != raster EPSG {}'.format(
                    name, meta.get('projected_crs_epsg'), crs))
            records.append({
                'field_shared_name': name,
                'country': country,
                'provider': parts['provider'],
                'farm': parts['farm'],
                'crop': parts['crop'],
                'year': parts['year'],
                'epsg': crs,
                'transform': transform,
                'width': width,
                'height': height,
                'centroid_lat': meta.get('centroid_latitude_wgs84'),
                'centroid_lon': meta.get('centroid_longitude_wgs84'),
                'adm_units': meta.get('adm_units'),
                'seeding_date': meta.get('seeding_date'),
                'harvesting_date': meta.get('harvesting_date'),
                'seeding_date_type': meta.get('seeding_date_type'),
                'target_side': {k: meta.get(k) for k in TARGET_SIDE_KEYS},
            })
    return records


def footprint(record):
    """Projected bounding box (xmin, ymin, xmax, ymax) of a field grid."""
    a, b, c, d, e, f = record['transform']
    if b != 0 or d != 0:
        raise ValueError('rotated grids are not supported')
    x0, x1 = c, c + a * record['width']
    y0, y1 = f, f + e * record['height']
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _iou(p, q):
    ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
    iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
    inter = ix * iy
    union = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter
    return inter / union if union > 0 else 0.0


def physical_field_groups(records, min_iou=0.3):
    """Group field seasons into physical fields by footprint overlap.

    Seasons in the same country and projected CRS whose grid footprints have
    IoU >= ``min_iou`` are linked (union-find), across farms and providers as
    well, so aliased ground cannot straddle a split. Returns
    ``(physical_id_by_name, cross_farm_links)``.
    """
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    boxes = [footprint(r) for r in records]
    cross_farm = []
    by_key = {}
    for i, r in enumerate(records):
        by_key.setdefault((r['country'], r['epsg']), []).append(i)
    for idx in by_key.values():
        for u in range(len(idx)):
            for v in range(u + 1, len(idx)):
                i, j = idx[u], idx[v]
                if _iou(boxes[i], boxes[j]) >= min_iou:
                    parent[find(i)] = find(j)
                    ri, rj = records[i], records[j]
                    if (ri['provider'], ri['farm']) != (rj['provider'], rj['farm']):
                        cross_farm.append((ri['field_shared_name'], rj['field_shared_name']))
    groups = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)
    physical = {}
    for members in groups.values():
        members.sort(key=lambda i: records[i]['field_shared_name'])
        head = records[members[0]]
        pid = '{}/{}/{}/pf-{}'.format(head['country'], head['provider'], head['farm'],
                                      parse_field_shared_name(head['field_shared_name'])['field'])
        for i in members:
            physical[records[i]['field_shared_name']] = pid
    return physical, cross_farm


def verify_cell_correspondence(raw_zip, record, rows, cols, targets, atol=1e-4):
    """Compare NetCDF per-cell targets with the raw yield-mask raster.

    Checks row/col bounds against the field grid, orientation (row = raster
    line, col = raster column) and value agreement, and reports how many
    valid raster cells have no NetCDF row. Returns a summary dict.
    """
    import rasterio

    member = '{}/{}/{}'.format(record['country'], record['field_shared_name'], YIELD_MASK)
    with rasterio.open(_vsizip(raw_zip, member)) as src:
        grid = src.read(1).astype(np.float64)
        nodata = src.nodata
        shape_ok = (src.width, src.height) == (record['width'], record['height'])
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    in_bounds = (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
    raster_vals = np.full(rows.shape, np.nan)
    raster_vals[in_bounds] = grid[rows[in_bounds], cols[in_bounds]]
    match = np.isclose(raster_vals, np.asarray(targets, dtype=np.float64), atol=atol)
    # orientation control: the same comparison with row/col swapped
    tb = (cols < grid.shape[0]) & (rows < grid.shape[1])
    tvals = np.full(rows.shape, np.nan)
    tvals[tb] = grid[cols[tb], rows[tb]]
    transposed = float(np.mean(np.isclose(tvals, targets, atol=atol)))
    valid_raster = np.isfinite(grid) & ((grid != nodata) if nodata is not None else True)
    covered = np.zeros_like(valid_raster)
    covered[rows[in_bounds], cols[in_bounds]] = True
    return {
        'field_shared_name': record['field_shared_name'],
        'raster_shape': list(grid.shape),
        'mask_matches_dem_grid': bool(shape_ok),
        'nodata': None if nodata is None else float(nodata),
        'rows_checked': int(rows.size),
        'out_of_bounds': int((~in_bounds).sum()),
        'target_match_fraction': float(match.mean()) if rows.size else None,
        'transposed_match_fraction': transposed,
        'valid_raster_cells': int(valid_raster.sum()),
        'valid_raster_cells_without_row': int((valid_raster & ~covered).sum()),
    }
