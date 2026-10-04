"""NOAA's nine US climate regions (Karl & Koss 1984; state-based), crop groups
and the pilot allocation (decisions D17/D18)."""
import numpy as np

NOAA_REGIONS = {
    'northwest': ('WA', 'OR', 'ID'),
    'west': ('CA', 'NV'),
    'southwest': ('UT', 'CO', 'AZ', 'NM'),
    'northern_rockies_plains': ('MT', 'WY', 'ND', 'SD', 'NE'),
    'upper_midwest': ('MN', 'WI', 'MI', 'IA'),
    'ohio_valley': ('IL', 'IN', 'OH', 'MO', 'KY', 'TN', 'WV'),
    'south': ('KS', 'OK', 'TX', 'AR', 'LA', 'MS'),
    'southeast': ('AL', 'GA', 'FL', 'SC', 'NC', 'VA'),
    'northeast': ('ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA', 'DE', 'MD', 'DC'),
}
REGIONS = tuple(NOAA_REGIONS)
STATE_REGION = {s: r for r, states in NOAA_REGIONS.items() for s in states}

# CDL cropland eligibility (D18): every cropland class is kept
CROPLAND_CODES = tuple(range(1, 62)) + tuple(range(66, 81)) + tuple(range(200, 256))
# crop groups used for cluster strata (points keep their exact CDL code)
GROUPS = ('corn', 'soybean', 'winter_wheat', 'spring_wheat', 'canola', 'other_crop')
GROUP_CODES = {'corn': (1,), 'soybean': (5,), 'winter_wheat': (24,), 'spring_wheat': (22, 23), 'canola': (31,)}
# YieldSAT crop vocabulary for knowledge strata / crop filters at training time
YIELDSAT_CROP = {1: 'corn', 5: 'soybean', 22: 'wheat', 23: 'wheat', 24: 'wheat', 31: 'rapeseed'}

# Pilot allocation (D17): clusters per region and the region's crop strata.
# 'wheat' = winter + spring wheat.
PILOT_ALLOCATION = {
    'upper_midwest': (15, ('corn', 'soybean')),
    'ohio_valley': (20, ('corn', 'soybean')),
    'northern_rockies_plains': (25, ('spring_wheat', 'corn', 'soybean', 'canola')),
    'south': (25, ('winter_wheat', 'soybean', 'corn')),
    'southeast': (10, ('corn', 'soybean', 'wheat')),
    'northwest': (12, ('wheat', 'canola')),
    'northeast': (8, ('corn', 'soybean')),
    'southwest': (5, ('wheat', 'corn')),
    'west': (5, ('wheat', 'corn')),
}
YEARS = (2021, 2022, 2023, 2024, 2025)


def group_lut():
    """uint8 CDL code -> group index (len(GROUPS) = not cropland)."""
    lut = np.full(256, len(GROUPS), dtype=np.uint8)
    for c in CROPLAND_CODES:
        lut[c] = GROUPS.index('other_crop')
    for g, codes in GROUP_CODES.items():
        for c in codes:
            lut[c] = GROUPS.index(g)
    return lut


def stratum_area(counts, stratum):
    """Area (pixel counts) of a pilot stratum from per-group counts (..., len(GROUPS))."""
    if stratum == 'wheat':
        return counts[..., GROUPS.index('winter_wheat')] + counts[..., GROUPS.index('spring_wheat')]
    return counts[..., GROUPS.index(stratum)]


def split_years(n, years=YEARS, offset=0):
    """Clusters per year, balanced (largest-remainder), rotated by ``offset`` so
    regions with few clusters do not all favour the first years."""
    base, extra = divmod(n, len(years))
    out = {y: base for y in years}
    for i in range(extra):
        out[years[(i + offset) % len(years)]] += 1
    return out


def largest_remainder(total, weights, floor=1):
    """Integer split of ``total`` proportional to ``weights`` with a per-item floor
    (while the total allows it)."""
    keys = list(weights)
    w = np.array([max(0.0, float(weights[k])) for k in keys])
    alloc = np.zeros(len(keys), dtype=int)
    live = w > 0
    if total <= 0 or not live.any():
        return dict(zip(keys, alloc.tolist()))
    f = min(floor, total // max(1, live.sum()))
    alloc[live] = f
    rest = total - alloc.sum()
    if rest > 0:
        share = w / w.sum() * rest
        alloc += np.floor(share).astype(int)
        rem = total - alloc.sum()
        order = np.argsort(-(share - np.floor(share)))
        for i in order[:rem]:
            alloc[i] += 1
    return dict(zip(keys, alloc.tolist()))
