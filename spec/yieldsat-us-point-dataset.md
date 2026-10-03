# US YieldSAT-style 10 m point dataset

**Built:** 2026-10-03. **Source:** `/home1/pupil/SMATousi/gdd-workspace/data`.
**Output:** `/home1/pupil/SMATousi/YieldSAT-US`.

This is a materialized dataset, not an adapter implementation. The user's final
scope was to build the dataset, download missing layers where needed, and leave
adapters for later. No existing repository code or source data was changed.
Temporary build/validation programs ran outside the repository. This specification
records the final data contract, decisions, counts and checks.

## Result and point definition

The primary export contains **90,757 field-year grid points**, representing
**596,553 finite yield measurements** across **28 distinct field-years**.
A location in a different year is a different sample.

| Crop | 2023 points | 2024 points | 2025 points | Total |
|---|---:|---:|---:|---:|
| Corn | 8,231 | 18,601 | 24,220 | 51,052 |
| Soybean | 24,113 | 15,592 | 0 | 39,705 |
| Total | 32,344 | 34,193 | 24,220 | **90,757** |

The 50 input folders include repeated exports. There are 21 exact duplicate
aliases and one alternate revision of an already represented field-year. Keeping
all 50 as independent samples would produce 177,854 rows and double-count many
measurements. All 50 full-time sampled source records are retained under `native/`;
`provenance/aliases.json` maps them to the 28 primary records.

The distinct field-years comprise 13 corn records (3/5/5 across 2023/2024/2025)
and 15 soybean records (10/5 across 2023/2024). `fields.json` also supplies 24
**overlap-based spatial groups** for later split construction. These are computed
from occupied cells, not asserted farm identities: link fields when occupied-cell
IoU >= 0.3 or intersection/minimum field area >= 0.8, then take connected groups.
The links and scores are saved in `provenance/physical_field_links.json`.

### Confirmed target aggregation

The supplied measurements are small polygons with the authoritative
`yieldMgHaM` attribute, in Mg/ha. The user explicitly approved:

1. Calculate each measurement polygon's centroid in projected coordinates.
2. Assign that centroid to one 10 m cell, separately within each field-year.
3. Take the **unweighted arithmetic mean** of finite `yieldMgHaM` observations
   assigned to that cell. One polygon record contributes one vote.

All source yield geometries use **EPSG:26915 (NAD83 / UTM zone 15N)**. The 10 m
lattice is anchored to coordinate multiples of 10 m; it is shared across fields.
For a field-local grid, `left = floor(min_centroid_x/10)*10` and
`top = ceil(max_centroid_y/10)*10`. Then
`col = floor((x-left)/10)` and `row = floor((top-y)/10)`.
Stored point coordinates are cell centers, not individual measurement positions.
A field only contributes cells containing at least one finite yield observation.

The portal's existing approximately 5 m yield rendering is **not** the source of
target averages. `yield_mg_h`, the similarly named intermediate column, is not
used. No yield scaling, outlier deletion, trimming or label interpolation was
added. For every cell, observation count and population standard deviation are
stored. Original source-row membership and measurement values are retained in
the per-record HDF5 archives. The selected records contain 1,803 additional
layout/nonfinite-label features; they do not contribute yield observations.

### Source revision exception

`yld_corn_2025_0013` and `yld_corn_2025_0026` both identify source `N2503` in 2025.
They share 19,749 exact geometry+yield records, but the first contains four
additional/different polygon fragments and the second contains three. Their
grids coincide, but two cell means differ (maximum difference about 0.622219
Mg/ha) and one cell count differs by one observation.

The primary dataset consistently uses the lexicographically first export,
`yld_corn_2025_0013`, with 19,753 finite observations. It does not blend two
processing revisions or pretend that this pair is an exact duplicate. Both
versions remain in `native/`; the exception and differing arrays are recorded
in the alias table. The other 21 aliases pass exact coordinate/target/count/std
and raw measurement/membership comparisons.

## Files and YieldSAT compatibility

```text
YieldSAT-US/
  US/merge_s2-soil-dem-weather-coords.nc
  fields.json
  band_schema.json
  manifest.json
  README.md
  native/<original_record_id>.h5          # all 50 records
  fields/<original_record_id>.json
  reports/summary.json
  reports/validation.json
  reports/field_counts.csv
  reports/layer_counts.csv
  reports/point_layer_counts.csv.gz
  provenance/aliases.json
  provenance/physical_field_links.json
  provenance/ssurgo_categories.json
  provenance/build_policy.json
  provenance/source_records/...
  provenance/supplements/...
```

The main file is genuine NetCDF4/HDF5, with dimension scales and string band
coordinates. It preserves the original YieldSAT core vocabulary and 24-slot
layout, with additional arrays rather than silently changing the 120-band core.

| Variable | Shape | Meaning |
|---|---|---|
| `sample` | `(90757,24,120)`, float32 | Original YieldSAT canonical band names/order. |
| `band` | `(120,)` | Names for `sample`. |
| `sample_extra` | `(90757,24,134)`, float32 | Additional source-specific predictors and ancillary values. |
| `extra_band` | `(134,)` | Names for `sample_extra`. |
| `times` | `(90757,24)`, float64 | Days since 1970-01-01, proleptic Gregorian. |
| `target` | `(90757,)`, **float64** | Arithmetic cell means, Mg/ha. |
| `index` | `(90757,)`, string | Deterministic UUID per field-year/grid cell. |
| `row`, `col` | `(90757,)` | Field-local grid coordinates, with integer-string dictionaries. |
| `x`, `y`, `longitude`, `latitude` | `(90757,)` | Cell center in EPSG:26915 and WGS84. |
| `yield_observation_count`, `target_std` | `(90757,)` | Target-side aggregation diagnostics. |
| `layer_available` | `(90757,254)`, uint8 | At least one finite slot for each channel. |
| `layer_valid_slot_count` | `(90757,254)`, uint8 | Number of finite slots, 0–24. |
| `available_layer_count` | `(90757,)` | All core+extra channels with any finite observation. |
| `available_predictor_layer_count` | `(90757,)` | Same count excluding uncertainty/coordinates/SCL. |
| `target_quality_flag` | `(90757,)` | Target-only quality bitmask; see below. |
| `target_implausible_observation_count` | `(90757,)` | Raw contributing measurements outside upstream plausible bounds. |

Categorical variables use YieldSAT's numeric-code-plus-numeric-string-attribute
convention: `field_shared_name`, `farm_identifier`, `country`, `crop`, `year`,
`seeding_date`, `harvesting_date`, `seeding_date_type`, plus source and spatial
identity codes. Country is **US**. Farm identity is genuinely unknown; `farm0` in
schema-compatible field names is a placeholder, not an inferred farm. Coordinates
are explicitly defined; the three core coordinate channels are unit-sphere xyz
from cell-center longitude/latitude, not an attempted reconstruction of the
original corpus's undocumented convention.

**Adapters are still required.** The current repository whitelists the four
original countries and their snapshots. It also does not consume `sample_extra`,
the new source semantics, target quality flags, or unknown planting dates.
No country aliases, fake source snapshots or hidden adapter patches were added.

## Layers and coverage

Each point has **254 declared channel positions**:
**120 core + 134 extra**. This count includes ancillary quantities, not only
scientific predictors. There are **196 predictor channels**, **54 SoilGrids
uncertainty channels**, **3 coordinates**, and **1 selected-date SCL channel**.
The target and target-quality diagnostics are excluded from all these counts.

| Family | Channels | Notes |
|---|---:|---|
| Sentinel-2 | 12 | B01/B09 supplemented at selected dates; other ten bands already present. |
| ERA5-Land | 18 | Four compatibility channels and 14 additional daily-variable aggregates. |
| Elevation | 1 | USGS 3DEP elevation. |
| Terrain | 5 | Aspect, slope, TWI, profile and plan curvature. |
| SoilGrids | 108 | Nine properties × six depths × mean/uncertainty. |
| POLARIS | 36 | Six properties × six depths. |
| SSURGO | 44 | Seven properties × six depths, plus drainage/hydrologic categories. |
| NAIP | 4 | R/G/B/NIR, with acquisition dates retained. |
| CDL | 3 | Explicit 2023, 2024 and 2025 class channels. |
| Sentinel-1 | 8 | VV/VH separately for ascending tracks 63, 92, 136 and 165. |
| Daymet | 7 | All seven source daily variables. |
| SMAP | 4 | AM/PM moisture and vegetation water content; raw flags retained separately. |
| Coordinates | 3 | Ancillary xyz. |
| S2 SCL | 1 | Ancillary selected-acquisition quality code. |
| **Total** | **254** | |

Across the primary points, **246–250 channels** have at least one valid slot;
median **248**, mean approximately **248.691**. Restricting this to predictor
channels gives **188–192 per point**. The unavailable channels are geographically
inapplicable/missing S1 orbit-track channels and, for 2,105 points, SMAP values
without acceptable retrieval quality. Missing is represented by NaN, never by a
fabricated zero. These counts mean “observed at least once,” not “complete on
all 24 dates.” Static values repeat across slots for compatibility and are not
24 independent measurements.

`reports/point_layer_counts.csv.gz` has one record for **every point**, with UUID,
field/year/crop, row/col, yield measurement count, total/core/extra/predictor
availability and counts per family. `layer_counts.csv` reports coverage of each
named channel; `field_counts.csv` reports each distinct field-year.

### Supplementary downloads

The source had all its expected families but only ten S2 bands and six SoilGrids
properties at four depths. Public downloads added:

- **B01 and B09 for all 672 selected S2 acquisitions**: 1,344 native 60 m band
  windows, matched to the exact indexed Planetary Computer STAC item IDs. Zero
  required selected-band supplements are missing. The source rasters were not
  changed. Storage access tokens were cached in process memory and not saved.
- SoilGrids `cec,cfvo,clay,nitrogen,phh2o,sand,silt,soc,bdod`, each at
  `0–5,5–15,15–30,30–60,60–100,100–200 cm`, requesting mean and uncertainty.
  Responses for all 50 source records are archived, including all 28 primary
  field-years. Core SoilGrids means and uncertainty channels are complete.

SoilGrids is a **field-centroid query broadcast to the field**, not a claimed
10 m soil survey. Raw mapped values and the API's units/d_factor are retained.
No unverified uncertainty weighting or physical-scale conversion was applied.

## Spatial and temporal processing

The output's 10 m lattice is a common join grid, not the native information
resolution of every source. CRS reprojection uses actual raster metadata.

- DEM, slope, TWI, curvature and NAIP: area-weighted averaging of valid source
  pixels into the 10 m grid. Aspect uses averaged sine/cosine and `atan2`, not an
  arithmetic average across the 0/360 boundary.
- S2/SCL, S1, POLARIS and CDL: nearest-source-pixel sampling after reprojection.
  Upsampling 20/60/250 m sources does not create additional spatial information.
- SSURGO: cell-center map-unit assignment. Numerical depth properties are
  averaged using component percentage × overlapping horizon thickness; missing
  values are excluded from the denominator. Drainage/hydrologic codes use the
  dominant component. The original component/horizon tables and codebooks are
  preserved, so these aggregates do not replace the source relationships.
- POLARIS OM remains **log10(percent)**. S1 remains **linear gamma0 power**.
  Core `curvature` is explicitly the source's **profile curvature**; plan
  curvature is an extra channel. These semantics differ from some original
  YieldSAT sources and must be respected by the future adapter.

The 24 slots divide the existing annual ERA5 source window into equal-duration
bins. Choose the S2 acquisition with lowest indexed in-field obscured fraction
in each bin, breaking ties toward the latest date. Its actual date is the slot
date; an empty imagery bin uses its end date for weather, with no invented S2
observation. In this completed export every primary field has 24 selected
acquisitions. Only SCL classes **4,5,6** contribute to core S2 values.

For core weather compatibility, temperatures are converted from source Celsius
to Kelvin and summed over **inclusive `[previous_date,current_date]`** intervals;
precipitation millimetres are converted to metres and summed. The first core
weather interval is missing, matching the original contract. Inclusive intervals
repeat a boundary day; do not sum these slots as disjoint seasonal rainfall.
Extra daily weather variables use means over **non-overlapping** intervals,
first starting at the known source-window start. Their full daily values remain
available. S1 uses mean positive linear power in the same non-overlapping
intervals, with tracks separated. SMAP means use retrieval flag exactly zero;
soil moisture additionally requires the source's `[0.02,0.5]` valid range.

The `native/` files preserve full acquisition-time **point samples**, not
unresampled source rasters: all ten original S2 bands and SCL at every indexed
acquisition, S1 VV/VH with dates/orbits, daily ERA5/Daymet/SMAP including raw QA
flags, NAIP dates, source measurement membership, core/extra views and metadata.
The 28 primary records contain **2,536 original S2 acquisitions** and **924 S1
acquisitions**. The extra downloaded B01/B09 windows cover the selected 672 dates;
we do not claim they were downloaded for every unselected native acquisition.
All 50 source-record archives remain available, including aliases/revisions.

CDL includes each stored release year even when later than a sample's harvest;
NAIP can be outside the observation window. This is a comprehensive archive,
**not a pre-cutoff-ready training tensor**. Future adapters must enforce product
release dates, observation cutoffs and vintage policies. Nothing was silently
made contemporaneous. Seven primary field-years (21,373 points) lack planting
dates: their date dictionary value is `UNKNOWN`, with type `missing`. Actual
dates were not fabricated from imagery or harvest timing.

## Target quality: retained, explicitly flagged

The source project's plausibility diagnostic is `0 < yield <= 50 Mg/ha`, used
there to flag, not remove, measurements. The export follows that policy:

- **2,310 cells** contain at least one contributing source value outside those
  bounds; **17,383 source measurements** are flagged.
- **2,309 cell averages** are outside the bounds.
- **2,376 cells** belong to the primary upstream review record
  `yld_soybean_2024_0003`; its repeated source aliases have the same flag.

The anomalous source values include enormous finite yields, not ordinary
agronomic variation. These labels must not silently enter model training as
trusted ground truth. They remain in this requested raw arithmetic-mean dataset;
a subsequent explicit label-quality policy can exclude or investigate them.

`target_quality_flag` uses bit 1 for any implausible contributing value, bit 2
for an implausible cell mean, and bit 4 for upstream field review. Flags, counts
and target standard deviations are **target-side metadata only**, never sensor
inputs or knowledge-pretraining evidence.

Targets use float64 (a deliberate precision extension to the original format),
because float32 introduces large absolute rounding errors on the corrupt extreme
values. Sensor arrays remain float32. No target clipping or alternative “cleaned”
mean is hidden in the dataset.

## Verification and provenance

The final validation recomputes cell assignments and arithmetic means directly
from the original shapefiles for all 90,757 points / 596,553 measurements. It
checks counts, cell centers, unique UUIDs, NetCDF dimensions, band ordering,
finite-value masks, per-point layer counts, temporal ordering, first-interval
weather masking, S2 cloud masking, complete canonical SoilGrids values and
target-quality flags. NetCDF arrays are also compared with the per-field HDF5
views. Floating arithmetic is checked at tight float64 tolerances; targets are
not compared only at loose model-training precision.

Source vector checksums, input raster sizes/mtimes, catalogs, CSVs, original
metadata, downloaded SoilGrids JSON, S2 STAC records and band windows are retained
under provenance. Final SHA-256, summary and verification results are in
`manifest.json`, `reports/summary.json` and `reports/validation.json`.

Spatial resampling follows the documented
[Rasterio resampling methods](https://rasterio.readthedocs.io/en/stable/topics/resampling.html)
and [reprojection interface](https://rasterio.readthedocs.io/en/stable/topics/reproject.html).
Centroids are computed in the metric projected CRS, following the
[GeoPandas CRS guidance](https://geopandas.org/en/stable/docs/user_guide/projections.html).
The neighboring downloader's `specs/data-layers.md`, its yield exporter and
`ingestion/portal_job.py` supply the local source semantics and plausibility
bounds. Neither repository's existing source code was edited.
