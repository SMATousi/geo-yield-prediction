# Unlabeled US YieldSAT-style point dataset

Build date: 2026-10-04. Source:
`/home1/pupil/SMATousi/gdd-pretraining/data/pretraining_aoi_2025`.
Output: `/home1/pupil/SMATousi/YieldSAT-US-Pretraining`.

## Scope and point definition

This dataset is for sensor-only pretraining. It contains **13,202,010 points**,
each the center of a distinct 10 m cell inside the source AOI boundary and
classified as cropland by CDL 2025.
It contains **no yield targets**, target statistics, invented crop labels,
planting dates, or harvest dates. CDL is a remotely mapped predictor, not an
observed yield/crop label. Existing training code and adapters are not changed.

The source is one AOI record, `pretraining_aoi_2025`, rather than individual
agricultural fields. Its boundary is the union of 432 PlanetScope scene footprints
from 2025, as recorded in the downloader manifest. PlanetScope pixel data is not
stored under the supplied source directory and is not part of this export.
The documented source area is 336,782.5 ha in EPSG:5070.

The final selection is **cropland only**. Starting with the 33,663,464 cell
centers inside the exact AOI polygon, retain CDL 2025 codes **1–61, 66–80,
and 200–255**, as sampled at the exact transformed cell center. The observed
retained codes are listed individually in `provenance/cropland_selection.json`.
This includes hay/alfalfa, tree crops and fallow/idle cropland, and excludes
grass/pasture (176), forests, water, wetlands, developed land, shrubland, barren
land, aquaculture and nodata. The class interpretation follows the
[USDA 2025 CDL dictionary](https://www.nass.usda.gov/Research_and_Science/Cropland/metadata/metadata_CDL25_FGDC-STD-001-1998.htm).
This is a declared cropland selection policy, not the separate USDA multi-year
Cultivated Layer product. No cells are selected using yield, imagery visibility
or cross-modal completeness.

The filter removes **20,461,454 non-cropland points**, retaining **13,202,010**.
Corn (CDL 1) accounts for **5,671,934** points, soybean (CDL 5) for **6,461,956**,
and other retained classes for **1,068,120**. These are remote land-cover
classifications, not measured crop ground truth. Each point preserves its original
AOI grid ID; all sensor arrays use the same selection and row order.

Holes and spaces outside the union are excluded. There is no overlap or duplicate
sampling within this region-year. The AOI can overlap the separate labeled US
corpus geographically; selected cropland locations are retained. A coordinates-only
overlap audit is saved as `reports/labeled_spatial_overlap.json`. Pretraining/
evaluation split exclusions remain an explicit downstream experimental decision,
and no yield values are read or transferred for this audit.

The lattice is EPSG:26915, NAD83 / UTM zone 15N, anchored to multiples of 10 m,
consistent with the labeled US dataset. Its transform is
`(10, 0, 343070, 0, -10, 4419660)`, with 7,506 columns and 6,073 rows.
Rasterization uses cell-center inclusion (`all_touched=False`). Selected cells
are ordered by row then column. `point_id = row * 7506 + col`; these are unique
within this AOI-year and need not be consecutive. Geographic coordinates are
WGS84 cell centers. The point count times 100 m² is a projected-grid area, not an
exact reproduction of the source's equal-area polygon measurement.

The AOI manifest was opened with SQLite read-only immutable mode, avoiding the
downloader's schema-migration side effects. The boundary and source record are
copied into output provenance. Source files are not modified.

## Channels and missing layers

The logical arrays retain the labeled US dataset's **120 core and 134 additional
channel positions**, in the same order, with 24 temporal slots. NaN means absent
or invalid; missing data is never replaced by zero. Availability reports count a
channel when it has at least one finite value across the 24 slots.

| Stored source family | Channel positions populated by this build | Processing |
|---|---:|---|
| Sentinel-2 | 10 reflectance + 1 SCL | Selected 24 dates; cloud masking for reflectance |
| Sentinel-1 | 4 | Ascending tracks 63 and 136, VV/VH separately |
| DEM | 1 | Valid-area mean to the 10 m grid |
| Terrain | 5 | Slope, aspect, profile/plan curvature, TWI |
| POLARIS | 36 | Six properties at six depths |
| SSURGO | 44 | Seven numerical properties at six depths, two categories |
| CDL | 1 | Stored 2025 release only |
| SMAP | 4 | AM/PM soil moisture and vegetation water content |
| Coordinates | 3 | Unit-sphere XYZ derived from longitude/latitude |
| Total potentially populated positions | **109** | Actual finite coverage varies by point |

The supplied directory and its downloader manifest contain exactly eight
completed source families: DEM, terrain, SSURGO, POLARIS, CDL, SMAP, Sentinel-1
and Sentinel-2. They do **not** contain ERA5-Land, Daymet, SoilGrids, NAIP,
Sentinel-2 B01/B09, CDL 2023/2024 or Sentinel-1 tracks 92/165. The corresponding
145 positions stay missing. The build includes all families available in the
requested directory; it does not claim that this AOI has all the products of
the separate labeled-field corpus. No region-wide values are fabricated by
copying a nearby labeled field or broadcasting a single regional centroid.

The original source indexes and metadata are retained. Native source raster
resolution is preserved in provenance: the grid does not turn 20–30 m optical,
soil or land-cover pixels, or 9 km SMAP cells, into independent 10 m observations.
SSURGO categories have an explicit codebook local to this dataset; category codes
must be joined by meaning when combining corpora.

## Spatial and temporal processing

DEM and terrain use valid-area averaging into EPSG:26915 cells. Aspect averages
sine and cosine before `atan2`, respecting the circular 0/360-degree boundary.
Near-zero resultant vectors are missing. Terrain comes from this AOI's actual
`*_8m.tif` products and DEM from `dem_10m.tif`; these differ from the labeled
corpus's 1 m lidar-derived inputs. Core `curvature` means **profile curvature**;
plan curvature is an additional channel.

POLARIS, CDL, Sentinel-1 and Sentinel-2 use exact cell-center coordinate
transformation followed by the containing source pixel (nearest-pixel sampling).
Approximate warp transformers are not used for those layers. A source-pixel
audit exposed neighboring-pixel choices near boundaries under default warping;
these layers were rebuilt using exact transformed centers before final validation.
POLARIS organic matter remains log10(percent). CDL class zero and raster nodata
are missing. Raw Sentinel-2 values retain source digital-number units; a later
adapter must apply the source scale and processing-baseline offsets consistently.

SSURGO assigns the map unit containing each cell center. For each target depth,
numerical properties are weighted by component percentage times overlapping
horizon thickness, excluding missing values from numerator and denominator.
Drainage class and hydrologic group use the dominant component. Map-unit keys,
original mapunit/horizon tables and the category codebook remain available.

The source window is 2025-01-01 through 2026-01-01 inclusive, following the AOI's
recorded coverage window. It is divided into 24 equal-duration bins. Within each
bin, select the Sentinel-2 acquisition with smallest indexed AOI-obscured
fraction, breaking ties toward the latest date and then item ID. One missing
cloud score is treated as 1.0 (unknown/worst) for selection. The indexed
score belongs to the source product; it does not guarantee valid imagery at
every point. A bin without an acquisition would have missing S2 and its end date
as the slot date. Selected item IDs and dates are explicit provenance.

The source has 158 Sentinel-2 records, all in EPSG:32615. The 24 selected
acquisitions are sampled to points. The ten available bands are B02, B03, B04,
B08, B05, B06, B07, B8A, B11 and B12; only SCL classes 4, 5 and 6 retain
reflectance. SCL is stored separately, including cloudy classifications.
No clear-sky values are interpolated or borrowed from another date.

The source has 52 Sentinel-1 records. VV and VH remain linear gamma0 power.
Positive finite observations are averaged per point, relative track and
non-overlapping interval `(previous_slot_date, slot_date]`; the first interval
starts at the source-window start. Ascending tracks 63 and 136 remain separate.
No last-interval extension beyond the last selected slot date is implied.

SMAP contains 70 EASE-Grid 2.0 cells across 366 dates; 58 of those cells
contain selected AOI point centers. Each point is transformed
to EPSG:6933 and assigned to its native 9 km cell using the source product's
9,008.055210146605 m lattice. The stored cell identifier indexes rows 290–296
and columns 912–921. Daily values are averaged over the same non-overlapping
slot intervals, using retrieval-quality flag exactly zero; soil moisture also
requires `0.02 <= value <= 0.5`. AM and PM remain separate. The full original
long-format daily table, including flags and missing rows, is preserved.

This is an archival pretraining dataset. Later adapters must enforce forecasting
cutoffs and product-release dates when appropriate. The AOI's 2026-01-01 window
anchor is not an actual harvest date. Neither the original 158-date S2 nor the
52-date S1 stack is duplicated as a full-time point tensor: the full indexes
and immutable source-raster paths are retained, while this export materializes
the 24-slot view. Resampling other dates requires the original source rasters.

## Storage contract

```text
YieldSAT-US-Pretraining/
  US/merge_s2-soil-dem-weather-coords.nc
  layers/<channel>.h5
  layers/<channel>__<slot>.h5
  layers/{row,col,index,longitude,latitude,smap_cell_index,ssurgo_mukey}.h5
  band_schema.json
  manifest.json
  README.md
  SPEC.md
  provenance/aoi_record.json
  provenance/aoi_mask.npy                  # full footprint, before crop filtering
  provenance/cropland_mask.npy             # final point selection
  provenance/cropland_selection.json
  provenance/grid.json
  provenance/selected_s2.json
  provenance/times.json
  provenance/ssurgo_categories.json
  provenance/source_inventory.json
  provenance/source_metadata/...
  reports/build.json
  reports/summary.json
  reports/layer_counts.csv
  reports/point_layer_counts.nc
  reports/validation.json
```

The main file is a NetCDF4/HDF5 **virtual dataset**. Logical arrays are
`sample[index, time, band]` and `sample_extra[index, time, extra_band]`, with
shapes `(13202010, 24, 120)` and `(13202010, 24, 134)`. Static physical arrays
are referenced across all time slots; dynamic physical arrays have one file per
channel and slot. Absent source channels are NaN virtual fill values. Compression
is gzip with shuffle. Processing works one physical array at a time rather than
allocating the full logical tensor.

**Move or copy the complete output directory**, keeping `US/` and `layers/` in
the same relative relationship. The small main `.nc` file is not a standalone
copy of all measurements. A missing virtual source file can appear as fill data,
so consumers must check the manifest's file inventory before training. The
virtual view requires an HDF5-backed reader supporting virtual datasets;
training adapters may alternatively read the physical arrays directly. Physical
arrays use 262,144-point compression chunks. Prefer spatially contiguous batch
reads or an adapter cache; arbitrary single-point reads across hundreds of
physical arrays can cause substantial read amplification.

There is no `target` variable or dummy zero-yield vector. Numeric metadata includes
point ID, local row/column, longitude/latitude, and SMAP cell index. Full region,
CRS, year and grid information are file attributes/provenance. No fictitious farm
or field labels are inserted to satisfy the supervised loader.

The per-point report stores each point ID, available-channel count,
available-predictor count, count of channels finite in all 24 slots, and total
finite slot-channel observations. Reports exclude all ground truth by design.

## Validation and completion

The completed build is checked for unique row-major point IDs, exact grid
membership, cropland eligibility for every point, coordinate transformations, per-array length, temporal dimensions,
all finite-value availability counts, all selected S2 cloud masks, 1,280 independent source-pixel spot checks,
696 SMAP source-table aggregation checks, relative
virtual-source resolution, and absence of target variables. Original source
file sizes and modification timestamps are checked for changes during the run.
Checksums and final measured storage/count summaries are written only after
validation completes. Counts and coverage are recorded in `reports/summary.json`
and `reports/layer_counts.csv`; the per-point report supplies the exact count
for each of the 13,202,010 points.

Temporary build and validation programs execute outside the repository. This
specification is the only project addition; existing scripts, dependencies,
training adapters and source datasets remain unchanged.
