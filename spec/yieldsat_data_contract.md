# Alternative Data Contract — YieldSAT Preprocessed

**Status:** Inspected source snapshot and implementation requirements, 2026-09-28.
Not implemented. This is a second selectable data-contract branch, not a Git
branch or a replacement of the downloader/full-layer contract.

## 1. Scope and source location

Contract key: `yieldsat_preprocessed_v1`. The existing downloader branch
remains `downloader_native_layers_v1`, specified in
[layer_integration.md](./layer_integration.md). Both must feed the shared encoder,
fusion and checkpoint-transfer interfaces, with branch-specific provenance and
adapters. Neither existing demo loader consumes this NetCDF today.

The user initially supplied the Argentina subdirectory; this contract covers its
parent corpus and all four country partitions:

```text
/home1/pupil/SMATousi/YieldSAT/Preprocessed/
  Argentina/merge_s2-soil-dem-weather-coords.nc
  Brazil/merge_s2-soil-dem-weather-coords.nc
  Germany/merge_s2-soil-dem-weather-coords.nc
  Uruguay/merge_s2-soil-dem-weather-coords.nc
```

Select one or more countries explicitly in configuration. Do not silently include
all countries when a single-country experiment was requested. Use country-qualified
sample, farm and physical-field identities; local codes are not globally unique.

Make the root configurable; do not hardcode this host path in training code.
Source files are read-only. Store indexes, split manifests, statistics and caches
under a separately configured writable artifact root. Do not resume extraction,
repack archives, or alter the source while inspecting/ingesting it.

Extraction of the overall corpus was reported in progress. Read-only HDF5 metadata
reads succeeded for all four files; five bounded row probes and companion raw-field
checks were performed for Argentina only. All observed file lengths match the
respective `Preprocessed.zip` central-directory entries. This is not a completed
whole-file CRC/checksum verification or confirmation that extraction has finished.

| Country | Rows × slots × features | File bytes | Archive CRC32 | Field-season / farm dictionary counts | Crop dictionary |
|---|---|---|---|---|---|
| Argentina | 5,325,807 × 24 × 120 | 62,861,086,332 | `bae1a7f0` | 751 / 57 | corn, soybean, wheat |
| Brazil | 4,260,262 × 24 × 120 | 50,280,040,289 | `1c400537` | 551 / 9 | corn, soybean, wheat |
| Germany | 609,645 × 24 × 120 | 7,193,431,158 | `b89c8acb` | 299 / 6 | rapeseed, wheat |
| Uruguay | 2,177,206 × 24 × 120 | 25,693,566,077 | `41592d6f` | 572 / 10 | soybean |

Dictionary cardinalities are metadata, not a complete row-coverage or unique
physical-field audit. Country-specific units, upstream transformations and quality
have not been verified by these metadata reads.

Before training, require a completed, immutable source snapshot: extraction
completion evidence, expected size, readability, bounded integrity probes and a
recorded fingerprint/checksum policy. Recheck across index construction and worker
startup; refuse files still changing or caches built from another snapshot. Stable
size alone is not proof of complete extraction. Full integrity checks are a task,
not something already performed during this specification work.

## 2. Shared schema and country-specific decoding

The `.nc` files have HDF5 signatures and NetCDF dimension-scale metadata. Inspection
used `h5py` in read-only mode, not full-array loads. All inspected datasets are
contiguous (`chunks=None`) and uncompressed. A lazy API does not imply efficient
arbitrary field gathers or unlimited parallel readers; benchmark actual reads.

| Dataset | Shape / dtype | Observed meaning or handling |
|---|---|---|
| `sample` | `(N,24,120)`, float32 | Per-row temporal feature vectors; `_FillValue=NaN`. Argentina alone has approximately 61.35 GB of feature values. |
| `index` | `(N,)`, variable-length strings | Sample UUIDs in inspected rows. Validate uniqueness in the complete index. |
| `target` | `(N,)`, float32 | Per-row spatial yield target; `_FillValue=NaN`; units/scale unresolved. |
| `times` | `(N,24)`, float64 | Per-file date origin and calendar; NaN slots occur in Argentina probes. |
| `time_step` | `(24,)`, int64 | Values 0–23, positional slots, not dates or a guaranteed regular interval. |
| `band` | `(120,)`, strings | Authoritative feature names/order; select by name, not fixed numeric offsets. |
| `field_shared_name` | `(N,)`, uint16 | Country-local field/crop/year dictionary; entries are not necessarily unique physical fields. |
| `farm_identifier` | `(N,)`, uint8 | Country-local farm dictionary; actual included coverage needs index audit. |
| `country`, `crop`, `year` | `(N,)`, uint8 | Country-local dictionaries; crops and year ranges vary by country. |
| `seeding_date`, `harvesting_date` | `(N,)`, uint16 | Attribute dictionaries of date strings, not epoch days. |
| `seeding_date_type` | `(N,)`, uint8 | Dictionary includes `provided_by_farmer`. |
| `row`, `col` | `(N,)`, uint16 | Encoded integer grid positions, with dictionaries; verify indexing and raster correspondence. |
| `stats-min/max/mean/std` | Each `(120,)`, float32 | Supplied feature statistics; population/procedure unknown. Not approved normalization inputs. |

Decode categorical/date metadata using each file's variable numeric-string attribute
keys. Code zero is a valid category: Argentina/Brazil `crop=0` means corn, Germany
`crop=0` means rapeseed, and Uruguay `crop=0` means soybean; Argentina `year=0` means 2017;
do not interpret HDF5's default zero fill value as universal missingness. Unknown
codes fail validation instead of silently becoming labels. Indexes preserve UUID,
field-season identity and physical-field grouping separately.

Argentina global attributes include 751 `<field_shared_name>_<>_yield_ground_truth` entries,
with mixed numeric/string values and 18 `UNKNOWN` values. Audit these attributes
separately for each country. They are scalar field metadata, separate from `target`; they must never enter predictor features,
pretraining losses, applicability evidence, normalization or split selection.

### Country-local time and feature metadata

All four files have the same set of 120 feature names; their stored order differs.
Build a name-to-channel mapping per file and reorder into a canonical registry
before batching countries together. A mapping from Argentina cannot index Uruguay
or any other file safely.

| Country | `times.units` | `times.calendar` | Year dictionary |
|---|---|---|---|
| Argentina | `days since 2021-10-28` | `proleptic_gregorian` | 2017–2024 |
| Brazil | `days since 2022-09-26` | `proleptic_gregorian` | 2017–2024 |
| Germany | `days since 2019-09-22` | `proleptic_gregorian` | 2016–2022 |
| Uruguay | `days since 2018-11-25` | `proleptic_gregorian` | 2018–2022 |

Never reuse another country's epoch or category dictionary. The detailed numeric
row examples below are Argentina observations, not verified values or conventions
for Brazil, Germany or Uruguay.

### Feature inventory: 120 named channels

| Group | Exact names / construction | Count |
|---|---|---|
| Optical | `B01,B02,B03,B04,B05,B06,B07,B08,B09,B11,B12,B8A` | 12 |
| Terrain | `aspect,curvature,dem,slope,twi` | 5 |
| Soil estimates | `cec,cfvo,clay,nitrogen,phh2o,sand,silt,soc` × depths `0-5,5-15,15-30,30-60,60-100,100-200` | 48 |
| Soil uncertainty | Same 48 soil channel names followed by `_uncertainty` | 48 |
| Weather | `temp_max,temp_mean,temp_min,total_prec` | 4 |
| Coordinate features | `coord_x,coord_y,coord_z` | 3 |

The actual band vector is not grouped in this table order: e.g. soil depths follow
string ordering and estimates/uncertainties are interleaved. Resolve all channels
by exact names and record a canonical property/depth order for tensors. Require
unique expected names and fail on unexpected schema changes.

## 3. Argentina row observations and unresolved corpus semantics

These are five Argentina row spot checks plus metadata, not whole-corpus or
other-country quality statistics:

- The first three rows belong to one soybean field-season, at `(row,col)`
  `(1,10)`, `(1,11)`, `(2,10)`. Their targets differ: approximately 1.4467, 1.2825,
  0.7017. Thus the observed labels are not simply one scalar repeated per field.
- The first row has dates only in slots 9–16: 2021-10-28, 2021-11-25,
  2021-12-27, 2022-01-21, 2022-02-23, 2022-03-27, 2022-04-16, 2022-04-29.
  Its recorded harvest is 2022-03-31: some stored observations are post-harvest.
  One valid date has no valid optical values, so date validity and source validity
  cannot be equated. Leading/trailing NaN times are common in the inspected rows.
- Static DEM/soil/coordinate values repeat through 24 slots in inspected rows,
  including slots with missing dates. Verify this across the corpus before collapsing
  static values; do not erase valid static information with a temporal padding mask.
- The first row's `slope` is about 1699.14. Its `temp_mean` includes 8583.55 and
  10099.28. These cannot be consumed as slope degrees or instantaneous temperature.
  Weather may have interval aggregation, but its operation and interval bounds have
  not been verified. A zero weather entry does not establish an actual zero value.
- `coord_x/y/z` are undocumented in the NetCDF. Do not assume latitude/longitude,
  metres, Earth-centred coordinates, invertibility or a correct hemispheric sign.
- NetCDF variables do not supply the physical units/scales/processing provenance
  needed for yield, soil estimates/uncertainties, optical values, weather or terrain.
  Channel names alone do not establish source product or scientific units.

### Optional raw provenance companion: evidence, not a second ingestion target

A bounded read of the existing neighboring archive
`/home1/pupil/SMATousi/YieldSAT/Raw/Raw.zip` inspected only the first sampled field
(`Argentina_DUP1_farm16_field156_soybean_2022`), without extraction or mutation:

- Its metadata JSON contains projected CRS EPSG:32720, centroid WGS84 coordinates,
  crop/season dates, provider and yield-map quality. Recover equivalent metadata
  for all included fields before claiming geographic evaluation.
- DEM derivatives and the sampled 12-band S2 raster use a 10 m, 74×137 grid with
  affine `(10,0,594140,0,-10,6551630)`. This supports the interpretation of stored
  rows as grid cells, but does not establish every field's grid or original source
  resolutions. The preprocessed corpus has already undergone spatial harmonization.
- `yield_masks/mean_scaled_yield_masked_regional_statistical_outlier.tif` has value
  1.4466667 at row 1, col 10, matching NetCDF row zero's target. It declares nodata
  **-1**, not the project's demo sentinel -9999. Companion number/std rasters exist.
- The field's weather CSV contains daily temperatures around 290 and precipitation
  decimals. Its transformation into NetCDF interval features still needs auditing.

No upstream preprocessing code or explanation of “scaled yield” was found in the
inspected preprocessed directory. Raw sample agreement verifies one pixel's storage
mapping, not target units, independence from field yield, or cleaning validity.
Recover processing provenance before training on a scientific target; in particular,
check whether field `yield_ground_truth` was used to produce scaled targets. If so,
document label construction and keep that value out of all model inputs. Do not
multiply/divide targets by ground truth or label them Mg/ha based on an assumption.

## 4. Proposed branch-specific model streams

These names deliberately preserve source/provenance differences from the US
full-layer registry. Register aliases only after source equivalence is verified.

| Stream | Input shape for point mode | Encoder / interpretation |
|---|---|---|
| `yieldsat_s2` | `(B,T,12)` | Temporal optical feature encoder; all stored bands are one sensor family. A raster/SAR encoder cannot consume these vectors unchanged. |
| `yieldsat_dem` | `(B,1)` | Static terrain-family elevation feature; unit/vintage unresolved. |
| `yieldsat_terrain` | `(B,4)` | Static aspect/curvature/slope/TWI; same family as DEM. Cyclic aspect only after units are confirmed. |
| `yieldsat_soil` | `(B,48)` | Depth/property-indexed static features. Encode uncertainty as an optional 48-feature ancillary input or reliability mask only after its definition is verified. |
| `yieldsat_weather` | `(B,T,4)` | Temporal interval features with per-feature/time validity and actual dates/interval support. Not the daily Daymet/ERA5-Land contract. |

`coord_*`, row/col, field identity and dates are metadata by default, not a sixth
scientific modality. IDs are for indexing/splits/provenance, not embeddings of farm
or field labels. Enable any verified physical positional encoding explicitly and
ablate it; unverified coordinate channels cannot drive geographic splits.
Crop/season metadata may be optional context only if available at the configured
prediction time. Actual harvesting date is not an automatic predictor feature.

For static groups, compare valid repetitions with declared tolerances, select one
verified representation and retain feature masks. Conflicting repetitions fail or
follow a recorded policy; never silently average away inconsistency. Weather may
repeat spatially, but its spatial support must be recovered rather than treating
all cells as independent station observations.

This branch cannot reconstruct original native grids from merged vectors. It is a
preprocessed/common-grid experiment, with results labelled accordingly. It does
not satisfy the native-grid requirement or LI-01–LI-15 full-layer compliance.
S1, SMAP, CDL, NAIP, PlanetScope and SSURGO/POLARIS are not identified in the
inspected feature inventories; do not
synthesize or infer them from the filenames or another contract's vocabulary.

## 5. Adapter, sampling and target contract

### YS-R01 — Lazy, versioned source access

Read bounded row ranges/windows from the HDF5-backed file; never evaluate
`sample[:]`, eager-load ~61 GB, or duplicate it per worker. Select a supported
backend during implementation; h5py inspection success does not select the training
backend. Declare any xarray/netCDF4/h5netcdf dependency if chosen. Open file handles
per worker, close them correctly and benchmark contiguous block sampling versus
random gathers. Optional indexed/sharded caches must be reproducible, bounded and
stored outside the source root; no source rewrite is required.

Construct the UUID/field-season/grid index in chunks, with compact on-disk metadata.
Validate duplicate UUIDs and duplicate `(field-season,row,col)` tuples. Explicitly
resolve duplicates before scatter or sampling; verify whether rows for a field are
contiguous rather than assuming ordering. Weight/sample by field/farm as appropriate
so millions of neighboring pixels do not dominate pretraining or reported metrics.

### YS-R02 — Point mode first, spatial reconstruction second

The initial selectable mode is `point_timeseries`: one grid-cell history and its
scalar target. Add a registered scalar yield head or a documented compatible
per-cell head and collation path; current dense heads do not automatically accept
these vector inputs. Scatter held-out predictions back to verified field row/col
grids for spatial evaluation. Preserve unobserved cells as invalid.

A later `field_patch` mode may group neighboring cells into a spatial window for
the existing dense heads. It requires verified grids, grouping, variable field sizes,
per-pixel times/masks and actual spatial encoder support. Do not reshape an arbitrary
batch of pixels into a square image. A point-mode experiment supplies within-field
cell predictions; it does not prove that spatial patch context has been learned.

### YS-R03 — Missingness, units and train-only normalization

Keep separate feature/time/source masks. NaNs, verified source sentinels and any
upstream padding/aggregation conventions need source-specific decoding before
normalization; zeros and uint category code zero are not generically missing.
An observed date need not mean that optical or weather is usable. Date-less temporal
values are ineligible until their time support is resolved; static groups can remain
usable. Reject target -1 if its equivalence to raw nodata is confirmed across the
snapshot, and always exclude nonfinite targets; do not rely only on -9999 masking.

Fit feature/target transformations on training groups only. Supplied `stats-*`
have unknown fitting populations and may include test fields: retain for auditing,
do not use for default normalization. Record conversion, aggregation, uncertainty
and target-unit policies per country. Choose pooled versus country-specific
training statistics explicitly; both exclude all held-out countries/groups. Never
pool incompatible physical units or target definitions merely because names match. Unit-unresolved groups are excluded from physical
knowledge constraints; supervised scientific training awaits a verified target.

### YS-R04 — Forecasting cutoff and availability

Set an explicit prediction cutoff/lead time. Decode `times` using its units and
calendar, and decode seeding/harvest dictionaries independently. Filter *all*
time-dependent sources consistently; weather aggregation intervals must end no
later than the cutoff. If an interval overlaps the cutoff, exclude it unless daily
provenance permits a valid recalculation. The first row demonstrates that “all 24
slots” can include information after harvest. Report any retrospective full-season
experiment separately from pre-harvest prediction. Harvest-date-based experimental
cutoffs do not authorize using the actual harvest date as an input available then.

### YS-R05 — Geographic and temporal leakage controls

Split by physical field/farm before sampling rows/patches. `field_shared_name`
includes crop/year, so derive and validate physical-field identity (country/provider/
farm/field) separately and keep all its seasons together. Check geometry-based
duplicates across aliases/farms when metadata is recovered. Require farm-held-out
baseline splits and geographic/block-separated splits for geographic generalization
claims. Without verified geography, label the result farm-held-out only.

Country-qualify local farm/field identifiers to prevent collisions. Support
single-country, pooled multi-country and leave-one-country-out experiments with
explicit manifests. Country holdouts are distinct from within-country farm/block
holdouts; report them separately. Calibration, normalization, knowledge-annotation
policy fitting and pretraining exclude held-out countries/groups in the main transfer
comparison. Report per-country/per-crop and macro-country metrics so Argentina's
row count does not dominate the pooled result. Audit cross-border geographic overlap
where appropriate.

Use training geography only for pretraining in the main transfer comparison; exclude
all rows/seasons of held-out physical fields. Report any transductive experiment
separately. No random pixel split may be the headline result. Loss/metrics use
valid targets; report per-crop metrics and both pixel-weighted and field-balanced
aggregates after units are verified.

### YS-R06 — Yield geometry and support

`target` is the default candidate per-cell label; global field-yield attributes and
raw count/std rasters are target-side diagnostics only. Audit scaling/cleaning,
measurement support, nodata and field-level consistency, including `UNKNOWN`
field attributes without automatically rejecting valid per-cell targets.
Do not repeat a field's scalar ground truth across cells as dense supervision.

Recover each field's CRS/affine/dimensions and valid spatial support from metadata
or matching raw assets; validate row/col orientation and off-by-one errors on
several fields. Row presence alone may reflect yield-data filtering, not a boundary
or full sensor footprint. Do not feed target/count/std-derived masks or statistics
into sensor inputs. Report that preprocessed row selection may already be label-
informed; it is not an independent unlabelled AOI corpus. Missing target coverage
cannot be represented as a valid zero-yield cell. Georeferenced outputs require a
validated transform, not a guessed inverse of `coord_*`.

### YS-R07 — Existing objectives and knowledge compatibility

Reuse existing losses only after shape/mask/support adapters are verified. Spatial
masked reconstruction requires spatial neighborhoods; for point mode use an
explicit temporal/feature masking variant and label its changed semantics. Forecast
the final *valid* eligible observation from earlier observations, not slot 23 by
position; audit aggregated interval targets and cross-source future leakage.
Use source-family-aware cross-modal/contrastive objectives: terrain derivatives and
soil uncertainties are not independent sensors, nor are separate S2 bands. Avoid
same/nearby-field negatives that are actually redundant observations.

[Relationship pretraining v2](./knowledge_pretraining.md) requires a separate
country-specific concept/tag mapping and reviewed evidence policies. Do not apply the
US layer-generator prompt verbatim. Confirm units, time/support, soil uncertainty
meaning and terrain definitions before teacher evidence or relationship scorers.
Model annotations remain offline, uncertain and yield-free. No provider calls are
part of this documentation task.

## 6. Implementation tasks and acceptance

All tasks remain open. These are a parallel Phase 3 adapter path with Phase 4
transfer/knowledge evaluation; completing it is not completion of the downloader
full-layer backlog.

| Task | Phase / dependencies | Deliverable and acceptance |
|---|---|---|
| **YS-01: Freeze and audit source snapshot** | Phase 3, first | Extraction/integrity evidence per selected country, shared schema and complete row/code coverage audit; read-only source access and stale-cache rejection. |
| **YS-02: Recover semantics and geometry** | Phase 3; YS-01 | Versioned country-specific provenance for optical/terrain/soil/weather/coordinate units, temporal aggregation, target scaling/cleaning and per-field CRS/affine. Verify several raw/preprocessed cell correspondences. Record unresolved items and prohibit unsupported physical claims. |
| **YS-03: Build index and grouped splits** | Phase 3; YS-01/02 | Chunked UUID/field/row/col index, duplicate policy, country-qualified physical-field/farm identities, single/pooled/country-holdout train/validation/test manifests and overlap checks; geographic split if verified geometry permits. |
| **YS-04: Implement lazy adapter and stream registry** | Phase 3; YS-01/02 | Per-country named-channel reordering, local categorical/date decoding, verified static collapse and configurable uncertainty handling. Worker-safe bounded reads and relocatable artifacts. |
| **YS-05: Masks, cutoff and normalization** | Phase 3; YS-03/04 | Independent source/feature/time/target masks, interval-aware cutoff and train-only statistics. Check valid code zero, NaNs, all-missing rows, post-harvest slots, sentinels and no test-statistics leakage. |
| **YS-06: Point encoders and yield head** | Phase 3; YS-04/05 | Vector temporal/static encoders, shared fusion/collation, scalar target head and explicit checkpoint compatibility. Real forward/backward works for missing sources and variable valid histories. |
| **YS-07: Objective and knowledge routing audit** | Phase 3 for existing objectives; Phase 4 for knowledge; YS-03/05/06 | Record which objectives are active in point/patch modes, adapt masking/forecast eligibility and prevent source-family/temporal leakage. country-reviewed concepts/teacher policies integrate with KP-01–KP-08 only after semantics are verified. |
| **YS-08: Real entry points and bounded pilot** | Phase 3; YS-03/05/06/07 existing-objective audit | Explicit contract selector, documented point-mode CPU/GPU pilot as available, measured I/O/memory, source exclusions and sensor-only checkpoint transfer. Report honest held-out per-country/per-crop/field-balanced metrics in verified units. |
| **YS-09: Spatial reconstruction / patch mode** | Phase 3 follow-up; YS-02/03/06/08 | Scatter point outputs to verified field grids; optionally implement spatial patch datasets/dense heads. Test sparse holes, bounds, orientation, duplicate cells and georeferencing; no invented dense labels. |
| **YS-10: Transfer and relationship evaluation** | Phase 4; YS-08 and relevant YS-07/KP tasks | Scratch/pretrained comparisons and knowledge controls on matching grouped splits/budgets; report point versus patch mode, annotation cost, excluded sources and absence of benefit if applicable. |

Initial pilot exit: verified target units/processing, accepted source snapshot,
grouped splits, bounded real-data training/transfer and correctly labelled evaluation.
Full branch acceptance additionally requires validated geometry and spatial output
reconstruction. Native-grid full-layer acceptance remains governed independently
by LI tasks. Open choices include backend/cache layout, prediction horizon, valid
weather interval interpretation, uncertainty use and whether patch mode is pursued.
