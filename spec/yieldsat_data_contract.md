# Alternative Data Contract — YieldSAT Preprocessed

**Status:** Inspected source snapshot and implementation requirements, 2026-09-28.
**Implementation update (2026-09-28, later the same day):** the point-mode pipeline
is implemented and has run on real data for all four countries. That covers
snapshot audit, semantics and geometry recovery, index and grouped splits,
cache/HDF5 adapters, masks, cutoffs, train-only normalization, one encoder per
stream, the scalar head, objective routing, the contract-selected entry point,
spatial reconstruction and a first transfer comparison. §7 is the step-by-step
progress log, with evidence and remaining limitations; §1–§6 keep the original
requirements. This is a second selectable data-contract branch, not a Git
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

Task status as of 2026-09-28 is in §7. These are a parallel Phase 3 adapter path with Phase 4
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

## 7. Implementation progress log (2026-09-28)

Everything below was executed in this repository's `geo-yield-phase0` environment
(Python 3.11, PyTorch 2.5.1, h5py 3.16, RTX 3090). Source files were opened
read-only. All artifacts live under `YIELDSAT_ARTIFACT_ROOT`
(`/root/yieldsat_artifacts` on this host, local ext4, outside the source root).
Source code:

| Module | Role |
|---|---|
| `dataset/yieldsat_schema.py` | Canonical 120-channel registry, stream definitions, per-file name→channel resolution, dictionary/date decoding, provenance table |
| `dataset/yieldsat_source.py` | Snapshot checks (signature, size, stability, CRC32), chunked row index, stale-artifact rejection |
| `dataset/yieldsat_geometry.py` | Read-only `Raw.zip` metadata/grid extraction, physical-field grouping, raw↔NetCDF cell verification |
| `dataset/yieldsat_cache.py` | Shared canonicalizer, one-pass audit, compact cache, per-field sufficient statistics |
| `dataset/yieldsat_splits.py` | Grouped split manifests (farm cluster, physical field, geographic block, country, year) with leakage checks |
| `dataset/yieldsat_dataset.py` | Point-mode dataset (cache and HDF5 backends), masks, cutoffs, train-only normalizer, field-balanced block sampler |
| `models_multimodal_encoder.py` | New encoder types `masked_temporal`, `masked_static`, `soil_profile` |
| `models_heads.py` | Registered `scalar_cell_yield` head |
| `models_yieldsat.py` | `YieldSATPointModel`: one encoder per stream → Perceiver fusion → scalar head; sensor-checkpoint export/import with compatibility checks |
| `yieldsat_objectives.py` | Objective routing table and point-mode pretext objectives |
| `util/yieldsat_eval.py` | Pixel, field-balanced and field-level metrics per country/crop plus macro; grid scatter and GeoTIFF writing |
| `yieldsat_prepare.py` | CLI: `snapshot`, `index`, `geometry`, `cache`, `semantics`, `splits` |
| `main_yieldsat_finetune.py` | Entry point with `--data_contract` selector, fine-tune/pretrain modes, reports |
| `tests/test_yieldsat.py` | 16 contract tests on a synthetic file with the real NetCDF layout |

Commands and flags are in [RUNNING.md](../RUNNING.md).

### YS-01 — Source snapshot frozen and audited ✅

- **Integrity.** `Preprocessed.zip` was removed from the source directory during
  this work (about 22:48); the extracted `.nc` files were untouched. The archive
  CRC32s recorded in §1 served as the reference. A whole-file streaming CRC32 of
  each `.nc` matched the record for all four countries: Germany `b89c8acb`,
  Uruguay `41592d6f`, Brazil `1c400537`, Argentina `bae1a7f0`
  (`audit/crc32_<country>.json`). Sizes matched too, and mtimes were identical
  before and after each read. `yieldsat_prepare.py snapshot` re-verifies HDF5
  signature, recorded size, stability over a 2 s interval, schema (rows × 24 ×
  120, all channels resolvable), last-row readability and CRC evidence for the
  same size/mtime. It fails on any mismatch.
- **Stale-artifact rejection.** Every index, cache, split and statistics file
  records a fingerprint: size, mtime_ns and SHA-256 of the first and last 4 MiB.
  Loaders raise `SnapshotError` when the fingerprint changes, and a cache without
  a completed manifest is ignored. Both are covered by tests.
- **Index audit** (`index/<country>/index_manifest.json`, all four countries):
  - UUIDs unique; every field season's rows contiguous.
  - **0 duplicate `(field, row, col)` cells.** An earlier probe that reported
    duplicates used an overflowing key and was wrong.
  - Every categorical code decodes; field/farm/crop/year coverage equals the
    dictionaries (751/57, 551/9, 299/6, 572/10 field seasons/farms).
  - Row/col dictionaries are identity maps except in **Brazil**, where codes must
    be decoded (rows up to 645 from 421 codes).
  - No target is NaN, -1 or ≤ 0.
  - Each field season has a single seeding and harvest date.
- **Full-corpus value audit.** One sequential pass per file, 5–33 min, 12.37 M
  rows (`cache/<country>/cache_manifest.json`):

| | Argentina | Brazil | Germany | Uruguay |
|---|---|---|---|---|
| Rows | 5,325,807 | 4,260,262 | 609,645 | 2,177,206 |
| Static repetition conflicts | 0 | 0 | 0 | 0 |
| Rows with TWI missing (only static gap) | 3,601,625 (68%) | 2,560,352 (60%) | 609,397 (99.96%) | 1,704,630 (78%) |
| Dated slots per row (mean) | 7.5 | 7.3 | 11.5 | 6.5 |
| Dated slots with all optical NaN | 9.3% | 10.5% | 42.0% | 6.4% |
| Dated slots with weather NaN / undated values | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Rows with first-dated-slot weather exactly 0 | 2,778,382 (52%) | 3,878,180 (91%) | 0 | 0 |
| Optical zeros / negatives / ±inf | 4,345 / 0 / 0 | 740 / 0 / 0 | 106 / 0 / 0 | 1,369 / 0 / 0 |

Dated slots occupy positions 7–23 (AR/BR), 7–20 (DE) and 10–17 (UY). Slots 0–6
are never dated, and "24 time steps" is not a common calendar.

### YS-02 — Semantics and geometry recovered ✅ (units partly inferred)

Sources:
- the YieldSAT project page and release notebooks (<https://yieldsat.github.io/>,
  CVPR 2026);
- `yieldsat_prepare.py semantics` on 200k cached rows per country
  (`audit/semantics_<country>.json`);
- `geometry` over the untouched `Raw.zip`.

The status column below matches `PROVENANCE` in `dataset/yieldsat_schema.py`.
Only *documented* entries may feed physical constraints.

| Group | Finding | Status |
|---|---|---|
| Target | Combine-harvester yield converted to standard-moisture dry yield, t/ha. Zero, crop-infeasible and ±3σ cells were removed upstream. Release notebooks label `target` t/ha. Per-cell medians fit this: AR corn 8.67, soybean 3.56, wheat 4.18; BR corn 6.69, soybean 4.13, wheat 3.55; DE rapeseed 3.88, wheat 9.67; UY soybean 2.32 t/ha. Some maxima exceed agronomic limits (AR corn 38.4 t/ha, UY soybean 15.0 t/ha). Upstream caps are set per provider (metadata `max_yield_per_hectare`) and are not re-applied here. | documented |
| Target ↔ raw | 20 randomly chosen field seasons (5 per country) were checked against `yield_masks/mean_scaled_yield_masked_regional_statistical_outlier.tif`. Every NetCDF cell equals the raster at `(row, col)` (100%). The transposed control matches 0.3–3.6%. No row is out of bounds, and every valid raster cell has exactly one row. Raw nodata is **-1**; no NetCDF target is -1, NaN or ≤ 0. | verified |
| Label construction | `yield_ground_truth` attributes are scalar field metadata (Uruguay: all 572 `UNKNOWN`; Argentina: 18). Per-cell targets vary within fields and are not a repeated scalar. Whether the field value entered the cell scaling is not stated upstream. It is kept in `target_side.json` and never loaded by the adapter. | partly open |
| Optical | Sentinel-2 L2A. The least-cloudy acquisition per slot; 20/60 m bands nearest-upsampled. Values are L2A DN (B04 median ≈ 700, B08 ≈ 3400). No negatives; zeros are rare (106 in DE, 1,369 in UY) and kept as values. SCL is not in the NetCDF. 42% (DE) and 6% (UY) of dated slots have all optical NaN (cloud-masked). | inferred |
| Weather | ERA5-Land daily, interpolated to the field centroid. Each slot holds the **sum of daily values over the inclusive interval `[t_(k-1), t_k]`**. `value/(dt+1)` is flat across interval lengths (median K per day for 5–50-day gaps: AR 290–295, BR 293–295, DE 278–292, UY 290–297), while `value/dt` drifts from about 350 K at 5 days to about 284–301 K at 50 days. So boundary days count in both neighbouring slots, temperatures are Kelvin-day sums and precipitation is metres. `temp_max ≥ temp_mean ≥ temp_min` holds on 100% of dated slots. Weather is constant within a field (max per-slot std ≤ 0.5 on sums of ~8,000). The first dated slot's interval start is unrecorded: AR/BR store exactly 0 there in 52%/91% of rows, and DE/UY store a sum over an unknown span. **Weather at each row's first dated slot is therefore treated as invalid** (`temporal_valid_mask`; per-field statistics v2 apply the same rule). Weather is available at `t_k`, so a cutoff at or after `t_k` is causal. | inferred |
| DEM / terrain | SRTM 30 m, cubic-upsampled; RichDEM derivatives. DEM looks like metres. Aspect spans 0–360 (degrees; cyclic encoding is optional, `--aspect_encoding cyclic`). Slope (median ≈ 2,700 in DE) is not degrees or percent; its scale is unresolved and it is only standardized. **TWI is NaN for 99.96% of Germany and 78% of Uruguay rows** and is masked per feature. | slope/TWI unresolved |
| Soil | SoilGrids 2.0 at 250 m, cubic-upsampled. Values sit in SoilGrids mapped units (clay 100–270 g/kg, phh2o 53–67 = pH×10, soc dg/kg). Uncertainty layers have a heavy tail (p99 3,737 for clay 0–5 cm in DE) and an unverified definition, so they are excluded by default (`--soil_uncertainty ancillary` enables them as a separate input group). | inferred / unresolved |
| Coordinates | `coord_x/y/z` are unit-norm 3-vectors whose spread inside one country is inconsistent with WGS84 geocentric positions. Unresolved and never used as input or for splits. | unresolved |
| Geometry | All 2,173 field seasons have metadata JSON and a DEM grid in `Raw.zip`: EPSG (UTM), 10 m affine, width/height, WGS84 centroid, provider, quality. `geometry/fields_geometry_*.json` has one record per season. | verified |
| Physical identity | Field numbers are unique per **season**, so names cannot link seasons of the same ground. Footprint union-find at bbox IoU ≥ 0.3 gives **1,059 physical fields** from 2,173 seasons (up to 91 seasons chained in one Argentine group). **146 overlaps cross farm IDs** (139 AR, 7 BR), e.g. `farm10_field105` ≡ `farm15_field135` at IoU 0.98. Some are same-year aliases (`farm19_field205_wheat_2021` ≡ `farm50_field666_wheat_2021`, IoU 0.99), which are possibly duplicated seasons. Farm clusters: AR 57 farm IDs → 41, BR 9 → 8, DE 6, UY 10. | verified |

### YS-03 — Index and grouped splits ✅

`yieldsat_prepare.py index` builds, per country: `rows.npz` (decoded field code,
grid row/col, target, seeding/harvest day), `uuids.npy`, `fields.json` (contiguous
row ranges, country-qualified `farm_id`/`season_id`) and `index_manifest.json`. It
takes 1.5–11.6 s per country. `yieldsat_splits.py` adds geometry-derived
`physical_field_id` and **farm clusters** (farms linked through a shared physical
field form one group). Each manifest is checked so that no physical field, and for
farm/country schemes no farm cluster, appears in two partitions. Otherwise
`check_split` raises. Manifests carry the source fingerprints, and loading one
against another snapshot fails.

Groups are assigned by largest-deficit greedy on row fractions. The first
implementation (sequential fill) overshot validation with coarse farm clusters, so
it was replaced before any reported run. Seed-0 manifests
(`artifact_root/splits/`, rows as fractions):

| Manifest | Label | Train | Val | Test | Excluded |
|---|---|---|---|---|---|
| `germany_farm_s0` | farm-held-out | 220 seasons / 0.59 | 43 / 0.18 | 36 / 0.23 | – |
| `pooled_farm_s0` | farm-held-out | 1,472 / 0.64 | 276 / 0.14 | 425 / 0.23 | – |
| `pooled_field_s0` | physical-field-held-out | 1,177 / 0.65 | 452 / 0.15 | 544 / 0.20 | – |
| `pooled_block20_s0` | geographic-block-held-out (20 km) | 1,412 / 0.65 | 315 / 0.12 | 446 / 0.24 | – |
| `loco_uruguay_s0` | leave-one-country-out (Uruguay) | 1,408 / 0.70 | 193 / 0.12 | 572 / 0.18 | – |
| `loyo_2022_s0` | leave-one-year-out (2022) | 1,015 / 0.39 | 179 / 0.07 | 333 / 0.16 | 646 / 0.37 |

LOYO removes all other-year seasons of any physical field seen in the test year,
so 37% of rows are excluded. That is the cost of honest temporal holdout on
repeatedly cropped ground. Brazil's pooled test partition is a single farm
cluster (190 seasons), so Brazil test numbers describe one farm. No random pixel
split exists in the tooling.

### YS-04 — Lazy adapter and stream registry ✅

- **Canonicalization** (`canonicalize_block`) resolves the 120 channels by exact
  name in each file. It produces temporal `(N,24,16)` [12 S2 + 4 weather] and
  static `(N,104)` [dem, 4 terrain, 48 soil, 48 soil uncertainty, 3 coords], plus
  times as days since 1970 using each file's own epoch. Static channels keep the
  first finite slot. Repetitions differing beyond `1e-6 + 1e-5·|v|` set a
  per-row conflict flag and are never averaged.
- **Backends.** The `cache` backend uses local `.npy` memmaps: ≈2 KB/row, built
  once and stamped with the source fingerprint. Loading refuses a partial cache or
  a changed source. The `h5` backend reads the NetCDF directly: file handle per
  worker, sorted rows grouped into contiguous slices of at most 8,192, and never
  `sample[:]`. Both produce identical tensors, checked on real Germany rows and in
  tests.
- **Measured I/O on this host (NFS source, NVMe cache).** Contiguous HDF5 reads
  run at 122 MB/s. Random single-row reads manage 678 rows/s. The h5 backend takes
  0.19 s per 256-row batch with 64-row blocks; the cache takes about 3 ms. The
  single cache/audit pass reads each file once, at about 120 MB/s aggregate for
  four parallel processes.
- **Streams and decoding.** Streams are `yieldsat_s2 (T,12)`,
  `yieldsat_weather (T,4)`, `yieldsat_dem (1)`, `yieldsat_terrain (4,
  or 5 with cyclic aspect)` and `yieldsat_soil (48, or 96 with uncertainty)`.
  Coordinates and IDs are never inputs. Crop is an optional context token (known
  at planting; `--no_crop_context` ablates it); harvest date is used only to place
  the cutoff.

### YS-05 — Masks, cutoff and normalization ✅

- **Masks.** Each item carries a value mask per stream, a `time_valid` slot mask
  and a target mask, all independent. A dated slot with NaN optical is invalid for
  optical only. Static features stay valid whatever the time mask says. Code 0 is
  a category, never missingness. Invalid values are set to 0 *after*
  standardization and always travel with their mask.
- **Cutoff.** Modes are `before_harvest` (default, 30 days), `after_seeding`,
  `harvest` and `all_slots`; the last two are labelled *retrospective* in
  reports. The rule `t_k ≤ cutoff` applies to every temporal stream. Given the
  inclusive weather intervals it is causal for weather, and the post-harvest slots
  seen in Argentina probes are excluded by all non-`all_slots` modes. Time
  features are days since seeding / 365 and day-of-year sin/cos, zeroed for
  ineligible slots.
- **Normalization.** Mean/std come from per-field sums accumulated in the audit
  pass, summed over *training* seasons only (pooled by default, `per_country`
  optional; a held-out country then has no statistics and is refused). The target
  is standardized the same way; metrics are computed after inverting to t/ha. The
  supplied `stats-*` are never read.

### YS-06 — Point encoders and yield head ✅

The exact layer-by-layer architecture, shapes and parameter counts are in
[architecture.md §9](./architecture.md#9-yieldsat-point-model--exact-architecture).

One dedicated encoder per stream, all built through `MultiModalEncoder`, which
keeps the learned missing-modality token, per-sample availability and modality
dropout:

| Stream | Encoder type | Design |
|---|---|---|
| `yieldsat_s2` | `masked_temporal` | Per slot `[values·mask, mask]` + time-feature projection + slot embedding. Transformer over slots with invalid slots removed from attention; CLS token → 1 token. |
| `yieldsat_weather` | `masked_temporal` | Same encoder class, separate weights. |
| `yieldsat_dem` | `masked_static` | MLP on `[value·mask, mask]`. |
| `yieldsat_terrain` | `masked_static` | MLP; aspect optionally sin/cos. |
| `yieldsat_soil` | `soil_profile` | One token per depth (8 properties + masks, optional uncertainties), depth embedding, transformer over the profile, CLS → 1 token. |

The five tokens (plus an optional crop token) go to `LatentFusionTransformer`
(8 latents, depth 2) and the registered `scalar_cell_yield` head (mean-pooled
latents → MLP → one value per cell). Defaults total 1.13 M parameters and use
0.39 GB GPU at batch 512.

`sensor_state_dict()` exports encoders + fusion only (no head, no crop embedding)
with a descriptor: contract, point mode, per-stream channel lists, dims.
`load_sensor_state_dict()` refuses a different contract/mode, embedding size or
channel list. Tests cover forward/backward with whole streams missing, empty
weather histories, modality dropout 0.5, soil uncertainty and cyclic aspect.

### YS-07 — Objective routing and leakage audit ✅ (knowledge part blocked)

`OBJECTIVE_ROUTING` in `yieldsat_objectives.py`, point mode:

| Objective | Point mode |
|---|---|
| supervised yield | active |
| masked spatial reconstruction | **inactive**: no spatial neighbourhood; `require_objective` raises |
| masked observation | active, *changed semantics*: hides 30% of observed values and reconstructs them in normalized input space; not MAE |
| temporal forecast | adapted to `forecast_last_valid`: predicts optical at each cell's last valid eligible slot; **all** temporal streams hidden from that slot on (no weather leakage); never slot 23 by position |
| masked modality / cross-modal | restricted to whole families (DEM + terrain are one family) |
| contrastive | restricted: family-level views; `contrastive_negative_mask` excludes same-physical-field negatives |
| knowledge relationships | **blocked**: needs the country-reviewed concept mapping of KP-01–KP-08 and verified slope/TWI/soil-uncertainty semantics; no annotations or provider calls were made |

Patch mode is not implemented, so every objective is marked *not implemented*
there. `YieldSATPointPretrainer` combines masked observation and last-valid
forecast. It is yield-free and never sees the crop token.

### YS-08 — Contract selector and bounded real-data pilots ✅

`main_yieldsat_finetune.py --data_contract yieldsat_preprocessed_v1` is the entry
point. Selecting `downloader_native_layers_v1` exits with "not implemented", which
is tested. Every run writes `report.json` with:
- split label and hash, cutoff/label policy (pre-harvest vs retrospective);
- the stream channel lists and excluded sources (coordinates, IDs, harvest date
  as input, supplied `stats-*`, ground-truth attributes, soil uncertainty unless
  enabled);
- the objective routing and provenance table;
- I/O rates, peak RSS/GPU memory, training history and best validation epoch;
- test metrics.

Runs are single-seed (seed 0) with the defaults in [RUNNING.md](../RUNNING.md):
cache backend, `before_harvest` 30-day cutoff, pooled train-only normalization,
crop context on, batch 512, 20×500 steps (Germany: 15×300), and field-balanced
sampling α = 0.5. The best epoch by validation pixel RMSE is evaluated on **every
held-out cell**. Metrics are in t/ha; R² is the coefficient of determination.
Commands are in `artifact_root/runs/run_all.sh`.

| Run (split label) | Test cells / seasons | Pixel RMSE | Pixel R² | Field-balanced RMSE | Field-level RMSE / R² | Bias | Macro-country pixel RMSE / R² |
|---|---|---|---|---|---|---|---|
| Germany only (farm-held-out) | 137,917 / 36 | 1.81 | 0.40 | 1.71 | 1.74 / 0.40 | +1.27 | – |
| Pooled 4 countries (farm-held-out) | 2,811,333 / 425 | 1.65 | 0.66 | 1.49 | 1.08 / 0.79 | +0.05 | 1.64 / 0.48 |
| Pooled (geographic-block-held-out, 20 km) | 2,913,259 / 446 | 1.45 | 0.66 | 1.33 | 0.91 / 0.81 | −0.08 | 1.52 / 0.32 |
| Leave-one-country-out → Uruguay | 2,177,206 / 572 | 1.51 | 0.08 | 1.49 | 1.05 / 0.04 | +0.74 | – |

Per country × crop, pooled farm-held-out (pixel RMSE / R², field-level RMSE / R²,
test seasons):

| | Pixel | Field level | n |
|---|---|---|---|
| Argentina corn | 2.63 / 0.65 | 1.81 / 0.72 | 21 |
| Argentina soybean | 0.99 / 0.56 | 0.71 / 0.72 | 55 |
| Argentina wheat | 2.12 / −0.14 | 2.10 / −0.42 | 23 |
| Brazil corn | 2.58 / 0.33 | 1.51 / 0.47 | 54 |
| Brazil soybean | 1.08 / 0.20 | 0.47 / 0.38 | 91 |
| Brazil wheat | 1.84 / −0.06 | 0.94 / −0.15 | 45 |
| Germany rapeseed | 1.69 / 0.08 | 0.97 / 0.23 | 30 |
| Germany wheat | 2.83 / −0.33 | 2.24 / −0.44 | 10 |
| Uruguay soybean | 1.20 / 0.32 | 0.65 / 0.67 | 96 |

Reading the numbers:
- Much of the pooled R² comes from separating crops and countries: corn ≈ 8 t/ha
  against soybean ≈ 3 t/ha. Within crop, skill ranges from useful (Argentine
  corn/soybean, Uruguay soybean at field level) to none (wheat everywhere,
  German crops).
- Germany alone has only 2 farm clusters for training and overfits after one
  epoch.
- Leaving Uruguay out gives nearly no within-country skill and a +0.74 t/ha bias,
  so cross-country transfer is not solved.
- Brazil's farm-held-out test set is a single farm cluster.
- These are single-seed pilot numbers, not tuned benchmarks. They are not directly
  comparable with the YieldSAT paper's CV10/LOYO/LORO protocols, which use other
  splits and may use all 24 slots.

Resources on this host:
- training throughput is 12.6–17.6 k samples/s per run with three runs sharing
  the GPU and 6 loader workers;
- data-loader wait is 3–4% of training time;
- test inference runs at 82–123 k cells/s;
- peak RSS is 2.35 GB and peak GPU memory 0.39 GB (0.58 GB when pretraining);
- a pooled 20-epoch run takes about 7 minutes wall time.

### YS-09 — Spatial reconstruction ✅ (point mode); patch mode not pursued

`util/yieldsat_eval.py` scatters held-out point predictions onto each field's
verified raster grid. Unobserved cells stay NaN; out-of-bounds or duplicate cells
raise (tested). A GeoTIFF is written only with the verified EPSG and affine from
`Raw.zip`; without geometry the tool refuses. Every run writes
`test_predictions.npz` (season, row, col, prediction, target) and three test
fields as GeoTIFF.

Check on `Argentina_DUP1_farm16_field156_soybean_2022` (pooled run):
- CRS is EPSG:32720 and the affine is identical to the raw grid (74×137);
- the prediction covers exactly the raw yield mask's 4,801 valid cells;
- within-field correlation with the observed map is 0.72 (mean 3.97 predicted vs
  4.36 t/ha observed).

This establishes the output geometry. It does not show that spatial context was
learned, because each cell is predicted independently. `field_patch` mode
(neighbourhood windows for the dense heads) is not implemented. Its routing
entries say so, and none of the results above are patch-mode results.

### YS-10 — Transfer evaluation ◐ (first comparison done; knowledge blocked)

Design, single seed:
- **Pretraining.** Yield-free `YieldSATPointPretrainer` (masked observation +
  last-valid forecast, no crop token) on the pooled farm-held-out **training**
  partition of all four countries: 1,472 seasons, no validation or test farms,
  20×500 steps. Loss fell from 1.12 to 0.17.
- **Fine-tuning.** Uruguay's training farms of the same manifest at 10% (42
  seasons) and 100% (417 seasons) label budgets, from scratch or from the sensor
  checkpoint (156 encoder/fusion tensors loaded after the compatibility check).
  Budgets, schedule and evaluation cells are identical: all 436,708 cells of the
  96 held-out Uruguay seasons.

| Uruguay, farm-held-out | Pixel RMSE / R² | Field-balanced RMSE | Field-level RMSE / R² | Bias |
|---|---|---|---|---|
| Scratch, 10% labels | 1.21 / 0.31 | 1.23 | 0.73 / 0.58 | +0.20 |
| Pretrained, 10% labels | 1.22 / 0.29 | 1.21 | 0.76 / 0.54 | +0.12 |
| Scratch, 100% labels | 1.20 / 0.32 | 1.21 | 0.71 / 0.61 | +0.36 |
| Pretrained, 100% labels | 1.24 / 0.27 | 1.22 | 0.71 / 0.60 | +0.34 |
| *Pooled 4-country supervised model (reference)* | 1.20 / 0.32 | 1.16 | 0.65 / 0.67 | – |

**Result: no benefit from this pretraining, at either budget.** Differences are
within what one seed can resolve. Uruguay's validation partition is a single farm
cluster, which makes early-stopping selection noisy (best epochs 0–6). Supervised
pooling across countries helps field-level error more than self-supervised
pretraining does.

Still open:
- multiple seeds and splits (block, LOYO);
- other pretext weights and longer schedules;
- patch-mode comparisons;
- knowledge controls, blocked by the KP-01–KP-08 prerequisites recorded under
  YS-07.

No annotation cost was incurred, because no annotations exist.

### Remaining limitations and open items

0. **Paper comparison.** Results are not yet comparable with the YieldSAT
   paper's per-crop CV10/LORO/LOYO tables. The plan is in
   [yieldsat_paper_comparison.md](./yieldsat_paper_comparison.md).

1. **Unresolved semantics.**
   - The slope scale and TWI definition are unknown (TWI is mostly missing).
   - The soil-uncertainty definition is unknown, and the stream is off by default.
   - `coord_*` meaning is unknown, and the field is never used.
   - Whether field ground truth entered upstream cell scaling is not stated.
   - Upstream crop caps were not re-applied: extreme targets such as AR corn
     38 t/ha remain in the data.
2. **Duplicate seasons.** Same-year aliased field seasons (e.g.
   `farm19_field205` ≡ `farm50_field666`, wheat 2021) are grouped for splitting
   but not deduplicated. A duplicate-season policy is still needed before
   benchmark claims.
3. **Brazil evaluation.** Brazil's pooled farm-held-out test set is one farm
   cluster. Report Brazil per-farm results with that caveat, or use block splits.
4. **Scope of the numbers.** All results are single-seed pilots with untuned
   hyperparameters. Only a 30-day pre-harvest cutoff was evaluated.
5. **Patch mode.** `field_patch` (spatial context for the dense heads) is not
   implemented.
6. **Knowledge objectives.** Blocked; see YS-07.
7. **Storage.** The cache needs ~25 GB of local disk. Without it, the `h5`
   backend works but is about 60× slower per batch over NFS.
