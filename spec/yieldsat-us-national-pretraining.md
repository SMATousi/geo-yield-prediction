# Nationwide US cropland pretraining corpus

**Status:** Design draft, 2026-10-04. This document specifies a future collection;
it does not launch downloads, reserve infrastructure, or modify training code.
Confirmed decisions and proposed defaults are distinguished below. Outstanding
user choices must be incorporated before the acquisition plan is finalized.

## Objective and confirmed scope

Build an unlabeled, multimodal corpus spanning **2021–2025**, with exactly
**20,000,000 accepted location-year samples per year**, **100,000,000 total**,
covering cropland across the **contiguous 48 US states (CONUS)**. Use a **hybrid
of shared locations and new annual locations**, as confirmed by the user.

A sample is one geographically identified 10 m grid location and its observations
for one growing-season/year window. Twenty-four timestamps at that location count as **one**
sample, not 24. A repeated location in another year is another location-year
sample but not a new geographical location. Report both totals. No measured
yield, artificial zero targets, or inferred farm boundaries are required.

The purpose is to increase diversity of climate, annual weather, crop systems,
soil and terrain, rather than inflate sample count through adjacent points with
identical coarse inputs. The corpus must support the existing sensor encoders
and relational knowledge-distillation workflow. Its acquisition and storage
contract is separate from supervised fine-tuning and future loader adapters.

The local unlabeled point build at
`/home1/pupil/SMATousi/YieldSAT-US-Pretraining` provides a processing pilot,
not a nationally representative sampling frame.
Its 2025 Missouri footprint must not consume a disproportionate national quota.
It may contribute qualifying records after national selection and validation;
locally retained points are not automatically included in the 100M budget.

## Decisions

| ID | Decision | State |
|---|---|---|
| D01 | 2021–2025; 20M location-years/year; 100M total | Confirmed |
| D02 | Contiguous 48 states; Alaska, Hawaii and territories excluded | Confirmed |
| D03 | Shared-location cohort plus new annual locations | Confirmed |
| D04 | Hybrid fraction | Proposed: nominal 10M shared + annual replenishment to 20M |
| D05 | Balanced climate-region and crop-type sampling, with weights | Confirmed |
| D06 | Points, not stored images; same 12 S2 bands as YieldSAT | Confirmed |
| D07 | 24 growing-season slots with aligned weather; versioned crop- and region-specific calendars; harvest-year sample identity | Confirmed |
| D08 | Exclude evaluation regions across all years | Confirmed; region list/boundaries pending |
| D09 | Storage, acquisition budget, compute environment and deadline | Intentionally blank, at user request |

Additional proposed defaults, open to revision: annual CDL crop eligibility
including hay/alfalfa, tree crops and fallow; a 10 m metric grid; a common mandatory
sensor package with optional enrichment; partitioned storage with shared coarse-grid
and static inputs. These are design proposals, not user-approved numerical
thresholds or service commitments.

The requested resource fields are deliberately unfilled:

| Resource | Value |
|---|---|
| Storage budget/location | |
| Acquisition/transfer budget | |
| Compute environment | |
| Completion deadline | |

## Sampling frame and location identity

### Cropland eligibility

For each year, use that year's pinned USDA CDL product, actual native resolution,
class dictionary and processing version. Include the agreed crop classes; the
local policy includes codes 1–61, 66–80 and 200–255, excludes grass/pasture (176),
and requires explicit validation of each code actually present. Do not silently
include unused/unknown codes merely because they fall in a numerical range.
The annual observed-code whitelist is part of the release manifest.

USDA provides historical national CDL coverage and its 2025 product is at 10 m;
older products may have different resolution. Read each selected asset's metadata
and do not reinterpret an upsampled class as independent fine-scale evidence.
[USDA releases](https://www.nass.usda.gov/Research_and_Science/Cropland/Release/),
[2025 metadata](https://www.nass.usda.gov/Research_and_Science/Cropland/metadata/metadata_CDL25_FGDC-STD-001-1998.htm).

Cropland masking is explicitly retrospective annual land-cover selection. It is
not measured crop ground truth or a guarantee of eligibility at an earlier
forecast date. Store CDL as selection/provenance metadata by default; expose it
as a predictor only through an explicit downstream policy. Do not let a knowledge
teacher infer an allegedly sensor-grounded crop concept solely from CDL labels.

### Grid

Recommended design: stable 10 m UTM-zone grids with deterministic half-open zone
ownership. Assign a cell to its zone by center; retain only one owner at a zone
boundary. Encode location ID as a versioned tuple `(grid_version, EPSG, row, col)`
and store WGS84 coordinates. Use EPSG:5070 for national area/quota accounting,
not one Missouri UTM projection for the entire US. Fix origins and boundary rules
before any sampling. Adjacent cells and terrain-processing halos at zone boundaries require
explicit reprojection and deduplication tests.

A 10 m grid is a sampling coordinate system. It does not make weather, soil,
land-cover or coarse optical bands native 10 m observations. Record native cell
IDs and spatial support for every source family.

### Hybrid annual sample

Proposed initial setting: a nominal **10M-location shared cohort**, chosen from
the 2021 sampling frame, plus replenishment each year to **20M accepted samples**.
The 50/50 fraction is not yet confirmed.

- Revisit shared locations while they remain cropland-eligible in that year and
  meet the declared minimum evidence policy. Preserve their stable location IDs.
- If a shared location becomes ineligible, skip that location-year and increase
  that year's replenishment quota. Do not relabel it or fill its observations
  with a neighboring location. Permit return to the shared cohort in later years.
- Do not define the shared cohort solely as the intersection of cropland in all
  five years: that would favor persistent land use and introduce future selection.
- Draw replenishment without duplicate locations within the year. Prefer locations
  not used by earlier replenishment cohorts, subject to feasibility and quotas.
  Record any authorized cross-year reuse rather than calling it new geography.
- Check shared-cohort retention against each year's climate/crop quotas. If crop
  transitions make the nominal shared fraction and quotas incompatible, resolve
  the conflict before scaling and publish the adjusted retention policy.
- Distinguish cohort membership, annual eligibility, selected candidate, rejected
  candidate and accepted sample. Count only accepted location-years toward 20M.

With an exactly persistent 10M shared cohort and five disjoint 10M refresh cohorts,
the idealized design yields **60M unique locations and 100M location-years**.
Attrition and quota-driven replacements change the unique-location count; 60M
is an illustration, not an acceptance guarantee.

## Diversity-aware allocation

Use the confirmed balanced policy through a hierarchical design rather than
uniformly sampling 20M pixels or dividing counts equally among states:

1. Establish geographic and agroclimatic strata from versioned, predeclared
   region/climate maps. Cross these with annual CDL crop groups, merging sparse
   combinations through a deterministic documented hierarchy.
2. Assign coverage floors to feasible strata, then distribute the remainder using
   cropland area and a bounded diversity oversampling factor. Publish targets,
   achieved counts, candidate availability and any shortfall redistribution.
3. Within strata, spread candidates across native weather cells and geographic
   blocks before adding more points to densely sampled areas. A proposed initial
   thinning unit is a 100 m selection block; the pilot must establish feasibility.
   One point per block is not a guarantee of 100 m pairwise separation.
4. Limit repeated points per coarse weather/soil cell through capacity-aware caps.
   Cap values must be derived from available eligible cells and annual quotas;
   an arbitrary hard cap must not make 20M infeasible or discard rare climates.
5. Use deterministic randomized selection without replacement, stable priorities,
   versioned seeds, and preselected reserve candidates for replacements. Count
   eligible pixels/blocks by tile and sample through hierarchical indexes; do not
   materialize every 10 m CONUS cell into a giant candidate-row database.

Publish the sampling frame, selected probabilities/weights where supported by the
actual design, and unweighted versus weighted summaries. Do not claim a simple
inverse probability weight corrects a multi-stage exclusion/replacement design
unless its inclusion probabilities have actually been derived. Calibrate a
reproducible weighting method during the pilot.

### Required diversity reports

For every year and for the combined corpus, report counts by state, climate
region, crop group, native ERA5/Daymet/SMAP cell, soil support unit, elevation and
terrain bins. Report unique locations, revisits, crop transitions, stratum quota
attainment, spatial clustering, and points per native cell (median/tails/max).

Also report weather coverage: unique cell-years, climate-normal ranges, annual
anomalies and growing-season temperature/precipitation distributions, with a
versioned reference climatology. Choose strata from independent geography or
climate normals; do not quietly select years/points using held-out target labels.
Many point records sharing one weather series are not independent climate samples.
These coverage metrics do not establish a statistical effective sample size.

## Sensor inventory and acquisition requirements

Acquire by source tile/grid and time block, then join selected points. **Do not
issue one public API request per point.** Cache source assets, spatial indexes,
processed tiles, and coarse time series across all points and years.

| Family | Proposed national role | Contract |
|---|---|---|
| Sentinel-2 L2A | Mandatory optical family | B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12; SCL/QA separately; dates, footprints, baseline, scale/offsets |
| ERA5-Land | Mandatory weather family | Native spatial cells; daily physical variables derived with correct units, accumulation resets and time basis |
| Daymet | Complementary weather | Native daily grid and calendar; retain separately from ERA5, without pretending identical products |
| DEM and terrain | Mandatory elevation/terrain family | Pinned national coverage and resolution; consistent slope/aspect/curvature/TWI derivation with buffers |
| SoilGrids | Mandatory shared soil contract | Eight existing properties × six depths, plus versioned uncertainty; bulk density as additional channels |
| SSURGO/gSSURGO and POLARIS | Proposed soil enrichment | Coverage-aware properties, depth alignment, component/horizon weights and categorical dictionaries |
| Sentinel-1 RTC | Proposed radar enrichment | Actual acquisition/relative orbit/direction, VV/VH and processing geometry; ragged records, not four Missouri tracks |
| SMAP | Proposed moisture enrichment | Native spatial cell, AM/PM observations and quality flags; no invented fine spatial variation |
| Annual CDL | Mandatory eligibility/provenance | Year-specific mask, native resolution, codebook and release date; input use separately controlled |
| NAIP | Optional context | Actual acquisition vintage and spectral availability; never imply annual or contemporaneous coverage |
| Coordinates | Metadata | Stable IDs, grids and native-source joins; intentional input use only |

Mandatory means a scientifically usable family under a pilot-calibrated missingness
policy, not that every date and every band must be finite. Define hard failures,
soft missingness and minimum temporal evidence separately. Global all-family
complete-case filtering would bias the sample toward easier regions and seasons.
Record failure reasons and compare rejection rates across strata. Do not replace
unavailable sources with invented zero values or a regional centroid broadcast.

The local source directory lacks several of these national products; its 254-slot
schema is not evidence that the corresponding measurements exist. The national
collector must actually acquire its required weather and soil families. Final
layer/version choices depend on the user budget and pilot, and deviations from
the mandatory package require a revised specification.

The S2 contract is the same twelve-band order as the existing YieldSAT core:
`B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12`. The ten bands present in the
local AOI are not sufficient to declare the national twelve-band requirement met;
B01 and B09 must also be acquired. The final corpus stores per-point spectra,
not 64×64 imagery or image tokens. SCL and validity/count metadata are separate.

### Source-specific precautions

- Sentinel-2 scale/offset and processing-baseline changes must be harmonized across
  2021–2025. Preserve raw provenance and publish the exact physical reflectance
  transform. Archive availability and baseline handling are documented by
  [Planetary Computer](https://planetarycomputer.microsoft.com/catalog) and its
  [change history](https://planetarycomputer.microsoft.com/docs/changelogs/history).
- Radar revisit availability changes across years: Sentinel-1B stopped acquiring
  after its December 2021 failure. Do not impose one year's cadence on all years
  or treat missing acquisitions as no backscatter.
  [ESA mission account](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-1/Sentinel-1B_journeys_back_to_Earth).
- Use the actual native ERA5-Land grid and published accumulation semantics;
  pin retrieval and aggregation versions.
  [Copernicus ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download).
- Daymet's calendar is not a generic leap-year Gregorian daily series; its guide
  describes omission of December 31 in leap years. Preserve dates and missingness
  during joins; do not shift subsequent weather by one day.
  [Daymet V4 guide](https://daacweb-prod.ornl.gov/DAYMET/guides/Daymet_Daily_V4.html).
- SoilGrids is a spatial prediction at 250 m and six standard depths; mapped units
  and uncertainty definitions require explicit conversion. A static soil product
  reused across five years is not five annual measurements.
  [ISRIC product description](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs.html),
  [layers and units](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html).
- A 10 m gSSURGO grid approximates soil map-unit polygons; preserve map-unit and
  component provenance rather than assuming independent soil observations per pixel.
  [NRCS gSSURGO](https://www.nrcs.usda.gov/resources/data-and-reports/gridded-soil-survey-geographic-gssurgo-database).
- Validate terrain value coverage and conditioning, not just file existence. The
  local pilot's TWI is finite at only 18,843 of 13,202,010 retained cropland points;
  source nodata coverage needs diagnosis before adopting that derivation nationally.

## Growing-season temporal contract

The delivered observation view has **24 slots per point-season**, containing
12 S2 bands plus aligned weather and other dynamic modalities. Full-acquisition
point histories and stored image patches are not required deliverables. Temporary
source staging and reusable native-cell daily weather caches may support production;
these are not extra point samples or a promise of a permanent full-time archive.

The confirmed growing-season definition uses **versioned crop- and region-specific
calendars**, with the sample year interpreted as **harvest year**. This avoids imposing a summer-only window on winter wheat or a single cycle
on double-cropped areas. A 2021 winter-crop sample may legitimately need autumn
2020 observations; those observations do not create a sixth sample year.

For every sample, record `season_year`, `season_start`, `season_end`,
calendar source/version, crop/calendar group, confidence and fallback reason.
Dates are calendar estimates, not invented observed planting/harvest dates.
Define windows for perennials, hay, fallow and double crops explicitly. Crop
calendar mappings and fallback windows must be frozen before national collection;
missing mappings cannot silently use a different region's summer season. Multiple
crop cycles still produce one record with 24 slots spanning the declared combined
window, with cycle metadata, unless the sample-count contract is revised.

Divide each declared season into 24 ordered, non-overlapping date intervals.
Pin inclusive/exclusive conventions, handling of short windows, source time zones,
leap days and each product's calendar. Store every interval's bounds and actual
satellite acquisition dates, quality flags and observation counts. Weather spans
the **entire slot interval**, not just the day an image was acquired; even a cloudy
optical slot retains its available weather. Do not truncate the last weather
interval at the last clear image date.

Select an observed clear S2 acquisition within each slot; tie-breaking, footprint
coverage and QA policy must be versioned. All 12 bands in a slot must belong to a
compatible acquisition, with native band-resolution masks. A composite is a future
explicit alternative requiring source-contributor provenance, not an unnoticed
change to the observed-spectrum policy. Missing slots remain masked; no off-season
image borrowing or silent temporal interpolation. Retain distinct band-valid and
clear-observation masks, especially at 10/20/60 m source-resolution boundaries.

Recommended core weather view: mean daily maximum temperature, mean daily mean
temperature, mean daily minimum temperature (degrees Celsius), and interval-total
precipitation (mm), each with valid-day counts and completeness flags. Final source
variable mappings and additional weather operators are frozen in the schema.
These physical aggregates must not be confused with the legacy dataset's summed
Kelvin temperatures and overlapping intervals. A later compatibility adapter must
convert explicitly rather than changing meanings under existing names. Derive
weather from native-cell source data and never multiply a coarse observation into
claimed independent 10 m measurements.

Canonical satellite shape is `(24,12)` per record. Core weather shape is `(24,4)`;
other approved dynamic layers share slot bounds or carry an explicit alignment
rule. Static layers remain one value vector per location/support unit. All slots
have time/quality/missingness metadata. Store source item identifiers and transforms
needed to audit values. Optional links to source imagery support provenance but
do not make a point record an image dataset.

## Evaluation integrity and knowledge pretraining

The user requires geographically excluded evaluation regions **across all years**.
The exact region/dataset list and boundary files remain unresolved. Do not start
national sampling without that exclusion manifest. Exclude all benchmark regions
and required source/context buffers, including earlier-year data used in winter
crop windows. Assign each remaining location permanently to a development or
pretraining split. Define a separate development-region allowance outside the
20M/year training quotas, or explicitly revise those quotas if development
records must be included in the 100M count. This counting choice must be recorded.

The intended 100M records are eligible pretraining records outside evaluation
regions; excluded candidates cannot be counted toward the annual target. Reserves
must replenish within allowed strata. No transductive geographic overlap is
permitted under the confirmed design.

Split before fitting normalization, climate thresholds, concept thresholds,
clustering or trainable encoders. Static/coarse source cells shared across a split
must be quantified; exclusion buffers should account for spatial context and the
scientific evaluation protocol. Random point splits do not provide geographic
independence. Publish overlap checks proving the agreed exclusion policy is met.

Adapt the scientific requirements of the [relational knowledge contract](yieldsat-image-knowledge-pretraining.md) to point evidence:
expert-approved concept/relation pairs, evidence at the appropriate native spatial
and temporal support, applicability abstention and auditable provenance. CLIP text
embeddings support the approved relation targets; no text is required at supervised
sensor-only inference. This point corpus does not itself supply DINO spatial inputs
or satisfy the existing image loader. DINO stays frozen in the separate image
workflow; point-model/knowledge adapters are a future task. S2 channel attention and elevation–S2
cross-attention belong to fusion fine-tuning, not this knowledge-pretraining stage.
Do not interpret thousands of points sharing a weather cell as thousands of
independent weather–soil concept judgments. Annotate representative clusters and
carry evidence scope; LLM/VLM calls are a separate budget, not one per 100M sample.
Image-specific VLM prompts cannot be applied to fabricated images made by
concatenating unrelated points.

## Storage and compute design

The production root and infrastructure are intentionally unspecified. The following
is a logical release layout, not a storage allocation or budget commitment.

```text
release/<version>/
  manifest.json                     # counts, versions, partitions, checksums
  schema/                           # semantic channels, units, masks, transforms
  sampling/                         # strata, quotas, seeds, candidates, exclusions
  points/year=<year>/region=<id>/... # point IDs, joins, cohort and selection metadata
  optical/...                       # 24-slot, 12-band point spectra and quality
  radar/...                         # orbit-aware acquisition tables and values
  weather/<product>/cell_window/...  # shared inputs/slot aggregates and point joins
  moisture/<product>/cell_year/...
  static/<product>/<version>/...     # values keyed by support unit/location
  views/24_slot/...                  # derived training views with manifest lineage
  splits/...
  reports/...
```

Use partitioned columnar metadata and chunked sensor arrays; evaluate Parquet plus
Zarr or sharded HDF5 in the pilot. Avoid a single enormous file, files per point,
or unbounded global HDF5 virtual-source fan-out. Write shards atomically; support
resume, bounded working memory, missing-source detection, checksums and explicit
schema migrations. Static properties and native-cell weather are stored once and
joined, with optional batch caches. Units and semantics take precedence over
blindly filling the local 120+134 layout. A later adapter may emit that view.
National orbit counts and optional layers require a versioned richer canonical
schema, not invented fixed channel assignments.

### Sizing scenarios, not cost estimates

| Representation | Arithmetic for 100M samples | Uncompressed payload |
|---|---|---:|
| Existing 24×254 float32 point view | 100M × 24 × 254 × 4 bytes | **2.4384 TB** |
| Same view stored as two-byte values | 100M × 24 × 254 × 2 bytes | **1.2192 TB** |
| 24-slot, 12-band optical point spectra, float32 | 100M × 24 × 12 × 4 bytes | **115.2 GB** |

These decimal-byte figures exclude masks, metadata, source-tile staging, indexes,
replicas and other temporary space. Two-byte storage is a
sizing scenario, not permission to quantize physical variables without an error
budget. Factorization/compression can reduce duplication; no ratio from the local
pilot should be extrapolated nationally. Shared-cell reuse, deduplication and read
amplification must be measured. These arithmetic estimates do not populate the
blank storage or acquisition budget fields.

The pilot must measure bytes per accepted location-year, unique source assets,
requests and transfer bytes, rejection/retry rates, worker peak memory, wall/CPU
time, read throughput and cost per sample. Estimate complete storage, temporary
high-water marks, egress and compute from stratified measurements with uncertainty.
Respect provider limits and bulk/cloud access mechanisms; never bypass limits or
launch unbounded pointwise calls.

## Tasks and acceptance gates

| Task | Work and deliverable | Acceptance |
|---|---|---|
| NP-00 | Resolve the hybrid fraction and evaluation boundaries; pin calendar products/mappings and freeze the release charter | Confirmed year/count scope, selection policy, modalities, evaluation and resource limits |
| NP-01 | Audit source products and license/access paths by year/region | Versioned coverage matrix; required-source gaps, auth needs and unit/calendar tests documented |
| NP-02 | Build national annual cropland frames and spatial exclusions | Native metadata pinned; zone-edge deduplication and class masks validated; no yield-dependent selection |
| NP-03 | Implement hybrid, stratified deterministic sampling and reserve queues | Quota sums 20M/year; feasible caps; reproducibility, cohort eligibility and replacement accounting |
| NP-04 | Run diverse multi-year acquisition pilot | Proposed 250k location-years across all five years and contrasting climates/crops; true counts and sizes measured |
| NP-05 | Validate canonical joins, calendars, units and temporal views | Exact-source spot checks; grid IDs, coarse-cell mappings, no crop/year shifts, masks and terrain coverage checks |
| NP-06 | Benchmark training access and cost; finalize infrastructure | Measured storage/throughput/worker plan and approved resource ceiling; point-series batch access demonstrated |
| NP-07 | Scale progressively, e.g. 1M then 5M then 20M/year | Each stage meets diversity, source quality, rejection and cost gates before expansion |
| NP-08 | Assemble and audit the full release | Exactly 20M accepted records in each of 2021–2025; 100M total; no duplicate location-year IDs; intact source lineage |
| NP-09 | Prepare adapters and knowledge-pretraining eligibility indexes | Separate future code task; compatible units/masks, frozen DINO, no target/text-inference dependency introduced |

The 250k pilot may contribute to final quotas only if it was selected under the
final design and passes the same checks; it is not an extra 250k on top of 100M.
Pilot quality thresholds must be fixed before scaling, including minimum optical
history, weather completeness, terrain validity and acceptable stratum shortfalls.
Reserves can replace failed candidates but cannot manufacture valid dates or
hide spatially structured missingness. If a quota is infeasible, report the
shortfall and revise the approved design instead of duplicating samples.

The final release report must give exact annual record counts; unique locations
and cohort transitions; all diversity and overlap summaries; per-point/per-family
availability; rejected candidates and reasons; storage and acquisition costs;
source versions, growing-season calendars, units and uncertainty conventions; corruption/missing
shard checks; and reproducible selection/build manifests. No national downloads
or model-training changes are authorized by this design document alone.
