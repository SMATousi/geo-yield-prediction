# Nationwide US cropland pretraining corpus

**Status (2026-10-05):**
- Pilot design confirmed (D13–D19).
- Builder implemented (`yieldsat_us/`, `yieldsat_us_pilot.py`).
- **Frame built:** 125 clusters, 249,952 points over 2021–2025, in
  `/home1/pupil/SMATousi/YieldSAT-US-Pilot`.
- **Extraction pending a NASS QuickStats key** (D16).
- Stage-0 knowledge pretraining showed no DEV gain
  (spec/pretraining/success-criteria.md, "Applied"), so the scale-up gate
  (D10) is not met. The pilot completes as planned, because it is cheap and
  useful for a parity check, but **no scaling beyond 250k** without new
  evidence.

Earlier status: design draft, 2026-10-04. This document specifies a future collection;
it does not launch downloads, reserve infrastructure, or modify training code.
Confirmed decisions and proposed defaults are distinguished below. Outstanding
user choices must be incorporated before the acquisition plan is finalized.

## Revised plan (2026-10-04, user decision)

The corpus is built **in measured stages, not committed to 100M up front**,
and only after the knowledge-pretraining pipeline has been validated on data
we already have. This revision takes precedence over conflicting statements
below. The 100M target (D01) stays the ceiling, not a commitment.

1. **Point model first.** All pretraining, adapters and evaluation use the
   point model ([yieldsat-point-knowledge-pretraining.md](./yieldsat-point-knowledge-pretraining.md)).
   The image model comes back later; nothing here needs image tiles (D06
   already stores points).
2. **Stage 0 — YieldSAT inputs only.** Knowledge pretraining is first
   validated on the YieldSAT inputs of all 9 pairs (~12M cells, labels
   never used), judged by [pretraining/success-criteria.md](./pretraining/success-criteria.md).
   US collection proceeds in parallel only through the pilot.
3. **Pilot (NP-04, 250k location-years)** in exactly YieldSAT-compatible
   semantics (point 4), then **1M → 5M → 20M** location-years. Each stage is
   kept only if it passes the **scale-up gate** of the success criteria:
   a DEV-mean gain ≥ +0.005 pixel R² against the previous stage with a CI
   excluding 0, and no regression of intrinsic criteria I1/I3. Scaling stops
   at the first stage that fails, and the result is reported. Diversity is
   expected to saturate well before 100M: ERA5-Land CONUS cropland is
   roughly 10⁴ cells, ~10⁵ cell-years over 2021–2025, so later points mostly
   repeat weather/soil cells.
4. **Feature semantics must match the YieldSAT contract**, or pretraining will
   not transfer:
   - The released training view must carry **exactly the YieldSAT point
     semantics**: the 24-slot scheme of `yieldsat_preprocessed_v1`, S2 in L2A
     digital numbers in the same 12-band order, and weather as interval sums
     with YieldSAT's operator (temperatures as Kelvin-day sums,
     precipitation as metre sums over the interval since the previous dated
     slot; the first dated slot is invalid), plus the same static channels
     and units (SoilGrids units as in the YieldSAT cache).
   - The physical view recommended in "Growing-season temporal contract"
     (daily means in °C, precipitation in mm) may be stored as well. The
     view used for pretraining is the YieldSAT-compatible one, or an
     explicit, tested adapter converts both corpora to one semantics.
     Silently mixing semantics is not allowed.
   - **Terrain must be fixed before use:** TWI is finite for only 0.1% of
     the local pilot's points, and slope's scale is unresolved in YieldSAT
     as well.
5. **Evaluation exclusion (D08):** all current evaluation sets (YieldSAT:
   Argentina, Brazil, Germany, Uruguay) lie outside the US, so a CONUS
   corpus cannot leak them. D08 only matters for a future US benchmark; until
   one is declared, the exclusion manifest is empty by decision, and that is
   recorded in the release manifest.
6. **Knowledge supervision on US points** uses the same rule-based estimators
   (with US strata: state/climate region × crop) and the same 6 approved
   rules as the YieldSAT stage.

## YieldSAT-aligned acquisition design (2026-10-04)

This section freezes how US points are built so that they match the
YieldSAT point contract (`yieldsat_preprocessed_v1`) the point model is
trained on. Measured YieldSAT conventions below come from the local caches
(spec/yieldsat_data_contract.md §2 plus the 2026-10-04 probes recorded
here). Where an earlier section of this spec disagrees, this section wins
for the pretraining view. The physical view stays optional.

### A. Measured YieldSAT conventions the US view must reproduce

| Item | YieldSAT (measured) | US pretraining view |
|---|---|---|
| Slot grid | The dates span **only ≈ seeding → harvest**: no date is > 60 d before seeding. The raw release holds every acquisition (16–131 per field, every 2–5 days); the **preprocessed NetCDF keeps one per calendar month** (p50 7–11 per season, all raw dates). The slot *position* is the month index: slot k = calendar month k counted from January of (harvest year − 1). That is a positional frame, not a 2-year data window. 92–97% of dated slots match exactly, the rest are one month later. Median gap 30 days; months outside the season are undated. | Same rule: harvest year from the crop calendar (D07). One acquisition per calendar month from the calendar seeding month through the harvest month. Other slots are undated (NaN date) |
| Acquisition per slot | The least-cloudy acquisition in the month (median day of month 20–23) | Among the month's acquisitions, prefer those clear at the point (SCL 4/5/6), ranked by scene `eo:cloud_cover`, then the latest date, then item ID. If none is clear, take the least-cloudy acquisition and set the optical values to NaN (a dated slot with masked optics, as in YieldSAT: 6–42% of its dated slots) |
| S2 bands and units | L2A, 12 bands `B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12`, DN = reflectance × 10,000, no negatives | Same order and units |
| Processing-baseline offset | **Harmonized**: no +1,000 shift after the January 2022 baseline 04.00 change. Low percentiles (B02 p0.5 ≈ 60–200 DN) are continuous across 2017–2024 in all four countries, so the BOA offset was removed. The pattern matches `COPERNICUS/S2_SR_HARMONIZED`. | For items with processing baseline ≥ 04.00 whose source did not already remove the offset, subtract 1,000 and clamp at 0. The source's offset flag and the baseline are stored per item |
| Spatial sampling of S2 | 20/60 m bands nearest-upsampled to the 10 m cell | The value of the native pixel containing the 10 m cell centre (exact transform, no warping), per band at its native resolution |
| Weather | ERA5-Land daily, field-centroid. Each dated slot holds the **sum of daily values over the inclusive interval [t_(k−1), t_k]**: temperatures in K·days, precipitation in m. The first dated slot is invalid | ERA5-Land daily at the native cell containing the point, same operator over the same acquisition dates (D07 interval rule replaced by this one for the pretraining view) |
| DEM / terrain | SRTM 30 m, cubic-upsampled; RichDEM slope (unresolved scale), aspect (degrees), curvature, TWI (mostly NaN) | SRTM 30 m (same product, not the 1 m lidar or 8 m local derivatives) with the same RichDEM derivatives. Validity rule v3 applies (|curvature| > 1,000 invalid) |
| Soil | SoilGrids 2.0, 250 m, mapped units (clay/silt/sand g/kg, SOC dg/kg, pH × 10), 8 properties × 6 depths | Same product and units, value at the containing 250 m cell |
| Season dates | Per field: farmer seeding and harvest dates | State × crop × year planting and harvest dates from USDA NASS Crop Progress (median-progress week), recorded as `seeding_date_type = calendar_estimate` |

The concept estimators (spec/yieldsat-point-knowledge-pretraining.md §3)
then apply unchanged, with US strata (state or climate region × crop).

**Dense series (optional).** The point model consumes the monthly view,
so the pretraining view is monthly for parity. The raw YieldSAT release also
offers the dense ~5-day series, which a later model variant could use. If
it does, the US reader stores every clear in-season acquisition per point,
≈ 5× the block reads of §B. The monthly view is then derived by selecting
one acquisition per month.

### B. Pixel-only Sentinel-2 acquisition (answers "can we download only the pixels?")

**Yes.** Whole scenes never need to be downloaded or kept on disk. There are
two pixel-only routes:

1. **COG range reads (recommended primary).** Sentinel-2 L2A is published
   as Cloud-Optimized GeoTIFFs with a STAC index:
   - AWS Earth Search `sentinel-2-l2a`: public, no authentication;
   - Microsoft Planetary Computer: free, signed URLs.

   Each band is internally tiled (≈ 1,024 × 1,024 pixel blocks). With
   GDAL/rasterio over HTTP, a point read fetches **only the compressed
   block(s) containing it** through HTTP range requests, about 1–2 MB per
   band block, instead of ≈ 1 GB per 12-band scene.
   - Values are sampled in memory and the block is discarded.
   - Temporary disk use is essentially zero (a bounded in-memory block
     cache).
   - Item IDs, baselines and offsets give exact provenance.
2. **Google Earth Engine** `COPERNICUS/S2_SR_HARMONIZED`: server-side
   monthly selection and point sampling (`sampleRegions`). Only the values
   come back (≈ 1.2 KB per point-season). The offset is already harmonized,
   which likely matches how YieldSAT was produced. Costs:
   - Earth Engine quotas and registration;
   - weaker item-level provenance;
   - batch-export throughput limits at tens of millions of points.

**Cost driver: bytes per block, not per pixel.** A block covers about
10 × 10 km at 10 m. An isolated point costs a whole block read per band per
acquisition, while a block with 1,000 sampled points costs the same.
Estimate per season and block:
- ≈ 12 months × (13 bands × ~1.5 MB);
- plus ≈ 6 acquisitions per month × the 20 m SCL block for clear-sky
  selection;
- ≈ 0.3 GB per block-season.

Two consequences:
- **Clustered sampling is required for the COG route.** Sample points within
  a limited set of block-sized clusters, e.g. ~2,000 points per cluster.
  - The 250k pilot is then ≈ 125 blocks ≈ 35 GB of streamed reads.
  - 20M points per year is ≈ 10k blocks ≈ 3 TB streamed per year, with
    nothing stored except the point values.
  - Uniformly scattered points would cost one block per point, ≈ 1,000×
    more traffic.
- Diversity quotas (D05) are then met at the cluster level. Clusters are the
  primary sampling units, stratified by climate region × dominant crop;
  points within a cluster are stratified by CDL class. Cluster sampling
  inflates the variance of between-cluster statistics; the diversity reports
  count clusters as well as points.

Stored per point and slot: 12 band values, SCL class, item ID, acquisition
date, baseline, offset applied, clear flag. Per point: grid ID, WGS84
coordinates, cluster ID, CDL class and calendar fields. The layout is the
YieldSAT cache layout (`temporal (N,24,16)`, `static`, `times`, an index and
a `fields`-like season table; no target).

### C. Relation to the existing local AOI build

`spec/yieldsat-us-pretraining-point-dataset.md` documents a separate build:
one Missouri AOI, 2025, 13.2M cropland points.
- It is **not** YieldSAT-aligned:
  - 10 S2 bands (no B01/B09);
  - 24 equal bins over a calendar year instead of the monthly harvest-year
    grid;
  - raw DN with no harmonization;
  - no ERA5-Land and no SoilGrids;
  - local 8 m terrain instead of SRTM.
- It may be used for loader tests and for the image-model branch later.
- It is not used for point-model pretraining unless it is rebuilt to §A.

### D. Decisions

| ID | Decision | Status |
|---|---|---|
| D13 | The pretraining view reproduces the measured YieldSAT conventions of §A (monthly 24-slot harvest-year grid, harmonized L2A DN, nearest native pixel, ERA5-Land inclusive interval sums, SRTM/RichDEM, SoilGrids mapped units) | Proposed 2026-10-04 (follows D11) |
| D14 | S2 is acquired pixel-only: COG range reads from the STAC archive (primary), GEE `S2_SR_HARMONIZED` as parity check/fallback; no scene downloads or scene staging | Proposed 2026-10-04 |
| D15 | Clustered sampling (block-sized clusters as primary sampling units, D05 quotas at cluster level) to make pixel-only reads efficient | **Confirmed 2026-10-04** |
| D16 | Season dates for US points come from NASS Crop Progress state × crop × year medians, marked as calendar estimates | Proposed 2026-10-04 |

### D2. Pilot decisions (2026-10-04, project lead)

| ID | Decision |
|---|---|
| D17 | Pilot strata: **NOAA's nine US climate regions** (state-based) × crop strata. The allocation is about 125 clusters of ~2,000 points (≈ 250k points), spread over 2021–2025 and balanced across years within each region: Upper Midwest 15, Ohio Valley 20, Northern Rockies & Plains 25, South 25, Southeast 10, Northwest 12, Northeast 8, Southwest 5, West 5 |
| D18 | **All CDL cropland classes are kept** (codes 1–61, 66–80, 200–255), not only the YieldSAT crops. Pretraining selects crops at training time (a crop filter in the training script) |
| D19 | Output: `/home1/pupil/SMATousi/YieldSAT-US-Pilot/`. That volume had 120 GB free on 2026-10-04, so national CDL downloads are transient (one year at a time, deleted after use) and S2 is pixel-only |

### D3. YieldSAT terrain conventions recovered from `Raw.zip` (2026-10-04)

The raw per-field rasters (`dem/{dem,slope,aspect,curvature,twi}`, 10 m)
show that the terrain derivatives were computed on the **native SRTM lat/lon
grid** and then upsampled:
- **slope** = 84,000 ± 3% × the true rise/run on every field checked. That is
  RichDEM `slope_riserun` with horizontal units in degrees (≈ 111 km per
  degree, latitude-scaled), which explains the "unresolved scale" of 1e3–1e4.
- **aspect**: downslope azimuth, clockwise from north. It matches a 10 m
  recomputation within 10–19° median (differences come from upsampling).
- **curvature**: correlates −0.65 to −0.83 with metre-based
  Zevenbergen–Thorne curvature, with a field-dependent scale (×1.1–2.2),
  consistent with computation at native resolution.
- **TWI**: NaN in most fields (100% of the German fields checked).

The US build computes slope (degree-unit rise/run) and aspect the same way
on native NASADEM 1-arcsecond tiles, then samples with cubic interpolation.
- Curvature: Zevenbergen–Thorne at native resolution, with its scale
  calibrated to the YieldSAT distribution in NP-04a.
- TWI: stored as NaN, as in most of YieldSAT.

### E. Pilot build tasks (NP-04, YieldSAT-aligned)

| ID | Deliverable | Acceptance |
|---|---|---|
| NP-04a | Parity tests on YieldSAT itself | (1) From `Raw.zip`, confirm the monthly selection rule: which raw acquisition each month keeps (least cloudy over the field?). (2) Re-extract S2 for ≥ 20 YieldSAT field seasons (all 4 countries) from the COG archive with the §A rules. Agreement with the cache: same slot dates for ≥ 90% of dated slots; median absolute band difference ≤ 2% of the value. This verifies harmonization, the monthly selection rule and nearest sampling before any US point is built |
| NP-04b | Cluster frame and sampler | CONUS cropland (CDL) block clusters, stratified by climate region × dominant crop; deterministic, seeded; 250k points in ≈ 125 clusters across 2021–2025 |
| NP-04c | Pixel-only S2 reader | STAC query per cluster and season; block-aligned range reads; monthly clear-sky selection; harmonization; streaming writes; bounded memory; measured bytes per point |
| NP-04d | ERA5-Land, SRTM/RichDEM, SoilGrids joins | Native-cell joins with YieldSAT operators and units; validity rules; first dated slot invalid |
| NP-04e | YieldSAT-format cache and audit | Cache, index and season table loadable by `YieldSATPointDataset` (no target); concept indices built; unit and range audit against YieldSAT distributions per channel |

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
| D04 | 10M shared locations + annual replenishment to 20M accepted samples | Confirmed |
| D05 | Balanced climate-region and crop-type sampling, with weights | Confirmed |
| D06 | Points, not stored images; same 12 S2 bands as YieldSAT | Confirmed |
| D07 | 24 growing-season slots with aligned weather; versioned crop- and region-specific calendars; harvest-year sample identity | Confirmed |
| D08 | Exclude evaluation regions across all years | Exclusion policy confirmed; current evaluation sets are outside the US, so the manifest is empty until a US benchmark is declared (2026-10-04) |
| D09 | Storage, acquisition budget, compute environment and deadline | Intentionally blank, at user request |
| D10 | Staged scaling (pilot → 1M → 5M → 20M location-years), each stage gated by measured DEV gains; 100M is a ceiling | Confirmed 2026-10-04 |
| D11 | Point model first; pretraining view in exact YieldSAT point semantics (or a tested adapter) | Confirmed 2026-10-04 |
| D12 | Stage 0 = knowledge pretraining on YieldSAT inputs before any US data is used | Confirmed 2026-10-04 |
| D13 | Pretraining view reproduces measured YieldSAT conventions (monthly harvest-year slots, harmonized DN, nearest native pixel, ERA5-Land inclusive sums, SRTM/RichDEM, SoilGrids) | Proposed 2026-10-04; see "YieldSAT-aligned acquisition design" |
| D14 | Pixel-only S2 (COG range reads; GEE harmonized as parity/fallback); no scene downloads | Proposed 2026-10-04 |
| D15 | Clustered sampling with block-sized clusters as primary sampling units | **Confirmed 2026-10-04** |
| D16 | Season dates from NASS Crop Progress state × crop × year medians | Proposed 2026-10-04 |

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

Confirmed design: a **10M-location shared cohort**, chosen from the 2021 sampling
frame, plus replenishment each year to **20M accepted samples**. When all shared
locations remain eligible, replenishment contributes 10M samples; otherwise it
increases to cover that year's shared-cohort shortfall.

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
The user has explicitly left the exact region/dataset list and boundary files
undecided. Do not infer exclusions from the local corpus or substitute arbitrary
regions. This remains an unresolved prerequisite: do not start national sampling
without the exclusion manifest. Exclude all benchmark regions
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
| NP-00 | Resolve evaluation boundaries; pin calendar products/mappings and freeze the release charter | Confirmed year/count scope, selection policy, modalities, evaluation and resource limits |
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

### F. Pilot build log

- 2026-10-04 — **Builder implemented** (`yieldsat_us/`, `yieldsat_us_pilot.py`
  with stages `frame` and `extract`).
  - **frame:** national CDL per harvest year (downloaded, used, deleted),
    10,230 m cluster cells in EPSG:5070, NOAA regions from Census state
    boundaries, clusters drawn in proportion to stratum area, about 2,000
    points per cluster (one 10 m cell per 100 m block, stratified by CDL
    class). 2021: 26 clusters across all 9 regions, 51,991 points.
  - **extract:**
    - S2 from AWS Earth Search COGs, pixel-only. One cluster (2,000 points):
      174 acquisitions scanned via SCL, 132 band reads, 23 s. Earth Search
      values are already harmonized (`earthsearch:boa_offset_applied`; July
      2023 blue p0.5 = 266 DN).
    - Weather from Open-Meteo: ERA5-Land temperatures. Precipitation comes
      from ERA5 at 0.25° because Open-Meteo serves no ERA5-Land
      precipitation; ERA5-Land precipitation is the interpolated ERA5
      forcing.
    - SoilGrids mean layers (uncertainty optional) and NASADEM terrain in the
      recovered YieldSAT conventions.
    - About 2 min per cluster.
  - **First cluster check** (2021 Minnesota corn) against YieldSAT ranges:
    B04/B08 p50 900/2,778 DN; slope p50 2,795 (YieldSAT 2,545–2,665);
    curvature p5–p95 ±0.39 (YieldSAT ≈ ±0.5); ≈ 287 K/day; 6 monthly dated
    slots at positions 15–21.
  - **Calendar:** no NASS QuickStats key is available, so all dates are
    currently the fallback crop windows (`calendar_fallback`).
- 2026-10-04 — **Frame complete for 2021–2025: 125 clusters, 249,952
  points.**
  - Per year: 26 / 25 / 24 / 25 / 25 clusters.
  - Per region, exactly the D17 allocation: Upper Midwest 15, Ohio Valley
    20, Northern Rockies & Plains 25, South 25, Southeast 10, Northwest 12,
    Northeast 8, Southwest 5, West 5.
  - Per stratum: corn 46, soybean 37, wheat 17, canola 10, winter wheat 10,
    spring wheat 5.
  - Transient CDL removed.
  - Extraction waits for a NASS QuickStats key (project lead's choice), so
    season dates come from Crop Progress. The test cluster with fallback
    dates was discarded.
