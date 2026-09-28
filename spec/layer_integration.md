# Full-Layer Integration and Compliance Tasks

**Status:** Requirements and implementation backlog, 2026-09-28. Not implemented.

This is the model-side contract for consuming `geo-data-downloader` outputs and
the separately managed PlanetScope corpus. It supersedes the flat-file demo layout
as the target for real-data integration. Existing demo loaders remain useful for
tests; their presence does not establish compliance with this document.

## 1. Sources and scope

This inventory is a snapshot of the neighboring project's documents and source,
not a new verification of downloads:

- [Per-field layer inventory](../../geo-data-downloader/specs/data-layers.md).
- [Pretraining corpus requirements](../../geo-data-downloader/specs/pretraining-corpus/requirements.md)
  and [objective requirements](../../geo-data-downloader/specs/pretraining-corpus/pretraining-objectives.md).
- [PlanetScope corpus](../../geo-data-downloader/specs/pretraining-corpus/coverage-and-tiling.md).
- [Catalog implementation](../../geo-data-downloader/geo_data_downloader/catalog.py)
  and [yield export implementation](../../geo-data-downloader/geo_data_downloader/datasets/yield_target.py).

The field inventory documents verified field-scale outputs. AOI-wide gridded
weather, SMAP, and SoilGrids, multi-tile imagery handling, and several resolution
choices are separate downloader requirements; do not assume that those assets
already exist. PlanetScope is managed outside the downloader. Downloader phase
numbers refer to a different roadmap from this repository.

| Corpus | Training sample | Labels |
|---|---|---|
| Unlabelled pretraining AOI | Geographic tile and observation window | No yield assets or fabricated labels required. |
| Supervised field corpus | Field-year or a spatial patch within it | Measured yield and valid target coverage required. |

## 2. Inventory and canonical model streams

The names below define the proposed vocabulary for adapters, experiment configs,
ablations, and expert-statement tags. A dataset may produce multiple model streams
when resolution or acquisition geometry differs. The registry records that
grouping rather than treating each file as a new scientific modality.

| Stream | Source artifacts and content | Native representation | Required handling |
|---|---|---|---|
| `dem` | `dem_1m.tif`, or named 10/30 m fallback; elevation above NAVD88 | Raster, metres; CRS varies | Read actual grid and vintage; resolution is not always 1 m. |
| `terrain_1m` | `slope_1m.tif`, `aspect_1m.tif` | Raster, degrees | Keep native grid; encode aspect cyclically with validity preserved. |
| `terrain_5m` | `twi_5m.tif`, plan/profile curvature | Raster; TWI dimensionless, curvature 1/m | Preserve derivative scale and curvature sign convention. |
| `soil_ssurgo` | Map-unit GeoPackage, map-unit CSV, component/horizon CSV | Polygons and depth-indexed attributes | Join by keys; retain depths, component fractions, drainage and hydrologic categories. |
| `soil_polaris` | Six properties × six depths, 36 GeoTIFFs | Approximately 30 m, EPSG:4326 | Explicit property/depth ordering; organic matter is log10(percent). |
| `soil_soilgrids` | `soilgrids_centroid.json` for fields; AOI raster/grid mode planned | Field point, 250 m nominal | Divide by recorded `d_factor`; do not invent within-field variation. |
| `cdl` | Annual CDL GeoTIFFs and class-summary JSON | 30 m categorical history | Preserve codes and years; class 0 is background; summaries cover the field only. |
| `weather_daymet` | Daily CSV and metadata; `prcp,tmax,tmin,srad,vp,swe,dayl` | Point series, 1 km source grid | Read seven named variables, units, served point, dates and missing cells. |
| `weather_era5land` | Daily CSV and metadata; ten weather variables plus soil moisture/temperature at four depths by default | Point series, approximately 11 km | Read configured columns, UTC dates, units and served-point offset. |
| `soil_moisture_smap` | Daily CSV and metadata; moisture, quality/surface flags, vegetation water content | Point series, 9 km; AM and PM separate | Keep overpasses separate; use retrieval validity and observation masks. |
| `sentinel1` | Two-band VV/VH GeoTIFF per date/orbit/track plus index | 10 m RTC gamma0 **linear power** | Separate tracks/orbits; retain dates, CRS and band identity. Do not assume dB. |
| `sentinel2_10m` | Per-date B02, B03, B04, B08 | 10 m reflectance ×10000 | Preserve native group and band order; decode scale explicitly. |
| `sentinel2_20m` | Per-date B05, B06, B07, B8A, B11, B12 | 20 m reflectance ×10000 | Preserve native group; no input upsampling to 10 m. |
| `naip` | One acquisition, R/G/B/NIR plus index | Approximately 0.6 m, uint8 DN | Retain date and outside-window flag; AOI inclusion remains a budget decision. |
| `planetscope` | Eight-band SR GeoTIFF, UDM2, JSON/XML metadata | Approximately 3 m imagery series | Separate external adapter; validate bands, scale and actual grid from metadata. |

POLARIS properties are `clay,sand,silt,om,ph,bd`; depths in cm are
`0_5,5_15,15_30,30_60,60_100,100_200`. SoilGrids exposes
`bdod,clay,phh2o,sand,silt,soc`, with depth and scaling metadata. SSURGO includes
texture, organic matter, pH, bulk density, available water capacity, drainage,
hydrologic group, component percentages and horizon depth bounds.

Ancillary assets are part of the contract: Sentinel-2 SCL, PlanetScope UDM2,
acquisition indexes, source sidecars, catalog geometry, QA and provenance. SCL and
UDM2 supply quality/validity information by default; they are not silently added
as reflectance channels or counted as independent contrastive modalities.
Precipitation is a variable in the weather sources, not a third weather dataset.
SMAP observations and ERA5-Land modelled soil moisture remain distinct sources.

The supervised `yield` dataset exports `<field_id>_yield.tif`, its metadata JSON,
and `<field_id>_polygons` shapefile components. The current exporter records Mg/ha
and identifies `yieldMgHaM` as the authoritative polygon attribute. Validate the
exported metadata; never substitute the similarly named `yield_mg_h` column.

## 3. Required ingestion and representation behavior

### LI-R01 — Catalog-driven discovery

Consume `<data_root>/<record_id>/catalog.json`, dataset indexes and sidecars.
Resolve asset `href` values relative to configured `data_root`, rather than
reconstructing paths using `<field>_<layer>.tif`. Read boundary geometry from the
catalog; no `_boundary.tif` is required from the downloader.

Preserve QA verdicts, flags and provenance. Make inclusion of `review` records an
explicit configuration and record exclusions. Verify asset readability and
distinguish failed/malformed assets from legitimate missing modalities. Offline
ingestion must not invoke downloader commands that can migrate or mutate a manifest.

Read raster metadata directly when required. The inspected catalog writer names a
field `resolution_m` even for geographic rasters, where its value comes from CRS
coordinate units. Do not interpret degrees as metres. Units and observation
details for tables come from sidecars/config; the catalog does not duplicate all
metadata.

### LI-R02 — Shared footprint, separate native grids

Define one geographic sample footprint and time window, then window each source
on its own grid. Maintain source CRS, affine transform, pixel size, valid extent
and acquisition time with tensors. Never resample all predictors onto a boundary,
yield or PlanetScope grid before feature extraction.

Within-source alignment needed for a temporal stack must have a declared reference
grid and interpolation policy; otherwise encode acquisitions separately. Cross-CRS
assets and SAR tracks must not be concatenated as though their pixels were
colocated. Fusion receives physical position/resolution/time metadata for tokens,
rather than relying solely on sequence index for correspondence.

### LI-R03 — Units and normalization

Validate channels and physical representations before normalization. Register
statistics by source, property, depth and resolution group as appropriate; fit
statistics on training data only and version them with preprocessing.

Existing S1/S2 statistics are not automatically valid for this corpus. Linear S1
gamma0 must not be treated as dB or normalized with unrelated DN statistics. Any
log conversion is explicit, recorded and masks nonpositive/invalid values first.
Handle S2 scaling, SoilGrids `d_factor`, and POLARIS OM deliberately. Keep OM in log
space or convert with `10**x` according to the declared representation; compare
physical percentages with SSURGO only after conversion. Do not assume PlanetScope
has Sentinel-2's scale.

### LI-R04 — Observation validity at every level

Read each raster's nodata, including float32 minimum, -9999, -32768 and optical
zero where declared. Carry separate masks for missing modalities, invalid pixels/
tokens, missing time steps/features, and supervised target coverage. Empty CSV
cells are missing, not observed zero. Apply cloud/quality masks before computing
statistics, targets or losses.

Batch padding remains masked. Every pretraining objective ignores unavailable
target rows and invalid target elements. Fusion excludes invalid input tokens;
the Phase 2 learned all-absent fallback is not an observed reconstruction target.
The current per-modality availability mask alone does not satisfy this requirement.

### LI-R05 — Time and information cutoff

Use actual observation dates, temporal gaps and quality indexes. Preserve daily
rows with missing values and separate SMAP AM/PM. Keep CDL years explicit. Do not
synthesize observations or assume equal intervals for imagery acquisitions.

Supervised experiments declare a prediction cutoff, defaulting to harvest start
for the initial pipeline. Filter every predictor by it; NAIP's nearest acquisition
may lie outside the window and must not silently admit future data. Temporal
forecasting uses only information available before the predicted step across all
conditioning sources. Yield values/statistics never enter pretraining predictors,
statement generation or sensor embeddings.

Availability includes product release time where relevant, not just the nominal
observation year. Harvest-year CDL published after the cutoff cannot silently
supply a current-season predictor; use admissible prior history or declare the
experiment's alternative information policy. Record the vintage/availability
policy for static products and crop/context metadata as well.

### LI-R06 — Sample index and objective eligibility

Index tile/field footprints, coordinates, windows and source availability. Measure
spatial/temporal co-occurrence with configured time tolerance; file existence alone
does not establish a valid positive pair. PlanetScope-only tile counts are not
multimodal sample counts.

Spatial reconstruction requires enough valid spatial tokens. Field weather/SMAP
and centroid SoilGrids are context/temporal modalities, not fine-resolution maps.
Eligibility depends on actual tokenization and coverage, not filenames or a
hardcoded source table. Forecasting requires a valid past and target. Contrastive
positives require co-occurrence; negatives require a configured minimum geographic
separation, verified in physical distance rather than tile-index distance.

The registry distinguishes a source family from its streams. Default whole-modality
dropout and leave-one-out prediction hold out all streams of a target family (for
example, both Sentinel-2 resolution groups), while predicting each target stream's
embedding. Contrastive source pairs are between distinct families by default;
band-group alignment is a separately declared experiment, not additional sensor
diversity. Keep target quality masks for eligibility and loss computation even
when their parent sensor is held out; do not expose target-derived content to the
predictor.

### LI-R07 — Encoders, fusion and batching

Use a source registry rather than the fixed `dem/sar/weather/soil/crop` configs.
Define native raster, irregular imagery sequence, daily series, soil polygon/
horizon, tabular/context and categorical history adapters. Validate encoder output
rank, token count and grid metadata at fusion.

Handle different tensor sizes/dates with masks and a bounded token budget. Do not
resize all sources merely to make `torch.stack` succeed. Resolve `grouped_vit`'s
rank defect before using it and the unified container contract before selecting
that container. Working containers can be extended without wiring every optional
head. The temporal encoder's current mean-pooling limitation remains D4; carrying
real dates and masks alone does not make it order-sensitive.

### LI-R08 — Targets and geographic evaluation

Choose and record the supervised output grid separately from predictor grids.
Inspect the portal raster transform: it is not guaranteed to be a square, aligned
5 m grid. Choose explicitly between registering its raster and rendering exported
yield polygons on the chosen grid. Record aggregation/interpolation/coverage and
preserve source checksums and units.

Use field-boundary and observed-yield masks for loss and metrics. Supervised samples
without valid targets are ineligible; unlabelled pretraining samples need no target.
Group repeated field identities across years and split by geographic blocks.
Prevent held-out evaluation geography from entering pretraining for the primary
transfer experiment; label any transductive experiment separately. Report yield
RMSE in Mg/ha alongside spatial metrics and modality ablations.

### LI-R09 — Lazy access and reproducibility

Build datasets from lightweight indexes, lazy raster windows/table reads, bounded
worker caches, and cache keys incorporating asset versions, footprints and
preprocessing. Do not eagerly reproject the AOI into parent-process memory. Record
sources, cutoffs, bands, units, masks, normalization, sampling, splits and token
budgets with each experiment.

## 4. Compliance task backlog

All tasks are open, ordered by dependency. `LI-` IDs are model-side tasks, not
downloader REQs. Documenting them does not claim implementation.

| Task | Phase and dependencies | Deliverable and acceptance |
|---|---|---|
| **LI-01: Inspect and index the handoff** | Phase 3, first | Inspect field catalogs and available AOI assets offline; record schemas, units, QA and absent sources. A relocated fixture works without original absolute paths or manifest writes. |
| **LI-02: Implement the source registry** | Phase 3; LI-01 | Define §2 streams, property/depth/band order, ancillary roles, grids, timestamps and aliases. Unknown names, inconsistent channels and unsupported units fail clearly. |
| **LI-03: Build native-footprint datasets** | Phase 3; LI-01/02 | Separate unlabelled tile and labelled field/patch modes with lazy reads and catalog geometry. Mixed-CRS fixtures retain separate grids over the same physical footprint. |
| **LI-04: Add static/soil/history adapters** | Phase 3; LI-02/03 | Read DEM, terrain, all POLARIS depths, SoilGrids point/grid assets, SSURGO joins and CDL history. Verify units, depths, categorical values and cyclic aspect. |
| **LI-05: Add daily-series adapters** | Phase 3; LI-02/03 | Read Daymet/ERA5-Land/SMAP columns and sidecars with configurable features, dates, coordinates and masks. Missing cells/dates and AM/PM round-trip faithfully; unavailable AOI sources remain explicitly absent. |
| **LI-06: Add imagery adapters** | Phase 3; LI-02/03 | Read S1 tracks/orbits, both S2 grids/SCL, NAIP metadata and external PlanetScope/UDM2. Validate scale/bands/geometry; reject accidental cross-track stacking and future acquisitions. |
| **LI-07: Implement transforms and train-only statistics** | Phase 3; LI-04/05/06 | Version conversions, masks and normalization; numerically verify linear S1, S2 scale, POLARIS OM and SoilGrids factors. Held-out values never enter fitted statistics. |
| **LI-08: Extend batching, encoders and fusion** | Phase 3; LI-03–07 | Carry dates, physical token coordinates/resolution, variable sizes and pixel/time masks through source-configured encoders and fusion. Masked/padded values cannot affect outputs; measure token/memory limits. |
| **LI-09: Implement co-occurrence and spatial sampling** | Phase 3; LI-03/05/06 | Build tile/window indexes and objective-eligibility reports; enforce time tolerance, geographic negatives and held-out blocks. Verify distances, shared footprints and missing-source eligibility. |
| **LI-10: Audit all five objective masks** | Phase 3; LI-08/09 | Exclude unavailable/invalid targets from every loss; prevent future-target leakage through conditioning sources. Empty eligible sets produce defined zero losses and counts. |
| **LI-11: Implement the yield adapter** | Phase 3; LI-03/07 | Read raster/polygons/metadata; select the output-grid policy and target/boundary masks. Verify `yieldMgHaM`, Mg/ha, irregular raster transforms and georeferenced output. |
| **LI-12: Connect real training and checkpoints** | Phase 3; LI-08–11 | Both entry points consume real datasets; synthetic input requires `--smoke-test`. Add explicit encoder/fusion transfer with compatibility checks. Real pretraining and supervised steps complete; fine-tuning loads intended weights. |
| **LI-13: Validate the pilot end to end** | Phase 3; LI-12 | Run a documented real-source pilot, report exclusions/co-occurrence and held-out spatial RMSE, and verify CPU executability. Record GPU results when available. Missing upstream AOI data is reported, never fabricated. |
| **LI-14: Profile and evaluate transfer** | Phase 4; LI-13 | Measure I/O, tokens, memory and fusion passes; compare scratch/pretrained on identical spatial splits and budgets. Tune weights using validation; report source/subset ablations. |
| **LI-15: Connect approved knowledge statements** | Phase 4; LI-02/10/14 | Route reviewed statement tags through the registry and implement [knowledge pretraining v1](./knowledge_pretraining.md). Compare approved/shuffled/no-text conditions; yield inference needs no text artifacts. |

Full-layer compliance requires adapters and contract tests for every listed source
and documented handling of genuinely unavailable assets. A pilot may use a
declared subset; it must not be labelled full-layer validation. Useful transfer
remains an experimental criterion, not a guaranteed implementation outcome.

## 5. External dependencies and recorded decisions

- Inspect which AOI assets exist; gridded Daymet/ERA5-Land/SMAP/SoilGrids and
  multi-CRS outputs are external dependencies until verified. Do not substitute
  one AOI-wide point for a required grid.
- Record DEM/terrain resolution, NAIP inclusion, tile footprint/stride, token
  budgets, time tolerance, negative separation, spatial blocks, output-grid policy
  and cutoff in experiment configuration. The downloader's proposed 192 m/64×64
  PlanetScope tile is a candidate, not a fixed model requirement.
- Probe actual PlanetScope band names, scale and quality conventions rather than
  filling undocumented fields with guesses.
- Pin the external data/spec snapshot with integration experiments; changes require
  schema/contract checks, not modifying the downloader from this repository.
