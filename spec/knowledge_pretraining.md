# Expert-Validated Statement Pretraining — v1

**Status:** Approved design, 2026-09-28. Not implemented.

This document specifies an optional addition to the field-scale System B
pretraining framework. The existing five objectives remain in place. The new
component uses general agronomic statements to condition cross-modal prediction
before supervised fine-tuning on measured yield.

## 1. Purpose and approved scope

A model drafts general statements about modality types and their relationships.
A human agronomic expert validates every statement admitted to the training
library. A frozen text encoder embeds the approved statements, and those
embeddings provide additional context to an auxiliary sensor-prediction task.

The approved v1 decisions are:

- Statements describe general agronomic relationships, not individual field
  observations or sample captions.
- The drafting model receives modality descriptions and relationships, not field
  sensor measurements or measured yield.
- Every training statement must be approved by a human expert.
- Knowledge contributes a soft, weighted auxiliary loss.
- Yield prediction uses sensor inputs only. Text is used during pretraining.
- V1 requires statements only: no executable conditions, numerical rule
  thresholds, logical predicates, or explicit statement-satisfaction scorer.

The longer-term intent is to encourage representations consistent with expert
knowledge connecting multiple modalities. V1 implements **statement-conditioned
cross-modal pretraining**, not direct enforcement of conditional agronomic rules.
Its effectiveness must be measured rather than inferred from expert approval.

## 2. Statement library

Store the library as a versioned JSONL artifact. Each record contains:

| Field | Meaning |
|---|---|
| `id` | Stable statement identifier. |
| `text` | The complete expert-approved natural-language statement. |
| `modalities` | Model modality names involved in the statement. |
| `review_status` | `draft`, `approved`, or `rejected`. Only `approved` is eligible. |
| `reviewer_id` | Identifier of the expert who approved this version. |
| `reviewed_at` | Approval timestamp. |
| `version` | Version of this statement. Editing approved text requires review again. |
| `generator` | Optional provenance for the drafting model and prompt version. |

The modality tags support routing and do not turn the statement into a structured
rule. They identify which inputs the text discusses; they contain no conditions,
thresholds, or numerical consequences. Require at least two distinct, recognized
modality names for this cross-modal objective.

Resolve names through the full-layer source registry in
[layer_integration.md](./layer_integration.md), rather than the prototype's generic
`dem/sar/weather/soil/crop` names. Tags can identify actual sources such as
`sentinel1`, `soil_polaris`, `weather_daymet`, or `soil_moisture_smap`, and specific
streams such as `sentinel2_10m` or `terrain_5m`. Qualified property/depth/band
references stay in the statement text or registry mapping; they do not introduce
executable conditions in v1. The library version pins its registry mapping.

Statement approval concerns agronomic meaning, including any qualifications
written in the text. No per-field statement review or executable rule review is
required in v1. Reject malformed records and duplicate identifiers; exclude
unapproved records from the training library.

## 3. Text encoding

Use a pretrained text encoder with frozen parameters and evaluation-mode behavior.
Choose its model and tokenizer during implementation; this spec does not select a
provider or model. Cache embeddings of the approved library rather than re-encode
unchanged statements on every training step.

Record the library version/hash, text encoder identifier and revision, tokenizer,
and preprocessing settings with each experiment. These identify the actual
knowledge input and allow a cache to be rebuilt or invalidated after changes.

The text encoder consumes statements only. It does not generate sample-specific
claims from sensor data during training.

## 4. Auxiliary prediction task

For an eligible field sample and target modality:

1. Encode the observed modalities using the existing native-resolution encoders.
2. Hold the target modality out of the predictor's sensor inputs. Its actual
   encoding supplies the reconstruction target only.
3. Select approved statements tagged with the target and at least one remaining
   observed modality. Average their cached text embeddings to obtain one
   conditioning vector for this prediction direction.
4. Fuse the remaining sensor tokens with the existing backbone and mean-pool its
   latents into a field representation.
5. Concatenate that representation with the text conditioning vector. A dedicated
   auxiliary MLP for the target modality predicts its pooled sensor embedding.
6. Compare the prediction with the detached, mean-pooled target embedding.

For shared sensor dimension `D` and text dimension `D_text`, the auxiliary head
maps `(B, D + D_text)` to `(B, D)`. Gradients update the remaining sensor encoders,
the fusion backbone, and the auxiliary head. The frozen text encoder and detached
target branch receive no gradients from this objective. Other pretraining
objectives continue to train the modality encoders.

The first version predicts pooled embeddings, consistent with the current
cross-modal objective. It does not reconstruct spatially distinct sensor pixels.

Samples may be unlabelled AOI tiles as well as fields. The same physical footprint,
time tolerance and source-validity rules from the full-layer contract apply. Hold
out the target source family, including its related resolution/track streams, by
default; select and predict individual eligible target-stream embeddings within
that held-out family. Do not count resolution groups from one sensor as independent
sources when routing a cross-modal statement.

### Eligibility and missing modalities

Compute this objective only for sample/target pairs with:

- an actually observed target modality;
- at least one actually observed conditioning modality; and
- at least one relevant approved statement.

Availability is evaluated per sample. A learned missing token is not an observed
target or a qualifying conditioning source. Missing target rows do not contribute
to the loss. If no sample/target pair is eligible, return a zero knowledge loss and
report zero eligible pairs.

Pool targets over valid observations/tokens only, excluding cloud, nodata and
padding; a present file with no valid coverage is not an eligible modality.
Co-occurrence means shared geographic coverage and appropriate observation times,
not merely files stored under one record. Known source conversions (linear SAR,
scaled reflectance, soil units) follow the source registry before encoding.

Selection by modality tags does **not** establish that a statement's agronomic
conditions hold in a particular field. The text provides context to the predictor;
v1 has no mechanism to test or activate those conditions explicitly.

## 5. Soft training objective

Use mean squared error in the shared sensor embedding space. Average over embedding
dimensions and eligible samples for each target direction, then average the active
directions:

```text
target_m = stop_gradient(mean_pool(sensor_encoder_m(input_m)))
prediction_m = knowledge_head_m(concat(fused_other_sensors, relevant_text))
L_knowledge = mean_over_active_directions(MSE_over_eligible_rows(prediction_m, target_m))
L_total = L_existing_five_objectives + lambda_knowledge * L_knowledge
```

Expose `lambda_knowledge` as an experiment configuration value; choose its value
using validation rather than fixing an unsupported default. The feature is
optional and disabled by default. When disabled, preserve the existing objective
behavior and perform no text encoding or additional fusion passes.

Use a separate auxiliary head from ordinary cross-modal prediction so the
statement-conditioned component can be enabled and ablated independently. Report
the unweighted knowledge loss, its weighted contribution, and eligible pair count.
Sample prediction directions if necessary to keep the added fusion cost bounded.

## 6. Fine-tuning and inference

Transfer the sensor encoders and fusion backbone to supervised yield fine-tuning.
Discard the text encoder, cached statements, and knowledge prediction heads from
the yield inference path. Loading a fine-tuned model and making predictions must
require no statement library, text input, or text-model access.

Measured yield remains the ground truth for supervised fine-tuning and evaluation.
The statement library supplies pretraining context; it supplies no yield labels.

## 7. Evaluation and acceptance

Compare three primary pretraining conditions:

1. The existing five objectives without knowledge conditioning.
2. The existing objectives plus approved statement conditioning.
3. The same added objective with shuffled statement assignments.

Also compare the added prediction head with its text input removed or replaced by
a constant vector. This separates the benefit of additional prediction training
from the benefit of statement content. Use comparable training budgets, the same
sensor data, spatial splits, fine-tuning protocol, and evaluation metrics.

Measure held-out cross-modal reconstruction error and downstream yield-map
performance. Report uncertainty across repeated runs where practical. Shuffled
text and no-text controls are necessary because the predictor may learn to ignore
the statements. Useful expert knowledge is not, by itself, evidence that this
conditioning mechanism transfers it into sensor representations.

Acceptance requires:

- Tests exclude unapproved statements and missing target rows from the objective.
- Tests verify that holding out a target prevents its embedding from entering the
  predictor and that text-encoder parameters remain frozen.
- Tests cover empty libraries, no eligible pairs, and heterogeneous availability.
- Disabling the feature preserves the existing pretraining path.
- Fine-tuning and inference run without text artifacts.
- Reproducible real-data comparisons establish whether the addition helps.
  Synthetic runs establish executability only.

Do not report logical rule satisfaction or violation rates for v1: it has no
executable definition of either. Report a lack of measured benefit or ignored text
as an experimental result, not as successful knowledge transfer.

## 8. Implementation placement and later work

Implement this as an optional extension of `MultimodalSelfSupervisedPretrain`.
The statement library and embedding cache belong to the pretraining data/config
path, rather than the supervised field-sample contract.

Real unlabelled field ingestion is a prerequisite for meaningful evaluation
(roadmap Phase 3). Integrate and profile the experimental objective with the
pretraining and transfer work in Phase 4. No implementation is claimed by this
document.

Later versions may explore explicit conditions, measurable consequences,
applicability checks, confidence-weighted rules, and statement-satisfaction
scoring. Those require a separate design decision and are outside the approved v1
scope.

## 9. Statement-generator prompt

**Prompt version:** `statement_generator_v1.0`.
**Layer vocabulary snapshot:** `layer_inventory_2026-09-28`.

Copy the complete block below into the drafting model. It is self-contained and
requests an initial batch of at most 60 statements. This prompt does not approve
its outputs or implement the pretraining objective. Record the actual generator
model/revision and generation settings externally; the model must not invent its
own provenance. Allocate globally unique statement IDs when importing multiple
batches, then retain those IDs through expert review and versioning.

```text
You are drafting an agronomic knowledge library for review by a human expert.
Your output will be used in statement-conditioned multimodal pretraining before
supervised crop-yield fine-tuning.

TASK
Generate up to 60 distinct, scientifically defensible, general agronomic
statements connecting the modalities listed below. You receive only the modality
inventory, not observations from any field. Do not describe a particular field,
invent measurements, estimate yield, or infer that any condition is currently
present. Every generated statement is a DRAFT requiring human expert approval.

The training task hides a sensor source and uses other sensor representations plus
approved statement embeddings to predict the held-out source representation.
Statements should therefore express useful relationships between observable soil,
terrain, weather, moisture, vegetation or crop-history information. Text is used
only during pretraining; yield inference uses sensor inputs only.

This version uses natural-language statements and modality tags only. Do not
generate executable rules, formal predicates, numerical thresholds, pseudo-labels,
loss functions, training code, or statement-satisfaction scores. Conditional
wording and exceptions belong in the natural-language statement itself.

AVAILABLE MODALITIES — USE THESE EXACT TAGS

1. dem
   USGS 3DEP elevation in metres above NAVD88. Field outputs are commonly 1 m,
   with named 10/30 m fallbacks. Actual CRS, resolution and acquisition vintage
   vary. Elevation is not itself a measurement of fertility, moisture or yield.

2. terrain_1m
   DEM-derived slope in degrees and aspect clockwise from north. Aspect is a
   circular quantity. These are terrain descriptors, not direct water observations.

3. terrain_5m
   DEM-derived topographic wetness index (TWI, dimensionless), plan curvature and
   profile curvature (1/m); positive profile curvature is convex-upward in this
   inventory. TWI/curvature use a separate 5 m grid. TWI is a terrain-based index,
   not a direct measurement of current soil moisture.

4. soil_ssurgo
   Soil map-unit polygons with component/horizon attributes: sand/silt/clay,
   organic matter, pH, bulk density, available water capacity, drainage class,
   hydrologic group, component percentages and horizon depth bounds. Texture and
   organic matter are percentages; depth bounds are centimetres. Attributes describe
   mapped soil components/horizons, not dense contemporaneous field measurements.

5. soil_polaris
   Approximately 30 m gridded estimates of clay, sand, silt, organic matter, pH
   and bulk density at 0–5, 5–15, 15–30, 30–60, 60–100 and 100–200 cm depths.
   Texture is percent; bulk density is g/cm3. Stored organic matter is log10(percent),
   not percent. Express agronomic relationships in clearly identified physical
   properties/depths; do not compare stored log values with raw percentages.

6. soil_soilgrids
   SoilGrids properties bdod, clay, phh2o, sand, silt and soc with depth metadata
   and source-specific scaling factors. Field outputs are centroid samples at
   250 m nominal resolution; AOI-wide raster/grid support is planned. Values need
   decoding using d_factor. Organic carbon and organic matter are different
   properties and are not interchangeable without a justified conversion.

7. cdl
   Annual 30 m categorical crop/land-cover rasters and in-field class summaries
   for the harvest year and prior years. Class 0 is background; codes are labels,
   not continuous numbers. Crop history describes mapped crop classes; it does
   not reveal irrigation, fertilizer, cultivar, tillage or other management.
   A harvest-year release may not be available before the prediction cutoff.

8. weather_daymet
   Daily weather: precipitation (prcp, mm/day), maximum/minimum temperature
   (tmax/tmin, degrees C), solar radiation (srad, W/m2), vapor pressure (vp, Pa),
   snow water equivalent (swe, kg/m2), and day length (dayl, seconds).
   Field outputs are point series from a 1 km source grid. Dates and missing
   observations are explicit. Precipitation is a variable here, not a new modality.

9. weather_era5land
   Daily reanalysis: maximum/minimum/mean 2 m temperature, precipitation, rain,
   snowfall, reference evapotranspiration, shortwave radiation, wind speed and
   relative humidity; also modelled soil moisture and temperature at 0–7,
   7–28, 28–100 and 100–255 cm depths. Field outputs are point series from an
   approximately 11 km grid, with UTC dates and recorded served-point offsets.
   Modelled soil moisture is not an independent in-field sensor observation.

10. soil_moisture_smap
    SMAP soil moisture and vegetation water content, with retrieval-quality and
    surface flags. AM and PM overpasses remain separate. Soil moisture is m3/m3;
    vegetation water content is kg/m2. The nominal grid is 9 km, with frequent
    observation gaps. Treat it as coarse contextual information, not a map of
    within-field moisture or an automatic measurement of the entire root zone.

11. sentinel1
    Sentinel-1 RTC gamma0 VV/VH radar backscatter at 10 m, stored as LINEAR POWER,
    not dB. Observations have acquisition dates, orbit/track and grid metadata;
    viewing geometry can differ. Radar response must not be treated as uniquely
    identifying soil moisture, crop biomass or another single agronomic variable.

12. sentinel2_10m
    Sentinel-2 surface reflectance: B02 blue, B03 green, B04 red, B08 near-infrared,
    at 10 m. Stored reflectance is scaled by 10000. Acquisition dates and SCL
    quality information accompany imagery; invalid/obscured pixels are masked.

13. sentinel2_20m
    Sentinel-2 surface reflectance: B05, B06, B07 red-edge bands, B8A narrow
    near-infrared, B11/B12 shortwave infrared, at 20 m. Stored reflectance is scaled
    by 10000. This is another resolution group of the SAME sensor as sentinel2_10m,
    not independent sensor evidence. Preserve distinctions in band sensitivity.

14. naip
    Aerial R/G/B/NIR imagery, approximately 0.6 m, uint8 digital numbers.
    Typically one acquisition rather than a seasonal time series. Its actual date
    can lie outside the field observation window; do not assume it depicts current
    crop conditions or is radiometrically equivalent to satellite reflectance.

15. planetscope
    Separately managed approximately 3 m eight-band surface-reflectance imagery
    with acquisition metadata and UDM2 quality masks. Exact band names/order and
    scale must be verified from metadata; they are not specified here. Do not
    invent PlanetScope band identities or apply Sentinel-2 scaling by assumption.

Ancillary SCL, UDM2, quality flags, dates and provenance are not extra modality tags.
Measured yield rasters/polygons are reserved for supervised fine-tuning and are
not inputs to statement generation. Several AOI source extensions are planned;
the inventory is not proof that every source is observed for every training sample.

STATEMENT REQUIREMENTS
1. Express one principal agronomic relationship per statement. Use one to three
   sentences, approximately 35–100 words, without filler or repetitive paraphrases.
2. Connect two to four of the allowed modality tags. Every tag must participate
   meaningfully in the statement; do not add unrelated tags to improve coverage.
   Include at least two source families. Sentinel-2's resolution groups are one
   family; DEM-derived terrain is not independent evidence from DEM. Relations
   between resolution groups alone or identities among terrain derivatives do
   not qualify as cross-source agronomic statements.
3. State important qualifications in the text: crop or growth stage, relevant
   soil depth, temporal order, moisture regime, spatial scale, viewing geometry,
   and management dependence where they materially change the relationship.
   Do not assign universal directional effects where the relationship is conditional.
4. Distinguish physical mechanisms, observational associations and sensor proxies.
   A reflectance index is not a direct yield measurement; radar backscatter is
   not uniquely determined by moisture; TWI is not observed wetness. Do not turn
   a plausible association into an unconditional causal claim.
5. Respect scale and time. Coarse weather/SMAP/centroid soil data cannot resolve
   fine within-field patterns. Contemporary relationships require compatible
   observation times; static soil/terrain and old NAIP imagery cannot automatically
   describe current canopy conditions. Soil depths and measurement supports must
   be comparable before asserting agreement across sources.
6. Missing or invalid values provide no agronomic evidence. Do not interpret cloud,
   nodata, a missing overpass or an empty CSV cell as low vegetation or dry soil.
7. Statements may mention an important unobserved condition as a qualification,
   but must not assert that fertilizer, irrigation, planting date or management
   is available in these layers. Do not introduce unavailable modalities or labels.
8. Favor established relationships a human agronomic expert can assess. Do not
   invent citations, numerical effect sizes, probabilities, yield values, or
   confidence scores. If you cannot defend a relationship, omit it.
9. Cover a balanced range of terrain–soil–water relationships, weather–soil–water
   relationships, environmental context–vegetation relationships, optical–radar
   relationships, crop-history context and cross-source soil interpretation.
   Include every source where a defensible relationship exists, but do not force
   unsupported statements or every possible modality pair to meet a quota.
10. Prioritize agronomic relationships useful for connecting sensor information.
    Do not fill the library with file-format facts, variable definitions,
    generic data-quality advice, management prescriptions or preprocessing steps.
    The inventory caveats constrain your statements; they are not the entire task.

OUTPUT
Return only valid JSONL: one JSON object per line, with no markdown fences,
headings, rationale, prose preamble or trailing commentary. Use exactly these keys:

{"id":"agronomy_v1_0001","text":"<your draft statement>","modalities":["<allowed tag>","<allowed tag>"],"review_status":"draft","reviewer_id":null,"reviewed_at":null,"version":1,"generator":{"prompt_version":"statement_generator_v1.0","layer_vocabulary":"layer_inventory_2026-09-28","model":null}}

Use sequential IDs starting at agronomy_v1_0001. Never set review_status to
approved or invent a reviewer, review timestamp or generator model identity.
Keep conditional qualifications inside text; do not add structured conditions,
consequences, thresholds, scores or other keys. Produce fewer than 60 statements
if necessary to preserve defensibility and avoid duplicates. Before returning,
check JSON validity, allowed tags, tag relevance, source-family diversity,
scientific qualifications, and that every record remains draft.
```

After generation, validate the JSONL and tag vocabulary, deduplicate by meaning,
and submit the draft text and tags to the human expert. Approval or revision is
recorded through the statement-library review fields. Text or tag changes to an
approved version require review again. Generation alone never makes a statement
eligible for the training library.
