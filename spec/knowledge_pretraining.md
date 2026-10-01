# Expert-Validated Relationship Pretraining — v2

**Status:** Approved replacement design direction, 2026-09-28. Not implemented.
The VLM/LLM applicability estimator is an experimental option to evaluate, not a
selected model or a validated annotator. Scoring contracts and operational choices
listed below must be resolved before implementation claims or training use.

This design supersedes v1's statement-conditioned held-out sensor prediction.
Keep the existing five self-supervised objectives. The optional addition now
uses expert-approved agronomic concepts and relationships to constrain the
representations of observed modalities directly. It does not append text to a
sensor reconstruction head. Existing ordinary cross-modal prediction remains.

## 1. Purpose and scope

The knowledge library describes concept A, concept B, and their qualified
relationship: for example, an elevation regime and a soil property within a
specified geographic context. That example is a design illustration, not an
approved universal elevation–soil rule.

- A drafting model receives the source inventory, not individual observations or
  measured yield. A human expert reviews all admitted concepts, relationships,
  qualifications and their training interpretation.
- Frozen text embeddings represent concept descriptions and complete relationship
  statements. Sensor-side projections connect observations to those concepts.
- Soft concept-grounding and relationship losses update sensor encoders. Both
  observed endpoints may receive gradients; no sensor must be hidden for this
  objective.
- Applicability can be estimated by a frozen VLM/LLM from permitted sensor evidence
  and metadata. Its output is uncertain auxiliary supervision, not ground truth.
- Yield fine-tuning and inference use sensor inputs only. No text model, rule
  library, applicability teacher or knowledge head is required at deployment.

V2 explicitly expands the old statements-only scope to include concept definitions,
applicability estimates and executable training scorers. Natural-language statements
remain the scientific source; a text embedding alone does not define a scorer.
This is knowledge-guided/self-supervised pretraining with weak concept supervision,
not purely unsupervised learning or a source of synthetic yield labels.

## 2. Versioned knowledge library

Store draft/reviewed relationships in JSONL, with a shared concept registry so a
concept has the same definition wherever it is reused. Resolve modality names,
properties, depths, bands, units and source families through
[layer_integration.md](./layer_integration.md). Pin the registry snapshot.

| Relationship field | Meaning |
|---|---|
| `id`, `version` | Stable relationship identifier and immutable revision. |
| `text` | Complete qualified agronomic statement. |
| `modalities` | Participating canonical streams; at least two source families. |
| `concept_a`, `concept_b` | Objects containing `id`, `description`, `modality`, `property_refs`. References must resolve through the source registry before use. |
| `relation` | Object containing `type`, `direction`, `description`, `qualifications`. Types: `association`, `ordered_association`, `compatibility`; direction: `a_to_b`, `b_to_a`, `symmetric`. Direction of association does not establish causality. |
| `applicability` | Object containing `context`, `required_evidence`, `unobservable_conditions`, `estimator_policy_ref`. Unknown context/evidence must permit abstention. |
| `scoring_contract_ref` | Versioned executable interpretation; null in drafts is allowed, but such a record cannot supply a relationship loss. |
| `review_status`, `reviewer_id`, `reviewed_at` | `draft`/`approved`/`rejected` and expert review provenance. |
| `generator` | Prompt/model/settings provenance; actual model identity is supplied by the generation wrapper. |

Concept definitions specify spatial support, time support, relevant soil depth,
and physical meaning. Terms such as “high”, “wet” or “vigorous” need a reviewed
reference context; do not assume globally valid numerical thresholds. Definition
changes require a new concept version and review of affected relationships.

A separate scoring-contract artifact records supported relation type, required
concept states/orderings, activation policy, score/loss formula, normalization,
weights or margins, admissible contrasts, and reviewer/version provenance.
An approved natural-language relationship becomes training-ready only when its
concept grounding, applicability policy and scorer have also been reviewed and
validated. Drafts and unresolved contracts are excluded. Never silently migrate
old v1 statements into approved v2 rules.

### Alternative YieldSAT source mapping

The [YieldSAT branch](./yieldsat_data_contract.md) uses a separate
versioned vocabulary and country-specific evidence mapping for its temporal
optical/weather and static terrain/soil streams. Confirm physical units, interval aggregation,
uncertainty meaning and target-free spatial support before applying concepts or
teacher prompts. Do not treat merged-grid vectors as original native-grid layers,
use yield metadata in applicability evidence, or run §11's US inventory prompt
unchanged. YS-07 specifies the branch integration with KP tasks; dedicated
country-aware YieldSAT generator/evidence prompts are future implementation work.
The [YieldSAT image knowledge-pretraining spec](./yieldsat-image-knowledge-pretraining.md)
now defines this branch's encoder, support, and transfer contracts. It is a
design, not a completed concept mapping or annotation library.
`yieldsat_objectives.OBJECTIVE_ROUTING` still marks knowledge relationships
*blocked* for the implemented point branch. Weather aggregation is characterized
(inclusive-interval Kelvin-day/metre sums), but slope and TWI scales, soil
uncertainty definition and coordinates remain unresolved.

## 3. Text references and sensor concept grounding

Use a frozen pretrained text encoder in evaluation mode. Cache separate embeddings
for both concept descriptions and the full qualified relationship. Pin encoder
and tokenizer revisions, preprocessing, library hashes and embedding-cache keys.
Select the model during implementation; generic text encoders are not assumed to
understand numeric sensor units or encode agronomic truth in cosine distances.

For valid sensor tokens, form a representation at the concept's actual support:
valid-token pooling for a tile/field, or local token grouping for a spatial patch.
Do not broadcast coarse SMAP/weather/centroid soil evidence as independent fine
pixel annotations. Match footprints, dates, depth and scale before relating sources.

A small modality-side projector maps the sensor representation into the fixed text
reference dimension. A prototype-based concept head produces concept scores, e.g.
normalized sensor/text similarity with a fitted temperature and bias. Its output
must be anchored against observed-property definitions or validated teacher labels;
raw cosine similarity is not automatically a calibrated presence probability.
Use source-appropriate multi-label or mutually exclusive concept-state heads.
Known-absent concepts are valid grounding negatives; unknown concepts are masked.
A student concept head receives only its declared sensor source and allowed metadata,
not the other endpoint's teacher prediction or cached labels as input features.

Train the projector/head and sensor encoder using soft concept targets from §4.
Keep full sensor embeddings available to fusion; concepts are an auxiliary
projection, not a mandatory bottleneck that discards all other information.
Frozen prototypes and known positive/negative examples anchor semantics and limit
arbitrary changes of the concept coordinate system.

## 4. Applicability estimation: VLM/LLM as an optional teacher

Separate two questions:

1. **Concept presence:** does the observed source support concept A or B?
2. **Relationship applicability:** are the relationship's antecedent/context
   qualifications observable and satisfied for this sample or comparison?

Neither is a yield label or proof of the statement. The relationship gate must
not depend on whether the predicted consequence agrees with the rule. Otherwise
it could hide violations by activating only already compatible pairs.

### Evidence and teacher inputs

Build an offline evidence package from QA-valid observations and allowed metadata:
physical-unit tables/summaries for soil, terrain and weather; suitable image crops
for a VLM; dates, resolution, depth, footprint, provenance and missingness. Choose
an LLM for explicit numeric/tabular evidence or a VLM when image interpretation
is actually needed. A rendering of DEM or soil values needs its physical legend;
a natural-image VLM must not be presumed competent on multispectral/SAR arrays.
Summaries must preserve the support of the concept they label.

The teacher receives reviewed concept/rule definitions and this package. It must
not receive measured yield, yield-derived summaries, held-out evaluation labels,
or observations beyond the relevant cutoff. It must not infer unobserved
irrigation, fertilizer or management. Unknown required conditions lead to
abstention unless the expert-approved policy explicitly permits a qualified
relationship without them. Never ask the model to decide whether a rule is true
by treating its own agronomic narrative as observed evidence.

Use deterministic source-property estimators as a grounding reference or alternative
where a reviewed definition makes a concept directly measurable. VLM/LLM use is
optional and must be compared with these where feasible. No provider calls or
sample data transfer are authorized by this specification change.

### Cached annotation contract

For each concept/sample/support, store:

- sample/support/window ID and concept/rule versions;
- `status`: `estimated`, `unknown`, or `not_observable`;
- nullable `presence_score` and separate nullable `confidence_score` in [0, 1];
- cited input asset/property/pixel-region/date references and missing evidence;
- for a relationship, a separate context applicability status/score and evidence;
- teacher model/revision, prompt/preprocessing/settings versions, input hash,
  cache version, and any deterministic-estimator or calibration revision.

A model's self-reported confidence is not calibrated reliability. Presence scores
become soft targets only after the estimator policy passes validation; confidence
weights require separate calibration or an explicit conservative policy validated
on reviewed examples. Unknown is never converted to zero or “concept absent”.
Reject non-finite/out-of-range scores, unsupported evidence references, malformed
records, incompatible units/supports and stale cache entries.

Review a stratified subset of annotations with an expert, covering positives,
negatives, unknowns, sources, seasons and regions. Fit calibration and choose
abstention policies on training/validation geography only; report error, calibration,
coverage and abstention. The user has approved expert review of every rule, not
mandatory manual review of every sample estimate. Freeze accepted estimators and
cache outputs offline. No joint updates from the student, no student-produced
labels, and no online teacher calls inside the training loop.

## 5. Relationship semantics and executable scorers

Do not define the loss as “make DEM and soil vectors similar” or “copy the distance
between two sentences”. Semantic similarity neither establishes an association
nor captures its direction, qualifications or magnitude.

Represent a relationship by its text embedding plus a reviewed scoring contract.
A bounded scorer evaluates sensor concept representations in the anchored concept
space. Support these first implementations only when the required evidence exists:

- **Ordered association:** compare eligible sample pairs in the same applicable
  context. A reviewed signed-order scorer softly penalizes reversed ordering of
  the specified endpoint properties/concept states, with a declared tolerance.
  Observed orderings, meaningful contrasts and qualifications must be available;
  do not rank arbitrary geographic pairs or force every exception to disappear.
- **Compatibility:** use a reviewed concept-state compatibility table or validated
  relation scorer. Penalize specifically declared incompatible combinations with
  soft weights. An ordinary association does not make alternative outcomes invalid.
- **General association:** a sentence alone cannot set a numerical compatibility
  matrix or conditional probability. Retain such records in the library, but skip
  their relationship loss until an expert-reviewed, validated contract exists.
  A future fitted probabilistic scorer must use training geography only and
  document the extra data supervision it introduces.

If a text-conditioned relation module is used, fit it separately using reviewed
concept-pair/relationship examples and admissible negative examples, validate it,
then freeze it before sensor pretraining. Relation-text swaps must affect its score
in expected ways. Do not let a jointly trained unconstrained MLP satisfy every rule
by changing its outputs while leaving sensor embeddings uninformative. A fixed
explicit table/order scorer is also supported; document when text is used to define
concept coordinates rather than materially controlling the relationship scorer.
Neither path claims automatically recovered relationship geometry from text.

Negatives must be justified by the scorer and evidence. Different fields, shuffled
rules or unobserved concepts are not automatically agronomically incompatible.
Use shuffled rules only as an experimental control, not as scientific labels.

The default knowledge branch acts on observed per-source encoder representations,
so both endpoints receive gradients through their concept projections. Gates,
teacher targets, cached text and the relation scorer are detached/frozen. This
branch does not directly train fusion by default; the existing five objectives do.
Any later fused-representation extension needs a separate leakage/gradient audit.

Freeze scorer parameters, not its computation on student inputs: preserve autograd
through the scorer into both endpoints. Training scorers must provide differentiable
soft penalties; hard decisions/order checks belong in eligibility or evaluation.
Specify any smooth surrogate and verify its gradients in the scoring contract.

## 6. Soft objectives, eligibility and gradient flow

For soft, independently present concepts, a first grounding implementation can use
weighted binary cross-entropy; use a declared categorical loss for exclusive states.
Choose the relationship loss from its reviewed contract, not from sentence cosine.

```text
h_a, h_b = valid_support(sensor_encoder_a(x_a)), valid_support(sensor_encoder_b(x_b))
p_a, p_b = concept_head_a(h_a, frozen_text_a), concept_head_b(h_b, frozen_text_b)
L_ground = weighted_known_concept_loss(p_a, p_b, detached_teacher_targets)
L_relation = weighted_applicable_relation_loss(p_a, p_b, frozen_relation_contract)
L_total = L_existing_five + lambda_ground * L_ground + lambda_relation * L_relation
```

Order scorers may consume a second matched sample pair. Weights combine reviewed
rule weights, calibrated annotation reliability and context applicability. They
are nonnegative, detached, bounded and fixed with respect to student predictions.
No eligibility decision may use measured yield or drop a pair solely because it
violates the relationship. Grounding remains eligible even when a relation is
inapplicable, if the individual concept target is reliable.

Relation eligibility requires two actually observed source families, valid shared
support/time/depth, an approved training-ready rule, accepted grounding/evidence,
and a supported scoring contract. Missing tokens, cloud/nodata/padding and stale
annotations are excluded. Related S2 grids or SAR tracks are not independent
source families. Follow full-layer spatial split, co-occurrence and cutoff rules.

Normalize each objective over its eligible weight sum, then balance active
concepts/rules so high-frequency statements cannot dominate. Return a defined zero
and zero counts when no eligible terms exist, with no NaNs or division by zero.
Report denominators, skipped reasons, active rules and unweighted/weighted losses.
Expose both knowledge weights, scorer margins, temperatures and sampling budgets
in versioned experiment configuration; select them on validation data.

The feature is optional and disabled by default. Disabled runs perform no text,
teacher, cache or extra-head work and preserve the existing training path. Retain
all five existing objectives to preserve signal beyond the selected concepts.
Monitor representation variance, concept discrimination and response to source
perturbations: low relationship loss alone cannot establish embedding quality.

## 7. Fine-tuning and inference

Transfer the sensor encoders and fusion backbone to supervised yield fine-tuning.
Discard text encoders/caches, applicability estimators, concept projections and
relationship scorers from the yield inference path. Attach the yield head and
train against actual measured yield. Loading and inference require no knowledge
artifacts, provider access or teacher annotations.

The transferred representations should retain useful information learned through
these auxiliary losses. This is an experimental aim, not a guarantee of useful
yield transfer or universal agronomic rule satisfaction.

## 8. Evaluation and acceptance

Compare on identical sensor data, spatial splits, seeds where feasible, and matched
budgets (report offline annotation cost separately):

1. Existing five-objective pretraining only.
2. Existing objectives plus concept grounding only.
3. Grounding plus reviewed relationships with the same annotations.
4. The same branch with shuffled relationship/concept assignments as a control.
5. Constant/no-text references with the same annotations and capacity; distinguish
   gains from weak concept supervision or explicit scorers from gains due to text.
6. Validated VLM/LLM versus deterministic or expert annotations on an overlapping
   measurable subset; report teacher coverage/error as well as training results.

Evaluate concept discrimination/calibration on independently expert-reviewed
examples, embedding variance and sensitivity, and conditional relationship scores
on held-out applicable supports. Report coverage and scorer definitions with
violation scores; never call them proof of agronomic truth. Test rule-text swaps
and sensor perturbations for shortcut behavior. Grounding and relationship gains
must be distinguished; inspect whether only projections improve while transferable
sensor features do not. Evaluate sensor-only downstream yield with the knowledge
branches removed, including low-label transfer and missing-source subsets.

Acceptance requires meaningful checks for:

- draft/unreviewed/unresolved rules and malformed/stale annotations excluded;
- unknown versus known-absent labels, cutoff/QA/support/depth eligibility, and
  missing/unobservable conditions handled without fabricated evidence;
- no consequence-based gate or arbitrary false-negative generation;
- frozen teacher/text/scorer parameters; gradients into both eligible sensor
  encoders; no detached student path; no teacher annotations as student inputs;
- well-defined zero losses for empty eligible sets and heterogeneous availability;
- bounded costs, reproducible provenance and no teacher calls in the train loop;
- disabled-path equivalence and text/teacher-free transfer/inference;
- real-data controlled comparisons and transparent reporting of no benefit.

Synthetic checks establish execution, not knowledge validity or transfer.

## 9. Implementation tasks and decisions

Implement as an optional extension of `MultimodalSelfSupervisedPretrain` after
real-data ingestion (Phase 3). These tasks expand Phase 4's LI-15, with LI-02/10/14
as prerequisites. All remain open.

| Task | Deliverable and acceptance |
|---|---|
| **KP-01: Library and review contracts** | V2 concept/rule/scorer schemas, registry references, versioning and review workflow; reject old/draft/unresolved records. Select a small expert-approved pilot of measurable relationships rather than force all layers to participate. |
| **KP-02: Evidence and applicability policies** | Define support/window units and permitted evidence per concept, optional deterministic definitions, unknown handling and context-only gates. Expert approves definitions and scorer meaning. |
| **KP-03: Applicability teacher pilot** | Select/evaluate frozen LLM/VLM and prompts; build offline evidence packages and versioned annotation cache. Measure expert-reviewed error/calibration/coverage and compare measurable baselines before admitting targets. |
| **KP-04: Frozen text and concept heads** | Select/pin text encoder, cache concept/rule embeddings and implement modality projections anchored by accepted annotations. Verify grounding and gradients without cross-source inputs. |
| **KP-05: Relationship scorers** | Implement reviewed order/compatibility contracts, or separately fit/validate/freeze a text-conditioned scorer. Establish meaningful contrasts, gradient paths and rule-swap sensitivity; unsupported associations remain inactive. |
| **KP-06: Objective integration** | Add optional independently weighted grounding/relation losses, eligibility masks, balanced normalization, bounded sampling and logging; keep existing objectives and disabled behavior. |
| **KP-07: Transfer and contract verification** | Cover §8 failure cases and strip auxiliary branches from supervised checkpoints/inference. Run real sensor-only transfer. |
| **KP-08: Controlled evaluation** | Execute §8 controls on spatial holdouts; report annotation cost, confidence validity, embedding quality and downstream yield. Document absence of benefit as an outcome. |

Operational choices to record during these tasks: pilot concepts/relations, text
encoder, applicability model and evidence format, annotation validation/calibration
policy, scorer type/contracts, spatial support, pairing strategy, weights and budget.
These are implementation decisions within this design, not claims resolved here.

## 10. Research context

[Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068) transfers
relations among representations rather than just individual outputs.
[Concept Bottleneck Models](https://arxiv.org/abs/2007.04612) ground learned
representations in annotated concepts; [Label-Free Concept Bottleneck Models](https://arxiv.org/abs/2304.06129)
explore construction without manually collected concept labels. These motivate
parts of this proposal. None establishes the validity of our agronomic rules,
LLM/VLM annotations, or this particular multimodal objective. Our concept branch
is auxiliary and does not impose a bottleneck on yield inference.

## 11. Relationship-generator prompt

**Prompt version:** `relationship_generator_v2.0`.
**Layer vocabulary snapshot:** `layer_inventory_2026-09-28`.

Copy the complete block below into the drafting model. It requests at most 30
relationship candidates, not sample annotations. Actual generator provenance is
recorded externally. Allocate globally unique IDs across imported batches and
retain identifiers through review. The old v1 prompt is superseded.

```text
You are drafting an agronomic relationship library for human expert review.
It will support soft constraints on multimodal sensor embeddings during
pretraining. Yield fine-tuning and inference use sensors only.

TASK
Generate up to 30 distinct, defensible relationship candidates. Each describes
concept A, concept B, their relationship, and conditions under which the
relationship is applicable. Encode no measurements from a particular field: you
receive only the inventory. Every candidate remains DRAFT.

Concept descriptions and the complete relationship will be encoded with a frozen
text encoder. Sensor representations will be grounded against the concepts using
validated observations or an optional frozen VLM/LLM applicability estimator.
A reviewed relation-specific scorer may then softly constrain the sensor
representations. Similar text vectors do not establish a physical relationship.
Your job is to draft definitions and qualifications, not invent its numerical loss.

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


RELATIONSHIP REQUIREMENTS
1. Use one principal relationship per candidate, with a concise complete text
   statement. Ground both endpoint concepts in observable properties from the
   inventory. Name units, depth, time and spatial support where material.
2. Use exactly two endpoint concepts for this first version. Each identifies a
   modality tag and its property references. Include any additional observed
   context sources in modalities. All tags must be relevant; require at least two
   source families. S2 resolution groups are one family; DEM and derived terrain
   do not provide independent source evidence.
3. State whether this is association, ordered_association, or compatibility.
   Choose direction a_to_b, b_to_a, or symmetric according to the described
   relation; directional association does not prove causation.
4. Do not infer that an association prohibits other outcomes. Do not translate
   generic co-occurrence into similarity of whole sensor embeddings. Do not
   invent compatibility tables, thresholds, probabilities, numerical effects,
   causal claims, model confidence, or scorer/loss code.
5. Define context and required evidence separately from the expected consequence.
   State unobservable conditions explicitly. A future applicability estimator
   must abstain when required conditions cannot be established; it must not infer
   context solely from whether the consequence already agrees with the statement.
6. Avoid globally undefined concepts such as high elevation without a reference
   context. Elevation alone does not determine a universal soil type. Respect
   region, crop/stage, soil depth, geometry, management and moisture qualifications.
7. Respect scales and observation dates. Coarse weather, soil and SMAP are not
   fine within-field measurements. Missing/invalid pixels are not concept absence.
   Old NAIP does not automatically describe current vegetation. No input or
   annotation may use measured yield or observations beyond the relevant cutoff.
8. Prefer relationships with measurable concepts and reviewable training meaning.
   General associations may be drafted but cannot train until a reviewed scoring
   contract exists. Do not invent such a contract: set scoring_contract_ref null.
9. Balance defensible terrain–soil–water, weather–water, environment–vegetation,
   optical–radar, crop-history and cross-source soil relationships. Do not force
   unsupported pairs or quotas. Distinguish mechanisms, associations and proxies;
   radar does not uniquely measure moisture and TWI is not observed current wetness.
10. Do not invent citations, field facts, unavailable management evidence, property
    identifiers, or PlanetScope bands/scales. property_refs are proposed inventory
    property references to resolve during review, not proof of registry validity.

OUTPUT
Return only JSONL, one object per line, without markdown or commentary. Use
exactly these keys and nested structures:

{"id":"agronomy_v2_0001","version":1,"text":"<qualified draft relationship>","modalities":["<tag A>","<tag B>"],"concept_a":{"id":"concept_v2_0001_a","description":"<observable concept A and reference context>","modality":"<tag A>","property_refs":["<inventory property reference>"]},"concept_b":{"id":"concept_v2_0001_b","description":"<observable concept B and reference context>","modality":"<tag B>","property_refs":["<inventory property reference>"]},"relation":{"type":"association","direction":"a_to_b","description":"<relationship meaning>","qualifications":["<material qualification>"]},"applicability":{"context":"<where and when applicable>","required_evidence":["<observable antecedent/context evidence>"],"unobservable_conditions":["<required condition not observed, if any>"],"estimator_policy_ref":null},"scoring_contract_ref":null,"review_status":"draft","reviewer_id":null,"reviewed_at":null,"generator":{"prompt_version":"relationship_generator_v2.0","layer_vocabulary":"layer_inventory_2026-09-28","model":null}}

Use sequential rule IDs starting agronomy_v2_0001. Reuse concept IDs only for
identical definitions; otherwise allocate distinct IDs. Lists can be empty when
there is no applicable item. Keep review fields draft/null, policy/scorer refs
null and generator model null. Produce fewer than 30 if defensibility requires it.
Check JSON validity, tag relevance, source-family diversity, observability,
qualifications and no fabricated numeric training targets before returning.
```

Validate schema/tags, resolve proposed property references, deduplicate concepts
and relationships, and submit all definitions and qualifications for expert review.
Prepare applicability/scoring policies separately before marking a rule training-ready.
Generation or a VLM/LLM applicability estimate never constitutes expert rule approval.
A separate sample-applicability prompt will be versioned and evaluated in KP-03;
this inventory-only generator prompt must not be reused to annotate sensor samples.
