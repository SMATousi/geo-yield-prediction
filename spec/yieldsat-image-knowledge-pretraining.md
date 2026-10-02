# YieldSAT image relational knowledge pretraining

**Status:** Design specification, 2026-10-01. No YieldSAT relationship library,
annotations, knowledge checkpoints, or knowledge-pretrained image model are
implemented by this document. This is the YieldSAT image specialization of
[expert-validated relationship pretraining](./knowledge_pretraining.md), and
precedes supervised training in the [YieldSAT image contract](./yieldsat-image-training.md).
The generic review, evidence, abstention, and scorer requirements remain in force.

## Objective and boundary

Transfer an expert-approved relation between **concept A** and **concept B**
from a frozen text/concept space to representations of **two observed sensor
modalities** that support those concepts. For example, a reviewed concept about
a rainfall regime can be paired with a reviewed concept about a subsequent
optical response, provided both are measurable at compatible time and spatial
support. This is a candidate pattern, not an approved agronomic rule. Both
sensor endpoints receive gradients when both encoders are trainable. There is no
hidden-sensor reconstruction conditioned on text, and no measured-yield target
in this pretraining objective.

Use the 64×64, full-coverage YieldSAT image dataset (`YieldSAT-Image-full`,
`--min-valid 1`), including per-cell `(24,12)` optical histories, weather
histories, static layers, timestamps, occupancy, and source provenance. “Full”
means all stored 24 slots, **not** 24 valid satellite acquisitions at every
cell. It covers the point corpus's recorded cells, not unlabeled pixels absent
from the original point corpus. The image builder's tile eligibility uses
finite yield, so this is *yield-free training on a label-selected corpus*, not
an independent unlabelled AOI sample. Neither `target` nor `valid_pixel`,
yield-derived statistics, or yield-based selection within a split may enter a
knowledge loss, concept annotation, or input feature. Use `source_row` only to
identify present source cells; it is not a feature.

The satellite [DINOv3 SAT-493M backbone](./yieldsat-image-training.md#inputs-and-separate-encoders)
and its cached RGB features remain frozen and unchanged. Pretrain and transfer
these **non-DINO image-model encoders**:

| Encoder in `YieldSATImageModel` | Sensor evidence | Knowledge support |
|---|---|---|
| `spec_enc` | Nine additional S2 bands on selected dated observations | Date-aware tile or local optical regions; masks per band. |
| `series_cell` + `series_enc` | All 12 S2 bands across 24 per-cell slots | Phenology/relative optical states over a reviewed time window; spatial regions after temporal encoding. |
| `weather_enc` | Four weather variables across 24 intervals | Field-season or declared time interval, never 4,096 independent 10 m weather labels. |
| `dem_enc` | Elevation | Spatial region at SRTM's effective support, not an asserted native 10 m measurement. |
| `terrain_enc` | Aspect, curvature, slope, TWI | Same SRTM-derived terrain family; unresolved scales block numeric concepts until audited. |
| `soil_cell` + `soil_enc` | Eight SoilGrids properties × six depths | Reviewed property and depth support; 250 m source resolution is not 10 m independent evidence. |

The generic relation task requires two distinct **sensor families**. `dem_enc`
and `terrain_enc` are separately trainable but share a source family, as do
`spec_enc` and `series_enc`; a pair within either family cannot masquerade as
independent cross-modal evidence. Such encoders can still receive other
knowledge pairs, grounding, and optional sensor self-supervision. The goal is
to give every non-DINO encoder an eligible training path; do not invent a rule
merely to give a stream a loss. Frozen DINO tokens may be evaluated as context
or a fixed reference, but no knowledge gradient or optimizer step may update
the DINO backbone/cache. Fusion, decoder, and yield heads are not trained by
the first knowledge stage. In particular, the planned S2 band-channel attention
and DEM–S2 cross-attention are **fusion fine-tuning modules only**, specified
in [YI-13–YI-14](./yieldsat-image-training.md#planned-fusion-contributions--2026-10-02).
No concept relation is applied to those attention weights, and no gradient from
the knowledge loss trains either module. A later fused knowledge objective
would need its own contract.

## YieldSAT concept and relation library

Maintain a versioned concept registry and expert-reviewed relationship JSONL.
Each record names two concept IDs, their **different sensor-family endpoints**,
the natural-language relationship, direction/type, qualifications, geographic
and crop/season scope, observation window, spatial support, required evidence,
abstention conditions, scorer version, and generator/reviewer provenance.
Concepts must resolve to exact [YieldSAT channels](../dataset/yieldsat_schema.py)
and units rather than generic words such as “soil” or “weather.” A relation
is training-ready only after a human expert approves both concepts, the
relation, its applicability policy, and its differentiable scoring contract.
Unreviewed drafts and generic US inventory statements are excluded.

Candidate **review topics**, not rules to train from yet:

| Pair of families | Candidate concept relationship to evaluate | Required qualification |
|---|---|---|
| Weather ↔ optical series | A reviewed precipitation/temperature regime and a later optical vegetation state or change. | Specify weather interval conversion, time lag, crop stage, valid S2 bands and clouds; do not claim rainfall alone caused the response. |
| Soil ↔ optical series | A reviewed soil-depth property regime and a seasonal optical response under a stated weather context. | Confirm property units/depth and coarse soil support; management and irrigation may make applicability unknown. |
| DEM/terrain ↔ soil | A reviewed topographic position and a soil-property regime within a justified local context. | No universal elevation→soil-type rule; confirm DEM/terrain semantics, geography and spatial support. |
| DEM/terrain ↔ optical series | A reviewed terrain regime and optical response conditional on crop/weather. | Confirm aspect/slope/TWI definitions; account for shadows and topographic artifacts. |

Terrain slope/TWI scales, soil uncertainty semantics, coordinates, and optical
BOA offset remain unresolved in the [YieldSAT data contract](./yieldsat_data_contract.md).
Do not use unresolved physical thresholds or uncertainty weights. Weather is
an inclusive-interval aggregate; its first dated interval has unknown start
and is invalid. Soil values are property-major across depths; static layers
are upsampled to the stored 10 m grid. All numerical definitions need explicit
country/crop reference populations fitted on training geography only.

## Observation units and applicability

An eligible training item is `(rule, field-season, tile or region, time window)`.
The two endpoints must refer to the same physical support or an explicitly
defined aggregation of it. Weather–optical relations should pool optical
evidence to a field/tile/time-window level compatible with field-centroid
weather. Soil–optical and terrain–optical relations may use local regions only
at the coarser source's effective footprint. Downweight repeated 10 m cells
from one 250 m soil cell or one weather field; balance by physical field,
season, country, crop, and rule. No random cell-level split across a field.

Apply the same observation cutoff to both endpoints. The `all_slots` policy is
retrospective; a prospective policy requires a cutoff known without actual
harvest outcome. Dates and weather intervals after cutoff are ineligible.
Missing/cloudy/undated S2 slots, padded cells and unsupported depth windows
abstain. The applicability gate must be set from reviewed context and sensor
quality **before** scoring the student relation; it may not switch off a
sample because the predicted relation is violated. Crops and country may
qualify a rule when known, but measured yield, yield maps, field-level yield,
and post-cutoff evidence are forbidden.

Ground individual concept presence with expert-labelled examples or a
validated deterministic estimator where a definition is measurable. An
offline frozen LLM/VLM may estimate concept applicability from permitted
sensor evidence under the [generic teacher contract](./knowledge_pretraining.md#4-applicability-estimation-vlmllm-as-an-optional-teacher).
Use it only after expert validation of calibration and abstention; every
generated **rule** is reviewed by a human expert, while sample estimates
remain uncertain and are audited on a stratified expert-reviewed subset.
No online teacher calls in training. Unknown is masked, not treated as absent.

## Relational distillation in embedding space

Cache frozen text embeddings `t_A`, `t_B` for the exact reviewed concept
descriptions and `t_R` for the full qualified relation, with encoder/tokenizer
revision and text hashes. A **frozen, expert-validated relation adapter** maps
the text triplet to a relation target `r_T` in a shared concept space. A pilot
may use the normalized displacement `normalize(t_B - t_A)` as a candidate
target, but a raw text-vector difference is **not** presumed to encode a
physical or causal relation. Validate it with approved positive, counterexample,
and relation-swap concept pairs; otherwise use a separately fitted and frozen
reviewed scorer. Association, ordering and compatibility need different
contracts. Preserve direction where the rule is directional; do not infer
causality from association.

For an eligible sensor pair, pool encoder features at the declared support,
then use small trainable modality projectors to put them in the same anchored
concept space:

```text
h_A = support_pool(E_A(x_A, mask_A, dates_A))
h_B = support_pool(E_B(x_B, mask_B, dates_B))
u_A = normalize(P_A(h_A));  u_B = normalize(P_B(h_B))
r_S = normalize(u_B - u_A)
L_relation = eligible_weight * (1 - dot(r_S, stop_gradient(r_T)))
L_total = L_sensor + λ_ground L_ground + λ_relation L_relation
```

`L_ground` anchors `u_A` and `u_B` to their frozen concept references using
independently accepted *soft* presence targets, known negatives, and abstention
masks. `L_sensor` is a documented masked sensor/reconstruction objective for
the trainable streams, or another validated information-preserving baseline;
the relation loss by itself can collapse encoders to concept constants. Where
an expert scorer encodes ordering/compatibility rather than displacement,
replace the displayed `L_relation` with that scorer's reviewed differentiable
loss; do not force a vector translation onto an unsuitable statement. Any
contrastive negatives must be expert-justified, not arbitrary other fields.
Targets, applicability weights and scorer parameters are detached; both
sensor encoders and their projectors receive gradients. Do not detach the
student score. Normalize loss by eligible *support weights* and balance active
rules to prevent a frequent field or easy concept from dominating. Zero
eligible pairs produce a finite zero relation loss and explicit skip counts.

The same sensor observations must substantiate concept presence; a projector
that always emits a concept prototype is a failure even if `L_relation` is
low. Audit sensor perturbation response, concept discrimination, embedding
variance, and within-rule exceptions. Do not interpret agreement with the
reviewed scorer as proof the agronomic rule is universally true.

## Checkpoint transfer and evaluation

Save sensor encoder weights separately from text/projectors/scorers with
explicit stream names and architecture hashes. The image model's DINO revision
and cache contract remain pinned, but its backbone is not overwritten. Transfer
the six non-DINO encoder groups above into the same v2 supervised image model;
initialize fusion, including S2 band-channel attention and DEM–S2
cross-attention when enabled, spatial decoder, level head and yield head by
the matched control protocol. The new attention modules are learned solely
from supervised yield fine-tuning; the knowledge checkpoint contains no
weights for them. At fine-tuning and inference, load **sensor inputs only**:
remove the text encoder, concept projectors, rules, applicability estimator,
and knowledge loss. A checkpoint must load and predict without access to any
knowledge artifact.

Use the existing grouped physical-field/farm/block/country folds. For the
primary inductive comparison, pretraining and annotation-policy fitting use
only training geography of each fold; validation/test field-seasons cannot
participate in sensor pretraining, even without yield labels. A separately
declared transductive experiment may use additional unlabeled geography but
must not be compared as the same protocol. Fit normalization, calibration,
reference populations and hyperparameters without held-out information.

Compare at identical data/splits, DINO cache, cutoff, seeds and supervised
label budgets and fusion configuration: image model from scratch except the common frozen DINO;
sensor self-supervision only; concept grounding only; grounding plus reviewed
relation distillation; and shuffled/incorrect relation assignments as a
diagnostic control. Include a no-text/constant-reference control, relation
and modality ablations, and optional validated teacher versus deterministic
annotations where both exist. Report per-rule eligibility/abstention,
grounding quality, relation scores on expert-reviewed held-out supports,
encoder gradient/variance checks, resource and annotation costs, and sensor-only
yield RMSE/MAE by field, country and crop. A gain in concept-head metrics
alone is not a successful transfer result.
Cross the knowledge/no-knowledge comparison with the four fusion settings in
YI-13–YI-14 so any gain is not attributed to relational pretraining when it
comes from the new supervised attention capacity.

## Implementation tasks and acceptance

1. **YIK-01 — Registry and review:** define YieldSAT-specific concept/rule
   schemas, exact channel/depth/unit references, support and cutoff policies;
   obtain expert approval for a small pilot library and every scorer. Resolve
   required source semantics before activating affected rules.
2. **YIK-02 — Evidence and annotations:** create target-free support records
   from image tensors, deterministic concept estimates where defensible, and
   an optional offline VLM/LLM pilot. Version provenance, audit abstention,
   and validate sample labels against expert review.
3. **YIK-03 — Text relation teacher:** pin and cache frozen concept/relation
   embeddings; validate the relation adapter/scorer on approved positive,
   negative and swapped examples before it supplies a gradient.
4. **YIK-04 — Sensor training:** attach disposable projectors to all eligible
   non-DINO encoders, implement support pooling, grounding, relation and
   sensor-preservation losses with masks, balanced sampling and gradient
   checks. Assert DINO weight/hash equality before and after pretraining.
5. **YIK-05 — Transfer and evaluation:** load only encoder weights into v2,
   verify text-free inference, and run the controlled grouped-split and
   low-label comparisons above. Publish failures and coverage, not only wins.

This feature is complete only when at least one expert-reviewed, supported
cross-family relation yields a nontrivial, well-grounded sensor training signal,
all active endpoints have verified gradients, DINO is unchanged, inference
needs no text, and downstream benefit or lack of benefit is measured on
held-out physical fields. The initial pilot need not claim a useful rule for
every stream; ungrounded streams remain untrained by relation loss and are
reported explicitly.
