# JEPA-style pretraining for the YieldSAT P3-NBR point model

**Status:** Design and experiment proposal, 2026-10-06. The user requested that
these options be documented, committed and pushed. This document does not select
a winning objective, implement training, launch jobs or authorize corpus scale-up.

Related specifications:
- [Point-model improvement and P3-NBR](yieldsat-improvement.md)
- [Dense temporal series](yieldsat-dense-series.md)
- [Point knowledge pretraining](yieldsat-point-knowledge-pretraining.md)
- [Pretraining success criteria](pretraining/success-criteria.md)
- [National US pretraining corpus](yieldsat-us-national-pretraining.md)

## Motivation and architecture boundary

P3-NBR retains the point model's separate sensor encoders and
`perceiver_summary` fusion. Its additional `yieldsat_s2_nbr` stream is the masked
mean of the 12 S2 bands in a **5×5 neighborhood, excluding the center**, for each
time slot. In the YieldSAT builder, neighbors belong to the same field season;
outside-field and missing observations do not contribute. This is an aggregated
point stream, not an image backbone or a representation of individual neighbors.
The dense variant supports 72 slots instead of the standard 24-slot frame.

The round-2 specification reports P3-NBR DEV pixel/field R² of 0.332/0.497.
Those results support keeping its architecture as the starting point, not a claim
that JEPA will improve it. The subsequent point-pretraining study did not establish
a DEV yield gain from its reconstruction SSL or knowledge objectives. The new
experiments must therefore compare against both scratch and existing SSL controls.

JEPA predicts **representations of withheld observations**, rather than their
raw values. The motivating reference is
[I-JEPA](https://arxiv.org/html/2301.08243v3): a context encoder and predictor
estimate target representations supplied by a slowly updated target encoder.
The agricultural objectives below are proposed adaptations, not published results
on YieldSAT. Point sequences do not require an image backbone to use this pattern.

## Common pretraining design

Let `C` be visible sensor context, `Y` an observed withheld target, and `q`
the target query (modality, time interval, and spatial role when applicable):

```text
student/context encoder E_theta(C)
    -> predictor P_phi(..., q)
    -> predicted target representation

target encoder E_xi(Y)
    -> stop-gradient target representation
```

Proposed loss:

`L_JEPA = weighted_mean_valid d(P_phi(E_theta(C), q), stopgrad(E_xi(Y)))`

Use a normalized embedding distance or normalized smooth-L1 loss, selected in a
small pilot and held fixed for comparisons. Update the target encoder by an
exponential moving average of the compatible student parameters; the EMA schedule,
target normalization and target support must be logged. The target branch has no
optimizer gradients and deterministic evaluation behavior. Initialize compatible
student/teacher components consistently.

Mask raw student values **and their input validity representation before encoding**.
Masking after a bidirectional temporal encoder would leave target information in
visible embeddings. Keep separate natural-observation validity, artificial hiding,
and target-loss validity. Score only actual valid withheld evidence. Unobserved
cloudy slots are not pseudo-targets. Query dates identify where to predict, but
must not include withheld values, derived indices or statistics computed from them.

Prefer targets anchored to a specified modality and temporal block. For the first
pilot, encode the observed target block in the teacher and predict its normalized
representation, or its valid per-slot tokens. Do not silently switch between
block-only and full-season-contextualized teacher targets. For strict future
prediction, the target encoder must not incorporate observations later than its
declared target interval. A fused full-input summary is a separate ablation:
shared static information could dominate it and make missing dynamics unnecessary.

`MaskedTemporalEncoder` already supports `output='tokens'` as well as summary
output. Expose the required target/student features through a pretraining-specific
path while preserving P3-NBR's summary-based supervised architecture and checkpoint
compatibility. Do not silently substitute the `perceiver_tokens` downstream model.

The student sensor encoders receive gradients through visible context; Perceiver
fusion may be pretrained when it participates in the predictor's context path,
as permitted by the point-pretraining contract. Ensure every intended trainable
stream is visible in some training tasks: an EMA-only target stream does not
receive learning gradients just because it supplies targets. Record which
parameters actually receive gradients and transfer.

Transfer compatible student sensor/fusion weights to yield fine-tuning. Discard
JEPA predictors and target encoders, and initialize the yield head independently.
No yield labels, crop-label shortcut tokens or text inputs are needed for JEPA.
The downstream sensor model retains its normal inference requirements. DINO and
image-model fusion additions are outside this point-model proposal.

## Candidate objectives and tradeoffs

| Option | Context and target | Expected benefit to test | Main drawbacks |
|---|---|---|---|
| **J1: Temporal block prediction** | Remaining S2 history plus allowed other sensors → a withheld contiguous S2 interval's embedding | Seasonal representations and tolerance of missing observations; closest fit to current encoders/data | Can become interpolation or calendar recognition; abstract targets may lose subtle yield-relevant variation |
| **J2: Weather-conditioned future prediction** | Earlier center/neighborhood optical history, site properties and permitted weather → a later optical interval's embedding | Crop development under changing conditions; potential early-prediction and year-transfer benefit | Unobserved management and disturbances make outcomes ambiguous; conditional averages and future-weather leakage are risks |
| **J3: Cross-modal prediction** | Available sensor families → a withheld family's embedding, rotating roles | Multimodal relationships, fusion learning and robustness to missing modalities | Some information is not predictable across modalities; shared information may dominate while unique useful detail is lost |
| **J4: Spatial context prediction** | Neighborhood history/context → center embedding, or center/context → surrounding-region embedding | Directly exercises the useful P3-NBR neighborhood stream | Spectral autocorrelation and shared native pixels make tasks trivial; oversmoothing may remove within-field variation |
| **J5: Cross-year conditioned prediction** | Another year's history at a revisited location plus target-year conditions → target-year optical embedding | Separates persistent site properties from annual response; exploits shared locations | Rotation/management ambiguity, location memorization, difficult time alignment and additional data requirements |

Benefits in this table are hypotheses. Lower latent prediction error is not proof
of useful agronomic information or improved yield prediction.

### J1 — Recommended first experiment

Hide contiguous blocks within the observed season, varying block location and
size while retaining informative context. A proposed initial masking range is
20–40% of valid optical observations, subject to a minimum context/target count
established in the pilot. This is a proposal, not a fixed production setting.
Sample by valid observations and elapsed time, not simply a fixed number of array
slots: a 24-slot frame can contain far fewer usable observations.

For this temporal task, hide the matching interval in **both center S2 and
neighborhood S2**, including all duplicate optical channels/derived features.
Leaving same-time neighborhood spectra visible can turn temporal prediction into
a spatial-copy shortcut. Weather and static evidence may remain visible under a
declared task policy. Compare against temporal interpolation, calendar-only and
static-context-only controls. This is bidirectional context completion, not an
early-season forecast; report it accordingly.

### J2 — Recommended extension

Predict a later S2 interval from earlier observations and site properties.
Separate two clearly named protocols:

- **J2-F, strict forecasting:** the student sees only evidence available at or
  before the prediction cutoff. Future realized weather and future neighbor
  observations are forbidden; any actual forecast product needs its issue time.
- **J2-W, weather-conditioned response prediction:** observed weather through the
  target interval is allowed, but future optical observations remain hidden.
  This is useful conditional pretraining, not an operational forecast with
  unknown future weather and not evidence of causal effects.

Use multiple elapsed-time horizons and actual date/interval metadata. For both
protocols, remove later optical values from every student stream before encoding.
Do not calculate input normalization or seasonal features from withheld future
observations. Compare against persistence and calendar/site/weather-only controls.
Consider uncertainty-aware predictors only after a deterministic baseline is
established; an average embedding can hide genuinely different plausible outcomes.

### J3 — Complementary objective

Example: weather, soil, terrain and visible early optical context predict a
withheld optical representation. Rotate source/target roles where there is a
scientifically plausible shared signal, and use modality-specific predictors or
queries. Balance losses per family so input width and number of valid slots do
not decide which encoder is trained most strongly.

Hiding the center S2 stream while retaining its same-time neighborhood is not a
clean optical-family-withheld task. Hide both for that experiment, or explicitly
name and evaluate it as spatial conditional prediction. Likewise, DEM and terrain
share source information; their prediction is not independent cross-family evidence.
Validate physical-information retention with probes. Use J3 with J1 before relying
on J3 alone, since unique sensor details may not be cross-modally predictable.

### J4 — Deferred spatial experiment

P3-NBR's neighborhood is an aggregate, so predicting individual spatial target
positions would require new neighborhood inputs rather than a new loss alone.
Initially compare center/aggregate targets using existing streams. The spatial
objective intentionally permits same-time neighbors, unlike J1/J2.

Center exclusion alone does not ensure independent evidence: a 20/60 m source
pixel can contribute to both center and neighbors on a 10 m grid. Audit native
footprints, use simple neighbor-copy controls, and consider larger/disjoint support
or selected native-resolution bands. Do not force nearby embeddings to be equal;
retain the predictor and monitor within-field variation and yield residuals.

For national point data, compute neighborhood summaries from actual adjacent
source-raster cells. Do not use arbitrary nearby sampled records as if they were
a complete 5×5 grid. Store contributor counts, masks and date rules. Points do
not supply field boundaries, so crop/region boundary policy needs explicit design
and a parity test against the YieldSAT same-field-season convention. Neighborhood
support must also respect evaluation exclusions. No stored image tiles are
required merely to retain the resulting point-level summary stream.

### J5 — Deferred cross-year experiment

Use persistent location IDs and pair only eligible, permitted years. Align by
actual dates and a declared season/phenology reference, handling winter crops,
rotations and calendar-estimated dates. Keep target-year weather conditioning
separate from strict forecasting claims.

Predict the target-year representation rather than forcing two years' embeddings
to be identical. Compare against location/static-only and year-shuffled controls,
report crop-transition subsets, and split held-out locations across all years.
Do not allow the shared cohort or static geography to bypass evaluation exclusions.

## Combining JEPA and relational knowledge

First evaluate JEPA without knowledge losses. Later compare matched arms:

`L = L_JEPA + lambda_ground * L_ground + lambda_relation * L_relation`

Optionally retaining raw reconstruction is another explicit ablation, not an
unreported default. Existing approved relations, applicability masks, reference
populations and frozen CLIP concept embeddings remain governed by the point
knowledge specification. JEPA targets are learned **sensor** embeddings; they
are not CLIP text embeddings. Text does not enter the yield inference path.

Keep JEPA-only and JEPA-plus-knowledge results separate. Intrinsic relation learning
previously did not establish yield benefit, so adding that objective cannot be
assumed to improve JEPA. Check gradient balance and sensor-information retention.

## Experiments, metrics and decision rules

Recommended first comparisons on the same P3-NBR architecture and data:

| Arm | Pretraining |
|---|---|
| J0 | None: scratch P3-NBR |
| JR | Existing masked-value reconstruction/forecast SSL, compute matched |
| J1 | Temporal block JEPA |
| J12 | J1 plus J2-W; J2-F is a separately labeled forecasting arm |
| J13 | J1 plus cross-modal JEPA, after the simpler comparison |
| JK | Selected JEPA arm plus approved relational knowledge, later |

Use the same sensor inputs, neighborhood preprocessing, normalization, fold
exclusions, label budgets and fine-tuning recipe. Preserve batches spanning many
field seasons; earlier tile-batched hybrids underperformed the point pipeline.
On clustered US data, also spread batches across clusters/native weather cells.
Compare 24-slot and 72-slot variants separately, with controls at the same slot
count. Changing temporal density is not evidence for the JEPA objective itself.
Do not transfer learned slot embeddings between different time grids without an
explicit mapping and parity test.

Match and report actual pretraining compute (including EMA teacher forward passes),
examples, steps and wall/GPU time; equal optimizer steps alone are not necessarily
compute matched. Evaluate multiple seeds using paired folds and a predeclared
confidence-interval procedure. Fit preprocessing and select hyperparameters on
training/development data. If reproducing the repository's paper protocol with
test-fold checkpoint selection, label that result explicitly and keep it separate
from an untouched-test generalization claim.

Required health/information checks:

- Per-stream and fused embedding variance/effective rank, before and after
  projection, plus constant-output baselines. EMA and stop-gradient are not a
  mathematical guarantee against collapse.
- Valid target counts and loss by modality, season, country/crop, cloud coverage
  and prediction horizon; no loss on naturally missing targets.
- Calendar-only, static/site-only, persistence/interpolation and neighborhood-copy
  controls, selected according to the objective.
- Frozen-embedding probes for optical state, weather, soil and terrain using
  training-only fits and held-out seasons; check preservation of within-field
  variation. Retain both ordinary and robust diagnostic loss summaries rather
  than concealing extreme-input behavior.
- End-to-end downstream pixel and field R², especially LOYO/LORO, together with
  10%/25% label-budget comparisons where feasible. Latent loss alone selects no winner.

The existing [success criteria](pretraining/success-criteria.md) remain the project
reference. P2/I3-style collapse and information-retention checks are relevant to
JEPA; concept/rule criteria apply only to knowledge-enabled arms. Before launching,
register a JEPA-specific success table with a positive paired downstream gain over
scratch and compute-matched JR, a predefined field-performance non-regression
margin, and per-protocol harm limits. Numerical JEPA thresholds require an explicit
experiment decision; this document does not quietly redefine existing criteria.

Do not scale to 100M points based on a falling JEPA loss. The national corpus's
staged acquisition and evidence gates remain in force; JEPA-only adaptations of
knowledge-specific gates need an explicit decision before being used for scale-up.

## Implementation tasks after objective selection

| Task | Deliverable and acceptance |
|---|---|
| JP-01 | Freeze first arms, dataset/time grid, target support, mask distributions, objective weights and JEPA success thresholds |
| JP-02 | Expose student/target features with checkpoint-compatible P3-NBR summaries; test clean transfer into the supervised model |
| JP-03 | Implement EMA teacher, predictor, masked loss and resumable state; verify stopped teacher gradients, online gradients and deterministic targets |
| JP-04 | Implement coupled center/neighbor masking and cutoff rules; tests must show that perturbing withheld values cannot alter student context |
| JP-05 | Audit neighborhood/date/native-pixel support, natural masks, teacher target validity and fold exclusions; define US-neighborhood parity before use |
| JP-06 | Run small smoke/health pilots and shortcut controls; measure memory and teacher compute before allocating a training suite |
| JP-07 | Run J0/JR/J1 paired DEV comparisons, followed by J12 if justified; publish failures as well as improvements |
| JP-08 | Evaluate J13/JK or deferred spatial/cross-year tasks only as identifiable additional experiments |

Open choices are objective selection, target pooling/token support, mask/horizon
settings, teacher/loss hyperparameters, first 24/72-slot data view, and final
numerical decision thresholds. The recommended starting point is **J1**, followed
by **J2-W**, with cross-modal, knowledge, spatial and cross-year additions tested
separately. No implementation changes accompany this specification.
