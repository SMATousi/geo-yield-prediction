# YieldSAT image knowledge pretraining: isolated implementation

**Date:** 2026-10-02. **Scope:** executable preparation, pilot, audit, pretraining,
diagnostics and supervised-transfer tooling. This is an implementation companion
to [the knowledge design](yieldsat-image-knowledge-pretraining.md), not a claim
that expert validation or scientific training has been completed.

The user requested new scripts and a separate specification, without modifying
existing code. All implementation lives in `yieldsat_knowledge/`; existing source
and specifications are untouched. Existing encoder classes, image packing,
channel registry and supervised trainer are imported. Credentials, cluster
submission files, expert approvals and production run artifacts are supplied
later. The runnable command sequence is in
[the package runbook](../yieldsat_knowledge/README.md).

## Implementation sequence and deliverables

| Step | Design task | Delivered files and behavior |
|---|---|---|
| 1 | YIK-01 registry/review | `common.py`, `make_library.py`, `validate.py`; 12 draft concepts and 6 draft relationships, exact channel ownership, distinct-family validation and human approval records. |
| 2 | YIK-02 evidence | `prepare.py`; grouped training-fold selection, target-free reader, normalization, temporal masks, 24-slot preview/evidence and artifact integrity hashes. |
| 3 | YIK-02 pilot | `pilot.py`, `generate.py`, three Markdown prompts; OpenRouter LLM/VLM dry-runs, bounded live calls, resumable response caches and strict response validation. |
| 4 | YIK-02/03 review gates | `audit.py`, gold/geometry JSON templates; independent expert annotation audit, candidate text-geometry calibration/audit, draft approval policies. |
| 5 | YIK-03 frozen references | `text_cache.py`; pinned text model, concept and qualified rule texts, frozen vectors, version/hash provenance. |
| 6 | YIK-04 training | `model.py`, `train.py`; six non-DINO streams, grounding, relational KD, sensor preservation, hierarchical sampler, controls, resume and exact encoder exports. |
| 7 | YIK-05 compatibility | `finetune.py`; new wrapper around the existing image trainer, strict transfer, normalization/cutoff preservation and sensor-only prediction. |
| 8 | YIK-04/05 diagnostics | `diagnostics.py`, integration tests and `scripts/check.sh`; encoder gradients, corruption sensitivity, variance, concept errors, rule coverage and offline transfer checks. |
| 9 | Handoff | package README/requirements, this spec, mirrored JSON/Markdown library; scheduler-neutral CLI commands and explicit pending scientific work. |

The generated library is stored identically at:

- `yieldsat_knowledge/assets/library.json` and `library.md`.
- `spec/yieldsat_knowledge_assets/library.json` and `library.md`.

The JSON is authoritative; Markdown is the human-readable review summary.
These are assistant-authored candidates, not externally generated or expert
approved. Optional OpenRouter redrafting cannot approve its own outputs.

## Data and information boundary

Accept canonical image HDF5 arrays: `temporal[64,64,24,16]`,
`static[64,64,104]`, `times[64,64,24]`, and `source_row[64,64]`.
Channel ordering must match `dataset/yieldsat_schema.py`. The preparation reader
never opens `target` or `valid_pixel`. Dummy arrays exist only to satisfy the
legacy packer's return interface and are removed before serialization.
Coordinates and soil uncertainty channels never enter concept evidence or a
trainable sensor stream. IDs/country/crop are evidence provenance/context and
sampling keys, not knowledge-model tensors.

The full-corpus v2 export has `min_valid_pixels: 1` (`--min-valid 1` when built).
“100% coverage” means coverage of the recorded point corpus, not 100% occupancy
or cloud-free acquisition at every image position/date. The currently mounted
`/home1/pupil/SMATousi/YieldSAT-Image` is the older `min_valid_pixels: 2048` export.
Production preparation refuses that export unless the explicit
`--allow-partial-corpus` diagnostic deviation is recorded in its manifest.
No new image corpus was built by this task.

Use only the fold's training physical fields. Season partitions must be disjoint;
physical fields shared with validation/test cause an error even in different
years. Fit sensor statistics on selected training tiles, with finite/occupied
and temporal validity masks. No yield-based filtering, normalization, or
applicability is added. The upstream image corpus remains label-selected, so
reports must say **yield-free pretraining on a label-selected corpus**.

`--window-days N` clips both sensor endpoints at seeding+N, without actual
harvest metadata in the evidence. Omitting it declares retrospective all-slots
training. All 24 storage slots remain represented; slots after cutoff are
masked. The original packer's series dates are explicitly zeroed where optical
observations are absent. A separate per-band series mask prevents reconstruction
from scoring a missing band as an observed zero. Weather's first interval is
invalid. Inclusive intervals and unknown physical conversions are exposed in
the evidence; summing overlapping interval rainfall is forbidden by the prompt.

| Student stream | Stored student tensor | Encoder state prefixes |
|---|---|---|
| Additional S2 bands | `[K,9,64,64]`, dated selected observations, band masks | `spec_enc` |
| Full optical series | `[64,64,24,12]`, slot mask/days, preservation-only band mask | `series_cell`, `series_enc` |
| Weather | `[24,4]` plus validity and three time features | `weather_enc` |
| Elevation | `[1,64,64]` plus validity | `dem_enc` |
| Terrain | `[5,64,64]`: sine/cosine aspect, curvature, slope, TWI | `terrain_enc` |
| Soil | `[48,64,64]`: eight properties × six depths | `soil_cell`, `soil_enc` |

Series/storage dimensions follow the current cell-major implementation, not an
older channel-major sketch. RGB DINO, fusion, decoder and yield heads are not
instantiated at pretraining time. This makes DINO mutation impossible within
this stage rather than merely omitting DINO from an optimizer parameter list.

## Candidate knowledge and teacher contract

The seed covers weather–optical, soil–optical, elevation–soil and
terrain–weather associations. It expressly rejects a universal altitude→soil
rule. Weather coarse support, irrigation/management, crop stage, units, local
reference distributions and aspect semantics can make candidates unobservable.
The expert may delete unsupported candidates; every non-DINO stream can still
receive the sensor objective without inventing a knowledge rule for it.

Each concept names exact channels, stream/family, description, support/window,
scope, unit provenance, reference requirements, abstention conditions and review.
Each rule names two concepts, qualified association/direction, context/window,
minimum presence, scorer version and review. Optical residual/series share a
family; DEM/terrain share a family. These pairs are not counted as independent
cross-family relationships.

The JSON library uses one versioned document containing concept/rule arrays,
rather than two JSONL registries. Evidence uses JSONL; one immutable response
file is used per teacher request. This keeps the requested JSON/Markdown mirror
simple without weakening record-level review or provenance.

OpenRouter uses `POST /api/v1/chat/completions`, JSON schema for annotations,
`require_parameters: true`, and disabled provider fallbacks. `--provider`
optionally fixes provider order. The API key comes only from the environment.
Model ID is explicitly selected by the operator; returned model/provider/usage
are retained. Request profiles include prompt, library, model/provider, vision,
token budget and temperature; evidence has a separate content hash. Existing
cache entries are validated before reuse. HTTP 429/selected 5xx/network errors
receive bounded retries; invalid model output is not quietly repaired.

No API calls happen without `--execute`. LLM and VLM pilots use separate caches;
country/crop strata are cycled reproducibly. The VLM receives one fixed-stretch
24-slot RGB contact sheet and the same numeric evidence, not invented views of
unobservable modalities. Concepts cite their own channels. Rules estimate
context applicability independently of student scores. Unknown/not-observable
must have null presence/context and confidence scores. Self-confidence is
recorded but not used as calibrated reliability.

Official API contracts used for the client:
[OpenRouter quickstart](https://openrouter.ai/docs/quickstart) and
[structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs).
Live provider behavior and the selected model still need a credentialed pilot.

## Expert gates and frozen relation targets

All active concepts/rules require a review record with reviewer/date/evidence.
Teacher policy auditing uses independent expert labels from training geography.
The initial executable gate requires at least 10 estimated labels for each kind
(concepts/rules), mean absolute error <= .25 and positive/negative coverage per
active ID. It reports Brier error, coverage and abstentions. Human approval must
also assess abstention suitability, unsupported confidence, stratum coverage and
sample size; this small aggregate gate is not a calibration guarantee.

Estimated targets receive fixed reliability 1. Their soft presence/applicability
values supply the weighting, not teacher confidence. Unknown values are masked.
The initial implementation does not invent deterministic physical thresholds or
unit conversions. Those can only follow a separate semantics/reference audit.

The frozen text cache encodes both concept descriptions and qualified rule
statements using frozen `CLIPTextModelWithProjection`, default checkpoint
`openai/clip-vit-base-patch32`, and a pinned 40-character model revision. Use
CLIP's pooled end-of-text output and pretrained text projection, followed by L2
normalization. Only the text tower is loaded. Respect the checkpoint context
length (77 tokens including special tokens for the default); reject overlong
text with record IDs rather than silently truncating rule qualifications. Cache
metadata records the encoder class, pooling, dimension and context limit. Old
mean-pooled caches and geometry policies must be rebuilt and revalidated. See
[the CLIP API](https://huggingface.co/docs/transformers/model_doc/clip). Random mock vectors are only
for offline smoke tests and are rejected by production training.

Only `directed_displacement_v1` is implemented:

```text
tA,tB,tR = frozen cached text vectors
rTeacher = normalize(tB - tA)
uA = normalize(PA(pool(EA(sensorA))))
uB = normalize(PB(pool(EB(sensorB))))
lossRelation = 1 - dot(normalize(uB-uA), stop_gradient(rTeacher))
```

`tR` is retained for exact semantic provenance but does not enter the v1 vector
formula. The candidate direction is not presumed to be a physical law. An
expert supplies positive/counterexample concept pairs in calibration and audit
partitions. Audit pairs must be independent alternate pairs, not just the
original endpoint pair or its reversal. Per-rule calibration finds a cosine
threshold; each rule must achieve at least .75 accuracy on the separate audit
pairs, then receive scorer-policy approval. Thresholds diagnose the candidate
geometry; training minimizes continuous directional disagreement. No learned
adapter fallback is implemented. Failed geometry blocks relational training.

## Training and preservation objectives

Original non-DINO encoder classes are instantiated with their original state
names. Each stream has a disposable linear projector into the text dimension.
Spatial tokens are pooled with observed-area weights to a tile-level support;
weather already has field/tile temporal support. No 10 m cell is treated as an
independent coarse-weather or soil annotation.

Soft grounding uses BCE on cosine-to-concept logits divided by .1, with unknowns
and entirely missing endpoints masked. Relational weights are detached products
of endpoint presences and rule context. Both presences must exceed the reviewed
rule threshold; context must be positive. Normalize within each active rule,
then average active rules. Both student endpoints receive gradients. Zero
eligible batches have finite zero knowledge loss; an entire epoch with no
requested grounding/relation terms raises an error instead of claiming success.

Sensor preservation randomly hides observed entries (default .2) and predicts
withheld standardized channel means over 4×4 spatial token regions; weather
predicts withheld temporal means. Series means are weighted by genuinely
observed withheld band values. This is deliberately a coarse auxiliary loss,
**not** a claim to reconstruct every raster pixel or the full temporal sequence.
All six streams receive this loss. Projectors/reconstruction heads are discarded
at transfer. Their auxiliary objectives reduce collapse incentives but do not
prove useful representations; diagnostics and downstream controls remain needed.

Sampling gives equal hierarchical mass to country, crop, physical field, season,
then tiles. This limits repeated coarse-footprint dominance. It is not explicit
SoilGrids footprint deduplication, because such footprint IDs are unavailable.
Rule losses are separately balanced among active rules. Training is one process
on one GPU (or CPU for smoke), with optional CUDA bfloat16 autocast, clipped
AdamW, deterministic seeds, and checkpointed sampler/PyTorch/CUDA RNG states.
Inputs, policies, text references, normalizer, split and encoder shapes are
hashed or recorded; incompatible resume is rejected. No API/network operation
or text-model inference occurs inside a training step.

## Transfer and evaluation boundary

The exported `model` dictionary contains only the eight listed state prefixes,
representing six streams. `finetune.py` requires the same split/export manifest,
K and embedding dimension and exact encoder key/shape agreement. It reuses the
existing image trainer through process-local adapters; no old source file is
edited. All S2+ADM streams and series are required for this initial transfer.
The wrapper keeps sensor normalization from pretraining and fits target
normalization only from supervised training labels. It preserves known-seeding
cutoffs and drops an entire cached DINO slot if any original pixel is post-cutoff,
since DINO self-attention can mix that pixel into all cached token positions.

The final image model receives sensor/date/crop inputs only. Normalization and
cutoff metadata must accompany inference; text/rules/pilots do not. Supervised
fusion and heads are initialized/trained by the existing workflow. The proposed
S2 channel attention and elevation–S2 cross-attention remain fusion-stage work;
this task supplies no pretraining weights or implementation for them.

Executable pretraining controls: sensor only, grounding, grounding+relation,
shuffled relation directions, and constant text references. Diagnostics report
concept errors, country/crop/rule eligibility, relation agreement, sample
embedding variance and sensor-removal response. They are labelled as training
support diagnostics, not held-out knowledge evaluation. Existing supervised
fold evaluation supplies prediction artifacts and yield metrics. Independent
expert-held-out evaluations, low-label/fusion-factorial experiments and scientific
claims must be run after expert/provider validation. No downstream gain is
claimed by shipping scripts.

## Verification and remaining execution

Offline integration covers target-free preparation on HDF5 files with no yield
datasets, temporal/occupancy masks, mirrored draft refusal, malformed teacher
responses, dry-run/no-network/cache behavior, unbalanced expert gold rejection,
non-DINO gradient flow, training/resume, strict encoder transfer, preservation of
normalization/cutoffs, text-free model forward and collapse/coverage diagnostics.
The offline suite contains 15 passing tests; all nine command entry points pass
`--help` checks. Tests ran with PyTorch 2.5.1 on CPU. CUDA was not available
to this process, so GPU/AMP execution remains a cluster verification step.
CLIP-specific tests use a tiny offline random CLIP text model to check frozen
projected outputs, normalization and rejection of overlong text. No pretrained
CLIP weights were downloaded or production embeddings generated in this check.
Synthetic approvals are explicitly test-only and never copied into production
assets. All test artifacts are temporary.

A real two-tile preparation using `pooled_field_s0`, a 120-day cutoff and the
explicit older-corpus override succeeded. A two-support VLM dry-run generated
requests without an API call. These verify compatibility with the mounted data,
not scientific validity or full-corpus coverage. No API key, model download,
production knowledge training or cluster job was used.

Before production execution:

1. Provide the full v2 image export path and the intended grouped fold/cutoff.
2. Obtain actual expert review; remove unobservable rules or supply audited
   units/reference context. Add alternate concepts for geometry validation.
3. Select/pin the text model and OpenRouter model/provider; supply credentials
   through environment secrets; run the pilot and independent expert audits.
4. Review/sign only passing annotation and geometry policies; regenerate caches
   whenever the library, evidence, teacher profile or text model changes.
5. Size GPU/storage with a short run, then use the supplied CLIs in the user's
   cluster submission scripts for matched controls, folds and seeds.
6. Run supervised transfer and independent evaluation; report unsupported rules,
   abstention, representation failure and lack of downstream benefit honestly.
