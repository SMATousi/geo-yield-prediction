# YieldSAT image knowledge pretraining

Standalone implementation of [the knowledge design](../spec/yieldsat-image-knowledge-pretraining.md).
Implementation decisions, boundaries, and acceptance status are in
[the implementation spec](../spec/yieldsat-knowledge-implementation.md).
All existing repository source files remain unchanged. Run commands from the repository root.

This stage trains **only the existing non-DINO encoders**. Frozen text concepts
provide grounding and relational targets; they are discarded before yield
fine-tuning/inference. DINO, fusion, DEM–S2 attention, S2 band attention and yield
heads are outside the knowledge optimizer. The attention contributions remain
separate supervised implementation work; this package does not implement them.

## Environment and offline verification

Use the existing image-training environment, or install
`python -m pip install -r yieldsat_knowledge/requirements.txt` after selecting the
cluster's PyTorch build. No OpenRouter SDK is required. Set `PYTHON` to the
appropriate interpreter for the check script:

```bash
PYTHON=.conda/phase0/bin/python bash yieldsat_knowledge/scripts/check.sh
```

Tests use fabricated data, annotations and clearly marked synthetic expert
approvals; none is scientific approval. Tests never call an API or download a
model. Production training refuses mock text references.

## 1. Review the concept and rule library

The authored seed contains **12 concepts and 6 qualified cross-family rules**.
JSON and Markdown copies are in `assets/` and `../spec/yieldsat_knowledge_assets/`.
Every record starts as `draft`. Do not mark a record approved without actual
human expert review. Keep approved working copies outside the seed directories:

```bash
export RUN=/path/to/persistent/knowledge/fold0
mkdir -p "$RUN"
cp yieldsat_knowledge/assets/library.json "$RUN/library.reviewed.json"
python -m yieldsat_knowledge.validate --library "$RUN/library.reviewed.json"
```

Have the expert refine or remove unsupported concepts/rules, define geographic,
crop, depth and time support and required references, and populate each retained
record's `review` with `status: approved`, `reviewer`, `reviewed_at`, and `evidence`.
Review the **directed displacement scoring contract**, not merely the wording.
Numeric/reference-dependent concepts must remain unknown until those reference
populations and units are independently supported. The seed deliberately includes
hard cases; an API key does not make those concepts observable.

Optional model-assisted redrafting is available with `generate.py`:

```bash
python -m yieldsat_knowledge.generate --model "$OPENROUTER_MODEL" --output "$RUN/new-drafts.json"
# After inspecting the saved request, append --execute to send it.
```

This always resets generated reviews to draft. `make_library.py` reproduces both
committed seed copies; it is a maintainer operation, not a production pipeline
step (it would overwrite edits to those seed copies).

## 2. Prepare training-fold evidence and tensors

Use the v2 full-coverage image export built with `--min-valid 1`. Here “full
coverage” means retaining the recorded point corpus, including partially
occupied edge tiles; it does not mean every tile has 4,096 observed cells.
The existing mount `/home1/pupil/SMATousi/YieldSAT-Image` currently declares
`min_valid_pixels: 2048`; production preparation rejects it. The cluster's full
export path must be supplied explicitly.

```bash
export IMAGE_ROOT=/path/to/YieldSAT-Image-full
export ARTIFACT_ROOT=/path/to/yieldsat_artifacts
export SPLIT=pooled_field_s0
python -m yieldsat_knowledge.prepare \
  --image-root "$IMAGE_ROOT" --artifact-root "$ARTIFACT_ROOT" \
  --split "$ARTIFACT_ROOT/splits/$SPLIT.json" \
  --output-dir "$RUN/prepared" --k-obs 4 --window-days 120
```

Omit `--window-days` for a declared retrospective/all-slots experiment. The
prospective cutoff is days after known seeding, never days before actual harvest.
Preparation only selects the fold's training geography, rejects field leakage,
fits sensor normalization on that selection, and writes compressed tensors,
24-date RGB contact sheets, numeric evidence, hashes and manifests. It never
reads the HDF5 `target` or `valid_pixel` datasets. Source occupancy supplies masks.
The source image corpus is nevertheless label-selected upstream.

For a quick diagnostic use `--max-tiles 8`. `--allow-partial-corpus` explicitly
permits the older image export for a labelled legacy experiment; this deviation
is recorded. Do not report such an experiment as v2 full-corpus training.
Preparation is serial and writes a new directory; reserve disk for all normalized
image histories. A failed incomplete directory should be replaced with a new
output path, not used as a finished artifact.

## 3. Run LLM and VLM applicability pilots

Select an OpenRouter model/provider supporting JSON-schema structured outputs;
the VLM model must also support image input. No default model is prescribed.
Inject `OPENROUTER_API_KEY` through the cluster secret mechanism/environment, not
command arguments or source files. Only `--execute` makes a network request.

```bash
export OPENROUTER_MODEL=provider/model-id
python -m yieldsat_knowledge.pilot \
  --prepared "$RUN/prepared" --library "$RUN/library.reviewed.json" \
  --model "$OPENROUTER_MODEL" --output-dir "$RUN/pilot-llm" --max-samples 32
python -m yieldsat_knowledge.pilot \
  --prepared "$RUN/prepared" --library "$RUN/library.reviewed.json" \
  --model "$OPENROUTER_MODEL" --output-dir "$RUN/pilot-vlm" --max-samples 32 --vision
```

Inspect prompts, evidence and dry-run request metadata; append `--execute` to
run. Pilot selection cycles through country/crop strata with a reproducible
seed. Increase `--max-samples` later to annotate the desired training supports;
valid cache entries are reused. Keep different teacher profiles in different
folders. Set `--provider PROVIDER_SLUG` if provider-level reproducibility is
needed; profile hashes include it. Each response records actual returned model,
provider, usage and request ID when supplied by OpenRouter.

The LLM receives numeric evidence and dates. The VLM additionally receives a
fixed-stretch contact sheet with all 24 slots, including empty slots; it cannot
see soil/weather/elevation directly in RGB. Only allowlisted metadata and
sensors leave the machine. The prompts prohibit yield inference, invented
references, unsupported units, or scoring context from the desired consequence.
Unknown/not-observable scores must be null, not zero. Self-confidence is recorded
but never treated as calibrated reliability. Malformed/incomplete responses fail.

## 4. Expert-audit the annotation policy

Create independent expert labels following `templates/annotation_gold.json`.
Cover every retained concept AND rule with known positive and negative examples;
stratify geography, crop, dates, missingness and difficult cases. Do not copy
teacher outputs into the gold file. The executable minimum is a pilot gate, not
a sufficient sample-size claim for a paper.

```bash
python -m yieldsat_knowledge.audit annotations \
  --cache "$RUN/pilot-vlm" --gold "$RUN/annotation_gold.json" \
  --output "$RUN/annotation_audit.json"
```

A passing report needs at least 10 estimated labels per kind, MAE <= 0.25 per
kind, and positive/negative gold coverage for every active ID. Brier errors,
coverage and abstention counts are reported. Review failures by stratum and
inspect inappropriate confident estimates/abstentions before approving the
policy. The script creates `annotation_audit.json.policy.json` as a **draft**;
the expert must populate its `review`. Reliability is fixed at 1 for accepted
estimated scores, with presence/context probabilities retained as soft targets.
No calibrated self-confidence weighting is claimed. Annotation policy and report
hashes must match at training time. Choose the LLM or VLM profile using training
geography evidence; do not mix their responses in one cache.

## 5. Freeze and validate text relation geometry

Use the frozen CLIP text tower and its pretrained text projection. The default
checkpoint is `openai/clip-vit-base-patch32`; `--model` can select another compatible
CLIP checkpoint. Pin its Hugging Face repository to an immutable 40-character commit (or use a local snapshot and
record its matching revision). No remote model code is enabled.

```bash
export TEXT_MODEL=openai/clip-vit-base-patch32
# Set TEXT_REVISION to the approved immutable checkpoint commit.
python -m yieldsat_knowledge.text_cache \
  --library "$RUN/library.reviewed.json" --model "$TEXT_MODEL" \
  --revision "$TEXT_REVISION" --output-dir "$RUN/text" --device cpu
python -m yieldsat_knowledge.audit geometry \
  --library "$RUN/library.reviewed.json" --text-cache "$RUN/text" \
  --examples "$RUN/geometry_examples.json" --output "$RUN/geometry_audit.json"
```

CLIP uses its pooled end-of-text representation plus learned projection, followed
by L2 normalization; it does not use generic token mean pooling. Only the text
tower is instantiated. Text longer than the checkpoint context (77 tokens for
the default, including special tokens) fails with the affected IDs; shorten and
re-review it instead of silently truncating qualifications. Existing text caches
and geometry policies must be rebuilt for this encoder.

Follow `templates/geometry_examples.json`: each rule needs positive/counterexample
pairs in separate calibration and audit partitions. Audit pairs cannot be the
original rule endpoints or their trivial reversal. Add expert-reviewed alternate
concepts before freezing the library, if needed. Changing the library invalidates
text, annotation and audit caches. The scorer must achieve audit accuracy >= .75
for every active rule, followed by human review of the produced geometry policy.
The threshold diagnoses discrimination; the student optimizes cosine alignment
to the frozen direction. Failed geometry means **stop using that rule/scorer**;
this implementation does not fit an alternative learned relation adapter.

## 6. Pretrain non-DINO encoders

```bash
python -m yieldsat_knowledge.train \
  --prepared "$RUN/prepared" --library "$RUN/library.reviewed.json" \
  --text-cache "$RUN/text" --annotation-cache "$RUN/pilot-vlm" \
  --annotation-audit "$RUN/annotation_audit.json" \
  --annotation-policy "$RUN/annotation_audit.json.policy.json" \
  --geometry-audit "$RUN/geometry_audit.json" \
  --geometry-policy "$RUN/geometry_audit.json.policy.json" \
  --output-dir "$RUN/relation" --objective relation \
  --device cuda --amp --embed-dim 192 --batch-size 2 \
  --epochs 30 --steps-per-epoch 100 --seed 0
```

Start with a small real-data run to size GPU memory; the full 24-slot cell branch
is expensive. Training is single-process/single-GPU. Launch independent fold/seed
jobs using your cluster scheduler; no scheduler or credentials are bundled.
No network/API/text-encoder calls occur during training. Resume using the same
arguments plus `--resume "$RUN/relation/checkpoint_last.pth"`; `--epochs` may be
increased. Inputs/configuration must otherwise match exactly.

Outputs: `checkpoint_last.pth` (optimizer, RNG, disposable heads for resume),
`encoders.pth` (strict transfer weights plus normalization/provenance),
`provenance.json`, `report.json`, and JSON epoch metrics on stdout.

Matched controls use new output directories and the same sensors/seeds:
`--objective sensor`; `--objective grounding`; `--objective relation`; relation
with `--control shuffle_relations`; relation with `--control constant_text`.
Sensor-only control still requires the matching text-cache artifact for a common
model layout, but does not optimize grounding/relation losses or require expert
approval. All streams still get the masked sensor-preservation objective.

```bash
python -m yieldsat_knowledge.diagnostics \
  --checkpoint "$RUN/relation/checkpoint_last.pth" --prepared "$RUN/prepared" \
  --library "$RUN/library.reviewed.json" --annotation-cache "$RUN/pilot-vlm" \
  --annotation-audit "$RUN/annotation_audit.json" \
  --annotation-policy "$RUN/annotation_audit.json.policy.json" \
  --output "$RUN/relation/diagnostics.json" --device cuda
```

Diagnostics report concept Brier error, rule eligibility/unknowns by country/crop,
relation agreement, sample embedding variance and response to removing sensor
inputs. These are explicitly **training-support diagnostics**, not independent
evidence of generalization or agronomic truth.

## 7. Transfer into the existing image workflow

```bash
python -m yieldsat_knowledge.finetune \
  --knowledge-checkpoint "$RUN/relation/encoders.pth" \
  --image_root "$IMAGE_ROOT" --artifact_root "$ARTIFACT_ROOT" \
  --countries Argentina Brazil Germany Uruguay --split "$SPLIT" \
  --embed_dim 192 --k_obs 4 --series --slot_coverage present \
  --level_head --batch_size 2 --num_workers 0 \
  --output_dir "$RUN/finetune"
```

Supply the usual `--dino_cache`/revision flags if the existing cache is not at its
default location. This wrapper reuses the original trainer and output format,
forces exact all-stream encoder transfer, retains pretrained sensor statistics,
fits yield normalization only on supervised training labels, and enforces the
same known-seeding cutoff. Use matching fold, K and embedding dimension. Leave
`--cutoff_mode all_slots` (the wrapper owns masking); donor mode is unsupported.
Cached DINO slots containing any post-cutoff pixel are discarded because their
global self-attention cannot be repaired by a per-pixel output mask.

The resulting supervised model takes sensors and permitted crop/date context
only. It needs no text model, rules, annotations or API key. Preserve the wrapper's
normalizer and cutoff contract when constructing inference data. Do not directly
use the original warm-start loader with a fresh normalizer, as that silently
changes the input distribution.

Downstream evaluation remains the existing grouped-fold workflow. Match frozen
DINO cache, splits, seeds, label budget and fusion settings across controls;
report field/country/crop yield metrics and failures as well as improvements.
Low-label jobs must restrict supervised labels without changing the pretraining
fold or admitting held-out fields. New fusion attention variants and independent
expert-held-out knowledge evaluation are follow-up experiments, not completed
results of this implementation.
