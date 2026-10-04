# YieldSAT pooled foundation model with knowledge pretraining

**Status:** specified 2026-10-03 (user request). Not implemented. Builds on
[yieldsat-improvement.md](./yieldsat-improvement.md) (DEV subset, pooled metric,
round-2 point-model features) and the knowledge-pretraining design in
[knowledge_pretraining.md](./knowledge_pretraining.md),
[yieldsat-image-knowledge-pretraining.md](./yieldsat-image-knowledge-pretraining.md)
and [yieldsat-knowledge-implementation.md](./yieldsat-knowledge-implementation.md).

## 1. Goal

Replace the nine country–crop models with **one model per fold, trained on all
countries and crops together**. The model is pretrained in two stages:
1. sensor self-supervised learning;
2. **expert-validated knowledge pretraining**, the user's main contribution.

It is then fine-tuned on yield.

The results must answer:
1. Does pooling across pairs beat the per-pair models?
2. Does self-supervised pretraining help on top of pooling?
3. **Does knowledge pretraining add a measurable benefit beyond (2)?** This
   is the before/after comparison, with controls showing that the
   *knowledge content*, not just extra training, causes it.
4. Does the result beat the paper's best models by a good margin (§9)?

Scores stay **per pair, on the paper's folds, with the paper's metric**
(pooled out-of-fold R²/RMSE, pixel and field level), so they compare
directly with the paper, the point suite and the DEV rounds.

## 2. Why pooled

- **Data per model grows 5–10×.** Small pairs (GER-R 111 seasons, GER-W 188,
  BRA-W 140) benefit most. The round-1 donor warm start was a crude version
  of this.
- **Shared physics:** crop phenology in S2, weather–yield response and
  soil/terrain effects are largely shared. Crop and country are conditioning
  inputs, not separate models.
- **Natural fit for pretraining:** self-supervised and knowledge
  pretraining use *all* fields' inputs (no labels), the regime where
  pretraining pays off.
- **Fewer, larger runs:** one model per joint fold (§3) instead of one per
  pair and fold.

## 3. Leak-free joint folds (FM-01)

Each pair's **test set stays exactly its paper fold** (the existing manifests).
Training is the union of all pairs' data **minus everything that would leak
the held-out unit into training**:

| Protocol | Joint fold = | Excluded from training (all pairs) | Models per policy |
|---|---|---|---|
| CV10 | fold *k* of every pair at once | each pair's fold-*k* test seasons. `strict`: also every season sharing a physical field with any test season (any crop) | 10 |
| LOYO | harvest year *Y*, held out **globally** | every season of year *Y* in every country and crop | 9 (2016–2024) |
| LORO | region *R* of one country | every season in *R*, any crop (provinces for Argentina, farms elsewhere, as in the point suite) | one per distinct region (counted by FM-01) |

- **Validation:** a 10% season carve-out of the joint training set, grouped
  by physical field, stratified by pair.
- **LOYO holds the year out globally:** a stricter choice than the paper's
  per-pair years. It removes any same-year weather/price/management signal
  from other countries.
  - A per-country variant (hold out *Y* only in the pair's country) can be
    added as a sensitivity row (open decision D2).
- **Pretraining** (stages P1/P2) uses only the joint fold's **training**
  seasons by default (inductive).
  - A transductive variant (all inputs, labels never) is an optional
    sensitivity row (open decision D1).
- **Acceptance (FM-01):**
  - a `check_joint_fold` audit proves for every joint fold that no test
    season, test year (LOYO) or test region (LORO) is in training,
    validation or pretraining data;
  - under `strict`, no test physical field is in them either;
  - each pair's test set equals its paper-fold test set exactly;
  - cell counts are reconciled with the per-pair manifests.

## 4. Model (FM-02)

- **Backbone:** the point model with field-balanced cell sampling, the best
  learner so far. Its S4/S5/S6 options (early fusion, season-level term,
  neighbourhood stream) are fixed by the round-2 DEV results before FM-02
  starts.
- **Conditioning:** crop and country embeddings, added to the fused
  representation and fed to the season-level head. Optionally FiLM
  (feature-wise scale and shift) of the head by crop.
- **Targets:** standardized per crop (yield scales differ: corn ~3× wheat)
  with train-fold statistics, then de-standardized per crop. Inputs are
  standardized with pooled train-fold statistics per channel.
- **Sampling:** field-balanced within a pair, **pair-balanced across
  pairs** (temperature-scaled by pair size), so Argentina and Uruguay do not
  dominate.
- **Capacity:** start at 2× the point model's width/depth; sized by the FM-05
  DEV sweep.
- **Inference:** sensor inputs only. No text, rule library or teacher is
  needed at deployment.

## 5. Pretraining

### P1 — sensor self-supervised pretraining (FM-03)

Existing point objectives (`yieldsat_objectives.py`):
- masked observation reconstruction;
- forecast of the last valid optical observation (temporal streams hidden
  from that slot on);
- masked modality / cross-family prediction (whole families masked);
- contrastive field-season views (negatives exclude the same physical
  field).

Pooled over all pairs' training-fold cells; checkpoint per joint fold.

### P2 — knowledge pretraining (FM-04), the contribution

Follows [knowledge_pretraining.md](./knowledge_pretraining.md) v2 and the
YieldSAT mapping in
[yieldsat-image-knowledge-pretraining.md](./yieldsat-image-knowledge-pretraining.md):

- **Library:** versioned concepts and relationships (`spec/yieldsat_knowledge_assets/`).
  The current 12 concepts and 6 relationships are drafts. **Only
  expert-approved entries with an approved scoring contract are used for
  training** (gate G1).
- **Frozen text references:** pinned text encoder (`text_cache.py`), concept
  and qualified-relationship embeddings.
- **Concept grounding:** modality-side projectors map sensor representations
  at the concept's **true support** into the text space.
  - Supports: field-season pooling for weather, soil and DEM context; cell
    or local groups for S2 and terrain.
  - Coarse evidence (weather, soil) is **never broadcast as independent
    per-cell labels**.
  - Soft targets come from reviewed deterministic estimators. An optional
    VLM/LLM teacher (FM-04b) is used only after its calibration audit.
  - Unknown means masked, never absent.
- **Relational distillation:** reviewed executable scorers (ordered
  association over eligible pairs in the same applicable context;
  compatibility tables). The relationship gate never depends on whether
  the prediction already agrees with the rule.
- **Sensor preservation:** P1 losses continue during P2 to prevent
  collapse; weighted, with collapse/variance diagnostics.
- **Information boundary:** no yield, yield-derived summaries, held-out
  data, coordinates or soil-uncertainty channels in evidence or as trainable
  inputs (as in the existing package).
- **Port from image to pooled point model:** the `yieldsat_knowledge`
  package (registry, evidence, text cache, grounding/relational heads,
  diagnostics) is adapted from the image encoder streams to the point
  model's per-stream encoders. Its knowledge routing for the point branch
  (`OBJECTIVE_ROUTING`, currently *blocked*) is enabled once G1 passes.

## 6. Fine-tuning and evaluation (FM-05, FM-06)

- **Fine-tuning:** supervised yield fine-tuning of the pretrained encoder
  plus head per joint fold, with the same budget and early stopping for
  every arm.
- **Label efficiency:** fine-tune with 5/10/25/50/100% of each pair's
  training seasons (sampled per fold, fixed seeds). This is where
  pretraining should matter most, so the curves are part of the
  contribution.
- **Evaluation:** per pair and protocol, pooled out-of-fold R²/RMSE (pixel
  and field), mean ± std over 3 seeds. Plus a paired per-fold comparison
  between arms (same fold, seed and cells) with bootstrap confidence
  intervals.

## 7. Experiment arms

| Arm | Model | Pretraining | Purpose |
|---|---|---|---|
| A0 | per-pair point model, round-2 winner | none | baseline (existing per-pair protocol) |
| A1 | pooled | none | effect of pooling |
| A2 | pooled | P1 | effect of self-supervised pretraining |
| **A3** | pooled | **P1 + P2** | **knowledge pretraining (contribution)** |
| A4 | pooled | P1 + P2 with a **shuffled concept/relationship library** (same losses and compute) | control: knowledge content vs regularization |
| A5 | pooled | P1 + P2 **without text references** (learned concept prototypes) | control: role of the language prior |
| A6 | pooled | P1 for as many extra steps as P2 | compute-matched control for A3 |

Run A1–A6 on DEV first (FM-05), then the full matrix (FM-07):
9 pairs × CV10/LOYO/LORO × paper/strict × S2/S2+ADM × 3 seeds, plus
label-efficiency curves on DEV.

## 8. Tasks

| ID | Deliverable | Acceptance |
|---|---|---|
| FM-01 | `yieldsat_joint_folds.py`: joint CV10/LOYO/LORO folds over all pairs, paper/strict, with `check_joint_fold` | §3 audit passes for every fold; per-pair test sets equal the paper folds |
| FM-02 | Pooled point model (crop/country conditioning, per-crop targets, pair-balanced sampler) | Unit tests; DEV run of A1 complete |
| FM-03 | Pooled P1 pretraining entry (existing objectives) and encoder transfer to fine-tuning | Resumable checkpoint per joint fold; strict encoder transfer test |
| FM-04 | Knowledge pretraining ported to the pooled point encoder: grounding, relational distillation, preservation; diagnostics | Gradient/collapse/coverage diagnostics; no yield access (boundary test) |
| FM-04b | Optional VLM/LLM teacher annotations (existing `pilot.py`/`generate.py`) with calibration audit | Calibrated or rejected per §4 of the knowledge spec |
| G1 | **Expert review gate:** approved concepts, relationships, scoring contracts and estimator policies | Signed review records; drafts never used for training |
| FM-05 | DEV runs of A1–A6 + label-efficiency curves; selection | `results/foundation_dev.md` with paired tests |
| FM-06 | Cluster suites and pools for the pooled runs (GPU-bound pods, like the point suite) | Smoke test on one GPU passes |
| FM-07 | Full matrix with the selected configuration; results tables vs the paper and per-pair models | `results/foundation_*.md` |
| FM-08 | Write-up: pooled vs per-pair; P1; **P2 vs controls**; label efficiency | Spec log + results with uncertainty |

**Order:** FM-01 → FM-02 → (FM-03 ∥ FM-04 scaffolding with the draft
library, test-only) → G1 → FM-05 → FM-06 → FM-07 → FM-08. Knowledge training
on real data waits for G1.

## 9. Success criteria

- **Pooling:** A1 ≥ A0 on the DEV pooled pixel and field R², and not worse
  on any protocol by > 0.02.
- **Knowledge pretraining:** A3 > A2, A4 and A6 on DEV pooled pixel and field
  R², with paired per-fold 95% bootstrap CIs excluding 0. The benefit should
  grow at small label fractions.
- **Paper:** the selected arm beats the paper's best by ≥ 0.03 DEV-mean
  pixel and field R², with no protocol below the paper (the criterion in
  [yieldsat-improvement.md](./yieldsat-improvement.md)), then on the full
  dataset.

## 10. Open decisions (user)

- **D1 — pretraining data:** inductive (training-fold inputs only, default)
  or additionally a transductive sensitivity row (all inputs, never labels).
- **D2 — LOYO:** global held-out year (default) or also the per-country
  variant.
- **D3 — expert review (G1):** who reviews and when. Until then P2 runs
  only as a test-only pipeline check on drafts.
- **D4 — teacher:** use the optional VLM/LLM teacher (OpenRouter model and
  credentials as a namespace secret) or deterministic estimators only.

### Decisions taken (2026-10-04)

- **D3 resolved:** the project lead declared the 6 relationships and 12
  concepts verified (G1 satisfied for these entries; recorded in the library).
- **D4 resolved:** informed rule-based concept estimators first; no VLM/LLM
  teacher for now.
- **Point model first:** knowledge pretraining is implemented and validated on
  the point model ([yieldsat-point-knowledge-pretraining.md](./yieldsat-point-knowledge-pretraining.md))
  before pooled fine-tuning. Success is judged by
  [pretraining/success-criteria.md](./pretraining/success-criteria.md).
- **Pretraining data:** stage 0 is YieldSAT inputs of all pairs; the US
  national corpus follows in gated stages
  ([yieldsat-us-national-pretraining.md](./yieldsat-us-national-pretraining.md)).

## 11. Progress log

- 2026-10-03 — Specified (this document). Waits on the round-2 DEV results
  for the backbone options (§4) and on G1 for real knowledge training.
