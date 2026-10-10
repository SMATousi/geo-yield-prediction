# Field encoder: learned field context for within-field yield

**Status (2026-10-10): FE-04 not passed (§7).**
- **Result:** within-field R² +0.006 on CV10 (not significant) and +0.015 on LOYO (significant, driven by ARG-W).
- **Pooled R²** is −0.009 and −0.017 (not significant).
- FE-05 is not launched. Follow-up variants are proposed in §7.

Related:
- [yieldsat-field-context.md](./yieldsat-field-context.md): hand-made field context (field S2 mean/std, within-field
  z-scores, edge distance). That spec's best model, dense p3-nbr + field context + relation loss, is the baseline
  here (`ours-p3nbr-dense-fc-rel10`).
- [yieldsat-relational-loss.md](./yieldsat-relational-loss.md): within-field metrics and the relation loss.
- [yieldsat-paper-models.md](./yieldsat-paper-models.md) §8: the paper's 5×5-window models are the weakest within
  fields. Local windows are not enough; field-scale context is the open lever.

## 1. Motivation

**Current field context** (field-context spec §4) is three extra streams per pixel:
- the field's per-slot S2 mean and std;
- the pixel's within-field S2 z-score;
- within-field terrain z-scores and edge distance.

**What it cannot express:** it describes the field only through its first two moments. It cannot show zones (a wet
corner, a sandy strip, a bimodal field), and it has no spatial layout: the model never learns where a pixel sits
relative to the rest of the field.

**Why field scale matters:** the semivariograms (field-context spec §3) show the predictable within-field structure
extends to 50–300 m, well beyond the 5×5 (±20 m) window.

**Proposal:** a learned field encoder. It sees a sample of the field's own pixels (their full S2 series and other
inputs), and the target pixel reads from it.

## 2. Model (options 1 + 2 + 3)

**Pixel embedding (time handled by the point model):**
- Every pixel, target or context, goes through the same point encoder: the dense p3-nbr streams plus the field-context
  streams, then the Perceiver fusion.
- Its fused latents are mean-pooled to h ∈ R^128.
- The temporal S2 (72 dense slots) is handled there by the masked temporal encoders, so the field encoder works on
  per-pixel vectors. There is no 4-D (time × space) encoder.

**Option 1, field set encoder:**
- **Context:** K = 64 pixels of the target's field season.
  - Training: a new random sample in every batch.
  - Evaluation: a fixed sample per field season (seeded), so every pixel of a field sees the same context.
- **Context tokens:** c_k = h_k + φ_abs(p_k − p̄). Here p_k is the grid position (cells), p̄ the context centroid, and
  φ_abs a Fourier-feature MLP of position / 50 cells (500 m).
- **Encoder:** one self-attention block over the K tokens (Set Transformer SAB), then attention pooling with Q = 4
  learned seeds (PMA). The output is the field tokens F ∈ R^{4×128}.

**Option 2, pixel-to-field cross-attention:**
- **Query:** the target's h.
- **Keys / values:** c_k + φ_rel(p_target − p_k), a Fourier-feature MLP of the relative offset.
- **Result:** r ∈ R^128. The pixel can relate itself to specific parts of the field ("greener than the pixels 100 m
  north"), the relation the relation loss supervises.

**Option 3, level / deviation split:**
- **Level** ℓ = MLP(mean(F)): the field's yield level, from field context only.
- **Deviation** d = MLP([h, r, mean(F)]): the pixel's offset from it.
- **Prediction:** ŷ = ℓ + d.
- **Loss:** MSE(ŷ, y) + λ_level · MSE(ℓ, ȳ_f) + λ_rel · L_rel(ŷ).
  - ȳ_f is the field-season mean target, standardized; it is a training target only.
  - λ_level = 1 and λ_rel = 1, the latter as in the baseline.
  - L_rel is invariant to per-field constants, so it supervises d. The level term ties ℓ to the field mean.

**Inputs at test time:** the target pixel's inputs and the context pixels' inputs, from the same field season. No
labels are used, as with the field-context streams. Field-context spec §4 lists the caveats; the footprint caveat
applies the same way.

## 3. Training

- **Batches by field:** each batch has 16 field seasons × 4 spatial clusters × 8 pixels (512 targets), plus 64
  context pixels per field (1,024 context pixels).
  - `FieldClusterBatchSampler(clusters_per_field=4)` builds the targets: it keeps the relation loss's same-field pairs
    and shares one context sample per field.
- **Budget, optimizer and protocol** are as for the baseline: dense p3-nbr arguments, field context, relation loss
  λ = 1, the paper protocol (no validation set, test-fold selection), early stopping.
- **Cost:** each step encodes 1,536 pixel series instead of 512, roughly 3× the GPU work per step.

## 4. Implementation

- **`dataset/yieldsat_field_context_batch.py`: `FieldContextDataset(YieldSATPointDataset)`.**
  - `get_batch(idx)` adds, for each field season in the batch, K context items, delivered as `ctx_inputs`,
    `ctx_masks`, `ctx_available`, `ctx_time_features`, `ctx_time_valid`, `ctx_crop`, `ctx_grid_row`, `ctx_grid_col`.
  - Group indices: `ctx_group` (G·K) and `tgt_group` (B).
  - Training draws a fresh random context; evaluation uses a fixed sample per season (seed = season index), so every
    pixel of a field sees the same context.
- **`models_yieldsat_field.py`: `FieldEncoderModel`.**
  - It wraps `YieldSATPointModel` (pixel encoder), adds the set encoder, cross-attention and level / deviation heads.
  - Same `loss` / `forward` / descriptor interface as the other models.
- **`main_yieldsat_finetune.py`:** `--model field_encoder`, `--fe_context` (64), `--fe_seeds` (4), `--fe_level_weight`
  (1.0), `--fe_clusters_per_field` (4).
- **Units:** `yieldsat_protocol_runs.py --batch fe-dev` (FE-04) and `--batch fe-all` (FE-05), tag
  `ours-p3nbr-dense-fe`.

## 5. Plan

| ID | Deliverable | Acceptance |
|---|---|---|
| FE-01 | Context dataset + field-grouped sampler | Unit tests: context items come from the target's field season; evaluation context is fixed per season; targets cluster by field; no labels in the inputs |
| FE-02 | `FieldEncoderModel` | Unit tests: forward/backward; ŷ = ℓ + d; permuting the context pixels leaves predictions unchanged (set invariance) |
| FE-03 | Trainer integration + local smoke | Trains; `report.json` with `within_field`; speed measured to size the cluster job |
| FE-04 | DEV test: 4 DEV pairs × CV10 + LOYO, vs `ours-p3nbr-dense-fc-rel10` | Within-field R² clearly up (paired field-season bootstrap), pooled R² not lower |
| FE-05 | If FE-04 passes: all 9 pairs × CV10/LOYO/LORO; full table with the paper models | As FE-04, all pairs |

## 6. Implementation notes and smoke (2026-10-10)

**Tests:** `tests/test_field_encoder.py`, 5 tests, all passing.
- Context items come from the target's field season.
- The evaluation context is fixed per season; the training context is random.
- Clusters are grouped by field.
- One level per field season (ŷ = ℓ + d).
- Permuting the context pixels leaves predictions unchanged.

The full suite passes (179).

**Local smoke** (dense GER-R CV10 fold 0, 4 × 200 steps; too short to compare accuracy):

| Model | Val R² by epoch | Test pooled R² | Within-field R² | Within-field r | s / epoch |
|---|---|---|---|---|---|
| Baseline (FC + relation loss) | 0.32 → 0.46 | 0.460 | 0.387 | 0.53 | 10 |
| Field encoder | 0.11 → 0.35 | 0.352 | 0.346 | 0.57 | 27 |

- The field encoder starts slower: new attention layers and a level head trained from scratch.
- Its within-field correlation is already higher.
- It costs 2.7× the GPU time per step (1,536 encoded pixels per batch).

**FE-04:** `cluster/nautilus/protocol_fe_dev_job.yaml`, 65 units (`cluster/tabm/protocol_fe_dev_units.json`), with the
same budget and protocol as `ours-p3nbr-dense-fc-rel10`. `protocol_fe_all_units.json` (152 units) is prepared for FE-05.

## 7. FE-04 results (2026-10-10)

All 65 runs finished (none failed). Data: `results/fe_dev_fe04.json`; script `results/rel_cmp.py`, with
`REL_BASE=ours-p3nbr-dense-fc-rel10`.
- **Comparison:** the field encoder vs the current best model (dense p3-nbr + field context + relation loss) on the
  same test pixels.
- **Intervals:** 95% field-season cluster bootstrap; **bold** = CI excludes 0.

| Protocol | Pair | Best: pooled / within | FE: pooled / within | Δ pooled R² | Δ within-field R² |
|---|---|---|---|---|---|
| CV10 | ARG-W | 0.822 / 0.394 | 0.821 / 0.403 | −0.001 [−0.018, +0.019] | +0.009 [−0.028, +0.052] |
| CV10 | BRA-C | 0.494 / 0.277 | 0.474 / 0.288 | −0.020 [−0.050, +0.008] | +0.011 [−0.000, +0.025] |
| CV10 | GER-R | 0.517 / 0.254 | 0.499 / 0.259 | −0.017 [−0.046, +0.011] | +0.005 [−0.012, +0.023] |
| CV10 | URG-S | 0.419 / 0.105 | 0.420 / 0.104 | +0.001 [−0.012, +0.015] | −0.001 [−0.007, +0.004] |
| **CV10** | **mean** | | | −0.009 [−0.021, +0.003] | +0.006 [−0.005, +0.018] |
| LOYO | ARG-W | 0.725 / 0.181 | 0.711 / 0.251 | −0.013 [−0.046, +0.018] | **+0.069 [+0.024, +0.125]** |
| LOYO | BRA-C | 0.207 / 0.200 | 0.161 / 0.191 | −0.047 [−0.116, +0.008] | −0.010 [−0.028, +0.007] |
| LOYO | GER-R | 0.291 / 0.208 | 0.309 / 0.220 | +0.019 [−0.052, +0.078] | +0.013 [−0.002, +0.029] |
| LOYO | URG-S | 0.345 / 0.086 | 0.317 / 0.075 | **−0.028 [−0.042, −0.014]** | **−0.011 [−0.017, −0.005]** |
| **LOYO** | **mean** | | | −0.017 [−0.042, +0.005] | **+0.015 [+0.002, +0.031]** |

**Reading:**
1. **Not a pass.**
   - Within-field R² is not clearly up: CV10 +0.006 is not significant; LOYO +0.015 is significant but driven by one
     pair, ARG-W (+0.069).
   - Pooled R² trends down: −0.009 and −0.017. URG-S LOYO is significantly worse on both metrics.
2. **Training curves** (32 finished folds at mid-run):
   - The field encoder peaks earlier: best epoch median 3 vs 5.
   - It fits the training data slightly worse: train loss 0.594 vs 0.578 at the best epoch.
   - So the extra capacity is not turning into better held-out prediction. "Stopping too early" is not the main
     issue.
3. **Where the pooled loss comes from:** the level head. ℓ is predicted from 64 context pixels and trained against
   the field mean. A field-level error moves every pixel of a field, which hurts pooled R² and leaves within-field R²
   untouched. This is consistent with the losses being largest where field levels are hard: BRA-C and URG-S LOYO.
4. **Field-scale structure does exist:** ARG-W LOYO within-field +0.069, the largest gain of any method so far on a
   held-out-year row.

**Proposed follow-up (FE-04b, DEV, decision: project lead).** Remove the level-head risk and start from a trained
encoder:
- **(a) Fold-matched warm start.** Initialize the point encoder from the same fold's
  `ours-p3nbr-dense-fc-rel10` checkpoint. This is the same training data, so there is no leakage. Train the field
  branch with a lower learning rate for the encoder.
- **(b) No separate level head:** ŷ = base prediction + d(h, r, F). The field encoder then only adds a within-field
  correction on top of the best model, and the relation loss supervises it.
- **(c)** K = 128 context pixels.
