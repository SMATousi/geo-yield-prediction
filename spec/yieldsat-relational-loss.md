# Relation-matching loss for within-field yield variability

**Status (2026-10-09): RL-01..RL-04 done (§6, §7). RL-04 gain is small: CV10
within-field R² +0.01 (significant), LOYO +0.01 (not significant), variability
ratio unchanged. Below the "clearly up" bar; RL-05 not launched (decision:
project lead).**

Related:
- [yieldsat-dense-series.md](./yieldsat-dense-series.md): the current best
  model, dense p3-nbr.
- [yieldsat-paper-protocol.md](./yieldsat-paper-protocol.md): protocol and
  pooled scoring.

## 1. Motivation

The most important output of the problem is the within-field yield pattern:
where a field yields more and where less. Today every pixel is trained
independently with MSE.

The total pixel variance splits exactly into between-field variance
(field levels) and within-field variance (the pattern). An independent
per-pixel loss mostly rewards getting field and year levels right.

Diagnostics so far (spec/yieldsat-dense-series.md §11):
- Even with each held-out year's mean set perfectly, dense p3-nbr's
  within-year Pearson r² on LOYO is only ≈ 0.33.
- The paper does not report within-field accuracy at all.

## 2. Loss

The aim is to preserve the relations between pixels of the same field,
not to make the pixels equal. A smoothness penalty Σ(ŷᵢ − ŷⱼ)² would
flatten the map and is explicitly **not** used.

**Relation-matching term** over pairs of pixels i, j of the same field
season:

  L_rel = Σ_(i,j) w_ij · Huber_δ( (ŷᵢ − ŷⱼ) − (yᵢ − yⱼ) ) / Σ w_ij

- If the true yields differ by Δ, the predictions must differ by Δ.
- The term is invariant to adding a constant to a field's predictions: it
  supervises the pattern, not the level.
- With all pairs weighted equally it equals the field-centred squared
  error up to a constant (before the Huber).

**Neighbour weighting w_ij:** local relations matter most (gradients, patch
edges). Pairs beyond a radius R are not used, and immediate neighbours
(1–2 cells) are down-weighted, because adjacent-cell differences carry the
most harvester noise (swath stripes, edge effects; noise variance doubles
in a difference).

**Huber:** robust to outlier differences from yield-map artefacts.

**Optional variance term:** against the shrinkage MSE produces,

  L_var = ( std_f(ŷ) − std_f(y) )² per field.

**Total:** L = MSE + λ_rel · L_rel (+ λ_var · L_var). The usual MSE keeps
the level supervised.

**Training only:** neighbours' targets enter the loss, never the inputs; at
test time the model sees inputs only.

**Batches:** field-clustered, e.g. 64 field seasons × 8 spatially clustered
pixels (grid row/col available), instead of 512 independent pixels.

## 3. Metrics (evaluation)

Per field season f, centre the targets on their mean and the predictions on
theirs: y′ = y − ȳ_f, ŷ′ = ŷ − mean_f(ŷ).

| Metric | Definition | Reads |
|---|---|---|
| **Within-field R² (pooled)** | 1 − Σ_f Σᵢ (ŷ′ − y′)² / Σ_f Σᵢ y′² | Pattern and amplitude inside fields, pixel-weighted. Main metric |
| Median per-field R² | Median over fields (≥ 50 pixels) of R²_f | A typical field |
| Median within-field r | Median over fields of Pearson(ŷ′, y′) | Pattern only, scale-free |
| Median variability ratio | Median over fields of std(ŷ′) / std(y′) | < 1: map too smooth (shrinkage) |
| Within-field share | Σ_f Σᵢ y′² / Σ (y − ȳ)² | Fraction of total pixel variance that is within-field |

These are proper evaluation metrics: they use labels to score predictions,
like any R². They are reported next to the standard pooled R² (the paper's
metric), which stays the comparison with the paper.

## 4. Baseline (current models)

Computed 2026-10-08 from the saved test predictions (`results/within_field_baseline.json`). Means over the 9 pairs; per-field statistics use fields with ≥ 50 pixels.

| Protocol | Model | Pooled R² (paper metric) | **Within-field R² (pooled)** | Median per-field R² | Median within-field r | Median variability ratio |
|---|---|---|---|---|---|---|
| CV10 | monthly p3-nbr | 0.511 | **0.248** | 0.251 | 0.510 | 0.563 |
| CV10 | dense p3-nbr | 0.540 | **0.276** | 0.287 | 0.538 | 0.579 |
| CV10 | dense + dropout | 0.535 | **0.282** | 0.284 | 0.536 | 0.536 |
| CV10 | TabM F1 | 0.509 | **0.268** | 0.269 | 0.525 | 0.512 |
| LOYO | monthly p3-nbr | 0.346 | **0.167** | 0.175 | 0.452 | 0.545 |
| LOYO | dense p3-nbr | 0.356 | **0.195** | 0.189 | 0.471 | 0.554 |
| LOYO | dense + dropout | 0.358 | **0.201** | 0.206 | 0.477 | 0.506 |
| LOYO | TabM F1 | 0.317 | **0.196** | 0.195 | 0.468 | 0.444 |
| LORO | monthly p3-nbr | 0.324 | **0.166** | 0.172 | 0.451 | 0.558 |
| LORO | dense p3-nbr | 0.371 | **0.208** | 0.208 | 0.482 | 0.569 |
| LORO | dense + dropout | 0.373 | **0.224** | 0.227 | 0.495 | 0.515 |
| LORO | TabM F1 | 0.293 | **0.202** | 0.207 | 0.474 | 0.460 |

**Within-field share of pixel variance: 0.518** (mean over pairs; the same for every model, since it depends only on the targets).

Per pair, dense p3-nbr (within-field R² / median within-field r / median variability ratio):

| Pair | CV10 | LOYO | LORO |
|---|---|---|---|
| ARG-C | 0.45 / 0.66 / 0.69 | 0.30 / 0.58 / 0.65 | 0.34 / 0.57 / 0.64 |
| ARG-S | 0.52 / 0.72 / 0.73 | 0.47 / 0.68 / 0.71 | 0.48 / 0.69 / 0.75 |
| ARG-W | 0.37 / 0.66 / 0.71 | 0.15 / 0.51 / 0.75 | 0.28 / 0.58 / 0.73 |
| BRA-C | 0.28 / 0.56 / 0.61 | 0.15 / 0.45 / 0.59 | 0.20 / 0.51 / 0.52 |
| BRA-S | 0.23 / 0.48 / 0.49 | 0.17 / 0.44 / 0.46 | 0.18 / 0.44 / 0.52 |
| BRA-W | 0.10 / 0.39 / 0.42 | 0.09 / 0.35 / 0.39 | 0.08 / 0.34 / 0.40 |
| GER-R | 0.23 / 0.53 / 0.63 | 0.17 / 0.46 / 0.56 | 0.13 / 0.47 / 0.62 |
| GER-W | 0.22 / 0.53 / 0.57 | 0.18 / 0.48 / 0.52 | 0.11 / 0.44 / 0.56 |
| URG-S | 0.10 / 0.30 / 0.35 | 0.09 / 0.29 / 0.35 | 0.08 / 0.29 / 0.37 |

**Reading:**
1. **Half the pixel variance is within fields**, but the best model explains only 20–28% of it (dense p3-nbr: CV10 0.276, LOYO 0.195, LORO 0.208).
2. **The pattern is partly right but the amplitude is far too low.**
   - Within-field correlation is ≈ 0.47–0.54.
   - The predicted maps carry only ≈ 55% of the true within-field variability (variability ratio ≈ 0.55).
   - Shrinkage alone costs R² even where the pattern is right.
3. **This is the target of the relation-matching loss (§2) and the variance term.** The success criterion for RL-04: within-field R² and the variability ratio up, pooled R² not lower.
4. **Model ranking holds within fields** (dense > monthly). TabM F1 is the smoothest (variability ratio 0.44–0.51).

## 5. Plan (if adopted)

| ID | Deliverable | Acceptance |
|---|---|---|
| RL-01 | Field-clustered batch sampler (K pixels per field season, spatially clustered) | Unit test: K pixels per field, all within radius R |
| RL-02 | Relation loss (Huber, distance weights, optional variance term) in `main_yieldsat_finetune.py` | Unit tests: invariant to per-field constant shifts; zero when predictions equal targets |
| RL-03 | Within-field metrics in `util/yieldsat_eval.py` and the reports | Match this spec's baseline numbers on existing predictions |
| RL-04 | DEV test: dense p3-nbr + L_rel, λ_rel ∈ {0.5, 1}, 4 DEV pairs × CV10 + LOYO | Within-field R² up clearly (paired fold bootstrap) without lowering pooled R² |
| RL-05 | If RL-04 passes: all pairs and protocols | As RL-04, all pairs |

## 6. Implementation (RL-01..RL-03, 2026-10-08)

- **Loss** (`yieldsat_relation_loss.py`): `relation_loss` builds all pairs in a
  batch with the same field season, Chebyshev grid distance 1..R (R = 5
  cells), both targets valid. Weight 0.5 for distance 1 (`--rel_near_weight`),
  1 otherwise. Huber δ = 1 on the normalized target scale (`--rel_delta`).
  `variance_loss` is the optional per-field std term (`--var_weight`, off
  in RL-04).
- **Sampler** (`FieldClusterBatchSampler` in `dataset/yieldsat_dataset.py`):
  used whenever `--rel_weight` or `--var_weight` > 0. A batch is
  batch_size / K clusters of K = 8 pixels (`--rel_cluster`). Each cluster:
  a field season drawn ∝ n_pixels^0.5, a uniform anchor pixel, then 7 more
  pixels within radius R of it. The MSE term sees the same batch, so the
  pixel mix shifts somewhat towards small fields.
- **Training** (`main_yieldsat_finetune.py`): L = MSE + `--rel_weight` ·
  L_rel (+ `--var_weight` · L_var). Logged as `train_rel` / `train_var`.
  Validation and test are unchanged (inputs only).
- **Metrics:** `within_field_metrics` (§3) for every test fold in
  `report.json` → `test.within_field`. Pair-level numbers are recomputed
  from the pooled `test_predictions.npz`, as for the baseline.
- **Tests:** `tests/test_relation_loss.py`:
  - zero loss at perfect predictions;
  - invariant to per-field constant shifts;
  - cross-field pairs ignored;
  - flattened fields penalized;
  - variance term;
  - metric extremes;
  - sampler clusters stay in one field season within radius R.
- **Smoke** (local, dense, GER-R CV10 fold 0, λ = 1, 2 × 150 steps only):
  - trains stably, `train_rel` ≈ 0.27;
  - test within-field R² 0.315, median variability ratio 0.63, pooled R²
    0.356.
  - Too short to compare with the baseline.

## 7. RL-04 (DEV test)

Job `smatousi-yieldsat-protocol-rel-dev`
(`cluster/nautilus/protocol_rel_dev_job.yaml`, units
`cluster/tabm/protocol_rel_dev_units.json`):
- **Runs:** 130, dense p3-nbr exactly as the baseline (same splits, seed,
  epochs, selection on the test fold, as in the paper protocol) plus
  `--rel_weight` 0.5 (`ours-p3nbr-dense-rel05`) or 1.0
  (`ours-p3nbr-dense-rel10`).
- **Coverage:** ARG-W, BRA-C, GER-R, URG-S × CV10 (10 folds) and LOYO.
- **Comparison:** per pair and protocol, paired with the baseline on the
  same test pixels.
  - Pooled R² and within-field R² deltas, with 95% CIs from a field-season
    cluster bootstrap (2000 replicates; a fold bootstrap has too few folds
    for LOYO).
  - Per-field medians reported alongside.

### 7.1 Results (2026-10-09)

All 130 runs finished (none failed). Data: `results/rel_dev_rl04.json`,
script `results/rel_cmp.py`. Fields are keyed by fold + season id, since the
saved season ids are per fold. The baseline numbers reproduce §4.

**Δ vs dense p3-nbr, [95% field-season bootstrap CI]:**

| Protocol | Pair | Base pooled R² | Base within R² | λ 0.5: Δ pooled | λ 0.5: Δ within | λ 1: Δ pooled | λ 1: Δ within |
|---|---|---|---|---|---|---|---|
| CV10 | ARG-W | 0.824 | 0.367 | +0.010 [−0.000, +0.023] | +0.013 [−0.009, +0.039] | +0.006 [−0.010, +0.019] | +0.015 [−0.004, +0.038] |
| CV10 | BRA-C | 0.502 | 0.275 | +0.004 [−0.011, +0.017] | +0.006 [−0.008, +0.018] | +0.008 [−0.005, +0.020] | **+0.014 [+0.002, +0.025]** |
| CV10 | GER-R | 0.521 | 0.229 | −0.001 [−0.019, +0.021] | +0.015 [+0.000, +0.031] | +0.010 [−0.012, +0.034] | +0.006 [−0.009, +0.023] |
| CV10 | URG-S | 0.419 | 0.098 | +0.001 [−0.006, +0.009] | +0.002 [−0.004, +0.009] | +0.001 [−0.009, +0.010] | +0.003 [−0.004, +0.008] |
| **CV10** | **mean** | | | +0.004 [−0.003, +0.011] | **+0.009 [+0.001, +0.017]** | +0.006 [−0.002, +0.014] | **+0.010 [+0.003, +0.017]** |
| LOYO | ARG-W | 0.742 | 0.151 | −0.010 [−0.038, +0.015] | +0.018 [−0.036, +0.088] | −0.013 [−0.036, +0.008] | +0.028 [−0.026, +0.098] |
| LOYO | BRA-C | 0.116 | 0.147 | +0.033 [−0.018, +0.086] | **+0.033 [+0.016, +0.049]** | +0.032 [−0.016, +0.075] | +0.015 [−0.004, +0.033] |
| LOYO | GER-R | 0.285 | 0.168 | +0.026 [−0.032, +0.088] | −0.004 [−0.034, +0.026] | +0.038 [−0.009, +0.083] | +0.002 [−0.020, +0.028] |
| LOYO | URG-S | 0.355 | 0.086 | **−0.014 [−0.027, −0.001]** | −0.005 [−0.011, +0.001] | −0.008 [−0.020, +0.004] | **−0.006 [−0.012, −0.001]** |
| **LOYO** | **mean** | | | +0.009 [−0.012, +0.030] | +0.010 [−0.006, +0.031] | +0.012 [−0.006, +0.029] | +0.010 [−0.006, +0.030] |

**Variability ratio and within-field r:**
- Unchanged within ±0.04 per pair.
- Example, CV10 ARG-W: ratio 0.71 → 0.68 / 0.67; r 0.66 → 0.66.

**Reading:**
1. **The loss does no harm and helps a little.**
   - Pooled R² is not lower on average (+0.004 to +0.012).
   - Within-field R² rises by ≈ +0.01 on average for both λ.
   - The gain is significant on CV10 (CI above 0), not on LOYO.
   - Relative to the baseline it is small: CV10 within-field 0.24 → 0.25,
     mean over these 4 pairs.
2. **It does not fix the shrinkage.**
   - The variability ratio stays ≈ 0.35–0.75: the predicted maps are as
     flat as before.
   - The pattern (r) improves only marginally.
   - Matching neighbour differences under an MSE-dominated objective still
     lets the model hedge toward small differences where it is unsure.
3. **λ = 1 ≥ λ = 0.5**, by a margin within noise.
4. **Against the RL-04 criterion** (within-field R² clearly up, pooled R² not
   lower), this is a pass on "not lower" but only a weak pass on "up".
   RL-05 is not launched automatically.
   - **Option A:** run RL-05 at λ = 1 anyway (cheap gain, no downside).
   - **Option B:** first test the variance term (λ_var > 0) or a larger
     λ_rel (2–4), which target the amplitude directly.

## 8. Image models, within-field (2026-10-09)

The image models' test predictions, scored on the same cells as the point
model. Matched by field season and grid cell (`yieldsat_image_compare.match_fold`).
Data: `results/image_within_field.json`.

Suites and setting:
- Image v1 = `image_full`; image v2 = `image_v2`; both 324 experiments.
- Rows below: S2+ADM, paper label policy, mean over 9 pairs × 3 seeds.
- These suites used validation selection. The point model is the one they
  were compared with (`before_full`).

Each cell gives pooled R² / within-field R² / median within-field r / median
variability ratio.

| Protocol | Image v1 | Image v2 | Point |
|---|---|---|---|
| CV10 | 0.257 / 0.122 / 0.41 / 0.52 | 0.233 / 0.079 / 0.38 / 0.59 | 0.43–0.45 / 0.22 / 0.49 / 0.56 |
| LOYO | 0.053 / 0.068 / 0.38 / 0.54 | 0.061 / 0.036 / 0.37 / 0.59 | 0.18–0.21 / 0.11 / 0.40–0.41 / 0.57–0.58 |
| LORO (farm) | −0.079 / 0.075 / 0.34 / 0.47 | 0.002 / 0.052 / 0.33 / 0.50 | 0.04–0.05 / 0.09 / 0.37–0.38 / 0.48–0.49 |
| LORO (province) | 0.188 / 0.092 / 0.45 / 0.64 | 0.206 / −0.004 / 0.43 / 0.85 | 0.46–0.48 / 0.20 / 0.52 / 0.66–0.67 |

Image beats point within-field in:
- image v1: 0/27 CV10, 14/27 LOYO, 7/18 LORO, 1/9 province rows;
- image v2: 0, 4, 3 and 0 rows.

Partial paper-protocol runs (ARG-S, against dense p3-nbr within-field R²):

| Run | Image | Dense p3-nbr |
|---|---|---|
| Image v1, CV10 | 0.22 | 0.52 |
| Image v2, CV10 | 0.32 | 0.52 |
| Image v2, LOYO | 0.26 | 0.47 |
| Hybrid-h1, CV10 | 0.40 | 0.51 |

**Reading:**
1. The image models are worse inside fields too, not only in level. Their
   within-field correlation r is lower in every protocol.
2. Within-field R² ≈ 2·r·k − k² (k = variability ratio), which is maximal at
   k = r.
   - Image v2 has more amplitude than v1 (k 0.59 vs 0.52) but lower r, so
     its within-field R² is lower.
   - Extreme case, province LORO: k = 0.85, r = 0.43 → R² ≈ 0. This is the
     observed −0.004.
   - More contrast without a better pattern costs R². This is the argument
     against the per-field spread term (§2, `--var_weight`).
