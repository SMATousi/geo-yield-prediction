# Relation-matching loss for within-field yield variability

**Status (2026-10-08): proposed; baseline computed (§4).** Decision pending
(project lead).

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
