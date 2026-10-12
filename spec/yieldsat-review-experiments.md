# Experiments for review robustness

**Status (2026-10-12): RX-3 done (§RX-3 results); RX-1 done; RX-2 running on the cluster.**

Related:
- [yieldsat-paper-models.md](./yieldsat-paper-models.md) §8: the main comparison (paper protocol).
- [yieldsat-input-information.md](./yieldsat-input-information.md): the field-specific gap (per-field oracle 0.50 vs
  cross-field 0.29) that RX-3 targets.

## RX-1: field-level R² and RMSE

The paper's Table 4 reports field-level R² and RMSE (t/ha) next to pixel R². Our tables have only pixel-level R²
(pooled and within-field).

**Additional metrics** for all six models × 9 pairs × 3 protocols, from the saved test predictions with no training:
- pixel RMSE (t/ha);
- field-level R² and field-level RMSE (t/ha). A field season's prediction and target are the means over its test
  cells; R² and RMSE are pooled over all test field seasons of all folds.

**Implementation:** `results/paper_models_table.py` gains these columns; a table in the paper.

## RX-2: clean validation protocol

The paper protocol (and every table so far) selects the training epoch on the test fold. RX-2 repeats the main
comparison with model selection on a validation set.

**Splits:** YieldSAT's own manifests with a validation partition (`paper_<protocol>_<pair>_…_paper_s0_foldNN`).
- They have the same test folds as the no-validation manifests (those merge the validation fields back into
  training), so the test pixels are identical.
- Selection uses the validation fold; the test fold is used only for the final score.

**Models:** our best model (`ours-p3nbr-dense-fc-rel10`) and MMGF (the best paper model), with the same arguments as
in the main runs; 9 pairs × CV10/LOYO/LORO, i.e. 2 × 217 runs.
- Tags: `ours-p3nbr-dense-fc-rel10-val` and `paper_mmgf-val`.

**Table:** pooled and within-field R² under both protocols, and our model vs MMGF under the clean protocol (paired
bootstrap).

**Expected outcome:** absolute values drop a little under clean selection; the question is whether the ranking and
the within-field gap hold.

## RX-3: prior-year field calibration (practical setting)

**Setting** (constraint from the project lead): the farmer supplies last year's yield map of the same physical field.
Nothing from the year being predicted is used.

**Data:**
- **Protocol:** LOYO, where the test year is held out. The model is trained on all other years.
- **Field identity:** `physical_field_id`.
  - 1,009 of 2,173 field seasons have an earlier season of the same field; 868 of these are the immediately preceding
    year.
  - In 908 the earlier season had the same crop; otherwise it is a rotation.
- **Cell alignment:** seasons of a field share the EPSG but rarely the grid (154 of 1,009 have an identical
  transform). Cells are matched by geographic cell centre (affine transform; nearest centre within 5 m).

**Ingredients** for a target field season (field F, year t) and its most recent earlier season (F, t′ < t):
- ŷ_t: our model's LOYO prediction for year t (out of sample).
- y_t′: the observed yield map of the earlier season, the farmer's data.
- ŷ_t′: our model's prediction for the earlier season from the LOYO fold that held out year t′ (out of sample). It
  measures the model's own error pattern on that field.

**Predictors**, all in within-field anomaly space and scored with within-field R²:

| ID | Prediction for year t | Field-specific information used |
|---|---|---|
| M0 | model: ŷ′_t | none |
| M1 | persistence: a · s · z(y_t′) | last year's pattern only |
| M2 | blend: b₁ ŷ′_t + b₂ s z(y_t′) | model + last year's pattern |
| M3 | residual transfer: ŷ′_t + c · s z(y_t′ − ŷ_t′) | the model's own field-specific error, as seen last year |
| M4 | M2 + M3 terms jointly | both |
| M5 | per-field amplitude: β_F ŷ′_t, with β_F fitted on (y′_t′, ŷ′_t′) | per-field scaling |

- z(·) is the within-field z-score.
- s = std(ŷ′_t): the model's predicted spread for this year, which needs no labels.
- **Coefficients** (a, b, c) are fitted by least squares on all (field season, earlier season) pairs of the same
  country–crop pair **whose current year ≠ t**: training years only, so no year-t label is used anywhere.
  - β_F uses only the earlier season of the field itself.

**Reported:**
- Per pair and pooled: within-field R² of M0–M5 on the field seasons that have an earlier season.
- Subsets: earlier season = same crop vs rotation, and earlier season = previous year.
- Reference: the per-field oracle (same-year labels; not attainable in practice).

**Question:** how much of the field-specific gap (§7.1 of the input-information spec) last year's map recovers.

## RX-3 results (2026-10-12)

**Setup:**
- Data: `results/prior_year.json`, made by `yieldsat_prior_year.py`.
- Model: the best model (`ours-p3nbr-dense-fc-rel10`), LOYO predictions.
- Cases: 737 test field seasons with an aligned earlier season of the same physical field (≥ 200 matched cells),
  3.85M cells in total.
- Metric: within-field R² on those seasons.

| Pair | Seasons (same crop / previous year) | M0 | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|---|---|
| ARG-C | 63 (10 / 44) | 0.345 | 0.097 | 0.351 | 0.345 | 0.353 | 0.244 |
| ARG-S | 170 (94 / 119) | 0.481 | 0.148 | 0.491 | 0.484 | 0.491 | 0.441 |
| ARG-W | 58 (11 / 44) | 0.184 | 0.048 | 0.218 | 0.182 | 0.218 | 0.125 |
| BRA-C | 57 (7 / 57) | 0.171 | 0.044 | 0.167 | 0.172 | 0.175 | 0.158 |
| BRA-S | 128 (64 / 126) | 0.251 | 0.045 | 0.251 | 0.252 | 0.253 | 0.220 |
| BRA-W | 57 (13 / 55) | 0.098 | 0.029 | 0.103 | 0.100 | 0.105 | 0.091 |
| GER-R | 30 (4 / 8) | 0.240 | 0.137 | 0.266 | 0.247 | 0.254 | 0.187 |
| GER-W | 107 (75 / 68) | 0.247 | 0.122 | 0.265 | 0.260 | 0.266 | 0.197 |
| URG-S | 67 (67 / 49) | 0.094 | -0.001 | 0.094 | 0.094 | 0.094 | 0.065 |
| **Mean** | 737 | **0.234** | **0.074** | **0.245** | **0.237** | **0.245** | **0.192** |

**Paired field-season bootstrap**, mean over pairs, Δ within-field R² vs M0 with 95% CI:

| Method | Δ vs M0 |
|---|---|
| M1 persistence | **-0.160 [-0.178, -0.141]** |
| M2 blend | **+0.011 [+0.005, +0.018]** |
| M3 residual transfer | **+0.003 [+0.002, +0.004]** |
| M4 blend + residual | **+0.011 [+0.005, +0.019]** |
| M5 per-field amplitude | **-0.042 [-0.059, -0.028]** |

**By subset:**

| Subset (pooled over pairs) | M0 | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|---|
| all | 0.246 | 0.070 | 0.251 | 0.248 | 0.254 | 0.205 |
| same crop | 0.269 | 0.094 | 0.280 | 0.274 | 0.281 | 0.243 |
| rotation | 0.238 | 0.062 | 0.241 | 0.239 | 0.244 | 0.191 |
| previous year | 0.237 | 0.068 | 0.241 | 0.238 | 0.245 | 0.195 |

**Reading:**
1. **Last year's yield map adds little:**
   - Blending it with the model raises within-field R² by +0.011 (0.234 → 0.245, significant).
   - The model's own error from last year, transferred to this year, adds +0.003 (significant but negligible).
   - Rescaling the model's amplitude per field from last year hurts (−0.042).
2. **Within-field yield patterns are weakly persistent across years.**
   - Persistence alone (last year's anomaly pattern, optimally scaled) reaches only 0.074, far below the model's
     0.234.
   - It is higher when last year had the same crop (0.094) than after a rotation (0.062).
   - The season's weather, which the model sees through S2 and weather inputs, reshapes the within-field pattern each
     year.
3. **The field-specific gap does not persist across years.** The per-field oracle's advantage (0.50 vs 0.29,
   input-information spec §7.1) comes from same-year labels. Last year's labels of the same field recover only ≈ 0.01
   of it.
   - The field-specific input-to-yield relation is largely season-specific (field × year interaction), not a fixed
     property of the field.
   - This tightens the paper's interpretation of the second gap: it cannot be closed by collecting one prior year per
     field.
4. **Practical message:** prior-year maps are a cheap, small improvement (+0.011) that combines additively with the
   model. They are not a substitute for better inputs.

## RX-1 results (2026-10-12)

**Setup:** `results/paper_models_table.json` (now with `pixel_rmse`, `field_r2`, `field_rmse`); cells give field R² /
pixel RMSE / field RMSE (t/ha), mean over the 9 pairs.

| Protocol | 3D-LSTM | 3D-ConvLSTM | AFF | MMGF | Ours (base) | Ours |
|---|---|---|---|---|---|---|
| CV10 | 0.768 / 1.50 / 0.71 | 0.735 / 1.57 / 0.77 | 0.789 / 1.47 / 0.67 | 0.791 / 1.46 / 0.67 | 0.836 / 1.40 / 0.58 | 0.818 / 1.39 / 0.61 |
| LOYO | 0.321 / 1.84 / 1.20 | 0.225 / 1.95 / 1.30 | 0.389 / 1.79 / 1.16 | 0.404 / 1.77 / 1.14 | 0.473 / 1.70 / 1.06 | 0.473 / 1.69 / 1.05 |
| LORO | 0.389 / 1.84 / 1.19 | 0.205 / 1.98 / 1.34 | 0.321 / 1.81 / 1.20 | 0.316 / 1.78 / 1.20 | 0.538 / 1.67 / 0.99 | 0.496 / 1.68 / 1.03 |

**Reading:**
1. **Both of our models beat every paper model on the paper's field-level metrics**, by large margins out of
   distribution: LOYO 0.473 vs 0.404 (MMGF); LORO 0.50–0.54 vs 0.39 (3D-LSTM). Field RMSE is lower everywhere.
2. **Trade-off:** field context + relation loss sharpens the within-field pattern at a small cost in field-level R²
   relative to the base model: −0.018 (CV10), −0.042 (LORO), unchanged on LOYO.
3. **Paper:** a main table (`tables/field_level.tex`) and a per-pair supplementary table (`tables/supp_field.tex`).
