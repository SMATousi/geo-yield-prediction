# Experiments for review robustness

**Status (2026-10-12): adopted (project lead). RX-1, RX-2 and RX-3 in progress.**

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
