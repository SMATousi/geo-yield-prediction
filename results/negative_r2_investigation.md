# Investigation: negative R² in the point and image suites (2026-10-02)

**Question.** Many point-suite and image-v1 rows show negative R², while the
paper's rows are mostly positive. Is that a bug in the model, outputs, training,
or in how we compute or compare the numbers?

**Short answer.** It is mainly a **metric mismatch**, not a bug:
- **We report** the mean of per-fold R² over test folds. In LOYO/LORO those
  folds are a single year or region with low yield variance, so R² goes
  negative easily.
- **The paper's means behave like R² on predictions pooled over all folds**
  (the release tutorial's computation).
- **Under the paper's computation**, our point model ties the paper's LSTM.
  It is still below the paper's best models, and the image v1 model is
  genuinely weaker.

Below: evidence, the causes ranked by impact, and candidate solutions for
decision. **Nothing has been changed yet.**

## 1. Evidence

### 1.1 Our faithful paper LSTM shows the same "negative" pattern

Our re-implementation of the paper's pixel LSTM reproduces the paper in CV10:
GER-R pixel R² 0.33 vs 0.36, and over all 9 pairs a per-fold mean of 0.34 and
a pooled value of 0.37, vs the paper's 0.37. We ran it on ARG-W LOYO (7 year
folds, local RTX 3090):

| ARG-W LOYO, S2 | Per-fold mean R² (our reporting) | Pooled R² | RMSE |
|---|---|---|---|
| Paper LSTM (paper's table) | **0.61 ± 0.73** | – | 1.49 |
| Our re-run of the paper LSTM | **−0.94 ± 1.51** | **0.51** | 1.71 |
| Our point model | −0.24 | **0.58** | 1.47 |

Per fold, the re-run gives R² 0.69, 0.63, 0.63 on the three large years and
−2.4 to −2.7 on the three small ones (test std 0.9–1.2 t/ha). The paper's
number is reproduced by the **pooled** value, not the per-fold mean, and our
model is better than the LSTM on both.

### 1.2 The paper's own mean ± std is impossible as a per-fold mean

R² ≤ 1 per fold. With n folds and mean m, the per-fold standard deviation
cannot exceed √(n·(1 − m)²) (achieved with n − 1 folds at 1 and one low
fold). With our fold counts:
- **31 paper rows violate this bound**, all in LOYO/LORO (14 LOYO field,
  2 LOYO pixel, 15 LORO field), and none in CV10.
- Examples: ARG-W LOYO field LSTM 0.77 ± 1.70 (max possible 0.61); ARG-C
  LOYO field AFF 0.33 ± 10.45 (max 1.9).

So the paper's means are not the averages of the per-fold values whose std
it reports. Pooled out-of-fold R² explains both.

### 1.3 Per-fold mean vs pooled R², everywhere (paper policy, mean over pairs and seeds)

| Suite | Protocol | Per-fold mean | Pooled | Paper LSTM (S2) |
|---|---|---|---|---|
| Point | CV10 | 0.43 | 0.44 | 0.37 |
| Point | LORO farm (BRA/GER/URG) | −0.05 | 0.06 | 0.01 |
| Point | LORO province (ARG) | 0.20 | 0.45 | 0.53 |
| Point | LOYO | −0.06 | 0.19 | 0.15 |
| Paper LSTM re-run | CV10 | 0.34 | 0.37 | 0.37 |
| Image v1 | CV10 | 0.17 | 0.25 | 0.37 |
| Image v1 | LOYO | −0.40 | 0.07 | 0.15 |

In CV10 the folds are random, so both computations agree and we match the
paper. In LOYO/LORO they diverge strongly.

LOYO per pair, S2, point model:

| Pair | Per-fold mean | Pooled | Paper LSTM |
|---|---|---|---|
| ARG-C | 0.12 | 0.35 | 0.39 |
| ARG-S | 0.26 | 0.48 | 0.46 |
| ARG-W | −0.24 | 0.58 | 0.61 |
| BRA-C | −0.13 | 0.08 | 0.15 |
| BRA-S | −0.09 | 0.05 | 0.04 |
| BRA-W | −0.13 | −0.07 | 0.08 |
| GER-R | −0.40 | −0.21 | −0.05 |
| GER-W | −0.08 | 0.02 | −0.68 |
| URG-S | 0.07 | 0.29 | 0.33 |

### 1.4 Where negative per-fold R² happens (all finished runs, W&B)

| | Point (2,580 runs) | Image v1 (2,456 runs) |
|---|---|---|
| Negative pixel R², all runs | 23% | 41% |
| CV10 / farm LORO / province LORO / LOYO | 2% / 37% / 17% / 46% | 21% / 62% / 34% / 60% |
| Test folds with ≤ 10 field seasons | 36–41% | 42–65% |
| Test folds with > 50 field seasons | 11% | 18% |

Removing each fold's mean bias raises LOYO per-fold R² by 0.1–0.2, so
year-level bias is part of the error. The larger part is the small test-fold
variance.

### 1.5 No pipeline bug found

- **Predictions align with targets:** all 2,449 image folds matched the
  point predictions cell by cell with identical targets.
- **Denormalization is right:** the median |bias| is 0.12 t/ha on large
  folds.
- **The paper's protocol is reproduced in CV10** by the LSTM re-run.
- **Field-level R²** is more negative still because a LOYO fold may hold only
  6–15 field seasons. The same metric issue applies.

## 2. Causes, ranked

| # | Cause | Effect | Evidence |
|---|---|---|---|
| C1 | **Metric mismatch.** We average per-fold R²; the paper's numbers behave like pooled out-of-fold R². | Most of the "negative" gap in LOYO/LORO (up to 0.8 R² per row); none in CV10 | §1.1–1.3 |
| C2 | **Per-fold R² is fragile on small test folds** (a single year or region with ≤ 15 seasons and low variance). | Negative values even at RMSE equal to the paper's | §1.1, 1.4 |
| C3 | **Point model below the paper's best models** (spatial 3D-ConvLSTM/3D-LSTM and AFF fusion): pooled CV10 0.43–0.45 vs 0.49–0.53; LOYO 0.17–0.21 vs 0.32–0.37; province LORO 0.44–0.46 vs 0.61–0.66. | A real gap of ~0.05–0.2 | §1.3 |
| C4 | **Weather/terrain/soil (ADM) adds almost nothing.** The paper's input-fusion LSTM gains up to +0.16 in LOYO; we gain +0.04. | Part of C3 | §1.3; earlier GER-R analysis |
| C5 | **Training stops being useful within the first epochs.** The best validation epoch is 4–7 of 60 (median), with ≤ 5 in 55–60% of runs. Validation seasons come from the training years and regions, so model selection does not see the year/region shift. | Overfitting to training fields; ~90% of compute unused; selection blind to LOYO/LORO shift | W&B `best_epoch`; ARG-W LOYO histories |
| C6 | **Year/region level bias.** | 0.1–0.2 per-fold R² in LOYO | §1.4 (debiased R²) |
| C7 | **Image v1: overfitting and noisy selection.** 6.3M parameters on ~200–700 tiles per pair; training loss falls to 0.27 (standardized) while validation R² stays near 0; validation has only 20–35 tiles, so the "best" epoch is random (e.g. 1, 17, 29). | Image v1 worse than point on identical cells in 97/108 rows | ARG-C histories; `image_vs_point.md` |
| C8 | **Image v1: only 4 optical dates** (the point model sees all 24 slots) **and partial coverage** (26–63% of cells). | Weaker temporal signal; metrics only on the tiled subset | design; addressed in v2 |

## 3. Candidate solutions (for decision)

| # | Solution | Addresses | Cost | Notes |
|---|---|---|---|---|
| S1 | **Report pooled out-of-fold R² and RMSE (pixel and field) as the headline**, with per-fold mean ± std alongside, in all `results/` tables. Ask the authors to confirm their computation. | C1, C2 | CPU only; existing predictions | Changes only the reporting; makes the paper comparison like for like |
| S2 | **Model selection that sees the shift:** for LOYO, validate on one held-out training year; for LORO, on a held-out training region (group-aware validation). Optionally checkpoint averaging/EMA. | C5, C6 | Re-run the suites | Better checkpoints for out-of-distribution folds |
| S3 | **Retune the point training recipe on a few pairs** (CV10 + LOYO on ARG-S, ARG-W, BRA-S, GER-R): lower LR, stronger weight decay and dropout, fewer epochs (≈ 10–15), field-level sampling, 5-seed ensembles like the paper's deep ensembles. | C3, C5 | Small sweep (~1 GPU-day), then a suite re-run (~1/4 of the current cost if epochs drop) | Today's 60 epochs mostly overfit after epoch 5 |
| S4 | **Input-level (early) fusion of ADM** as in the paper's LSTM: concatenate weather/terrain/soil to the S2 bands per time step, alongside or instead of late token fusion. Also test soil uncertainty and coordinates (with the leakage-safe `strict` policy reported next to it). | C4 | Model change + suite re-run | The paper's largest ADM gains come from simple input fusion |
| S5 | **A year/region level term for the point model**, like the v2 image level head: a season-level head (weather, crop, timing) predicting the field-season mean, plus a residual. | C6 | Model change + re-run | v2 tests this idea for images |
| S6 | **Spatial context for the point model:** a 3×3 or 5×5 neighbourhood patch per cell (the paper's best models are spatial), or rely on image v2. | C3 | Model/dataset change + re-run | Directly targets the gap to 3D-ConvLSTM |
| S7 | **Image model regularization:** flips and 90° rotations (yield-invariant augmentation); smaller width (D = 128) and dropout; pooled multi-pair pretraining then per-pair fine-tuning; validation on all tiles of the validation seasons (v2 already does this); last-k checkpoint averaging instead of a single noisy best epoch. | C7 | Model change + re-run of the image suite | v2 adds the time series, level head and coverage but not augmentation or regularization |
| S8 | **Let image v2 finish before changing it** and judge it with S1's pooled metric on identical cells. | C7, C8 | Already running | Tells whether the temporal branch closes the gap |

**Suggested order:**
1. S1 now: no training cost; it immediately shows where we stand in the
   paper's terms.
2. S8: v2 results in a few hours.
3. A small sweep combining S2 + S3 (+ S4) on a few pairs before any full
   re-run.
4. S5–S7 according to the sweep.
