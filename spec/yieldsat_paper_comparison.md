# YieldSAT Paper-Comparison Protocol

**Status:** Plan, 2026-09-29. Not implemented. It builds on the implemented point
pipeline in [yieldsat_data_contract.md](./yieldsat_data_contract.md) §7. The model
under test is specified in [architecture.md](./architecture.md#9-yieldsat-point-model--exact-architecture).

## 1. Why the current results are not comparable

The runs in `results/yieldsat/` (farm/block/country-held-out, pooled crops, 30-day
pre-harvest cutoff) use a different protocol from the YieldSAT paper (CVPR 2026).
The paper's benchmark ([project results page](https://yieldsat.github.io/result/))
reports:

| Aspect | Paper | Current runs |
|---|---|---|
| Unit of evaluation | **one model per country–crop pair**, nine pairs: ARG-C/S/W, BRA-C/S/W, GER-R/W, URG-S | pooled across crops and countries |
| Protocols | **CV10** (10-fold CV), **LORO** (leave-one-region-out), **LOYO** (leave-one-year-out) | farm, block, country and year holdouts from our own manifests |
| Inputs | Sentinel-2 only, or Sentinel-2 + "ADM" (all auxiliary layers) | all five streams, soil uncertainty off |
| Time window | apparently all 24 time steps (retrospective) | `before_harvest` 30-day cutoff |
| Metrics | R², RMSE at **field level** and **subfield (pixel) level** | same metrics plus field-balanced and macro-country |
| Baselines | Transformer, LSTM, 3D-ConvLSTM, 3D-LSTM (Sentinel-2 or input fusion), AFF, MMGF (feature fusion) | `YieldSATPointModel` only |

The scales are similar. URG-S under CV10 has pixel RMSE 1.19–1.26 t/ha (R²
0.38–0.43) and field R² 0.72–0.82 across the paper's baselines; our pooled
farm-held-out model reaches pixel RMSE 1.20 (R² 0.32) and field R² 0.67 on
Uruguay. The protocols differ, so these numbers must not be compared as a result.

## 2. Protocol facts still to be pinned down

Recover these from the paper or its code before any comparison, and record the
answers here:

- **PC-Q1 CV10 fold grouping.** The release ML tutorial uses `GroupKFold` on
  `field_shared_name`, i.e. one **field season**. Seasons of the same physical
  ground in other years then fall into training folds. Our geometry recovery found
  2,173 seasons on only 1,059 physical fields, with up to 91 seasons chained on
  one Argentine ground.
- **PC-Q2 LORO region definition.** Candidates are `adm_units` in the `Raw.zip`
  metadata (e.g. `adm_1`/`adm_2`), farms, or providers. Also needed: the number of
  folds per country.
- **PC-Q3 LOYO years.** Which years rotate per pair. Whether other seasons of the
  same ground stay in training (the paper most likely allows it; our `year` scheme
  excludes them).
- **PC-Q4 Time window.** All 24 slots, or seeding-to-harvest only. The release
  notebook says observations outside the growing season are masked, but our audit
  found post-harvest dates in stored slots.
- **PC-Q5 Normalization and fill.** Supplied `stats-*` vs. train-fold statistics.
  The tutorial fills NaN with -1.
- **PC-Q6 Metric computation.** Out-of-fold predictions pooled over all folds
  (tutorial behaviour) vs. mean of per-fold metrics. Field level is the mean of
  pixel predictions vs. the mean of pixel targets per `field_shared_name`
  (tutorial), not the `yield_ground_truth` attribute.
- **PC-Q7 Training budget and model selection.** Epochs, early-stopping data (a
  held-out validation fold or none) and seeds.

## 3. Implementation tasks

| Task | Deliverable | Acceptance |
|---|---|---|
| **PC-01 Paper-compatible split schemes** | Add `cv` to `dataset/yieldsat_splits.py`: k folds within one country–crop, grouped by season (`--cv_group season`, paper-compatible) or physical field (`--cv_group physical`, leakage-safe). Add `loro` grouped by the PC-Q2 region key. Add LOYO rotation over all years with `--loyo_policy paper` (keep other seasons of test ground) or `strict` (current behaviour). One manifest per fold, validation drawn from training groups. | Folds cover every season exactly once as test; `check_split` passes for the chosen grouping; manifests record the policy. |
| **PC-02 Per-pair training** | `--crops` filter in `main_yieldsat_finetune.py` restricting all partitions to one crop; crop context off automatically when one crop is selected. | A run on `Germany` + `rapeseed` sees only rapeseed seasons; report states the pair. |
| **PC-03 Input configurations** | Paper **S2** = `--streams yieldsat_s2`. Paper **S2 + ADM** = all streams (optionally `--soil_uncertainty ancillary`, since ADM likely includes it). Both with `--cutoff_mode all_slots` (labelled retrospective). Optional `--norm_source supplied` and `--fill_value -1` ablations to mirror PC-Q5, reported separately. | Report label policy says retrospective; stream lists match the paper row. |
| **PC-04 Fold aggregator** | `yieldsat_collect_results.py --aggregate_folds` (or a sibling script) merges `test_predictions.npz` of all folds per pair and computes pixel and field R²/RMSE on pooled out-of-fold predictions, plus mean ± std over folds. | Each cell appears once across folds; metrics reproduce `util/yieldsat_eval.py` on a single fold. |
| **PC-05 Protocol validation** | Re-implement the paper's small LSTM baseline (S2 only, 1 layer, hidden 64, last-step readout, MSE, Adam 1e-3, NaN→-1, as in the release tutorial) as a registered point baseline. Run it on GER-R, CV10. | Result within the paper's LSTM row tolerance, or the discrepancy explained, **before** our model's numbers are compared. |
| **PC-06 Run matrix** | CV10: 9 pairs × 10 folds × {S2, S2+ADM} = 180 runs of `YieldSATPointModel` (+ LSTM baseline for validation). LORO: one run per region per pair. LOYO: one run per year per pair. Store under `results/yieldsat/paper_protocol/`. | Estimated 3–5 GPU-hours for CV10 at 2–5 min/run with three concurrent runs; runs launched from a checked-in script. |
| **PC-07 Comparison table** | Transcribe the paper table into `results/yieldsat/paper_benchmark.csv` (protocol, modalities, fusion, model, pair, level, R², RMSE). Produce a side-by-side table per protocol and pair with our model, the LSTM re-run and the paper's best baseline. | Every row states split grouping, time window and normalization; differences from the paper protocol are listed next to the numbers. |
| **PC-08 Leakage sensitivity** | Repeat CV10 with `--cv_group physical` and LOYO with `strict`. | Paper-compatible and leakage-safe numbers reported together; the gap quantifies cross-season leakage. |

## 4. Rules for reporting

- Label every paper-protocol result **retrospective** when it uses all 24 slots.
  Keep our pre-harvest (`before_harvest`) results as a separate table.
- Keep known protocol differences visible next to the numbers:
  - train-fold normalization vs. supplied statistics;
  - masking of weather at each cell's first dated slot;
  - the season vs. physical-field grouping.
- Report per pair. Pooled or macro numbers are extra and never replace the
  per-pair table.
- Single-seed differences smaller than the fold-to-fold standard deviation are not
  claims of improvement.
