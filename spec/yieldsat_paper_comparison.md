# YieldSAT Paper-Comparison Protocol

**Status:** 2026-09-29: tooling implemented (PC-01–PC-04, PC-06, PC-07). PC-05
protocol check run on GER-R and PC-08 leakage quantified. See §5 for the
progress log. Originally a plan, 2026-09-29. It builds on the implemented point
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

## 5. Progress log

### 2026-09-29 — Protocol questions answered from the paper

Sources: the paper text (arXiv:2604.00940 v1, §5 and Appendix A.3) and the
release tutorial.

| Question | Answer | Consequence |
|---|---|---|
| PC-Q1 CV10 grouping | "stratified, grouped 10-fold cross-validation, where pixels are grouped by field and stratified by region" (§5). "Field" means a field season (`field_shared_name`), as in the tutorial. | `cv` scheme with `--group season` (paper). `physical` is our leakage-safe variant. |
| PC-Q2 LORO region | "a set of fields belonging to a single farmer or to a local data provider" (§5.2). | Region = `farm_identifier` (country-qualified), one fold per farm; `strict` uses farm clusters. Fold counts: ARG-C 29, ARG-S 44, ARG-W 21, BRA-C 7, BRA-S 9, BRA-W 6, GER-R 6, GER-W 6, URG-S 10. |
| PC-Q3 LOYO | One fold per year. The paper does not say whether other seasons of the same ground stay in training; we assume they do (`paper` policy). | `strict` excludes them (e.g. BRA-S: 1,442 season-fold exclusions). |
| PC-Q4 Time window | "a unified time series of 24 time steps, encompassing all available data modalities" (§4.5). | `--cutoff_mode all_slots` (labelled retrospective). |
| PC-Q5 Normalization | **Not stated.** The tutorial uses raw values with NaN→-1 and a raw target. | `--normalization {train,supplied,none}`, `--target_normalization`, `--fill_value`; tested in PC-05. |
| PC-Q6 Metrics | "The metrics are presented as the average across the folds"; field level: "all pixels in the same field are averaged and compared with the field's averaged ground truth" (§5). Appendix tables give mean ± std over folds. | The fold aggregator reports fold mean ± std (primary) plus pooled out-of-fold values. |
| PC-Q7 Hyperparameters | **Not stated** in the paper (epochs, optimizer, model selection, seeds). The deep ensembles in §5.2 use 5 members. | The LSTM preset follows the tutorial: Adam 1e-3, batch 1028, 15 full-pass epochs, MSE, hidden 64. |
| Model selection | Not stated. The tutorial reports the best epoch on the evaluated split, which is optimistic. | We select on a validation carve-out (10% of training groups, `--val_frac`); `--val_frac 0` uses the last epoch. |

Source discrepancies, recorded in `results/yieldsat/paper_benchmark.csv`
(`sources_agree`):
- 517/540 means agree between the PDF appendix and the project results page.
- In the PDF, the LORO "S2+ADM input-fusion 3D-ConvLSTM" rows (Tables 17/18)
  duplicate the S2 3D-ConvLSTM row, while the web page has distinct values.
- The GER-W LSTM CV10 field R² is 0.55 in the PDF and −0.87 on the web page.
- Three URG-S cells differ by 0.01–0.02.
- The Table 16 caption says "LORO, field level", but its header and content are
  LOYO pixel level.

### 2026-09-29 — PC-01 to PC-04, PC-06, PC-07 implemented

- **PC-01** `dataset/yieldsat_splits.make_paper_folds` and
  `yieldsat_prepare.py folds` produce per-pair fold manifests:
  - `cv`: sklearn `StratifiedGroupKFold` on pixels (one sample per 100 cells),
    grouped by season or physical field and stratified by region;
  - `loro` and `loyo` as above;
  - `paper` or `strict` leakage policy, with a 10% validation carve-out;
  - an index file `<prefix>.folds.json` per experiment.

  Coverage (every season is test exactly once) and the fold-key separation are
  checked. Each manifest records `physical_overlap_test_seasons`.
- **PC-02** `--crops`: all partitions are restricted to the pair's crop, and the
  crop token is disabled automatically for a single crop.
- **PC-03** Inputs: `s2` = `--streams yieldsat_s2`; `s2_adm` = all five streams.
  New options: `--normalization {train,supplied,none}`,
  `--target_normalization {train,none}`, `--fill_value`, `--model paper_lstm`,
  `--optimizer {adamw,adam}`, `--lr_schedule {cosine,constant}`, `--grad_clip`,
  `--no_amp`, and `--steps_per_epoch 0` (full pass). Reports record the pair
  filter and normalization policy.
- **PC-04** `yieldsat_collect_results.aggregate_folds`: fold mean ± std of pixel
  and field R²/RMSE plus pooled out-of-fold metrics. It refuses cells that
  appear in two test folds. Output: `aggregate.json` per pair.
- **PC-06** `yieldsat_paper_runs.py`: builds missing fold manifests, runs the
  (protocol × inputs × model × pair × fold) matrix with a concurrency limit,
  resumes finished folds, and aggregates. `--epochs`/`--steps_per_epoch`
  overrides, `--tag` names long-training variants, and extra arguments after
  `--` go to `main_yieldsat_finetune.py`. Presets: `ours` (point model,
  `--fusion` choice) and `paper_lstm`. Layout:
  `runs/paper/<protocol>_<group>_<policy>_s<seed>/<inputs>/<model>[_tag]/<pair>/fold<ii>/`.
- **PC-07** `util/yieldsat_paper_tables.py` transcribes appendix Tables 13–18
  from the PDF text layer into `results/yieldsat/paper_benchmark.csv`: 540 rows
  of mean ± std, cross-checked against the results page.
  `yieldsat_paper_compare.py` writes `comparison.csv` and `comparison.md`: the
  paper's LSTM, its best model per input set, and our experiments per protocol,
  level and pair. Leakage-sensitivity runs are marked separately.
- **Tests:** 4 new tests (fold protocols and policies, table-cell splitting,
  fold aggregation with duplicate detection, normalization policies and the LSTM
  baseline). Suite: 81 passed, 2 expected failures.

### 2026-09-29 — PC-08 leakage quantified (fold manifests, seed 0)

For each pair, the table counts test seasons (summed over folds) whose physical
ground also appears in the training partition under the paper-compatible
policy. The strict policy removes all of this overlap; for LOYO it does so by
excluding seasons.

| Pair | CV10 (season grouping) | LORO (farm) | LOYO (year) | LOYO strict: excluded season-folds |
|---|---|---|---|---|
| ARG-C | 73 | 14 | 57 | 89 |
| ARG-S | 211 | 46 | 208 | 357 |
| ARG-W | 71 | 2 | 64 | 99 |
| BRA-C | 115 | 4 | 111 | 272 |
| BRA-S | 283 | 23 | 283 | 1,442 |
| BRA-W | 138 | 13 | 138 | 395 |
| GER-R | 35 | 0 | 34 | 36 |
| GER-W | 149 | 0 | 146 | 357 |
| URG-S | 130 | 0 | 136 | 165 |

Under the paper's CV10, a large share of test seasons have the same ground in
training from other years. For example, BRA-S has 283 of 293 seasons and GER-W
149 of 188. Soil and terrain inputs are then identical between train and test
cells, so paper-protocol numbers are likely optimistic for ADM inputs. Always
report `--group physical --policy strict` alongside them.
