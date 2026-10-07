# Dense raw Sentinel-2 series for the point models

**Status (2026-10-06): specified; implementation started (project lead:
"write the specs and start training"; train with 64 GPUs once ready).**

Related:
- [yieldsat-paper-protocol.md](./yieldsat-paper-protocol.md): protocol and
  scoring used for every result here.
- [yieldsat-paper-reproduction.md](./yieldsat-paper-reproduction.md): D5 and
  RP-07, the dense series the paper's MMAF/AFF model used.
- [yieldsat_data_contract.md](./yieldsat_data_contract.md): the monthly
  contract this extends.

## 1. Why

- **What our models train on now.** The preprocessed YieldSAT release,
  which keeps **one acquisition per calendar month** per field (24 slots).
- **What `Raw.zip` holds.** Every Sentinel-2 acquisition of every field
  season: 261,880 rasters for ~2,200 field seasons, ~118 per season on
  average. It also has a per-pixel scene classification (SCL) mask for
  each acquisition and 10 daily weather variables per field.
- **Why it matters for the gap.**
  - The thesis' MMAF (the paper's "AFF", best model in 14 of 24 DEV row ×
    level entries) used the dense series and daily weather.
  - Under the paper's protocol and pooled scoring we beat the paper's LSTM
    in every protocol. We are still 0.02–0.07 pixel R² below its best
    model (spec/yieldsat-paper-protocol.md §6).

## 2. Verified facts about `Raw.zip` (2026-10-06)

- **Location.** `/home1/pupil/SMATousi/YieldSAT/Raw/Raw.zip` (19.5 GB) and
  `/data/YieldSAT/Raw.zip` on the PVC.
- **Layout.** One folder per field season
  `<Country>/<field_shared_name>/` with:
  - `s2_images/S2_L2A_<YYYYMMDD>.tif`: 12 bands B01–B12 (B8A included,
    canonical order), uint16, nodata 0, 10 m, the field's UTM CRS;
  - `scl_masks/S2_L2A_SCL_<YYYYMMDD>.tif`: the per-pixel SCL class for the
    same date;
  - `weather/<name>.csv`: daily rows starting before seeding. Columns:
    Temp_mean/max/min (K), Total_prec (m), S_rad, Temp_dew, Wndsp,
    RH_min/mean/max;
  - `dem/` (5 rasters), `soil/` (8 properties, 0–200 cm),
    `yield_masks/`, `metadata-*.json`.
- **Exact parity with our cache.** Indexing a raw raster at the cache's
  `(grid_row, grid_col)` gives exactly the cache's S2 values at the
  acquisition dates the monthly release kept.
  - Checked on Argentina_DUP1_farm15_field130_soybean_2018: 4 dated slots,
    10,575 cells each, maximum |difference| 0.
  - The same field has 27 acquisitions in its season; the release keeps 4.
- **Weather.** The release's weather is the inclusive sum of these daily
  values between consecutive kept dates (verified earlier,
  spec/yieldsat-paper-reproduction.md).

## 3. Design

| ID | Decision |
|---|---|
| D1 | **Slot grid: 10-day bins.** 3 per calendar month (days 1–10, 11–20, 21–end) from January of harvest year − 1 to December of the harvest year: **72 slots**. Slot k ↔ month k // 3, third k % 3. This refines the monthly contract (slot m = bins 3m..3m+2) and keeps its season window: bins outside [seeding, harvest] are masked |
| D2 | **Per cell and bin:** the clear acquisition (SCL 4 vegetation, 5 bare soil, 6 water) closest to the bin centre. If none is clear, the bin is masked for that cell. S2 values are the raw DN, which equal the cache's (§2) |
| D3 | **Weather per bin:** the sum of the daily values over the bin's days inside the season window. This is the release's interval-sum operator at bin resolution, field level. The 4 contract channels (temp_max, temp_mean, temp_min, total_prec) form v1; the 6 extra daily variables are a later option (D9) |
| D4 | **Times:** days since 1970 of the selected acquisition per cell and bin. A weather-only bin (no clear S2) keeps the bin centre date |
| D5 | **Storage:** `<artifact_root>/cache_dense/<Country>/`. `temporal.npy` float16 (N, 72, 16) in the canonical 16-channel order; `times.npy` float32 (N, 72); `static.npy`, field stats and the index are reused from `cache/` (same rows, same order); `manifest.json` (n_slots 72, bin rule, SCL classes, source) |
| D6 | **Neighbourhood stream:** `yieldsat_build_neighbourhood.py` generalized to T slots. It writes `neighbourhood_dense/<Country>/s2_nbr5.npy` (N, 72, 12) from the dense cache |
| D7 | **Code:** the slot count comes from the cache manifest (default 24). It flows through the cache helpers, `YieldSATPointDataset`, the flat (TabM) view, the neighbourhood builder and `YieldSATPointModel(num_slots=…)`. Selection: `--cache_name cache_dense` (and `--neighbourhood_name`) |
| D8 | **Leakage:** inputs only (no labels, no split), like the monthly cache. Normalization stays train-only per fold |
| D9 | (later) The 6 extra daily weather variables as additional temporal channels |

Expected size: 12.4M cells × 72 × 16 × 2 B ≈ 28 GB for temporal, plus
3.6 GB for times and ~21 GB for the dense neighbourhood. It is built on the
PVC (local disk has 67 GB free).

## 4. Models and protocol

The protocol is spec/yieldsat-paper-protocol.md:
- per-pair, no validation set, selection on the test fold;
- pooled out-of-fold scoring;
- all 9 pairs × CV10/LOYO/LORO (217 folds).

| Tag | Model | Compared with |
|---|---|---|
| `ours-p3nbr-dense` | p3-nbr on the dense cache (72 slots) + dense 5×5 neighbourhood | `ours-p3nbr` (monthly) |
| `tabm-f1-dense` | TabM F1 (fixed configuration) on the flat dense view | `tabm-f1` |
| `tabm-f0-dense` | TabM F0 on the flat dense view | `tabm-f0` |

The comparison is paired, on the same folds. Success means a pooled
pixel R² gain over the monthly version (mean over pairs) with a paired
fold bootstrap CI excluding 0.

## 5. Cluster execution: 64 GPUs without the utilization flag

Rule (project lead): pods must never fall below 20% use of their
requested GPU memory or RAM (spec/yieldsat-paper-protocol.md §7; memory
note).
1. **Measure before submitting.** Run each model locally on the largest
   pair (ARG-S) and the smallest (GER-R). Record:
   - peak PSS of the process and its loader workers;
   - peak GPU memory;
   - CPU use.
2. **Pack runs per GPU.** Use `pool --concurrency N` so each GPU's memory
   use stays ≥ 25% of the card. N comes from the measured GPU memory of
   one run.
3. **Request memory at ~1.3–1.5× the measured peak** of N concurrent runs,
   and check the smallest pair still stays above 20%.
4. **Request CPU** at the measured CPU of N runs.
5. **Verify with `kubectl top` and `nvidia-smi`** on the first pods. Fix the
   job files before scaling to 64 GPUs.
6. **The dense build is a CPU job,** sized the same way (measured first on
   one country).

## 6. Tasks

| ID | Deliverable | Acceptance |
|---|---|---|
| DS-01 | `yieldsat_build_dense.py` (D1–D5): reads `Raw.zip` field folders, writes `cache_dense/` | Parity: for every dated monthly slot of ≥ 20 sampled seasons, the dense bin holding that date reproduces the cache values exactly where that acquisition is selected. Row counts and order equal `cache/`. Clear-observation statistics reported |
| DS-02 | D7: slot count from the cache manifest through the dataset, flat view, neighbourhood builder and model | Unit tests: 24-slot behaviour unchanged (same batches as before); a 72-slot cache trains end to end locally |
| DS-03 | Dense neighbourhood (D6) | Shape/alignment checks; NaN fraction reported |
| DS-04 | Cluster build of the dense cache and neighbourhood on the PVC | All 4 countries; manifest written |
| DS-05 | Resource measurement (§5) and job files | Requests within 1.3–1.5× of measured peaks; first pods ≥ 20% on GPU memory and RAM |
| DS-06 | Training: 3 models × 217 folds on 64 GPUs | All units done or explained |
| DS-07 | Report: paired dense vs monthly, pooled scoring, per pair + mean | Added to this spec and results/ |

## 7. Log

- 2026-10-06:
  - Spec written.
  - Raw layout and the exact raw↔cache parity were verified (§2).
  - Before starting, every cluster job was stopped on the project lead's
    request. Their results so far were gathered: pretrained-encoder p3-nbr
    99/326 units in `results/protocol_pk_partial.json`; the image batch's
    53 units are on the PVC.
- 2026-10-06, DS-01 (local test, 20 German field seasons, 4.4 s on 8 workers):
  - 2,052 in-season acquisitions (~103 per season).
  - Clear S2 slots per cell: **14.1 dense vs 6.9 monthly**.
  - **Parity:** in 37,372/46,669 sampled monthly dated slots the dense bin
    selected the same acquisition. All 37,372 equal the monthly values
    exactly after float16 rounding (max |diff| 2 DN; float16 steps 2 above
    2,048 and 4 above 4,096, i.e. ≤ 0.05% relative).
  - The other slots differ because the dense rule takes a clear acquisition
    nearest the bin centre, while the release takes one acquisition per
    field and month regardless of per-pixel cloud.
  - Weather bins: temp_mean ~2,850 K·day per 10 days; precipitation 0.01–0.04 m.
  - D7: slot count and cache directories come from `YIELDSAT_NUM_SLOTS`,
    `YIELDSAT_CACHE_NAME` and `YIELDSAT_NBR_NAME` (defaults unchanged). They
    reach the cache, dataset, flat view, neighbourhood builder and
    `YieldSATPointModel(num_slots)`. Pool units can set them via `env`.
    Existing tests pass (40).

- 2026-10-06, DS-02/03/05 (local, full Germany and Argentina dense builds):
  - **Build speed:** Argentina dense cache in 3.4 min (16 workers); Germany
    in ~30 s. Dense field stats are computed in the builder: temporal from
    the dense values, static copied from the monthly cache.
  - **p3-nbr dense trains end to end at 72 slots.**

    | | Peak PSS (process + loader workers) | GPU memory |
    |---|---|---|
    | GER-R | 3.0 GiB | 3.1 GB |
    | ARG-S | 10.8 GiB | 3.1 GB |

  - **Early signal, same tiny budget (2 × 200 steps), ARG-S LOYO fold 4:**
    dense test pixel R² 0.484 vs monthly 0.158. GER-R CV fold 1:
    0.419 / 0.70 field. These are smoke numbers, not results.
  - **TabM dense (F0 ~1,400 columns, F1 ~2,300):** CUDA out of memory even
    on GER-R (F0 > 21 GB, F1 tried to allocate 9 GB more). The embedded
    column width is too large for the current TabM set-up. **Deferred:** it
    needs a design change (fewer columns, e.g. monthly summaries of the
    dense series, or a smaller batch / embedding). The first training wave
    is p3-nbr dense only.
  - **Builders:** dense build 6.9 GiB PSS (16 workers, Germany);
    neighbourhood 3.3 GiB. Cluster build job: 16 CPU / 12 Gi, no GPU.
  - **64-GPU sizing (§5):** 3 runs per GPU on 24 GB cards only (A10, RTX
    3090, RTX 4090).
    - GPU memory ≈ 9.3 GB of 24 GB (39%).
    - RAM request 36 Gi: 3 ARG-S runs ≈ 32 GiB (90%), 3 GER-R runs
      ≈ 9 GiB (25%).
    - 10 CPU, 4 Gi shm.
    - 64 pods × 3 = 192 slots for 217 units. The last units run one per GPU
      briefly at the tail.
- 2026-10-06, DS-04 (cluster build):
  - **First attempt:** reading `Raw.zip` randomly over CephFS left the
    workers I/O-bound (0.47 of 16 CPUs).
  - **Second attempt:** with `Raw.zip` copied to local scratch first
    (9 min), it was OOM-killed at 12 Gi. The memory had only been measured
    on Germany; Argentina peaks at 26 GiB PSS with 12 workers, mostly shared
    output pages.
  - **Resubmitted** at 32 Gi / 12 CPU.
- 2026-10-06, build memory:
  - **Third cluster attempt:** OOM-killed at 32 Gi. Profiling (per-process
    PSS against the build phase) found four causes, all fixed without
    changing the output (verified identical):

    | Cause | Fix |
    |---|---|
    | The parent re-read the lazy `rows.npz` per field and kept 751 full copies alive (24.7 GB) | Load the row arrays once and copy each field's slice |
    | Forked workers duplicated the parent heap | Spawned workers read zip members by offset (pread + inflate), without parsing the 266k-entry directory |
    | Output memory maps (12 GB file pages) | Per-field `pwrite` with fdatasync + page drop |
    | The stats pass mapped the whole output | Block `pread` |

  - **Result for Argentina:** dense build 26 → **2.3 GiB** peak PSS;
    neighbourhood builder (same treatment) 19.8 → **7.7 GiB**.
  - **Saturated S2 DN above float16's range** (18,103 values in Argentina,
    mostly B01, ~1e-4) are now dropped as NaN instead of becoming inf.
  - **Build job:** 12 CPU / 12 Gi.
- 2026-10-06, DS-04 continued:
  - **Fourth attempt:** failed on one raster read from the pod's local copy
    of `Raw.zip`; the pod's logs were lost with the job.
    - A diagnostic pod read ~14,000 Argentina rasters (every 7th) straight
      from the PVC with no failure. The cluster and local stack are
      identical (rasterio 1.4.4, GDAL 3.10.3).
    - The builder now reads the PVC directly, loops on short `pread`s, and
      retries an unreadable acquisition once, then skips and counts it
      (`unreadable_acquisitions_skipped` in the manifest).
  - **The PVC read is throughput-bound:** ~0.2–0.5 CPU and 0.3–1 GiB at
    12 or 32 workers, ~3.4 min per 100 Argentina fields.
    - Resized to 1 CPU / 4 Gi (measured 54% CPU, 26% memory).
    - The neighbourhood step is a separate job (`dense_nbr_job.yaml`,
      2 CPU / 10 Gi for its 7.7 GiB peak), started automatically after the
      build.

## 8. Results (DS-06/07, 2026-10-07)

All 217 units are done; none were lost. Hardware failures during the run:
- **Two GPUs with memory held outside our pods:** the pool's GPU check now
  requires ≥ 3 GiB free. Affected units were re-queued.
- **One bad node:** excluded in the job files.

Full per-pair tables: `results/dense_p3nbr_final.md`. Data:
`results/dense_p3nbr_final.json`. Pooled out-of-fold R², mean over the
9 pairs (pixel / field):

| Protocol | Paper LSTM S2+ADM | Paper best | Monthly p3-nbr | **Dense p3-nbr** | Dense ≥ paper best (pixel) | Dense − monthly (paired fold bootstrap 95% CI) |
|---|---|---|---|---|---|---|
| CV10 | 0.46 / 0.72 | 0.53 / 0.82 | 0.51 / 0.80 | **0.54 / 0.84** | 5/9 pairs | pixel +0.029 [+0.022, +0.037]; field +0.036 [+0.024, +0.047] |
| LOYO | 0.31 / 0.43 | 0.38 / 0.54 | 0.35 / 0.45 | 0.36 / 0.47 | 5/9 pairs | pixel +0.010 [−0.032, +0.054]; field +0.026 [−0.058, +0.103] |
| LORO | 0.23 / 0.22 | 0.39 / 0.58 | 0.32 / 0.46 | 0.37 / 0.54 | 3/9 pairs | pixel +0.047 [+0.029, +0.071]; field +0.075 [+0.031, +0.144] |

Findings:
1. **The dense series helps where it is measurable.**
   - CV10 and LORO gains are significant (CIs exclude 0).
   - LOYO is positive on average but not significant. It improves strongly
     on ARG-C, ARG-W, BRA-S, BRA-W and URG-S, and regresses on BRA-C
     (0.12 vs 0.28 monthly) and GER-W (0.08 vs 0.22).
2. **CV10: we beat the paper's best model on average:** 0.54 vs 0.53 pixel
   and 0.84 vs 0.82 field. This is the first time any of our models
   exceeds it (same protocol and metric).
3. **LOYO / LORO: still below the paper's best on average** (−0.03 / −0.02
   pixel), though above it on 5/9 and 3/9 pairs.
4. **Next candidates:**
   - The two LOYO regressions (BRA-C, GER-W): year-level offset; check the
     dense weather sums for those years.
   - Dense TabM: needs a narrower flat view.
   - Pretrained encoders at 72 slots.

## 9. Knowledge pretraining on the dense series (project lead, 2026-10-07)

The 24-slot pretrained encoders (pk_dev1r) cannot load into the 72-slot
model, so pretraining is re-run on the dense cache.

| Item | Decision |
|---|---|
| Arms | **A3** (SSL + knowledge grounding + relational distillation) and **A7** (the same with random targets; control), as in spec/yieldsat-point-knowledge-pretraining.md |
| Units | The same 46 leakage-audited pretraining units (`pretrain_units_dev_s0`): all 4 countries, inputs only, each excluding the seasons its folds test on |
| Model / budget | As pk_dev1r: point model, perceiver summary, S2+ADM, no neighbourhood stream, 30 epochs × 500 steps × 512 cells, `--norm_pooling per_country`, `--pretrain_diagnostics`. Dense env (72 slots), `--weather_first_slot keep` |
| Knowledge targets | Unchanged: the per-cell concept indices (`knowledge/<Country>/concept_raw_c30.npz`, from the monthly cache; label-free cell-season properties that do not depend on the slot layout) and the frozen CLIP text vectors |
| Fine-tuning | Dense p3-nbr (§4 budget) from each fold's unit checkpoint. Encoders only; the neighbourhood encoder starts fresh (`--encoders_only_transfer --init_nonstrict_streams`). Run on the 163 per-pair folds whose test seasons no unit saw (all LOYO, all but 4 LORO, CV10 for the 4 DEV pairs only) |
| Comparison | Paired with dense p3-nbr from scratch on the same folds: pooled out-of-fold R² per pair and protocol, paired fold bootstrap CI. A3 − scratch is the pretraining effect; A3 − A7 is the knowledge effect |
| Execution | Two jobs: pretraining (92 runs) first, then fine-tuning (326 runs), so no fine-tuning unit starts before its checkpoint exists. Resources measured on a smoke pod first (§5 rule) |

Not in this round: observation-dropout augmentation (the fix proposed by the
LOYO diagnosis), kept separate so the pretraining comparison stays clean.

### LOYO diagnosis (2026-10-07)

- **BRA-C and GER-W LOYO are worse dense than monthly because of an
  observation-density shift, not a pipeline error.**
  - Clear S2 observations per cell in the dense cache drop to about a
    third in the early years (GER-W 2016: 5.2 vs 13–18 later; BRA-C 2017:
    6.1), when only Sentinel-2A was imaging.
  - The dense model, trained mostly on observation-rich years,
    underpredicts the high-yield sparse year GER-W 2016 by 3.2 t/ha
    (fold R² −0.91 vs −0.22 monthly). This one year dominates the pooled
    score.
  - Year-level bias of 1–3 t/ha affects both models.
- **Data checks are clean:** no weather NaN on dated slots; plausible
  seasonal sums and peak NDVI in every year.
- **Candidate fix (later round):** random S2 observation dropout during
  training.

### 9.1 Results (2026-10-07)

All 92 pretraining units and all 326 fine-tuning units are done; the
comparison covers 163 folds with all three arms. The table gives the
change in pooled R², averaged over the pairs of each protocol, with paired
fold bootstrap 95% CIs (`results/pk_dense_final.json`):

| Protocol | A3 − scratch pixel | A7 − scratch pixel | A3 − A7 pixel | A3 − scratch field | A7 − scratch field | A3 − A7 field |
|---|---|---|---|---|---|---|
| CV10 | -0.003 [-0.011, +0.006] | -0.003 [-0.012, +0.004] | +0.001 [-0.005, +0.007] | -0.002 [-0.010, +0.007] | -0.006 [-0.013, +0.003] | +0.004 [-0.001, +0.010] |
| LOYO | -0.001 [-0.034, +0.031] | -0.004 [-0.030, +0.020] | +0.003 [-0.016, +0.019] | +0.004 [-0.059, +0.075] | -0.021 [-0.087, +0.031] | +0.025 [-0.005, +0.070] |
| LORO | -0.009 [-0.027, +0.011] | -0.016 [-0.039, +0.007] | +0.006 [-0.010, +0.026] | -0.041 [-0.071, +0.015] | -0.063 [-0.105, -0.001] | +0.022 [-0.012, +0.057] |
| ALL | -0.005 [-0.017, +0.010] | -0.008 [-0.021, +0.005] | +0.004 [-0.007, +0.014] | -0.016 [-0.044, +0.019] | -0.036 [-0.061, -0.003] | +0.020 [+0.002, +0.040] |

**Conclusion: knowledge pretraining does not change yield accuracy on the
dense series either.**
- A3 − scratch is within ±0.01 pixel in every protocol, and every CI
  includes 0.
- A3 − A7 (the knowledge effect) is not significant within any single
  protocol.
- Across all protocols, the field-level A3 − A7 difference is significant:
  +0.020 [+0.002, +0.040]. This comes from the control *losing* field
  accuracy (A7 − scratch −0.036 [−0.061, −0.003]; LORO −0.063), not from
  A3 gaining (A3 − scratch −0.016, CI includes 0).
- So knowledge targets avoid the harm random targets do, but neither beats
  training from scratch.
- This matches the monthly study (§0 of
  spec/yieldsat-point-knowledge-pretraining.md), now under test-fold
  selection and with the dense series.

Operational notes:
- Fine-tuning was resubmitted once at 28 Gi: at 40 Gi the fleet used 19% of
  requested memory. After the resize the fleet used 70%.

## 10. S2 observation-dropout augmentation (project lead, 2026-10-07)

- **Cause it targets:** the LOYO regressions in §9.
- **Option:** `--s2_obs_dropout K` (training only;
  `YieldSATPointDataset(s2_obs_dropout=K)`). Each sample keeps a fraction
  r ~ U(K, 1) of its S2 slots.
  - The centre S2 values and the 5×5 neighbourhood are dropped at the same
    slots. Dates and weather are kept, as in a sparse year.
  - Check on GER-W: K = 0.3 keeps 64% of S2/neighbourhood observations
    (expected 65%). Weather is untouched, and nothing is ever added.
- **First run:** BRA-C and GER-W LOYO only (13 folds), K = 0.3 and
  K = 0.1. Compared with dense p3-nbr without augmentation on the same
  folds (pooled LOYO R² per pair, per-fold differences).
  - Units: `cluster/tabm/protocol_obsdrop_units.json` (26).
  - Job: `cluster/nautilus/protocol_obsdrop_job.yaml`.
