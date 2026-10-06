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
