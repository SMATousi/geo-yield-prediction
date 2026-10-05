# Reproducing the YieldSAT paper's models (sanity check)

**Status:** specified 2026-10-05 (user request). The gap between our best
models (DEV pixel ≈ 0.35) and the paper's best (≈ 0.46) is large. Before
more modelling we reproduce the paper's models on our data and protocol
pipeline:
- if we match their numbers, our results are comparable;
- if not, the difference is a data or protocol problem to find, which may
  improve every model.

## 1. Sources

- **Paper:** YieldSAT, arXiv:2604.00940 v1, §5, Table 4, Appendix A.3
  Tables 13–18 (already transcribed in `results/yieldsat/paper_benchmark.csv`).
  - "All models are trained at the pixel level ... some methods include
    spatial neighborhood information by processing a pixel with its
    surroundings using a 3D-CNN block or a ConvLSTM."
  - "The metrics are presented as the average across the folds."
- **Model details:** M. Miranda, PhD thesis, RPTU 2025, *Domain-Informed
  Neural Networks for Earth Observation* (the author of [32]). §4.4–4.5 and
  Appendix A.1.2:
  - the input-fusion (IF) baselines;
  - the 3D-LSTM;
  - the multimodal attention fusion model (MMAF). It is our best guess for
    the paper's "AFF [32]": same author, same feature-fusion design.
- **Related:** [13] Helber et al., IGARSS 2024 (3D-ConvLSTM, many-to-many
  spatio-temporal); [36] Pathak et al., IGARSS 2023 (LSTM baseline).

## 2. What the sources say about the pipeline

| Item | Paper / thesis | Our pipeline | Difference |
|---|---|---|---|
| Time axis | 24 calendar months from 1 Jan of the seeding year (SY − 1 if SY = HY); one cloud-free S2 image per month (SCL 4/5); before seeding / after harvest / cloudy masked | Same: the preprocessed NetCDF, slot = month (verified 2026-10-05) | none |
| Weather | Daily weather aggregated between consecutive S2 timestamps (sums) | Same operator; **we mask the first dated slot** (unknown interval start) | **D1** |
| Normalization | Not stated; the release tutorial uses raw inputs with NaN → −1. Our PC-05 reproduced the paper's LSTM only with the file's supplied `stats-*` (0.33 vs 0.36) | Train-fold z-score (all our models) | **D2** (PC-05: supplied 0.33, train-fold 0.28 on GER-R CV10, S2-only LSTM) |
| Coordinates | **Used as input** (lat/lon → unit sphere) in the neighbourhood models | Excluded as metadata | **D3** |
| Spatial context | **5×5 window** around each pixel from the per-field image data (3D-LSTM, MMAF) | Point models: none; S6/F1: 5×5 neighbourhood mean of S2 only | **D4** (the main suspected cause of the gap) |
| S2 series in MMAF | **Dense time series** (every acquisition, padded with −1); the release has 16–131 dates per field in `Raw.zip` | Monthly subsample only | **D5** |
| Static inputs | IF: repeated over time steps. MMAF: separate encoders (conv2d 5×5 + FC) | Repeated (LSTM); separate encoders (our point model) | – |
| Metric | Fold mean (paper text); pooled out-of-fold investigated in results/negative_r2_investigation.md | Pooled (since 2026-10-03) | **D6**: report both |
| Static validity | Not described | v3 masks corrupt Argentina curvature | D7 (minor) |
| Training | IF-LSTM: 2 × 128, 2 FC (128 → 1), BN, ReLU, Adam 1e-3, batch 1,024, ≤ 50 epochs, early stop 8. 3D-LSTM: conv3d kernel (1, 5, 5) → 64 ch, BN, LeakyReLU, LSTM 2 × 64, Adam 6e-3, batch 2,048, ≤ 50 epochs, reduce-on-plateau, early stop 10, random 90° rotations, temporal dropout 0.2. MMAF: S2 via the 3D-LSTM block, DEM/soil/coords via conv2d (5, 5) + FC, weather via LSTM, each → 64; learnable-query scaled dot-product attention pooling (dropout 0.2) → linear | — | implemented as described |

## 3. Plan

### R0 — Metric and protocol switches (cheap)

- Every reproduction run reports **fold mean ± std** (the paper's text) and
  **pooled** out-of-fold R²/RMSE, at pixel and field level.
- New input switches in the point data path:
  - `--weather_first_slot keep` (D1);
  - `--normalization supplied` (exists; D2);
  - `--coords` as an input stream (D3).

### R1 — Input-fusion LSTM, S2+ADM (the cleanest test)

- **Model:** the thesis IF-LSTM configuration (§2) on the preprocessed
  NetCDF, the paper's released format. Rows: the 12 DEV rows first, then
  all 9 pairs × CV10.
- **Target:** the paper's "S2+ADM, Input Fusion, LSTM" rows (Tables 4,
  13–18). Example: GER-R CV10 pixel R² 0.47, field 0.81.
- **Variants** (one at a time from the paper-like default supplied stats +
  first slot kept + coords):
  - our train-fold normalization;
  - first slot masked;
  - no coords.

  These quantify D1–D3 directly.
- **Pass:** fold-mean R² within one fold std of the paper (or ±0.03) on most
  rows. If R1 fails, the gap is in data or protocol, not architecture; stop
  and investigate before R2/R3.

### R2 — 3D-LSTM with 5×5 neighbourhoods (D4)

- **Input:** for each pixel, the 5×5 window of the field's grid (from cache
  `grid_row`/`grid_col`), cells outside the field or missing = −1, all IF
  channels per month; X ∈ R^{B×24×5×5}.
- **Model and training** exactly as §2.
- **Target:** the paper's 3D-LSTM rows ("S2" and "S2+ADM input fusion").

### R3 — MMAF ("AFF") with neighbourhoods (D4, D5)

- **R3a:** the monthly S2 series (what we have), per-modality encoders,
  attention pooling, coords.
- **R3b:** the dense S2 series from `Raw.zip` (per-field rasters for every
  acquisition) plus daily weather from the raw CSVs, padded with −1. This
  needs a dense-series builder; done only if R3a falls short of the AFF rows.
- **Target:** the paper's AFF rows (the best model on 14 of 24 DEV
  row × level entries).

### Decision after R1–R3

- **Reproduced:** our results are comparable. Every point-model gap is then
  architectural (spatial context and dense series). Port the winning
  ingredients (5×5 windows, coords, dense S2) into TabM and the point model.
- **Not reproduced:** locate the difference with the ablations, fix it in the
  shared data path, and rerun the affected baselines.

## 4. Tasks

| ID | Deliverable | Acceptance |
|---|---|---|
| RP-01 | Fold-mean + pooled reporting for any run root (`yieldsat_repro_report.py`); paper rows joined | Matches `results/yieldsat/paper_benchmark.csv` keys |
| RP-02 | Input switches: first-slot weather keep, coords stream | Unit tests; default behaviour unchanged |
| RP-03 | IF-LSTM thesis configuration (`--model paper_lstm2`) | Same plumbing as `paper_lstm` |
| RP-04 | R1 runs (DEV CV10/LOYO/LORO + 9 pairs CV10) and variants | `results/repro_r1.md` |
| RP-05 | 5×5 neighbourhood window dataset from the point cache + 3D-LSTM | Window tests (field boundary padding, rotation augmentation); `results/repro_r2.md` |
| RP-06 | MMAF (monthly) | `results/repro_r3.md` |
| RP-07 | Dense-series builder from `Raw.zip` + MMAF dense (only if needed) | Dense dates per field match the raw S2 file list |

## 5. Progress log

- 2026-10-05 — Specified from the paper and the Miranda thesis
  (§4.4–4.5, A.1.2).
  - Our known differences from their pipeline: D1 first-slot weather
    masking, D2 normalization, D3 coordinates as input, D4 5×5 spatial
    windows, D5 dense S2 series for MMAF, D6 the metric.
  - Notable: the paper's plain pixel LSTM with S2+ADM (no spatial context)
    reports GER-R CV10 0.47 pixel, while our point model gets 0.38 pooled
    there. R1 tests this first.
