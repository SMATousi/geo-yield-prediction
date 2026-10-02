# YieldSAT: beating the paper's best models (improvement plan)

Started 2026-10-02 (user request). Context: `results/` and
`results/negative_r2_investigation.md`. With the paper's metric (pooled
out-of-fold R²), our point model ties the paper's LSTM but is 0.05–0.2 R²
below the paper's best models (3D-ConvLSTM/3D-LSTM spatial models and AFF
fusion). Image v1/v2 are below the point model.

**Goal:** beat the paper's best model by a good margin on a fast development
subset, then run the winning setting on the full dataset.

## 1. Development subset (DEV)

To avoid running everything for each idea, every candidate is evaluated on a
fixed subset of country-crop pairs:

| Pair | Country / crop | Why | CV10 / LOYO / LORO folds |
|---|---|---|---|
| ARG-W | Argentina, wheat | Largest LOYO gap (ours 0.46 vs best 0.72); province LORO | 10 / 7 / 6 (provinces) |
| BRA-C | Brazil, corn | Corn; weak LOYO/LORO | 10 / 6 / 7 (farms) |
| GER-R | Germany, rapeseed | Small pair; worst LORO (−0.25 vs 0.15) | 10 / 7 / 6 (farms) |
| URG-S | Uruguay, soybean | Soybean (the main crop); 10 farm regions | 10 / 5 / 10 (farms) |

- Every country and four crops; small to medium pairs (283–2,504 tiles).
- **Matrix:** paper policy, S2+ADM (the paper's best rows are S2+ADM),
  seed 0 → **94 runs per candidate**. The fold manifests are those of the
  point suite.
- **Metric:** pooled out-of-fold pixel and field R²/RMSE (the paper's
  computation), from `yieldsat_pooled_metrics.py`.
- **Reference values** (pooled pixel R², S2+ADM):

  | Pair | CV10 ours / best | LOYO ours / best | LORO ours / best |
  |---|---|---|---|
  | ARG-W | 0.74 / 0.84 | 0.46 / 0.72 | 0.57 / 0.78 |
  | BRA-C | 0.43 / 0.46 | 0.12 / 0.29 | 0.22 / 0.37 |
  | GER-R | 0.38 / 0.49 | 0.09 / 0.31 | −0.25 / 0.15 |
  | URG-S | 0.38 / 0.43 | 0.27 / 0.32 | 0.31 / 0.36 |

  The DEV mean is 0.31 for our point model vs 0.46 for the paper's best.

**Success criterion ("good margin"):**
- the DEV mean pooled pixel R² is ≥ the paper-best DEV mean + 0.03, and the
  same at field level;
- no protocol's DEV mean is below the paper's best for that protocol.

Only a candidate that meets it moves to the full dataset (all 9 pairs ×
CV10/LOYO/LORO × paper/strict × S2/S2+ADM × 3 seeds).

## 2. Solutions (user selection: S4, S5, S6, S7, S9, S10)

All solutions are options of one new **hybrid model**
(`models_yieldsat_hybrid.py`; `main_yieldsat_image.py --arch hybrid`). It
predicts every cell of a 64×64 tile (full-coverage build, every point cell)
and combines a **per-cell point branch** with optional **spatial context**,
so each solution is a switch and can be ablated:

| ID | Solution | Implementation (flag) |
|---|---|---|
| S4 | **Early fusion of ADM** (as the paper's input-fusion LSTM) | Per cell and per time step, the S2 bands, weather, and static layers (DEM, terrain, soil, broadcast over time) are concatenated with masks and date features. A temporal transformer encodes this one fused sequence. `--cell_fusion early` (vs `s2` = S2 series only) |
| S5 | **Year/region level term** | Prediction = tile level + cell residual. The level head reads only season-wide inputs (the tile's weather series, crop, seeding day of year). An auxiliary loss ties it to the tile's mean target. `--level_head` |
| S6 | **Local spatial context** | Three masked 3×3 conv layers over the map of per-cell embeddings (7×7-cell neighbourhood of all modalities). `--context local` |
| S10 | **S2 and elevation as 64×64 maps, everything else as points** | A tile-level image branch over S2 (cached DINO RGB tokens + nine-band maps of the K = 4 observations) and the DEM map (U-Net, 64×64 output). Its features are gathered at each cell and fused with the cell's point embedding (weather, terrain, soil, full S2 series). `--context maps` |
| S9 | **Cross-attention between the elevation map and the RGB image** | In the S10 image branch, DEM tokens (8×8) attend to the RGB/DINO tokens and vice versa, with residual updates, before decoding. `--dem_rgb_xattn` |
| S7 | **Augmentation** | Random flips and 90° rotations of the tile (all maps, target and masks together; the 4×4 DINO token grid is permuted accordingly), training only. `--augment` |

**Training** (shared):
- every tile of the training seasons; loss = MSE over valid cells,
  cell-weighted like the point model, + λ · level loss with S5;
- AdamW with cosine schedule; best epoch on the validation seasons (all
  cells, pooled);
- predictions are written in the point format, so the pooled tools apply
  unchanged.

## 3. Experiment plan

| Step | Candidates on DEV | Question |
|---|---|---|
| E1 | hybrid `--cell_fusion early` | Does early fusion (S4) beat our late-fusion point model? |
| E2 | E1 + `--level_head` | S5 |
| E3 | E2 + `--context local` | S6 |
| E4 | E2 + `--context maps` | S10 |
| E5 | E4 + `--dem_rgb_xattn` | S9 |
| E6 | best of E3–E5 + `--augment` | S7 |

E1–E6 run together as one DEV suite (6 × 94 runs). The next round combines
the winners and tunes them. Each step's results and decisions are logged
below.

## 4. Progress log

- 2026-10-02 — Plan written. DEV subset and success criterion fixed.
- 2026-10-02 — **Implementation** (commit `f5b7cd0`): `models_yieldsat_hybrid.py`
  (`CellEncoder` S4, `LevelHead` S5, `LocalContext` S6, map branch S10 =
  `YieldSATImageModel` restricted to DINO + nine bands + DEM, via
  `adm_streams` and `features()`, with S9 `dem_rgb_xattn`); `augment_tile`
  (S7: flips/rotations of every map, target, masks, grid row/col and the DINO
  token grid; aspect sin/cos rotated with the map); `main_yieldsat_image.py
  --arch hybrid --cell_fusion --context --level_head --dem_rgb_xattn
  --augment`.
  - Tests: 31 image + 36 point tests pass. They cover each variant's
    gradients, absent cells never predicted, augmentation identity and
    aspect rotation, and hybrid runs end to end. Image-model defaults are
    unchanged (same parameters and outputs).
  - Small spec deviation: S9's DEM tokens are on the 4×4 token grid (the
    same grid as the DINO tokens), not 8×8.
- 2026-10-02 — **Local calibration** (GER-R CV10 fold 0, S2+ADM, RTX 3090;
  test = the same 31,880 cells as the point run):

  | Setting | Test pixel R² | Pixel RMSE | Field R² | Wall |
  |---|---|---|---|---|
  | Point model (`before_full` recipe) | 0.226 | 1.485 | 0.440 | – |
  | Hybrid early fusion, batch 8, lr 5e-4, 40 ep | 0.026 | 1.665 | 0.097 | 3 min |
  | … batch 4, lr 1e-3, 40 ep | 0.242 | 1.469 | 0.598 | 6.5 min |
  | … **batch 8, lr 2e-3, 80 ep** | **0.293** | **1.419** | **0.649** | 10 min |
  | Maps + S9 + S5 + S7, batch 8, lr 5e-4, 8 ep (smoke) | 0.090 | 1.610 | 0.323 | 1 min |

  - The default image recipe (lr 5e-4, 60 epochs) under-trains the hybrid:
    validation R² was still rising, and the best epoch was 33 of 40.
  - With lr 2e-3 and 80 epochs, early fusion alone beats the point model on
    this fold (+0.07 pixel R², +0.21 field R²).
  - The DEV recipe is therefore batch 8 tiles, lr 2e-3, 80 epochs, with all
    tiles of the training seasons (`--train_min_valid 0`).
- 2026-10-02 — **Round 1 submitted** (`cluster/suites/dev_r1.yaml`,
  `cluster/nautilus/dev_r1_pools.yaml`; W&B project `yieldsat-cvpr27-improve`).
  - **Configurations:** h1-early (S4), h2-level (+S5), h3-local (+S6), h4-maps
    (+S10), h5-maps-xattn (+S9), h6a-local-aug (h3 + S7), h6b-maps-xattn-aug
    (h5 + S7).
  - **Size:** 7 × 94 = 658 runs, ~119 GPU-hours (≈ 4–5.5 h on 30 A10s);
    2 × 15 A10 pods (16 CPU, 48 GiB).
  - **Data:** `YieldSAT-Image-full` on the PVC; sparse per-job staging.
