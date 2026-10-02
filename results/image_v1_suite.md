# Image model v1 results (image_full suite)

Generated 2026-10-02 00:16 UTC from W&B project `yieldsat-cvpr27-image` (suite `image_full`) by `yieldsat_results_tables.py`.
**2456 of 2460 planned runs finished.** Values are means ± std over finished folds × seeds.

- **Pixel**: every held-out cell counts once. **Field**: per field season, mean prediction vs mean target; field R² only over folds with ≥ 3 test field seasons.
- **RMSE** in t/ha. **Policy** `paper` = the paper's grouping; `strict` additionally removes training seasons sharing a physical field with the test fold.
- **Paper** numbers are fold means from Pathak et al. (CVPR 2026, `results/yieldsat/paper_benchmark.csv`); "best" is the best model per row and modality set.

Image model v1: 64x64 tiles, frozen DINOv3 + per-modality encoders, Perceiver fusion (spec/yieldsat-image-training.md). Metrics cover the valid cells of the test tiles (26-63% of each country's point cells), not whole fields, so they are not directly comparable with the paper; see image_vs_point.md for the comparison with the point model on identical cells. GER-R/GER-W are warm-started from donors. 126 folds without test tiles are not in the plan.

## CV10 (10-fold, grouped by field season)

| Pair | Model | Policy | Inputs | Runs done | Pixel R² | Pixel RMSE | Field R² | Field RMSE | Paper LSTM pixel R² / RMSE (S2) | Paper best pixel R² / RMSE | Paper best field R² / RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ARG-C | Image v1 | paper | S2 | 30/30 | 0.12 ± 0.30 | 3.20 ± 0.64 | 0.27 ± 0.32 | 2.33 ± 0.59 | 0.59 / 2.40 | 0.65 / 2.20 (3D-ConvLSTM) | 0.84 / 1.13 (3D-ConvLSTM) |
| ARG-S | Image v1 | paper | S2 | 30/30 | 0.50 ± 0.12 | 1.03 ± 0.11 | 0.60 ± 0.09 | 0.74 ± 0.12 | 0.60 / 0.96 | 0.65 / 0.90 (3D-ConvLSTM) | 0.79 / 0.55 (3D-ConvLSTM) |
| ARG-W | Image v1 | paper | S2 | 30/30 | 0.53 ± 0.18 | 1.53 ± 0.30 | 0.63 ± 0.24 | 1.20 ± 0.33 | 0.74 / 1.21 | 0.79 / 1.10 (3D-ConvLSTM) | 0.92 / 0.62 (3D-ConvLSTM) |
| BRA-C | Image v1 | paper | S2 | 30/30 | 0.23 ± 0.14 | 2.49 ± 0.55 | 0.31 ± 0.32 | 1.44 ± 0.41 | 0.42 / 2.20 | 0.46 / 2.13 (3D-LSTM) | 0.82 / 0.74 (3D-ConvLSTM) |
| BRA-S | Image v1 | paper | S2 | 30/30 | 0.23 ± 0.10 | 1.03 ± 0.06 | 0.33 ± 0.21 | 0.54 ± 0.11 | 0.34 / 0.98 | 0.39 / 0.94 (3D-ConvLSTM) | 0.76 / 0.34 (3D-ConvLSTM) |
| BRA-W | Image v1 | paper | S2 | 30/30 | 0.10 ± 0.08 | 1.43 ± 0.20 | 0.27 ± 0.24 | 0.63 ± 0.24 | 0.22 / 1.39 | 0.24 / 1.37 (3D-LSTM) | 0.73 / 0.42 (3D-ConvLSTM) |
| GER-R | Image v1 | paper | S2 | 30/30 | -0.29 ± 0.51 | 1.34 ± 0.31 | -1.57 ± 3.15 | 0.98 ± 0.42 | 0.36 / 1.33 | 0.49 / 1.20 (3D-ConvLSTM) | 0.82 / 0.57 (3D-LSTM) |
| GER-W | Image v1 | paper | S2 | 30/30 | -0.24 ± 0.52 | 2.22 ± 0.48 | -1.10 ± 1.47 | 1.17 ± 0.89 | -0.32 / 3.38 | 0.34 / 2.40 (3D-ConvLSTM) | 0.65 / 1.12 (3D-ConvLSTM) |
| URG-S | Image v1 | paper | S2 | 30/30 | 0.31 ± 0.13 | 1.19 ± 0.10 | 0.53 ± 0.29 | 0.64 ± 0.13 | 0.37 / 1.26 | 0.41 / 1.22 (3D-ConvLSTM) | 0.77 / 0.51 (3D-ConvLSTM) |
| ARG-C | Image v1 | paper | S2+ADM | 30/30 | 0.12 ± 0.23 | 3.20 ± 0.38 | 0.30 ± 0.27 | 2.25 ± 0.40 | 0.59 / 2.40 | 0.70 / 2.03 (AFF) | 0.84 / 1.12 (AFF) |
| ARG-S | Image v1 | paper | S2+ADM | 30/30 | 0.44 ± 0.15 | 1.09 ± 0.15 | 0.59 ± 0.10 | 0.75 ± 0.12 | 0.60 / 0.96 | 0.73 / 0.79 (AFF) | 0.84 / 0.49 (AFF) |
| ARG-W | Image v1 | paper | S2+ADM | 30/30 | 0.54 ± 0.24 | 1.51 ± 0.41 | 0.66 ± 0.21 | 1.15 ± 0.38 | 0.74 / 1.21 | 0.84 / 0.96 (AFF) | 0.92 / 0.60 (AFF) |
| BRA-C | Image v1 | paper | S2+ADM | 30/30 | 0.23 ± 0.13 | 2.49 ± 0.53 | 0.36 ± 0.28 | 1.37 ± 0.36 | 0.42 / 2.20 | 0.46 / 2.12 (AFF) | 0.84 / 0.70 (AFF) |
| BRA-S | Image v1 | paper | S2+ADM | 30/30 | 0.26 ± 0.08 | 1.01 ± 0.07 | 0.36 ± 0.25 | 0.53 ± 0.11 | 0.34 / 0.98 | 0.44 / 0.90 (AFF) | 0.80 / 0.31 (AFF) |
| BRA-W | Image v1 | paper | S2+ADM | 30/30 | 0.05 ± 0.10 | 1.47 ± 0.21 | 0.10 ± 0.28 | 0.68 ± 0.23 | 0.22 / 1.39 | 0.24 / 1.37 (AFF) | 0.72 / 0.42 (3D-ConvLSTM) |
| GER-R | Image v1 | paper | S2+ADM | 30/30 | -0.32 ± 0.58 | 1.36 ± 0.33 | -1.24 ± 2.10 | 0.99 ± 0.45 | 0.36 / 1.33 | 0.49 / 1.20 (AFF) | 0.81 / 0.59 (3D-LSTM) |
| GER-W | Image v1 | paper | S2+ADM | 30/30 | -0.06 ± 0.32 | 2.05 ± 0.32 | -0.44 ± 1.45 | 0.93 ± 0.48 | -0.32 / 3.38 | 0.44 / 2.20 (AFF) | 0.77 / 0.90 (MMGF) |
| URG-S | Image v1 | paper | S2+ADM | 30/30 | 0.32 ± 0.14 | 1.18 ± 0.09 | 0.59 ± 0.29 | 0.59 ± 0.13 | 0.37 / 1.26 | 0.43 / 1.19 (AFF) | 0.81 / 0.46 (AFF) |
| ARG-C | Image v1 | strict | S2 | 30/30 | 0.18 ± 0.25 | 3.21 ± 0.36 | 0.21 ± 0.31 | 2.43 ± 0.33 | 0.59 / 2.40 | 0.65 / 2.20 (3D-ConvLSTM) | 0.84 / 1.13 (3D-ConvLSTM) |
| ARG-S | Image v1 | strict | S2 | 30/30 | 0.48 ± 0.09 | 1.05 ± 0.10 | 0.54 ± 0.12 | 0.78 ± 0.11 | 0.60 / 0.96 | 0.65 / 0.90 (3D-ConvLSTM) | 0.79 / 0.55 (3D-ConvLSTM) |
| ARG-W | Image v1 | strict | S2 | 29/30 | 0.50 ± 0.20 | 1.57 ± 0.30 | 0.62 ± 0.20 | 1.25 ± 0.40 | 0.74 / 1.21 | 0.79 / 1.10 (3D-ConvLSTM) | 0.92 / 0.62 (3D-ConvLSTM) |
| BRA-C | Image v1 | strict | S2 | 30/30 | 0.15 ± 0.22 | 2.52 ± 0.54 | -0.10 ± 1.17 | 1.38 ± 0.41 | 0.42 / 2.20 | 0.46 / 2.13 (3D-LSTM) | 0.82 / 0.74 (3D-ConvLSTM) |
| BRA-S | Image v1 | strict | S2 | 30/30 | 0.20 ± 0.09 | 1.05 ± 0.05 | 0.28 ± 0.27 | 0.58 ± 0.13 | 0.34 / 0.98 | 0.39 / 0.94 (3D-ConvLSTM) | 0.76 / 0.34 (3D-ConvLSTM) |
| BRA-W | Image v1 | strict | S2 | 30/30 | 0.05 ± 0.07 | 1.47 ± 0.18 | -0.25 ± 0.81 | 0.67 ± 0.16 | 0.22 / 1.39 | 0.24 / 1.37 (3D-LSTM) | 0.73 / 0.42 (3D-ConvLSTM) |
| GER-R | Image v1 | strict | S2 | 30/30 | -0.73 ± 1.20 | 1.36 ± 0.31 | -3.64 ± 6.16 | 1.02 ± 0.34 | 0.36 / 1.33 | 0.49 / 1.20 (3D-ConvLSTM) | 0.82 / 0.57 (3D-LSTM) |
| GER-W | Image v1 | strict | S2 | 30/30 | -0.24 ± 0.55 | 2.17 ± 0.45 | -0.23 ± 0.15 | 1.20 ± 0.69 | -0.32 / 3.38 | 0.34 / 2.40 (3D-ConvLSTM) | 0.65 / 1.12 (3D-ConvLSTM) |
| URG-S | Image v1 | strict | S2 | 30/30 | 0.31 ± 0.13 | 1.20 ± 0.13 | 0.50 ± 0.26 | 0.67 ± 0.20 | 0.37 / 1.26 | 0.41 / 1.22 (3D-ConvLSTM) | 0.77 / 0.51 (3D-ConvLSTM) |
| ARG-C | Image v1 | strict | S2+ADM | 30/30 | 0.18 ± 0.29 | 3.21 ± 0.48 | 0.28 ± 0.27 | 2.35 ± 0.40 | 0.59 / 2.40 | 0.70 / 2.03 (AFF) | 0.84 / 1.12 (AFF) |
| ARG-S | Image v1 | strict | S2+ADM | 30/30 | 0.46 ± 0.17 | 1.07 ± 0.17 | 0.59 ± 0.15 | 0.74 ± 0.14 | 0.60 / 0.96 | 0.73 / 0.79 (AFF) | 0.84 / 0.49 (AFF) |
| ARG-W | Image v1 | strict | S2+ADM | 30/30 | 0.48 ± 0.25 | 1.58 ± 0.39 | 0.61 ± 0.20 | 1.24 ± 0.38 | 0.74 / 1.21 | 0.84 / 0.96 (AFF) | 0.92 / 0.60 (AFF) |
| BRA-C | Image v1 | strict | S2+ADM | 30/30 | 0.18 ± 0.24 | 2.45 ± 0.46 | 0.11 ± 0.93 | 1.26 ± 0.32 | 0.42 / 2.20 | 0.46 / 2.12 (AFF) | 0.84 / 0.70 (AFF) |
| BRA-S | Image v1 | strict | S2+ADM | 30/30 | 0.23 ± 0.08 | 1.03 ± 0.05 | 0.40 ± 0.18 | 0.53 ± 0.09 | 0.34 / 0.98 | 0.44 / 0.90 (AFF) | 0.80 / 0.31 (AFF) |
| BRA-W | Image v1 | strict | S2+ADM | 30/30 | 0.07 ± 0.09 | 1.46 ± 0.18 | -0.03 ± 0.68 | 0.61 ± 0.14 | 0.22 / 1.39 | 0.24 / 1.37 (AFF) | 0.72 / 0.42 (3D-ConvLSTM) |
| GER-R | Image v1 | strict | S2+ADM | 30/30 | -0.59 ± 1.09 | 1.33 ± 0.34 | -3.08 ± 6.09 | 0.96 ± 0.37 | 0.36 / 1.33 | 0.49 / 1.20 (AFF) | 0.81 / 0.59 (3D-LSTM) |
| GER-W | Image v1 | strict | S2+ADM | 30/30 | -0.03 ± 0.25 | 2.01 ± 0.42 | -0.25 ± 0.17 | 0.94 ± 0.59 | -0.32 / 3.38 | 0.44 / 2.20 (AFF) | 0.77 / 0.90 (MMGF) |
| URG-S | Image v1 | strict | S2+ADM | 30/30 | 0.29 ± 0.14 | 1.21 ± 0.12 | 0.49 ± 0.28 | 0.67 ± 0.19 | 0.37 / 1.26 | 0.43 / 1.19 (AFF) | 0.81 / 0.46 (AFF) |

## LORO, provinces (Argentina; the paper's regions)

| Pair | Model | Policy | Inputs | Runs done | Pixel R² | Pixel RMSE | Field R² | Field RMSE | Paper LSTM pixel R² / RMSE (S2) | Paper best pixel R² / RMSE | Paper best field R² / RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ARG-C | Image v1 | paper | S2 | 18/18 | -0.05 ± 0.26 | 3.31 ± 0.62 | -0.45 ± 1.26 | 2.66 ± 0.57 | 0.44 / 2.79 | 0.54 / 2.52 (3D-ConvLSTM) | 0.68 / 1.59 (3D-ConvLSTM) |
| ARG-S | Image v1 | paper | S2 | 24/24 | 0.22 ± 0.25 | 1.15 ± 0.13 | 0.17 ± 0.49 | 0.73 ± 0.34 | 0.55 / 1.03 | 0.59 / 0.98 (3D-LSTM) | 0.70 / 0.67 (3D-LSTM) |
| ARG-W | Image v1 | paper | S2 | 18/18 | 0.11 ± 0.57 | 1.53 ± 0.35 | -0.06 ± 1.01 | 1.18 ± 0.46 | 0.61 / 1.48 | 0.71 / 1.30 (3D-LSTM) | 0.81 / 0.92 (3D-LSTM) |
| ARG-C | Image v1 | paper | S2+ADM | 18/18 | 0.05 ± 0.16 | 3.22 ± 0.79 | -0.05 ± 0.40 | 2.56 ± 0.84 | 0.44 / 2.79 | 0.54 / 2.52 (3D-ConvLSTM) | 0.68 / 1.59 (3D-ConvLSTM) |
| ARG-S | Image v1 | paper | S2+ADM | 24/24 | 0.13 ± 0.22 | 1.23 ± 0.23 | -0.47 ± 2.64 | 0.83 ± 0.38 | 0.55 / 1.03 | 0.65 / 0.90 (AFF) | 0.78 / 0.57 (AFF) |
| ARG-W | Image v1 | paper | S2+ADM | 18/18 | -0.39 ± 0.83 | 1.88 ± 0.32 | -0.19 ± 0.50 | 1.63 ± 0.34 | 0.61 / 1.48 | 0.78 / 1.12 (AFF) | 0.87 / 0.78 (AFF) |
| ARG-C | Image v1 | strict | S2 | 18/18 | -0.19 ± 0.50 | 3.42 ± 0.49 | -0.93 ± 2.22 | 2.76 ± 0.45 | 0.44 / 2.79 | 0.54 / 2.52 (3D-ConvLSTM) | 0.68 / 1.59 (3D-ConvLSTM) |
| ARG-S | Image v1 | strict | S2 | 24/24 | 0.18 ± 0.24 | 1.17 ± 0.11 | -0.03 ± 0.53 | 0.79 ± 0.27 | 0.55 / 1.03 | 0.59 / 0.98 (3D-LSTM) | 0.70 / 0.67 (3D-LSTM) |
| ARG-W | Image v1 | strict | S2 | 18/18 | 0.01 ± 0.58 | 1.62 ± 0.35 | -0.25 ± 1.18 | 1.29 ± 0.52 | 0.61 / 1.48 | 0.71 / 1.30 (3D-LSTM) | 0.81 / 0.92 (3D-LSTM) |
| ARG-C | Image v1 | strict | S2+ADM | 18/18 | 0.07 ± 0.23 | 3.15 ± 0.76 | -0.14 ± 0.69 | 2.53 ± 0.76 | 0.44 / 2.79 | 0.54 / 2.52 (3D-ConvLSTM) | 0.68 / 1.59 (3D-ConvLSTM) |
| ARG-S | Image v1 | strict | S2+ADM | 24/24 | 0.18 ± 0.18 | 1.19 ± 0.18 | -0.27 ± 1.87 | 0.81 ± 0.31 | 0.55 / 1.03 | 0.65 / 0.90 (AFF) | 0.78 / 0.57 (AFF) |
| ARG-W | Image v1 | strict | S2+ADM | 18/18 | -0.27 ± 0.66 | 1.83 ± 0.33 | -0.22 ± 0.69 | 1.56 ± 0.39 | 0.61 / 1.48 | 0.78 / 1.12 (AFF) | 0.87 / 0.78 (AFF) |

## LORO, farm regions

| Pair | Model | Policy | Inputs | Runs done | Pixel R² | Pixel RMSE | Field R² | Field RMSE | Paper LSTM pixel R² / RMSE (S2) | Paper best pixel R² / RMSE | Paper best field R² / RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BRA-C | Image v1 | paper | S2 | 18/18 | -0.08 ± 0.39 | 2.59 ± 0.36 | -0.56 ± 0.92 | 1.65 ± 0.38 | 0.27 / 2.46 | 0.34 / 2.33 (3D-LSTM) | 0.59 / 1.13 (3D-LSTM) |
| BRA-S | Image v1 | paper | S2 | 24/24 | -0.01 ± 0.25 | 1.09 ± 0.14 | -1.60 ± 3.65 | 0.66 ± 0.18 | 0.16 / 1.10 | 0.22 / 1.06 (3D-LSTM) | 0.41 / 0.53 (3D-LSTM) |
| BRA-W | Image v1 | paper | S2 | 15/15 | 0.00 ± 0.10 | 1.41 ± 0.17 | -0.26 ± 0.42 | 0.71 ± 0.10 | 0.11 / 1.48 | 0.20 / 1.41 (3D-LSTM) | 0.48 / 0.57 (3D-LSTM) |
| GER-R | Image v1 | paper | S2 | 18/18 | -0.65 ± 0.75 | 1.41 ± 0.46 | -3.00 ± 3.95 | 1.10 ± 0.51 | -0.05 / 1.71 | 0.17 / 1.52 (3D-LSTM) | 0.26 / 1.16 (3D-LSTM) |
| GER-W | Image v1 | paper | S2 | 18/18 | -1.11 ± 1.18 | 2.36 ± 0.83 | -6.62 ± 5.25 | 1.73 ± 1.03 | -0.78 / 3.93 | 0.10 / 2.79 (Transformer) | 0.14 / 1.76 (3D-ConvLSTM) |
| URG-S | Image v1 | paper | S2 | 17/18 | -0.26 ± 0.77 | 1.25 ± 0.17 | -2.09 ± 4.57 | 0.84 ± 0.20 | 0.34 / 1.28 | 0.36 / 1.26 (3D-ConvLSTM) | 0.68 / 0.61 (3D-ConvLSTM) |
| BRA-C | Image v1 | paper | S2+ADM | 18/18 | -0.01 ± 0.22 | 2.53 ± 0.38 | -0.25 ± 0.60 | 1.46 ± 0.49 | 0.27 / 2.46 | 0.37 / 2.29 (3D-LSTM) | 0.65 / 1.05 (3D-LSTM) |
| BRA-S | Image v1 | paper | S2+ADM | 24/24 | -0.07 ± 0.35 | 1.11 ± 0.16 | -2.40 ± 5.45 | 0.69 ± 0.24 | 0.16 / 1.10 | 0.35 / 0.97 (AFF) | 0.63 / 0.42 (AFF) |
| BRA-W | Image v1 | paper | S2+ADM | 15/15 | -0.04 ± 0.09 | 1.44 ± 0.20 | -0.29 ± 0.32 | 0.73 ± 0.12 | 0.11 / 1.48 | 0.18 / 1.42 (AFF) | 0.52 / 0.55 (AFF) |
| GER-R | Image v1 | paper | S2+ADM | 18/18 | -0.82 ± 0.96 | 1.46 ± 0.44 | -8.04 ± 14.17 | 1.20 ± 0.38 | -0.05 / 1.71 | 0.15 / 1.55 (AFF) | 0.25 / 1.16 (3D-ConvLSTM) |
| GER-W | Image v1 | paper | S2+ADM | 18/18 | -1.79 ± 2.32 | 2.64 ± 1.28 | -9.49 ± 7.50 | 2.04 ± 1.48 | -0.78 / 3.93 | 0.07 / 2.84 (3D-ConvLSTM) | 0.14 / 1.76 (3D-ConvLSTM) |
| URG-S | Image v1 | paper | S2+ADM | 18/18 | 0.01 ± 0.32 | 1.17 ± 0.19 | -1.73 ± 4.63 | 0.70 ± 0.21 | 0.34 / 1.28 | 0.36 / 1.26 (3D-ConvLSTM) | 0.68 / 0.61 (3D-ConvLSTM) |
| BRA-C | Image v1 | strict | S2 | 15/15 | -0.06 ± 0.23 | 2.65 ± 0.35 | -0.88 ± 1.57 | 1.62 ± 0.34 | 0.27 / 2.46 | 0.34 / 2.33 (3D-LSTM) | 0.59 / 1.13 (3D-LSTM) |
| BRA-S | Image v1 | strict | S2 | 21/21 | -0.03 ± 0.23 | 1.10 ± 0.13 | -1.77 ± 3.98 | 0.67 ± 0.20 | 0.16 / 1.10 | 0.22 / 1.06 (3D-LSTM) | 0.41 / 0.53 (3D-LSTM) |
| BRA-W | Image v1 | strict | S2 | 12/12 | -0.02 ± 0.09 | 1.47 ± 0.18 | -0.18 ± 0.39 | 0.72 ± 0.07 | 0.11 / 1.48 | 0.20 / 1.41 (3D-LSTM) | 0.48 / 0.57 (3D-LSTM) |
| GER-R | Image v1 | strict | S2 | 18/18 | -0.60 ± 0.81 | 1.39 ± 0.45 | -5.40 ± 10.04 | 1.11 ± 0.44 | -0.05 / 1.71 | 0.17 / 1.52 (3D-LSTM) | 0.26 / 1.16 (3D-LSTM) |
| GER-W | Image v1 | strict | S2 | 18/18 | -1.12 ± 1.07 | 2.38 ± 0.79 | -6.39 ± 4.59 | 1.75 ± 0.96 | -0.78 / 3.93 | 0.10 / 2.79 (Transformer) | 0.14 / 1.76 (3D-ConvLSTM) |
| URG-S | Image v1 | strict | S2 | 18/18 | -0.19 ± 0.69 | 1.23 ± 0.18 | -1.53 ± 3.94 | 0.81 ± 0.22 | 0.34 / 1.28 | 0.36 / 1.26 (3D-ConvLSTM) | 0.68 / 0.61 (3D-ConvLSTM) |
| BRA-C | Image v1 | strict | S2+ADM | 15/15 | -0.09 ± 0.24 | 2.69 ± 0.31 | -0.92 ± 1.81 | 1.64 ± 0.35 | 0.27 / 2.46 | 0.37 / 2.29 (3D-LSTM) | 0.65 / 1.05 (3D-LSTM) |
| BRA-S | Image v1 | strict | S2+ADM | 21/21 | -0.18 ± 0.32 | 1.17 ± 0.14 | -3.21 ± 6.37 | 0.77 ± 0.19 | 0.16 / 1.10 | 0.35 / 0.97 (AFF) | 0.63 / 0.42 (AFF) |
| BRA-W | Image v1 | strict | S2+ADM | 11/12 | -0.08 ± 0.14 | 1.50 ± 0.23 | -0.25 ± 0.40 | 0.75 ± 0.13 | 0.11 / 1.48 | 0.18 / 1.42 (AFF) | 0.52 / 0.55 (AFF) |
| GER-R | Image v1 | strict | S2+ADM | 18/18 | -0.65 ± 0.71 | 1.41 ± 0.42 | -3.77 ± 5.78 | 1.11 ± 0.39 | -0.05 / 1.71 | 0.15 / 1.55 (AFF) | 0.25 / 1.16 (3D-ConvLSTM) |
| GER-W | Image v1 | strict | S2+ADM | 18/18 | -1.81 ± 2.31 | 2.65 ± 1.27 | -9.49 ± 7.50 | 2.05 ± 1.46 | -0.78 / 3.93 | 0.07 / 2.84 (3D-ConvLSTM) | 0.14 / 1.76 (3D-ConvLSTM) |
| URG-S | Image v1 | strict | S2+ADM | 18/18 | 0.09 ± 0.20 | 1.13 ± 0.19 | -1.17 ± 3.42 | 0.63 ± 0.19 | 0.34 / 1.28 | 0.36 / 1.26 (3D-ConvLSTM) | 0.68 / 0.61 (3D-ConvLSTM) |

## LOYO (leave one year out)

| Pair | Model | Policy | Inputs | Runs done | Pixel R² | Pixel RMSE | Field R² | Field RMSE | Paper LSTM pixel R² / RMSE (S2) | Paper best pixel R² / RMSE | Paper best field R² / RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ARG-C | Image v1 | paper | S2 | 24/24 | -0.03 ± 0.41 | 3.18 ± 0.52 | -0.27 ± 1.02 | 2.52 ± 0.47 | 0.39 / 2.90 | 0.48 / 2.68 (3D-ConvLSTM) | 0.61 / 1.75 (3D-ConvLSTM) |
| ARG-S | Image v1 | paper | S2 | 24/24 | 0.13 ± 0.38 | 1.09 ± 0.18 | -0.09 ± 0.58 | 0.86 ± 0.23 | 0.46 / 1.12 | 0.55 / 1.03 (3D-LSTM) | 0.66 / 0.70 (3D-LSTM) |
| ARG-W | Image v1 | paper | S2 | 21/21 | -0.68 ± 1.84 | 1.67 ± 0.62 | -0.60 ± 1.94 | 1.27 ± 0.68 | 0.61 / 1.49 | 0.68 / 1.35 (3D-LSTM) | 0.78 / 1.00 (3D-ConvLSTM) |
| BRA-C | Image v1 | paper | S2 | 18/18 | -0.16 ± 0.57 | 2.71 ± 0.84 | -1.12 ± 2.58 | 1.73 ± 0.63 | 0.15 / 2.66 | 0.21 / 2.56 (3D-LSTM) | 0.36 / 1.42 (3D-LSTM) |
| BRA-S | Image v1 | paper | S2 | 24/24 | -0.10 ± 0.37 | 1.17 ± 0.20 | -0.70 ± 1.14 | 0.69 ± 0.28 | 0.04 / 1.18 | 0.19 / 1.08 (3D-ConvLSTM) | 0.36 / 0.55 (3D-ConvLSTM) |
| BRA-W | Image v1 | paper | S2 | 21/21 | -0.07 ± 0.21 | 1.47 ± 0.39 | -0.86 ± 1.41 | 0.78 ± 0.29 | 0.08 / 1.50 | 0.08 / 1.50 (LSTM) | -0.05 / 0.82 (LSTM) |
| GER-R | Image v1 | paper | S2 | 21/21 | -1.65 ± 3.23 | 1.57 ± 0.64 | -1.44 ± 1.15 | 1.21 ± 0.67 | -0.05 / 1.71 | 0.21 / 1.49 (3D-ConvLSTM) | 0.43 / 1.02 (3D-ConvLSTM) |
| GER-W | Image v1 | paper | S2 | 12/12 | -0.35 ± 0.71 | 2.26 ± 0.38 | -1.46 ± 2.55 | 1.32 ± 0.67 | -0.68 / 3.82 | 0.15 / 2.71 (3D-ConvLSTM) | 0.23 / 1.67 (3D-ConvLSTM) |
| URG-S | Image v1 | paper | S2 | 15/15 | -0.11 ± 0.29 | 1.27 ± 0.27 | -0.80 ± 0.96 | 0.80 ± 0.31 | 0.33 / 1.29 | 0.35 / 1.27 (3D-LSTM) | 0.63 / 0.65 (3D-ConvLSTM) |
| ARG-C | Image v1 | paper | S2+ADM | 23/24 | -0.35 ± 0.90 | 3.46 ± 0.66 | -0.64 ± 1.69 | 2.72 ± 0.63 | 0.39 / 2.90 | 0.44 / 2.78 (LSTM) | 0.42 / 2.13 (LSTM) |
| ARG-S | Image v1 | paper | S2+ADM | 24/24 | -0.00 ± 0.66 | 1.16 ± 0.27 | -0.18 ± 0.64 | 0.91 ± 0.29 | 0.46 / 1.12 | 0.66 / 0.89 (AFF) | 0.77 / 0.59 (AFF) |
| ARG-W | Image v1 | paper | S2+ADM | 21/21 | -1.96 ± 4.81 | 1.89 ± 1.12 | -2.61 ± 5.57 | 1.54 ± 1.16 | 0.61 / 1.49 | 0.72 / 1.27 (AFF) | 0.81 / 0.93 (AFF) |
| BRA-C | Image v1 | paper | S2+ADM | 18/18 | -0.23 ± 0.46 | 2.80 ± 0.75 | -1.16 ± 1.93 | 1.78 ± 0.57 | 0.15 / 2.66 | 0.29 / 2.43 (AFF) | 0.45 / 1.32 (3D-ConvLSTM) |
| BRA-S | Image v1 | paper | S2+ADM | 24/24 | -0.05 ± 0.26 | 1.15 ± 0.17 | -0.61 ± 0.85 | 0.68 ± 0.23 | 0.04 / 1.18 | 0.29 / 1.02 (AFF) | 0.45 / 0.52 (3D-LSTM) |
| BRA-W | Image v1 | paper | S2+ADM | 21/21 | -0.05 ± 0.16 | 1.46 ± 0.39 | -0.74 ± 0.85 | 0.78 ± 0.28 | 0.08 / 1.50 | 0.11 / 1.48 (AFF) | 0.14 / 0.74 (AFF) |
| GER-R | Image v1 | paper | S2+ADM | 21/21 | -1.25 ± 1.96 | 1.47 ± 0.54 | -0.79 ± 0.76 | 1.12 ± 0.50 | -0.05 / 1.71 | 0.31 / 1.38 (LSTM) | 0.58 / 0.87 (LSTM) |
| GER-W | Image v1 | paper | S2+ADM | 12/12 | 0.01 ± 0.13 | 2.03 ± 0.28 | 0.12 ± 0.57 | 0.89 ± 0.30 | -0.68 / 3.82 | 0.22 / 2.60 (3D-ConvLSTM) | 0.46 / 1.39 (3D-ConvLSTM) |
| URG-S | Image v1 | paper | S2+ADM | 15/15 | -0.27 ± 0.28 | 1.36 ± 0.29 | -1.26 ± 1.07 | 0.94 ± 0.39 | 0.33 / 1.29 | 0.32 / 1.30 (AFF) | 0.56 / 0.71 (3D-ConvLSTM) |
| ARG-C | Image v1 | strict | S2 | 24/24 | -0.08 ± 0.48 | 3.23 ± 0.47 | -0.68 ± 1.95 | 2.57 ± 0.38 | 0.39 / 2.90 | 0.48 / 2.68 (3D-ConvLSTM) | 0.61 / 1.75 (3D-ConvLSTM) |
| ARG-S | Image v1 | strict | S2 | 24/24 | 0.07 ± 0.39 | 1.13 ± 0.20 | -0.16 ± 0.61 | 0.89 ± 0.27 | 0.46 / 1.12 | 0.55 / 1.03 (3D-LSTM) | 0.66 / 0.70 (3D-LSTM) |
| ARG-W | Image v1 | strict | S2 | 21/21 | -0.57 ± 1.28 | 1.73 ± 0.62 | -0.49 ± 1.49 | 1.36 ± 0.73 | 0.61 / 1.49 | 0.68 / 1.35 (3D-LSTM) | 0.78 / 1.00 (3D-ConvLSTM) |
| BRA-C | Image v1 | strict | S2 | 15/15 | -0.15 ± 0.19 | 2.83 ± 0.99 | -1.00 ± 0.96 | 1.87 ± 0.80 | 0.15 / 2.66 | 0.21 / 2.56 (3D-LSTM) | 0.36 / 1.42 (3D-LSTM) |
| BRA-S | Image v1 | strict | S2 | 24/24 | -0.19 ± 0.29 | 1.22 ± 0.13 | -0.99 ± 1.04 | 0.75 ± 0.18 | 0.04 / 1.18 | 0.19 / 1.08 (3D-ConvLSTM) | 0.36 / 0.55 (3D-ConvLSTM) |
| BRA-W | Image v1 | strict | S2 | 21/21 | -0.05 ± 0.16 | 1.45 ± 0.35 | -0.51 ± 0.78 | 0.72 ± 0.21 | 0.08 / 1.50 | 0.08 / 1.50 (LSTM) | -0.05 / 0.82 (LSTM) |
| GER-R | Image v1 | strict | S2 | 21/21 | -1.17 ± 1.47 | 1.52 ± 0.60 | -1.25 ± 1.36 | 1.16 ± 0.60 | -0.05 / 1.71 | 0.21 / 1.49 (3D-ConvLSTM) | 0.43 / 1.02 (3D-ConvLSTM) |
| GER-W | Image v1 | strict | S2 | 12/12 | -0.78 ± 0.95 | 2.62 ± 0.54 | -3.34 ± 3.08 | 1.91 ± 0.73 | -0.68 / 3.82 | 0.15 / 2.71 (3D-ConvLSTM) | 0.23 / 1.67 (3D-ConvLSTM) |
| URG-S | Image v1 | strict | S2 | 15/15 | -0.03 ± 0.20 | 1.22 ± 0.24 | -0.40 ± 0.77 | 0.71 ± 0.29 | 0.33 / 1.29 | 0.35 / 1.27 (3D-LSTM) | 0.63 / 0.65 (3D-ConvLSTM) |
| ARG-C | Image v1 | strict | S2+ADM | 24/24 | -0.18 ± 0.63 | 3.30 ± 0.57 | -0.61 ± 1.79 | 2.66 ± 0.51 | 0.39 / 2.90 | 0.44 / 2.78 (LSTM) | 0.42 / 2.13 (LSTM) |
| ARG-S | Image v1 | strict | S2+ADM | 24/24 | -0.01 ± 0.64 | 1.16 ± 0.26 | -0.14 ± 0.63 | 0.89 ± 0.28 | 0.46 / 1.12 | 0.66 / 0.89 (AFF) | 0.77 / 0.59 (AFF) |
| ARG-W | Image v1 | strict | S2+ADM | 21/21 | -1.63 ± 3.49 | 2.01 ± 0.94 | -2.32 ± 4.00 | 1.66 ± 0.95 | 0.61 / 1.49 | 0.72 / 1.27 (AFF) | 0.81 / 0.93 (AFF) |
| BRA-C | Image v1 | strict | S2+ADM | 15/15 | -0.09 ± 0.18 | 2.76 ± 0.99 | -0.54 ± 0.47 | 1.70 ± 0.83 | 0.15 / 2.66 | 0.29 / 2.43 (AFF) | 0.45 / 1.32 (3D-ConvLSTM) |
| BRA-S | Image v1 | strict | S2+ADM | 24/24 | -0.32 ± 0.44 | 1.27 ± 0.15 | -1.26 ± 1.23 | 0.80 ± 0.22 | 0.04 / 1.18 | 0.29 / 1.02 (AFF) | 0.45 / 0.52 (3D-LSTM) |
| BRA-W | Image v1 | strict | S2+ADM | 21/21 | -0.20 ± 0.21 | 1.54 ± 0.32 | -0.82 ± 1.14 | 0.78 ± 0.23 | 0.08 / 1.50 | 0.11 / 1.48 (AFF) | 0.14 / 0.74 (AFF) |
| GER-R | Image v1 | strict | S2+ADM | 21/21 | -1.12 ± 1.58 | 1.47 ± 0.53 | -0.87 ± 0.83 | 1.11 ± 0.51 | -0.05 / 1.71 | 0.31 / 1.38 (LSTM) | 0.58 / 0.87 (LSTM) |
| GER-W | Image v1 | strict | S2+ADM | 12/12 | -0.25 ± 0.34 | 2.30 ± 0.62 | -1.61 ± 1.21 | 1.56 ± 0.75 | -0.68 / 3.82 | 0.22 / 2.60 (3D-ConvLSTM) | 0.46 / 1.39 (3D-ConvLSTM) |
| URG-S | Image v1 | strict | S2+ADM | 15/15 | -0.27 ± 0.39 | 1.36 ± 0.34 | -1.28 ± 1.29 | 0.92 ± 0.40 | 0.33 / 1.29 | 0.32 / 1.30 (AFF) | 0.56 / 0.71 (3D-ConvLSTM) |
