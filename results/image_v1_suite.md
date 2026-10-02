# Image model v1 results (image_full suite)

Generated 2026-10-02 03:26 UTC by `yieldsat_results_tables.py` from pooled out-of-fold predictions (cluster results, image_full).
**324 of 324 experiments (pair × protocol × policy × inputs × seed) complete.**

- **Metric = the paper's computation:** for each experiment, the held-out predictions of all folds are pooled (each cell is held out exactly once) and R²/RMSE are computed once. Values are mean ± std of these pooled scores over seeds; a seed counts only when all its folds are finished.
- **Pixel**: every held-out 10 m cell. **Field**: per field season, mean prediction vs mean target. R² = 1 − SSE/SST; RMSE in t/ha.
- **Policy** `paper` = the paper's grouping; `strict` additionally removes training seasons that share a physical field with the test fold.
- **Paper** numbers are from Pathak et al. (CVPR 2026, `results/yieldsat/paper_benchmark.csv`); "best" is the best model per row and modality set. Why pooled: [negative_r2_investigation.md](negative_r2_investigation.md).

Image model v1: 64x64 tiles, frozen DINOv3 + per-modality encoders, Perceiver fusion (spec/yieldsat-image-training.md). Metrics cover the valid cells of the image test tiles (26-63% of each country's point cells), not whole fields, so they are not directly comparable with the paper; see image_vs_point.md for the comparison with the point model on identical cells. GER-R/GER-W are warm-started from donors. 126 folds without test tiles are not in the plan.

## CV10 (10-fold, grouped by field season)

| Pair | Model | Policy | Inputs | Seeds complete | Pixel R² | Pixel RMSE | Field R² | Field RMSE | Paper LSTM pixel R² / RMSE (S2) | Paper best pixel R² / RMSE | Paper best field R² / RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ARG-C | Image v1 | paper | S2 | 3/3 | 0.25 ± 0.02 | 3.27 ± 0.04 | 0.39 ± 0.02 | 2.37 ± 0.04 | 0.59 / 2.40 | 0.65 / 2.20 (3D-ConvLSTM) | 0.84 / 1.13 (3D-ConvLSTM) |
| ARG-S | Image v1 | paper | S2 | 3/3 | 0.53 ± 0.02 | 1.03 ± 0.02 | 0.60 ± 0.01 | 0.75 ± 0.01 | 0.60 / 0.96 | 0.65 / 0.90 (3D-ConvLSTM) | 0.79 / 0.55 (3D-ConvLSTM) |
| ARG-W | Image v1 | paper | S2 | 3/3 | 0.58 ± 0.02 | 1.56 ± 0.03 | 0.66 ± 0.02 | 1.25 ± 0.04 | 0.74 / 1.21 | 0.79 / 1.10 (3D-ConvLSTM) | 0.92 / 0.62 (3D-ConvLSTM) |
| BRA-C | Image v1 | paper | S2 | 3/3 | 0.25 ± 0.03 | 2.57 ± 0.05 | 0.38 ± 0.04 | 1.51 ± 0.06 | 0.42 / 2.20 | 0.46 / 2.13 (3D-LSTM) | 0.82 / 0.74 (3D-ConvLSTM) |
| BRA-S | Image v1 | paper | S2 | 3/3 | 0.25 ± 0.01 | 1.03 ± 0.01 | 0.39 ± 0.04 | 0.56 ± 0.02 | 0.34 / 0.98 | 0.39 / 0.94 (3D-ConvLSTM) | 0.76 / 0.34 (3D-ConvLSTM) |
| BRA-W | Image v1 | paper | S2 | 3/3 | 0.11 ± 0.00 | 1.45 ± 0.00 | 0.24 ± 0.02 | 0.67 ± 0.01 | 0.22 / 1.39 | 0.24 / 1.37 (3D-LSTM) | 0.73 / 0.42 (3D-ConvLSTM) |
| GER-R | Image v1 | paper | S2 | 3/3 | -0.09 ± 0.15 | 1.42 ± 0.10 | -0.27 ± 0.36 | 1.11 ± 0.16 | 0.36 / 1.33 | 0.49 / 1.20 (3D-ConvLSTM) | 0.82 / 0.57 (3D-LSTM) |
| GER-W | Image v1 | paper | S2 | 3/3 | 0.00 ± 0.07 | 2.26 ± 0.08 | -0.23 ± 0.16 | 1.56 ± 0.10 | -0.32 / 3.38 | 0.34 / 2.40 (3D-ConvLSTM) | 0.65 / 1.12 (3D-ConvLSTM) |
| URG-S | Image v1 | paper | S2 | 3/3 | 0.36 ± 0.01 | 1.19 ± 0.01 | 0.60 ± 0.02 | 0.66 ± 0.02 | 0.37 / 1.26 | 0.41 / 1.22 (3D-ConvLSTM) | 0.77 / 0.51 (3D-ConvLSTM) |
| ARG-C | Image v1 | paper | S2+ADM | 3/3 | 0.27 ± 0.02 | 3.22 ± 0.05 | 0.44 ± 0.03 | 2.27 ± 0.07 | 0.59 / 2.40 | 0.70 / 2.03 (AFF) | 0.84 / 1.12 (AFF) |
| ARG-S | Image v1 | paper | S2+ADM | 3/3 | 0.47 ± 0.03 | 1.09 ± 0.03 | 0.60 ± 0.02 | 0.76 ± 0.02 | 0.60 / 0.96 | 0.73 / 0.79 (AFF) | 0.84 / 0.49 (AFF) |
| ARG-W | Image v1 | paper | S2+ADM | 3/3 | 0.58 ± 0.04 | 1.56 ± 0.07 | 0.69 ± 0.02 | 1.20 ± 0.05 | 0.74 / 1.21 | 0.84 / 0.96 (AFF) | 0.92 / 0.60 (AFF) |
| BRA-C | Image v1 | paper | S2+ADM | 3/3 | 0.26 ± 0.01 | 2.56 ± 0.03 | 0.44 ± 0.00 | 1.44 ± 0.01 | 0.42 / 2.20 | 0.46 / 2.12 (AFF) | 0.84 / 0.70 (AFF) |
| BRA-S | Image v1 | paper | S2+ADM | 3/3 | 0.27 ± 0.00 | 1.01 ± 0.00 | 0.43 ± 0.02 | 0.54 ± 0.01 | 0.34 / 0.98 | 0.44 / 0.90 (AFF) | 0.80 / 0.31 (AFF) |
| BRA-W | Image v1 | paper | S2+ADM | 3/3 | 0.05 ± 0.01 | 1.49 ± 0.01 | 0.13 ± 0.06 | 0.71 ± 0.03 | 0.22 / 1.39 | 0.24 / 1.37 (AFF) | 0.72 / 0.42 (3D-ConvLSTM) |
| GER-R | Image v1 | paper | S2+ADM | 3/3 | -0.13 ± 0.04 | 1.45 ± 0.03 | -0.32 ± 0.05 | 1.14 ± 0.02 | 0.36 / 1.33 | 0.49 / 1.20 (AFF) | 0.81 / 0.59 (3D-LSTM) |
| GER-W | Image v1 | paper | S2+ADM | 3/3 | 0.16 ± 0.07 | 2.08 ± 0.09 | 0.39 ± 0.18 | 1.09 ± 0.17 | -0.32 / 3.38 | 0.44 / 2.20 (AFF) | 0.77 / 0.90 (MMGF) |
| URG-S | Image v1 | paper | S2+ADM | 3/3 | 0.37 ± 0.01 | 1.18 ± 0.01 | 0.67 ± 0.01 | 0.60 ± 0.01 | 0.37 / 1.26 | 0.43 / 1.19 (AFF) | 0.81 / 0.46 (AFF) |
| ARG-C | Image v1 | strict | S2 | 3/3 | 0.26 ± 0.02 | 3.24 ± 0.03 | 0.34 ± 0.01 | 2.46 ± 0.03 | 0.59 / 2.40 | 0.65 / 2.20 (3D-ConvLSTM) | 0.84 / 1.13 (3D-ConvLSTM) |
| ARG-S | Image v1 | strict | S2 | 3/3 | 0.51 ± 0.00 | 1.06 ± 0.00 | 0.56 ± 0.01 | 0.79 ± 0.01 | 0.60 / 0.96 | 0.65 / 0.90 (3D-ConvLSTM) | 0.79 / 0.55 (3D-ConvLSTM) |
| ARG-W | Image v1 | strict | S2 | 3/3 | 0.56 ± 0.01 | 1.59 ± 0.01 | 0.64 ± 0.02 | 1.29 ± 0.04 | 0.74 / 1.21 | 0.79 / 1.10 (3D-ConvLSTM) | 0.92 / 0.62 (3D-ConvLSTM) |
| BRA-C | Image v1 | strict | S2 | 3/3 | 0.23 ± 0.04 | 2.61 ± 0.07 | 0.38 ± 0.08 | 1.52 ± 0.10 | 0.42 / 2.20 | 0.46 / 2.13 (3D-LSTM) | 0.82 / 0.74 (3D-ConvLSTM) |
| BRA-S | Image v1 | strict | S2 | 3/3 | 0.22 ± 0.01 | 1.05 ± 0.01 | 0.30 ± 0.04 | 0.60 ± 0.02 | 0.34 / 0.98 | 0.39 / 0.94 (3D-ConvLSTM) | 0.76 / 0.34 (3D-ConvLSTM) |
| BRA-W | Image v1 | strict | S2 | 3/3 | 0.08 ± 0.01 | 1.47 ± 0.00 | 0.10 ± 0.03 | 0.72 ± 0.01 | 0.22 / 1.39 | 0.24 / 1.37 (3D-LSTM) | 0.73 / 0.42 (3D-ConvLSTM) |
| GER-R | Image v1 | strict | S2 | 3/3 | -0.10 ± 0.08 | 1.43 ± 0.05 | -0.29 ± 0.13 | 1.13 ± 0.06 | 0.36 / 1.33 | 0.49 / 1.20 (3D-ConvLSTM) | 0.82 / 0.57 (3D-LSTM) |
| GER-W | Image v1 | strict | S2 | 3/3 | 0.01 ± 0.07 | 2.25 ± 0.07 | -0.17 ± 0.14 | 1.53 ± 0.09 | -0.32 / 3.38 | 0.34 / 2.40 (3D-ConvLSTM) | 0.65 / 1.12 (3D-ConvLSTM) |
| URG-S | Image v1 | strict | S2 | 3/3 | 0.35 ± 0.01 | 1.20 ± 0.01 | 0.56 ± 0.01 | 0.69 ± 0.01 | 0.37 / 1.26 | 0.41 / 1.22 (3D-ConvLSTM) | 0.77 / 0.51 (3D-ConvLSTM) |
| ARG-C | Image v1 | strict | S2+ADM | 3/3 | 0.25 ± 0.03 | 3.26 ± 0.06 | 0.37 ± 0.02 | 2.40 ± 0.05 | 0.59 / 2.40 | 0.70 / 2.03 (AFF) | 0.84 / 1.12 (AFF) |
| ARG-S | Image v1 | strict | S2+ADM | 3/3 | 0.48 ± 0.02 | 1.09 ± 0.02 | 0.60 ± 0.02 | 0.75 ± 0.02 | 0.60 / 0.96 | 0.73 / 0.79 (AFF) | 0.84 / 0.49 (AFF) |
| ARG-W | Image v1 | strict | S2+ADM | 3/3 | 0.55 ± 0.02 | 1.62 ± 0.04 | 0.64 ± 0.02 | 1.29 ± 0.03 | 0.74 / 1.21 | 0.84 / 0.96 (AFF) | 0.92 / 0.60 (AFF) |
| BRA-C | Image v1 | strict | S2+ADM | 3/3 | 0.28 ± 0.03 | 2.52 ± 0.05 | 0.52 ± 0.06 | 1.33 ± 0.09 | 0.42 / 2.20 | 0.46 / 2.12 (AFF) | 0.84 / 0.70 (AFF) |
| BRA-S | Image v1 | strict | S2+ADM | 3/3 | 0.24 ± 0.02 | 1.03 ± 0.01 | 0.41 ± 0.04 | 0.55 ± 0.02 | 0.34 / 0.98 | 0.44 / 0.90 (AFF) | 0.80 / 0.31 (AFF) |
| BRA-W | Image v1 | strict | S2+ADM | 3/3 | 0.10 ± 0.02 | 1.46 ± 0.02 | 0.27 ± 0.09 | 0.65 ± 0.04 | 0.22 / 1.39 | 0.24 / 1.37 (AFF) | 0.72 / 0.42 (3D-ConvLSTM) |
| GER-R | Image v1 | strict | S2+ADM | 3/3 | -0.09 ± 0.07 | 1.42 ± 0.05 | -0.23 ± 0.17 | 1.09 ± 0.08 | 0.36 / 1.33 | 0.49 / 1.20 (AFF) | 0.81 / 0.59 (3D-LSTM) |
| GER-W | Image v1 | strict | S2+ADM | 3/3 | 0.08 ± 0.07 | 2.17 ± 0.08 | 0.03 ± 0.17 | 1.39 ± 0.12 | -0.32 / 3.38 | 0.44 / 2.20 (AFF) | 0.77 / 0.90 (MMGF) |
| URG-S | Image v1 | strict | S2+ADM | 3/3 | 0.34 ± 0.03 | 1.21 ± 0.03 | 0.56 ± 0.06 | 0.69 ± 0.05 | 0.37 / 1.26 | 0.43 / 1.19 (AFF) | 0.81 / 0.46 (AFF) |

## LORO, provinces (Argentina; the paper's regions)

| Pair | Model | Policy | Inputs | Seeds complete | Pixel R² | Pixel RMSE | Field R² | Field RMSE | Paper LSTM pixel R² / RMSE (S2) | Paper best pixel R² / RMSE | Paper best field R² / RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ARG-C | Image v1 | paper | S2 | 3/3 | 0.11 ± 0.07 | 3.55 ± 0.13 | 0.13 ± 0.11 | 2.82 ± 0.17 | 0.44 / 2.79 | 0.54 / 2.52 (3D-ConvLSTM) | 0.68 / 1.59 (3D-ConvLSTM) |
| ARG-S | Image v1 | paper | S2 | 3/3 | 0.39 ± 0.02 | 1.18 ± 0.02 | 0.39 ± 0.04 | 0.93 ± 0.03 | 0.55 / 1.03 | 0.59 / 0.98 (3D-LSTM) | 0.70 / 0.67 (3D-LSTM) |
| ARG-W | Image v1 | paper | S2 | 3/3 | 0.55 ± 0.01 | 1.61 ± 0.02 | 0.63 ± 0.01 | 1.31 ± 0.02 | 0.61 / 1.48 | 0.71 / 1.30 (3D-LSTM) | 0.81 / 0.92 (3D-LSTM) |
| ARG-C | Image v1 | paper | S2+ADM | 3/3 | 0.12 ± 0.03 | 3.53 ± 0.06 | 0.18 ± 0.06 | 2.74 ± 0.10 | 0.44 / 2.79 | 0.54 / 2.52 (3D-ConvLSTM) | 0.68 / 1.59 (3D-ConvLSTM) |
| ARG-S | Image v1 | paper | S2+ADM | 3/3 | 0.13 ± 0.06 | 1.41 ± 0.05 | 0.07 ± 0.05 | 1.15 ± 0.03 | 0.55 / 1.03 | 0.65 / 0.90 (AFF) | 0.78 / 0.57 (AFF) |
| ARG-W | Image v1 | paper | S2+ADM | 3/3 | 0.32 ± 0.15 | 1.98 ± 0.21 | 0.35 ± 0.16 | 1.73 ± 0.21 | 0.61 / 1.48 | 0.78 / 1.12 (AFF) | 0.87 / 0.78 (AFF) |
| ARG-C | Image v1 | strict | S2 | 3/3 | 0.08 ± 0.06 | 3.61 ± 0.13 | 0.07 ± 0.05 | 2.92 ± 0.07 | 0.44 / 2.79 | 0.54 / 2.52 (3D-ConvLSTM) | 0.68 / 1.59 (3D-ConvLSTM) |
| ARG-S | Image v1 | strict | S2 | 3/3 | 0.36 ± 0.03 | 1.20 ± 0.02 | 0.38 ± 0.03 | 0.94 ± 0.02 | 0.55 / 1.03 | 0.59 / 0.98 (3D-LSTM) | 0.70 / 0.67 (3D-LSTM) |
| ARG-W | Image v1 | strict | S2 | 3/3 | 0.46 ± 0.07 | 1.76 ± 0.11 | 0.53 ± 0.05 | 1.48 ± 0.08 | 0.61 / 1.48 | 0.71 / 1.30 (3D-LSTM) | 0.81 / 0.92 (3D-LSTM) |
| ARG-C | Image v1 | strict | S2+ADM | 3/3 | 0.16 ± 0.09 | 3.44 ± 0.19 | 0.24 ± 0.14 | 2.63 ± 0.25 | 0.44 / 2.79 | 0.54 / 2.52 (3D-ConvLSTM) | 0.68 / 1.59 (3D-ConvLSTM) |
| ARG-S | Image v1 | strict | S2+ADM | 3/3 | 0.22 ± 0.03 | 1.33 ± 0.02 | 0.18 ± 0.01 | 1.08 ± 0.01 | 0.55 / 1.03 | 0.65 / 0.90 (AFF) | 0.78 / 0.57 (AFF) |
| ARG-W | Image v1 | strict | S2+ADM | 3/3 | 0.32 ± 0.10 | 1.97 ± 0.15 | 0.37 ± 0.12 | 1.71 ± 0.16 | 0.61 / 1.48 | 0.78 / 1.12 (AFF) | 0.87 / 0.78 (AFF) |

## LORO, farm regions

| Pair | Model | Policy | Inputs | Seeds complete | Pixel R² | Pixel RMSE | Field R² | Field RMSE | Paper LSTM pixel R² / RMSE (S2) | Paper best pixel R² / RMSE | Paper best field R² / RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BRA-C | Image v1 | paper | S2 | 3/3 | 0.14 ± 0.03 | 2.75 ± 0.04 | 0.18 ± 0.08 | 1.74 ± 0.09 | 0.27 / 2.46 | 0.34 / 2.33 (3D-LSTM) | 0.59 / 1.13 (3D-LSTM) |
| BRA-S | Image v1 | paper | S2 | 3/3 | 0.08 ± 0.02 | 1.14 ± 0.01 | 0.02 ± 0.07 | 0.71 ± 0.02 | 0.16 / 1.10 | 0.22 / 1.06 (3D-LSTM) | 0.41 / 0.53 (3D-LSTM) |
| BRA-W | Image v1 | paper | S2 | 3/3 | 0.06 ± 0.01 | 1.49 ± 0.01 | 0.06 ± 0.10 | 0.74 ± 0.04 | 0.11 / 1.48 | 0.20 / 1.41 (3D-LSTM) | 0.48 / 0.57 (3D-LSTM) |
| GER-R | Image v1 | paper | S2 | 3/3 | -0.36 ± 0.10 | 1.59 ± 0.06 | -0.84 ± 0.16 | 1.34 ± 0.06 | -0.05 / 1.71 | 0.17 / 1.52 (3D-LSTM) | 0.26 / 1.16 (3D-LSTM) |
| GER-W | Image v1 | paper | S2 | 3/3 | -0.23 ± 0.07 | 2.51 ± 0.07 | -0.72 ± 0.10 | 1.85 ± 0.06 | -0.78 / 3.93 | 0.10 / 2.79 (Transformer) | 0.14 / 1.76 (3D-ConvLSTM) |
| URG-S | Image v1 | paper | S2 | 3/3 | 0.20 ± 0.06 | 1.34 ± 0.05 | 0.27 ± 0.09 | 0.89 ± 0.06 | 0.34 / 1.28 | 0.36 / 1.26 (3D-ConvLSTM) | 0.68 / 0.61 (3D-ConvLSTM) |
| BRA-C | Image v1 | paper | S2+ADM | 3/3 | 0.13 ± 0.03 | 2.77 ± 0.04 | 0.22 ± 0.06 | 1.70 ± 0.07 | 0.27 / 2.46 | 0.37 / 2.29 (3D-LSTM) | 0.65 / 1.05 (3D-LSTM) |
| BRA-S | Image v1 | paper | S2+ADM | 3/3 | 0.09 ± 0.07 | 1.13 ± 0.04 | 0.01 ± 0.18 | 0.71 ± 0.06 | 0.16 / 1.10 | 0.35 / 0.97 (AFF) | 0.63 / 0.42 (AFF) |
| BRA-W | Image v1 | paper | S2+ADM | 3/3 | 0.00 ± 0.02 | 1.53 ± 0.02 | -0.03 ± 0.05 | 0.78 ± 0.02 | 0.11 / 1.48 | 0.18 / 1.42 (AFF) | 0.52 / 0.55 (AFF) |
| GER-R | Image v1 | paper | S2+ADM | 3/3 | -0.40 ± 0.21 | 1.61 ± 0.13 | -0.87 ± 0.45 | 1.35 ± 0.17 | -0.05 / 1.71 | 0.15 / 1.55 (AFF) | 0.25 / 1.16 (3D-ConvLSTM) |
| GER-W | Image v1 | paper | S2+ADM | 3/3 | -0.53 ± 0.16 | 2.80 ± 0.15 | -1.60 ± 0.49 | 2.27 ± 0.21 | -0.78 / 3.93 | 0.07 / 2.84 (3D-ConvLSTM) | 0.14 / 1.76 (3D-ConvLSTM) |
| URG-S | Image v1 | paper | S2+ADM | 3/3 | 0.25 ± 0.02 | 1.30 ± 0.02 | 0.42 ± 0.04 | 0.79 ± 0.03 | 0.34 / 1.28 | 0.36 / 1.26 (3D-ConvLSTM) | 0.68 / 0.61 (3D-ConvLSTM) |
| BRA-C | Image v1 | strict | S2 | 3/3 | 0.10 ± 0.03 | 2.81 ± 0.04 | 0.09 ± 0.09 | 1.84 ± 0.09 | 0.27 / 2.46 | 0.34 / 2.33 (3D-LSTM) | 0.59 / 1.13 (3D-LSTM) |
| BRA-S | Image v1 | strict | S2 | 3/3 | 0.06 ± 0.02 | 1.15 ± 0.01 | -0.05 ± 0.05 | 0.73 ± 0.02 | 0.16 / 1.10 | 0.22 / 1.06 (3D-LSTM) | 0.41 / 0.53 (3D-LSTM) |
| BRA-W | Image v1 | strict | S2 | 3/3 | 0.03 ± 0.03 | 1.51 ± 0.02 | 0.07 ± 0.04 | 0.74 ± 0.01 | 0.11 / 1.48 | 0.20 / 1.41 (3D-LSTM) | 0.48 / 0.57 (3D-LSTM) |
| GER-R | Image v1 | strict | S2 | 3/3 | -0.36 ± 0.16 | 1.59 ± 0.09 | -0.85 ± 0.29 | 1.35 ± 0.11 | -0.05 / 1.71 | 0.17 / 1.52 (3D-LSTM) | 0.26 / 1.16 (3D-LSTM) |
| GER-W | Image v1 | strict | S2 | 3/3 | -0.25 ± 0.07 | 2.53 ± 0.07 | -0.71 ± 0.12 | 1.85 ± 0.06 | -0.78 / 3.93 | 0.10 / 2.79 (Transformer) | 0.14 / 1.76 (3D-ConvLSTM) |
| URG-S | Image v1 | strict | S2 | 3/3 | 0.21 ± 0.05 | 1.33 ± 0.04 | 0.26 ± 0.09 | 0.90 ± 0.06 | 0.34 / 1.28 | 0.36 / 1.26 (3D-ConvLSTM) | 0.68 / 0.61 (3D-ConvLSTM) |
| BRA-C | Image v1 | strict | S2+ADM | 3/3 | 0.09 ± 0.04 | 2.83 ± 0.07 | 0.10 ± 0.10 | 1.83 ± 0.10 | 0.27 / 2.46 | 0.37 / 2.29 (3D-LSTM) | 0.65 / 1.05 (3D-LSTM) |
| BRA-S | Image v1 | strict | S2+ADM | 3/3 | -0.00 ± 0.04 | 1.19 ± 0.03 | -0.17 ± 0.12 | 0.77 ± 0.04 | 0.16 / 1.10 | 0.35 / 0.97 (AFF) | 0.63 / 0.42 (AFF) |
| BRA-W | Image v1 | strict | S2+ADM | 3/3 | -0.05 ± 0.04 | 1.57 ± 0.03 | -0.07 ± 0.08 | 0.79 ± 0.03 | 0.11 / 1.48 | 0.18 / 1.42 (AFF) | 0.52 / 0.55 (AFF) |
| GER-R | Image v1 | strict | S2+ADM | 3/3 | -0.33 ± 0.08 | 1.57 ± 0.05 | -0.71 ± 0.20 | 1.29 ± 0.07 | -0.05 / 1.71 | 0.15 / 1.55 (AFF) | 0.25 / 1.16 (3D-ConvLSTM) |
| GER-W | Image v1 | strict | S2+ADM | 3/3 | -0.54 ± 0.17 | 2.81 ± 0.15 | -1.60 ± 0.49 | 2.27 ± 0.21 | -0.78 / 3.93 | 0.07 / 2.84 (3D-ConvLSTM) | 0.14 / 1.76 (3D-ConvLSTM) |
| URG-S | Image v1 | strict | S2+ADM | 3/3 | 0.25 ± 0.03 | 1.29 ± 0.02 | 0.43 ± 0.06 | 0.79 ± 0.04 | 0.34 / 1.28 | 0.36 / 1.26 (3D-ConvLSTM) | 0.68 / 0.61 (3D-ConvLSTM) |

## LOYO (leave one year out)

| Pair | Model | Policy | Inputs | Seeds complete | Pixel R² | Pixel RMSE | Field R² | Field RMSE | Paper LSTM pixel R² / RMSE (S2) | Paper best pixel R² / RMSE | Paper best field R² / RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ARG-C | Image v1 | paper | S2 | 3/3 | 0.18 ± 0.04 | 3.42 ± 0.08 | 0.26 ± 0.04 | 2.61 ± 0.07 | 0.39 / 2.90 | 0.48 / 2.68 (3D-ConvLSTM) | 0.61 / 1.75 (3D-ConvLSTM) |
| ARG-S | Image v1 | paper | S2 | 3/3 | 0.41 ± 0.02 | 1.15 ± 0.02 | 0.38 ± 0.02 | 0.94 ± 0.01 | 0.46 / 1.12 | 0.55 / 1.03 (3D-LSTM) | 0.66 / 0.70 (3D-LSTM) |
| ARG-W | Image v1 | paper | S2 | 3/3 | 0.42 ± 0.08 | 1.82 ± 0.13 | 0.52 ± 0.06 | 1.49 ± 0.09 | 0.61 / 1.49 | 0.68 / 1.35 (3D-LSTM) | 0.78 / 1.00 (3D-ConvLSTM) |
| BRA-C | Image v1 | paper | S2 | 3/3 | 0.05 ± 0.05 | 2.89 ± 0.07 | 0.04 ± 0.11 | 1.88 ± 0.11 | 0.15 / 2.66 | 0.21 / 2.56 (3D-LSTM) | 0.36 / 1.42 (3D-LSTM) |
| BRA-S | Image v1 | paper | S2 | 3/3 | 0.02 ± 0.01 | 1.18 ± 0.00 | -0.20 ± 0.03 | 0.78 ± 0.01 | 0.04 / 1.18 | 0.19 / 1.08 (3D-ConvLSTM) | 0.36 / 0.55 (3D-ConvLSTM) |
| BRA-W | Image v1 | paper | S2 | 3/3 | -0.00 ± 0.04 | 1.53 ± 0.03 | -0.18 ± 0.13 | 0.83 ± 0.04 | 0.08 / 1.50 | 0.08 / 1.50 (LSTM) | -0.05 / 0.82 (LSTM) |
| GER-R | Image v1 | paper | S2 | 3/3 | -0.35 ± 0.16 | 1.58 ± 0.10 | -0.71 ± 0.29 | 1.29 ± 0.11 | -0.05 / 1.71 | 0.21 / 1.49 (3D-ConvLSTM) | 0.43 / 1.02 (3D-ConvLSTM) |
| GER-W | Image v1 | paper | S2 | 3/3 | -0.10 ± 0.05 | 2.37 ± 0.06 | -0.10 ± 0.09 | 1.48 ± 0.06 | -0.68 / 3.82 | 0.15 / 2.71 (3D-ConvLSTM) | 0.23 / 1.67 (3D-ConvLSTM) |
| URG-S | Image v1 | paper | S2 | 3/3 | 0.19 ± 0.06 | 1.34 ± 0.05 | 0.20 ± 0.15 | 0.93 ± 0.09 | 0.33 / 1.29 | 0.35 / 1.27 (3D-LSTM) | 0.63 / 0.65 (3D-ConvLSTM) |
| ARG-C | Image v1 | paper | S2+ADM | 3/3 | 0.02 ± 0.06 | 3.72 ± 0.12 | 0.12 ± 0.05 | 2.84 ± 0.08 | 0.39 / 2.90 | 0.44 / 2.78 (LSTM) | 0.42 / 2.13 (LSTM) |
| ARG-S | Image v1 | paper | S2+ADM | 3/3 | 0.30 ± 0.09 | 1.25 ± 0.08 | 0.26 ± 0.10 | 1.02 ± 0.07 | 0.46 / 1.12 | 0.66 / 0.89 (AFF) | 0.77 / 0.59 (AFF) |
| ARG-W | Image v1 | paper | S2+ADM | 3/3 | 0.20 ± 0.05 | 2.15 ± 0.07 | 0.28 ± 0.06 | 1.83 ± 0.08 | 0.61 / 1.49 | 0.72 / 1.27 (AFF) | 0.81 / 0.93 (AFF) |
| BRA-C | Image v1 | paper | S2+ADM | 3/3 | -0.04 ± 0.06 | 3.02 ± 0.08 | -0.05 ± 0.07 | 1.97 ± 0.06 | 0.15 / 2.66 | 0.29 / 2.43 (AFF) | 0.45 / 1.32 (3D-ConvLSTM) |
| BRA-S | Image v1 | paper | S2+ADM | 3/3 | 0.07 ± 0.02 | 1.14 ± 0.01 | -0.08 ± 0.06 | 0.74 ± 0.02 | 0.04 / 1.18 | 0.29 / 1.02 (AFF) | 0.45 / 0.52 (3D-LSTM) |
| BRA-W | Image v1 | paper | S2+ADM | 3/3 | 0.01 ± 0.00 | 1.53 ± 0.00 | -0.16 ± 0.01 | 0.82 ± 0.00 | 0.08 / 1.50 | 0.11 / 1.48 (AFF) | 0.14 / 0.74 (AFF) |
| GER-R | Image v1 | paper | S2+ADM | 3/3 | -0.15 ± 0.04 | 1.46 ± 0.03 | -0.37 ± 0.13 | 1.16 ± 0.05 | -0.05 / 1.71 | 0.31 / 1.38 (LSTM) | 0.58 / 0.87 (LSTM) |
| GER-W | Image v1 | paper | S2+ADM | 3/3 | 0.03 ± 0.01 | 2.23 ± 0.02 | 0.47 ± 0.06 | 1.03 ± 0.05 | -0.68 / 3.82 | 0.22 / 2.60 (3D-ConvLSTM) | 0.46 / 1.39 (3D-ConvLSTM) |
| URG-S | Image v1 | paper | S2+ADM | 3/3 | 0.03 ± 0.07 | 1.47 ± 0.05 | -0.17 ± 0.17 | 1.13 ± 0.09 | 0.33 / 1.29 | 0.32 / 1.30 (AFF) | 0.56 / 0.71 (3D-ConvLSTM) |
| ARG-C | Image v1 | strict | S2 | 3/3 | 0.15 ± 0.09 | 3.47 ± 0.19 | 0.22 ± 0.11 | 2.68 ± 0.19 | 0.39 / 2.90 | 0.48 / 2.68 (3D-ConvLSTM) | 0.61 / 1.75 (3D-ConvLSTM) |
| ARG-S | Image v1 | strict | S2 | 3/3 | 0.36 ± 0.06 | 1.20 ± 0.06 | 0.30 ± 0.08 | 0.99 ± 0.06 | 0.46 / 1.12 | 0.55 / 1.03 (3D-LSTM) | 0.66 / 0.70 (3D-LSTM) |
| ARG-W | Image v1 | strict | S2 | 3/3 | 0.27 ± 0.08 | 2.06 ± 0.11 | 0.33 ± 0.07 | 1.76 ± 0.10 | 0.61 / 1.49 | 0.68 / 1.35 (3D-LSTM) | 0.78 / 1.00 (3D-ConvLSTM) |
| BRA-C | Image v1 | strict | S2 | 3/3 | -0.02 ± 0.05 | 3.31 ± 0.08 | -0.20 ± 0.12 | 2.19 ± 0.11 | 0.15 / 2.66 | 0.21 / 2.56 (3D-LSTM) | 0.36 / 1.42 (3D-LSTM) |
| BRA-S | Image v1 | strict | S2 | 3/3 | -0.06 ± 0.00 | 1.22 ± 0.00 | -0.25 ± 0.01 | 0.80 ± 0.00 | 0.04 / 1.18 | 0.19 / 1.08 (3D-ConvLSTM) | 0.36 / 0.55 (3D-ConvLSTM) |
| BRA-W | Image v1 | strict | S2 | 3/3 | 0.03 ± 0.04 | 1.51 ± 0.03 | 0.02 ± 0.11 | 0.76 ± 0.04 | 0.08 / 1.50 | 0.08 / 1.50 (LSTM) | -0.05 / 0.82 (LSTM) |
| GER-R | Image v1 | strict | S2 | 3/3 | -0.31 ± 0.09 | 1.56 ± 0.05 | -0.72 ± 0.19 | 1.30 ± 0.07 | -0.05 / 1.71 | 0.21 / 1.49 (3D-ConvLSTM) | 0.43 / 1.02 (3D-ConvLSTM) |
| GER-W | Image v1 | strict | S2 | 3/3 | -0.49 ± 0.10 | 2.77 ± 0.09 | -1.29 ± 0.39 | 2.13 ± 0.18 | -0.68 / 3.82 | 0.15 / 2.71 (3D-ConvLSTM) | 0.23 / 1.67 (3D-ConvLSTM) |
| URG-S | Image v1 | strict | S2 | 3/3 | 0.24 ± 0.02 | 1.30 ± 0.02 | 0.35 ± 0.06 | 0.84 ± 0.04 | 0.33 / 1.29 | 0.35 / 1.27 (3D-LSTM) | 0.63 / 0.65 (3D-ConvLSTM) |
| ARG-C | Image v1 | strict | S2+ADM | 3/3 | 0.11 ± 0.08 | 3.55 ± 0.15 | 0.19 ± 0.10 | 2.72 ± 0.16 | 0.39 / 2.90 | 0.44 / 2.78 (LSTM) | 0.42 / 2.13 (LSTM) |
| ARG-S | Image v1 | strict | S2+ADM | 3/3 | 0.30 ± 0.05 | 1.26 ± 0.04 | 0.28 ± 0.02 | 1.01 ± 0.01 | 0.46 / 1.12 | 0.66 / 0.89 (AFF) | 0.77 / 0.59 (AFF) |
| ARG-W | Image v1 | strict | S2+ADM | 3/3 | 0.01 ± 0.06 | 2.39 ± 0.08 | 0.13 ± 0.08 | 2.01 ± 0.09 | 0.61 / 1.49 | 0.72 / 1.27 (AFF) | 0.81 / 0.93 (AFF) |
| BRA-C | Image v1 | strict | S2+ADM | 3/3 | 0.01 ± 0.00 | 3.26 ± 0.00 | -0.11 ± 0.02 | 2.11 ± 0.02 | 0.15 / 2.66 | 0.29 / 2.43 (AFF) | 0.45 / 1.32 (3D-ConvLSTM) |
| BRA-S | Image v1 | strict | S2+ADM | 3/3 | -0.17 ± 0.09 | 1.29 ± 0.05 | -0.45 ± 0.21 | 0.86 ± 0.06 | 0.04 / 1.18 | 0.29 / 1.02 (AFF) | 0.45 / 0.52 (3D-LSTM) |
| BRA-W | Image v1 | strict | S2+ADM | 3/3 | -0.08 ± 0.04 | 1.59 ± 0.03 | -0.10 ± 0.12 | 0.80 ± 0.04 | 0.08 / 1.50 | 0.11 / 1.48 (AFF) | 0.14 / 0.74 (AFF) |
| GER-R | Image v1 | strict | S2+ADM | 3/3 | -0.18 ± 0.13 | 1.48 ± 0.08 | -0.42 ± 0.34 | 1.17 ± 0.14 | -0.05 / 1.71 | 0.31 / 1.38 (LSTM) | 0.58 / 0.87 (LSTM) |
| GER-W | Image v1 | strict | S2+ADM | 3/3 | -0.43 ± 0.13 | 2.71 ± 0.12 | -1.19 ± 0.36 | 2.09 ± 0.17 | -0.68 / 3.82 | 0.22 / 2.60 (3D-ConvLSTM) | 0.46 / 1.39 (3D-ConvLSTM) |
| URG-S | Image v1 | strict | S2+ADM | 3/3 | 0.01 ± 0.03 | 1.48 ± 0.02 | -0.17 ± 0.10 | 1.13 ± 0.05 | 0.33 / 1.29 | 0.32 / 1.30 (AFF) | 0.56 / 0.71 (3D-ConvLSTM) |
