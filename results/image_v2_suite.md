# Image model v2 results (image_v2 suite)

Generated 2026-10-02 22:41 UTC by `yieldsat_results_tables.py` from pooled out-of-fold predictions (cluster results, image_v2).
**324 of 324 experiments (pair × protocol × policy × inputs × seed) complete.**

- **Metric = the paper's computation:** for each experiment, the held-out predictions of all folds are pooled (each cell is held out exactly once) and R²/RMSE are computed once. Values are mean ± std of these pooled scores over seeds; a seed counts only when all its folds are finished.
- **Pixel**: every held-out 10 m cell. **Field**: per field season, mean prediction vs mean target. R² = 1 − SSE/SST; RMSE in t/ha.
- **Policy** `paper` = the paper's grouping; `strict` additionally removes training seasons that share a physical field with the test fold.
- **Paper** numbers are from Pathak et al. (CVPR 2026, `results/yieldsat/paper_benchmark.csv`); "best" is the best model per row and modality set. Why pooled: [negative_r2_investigation.md](negative_r2_investigation.md).

Image model v2 (plan v2 in spec/yieldsat-image-training.md): v1 + per-pixel full S2 time-series branch, tile level + residual head, full-coverage tiles (training on tiles with >= 2,048 valid cells; validation and test on every cell). Metrics cover 100% of the point model's cells, so they are directly comparable with the paper and the point suite. GER-R/GER-W are warm-started from v2 donors.

## CV10 (10-fold, grouped by field season)

| Pair | Model | Policy | Inputs | Seeds complete | Pixel R² | Pixel RMSE | Field R² | Field RMSE | Paper LSTM pixel R² / RMSE (S2) | Paper best pixel R² / RMSE | Paper best field R² / RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ARG-C | Image v2 | paper | S2 | 3/3 | 0.24 ± 0.05 | 3.23 ± 0.10 | 0.39 ± 0.03 | 2.20 ± 0.05 | 0.59 / 2.40 | 0.65 / 2.20 (3D-ConvLSTM) | 0.84 / 1.13 (3D-ConvLSTM) |
| ARG-S | Image v2 | paper | S2 | 3/3 | 0.48 ± 0.00 | 1.10 ± 0.00 | 0.60 ± 0.01 | 0.77 ± 0.01 | 0.60 / 0.96 | 0.65 / 0.90 (3D-ConvLSTM) | 0.79 / 0.55 (3D-ConvLSTM) |
| ARG-W | Image v2 | paper | S2 | 3/3 | 0.53 ± 0.05 | 1.64 ± 0.09 | 0.63 ± 0.06 | 1.29 ± 0.10 | 0.74 / 1.21 | 0.79 / 1.10 (3D-ConvLSTM) | 0.92 / 0.62 (3D-ConvLSTM) |
| BRA-C | Image v2 | paper | S2 | 3/3 | 0.28 ± 0.02 | 2.45 ± 0.03 | 0.48 ± 0.03 | 1.28 ± 0.04 | 0.42 / 2.20 | 0.46 / 2.13 (3D-LSTM) | 0.82 / 0.74 (3D-ConvLSTM) |
| BRA-S | Image v2 | paper | S2 | 3/3 | 0.24 ± 0.02 | 1.05 ± 0.01 | 0.46 ± 0.04 | 0.51 ± 0.02 | 0.34 / 0.98 | 0.39 / 0.94 (3D-ConvLSTM) | 0.76 / 0.34 (3D-ConvLSTM) |
| BRA-W | Image v2 | paper | S2 | 3/3 | 0.08 ± 0.01 | 1.50 ± 0.01 | 0.28 ± 0.07 | 0.68 ± 0.03 | 0.22 / 1.39 | 0.24 / 1.37 (3D-LSTM) | 0.73 / 0.42 (3D-ConvLSTM) |
| GER-R | Image v2 | paper | S2 | 3/3 | 0.01 ± 0.11 | 1.66 ± 0.09 | -0.11 ± 0.20 | 1.41 ± 0.13 | 0.36 / 1.33 | 0.49 / 1.20 (3D-ConvLSTM) | 0.82 / 0.57 (3D-LSTM) |
| GER-W | Image v2 | paper | S2 | 3/3 | 0.01 ± 0.01 | 2.93 ± 0.02 | -0.15 ± 0.05 | 2.03 ± 0.04 | -0.32 / 3.38 | 0.34 / 2.40 (3D-ConvLSTM) | 0.65 / 1.12 (3D-ConvLSTM) |
| URG-S | Image v2 | paper | S2 | 3/3 | 0.25 ± 0.00 | 1.37 ± 0.00 | 0.44 ± 0.03 | 0.80 ± 0.02 | 0.37 / 1.26 | 0.41 / 1.22 (3D-ConvLSTM) | 0.77 / 0.51 (3D-ConvLSTM) |
| ARG-C | Image v2 | paper | S2+ADM | 3/3 | 0.24 ± 0.04 | 3.24 ± 0.09 | 0.39 ± 0.03 | 2.18 ± 0.05 | 0.59 / 2.40 | 0.70 / 2.03 (AFF) | 0.84 / 1.12 (AFF) |
| ARG-S | Image v2 | paper | S2+ADM | 3/3 | 0.49 ± 0.01 | 1.09 ± 0.01 | 0.61 ± 0.01 | 0.76 ± 0.01 | 0.60 / 0.96 | 0.73 / 0.79 (AFF) | 0.84 / 0.49 (AFF) |
| ARG-W | Image v2 | paper | S2+ADM | 3/3 | 0.53 ± 0.01 | 1.64 ± 0.02 | 0.65 ± 0.02 | 1.26 ± 0.04 | 0.74 / 1.21 | 0.84 / 0.96 (AFF) | 0.92 / 0.60 (AFF) |
| BRA-C | Image v2 | paper | S2+ADM | 3/3 | 0.26 ± 0.03 | 2.48 ± 0.05 | 0.43 ± 0.04 | 1.34 ± 0.05 | 0.42 / 2.20 | 0.46 / 2.12 (AFF) | 0.84 / 0.70 (AFF) |
| BRA-S | Image v2 | paper | S2+ADM | 3/3 | 0.25 ± 0.01 | 1.04 ± 0.01 | 0.48 ± 0.03 | 0.50 ± 0.02 | 0.34 / 0.98 | 0.44 / 0.90 (AFF) | 0.80 / 0.31 (AFF) |
| BRA-W | Image v2 | paper | S2+ADM | 3/3 | 0.06 ± 0.01 | 1.52 ± 0.01 | 0.20 ± 0.03 | 0.71 ± 0.02 | 0.22 / 1.39 | 0.24 / 1.37 (AFF) | 0.72 / 0.42 (3D-ConvLSTM) |
| GER-R | Image v2 | paper | S2+ADM | 3/3 | -0.01 ± 0.04 | 1.68 ± 0.03 | -0.19 ± 0.07 | 1.46 ± 0.05 | 0.36 / 1.33 | 0.49 / 1.20 (AFF) | 0.81 / 0.59 (3D-LSTM) |
| GER-W | Image v2 | paper | S2+ADM | 3/3 | 0.04 ± 0.03 | 2.89 ± 0.05 | -0.08 ± 0.06 | 1.98 ± 0.05 | -0.32 / 3.38 | 0.44 / 2.20 (AFF) | 0.77 / 0.90 (MMGF) |
| URG-S | Image v2 | paper | S2+ADM | 3/3 | 0.24 ± 0.01 | 1.38 ± 0.01 | 0.43 ± 0.02 | 0.81 ± 0.01 | 0.37 / 1.26 | 0.43 / 1.19 (AFF) | 0.81 / 0.46 (AFF) |
| ARG-C | Image v2 | strict | S2 | 3/3 | 0.22 ± 0.01 | 3.28 ± 0.03 | 0.31 ± 0.03 | 2.33 ± 0.05 | 0.59 / 2.40 | 0.65 / 2.20 (3D-ConvLSTM) | 0.84 / 1.13 (3D-ConvLSTM) |
| ARG-S | Image v2 | strict | S2 | 3/3 | 0.51 ± 0.01 | 1.07 ± 0.01 | 0.61 ± 0.01 | 0.76 ± 0.01 | 0.60 / 0.96 | 0.65 / 0.90 (3D-ConvLSTM) | 0.79 / 0.55 (3D-ConvLSTM) |
| ARG-W | Image v2 | strict | S2 | 3/3 | 0.43 ± 0.05 | 1.80 ± 0.08 | 0.53 ± 0.09 | 1.45 ± 0.14 | 0.74 / 1.21 | 0.79 / 1.10 (3D-ConvLSTM) | 0.92 / 0.62 (3D-ConvLSTM) |
| BRA-C | Image v2 | strict | S2 | 3/3 | 0.24 ± 0.01 | 2.52 ± 0.01 | 0.39 ± 0.02 | 1.38 ± 0.03 | 0.42 / 2.20 | 0.46 / 2.13 (3D-LSTM) | 0.82 / 0.74 (3D-ConvLSTM) |
| BRA-S | Image v2 | strict | S2 | 3/3 | 0.21 ± 0.00 | 1.07 ± 0.00 | 0.40 ± 0.02 | 0.54 ± 0.01 | 0.34 / 0.98 | 0.39 / 0.94 (3D-ConvLSTM) | 0.76 / 0.34 (3D-ConvLSTM) |
| BRA-W | Image v2 | strict | S2 | 3/3 | 0.06 ± 0.01 | 1.52 ± 0.01 | 0.22 ± 0.06 | 0.70 ± 0.03 | 0.22 / 1.39 | 0.24 / 1.37 (3D-LSTM) | 0.73 / 0.42 (3D-ConvLSTM) |
| GER-R | Image v2 | strict | S2 | 3/3 | 0.03 ± 0.08 | 1.65 ± 0.07 | -0.11 ± 0.14 | 1.41 ± 0.09 | 0.36 / 1.33 | 0.49 / 1.20 (3D-ConvLSTM) | 0.82 / 0.57 (3D-LSTM) |
| GER-W | Image v2 | strict | S2 | 3/3 | 0.00 ± 0.01 | 2.95 ± 0.01 | -0.16 ± 0.02 | 2.04 ± 0.01 | -0.32 / 3.38 | 0.34 / 2.40 (3D-ConvLSTM) | 0.65 / 1.12 (3D-ConvLSTM) |
| URG-S | Image v2 | strict | S2 | 3/3 | 0.25 ± 0.01 | 1.36 ± 0.00 | 0.43 ± 0.02 | 0.80 ± 0.01 | 0.37 / 1.26 | 0.41 / 1.22 (3D-ConvLSTM) | 0.77 / 0.51 (3D-ConvLSTM) |
| ARG-C | Image v2 | strict | S2+ADM | 3/3 | 0.24 ± 0.01 | 3.23 ± 0.02 | 0.35 ± 0.05 | 2.25 ± 0.09 | 0.59 / 2.40 | 0.70 / 2.03 (AFF) | 0.84 / 1.12 (AFF) |
| ARG-S | Image v2 | strict | S2+ADM | 3/3 | 0.50 ± 0.01 | 1.07 ± 0.01 | 0.60 ± 0.00 | 0.76 ± 0.00 | 0.60 / 0.96 | 0.73 / 0.79 (AFF) | 0.84 / 0.49 (AFF) |
| ARG-W | Image v2 | strict | S2+ADM | 3/3 | 0.47 ± 0.01 | 1.75 ± 0.02 | 0.58 ± 0.01 | 1.39 ± 0.01 | 0.74 / 1.21 | 0.84 / 0.96 (AFF) | 0.92 / 0.60 (AFF) |
| BRA-C | Image v2 | strict | S2+ADM | 3/3 | 0.21 ± 0.02 | 2.56 ± 0.03 | 0.35 ± 0.04 | 1.42 ± 0.05 | 0.42 / 2.20 | 0.46 / 2.12 (AFF) | 0.84 / 0.70 (AFF) |
| BRA-S | Image v2 | strict | S2+ADM | 3/3 | 0.22 ± 0.01 | 1.06 ± 0.01 | 0.41 ± 0.01 | 0.53 ± 0.01 | 0.34 / 0.98 | 0.44 / 0.90 (AFF) | 0.80 / 0.31 (AFF) |
| BRA-W | Image v2 | strict | S2+ADM | 3/3 | 0.05 ± 0.01 | 1.53 ± 0.01 | 0.26 ± 0.05 | 0.68 ± 0.02 | 0.22 / 1.39 | 0.24 / 1.37 (AFF) | 0.72 / 0.42 (3D-ConvLSTM) |
| GER-R | Image v2 | strict | S2+ADM | 3/3 | -0.02 ± 0.07 | 1.69 ± 0.06 | -0.20 ± 0.08 | 1.47 ± 0.05 | 0.36 / 1.33 | 0.49 / 1.20 (AFF) | 0.81 / 0.59 (3D-LSTM) |
| GER-W | Image v2 | strict | S2+ADM | 3/3 | 0.04 ± 0.04 | 2.89 ± 0.06 | -0.06 ± 0.08 | 1.95 ± 0.07 | -0.32 / 3.38 | 0.44 / 2.20 (AFF) | 0.77 / 0.90 (MMGF) |
| URG-S | Image v2 | strict | S2+ADM | 3/3 | 0.24 ± 0.01 | 1.38 ± 0.01 | 0.43 ± 0.01 | 0.81 ± 0.01 | 0.37 / 1.26 | 0.43 / 1.19 (AFF) | 0.81 / 0.46 (AFF) |

## LORO, provinces (Argentina; the paper's regions)

| Pair | Model | Policy | Inputs | Seeds complete | Pixel R² | Pixel RMSE | Field R² | Field RMSE | Paper LSTM pixel R² / RMSE (S2) | Paper best pixel R² / RMSE | Paper best field R² / RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ARG-C | Image v2 | paper | S2 | 3/3 | 0.03 ± 0.06 | 3.66 ± 0.11 | 0.04 ± 0.11 | 2.74 ± 0.16 | 0.44 / 2.79 | 0.54 / 2.52 (3D-ConvLSTM) | 0.68 / 1.59 (3D-ConvLSTM) |
| ARG-S | Image v2 | paper | S2 | 3/3 | 0.35 ± 0.00 | 1.23 ± 0.00 | 0.40 ± 0.02 | 0.93 ± 0.02 | 0.55 / 1.03 | 0.59 / 0.98 (3D-LSTM) | 0.70 / 0.67 (3D-LSTM) |
| ARG-W | Image v2 | paper | S2 | 3/3 | 0.30 ± 0.02 | 1.99 ± 0.03 | 0.40 ± 0.01 | 1.65 ± 0.02 | 0.61 / 1.48 | 0.71 / 1.30 (3D-LSTM) | 0.81 / 0.92 (3D-LSTM) |
| ARG-C | Image v2 | paper | S2+ADM | 3/3 | 0.04 ± 0.05 | 3.64 ± 0.09 | 0.05 ± 0.08 | 2.73 ± 0.11 | 0.44 / 2.79 | 0.54 / 2.52 (3D-ConvLSTM) | 0.68 / 1.59 (3D-ConvLSTM) |
| ARG-S | Image v2 | paper | S2+ADM | 3/3 | 0.28 ± 0.10 | 1.29 ± 0.09 | 0.36 ± 0.12 | 0.97 ± 0.09 | 0.55 / 1.03 | 0.65 / 0.90 (AFF) | 0.78 / 0.57 (AFF) |
| ARG-W | Image v2 | paper | S2+ADM | 3/3 | 0.29 ± 0.09 | 2.01 ± 0.13 | 0.40 ± 0.09 | 1.65 ± 0.12 | 0.61 / 1.48 | 0.78 / 1.12 (AFF) | 0.87 / 0.78 (AFF) |
| ARG-C | Image v2 | strict | S2 | 3/3 | 0.10 ± 0.05 | 3.54 ± 0.10 | 0.13 ± 0.07 | 2.61 ± 0.10 | 0.44 / 2.79 | 0.54 / 2.52 (3D-ConvLSTM) | 0.68 / 1.59 (3D-ConvLSTM) |
| ARG-S | Image v2 | strict | S2 | 3/3 | 0.34 ± 0.05 | 1.24 ± 0.05 | 0.37 ± 0.10 | 0.96 ± 0.07 | 0.55 / 1.03 | 0.59 / 0.98 (3D-LSTM) | 0.70 / 0.67 (3D-LSTM) |
| ARG-W | Image v2 | strict | S2 | 3/3 | 0.36 ± 0.06 | 1.91 ± 0.09 | 0.46 ± 0.07 | 1.57 ± 0.10 | 0.61 / 1.48 | 0.71 / 1.30 (3D-LSTM) | 0.81 / 0.92 (3D-LSTM) |
| ARG-C | Image v2 | strict | S2+ADM | 3/3 | 0.14 ± 0.02 | 3.45 ± 0.05 | 0.22 ± 0.05 | 2.47 ± 0.08 | 0.44 / 2.79 | 0.54 / 2.52 (3D-ConvLSTM) | 0.68 / 1.59 (3D-ConvLSTM) |
| ARG-S | Image v2 | strict | S2+ADM | 3/3 | 0.32 ± 0.04 | 1.26 ± 0.04 | 0.40 ± 0.03 | 0.94 ± 0.03 | 0.55 / 1.03 | 0.65 / 0.90 (AFF) | 0.78 / 0.57 (AFF) |
| ARG-W | Image v2 | strict | S2+ADM | 3/3 | 0.31 ± 0.03 | 1.98 ± 0.05 | 0.43 ± 0.04 | 1.62 ± 0.05 | 0.61 / 1.48 | 0.78 / 1.12 (AFF) | 0.87 / 0.78 (AFF) |

## LORO, farm regions

| Pair | Model | Policy | Inputs | Seeds complete | Pixel R² | Pixel RMSE | Field R² | Field RMSE | Paper LSTM pixel R² / RMSE (S2) | Paper best pixel R² / RMSE | Paper best field R² / RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BRA-C | Image v2 | paper | S2 | 3/3 | 0.19 ± 0.01 | 2.59 ± 0.02 | 0.28 ± 0.06 | 1.50 ± 0.06 | 0.27 / 2.46 | 0.34 / 2.33 (3D-LSTM) | 0.59 / 1.13 (3D-LSTM) |
| BRA-S | Image v2 | paper | S2 | 3/3 | 0.13 ± 0.03 | 1.12 ± 0.02 | 0.19 ± 0.08 | 0.62 ± 0.03 | 0.16 / 1.10 | 0.22 / 1.06 (3D-LSTM) | 0.41 / 0.53 (3D-LSTM) |
| BRA-W | Image v2 | paper | S2 | 3/3 | 0.03 ± 0.02 | 1.55 ± 0.02 | 0.16 ± 0.04 | 0.73 ± 0.02 | 0.11 / 1.48 | 0.20 / 1.41 (3D-LSTM) | 0.48 / 0.57 (3D-LSTM) |
| GER-R | Image v2 | paper | S2 | 3/3 | -0.12 ± 0.07 | 1.77 ± 0.05 | -0.33 ± 0.10 | 1.55 ± 0.06 | -0.05 / 1.71 | 0.17 / 1.52 (3D-LSTM) | 0.26 / 1.16 (3D-LSTM) |
| GER-W | Image v2 | paper | S2 | 3/3 | -0.23 ± 0.23 | 3.26 ± 0.30 | -0.62 ± 0.47 | 2.40 ± 0.34 | -0.78 / 3.93 | 0.10 / 2.79 (Transformer) | 0.14 / 1.76 (3D-ConvLSTM) |
| URG-S | Image v2 | paper | S2 | 3/3 | 0.18 ± 0.02 | 1.43 ± 0.02 | 0.35 ± 0.05 | 0.86 ± 0.03 | 0.34 / 1.28 | 0.36 / 1.26 (3D-ConvLSTM) | 0.68 / 0.61 (3D-ConvLSTM) |
| BRA-C | Image v2 | paper | S2+ADM | 3/3 | 0.15 ± 0.04 | 2.65 ± 0.06 | 0.19 ± 0.08 | 1.60 ± 0.08 | 0.27 / 2.46 | 0.37 / 2.29 (3D-LSTM) | 0.65 / 1.05 (3D-LSTM) |
| BRA-S | Image v2 | paper | S2+ADM | 3/3 | 0.13 ± 0.03 | 1.12 ± 0.02 | 0.19 ± 0.07 | 0.62 ± 0.03 | 0.16 / 1.10 | 0.35 / 0.97 (AFF) | 0.63 / 0.42 (AFF) |
| BRA-W | Image v2 | paper | S2+ADM | 3/3 | 0.01 ± 0.02 | 1.56 ± 0.01 | 0.10 ± 0.06 | 0.75 ± 0.02 | 0.11 / 1.48 | 0.18 / 1.42 (AFF) | 0.52 / 0.55 (AFF) |
| GER-R | Image v2 | paper | S2+ADM | 3/3 | -0.20 ± 0.06 | 1.83 ± 0.05 | -0.45 ± 0.20 | 1.61 ± 0.11 | -0.05 / 1.71 | 0.15 / 1.55 (AFF) | 0.25 / 1.16 (3D-ConvLSTM) |
| GER-W | Image v2 | paper | S2+ADM | 3/3 | -0.26 ± 0.15 | 3.30 ± 0.20 | -0.66 ± 0.34 | 2.44 ± 0.25 | -0.78 / 3.93 | 0.07 / 2.84 (3D-ConvLSTM) | 0.14 / 1.76 (3D-ConvLSTM) |
| URG-S | Image v2 | paper | S2+ADM | 3/3 | 0.17 ± 0.02 | 1.44 ± 0.01 | 0.32 ± 0.04 | 0.88 ± 0.02 | 0.34 / 1.28 | 0.36 / 1.26 (3D-ConvLSTM) | 0.68 / 0.61 (3D-ConvLSTM) |
| BRA-C | Image v2 | strict | S2 | 3/3 | 0.15 ± 0.01 | 2.67 ± 0.02 | 0.19 ± 0.03 | 1.59 ± 0.03 | 0.27 / 2.46 | 0.34 / 2.33 (3D-LSTM) | 0.59 / 1.13 (3D-LSTM) |
| BRA-S | Image v2 | strict | S2 | 3/3 | 0.07 ± 0.02 | 1.16 ± 0.01 | 0.10 ± 0.09 | 0.66 ± 0.03 | 0.16 / 1.10 | 0.22 / 1.06 (3D-LSTM) | 0.41 / 0.53 (3D-LSTM) |
| BRA-W | Image v2 | strict | S2 | 3/3 | -0.03 ± 0.02 | 1.59 ± 0.01 | -0.08 ± 0.08 | 0.83 ± 0.03 | 0.11 / 1.48 | 0.20 / 1.41 (3D-LSTM) | 0.48 / 0.57 (3D-LSTM) |
| GER-R | Image v2 | strict | S2 | 3/3 | -0.14 ± 0.13 | 1.79 ± 0.10 | -0.37 ± 0.22 | 1.57 ± 0.13 | -0.05 / 1.71 | 0.17 / 1.52 (3D-LSTM) | 0.26 / 1.16 (3D-LSTM) |
| GER-W | Image v2 | strict | S2 | 3/3 | -0.22 ± 0.22 | 3.25 ± 0.29 | -0.61 ± 0.44 | 2.39 ± 0.32 | -0.78 / 3.93 | 0.10 / 2.79 (Transformer) | 0.14 / 1.76 (3D-ConvLSTM) |
| URG-S | Image v2 | strict | S2 | 3/3 | 0.16 ± 0.01 | 1.44 ± 0.00 | 0.31 ± 0.05 | 0.89 ± 0.03 | 0.34 / 1.28 | 0.36 / 1.26 (3D-ConvLSTM) | 0.68 / 0.61 (3D-ConvLSTM) |
| BRA-C | Image v2 | strict | S2+ADM | 3/3 | 0.13 ± 0.02 | 2.69 ± 0.02 | 0.17 ± 0.05 | 1.61 ± 0.05 | 0.27 / 2.46 | 0.37 / 2.29 (3D-LSTM) | 0.65 / 1.05 (3D-LSTM) |
| BRA-S | Image v2 | strict | S2+ADM | 3/3 | 0.08 ± 0.02 | 1.15 ± 0.01 | 0.13 ± 0.09 | 0.64 ± 0.03 | 0.16 / 1.10 | 0.35 / 0.97 (AFF) | 0.63 / 0.42 (AFF) |
| BRA-W | Image v2 | strict | S2+ADM | 3/3 | -0.06 ± 0.04 | 1.62 ± 0.03 | -0.07 ± 0.09 | 0.82 ± 0.03 | 0.11 / 1.48 | 0.18 / 1.42 (AFF) | 0.52 / 0.55 (AFF) |
| GER-R | Image v2 | strict | S2+ADM | 3/3 | -0.12 ± 0.05 | 1.77 ± 0.04 | -0.35 ± 0.13 | 1.56 ± 0.08 | -0.05 / 1.71 | 0.15 / 1.55 (AFF) | 0.25 / 1.16 (3D-ConvLSTM) |
| GER-W | Image v2 | strict | S2+ADM | 3/3 | -0.27 ± 0.14 | 3.32 ± 0.18 | -0.69 ± 0.29 | 2.46 ± 0.21 | -0.78 / 3.93 | 0.07 / 2.84 (3D-ConvLSTM) | 0.14 / 1.76 (3D-ConvLSTM) |
| URG-S | Image v2 | strict | S2+ADM | 3/3 | 0.15 ± 0.01 | 1.45 ± 0.01 | 0.30 ± 0.03 | 0.89 ± 0.02 | 0.34 / 1.28 | 0.36 / 1.26 (3D-ConvLSTM) | 0.68 / 0.61 (3D-ConvLSTM) |

## LOYO (leave one year out)

| Pair | Model | Policy | Inputs | Seeds complete | Pixel R² | Pixel RMSE | Field R² | Field RMSE | Paper LSTM pixel R² / RMSE (S2) | Paper best pixel R² / RMSE | Paper best field R² / RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ARG-C | Image v2 | paper | S2 | 3/3 | -0.01 ± 0.08 | 3.74 ± 0.15 | 0.03 ± 0.09 | 2.76 ± 0.13 | 0.39 / 2.90 | 0.48 / 2.68 (3D-ConvLSTM) | 0.61 / 1.75 (3D-ConvLSTM) |
| ARG-S | Image v2 | paper | S2 | 3/3 | 0.41 ± 0.03 | 1.18 ± 0.03 | 0.44 ± 0.05 | 0.91 ± 0.04 | 0.46 / 1.12 | 0.55 / 1.03 (3D-LSTM) | 0.66 / 0.70 (3D-LSTM) |
| ARG-W | Image v2 | paper | S2 | 3/3 | 0.17 ± 0.05 | 2.17 ± 0.07 | 0.27 ± 0.07 | 1.82 ± 0.08 | 0.61 / 1.49 | 0.68 / 1.35 (3D-LSTM) | 0.78 / 1.00 (3D-ConvLSTM) |
| BRA-C | Image v2 | paper | S2 | 3/3 | 0.15 ± 0.04 | 2.67 ± 0.07 | 0.22 ± 0.10 | 1.56 ± 0.10 | 0.15 / 2.66 | 0.21 / 2.56 (3D-LSTM) | 0.36 / 1.42 (3D-LSTM) |
| BRA-S | Image v2 | paper | S2 | 3/3 | -0.01 ± 0.03 | 1.21 ± 0.02 | -0.25 ± 0.06 | 0.77 ± 0.02 | 0.04 / 1.18 | 0.19 / 1.08 (3D-ConvLSTM) | 0.36 / 0.55 (3D-ConvLSTM) |
| BRA-W | Image v2 | paper | S2 | 3/3 | 0.00 ± 0.03 | 1.57 ± 0.02 | -0.07 ± 0.07 | 0.82 ± 0.03 | 0.08 / 1.50 | 0.08 / 1.50 (LSTM) | -0.05 / 0.82 (LSTM) |
| GER-R | Image v2 | paper | S2 | 3/3 | -0.01 ± 0.06 | 1.68 ± 0.05 | -0.14 ± 0.11 | 1.43 ± 0.07 | -0.05 / 1.71 | 0.21 / 1.49 (3D-ConvLSTM) | 0.43 / 1.02 (3D-ConvLSTM) |
| GER-W | Image v2 | paper | S2 | 3/3 | -0.01 ± 0.04 | 2.96 ± 0.06 | -0.11 ± 0.09 | 2.00 ± 0.09 | -0.68 / 3.82 | 0.15 / 2.71 (3D-ConvLSTM) | 0.23 / 1.67 (3D-ConvLSTM) |
| URG-S | Image v2 | paper | S2 | 3/3 | -0.01 ± 0.03 | 1.58 ± 0.03 | -0.13 ± 0.06 | 1.14 ± 0.03 | 0.33 / 1.29 | 0.35 / 1.27 (3D-LSTM) | 0.63 / 0.65 (3D-ConvLSTM) |
| ARG-C | Image v2 | paper | S2+ADM | 3/3 | 0.03 ± 0.02 | 3.67 ± 0.04 | 0.09 ± 0.04 | 2.68 ± 0.06 | 0.39 / 2.90 | 0.44 / 2.78 (LSTM) | 0.42 / 2.13 (LSTM) |
| ARG-S | Image v2 | paper | S2+ADM | 3/3 | 0.37 ± 0.07 | 1.22 ± 0.07 | 0.37 ± 0.11 | 0.95 ± 0.08 | 0.46 / 1.12 | 0.66 / 0.89 (AFF) | 0.77 / 0.59 (AFF) |
| ARG-W | Image v2 | paper | S2+ADM | 3/3 | 0.15 ± 0.04 | 2.20 ± 0.05 | 0.26 ± 0.03 | 1.84 ± 0.04 | 0.61 / 1.49 | 0.72 / 1.27 (AFF) | 0.81 / 0.93 (AFF) |
| BRA-C | Image v2 | paper | S2+ADM | 3/3 | 0.12 ± 0.03 | 2.70 ± 0.04 | 0.17 ± 0.05 | 1.61 ± 0.05 | 0.15 / 2.66 | 0.29 / 2.43 (AFF) | 0.45 / 1.32 (3D-ConvLSTM) |
| BRA-S | Image v2 | paper | S2+ADM | 3/3 | 0.02 ± 0.05 | 1.19 ± 0.03 | -0.19 ± 0.16 | 0.76 ± 0.05 | 0.04 / 1.18 | 0.29 / 1.02 (AFF) | 0.45 / 0.52 (3D-LSTM) |
| BRA-W | Image v2 | paper | S2+ADM | 3/3 | 0.00 ± 0.01 | 1.57 ± 0.01 | 0.01 ± 0.05 | 0.79 ± 0.02 | 0.08 / 1.50 | 0.11 / 1.48 (AFF) | 0.14 / 0.74 (AFF) |
| GER-R | Image v2 | paper | S2+ADM | 3/3 | -0.09 ± 0.03 | 1.74 ± 0.02 | -0.25 ± 0.08 | 1.50 ± 0.05 | -0.05 / 1.71 | 0.31 / 1.38 (LSTM) | 0.58 / 0.87 (LSTM) |
| GER-W | Image v2 | paper | S2+ADM | 3/3 | 0.01 ± 0.02 | 2.93 ± 0.02 | -0.09 ± 0.06 | 1.98 ± 0.06 | -0.68 / 3.82 | 0.22 / 2.60 (3D-ConvLSTM) | 0.46 / 1.39 (3D-ConvLSTM) |
| URG-S | Image v2 | paper | S2+ADM | 3/3 | -0.06 ± 0.13 | 1.62 ± 0.10 | -0.23 ± 0.28 | 1.18 ± 0.14 | 0.33 / 1.29 | 0.32 / 1.30 (AFF) | 0.56 / 0.71 (3D-ConvLSTM) |
| ARG-C | Image v2 | strict | S2 | 3/3 | 0.12 ± 0.02 | 3.49 ± 0.04 | 0.18 ± 0.02 | 2.54 ± 0.03 | 0.39 / 2.90 | 0.48 / 2.68 (3D-ConvLSTM) | 0.61 / 1.75 (3D-ConvLSTM) |
| ARG-S | Image v2 | strict | S2 | 3/3 | 0.38 ± 0.01 | 1.20 ± 0.01 | 0.41 ± 0.03 | 0.93 ± 0.02 | 0.46 / 1.12 | 0.55 / 1.03 (3D-LSTM) | 0.66 / 0.70 (3D-LSTM) |
| ARG-W | Image v2 | strict | S2 | 3/3 | 0.08 ± 0.12 | 2.28 ± 0.15 | 0.13 ± 0.15 | 1.99 ± 0.17 | 0.61 / 1.49 | 0.68 / 1.35 (3D-LSTM) | 0.78 / 1.00 (3D-ConvLSTM) |
| BRA-C | Image v2 | strict | S2 | 3/3 | -0.17 ± 0.02 | 3.12 ± 0.02 | -0.53 ± 0.01 | 2.19 ± 0.01 | 0.15 / 2.66 | 0.21 / 2.56 (3D-LSTM) | 0.36 / 1.42 (3D-LSTM) |
| BRA-S | Image v2 | strict | S2 | 3/3 | -0.12 ± 0.01 | 1.27 ± 0.01 | -0.34 ± 0.06 | 0.80 ± 0.02 | 0.04 / 1.18 | 0.19 / 1.08 (3D-ConvLSTM) | 0.36 / 0.55 (3D-ConvLSTM) |
| BRA-W | Image v2 | strict | S2 | 3/3 | -0.05 ± 0.01 | 1.61 ± 0.01 | -0.18 ± 0.07 | 0.87 ± 0.02 | 0.08 / 1.50 | 0.08 / 1.50 (LSTM) | -0.05 / 0.82 (LSTM) |
| GER-R | Image v2 | strict | S2 | 3/3 | -0.03 ± 0.02 | 1.70 ± 0.01 | -0.18 ± 0.08 | 1.45 ± 0.05 | -0.05 / 1.71 | 0.21 / 1.49 (3D-ConvLSTM) | 0.43 / 1.02 (3D-ConvLSTM) |
| GER-W | Image v2 | strict | S2 | 3/3 | -0.21 ± 0.00 | 3.24 ± 0.01 | -0.49 ± 0.02 | 2.32 ± 0.02 | -0.68 / 3.82 | 0.15 / 2.71 (3D-ConvLSTM) | 0.23 / 1.67 (3D-ConvLSTM) |
| URG-S | Image v2 | strict | S2 | 3/3 | -0.10 ± 0.06 | 1.65 ± 0.04 | -0.28 ± 0.12 | 1.21 ± 0.06 | 0.33 / 1.29 | 0.35 / 1.27 (3D-LSTM) | 0.63 / 0.65 (3D-ConvLSTM) |
| ARG-C | Image v2 | strict | S2+ADM | 3/3 | 0.10 ± 0.04 | 3.53 ± 0.09 | 0.18 ± 0.05 | 2.53 ± 0.07 | 0.39 / 2.90 | 0.44 / 2.78 (LSTM) | 0.42 / 2.13 (LSTM) |
| ARG-S | Image v2 | strict | S2+ADM | 3/3 | 0.39 ± 0.02 | 1.19 ± 0.02 | 0.41 ± 0.01 | 0.93 ± 0.01 | 0.46 / 1.12 | 0.66 / 0.89 (AFF) | 0.77 / 0.59 (AFF) |
| ARG-W | Image v2 | strict | S2+ADM | 3/3 | 0.06 ± 0.05 | 2.31 ± 0.06 | 0.16 ± 0.06 | 1.95 ± 0.07 | 0.61 / 1.49 | 0.72 / 1.27 (AFF) | 0.81 / 0.93 (AFF) |
| BRA-C | Image v2 | strict | S2+ADM | 3/3 | -0.18 ± 0.07 | 3.14 ± 0.09 | -0.59 ± 0.17 | 2.23 ± 0.12 | 0.15 / 2.66 | 0.29 / 2.43 (AFF) | 0.45 / 1.32 (3D-ConvLSTM) |
| BRA-S | Image v2 | strict | S2+ADM | 3/3 | -0.09 ± 0.04 | 1.26 ± 0.02 | -0.21 ± 0.15 | 0.76 ± 0.05 | 0.04 / 1.18 | 0.29 / 1.02 (AFF) | 0.45 / 0.52 (3D-LSTM) |
| BRA-W | Image v2 | strict | S2+ADM | 3/3 | -0.05 ± 0.01 | 1.61 ± 0.01 | -0.13 ± 0.07 | 0.85 ± 0.02 | 0.08 / 1.50 | 0.11 / 1.48 (AFF) | 0.14 / 0.74 (AFF) |
| GER-R | Image v2 | strict | S2+ADM | 3/3 | -0.12 ± 0.07 | 1.77 ± 0.06 | -0.31 ± 0.06 | 1.54 ± 0.04 | -0.05 / 1.71 | 0.31 / 1.38 (LSTM) | 0.58 / 0.87 (LSTM) |
| GER-W | Image v2 | strict | S2+ADM | 3/3 | -0.23 ± 0.02 | 3.27 ± 0.02 | -0.49 ± 0.12 | 2.32 ± 0.09 | -0.68 / 3.82 | 0.22 / 2.60 (3D-ConvLSTM) | 0.46 / 1.39 (3D-ConvLSTM) |
| URG-S | Image v2 | strict | S2+ADM | 3/3 | -0.06 ± 0.06 | 1.62 ± 0.05 | -0.20 ± 0.14 | 1.17 ± 0.07 | 0.33 / 1.29 | 0.32 / 1.30 (AFF) | 0.56 / 0.71 (3D-ConvLSTM) |
