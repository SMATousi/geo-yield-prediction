# Point model results (before_full suite)

Generated 2026-10-02 00:09 UTC from W&B project `yieldsat-cvpr27` (suite `before_full`) by `yieldsat_results_tables.py`.
**2734 of 2766 planned runs finished.** Values are means ± std over finished folds × seeds.

- **Pixel**: every held-out cell counts once. **Field**: per field season, mean prediction vs mean target; field R² only over folds with ≥ 3 test field seasons.
- **RMSE** in t/ha. **Policy** `paper` = the paper's grouping; `strict` additionally removes training seasons sharing a physical field with the test fold.
- **Paper** numbers are fold means from Pathak et al. (CVPR 2026, `results/yieldsat/paper_benchmark.csv`); "best" is the best model per row and modality set.

Argentine LORO uses provinces (the paper's regions); the earlier farm-level Argentine LORO runs were retired and are not shown. The paper LSTM was re-run for CV10 only.

## CV10 (10-fold, grouped by field season)

| Pair | Model | Policy | Inputs | Runs done | Pixel R² | Pixel RMSE | Field R² | Field RMSE | Paper LSTM pixel R² / RMSE (S2) | Paper best pixel R² / RMSE | Paper best field R² / RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ARG-C | Ours (point) | paper | S2 | 30/30 | 0.52 ± 0.15 | 2.40 ± 0.38 | 0.69 ± 0.19 | 1.42 ± 0.49 | 0.59 / 2.40 | 0.65 / 2.20 (3D-ConvLSTM) | 0.84 / 1.13 (3D-ConvLSTM) |
| ARG-C | Paper LSTM (our re-run) | paper | S2 | 10/10 | 0.48 ± 0.21 | 2.47 ± 0.42 | 0.66 ± 0.21 | 1.43 ± 0.48 | 0.59 / 2.40 | 0.65 / 2.20 (3D-ConvLSTM) | 0.84 / 1.13 (3D-ConvLSTM) |
| ARG-S | Ours (point) | paper | S2 | 30/30 | 0.61 ± 0.09 | 0.93 ± 0.11 | 0.73 ± 0.10 | 0.61 ± 0.15 | 0.60 / 0.96 | 0.65 / 0.90 (3D-ConvLSTM) | 0.79 / 0.55 (3D-ConvLSTM) |
| ARG-S | Paper LSTM (our re-run) | paper | S2 | 10/10 | 0.56 ± 0.12 | 0.99 ± 0.14 | 0.69 ± 0.11 | 0.66 ± 0.16 | 0.60 / 0.96 | 0.65 / 0.90 (3D-ConvLSTM) | 0.79 / 0.55 (3D-ConvLSTM) |
| ARG-W | Ours (point) | paper | S2 | 30/30 | 0.72 ± 0.11 | 1.18 ± 0.22 | 0.85 ± 0.09 | 0.75 ± 0.25 | 0.74 / 1.21 | 0.79 / 1.10 (3D-ConvLSTM) | 0.92 / 0.62 (3D-ConvLSTM) |
| ARG-W | Paper LSTM (our re-run) | paper | S2 | 10/10 | 0.67 ± 0.15 | 1.28 ± 0.31 | 0.79 ± 0.19 | 0.88 ± 0.37 | 0.74 / 1.21 | 0.79 / 1.10 (3D-ConvLSTM) | 0.92 / 0.62 (3D-ConvLSTM) |
| BRA-C | Ours (point) | paper | S2 | 30/30 | 0.40 ± 0.09 | 2.16 ± 0.40 | 0.73 ± 0.14 | 0.82 ± 0.16 | 0.42 / 2.20 | 0.46 / 2.13 (3D-LSTM) | 0.82 / 0.74 (3D-ConvLSTM) |
| BRA-C | Paper LSTM (our re-run) | paper | S2 | 10/10 | 0.39 ± 0.07 | 2.20 ± 0.38 | 0.68 ± 0.12 | 0.91 ± 0.16 | 0.42 / 2.20 | 0.46 / 2.13 (3D-LSTM) | 0.82 / 0.74 (3D-ConvLSTM) |
| BRA-S | Ours (point) | paper | S2 | 30/30 | 0.32 ± 0.13 | 0.97 ± 0.05 | 0.61 ± 0.19 | 0.39 ± 0.06 | 0.34 / 0.98 | 0.39 / 0.94 (3D-ConvLSTM) | 0.76 / 0.34 (3D-ConvLSTM) |
| BRA-S | Paper LSTM (our re-run) | paper | S2 | 10/10 | 0.31 ± 0.10 | 0.99 ± 0.05 | 0.61 ± 0.12 | 0.41 ± 0.07 | 0.34 / 0.98 | 0.39 / 0.94 (3D-ConvLSTM) | 0.76 / 0.34 (3D-ConvLSTM) |
| BRA-W | Ours (point) | paper | S2 | 30/30 | 0.16 ± 0.06 | 1.43 ± 0.13 | 0.50 ± 0.22 | 0.54 ± 0.16 | 0.22 / 1.39 | 0.24 / 1.37 (3D-LSTM) | 0.73 / 0.42 (3D-ConvLSTM) |
| BRA-W | Paper LSTM (our re-run) | paper | S2 | 10/10 | 0.18 ± 0.06 | 1.40 ± 0.11 | 0.56 ± 0.18 | 0.50 ± 0.16 | 0.22 / 1.39 | 0.24 / 1.37 (3D-LSTM) | 0.73 / 0.42 (3D-ConvLSTM) |
| GER-R | Ours (point) | paper | S2 | 30/30 | 0.33 ± 0.11 | 1.34 ± 0.21 | 0.61 ± 0.16 | 0.78 ± 0.17 | 0.36 / 1.33 | 0.49 / 1.20 (3D-ConvLSTM) | 0.82 / 0.57 (3D-LSTM) |
| GER-R | Paper LSTM (our re-run) | paper | S2 | 10/10 | 0.33 ± 0.11 | 1.34 ± 0.17 | 0.64 ± 0.17 | 0.75 ± 0.17 | 0.36 / 1.33 | 0.49 / 1.20 (3D-ConvLSTM) | 0.82 / 0.57 (3D-LSTM) |
| GER-W | Ours (point) | paper | S2 | 30/30 | 0.30 ± 0.14 | 2.43 ± 0.34 | 0.56 ± 0.17 | 1.20 ± 0.22 | -0.32 / 3.38 | 0.34 / 2.40 (3D-ConvLSTM) | 0.65 / 1.12 (3D-ConvLSTM) |
| GER-W | Paper LSTM (our re-run) | paper | S2 | 10/10 | 0.23 ± 0.11 | 2.55 ± 0.28 | 0.36 ± 0.22 | 1.46 ± 0.21 | -0.32 / 3.38 | 0.34 / 2.40 (3D-ConvLSTM) | 0.65 / 1.12 (3D-ConvLSTM) |
| URG-S | Ours (point) | paper | S2 | 30/30 | 0.36 ± 0.05 | 1.26 ± 0.07 | 0.70 ± 0.07 | 0.58 ± 0.08 | 0.37 / 1.26 | 0.41 / 1.22 (3D-ConvLSTM) | 0.77 / 0.51 (3D-ConvLSTM) |
| URG-S | Paper LSTM (our re-run) | paper | S2 | 10/10 | 0.36 ± 0.05 | 1.25 ± 0.07 | 0.69 ± 0.07 | 0.59 ± 0.07 | 0.37 / 1.26 | 0.41 / 1.22 (3D-ConvLSTM) | 0.77 / 0.51 (3D-ConvLSTM) |
| ARG-C | Ours (point) | paper | S2+ADM | 30/30 | 0.54 ± 0.13 | 2.34 ± 0.27 | 0.70 ± 0.13 | 1.38 ± 0.34 | 0.59 / 2.40 | 0.70 / 2.03 (AFF) | 0.84 / 1.12 (AFF) |
| ARG-C | Paper LSTM (our re-run) | paper | S2+ADM | 10/10 | 0.24 ± 0.40 | 2.95 ± 0.63 | 0.40 ± 0.39 | 1.91 ± 0.51 | 0.59 / 2.40 | 0.70 / 2.03 (AFF) | 0.84 / 1.12 (AFF) |
| ARG-S | Ours (point) | paper | S2+ADM | 30/30 | 0.62 ± 0.06 | 0.92 ± 0.09 | 0.74 ± 0.10 | 0.60 ± 0.14 | 0.60 / 0.96 | 0.73 / 0.79 (AFF) | 0.84 / 0.49 (AFF) |
| ARG-S | Paper LSTM (our re-run) | paper | S2+ADM | 10/10 | 0.52 ± 0.08 | 1.04 ± 0.08 | 0.66 ± 0.10 | 0.69 ± 0.12 | 0.60 / 0.96 | 0.73 / 0.79 (AFF) | 0.84 / 0.49 (AFF) |
| ARG-W | Ours (point) | paper | S2+ADM | 30/30 | 0.71 ± 0.13 | 1.19 ± 0.25 | 0.83 ± 0.13 | 0.79 ± 0.30 | 0.74 / 1.21 | 0.84 / 0.96 (AFF) | 0.92 / 0.60 (AFF) |
| ARG-W | Paper LSTM (our re-run) | paper | S2+ADM | 10/10 | 0.64 ± 0.15 | 1.33 ± 0.27 | 0.73 ± 0.23 | 0.98 ± 0.38 | 0.74 / 1.21 | 0.84 / 0.96 (AFF) | 0.92 / 0.60 (AFF) |
| BRA-C | Ours (point) | paper | S2+ADM | 30/30 | 0.42 ± 0.10 | 2.14 ± 0.43 | 0.77 ± 0.09 | 0.78 ± 0.19 | 0.42 / 2.20 | 0.46 / 2.12 (AFF) | 0.84 / 0.70 (AFF) |
| BRA-C | Paper LSTM (our re-run) | paper | S2+ADM | 10/10 | 0.24 ± 0.16 | 2.42 ± 0.30 | 0.49 ± 0.33 | 1.09 ± 0.27 | 0.42 / 2.20 | 0.46 / 2.12 (AFF) | 0.84 / 0.70 (AFF) |
| BRA-S | Ours (point) | paper | S2+ADM | 30/30 | 0.35 ± 0.10 | 0.95 ± 0.05 | 0.68 ± 0.13 | 0.36 ± 0.07 | 0.34 / 0.98 | 0.44 / 0.90 (AFF) | 0.80 / 0.31 (AFF) |
| BRA-S | Paper LSTM (our re-run) | paper | S2+ADM | 10/10 | 0.21 ± 0.12 | 1.05 ± 0.07 | 0.49 ± 0.36 | 0.44 ± 0.08 | 0.34 / 0.98 | 0.44 / 0.90 (AFF) | 0.80 / 0.31 (AFF) |
| BRA-W | Ours (point) | paper | S2+ADM | 30/30 | 0.18 ± 0.07 | 1.41 ± 0.13 | 0.59 ± 0.22 | 0.47 ± 0.13 | 0.22 / 1.39 | 0.24 / 1.37 (AFF) | 0.72 / 0.42 (3D-ConvLSTM) |
| BRA-W | Paper LSTM (our re-run) | paper | S2+ADM | 10/10 | 0.04 ± 0.07 | 1.53 ± 0.14 | 0.37 ± 0.17 | 0.62 ± 0.18 | 0.22 / 1.39 | 0.24 / 1.37 (AFF) | 0.72 / 0.42 (3D-ConvLSTM) |
| GER-R | Ours (point) | paper | S2+ADM | 29/30 | 0.36 ± 0.12 | 1.31 ± 0.20 | 0.65 ± 0.21 | 0.74 ± 0.21 | 0.36 / 1.33 | 0.49 / 1.20 (AFF) | 0.81 / 0.59 (3D-LSTM) |
| GER-R | Paper LSTM (our re-run) | paper | S2+ADM | 10/10 | 0.21 ± 0.19 | 1.45 ± 0.26 | 0.44 ± 0.32 | 0.93 ± 0.28 | 0.36 / 1.33 | 0.49 / 1.20 (AFF) | 0.81 / 0.59 (3D-LSTM) |
| GER-W | Ours (point) | paper | S2+ADM | 30/30 | 0.35 ± 0.12 | 2.34 ± 0.31 | 0.61 ± 0.14 | 1.14 ± 0.19 | -0.32 / 3.38 | 0.44 / 2.20 (AFF) | 0.77 / 0.90 (MMGF) |
| GER-W | Paper LSTM (our re-run) | paper | S2+ADM | 10/10 | 0.24 ± 0.14 | 2.53 ± 0.32 | 0.48 ± 0.24 | 1.30 ± 0.27 | -0.32 / 3.38 | 0.44 / 2.20 (AFF) | 0.77 / 0.90 (MMGF) |
| URG-S | Ours (point) | paper | S2+ADM | 29/30 | 0.37 ± 0.04 | 1.25 ± 0.07 | 0.70 ± 0.06 | 0.58 ± 0.07 | 0.37 / 1.26 | 0.43 / 1.19 (AFF) | 0.81 / 0.46 (AFF) |
| URG-S | Paper LSTM (our re-run) | paper | S2+ADM | 10/10 | 0.30 ± 0.05 | 1.32 ± 0.05 | 0.61 ± 0.08 | 0.66 ± 0.09 | 0.37 / 1.26 | 0.43 / 1.19 (AFF) | 0.81 / 0.46 (AFF) |
| ARG-C | Ours (point) | strict | S2 | 30/30 | 0.54 ± 0.18 | 2.39 ± 0.41 | 0.67 ± 0.19 | 1.45 ± 0.39 | 0.59 / 2.40 | 0.65 / 2.20 (3D-ConvLSTM) | 0.84 / 1.13 (3D-ConvLSTM) |
| ARG-S | Ours (point) | strict | S2 | 30/30 | 0.61 ± 0.06 | 0.94 ± 0.07 | 0.73 ± 0.09 | 0.61 ± 0.11 | 0.60 / 0.96 | 0.65 / 0.90 (3D-ConvLSTM) | 0.79 / 0.55 (3D-ConvLSTM) |
| ARG-W | Ours (point) | strict | S2 | 30/30 | 0.72 ± 0.10 | 1.20 ± 0.24 | 0.85 ± 0.09 | 0.79 ± 0.28 | 0.74 / 1.21 | 0.79 / 1.10 (3D-ConvLSTM) | 0.92 / 0.62 (3D-ConvLSTM) |
| BRA-C | Ours (point) | strict | S2 | 30/30 | 0.33 ± 0.16 | 2.23 ± 0.44 | 0.32 ± 0.99 | 0.92 ± 0.25 | 0.42 / 2.20 | 0.46 / 2.13 (3D-LSTM) | 0.82 / 0.74 (3D-ConvLSTM) |
| BRA-S | Ours (point) | strict | S2 | 30/30 | 0.34 ± 0.11 | 0.97 ± 0.07 | 0.65 ± 0.14 | 0.40 ± 0.08 | 0.34 / 0.98 | 0.39 / 0.94 (3D-ConvLSTM) | 0.76 / 0.34 (3D-ConvLSTM) |
| BRA-W | Ours (point) | strict | S2 | 30/30 | 0.15 ± 0.08 | 1.42 ± 0.17 | 0.46 ± 0.29 | 0.48 ± 0.16 | 0.22 / 1.39 | 0.24 / 1.37 (3D-LSTM) | 0.73 / 0.42 (3D-ConvLSTM) |
| GER-R | Ours (point) | strict | S2 | 30/30 | 0.22 ± 0.27 | 1.40 ± 0.21 | 0.50 ± 0.47 | 0.83 ± 0.30 | 0.36 / 1.33 | 0.49 / 1.20 (3D-ConvLSTM) | 0.82 / 0.57 (3D-LSTM) |
| GER-W | Ours (point) | strict | S2 | 30/30 | 0.30 ± 0.14 | 2.40 ± 0.25 | 0.53 ± 0.23 | 1.18 ± 0.32 | -0.32 / 3.38 | 0.34 / 2.40 (3D-ConvLSTM) | 0.65 / 1.12 (3D-ConvLSTM) |
| URG-S | Ours (point) | strict | S2 | 30/30 | 0.35 ± 0.07 | 1.26 ± 0.11 | 0.70 ± 0.13 | 0.57 ± 0.15 | 0.37 / 1.26 | 0.41 / 1.22 (3D-ConvLSTM) | 0.77 / 0.51 (3D-ConvLSTM) |
| ARG-C | Ours (point) | strict | S2+ADM | 30/30 | 0.56 ± 0.16 | 2.35 ± 0.39 | 0.72 ± 0.17 | 1.34 ± 0.39 | 0.59 / 2.40 | 0.70 / 2.03 (AFF) | 0.84 / 1.12 (AFF) |
| ARG-S | Ours (point) | strict | S2+ADM | 30/30 | 0.62 ± 0.07 | 0.92 ± 0.09 | 0.76 ± 0.10 | 0.58 ± 0.12 | 0.60 / 0.96 | 0.73 / 0.79 (AFF) | 0.84 / 0.49 (AFF) |
| ARG-W | Ours (point) | strict | S2+ADM | 30/30 | 0.73 ± 0.11 | 1.18 ± 0.24 | 0.85 ± 0.09 | 0.78 ± 0.26 | 0.74 / 1.21 | 0.84 / 0.96 (AFF) | 0.92 / 0.60 (AFF) |
| BRA-C | Ours (point) | strict | S2+ADM | 30/30 | 0.34 ± 0.14 | 2.21 ± 0.43 | 0.50 ± 0.52 | 0.86 ± 0.22 | 0.42 / 2.20 | 0.46 / 2.12 (AFF) | 0.84 / 0.70 (AFF) |
| BRA-S | Ours (point) | strict | S2+ADM | 30/30 | 0.35 ± 0.10 | 0.96 ± 0.06 | 0.66 ± 0.15 | 0.39 ± 0.09 | 0.34 / 0.98 | 0.44 / 0.90 (AFF) | 0.80 / 0.31 (AFF) |
| BRA-W | Ours (point) | strict | S2+ADM | 30/30 | 0.14 ± 0.11 | 1.43 ± 0.19 | 0.38 ± 0.36 | 0.50 ± 0.15 | 0.22 / 1.39 | 0.24 / 1.37 (AFF) | 0.72 / 0.42 (3D-ConvLSTM) |
| GER-R | Ours (point) | strict | S2+ADM | 28/30 | 0.33 ± 0.14 | 1.29 ± 0.16 | 0.62 ± 0.26 | 0.74 ± 0.19 | 0.36 / 1.33 | 0.49 / 1.20 (AFF) | 0.81 / 0.59 (3D-LSTM) |
| GER-W | Ours (point) | strict | S2+ADM | 30/30 | 0.31 ± 0.15 | 2.37 ± 0.24 | 0.54 ± 0.24 | 1.15 ± 0.30 | -0.32 / 3.38 | 0.44 / 2.20 (AFF) | 0.77 / 0.90 (MMGF) |
| URG-S | Ours (point) | strict | S2+ADM | 30/30 | 0.37 ± 0.07 | 1.24 ± 0.12 | 0.71 ± 0.15 | 0.56 ± 0.16 | 0.37 / 1.26 | 0.43 / 1.19 (AFF) | 0.81 / 0.46 (AFF) |

## LORO, provinces (Argentina; the paper's regions)

| Pair | Model | Policy | Inputs | Runs done | Pixel R² | Pixel RMSE | Field R² | Field RMSE | Paper LSTM pixel R² / RMSE (S2) | Paper best pixel R² / RMSE | Paper best field R² / RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ARG-C | Ours (point) | paper | S2 | 18/18 | 0.04 ± 0.40 | 3.09 ± 0.47 | -0.66 ± 2.01 | 2.32 ± 0.44 | 0.44 / 2.79 | 0.54 / 2.52 (3D-ConvLSTM) | 0.68 / 1.59 (3D-ConvLSTM) |
| ARG-S | Ours (point) | paper | S2 | 24/24 | 0.38 ± 0.14 | 1.09 ± 0.14 | 0.47 ± 0.19 | 0.69 ± 0.23 | 0.55 / 1.03 | 0.59 / 0.98 (3D-LSTM) | 0.70 / 0.67 (3D-LSTM) |
| ARG-W | Ours (point) | paper | S2 | 17/18 | 0.10 ± 0.65 | 1.47 ± 0.38 | 0.19 ± 0.91 | 1.12 ± 0.38 | 0.61 / 1.48 | 0.71 / 1.30 (3D-LSTM) | 0.81 / 0.92 (3D-LSTM) |
| ARG-C | Ours (point) | paper | S2+ADM | 18/18 | 0.10 ± 0.48 | 2.94 ± 0.34 | -0.67 ± 2.42 | 2.19 ± 0.48 | 0.44 / 2.79 | 0.54 / 2.52 (3D-ConvLSTM) | 0.68 / 1.59 (3D-ConvLSTM) |
| ARG-S | Ours (point) | paper | S2+ADM | 24/24 | 0.37 ± 0.16 | 1.09 ± 0.17 | 0.43 ± 0.24 | 0.68 ± 0.23 | 0.55 / 1.03 | 0.65 / 0.90 (AFF) | 0.78 / 0.57 (AFF) |
| ARG-W | Ours (point) | paper | S2+ADM | 15/18 | 0.13 ± 0.57 | 1.42 ± 0.40 | 0.11 ± 0.92 | 1.03 ± 0.43 | 0.61 / 1.48 | 0.78 / 1.12 (AFF) | 0.87 / 0.78 (AFF) |
| ARG-C | Ours (point) | strict | S2 | 18/18 | 0.13 ± 0.37 | 2.93 ± 0.35 | -0.36 ± 1.72 | 2.07 ± 0.35 | 0.44 / 2.79 | 0.54 / 2.52 (3D-ConvLSTM) | 0.68 / 1.59 (3D-ConvLSTM) |
| ARG-S | Ours (point) | strict | S2 | 19/24 | 0.40 ± 0.14 | 1.08 ± 0.15 | 0.49 ± 0.20 | 0.70 ± 0.23 | 0.55 / 1.03 | 0.59 / 0.98 (3D-LSTM) | 0.70 / 0.67 (3D-LSTM) |
| ARG-W | Ours (point) | strict | S2 | 14/18 | -0.00 ± 0.75 | 1.41 ± 0.37 | 0.12 ± 1.08 | 1.06 ± 0.36 | 0.61 / 1.48 | 0.71 / 1.30 (3D-LSTM) | 0.81 / 0.92 (3D-LSTM) |
| ARG-C | Ours (point) | strict | S2+ADM | 18/18 | 0.09 ± 0.41 | 3.01 ± 0.44 | -0.60 ± 1.94 | 2.27 ± 0.34 | 0.44 / 2.79 | 0.54 / 2.52 (3D-ConvLSTM) | 0.68 / 1.59 (3D-ConvLSTM) |
| ARG-S | Ours (point) | strict | S2+ADM | 24/24 | 0.40 ± 0.16 | 1.07 ± 0.18 | 0.51 ± 0.23 | 0.65 ± 0.26 | 0.55 / 1.03 | 0.65 / 0.90 (AFF) | 0.78 / 0.57 (AFF) |
| ARG-W | Ours (point) | strict | S2+ADM | 12/18 | -0.03 ± 0.78 | 1.60 ± 0.32 | -0.04 ± 1.17 | 1.25 ± 0.37 | 0.61 / 1.48 | 0.78 / 1.12 (AFF) | 0.87 / 0.78 (AFF) |

## LORO, farm regions

| Pair | Model | Policy | Inputs | Runs done | Pixel R² | Pixel RMSE | Field R² | Field RMSE | Paper LSTM pixel R² / RMSE (S2) | Paper best pixel R² / RMSE | Paper best field R² / RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BRA-C | Ours (point) | paper | S2 | 21/21 | 0.17 ± 0.23 | 2.22 ± 0.62 | -0.74 ± 1.88 | 1.17 ± 0.63 | 0.27 / 2.46 | 0.34 / 2.33 (3D-LSTM) | 0.59 / 1.13 (3D-LSTM) |
| BRA-S | Ours (point) | paper | S2 | 27/27 | 0.01 ± 0.21 | 1.20 ± 0.35 | -1.58 ± 3.58 | 0.77 ± 0.32 | 0.16 / 1.10 | 0.22 / 1.06 (3D-LSTM) | 0.41 / 0.53 (3D-LSTM) |
| BRA-W | Ours (point) | paper | S2 | 18/18 | 0.10 ± 0.12 | 1.31 ± 0.32 | 0.11 ± 0.53 | 0.54 ± 0.25 | 0.11 / 1.48 | 0.20 / 1.41 (3D-LSTM) | 0.48 / 0.57 (3D-LSTM) |
| GER-R | Ours (point) | paper | S2 | 18/18 | -0.44 ± 0.85 | 1.64 ± 0.45 | -0.74 ± 1.13 | 1.24 ± 0.45 | -0.05 / 1.71 | 0.17 / 1.52 (3D-LSTM) | 0.26 / 1.16 (3D-LSTM) |
| GER-W | Ours (point) | paper | S2 | 18/18 | -0.28 ± 0.45 | 2.58 ± 0.70 | -1.10 ± 1.74 | 1.87 ± 0.82 | -0.78 / 3.93 | 0.10 / 2.79 (Transformer) | 0.14 / 1.76 (3D-ConvLSTM) |
| URG-S | Ours (point) | paper | S2 | 30/30 | 0.11 ± 0.24 | 1.18 ± 0.26 | 0.04 ± 0.84 | 0.60 ± 0.22 | 0.34 / 1.28 | 0.36 / 1.26 (3D-ConvLSTM) | 0.68 / 0.61 (3D-ConvLSTM) |
| BRA-C | Ours (point) | paper | S2+ADM | 21/21 | 0.19 ± 0.18 | 2.17 ± 0.51 | -0.28 ± 0.95 | 1.11 ± 0.52 | 0.27 / 2.46 | 0.37 / 2.29 (3D-LSTM) | 0.65 / 1.05 (3D-LSTM) |
| BRA-S | Ours (point) | paper | S2+ADM | 27/27 | 0.04 ± 0.23 | 1.19 ± 0.38 | -1.92 ± 4.52 | 0.74 ± 0.30 | 0.16 / 1.10 | 0.35 / 0.97 (AFF) | 0.63 / 0.42 (AFF) |
| BRA-W | Ours (point) | paper | S2+ADM | 18/18 | 0.08 ± 0.13 | 1.32 ± 0.29 | 0.08 ± 0.28 | 0.57 ± 0.21 | 0.11 / 1.48 | 0.18 / 1.42 (AFF) | 0.52 / 0.55 (AFF) |
| GER-R | Ours (point) | paper | S2+ADM | 18/18 | -0.54 ± 0.77 | 1.71 ± 0.51 | -1.11 ± 1.32 | 1.37 ± 0.53 | -0.05 / 1.71 | 0.15 / 1.55 (AFF) | 0.25 / 1.16 (3D-ConvLSTM) |
| GER-W | Ours (point) | paper | S2+ADM | 18/18 | -0.20 ± 0.42 | 2.54 ± 0.85 | -0.93 ± 1.33 | 1.87 ± 0.95 | -0.78 / 3.93 | 0.07 / 2.84 (3D-ConvLSTM) | 0.14 / 1.76 (3D-ConvLSTM) |
| URG-S | Ours (point) | paper | S2+ADM | 26/30 | 0.18 ± 0.16 | 1.15 ± 0.27 | 0.28 ± 0.41 | 0.58 ± 0.26 | 0.34 / 1.28 | 0.36 / 1.26 (3D-ConvLSTM) | 0.68 / 0.61 (3D-ConvLSTM) |
| BRA-C | Ours (point) | strict | S2 | 18/18 | 0.06 ± 0.17 | 2.35 ± 0.54 | -0.85 ± 1.23 | 1.36 ± 0.42 | 0.27 / 2.46 | 0.34 / 2.33 (3D-LSTM) | 0.59 / 1.13 (3D-LSTM) |
| BRA-S | Ours (point) | strict | S2 | 24/24 | -0.02 ± 0.27 | 1.23 ± 0.35 | -1.78 ± 3.08 | 0.76 ± 0.21 | 0.16 / 1.10 | 0.22 / 1.06 (3D-LSTM) | 0.41 / 0.53 (3D-LSTM) |
| BRA-W | Ours (point) | strict | S2 | 15/15 | 0.11 ± 0.10 | 1.31 ± 0.34 | 0.07 ± 0.36 | 0.59 ± 0.23 | 0.11 / 1.48 | 0.20 / 1.41 (3D-LSTM) | 0.48 / 0.57 (3D-LSTM) |
| GER-R | Ours (point) | strict | S2 | 18/18 | -0.64 ± 1.01 | 1.71 ± 0.39 | -1.01 ± 1.27 | 1.31 ± 0.41 | -0.05 / 1.71 | 0.17 / 1.52 (3D-LSTM) | 0.26 / 1.16 (3D-LSTM) |
| GER-W | Ours (point) | strict | S2 | 18/18 | -0.20 ± 0.34 | 2.52 ± 0.69 | -0.92 ± 1.48 | 1.82 ± 0.76 | -0.78 / 3.93 | 0.10 / 2.79 (Transformer) | 0.14 / 1.76 (3D-ConvLSTM) |
| URG-S | Ours (point) | strict | S2 | 30/30 | 0.12 ± 0.23 | 1.17 ± 0.27 | 0.09 ± 0.88 | 0.58 ± 0.26 | 0.34 / 1.28 | 0.36 / 1.26 (3D-ConvLSTM) | 0.68 / 0.61 (3D-ConvLSTM) |
| BRA-C | Ours (point) | strict | S2+ADM | 18/18 | 0.06 ± 0.17 | 2.35 ± 0.52 | -0.92 ± 1.68 | 1.37 ± 0.43 | 0.27 / 2.46 | 0.37 / 2.29 (3D-LSTM) | 0.65 / 1.05 (3D-LSTM) |
| BRA-S | Ours (point) | strict | S2+ADM | 24/24 | 0.08 ± 0.11 | 1.18 ± 0.31 | -0.90 ± 1.89 | 0.66 ± 0.13 | 0.16 / 1.10 | 0.35 / 0.97 (AFF) | 0.63 / 0.42 (AFF) |
| BRA-W | Ours (point) | strict | S2+ADM | 15/15 | 0.10 ± 0.13 | 1.33 ± 0.36 | -0.00 ± 0.36 | 0.58 ± 0.29 | 0.11 / 1.48 | 0.18 / 1.42 (AFF) | 0.52 / 0.55 (AFF) |
| GER-R | Ours (point) | strict | S2+ADM | 18/18 | -0.42 ± 0.74 | 1.64 ± 0.46 | -0.88 ± 1.24 | 1.28 ± 0.46 | -0.05 / 1.71 | 0.15 / 1.55 (AFF) | 0.25 / 1.16 (3D-ConvLSTM) |
| GER-W | Ours (point) | strict | S2+ADM | 17/18 | -0.16 ± 0.32 | 2.49 ± 0.75 | -0.81 ± 1.05 | 1.81 ± 0.78 | -0.78 / 3.93 | 0.07 / 2.84 (3D-ConvLSTM) | 0.14 / 1.76 (3D-ConvLSTM) |
| URG-S | Ours (point) | strict | S2+ADM | 28/30 | 0.18 ± 0.15 | 1.12 ± 0.28 | 0.34 ± 0.44 | 0.53 ± 0.28 | 0.34 / 1.28 | 0.36 / 1.26 (3D-ConvLSTM) | 0.68 / 0.61 (3D-ConvLSTM) |

## LOYO (leave one year out)

| Pair | Model | Policy | Inputs | Runs done | Pixel R² | Pixel RMSE | Field R² | Field RMSE | Paper LSTM pixel R² / RMSE (S2) | Paper best pixel R² / RMSE | Paper best field R² / RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ARG-C | Ours (point) | paper | S2 | 24/24 | 0.12 ± 0.36 | 3.01 ± 0.35 | 0.02 ± 0.78 | 2.14 ± 0.63 | 0.39 / 2.90 | 0.48 / 2.68 (3D-ConvLSTM) | 0.61 / 1.75 (3D-ConvLSTM) |
| ARG-S | Ours (point) | paper | S2 | 24/24 | 0.26 ± 0.23 | 1.10 ± 0.18 | 0.17 ± 0.40 | 0.79 ± 0.19 | 0.46 / 1.12 | 0.55 / 1.03 (3D-LSTM) | 0.66 / 0.70 (3D-LSTM) |
| ARG-W | Ours (point) | paper | S2 | 21/21 | -0.24 ± 0.94 | 1.47 ± 0.42 | -0.68 ± 1.98 | 1.11 ± 0.44 | 0.61 / 1.49 | 0.68 / 1.35 (3D-LSTM) | 0.78 / 1.00 (3D-ConvLSTM) |
| BRA-C | Ours (point) | paper | S2 | 18/18 | -0.13 ± 0.42 | 2.73 ± 0.83 | -1.40 ± 2.78 | 1.72 ± 0.75 | 0.15 / 2.66 | 0.21 / 2.56 (3D-LSTM) | 0.36 / 1.42 (3D-LSTM) |
| BRA-S | Ours (point) | paper | S2 | 24/24 | -0.09 ± 0.27 | 1.18 ± 0.16 | -0.81 ± 1.03 | 0.66 ± 0.26 | 0.04 / 1.18 | 0.19 / 1.08 (3D-ConvLSTM) | 0.36 / 0.55 (3D-ConvLSTM) |
| BRA-W | Ours (point) | paper | S2 | 21/21 | -0.13 ± 0.32 | 1.54 ± 0.45 | -1.33 ± 3.04 | 0.81 ± 0.48 | 0.08 / 1.50 | 0.08 / 1.50 (LSTM) | -0.05 / 0.82 (LSTM) |
| GER-R | Ours (point) | paper | S2 | 21/21 | -0.40 ± 0.69 | 1.69 ± 0.44 | -6.70 ± 19.51 | 1.27 ± 0.43 | -0.05 / 1.71 | 0.21 / 1.49 (3D-ConvLSTM) | 0.43 / 1.02 (3D-ConvLSTM) |
| GER-W | Ours (point) | paper | S2 | 21/21 | -0.08 ± 0.47 | 2.78 ± 0.82 | -0.62 ± 1.30 | 1.92 ± 0.69 | -0.68 / 3.82 | 0.15 / 2.71 (3D-ConvLSTM) | 0.23 / 1.67 (3D-ConvLSTM) |
| URG-S | Ours (point) | paper | S2 | 15/15 | 0.07 ± 0.10 | 1.26 ± 0.15 | 0.19 ± 0.41 | 0.62 ± 0.22 | 0.33 / 1.29 | 0.35 / 1.27 (3D-LSTM) | 0.63 / 0.65 (3D-ConvLSTM) |
| ARG-C | Ours (point) | paper | S2+ADM | 24/24 | 0.09 ± 0.39 | 3.07 ± 0.51 | -0.07 ± 0.90 | 2.25 ± 0.85 | 0.39 / 2.90 | 0.44 / 2.78 (LSTM) | 0.42 / 2.13 (LSTM) |
| ARG-S | Ours (point) | paper | S2+ADM | 24/24 | 0.25 ± 0.23 | 1.10 ± 0.15 | 0.15 ± 0.44 | 0.78 ± 0.15 | 0.46 / 1.12 | 0.66 / 0.89 (AFF) | 0.77 / 0.59 (AFF) |
| ARG-W | Ours (point) | paper | S2+ADM | 21/21 | -0.79 ± 1.71 | 1.67 ± 0.56 | -1.90 ± 3.55 | 1.37 ± 0.65 | 0.61 / 1.49 | 0.72 / 1.27 (AFF) | 0.81 / 0.93 (AFF) |
| BRA-C | Ours (point) | paper | S2+ADM | 18/18 | -0.03 ± 0.18 | 2.64 ± 0.76 | -0.55 ± 0.73 | 1.59 ± 0.63 | 0.15 / 2.66 | 0.29 / 2.43 (AFF) | 0.45 / 1.32 (3D-ConvLSTM) |
| BRA-S | Ours (point) | paper | S2+ADM | 24/24 | -0.02 ± 0.26 | 1.14 ± 0.17 | -0.61 ± 1.08 | 0.61 ± 0.25 | 0.04 / 1.18 | 0.29 / 1.02 (AFF) | 0.45 / 0.52 (3D-LSTM) |
| BRA-W | Ours (point) | paper | S2+ADM | 21/21 | -0.11 ± 0.35 | 1.53 ± 0.46 | -1.28 ± 3.22 | 0.82 ± 0.50 | 0.08 / 1.50 | 0.11 / 1.48 (AFF) | 0.14 / 0.74 (AFF) |
| GER-R | Ours (point) | paper | S2+ADM | 21/21 | 0.08 ± 0.25 | 1.42 ± 0.43 | -1.59 ± 4.34 | 0.98 ± 0.39 | -0.05 / 1.71 | 0.31 / 1.38 (LSTM) | 0.58 / 0.87 (LSTM) |
| GER-W | Ours (point) | paper | S2+ADM | 21/21 | -0.02 ± 0.36 | 2.73 ± 0.87 | -0.53 ± 0.97 | 1.92 ± 0.76 | -0.68 / 3.82 | 0.22 / 2.60 (3D-ConvLSTM) | 0.46 / 1.39 (3D-ConvLSTM) |
| URG-S | Ours (point) | paper | S2+ADM | 15/15 | 0.06 ± 0.10 | 1.27 ± 0.15 | 0.14 ± 0.42 | 0.64 ± 0.22 | 0.33 / 1.29 | 0.32 / 1.30 (AFF) | 0.56 / 0.71 (3D-ConvLSTM) |
| ARG-C | Ours (point) | strict | S2 | 24/24 | 0.09 ± 0.29 | 3.13 ± 0.57 | 0.03 ± 0.68 | 2.25 ± 0.83 | 0.39 / 2.90 | 0.48 / 2.68 (3D-ConvLSTM) | 0.61 / 1.75 (3D-ConvLSTM) |
| ARG-S | Ours (point) | strict | S2 | 24/24 | 0.19 ± 0.32 | 1.13 ± 0.20 | 0.04 ± 0.54 | 0.82 ± 0.21 | 0.46 / 1.12 | 0.55 / 1.03 (3D-LSTM) | 0.66 / 0.70 (3D-LSTM) |
| ARG-W | Ours (point) | strict | S2 | 21/21 | -0.86 ± 2.20 | 1.66 ± 0.56 | -2.46 ± 6.09 | 1.32 ± 0.58 | 0.61 / 1.49 | 0.68 / 1.35 (3D-LSTM) | 0.78 / 1.00 (3D-ConvLSTM) |
| BRA-C | Ours (point) | strict | S2 | 18/18 | -0.30 ± 0.30 | 2.95 ± 0.83 | -1.64 ± 1.79 | 1.97 ± 0.64 | 0.15 / 2.66 | 0.21 / 2.56 (3D-LSTM) | 0.36 / 1.42 (3D-LSTM) |
| BRA-S | Ours (point) | strict | S2 | 23/24 | -0.27 ± 0.41 | 1.26 ± 0.18 | -1.58 ± 1.49 | 0.77 ± 0.25 | 0.04 / 1.18 | 0.19 / 1.08 (3D-ConvLSTM) | 0.36 / 0.55 (3D-ConvLSTM) |
| BRA-W | Ours (point) | strict | S2 | 21/21 | -0.19 ± 0.41 | 1.56 ± 0.39 | -2.31 ± 4.12 | 0.90 ± 0.37 | 0.08 / 1.50 | 0.08 / 1.50 (LSTM) | -0.05 / 0.82 (LSTM) |
| GER-R | Ours (point) | strict | S2 | 21/21 | -0.14 ± 0.34 | 1.57 ± 0.43 | -5.54 ± 14.75 | 1.19 ± 0.43 | -0.05 / 1.71 | 0.21 / 1.49 (3D-ConvLSTM) | 0.43 / 1.02 (3D-ConvLSTM) |
| GER-W | Ours (point) | strict | S2 | 21/21 | -0.14 ± 0.43 | 2.87 ± 0.85 | -0.75 ± 1.17 | 2.02 ± 0.74 | -0.68 / 3.82 | 0.15 / 2.71 (3D-ConvLSTM) | 0.23 / 1.67 (3D-ConvLSTM) |
| URG-S | Ours (point) | strict | S2 | 15/15 | 0.07 ± 0.10 | 1.26 ± 0.15 | 0.21 ± 0.44 | 0.61 ± 0.21 | 0.33 / 1.29 | 0.35 / 1.27 (3D-LSTM) | 0.63 / 0.65 (3D-ConvLSTM) |
| ARG-C | Ours (point) | strict | S2+ADM | 24/24 | 0.14 ± 0.32 | 3.03 ± 0.46 | 0.03 ± 0.81 | 2.21 ± 0.79 | 0.39 / 2.90 | 0.44 / 2.78 (LSTM) | 0.42 / 2.13 (LSTM) |
| ARG-S | Ours (point) | strict | S2+ADM | 24/24 | 0.22 ± 0.29 | 1.12 ± 0.16 | 0.06 ± 0.57 | 0.81 ± 0.17 | 0.46 / 1.12 | 0.66 / 0.89 (AFF) | 0.77 / 0.59 (AFF) |
| ARG-W | Ours (point) | strict | S2+ADM | 21/21 | -0.57 ± 1.21 | 1.66 ± 0.44 | -1.11 ± 2.61 | 1.29 ± 0.52 | 0.61 / 1.49 | 0.72 / 1.27 (AFF) | 0.81 / 0.93 (AFF) |
| BRA-C | Ours (point) | strict | S2+ADM | 18/18 | -0.58 ± 0.82 | 3.12 ± 0.71 | -2.47 ± 4.89 | 2.04 ± 0.69 | 0.15 / 2.66 | 0.29 / 2.43 (AFF) | 0.45 / 1.32 (3D-ConvLSTM) |
| BRA-S | Ours (point) | strict | S2+ADM | 24/24 | -0.19 ± 0.30 | 1.23 ± 0.15 | -1.53 ± 2.00 | 0.74 ± 0.21 | 0.04 / 1.18 | 0.29 / 1.02 (AFF) | 0.45 / 0.52 (3D-LSTM) |
| BRA-W | Ours (point) | strict | S2+ADM | 21/21 | -0.12 ± 0.23 | 1.54 ± 0.44 | -0.87 ± 1.06 | 0.85 ± 0.41 | 0.08 / 1.50 | 0.11 / 1.48 (AFF) | 0.14 / 0.74 (AFF) |
| GER-R | Ours (point) | strict | S2+ADM | 20/21 | -0.23 ± 1.05 | 1.53 ± 0.31 | -11.06 ± 34.55 | 1.13 ± 0.24 | -0.05 / 1.71 | 0.31 / 1.38 (LSTM) | 0.58 / 0.87 (LSTM) |
| GER-W | Ours (point) | strict | S2+ADM | 21/21 | -0.09 ± 0.49 | 2.80 ± 0.90 | -0.68 ± 1.33 | 1.95 ± 0.83 | -0.68 / 3.82 | 0.22 / 2.60 (3D-ConvLSTM) | 0.46 / 1.39 (3D-ConvLSTM) |
| URG-S | Ours (point) | strict | S2+ADM | 15/15 | 0.07 ± 0.12 | 1.26 ± 0.16 | 0.12 ± 0.55 | 0.64 ± 0.23 | 0.33 / 1.29 | 0.32 / 1.30 (AFF) | 0.56 / 0.71 (3D-ConvLSTM) |
