# Point model results (before_full suite)

Generated 2026-10-02 03:26 UTC by `yieldsat_results_tables.py` from pooled out-of-fold predictions (cluster results, before_full).
**342 of 342 experiments (pair × protocol × policy × inputs × seed) complete.**

- **Metric = the paper's computation:** for each experiment, the held-out predictions of all folds are pooled (each cell is held out exactly once) and R²/RMSE are computed once. Values are mean ± std of these pooled scores over seeds; a seed counts only when all its folds are finished.
- **Pixel**: every held-out 10 m cell. **Field**: per field season, mean prediction vs mean target. R² = 1 − SSE/SST; RMSE in t/ha.
- **Policy** `paper` = the paper's grouping; `strict` additionally removes training seasons that share a physical field with the test fold.
- **Paper** numbers are from Pathak et al. (CVPR 2026, `results/yieldsat/paper_benchmark.csv`); "best" is the best model per row and modality set. Why pooled: [negative_r2_investigation.md](negative_r2_investigation.md).

Argentine LORO uses provinces (the paper's regions); the earlier farm-level Argentine LORO runs were retired and are not shown. The paper LSTM was re-run for CV10 only.

## CV10 (10-fold, grouped by field season)

| Pair | Model | Policy | Inputs | Seeds complete | Pixel R² | Pixel RMSE | Field R² | Field RMSE | Paper LSTM pixel R² / RMSE (S2) | Paper best pixel R² / RMSE | Paper best field R² / RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ARG-C | Ours (point) | paper | S2 | 3/3 | 0.58 ± 0.03 | 2.42 ± 0.09 | 0.71 ± 0.06 | 1.49 ± 0.17 | 0.59 / 2.40 | 0.65 / 2.20 (3D-ConvLSTM) | 0.84 / 1.13 (3D-ConvLSTM) |
| ARG-C | Paper LSTM (our re-run) | paper | S2 | 1/1 | 0.55 | 2.49 | 0.71 | 1.51 | 0.59 / 2.40 | 0.65 / 2.20 (3D-ConvLSTM) | 0.84 / 1.13 (3D-ConvLSTM) |
| ARG-S | Ours (point) | paper | S2 | 3/3 | 0.62 ± 0.01 | 0.94 ± 0.02 | 0.73 ± 0.02 | 0.62 ± 0.02 | 0.60 / 0.96 | 0.65 / 0.90 (3D-ConvLSTM) | 0.79 / 0.55 (3D-ConvLSTM) |
| ARG-S | Paper LSTM (our re-run) | paper | S2 | 1/1 | 0.58 | 0.99 | 0.69 | 0.67 | 0.60 / 0.96 | 0.65 / 0.90 (3D-ConvLSTM) | 0.79 / 0.55 (3D-ConvLSTM) |
| ARG-W | Ours (point) | paper | S2 | 3/3 | 0.75 ± 0.01 | 1.20 ± 0.01 | 0.86 ± 0.01 | 0.79 ± 0.03 | 0.74 / 1.21 | 0.79 / 1.10 (3D-ConvLSTM) | 0.92 / 0.62 (3D-ConvLSTM) |
| ARG-W | Paper LSTM (our re-run) | paper | S2 | 1/1 | 0.70 | 1.32 | 0.80 | 0.95 | 0.74 / 1.21 | 0.79 / 1.10 (3D-ConvLSTM) | 0.92 / 0.62 (3D-ConvLSTM) |
| BRA-C | Ours (point) | paper | S2 | 3/3 | 0.42 ± 0.02 | 2.19 ± 0.04 | 0.78 ± 0.04 | 0.83 ± 0.08 | 0.42 / 2.20 | 0.46 / 2.13 (3D-LSTM) | 0.82 / 0.74 (3D-ConvLSTM) |
| BRA-C | Paper LSTM (our re-run) | paper | S2 | 1/1 | 0.41 | 2.22 | 0.73 | 0.92 | 0.42 / 2.20 | 0.46 / 2.13 (3D-LSTM) | 0.82 / 0.74 (3D-ConvLSTM) |
| BRA-S | Ours (point) | paper | S2 | 3/3 | 0.35 ± 0.01 | 0.97 ± 0.01 | 0.67 ± 0.02 | 0.40 ± 0.01 | 0.34 / 0.98 | 0.39 / 0.94 (3D-ConvLSTM) | 0.76 / 0.34 (3D-ConvLSTM) |
| BRA-S | Paper LSTM (our re-run) | paper | S2 | 1/1 | 0.32 | 0.99 | 0.65 | 0.41 | 0.34 / 0.98 | 0.39 / 0.94 (3D-ConvLSTM) | 0.76 / 0.34 (3D-ConvLSTM) |
| BRA-W | Ours (point) | paper | S2 | 3/3 | 0.16 ± 0.02 | 1.44 ± 0.01 | 0.51 ± 0.05 | 0.56 ± 0.03 | 0.22 / 1.39 | 0.24 / 1.37 (3D-LSTM) | 0.73 / 0.42 (3D-ConvLSTM) |
| BRA-W | Paper LSTM (our re-run) | paper | S2 | 1/1 | 0.19 | 1.41 | 0.57 | 0.52 | 0.22 / 1.39 | 0.24 / 1.37 (3D-LSTM) | 0.73 / 0.42 (3D-ConvLSTM) |
| GER-R | Ours (point) | paper | S2 | 3/3 | 0.34 ± 0.02 | 1.36 ± 0.02 | 0.64 ± 0.02 | 0.81 ± 0.02 | 0.36 / 1.33 | 0.49 / 1.20 (3D-ConvLSTM) | 0.82 / 0.57 (3D-LSTM) |
| GER-R | Paper LSTM (our re-run) | paper | S2 | 1/1 | 0.35 | 1.35 | 0.67 | 0.77 | 0.36 / 1.33 | 0.49 / 1.20 (3D-ConvLSTM) | 0.82 / 0.57 (3D-LSTM) |
| GER-W | Ours (point) | paper | S2 | 3/3 | 0.31 ± 0.01 | 2.45 ± 0.01 | 0.58 ± 0.02 | 1.23 ± 0.03 | -0.32 / 3.38 | 0.34 / 2.40 (3D-ConvLSTM) | 0.65 / 1.12 (3D-ConvLSTM) |
| GER-W | Paper LSTM (our re-run) | paper | S2 | 1/1 | 0.24 | 2.57 | 0.40 | 1.48 | -0.32 / 3.38 | 0.34 / 2.40 (3D-ConvLSTM) | 0.65 / 1.12 (3D-ConvLSTM) |
| URG-S | Ours (point) | paper | S2 | 3/3 | 0.36 ± 0.01 | 1.26 ± 0.01 | 0.70 ± 0.01 | 0.59 ± 0.01 | 0.37 / 1.26 | 0.41 / 1.22 (3D-ConvLSTM) | 0.77 / 0.51 (3D-ConvLSTM) |
| URG-S | Paper LSTM (our re-run) | paper | S2 | 1/1 | 0.37 | 1.25 | 0.70 | 0.59 | 0.37 / 1.26 | 0.41 / 1.22 (3D-ConvLSTM) | 0.77 / 0.51 (3D-ConvLSTM) |
| ARG-C | Ours (point) | paper | S2+ADM | 3/3 | 0.60 ± 0.03 | 2.35 ± 0.08 | 0.74 ± 0.02 | 1.42 ± 0.05 | 0.59 / 2.40 | 0.70 / 2.03 (AFF) | 0.84 / 1.12 (AFF) |
| ARG-C | Paper LSTM (our re-run) | paper | S2+ADM | 1/1 | 0.35 | 3.00 | 0.50 | 1.98 | 0.59 / 2.40 | 0.70 / 2.03 (AFF) | 0.84 / 1.12 (AFF) |
| ARG-S | Ours (point) | paper | S2+ADM | 3/3 | 0.64 ± 0.00 | 0.92 ± 0.00 | 0.74 ± 0.01 | 0.61 ± 0.01 | 0.60 / 0.96 | 0.73 / 0.79 (AFF) | 0.84 / 0.49 (AFF) |
| ARG-S | Paper LSTM (our re-run) | paper | S2+ADM | 1/1 | 0.54 | 1.04 | 0.66 | 0.70 | 0.60 / 0.96 | 0.73 / 0.79 (AFF) | 0.84 / 0.49 (AFF) |
| ARG-W | Ours (point) | paper | S2+ADM | 3/3 | 0.74 ± 0.00 | 1.22 ± 0.01 | 0.84 ± 0.01 | 0.84 ± 0.02 | 0.74 / 1.21 | 0.84 / 0.96 (AFF) | 0.92 / 0.60 (AFF) |
| ARG-W | Paper LSTM (our re-run) | paper | S2+ADM | 1/1 | 0.68 | 1.35 | 0.76 | 1.04 | 0.74 / 1.21 | 0.84 / 0.96 (AFF) | 0.92 / 0.60 (AFF) |
| BRA-C | Ours (point) | paper | S2+ADM | 3/3 | 0.43 ± 0.02 | 2.18 ± 0.03 | 0.80 ± 0.00 | 0.80 ± 0.01 | 0.42 / 2.20 | 0.46 / 2.12 (AFF) | 0.84 / 0.70 (AFF) |
| BRA-C | Paper LSTM (our re-run) | paper | S2+ADM | 1/1 | 0.29 | 2.43 | 0.60 | 1.12 | 0.42 / 2.20 | 0.46 / 2.12 (AFF) | 0.84 / 0.70 (AFF) |
| BRA-S | Ours (point) | paper | S2+ADM | 3/3 | 0.37 ± 0.01 | 0.95 ± 0.01 | 0.72 ± 0.01 | 0.37 ± 0.01 | 0.34 / 0.98 | 0.44 / 0.90 (AFF) | 0.80 / 0.31 (AFF) |
| BRA-S | Paper LSTM (our re-run) | paper | S2+ADM | 1/1 | 0.23 | 1.06 | 0.58 | 0.45 | 0.34 / 0.98 | 0.44 / 0.90 (AFF) | 0.80 / 0.31 (AFF) |
| BRA-W | Ours (point) | paper | S2+ADM | 3/3 | 0.19 ± 0.00 | 1.41 ± 0.00 | 0.62 ± 0.04 | 0.49 ± 0.02 | 0.22 / 1.39 | 0.24 / 1.37 (AFF) | 0.72 / 0.42 (3D-ConvLSTM) |
| BRA-W | Paper LSTM (our re-run) | paper | S2+ADM | 1/1 | 0.05 | 1.53 | 0.37 | 0.63 | 0.22 / 1.39 | 0.24 / 1.37 (AFF) | 0.72 / 0.42 (3D-ConvLSTM) |
| GER-R | Ours (point) | paper | S2+ADM | 3/3 | 0.38 ± 0.01 | 1.32 ± 0.01 | 0.67 ± 0.02 | 0.77 ± 0.03 | 0.36 / 1.33 | 0.49 / 1.20 (AFF) | 0.81 / 0.59 (3D-LSTM) |
| GER-R | Paper LSTM (our re-run) | paper | S2+ADM | 1/1 | 0.22 | 1.48 | 0.47 | 0.98 | 0.36 / 1.33 | 0.49 / 1.20 (AFF) | 0.81 / 0.59 (3D-LSTM) |
| GER-W | Ours (point) | paper | S2+ADM | 3/3 | 0.36 ± 0.01 | 2.36 ± 0.02 | 0.63 ± 0.03 | 1.16 ± 0.05 | -0.32 / 3.38 | 0.44 / 2.20 (AFF) | 0.77 / 0.90 (MMGF) |
| GER-W | Paper LSTM (our re-run) | paper | S2+ADM | 1/1 | 0.25 | 2.55 | 0.50 | 1.34 | -0.32 / 3.38 | 0.44 / 2.20 (AFF) | 0.77 / 0.90 (MMGF) |
| URG-S | Ours (point) | paper | S2+ADM | 3/3 | 0.38 ± 0.01 | 1.25 ± 0.01 | 0.71 ± 0.00 | 0.58 ± 0.00 | 0.37 / 1.26 | 0.43 / 1.19 (AFF) | 0.81 / 0.46 (AFF) |
| URG-S | Paper LSTM (our re-run) | paper | S2+ADM | 1/1 | 0.30 | 1.32 | 0.62 | 0.66 | 0.37 / 1.26 | 0.43 / 1.19 (AFF) | 0.81 / 0.46 (AFF) |
| ARG-C | Ours (point) | strict | S2 | 3/3 | 0.57 ± 0.03 | 2.43 ± 0.08 | 0.72 ± 0.01 | 1.50 ± 0.04 | 0.59 / 2.40 | 0.65 / 2.20 (3D-ConvLSTM) | 0.84 / 1.13 (3D-ConvLSTM) |
| ARG-S | Ours (point) | strict | S2 | 3/3 | 0.62 ± 0.00 | 0.94 ± 0.01 | 0.73 ± 0.01 | 0.62 ± 0.01 | 0.60 / 0.96 | 0.65 / 0.90 (3D-ConvLSTM) | 0.79 / 0.55 (3D-ConvLSTM) |
| ARG-W | Ours (point) | strict | S2 | 3/3 | 0.74 ± 0.02 | 1.22 ± 0.04 | 0.84 ± 0.01 | 0.85 ± 0.04 | 0.74 / 1.21 | 0.79 / 1.10 (3D-ConvLSTM) | 0.92 / 0.62 (3D-ConvLSTM) |
| BRA-C | Ours (point) | strict | S2 | 3/3 | 0.40 ± 0.01 | 2.23 ± 0.03 | 0.72 ± 0.00 | 0.93 ± 0.01 | 0.42 / 2.20 | 0.46 / 2.13 (3D-LSTM) | 0.82 / 0.74 (3D-ConvLSTM) |
| BRA-S | Ours (point) | strict | S2 | 3/3 | 0.34 ± 0.01 | 0.97 ± 0.01 | 0.66 ± 0.03 | 0.40 ± 0.02 | 0.34 / 0.98 | 0.39 / 0.94 (3D-ConvLSTM) | 0.76 / 0.34 (3D-ConvLSTM) |
| BRA-W | Ours (point) | strict | S2 | 3/3 | 0.18 ± 0.02 | 1.42 ± 0.02 | 0.57 ± 0.05 | 0.52 ± 0.03 | 0.22 / 1.39 | 0.24 / 1.37 (3D-LSTM) | 0.73 / 0.42 (3D-ConvLSTM) |
| GER-R | Ours (point) | strict | S2 | 3/3 | 0.30 ± 0.04 | 1.40 ± 0.04 | 0.61 ± 0.03 | 0.84 ± 0.03 | 0.36 / 1.33 | 0.49 / 1.20 (3D-ConvLSTM) | 0.82 / 0.57 (3D-LSTM) |
| GER-W | Ours (point) | strict | S2 | 3/3 | 0.33 ± 0.02 | 2.41 ± 0.03 | 0.58 ± 0.02 | 1.23 ± 0.03 | -0.32 / 3.38 | 0.34 / 2.40 (3D-ConvLSTM) | 0.65 / 1.12 (3D-ConvLSTM) |
| URG-S | Ours (point) | strict | S2 | 3/3 | 0.36 ± 0.00 | 1.27 ± 0.00 | 0.70 ± 0.01 | 0.59 ± 0.01 | 0.37 / 1.26 | 0.41 / 1.22 (3D-ConvLSTM) | 0.77 / 0.51 (3D-ConvLSTM) |
| ARG-C | Ours (point) | strict | S2+ADM | 3/3 | 0.59 ± 0.02 | 2.39 ± 0.04 | 0.75 ± 0.02 | 1.39 ± 0.05 | 0.59 / 2.40 | 0.70 / 2.03 (AFF) | 0.84 / 1.12 (AFF) |
| ARG-S | Ours (point) | strict | S2+ADM | 3/3 | 0.63 ± 0.01 | 0.92 ± 0.02 | 0.76 ± 0.02 | 0.60 ± 0.02 | 0.60 / 0.96 | 0.73 / 0.79 (AFF) | 0.84 / 0.49 (AFF) |
| ARG-W | Ours (point) | strict | S2+ADM | 3/3 | 0.75 ± 0.01 | 1.21 ± 0.03 | 0.85 ± 0.02 | 0.83 ± 0.05 | 0.74 / 1.21 | 0.84 / 0.96 (AFF) | 0.92 / 0.60 (AFF) |
| BRA-C | Ours (point) | strict | S2+ADM | 3/3 | 0.41 ± 0.01 | 2.22 ± 0.02 | 0.73 ± 0.01 | 0.92 ± 0.02 | 0.42 / 2.20 | 0.46 / 2.12 (AFF) | 0.84 / 0.70 (AFF) |
| BRA-S | Ours (point) | strict | S2+ADM | 3/3 | 0.35 ± 0.01 | 0.97 ± 0.01 | 0.67 ± 0.02 | 0.40 ± 0.01 | 0.34 / 0.98 | 0.44 / 0.90 (AFF) | 0.80 / 0.31 (AFF) |
| BRA-W | Ours (point) | strict | S2+ADM | 3/3 | 0.17 ± 0.01 | 1.43 ± 0.01 | 0.53 ± 0.06 | 0.55 ± 0.03 | 0.22 / 1.39 | 0.24 / 1.37 (AFF) | 0.72 / 0.42 (3D-ConvLSTM) |
| GER-R | Ours (point) | strict | S2+ADM | 3/3 | 0.39 ± 0.03 | 1.30 ± 0.04 | 0.69 ± 0.04 | 0.74 ± 0.05 | 0.36 / 1.33 | 0.49 / 1.20 (AFF) | 0.81 / 0.59 (3D-LSTM) |
| GER-W | Ours (point) | strict | S2+ADM | 3/3 | 0.35 ± 0.03 | 2.38 ± 0.05 | 0.61 ± 0.04 | 1.18 ± 0.06 | -0.32 / 3.38 | 0.44 / 2.20 (AFF) | 0.77 / 0.90 (MMGF) |
| URG-S | Ours (point) | strict | S2+ADM | 3/3 | 0.38 ± 0.00 | 1.24 ± 0.00 | 0.71 ± 0.00 | 0.58 ± 0.00 | 0.37 / 1.26 | 0.43 / 1.19 (AFF) | 0.81 / 0.46 (AFF) |

## LORO, provinces (Argentina; the paper's regions)

| Pair | Model | Policy | Inputs | Seeds complete | Pixel R² | Pixel RMSE | Field R² | Field RMSE | Paper LSTM pixel R² / RMSE (S2) | Paper best pixel R² / RMSE | Paper best field R² / RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ARG-C | Ours (point) | paper | S2 | 3/3 | 0.21 ± 0.04 | 3.30 ± 0.09 | 0.25 ± 0.07 | 2.43 ± 0.12 | 0.44 / 2.79 | 0.54 / 2.52 (3D-ConvLSTM) | 0.68 / 1.59 (3D-ConvLSTM) |
| ARG-S | Ours (point) | paper | S2 | 3/3 | 0.49 ± 0.02 | 1.09 ± 0.02 | 0.60 ± 0.02 | 0.77 ± 0.02 | 0.55 / 1.03 | 0.59 / 0.98 (3D-LSTM) | 0.70 / 0.67 (3D-LSTM) |
| ARG-W | Ours (point) | paper | S2 | 3/3 | 0.61 ± 0.04 | 1.49 ± 0.08 | 0.72 ± 0.03 | 1.13 ± 0.07 | 0.61 / 1.48 | 0.71 / 1.30 (3D-LSTM) | 0.81 / 0.92 (3D-LSTM) |
| ARG-C | Ours (point) | paper | S2+ADM | 3/3 | 0.31 ± 0.04 | 3.08 ± 0.09 | 0.34 ± 0.09 | 2.27 ± 0.15 | 0.44 / 2.79 | 0.54 / 2.52 (3D-ConvLSTM) | 0.68 / 1.59 (3D-ConvLSTM) |
| ARG-S | Ours (point) | paper | S2+ADM | 3/3 | 0.51 ± 0.05 | 1.07 ± 0.05 | 0.62 ± 0.06 | 0.75 ± 0.05 | 0.55 / 1.03 | 0.65 / 0.90 (AFF) | 0.78 / 0.57 (AFF) |
| ARG-W | Ours (point) | paper | S2+ADM | 3/3 | 0.57 ± 0.04 | 1.57 ± 0.07 | 0.69 ± 0.04 | 1.19 ± 0.07 | 0.61 / 1.48 | 0.78 / 1.12 (AFF) | 0.87 / 0.78 (AFF) |
| ARG-C | Ours (point) | strict | S2 | 3/3 | 0.33 ± 0.02 | 3.05 ± 0.04 | 0.42 ± 0.06 | 2.13 ± 0.11 | 0.44 / 2.79 | 0.54 / 2.52 (3D-ConvLSTM) | 0.68 / 1.59 (3D-ConvLSTM) |
| ARG-S | Ours (point) | strict | S2 | 3/3 | 0.49 ± 0.03 | 1.09 ± 0.03 | 0.60 ± 0.02 | 0.76 ± 0.02 | 0.55 / 1.03 | 0.59 / 0.98 (3D-LSTM) | 0.70 / 0.67 (3D-LSTM) |
| ARG-W | Ours (point) | strict | S2 | 3/3 | 0.62 ± 0.03 | 1.48 ± 0.05 | 0.74 ± 0.03 | 1.08 ± 0.06 | 0.61 / 1.48 | 0.71 / 1.30 (3D-LSTM) | 0.81 / 0.92 (3D-LSTM) |
| ARG-C | Ours (point) | strict | S2+ADM | 3/3 | 0.27 ± 0.03 | 3.19 ± 0.07 | 0.31 ± 0.05 | 2.33 ± 0.09 | 0.44 / 2.79 | 0.54 / 2.52 (3D-ConvLSTM) | 0.68 / 1.59 (3D-ConvLSTM) |
| ARG-S | Ours (point) | strict | S2+ADM | 3/3 | 0.53 ± 0.02 | 1.04 ± 0.02 | 0.64 ± 0.02 | 0.73 ± 0.02 | 0.55 / 1.03 | 0.65 / 0.90 (AFF) | 0.78 / 0.57 (AFF) |
| ARG-W | Ours (point) | strict | S2+ADM | 3/3 | 0.55 ± 0.03 | 1.60 ± 0.05 | 0.67 ± 0.01 | 1.22 ± 0.02 | 0.61 / 1.48 | 0.78 / 1.12 (AFF) | 0.87 / 0.78 (AFF) |

## LORO, farm regions

| Pair | Model | Policy | Inputs | Seeds complete | Pixel R² | Pixel RMSE | Field R² | Field RMSE | Paper LSTM pixel R² / RMSE (S2) | Paper best pixel R² / RMSE | Paper best field R² / RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BRA-C | Ours (point) | paper | S2 | 3/3 | 0.12 ± 0.05 | 2.70 ± 0.07 | 0.09 ± 0.10 | 1.69 ± 0.09 | 0.27 / 2.46 | 0.34 / 2.33 (3D-LSTM) | 0.59 / 1.13 (3D-LSTM) |
| BRA-S | Ours (point) | paper | S2 | 3/3 | 0.13 ± 0.03 | 1.12 ± 0.02 | 0.19 ± 0.06 | 0.62 ± 0.02 | 0.16 / 1.10 | 0.22 / 1.06 (3D-LSTM) | 0.41 / 0.53 (3D-LSTM) |
| BRA-W | Ours (point) | paper | S2 | 3/3 | 0.06 ± 0.04 | 1.52 ± 0.03 | 0.22 ± 0.06 | 0.70 ± 0.03 | 0.11 / 1.48 | 0.20 / 1.41 (3D-LSTM) | 0.48 / 0.57 (3D-LSTM) |
| GER-R | Ours (point) | paper | S2 | 3/3 | -0.13 ± 0.07 | 1.77 ± 0.05 | -0.01 ± 0.12 | 1.35 ± 0.08 | -0.05 / 1.71 | 0.17 / 1.52 (3D-LSTM) | 0.26 / 1.16 (3D-LSTM) |
| GER-W | Ours (point) | paper | S2 | 3/3 | -0.06 ± 0.03 | 3.04 ± 0.05 | -0.31 ± 0.12 | 2.17 ± 0.10 | -0.78 / 3.93 | 0.10 / 2.79 (Transformer) | 0.14 / 1.76 (3D-ConvLSTM) |
| URG-S | Ours (point) | paper | S2 | 3/3 | 0.31 ± 0.00 | 1.31 ± 0.00 | 0.59 ± 0.01 | 0.68 ± 0.01 | 0.34 / 1.28 | 0.36 / 1.26 (3D-ConvLSTM) | 0.68 / 0.61 (3D-ConvLSTM) |
| BRA-C | Ours (point) | paper | S2+ADM | 3/3 | 0.22 ± 0.04 | 2.54 ± 0.07 | 0.32 ± 0.07 | 1.46 ± 0.08 | 0.27 / 2.46 | 0.37 / 2.29 (3D-LSTM) | 0.65 / 1.05 (3D-LSTM) |
| BRA-S | Ours (point) | paper | S2+ADM | 3/3 | 0.18 ± 0.03 | 1.09 ± 0.02 | 0.29 ± 0.04 | 0.58 ± 0.02 | 0.16 / 1.10 | 0.35 / 0.97 (AFF) | 0.63 / 0.42 (AFF) |
| BRA-W | Ours (point) | paper | S2+ADM | 3/3 | 0.09 ± 0.02 | 1.50 ± 0.02 | 0.27 ± 0.06 | 0.68 ± 0.03 | 0.11 / 1.48 | 0.18 / 1.42 (AFF) | 0.52 / 0.55 (AFF) |
| GER-R | Ours (point) | paper | S2+ADM | 3/3 | -0.25 ± 0.02 | 1.87 ± 0.01 | -0.32 ± 0.06 | 1.54 ± 0.03 | -0.05 / 1.71 | 0.15 / 1.55 (AFF) | 0.25 / 1.16 (3D-ConvLSTM) |
| GER-W | Ours (point) | paper | S2+ADM | 3/3 | -0.24 ± 0.10 | 3.28 ± 0.13 | -0.71 ± 0.22 | 2.48 ± 0.16 | -0.78 / 3.93 | 0.07 / 2.84 (3D-ConvLSTM) | 0.14 / 1.76 (3D-ConvLSTM) |
| URG-S | Ours (point) | paper | S2+ADM | 3/3 | 0.31 ± 0.03 | 1.31 ± 0.03 | 0.57 ± 0.05 | 0.70 ± 0.04 | 0.34 / 1.28 | 0.36 / 1.26 (3D-ConvLSTM) | 0.68 / 0.61 (3D-ConvLSTM) |
| BRA-C | Ours (point) | strict | S2 | 3/3 | 0.10 ± 0.05 | 2.73 ± 0.07 | 0.14 ± 0.08 | 1.64 ± 0.08 | 0.27 / 2.46 | 0.34 / 2.33 (3D-LSTM) | 0.59 / 1.13 (3D-LSTM) |
| BRA-S | Ours (point) | strict | S2 | 3/3 | 0.05 ± 0.04 | 1.17 ± 0.02 | 0.05 ± 0.08 | 0.68 ± 0.03 | 0.16 / 1.10 | 0.22 / 1.06 (3D-LSTM) | 0.41 / 0.53 (3D-LSTM) |
| BRA-W | Ours (point) | strict | S2 | 3/3 | 0.09 ± 0.02 | 1.50 ± 0.01 | 0.17 ± 0.10 | 0.72 ± 0.04 | 0.11 / 1.48 | 0.20 / 1.41 (3D-LSTM) | 0.48 / 0.57 (3D-LSTM) |
| GER-R | Ours (point) | strict | S2 | 3/3 | -0.20 ± 0.10 | 1.83 ± 0.07 | -0.08 ± 0.14 | 1.39 ± 0.09 | -0.05 / 1.71 | 0.17 / 1.52 (3D-LSTM) | 0.26 / 1.16 (3D-LSTM) |
| GER-W | Ours (point) | strict | S2 | 3/3 | -0.05 ± 0.07 | 3.01 ± 0.10 | -0.30 ± 0.14 | 2.16 ± 0.12 | -0.78 / 3.93 | 0.10 / 2.79 (Transformer) | 0.14 / 1.76 (3D-ConvLSTM) |
| URG-S | Ours (point) | strict | S2 | 3/3 | 0.29 ± 0.00 | 1.33 ± 0.00 | 0.56 ± 0.01 | 0.71 ± 0.01 | 0.34 / 1.28 | 0.36 / 1.26 (3D-ConvLSTM) | 0.68 / 0.61 (3D-ConvLSTM) |
| BRA-C | Ours (point) | strict | S2+ADM | 3/3 | 0.12 ± 0.05 | 2.70 ± 0.07 | 0.13 ± 0.11 | 1.65 ± 0.10 | 0.27 / 2.46 | 0.37 / 2.29 (3D-LSTM) | 0.65 / 1.05 (3D-LSTM) |
| BRA-S | Ours (point) | strict | S2+ADM | 3/3 | 0.13 ± 0.02 | 1.12 ± 0.01 | 0.23 ± 0.01 | 0.61 ± 0.01 | 0.16 / 1.10 | 0.35 / 0.97 (AFF) | 0.63 / 0.42 (AFF) |
| BRA-W | Ours (point) | strict | S2+ADM | 3/3 | 0.05 ± 0.02 | 1.53 ± 0.01 | 0.10 ± 0.10 | 0.75 ± 0.04 | 0.11 / 1.48 | 0.18 / 1.42 (AFF) | 0.52 / 0.55 (AFF) |
| GER-R | Ours (point) | strict | S2+ADM | 3/3 | -0.12 ± 0.05 | 1.77 ± 0.04 | -0.10 ± 0.02 | 1.41 ± 0.01 | -0.05 / 1.71 | 0.15 / 1.55 (AFF) | 0.25 / 1.16 (3D-ConvLSTM) |
| GER-W | Ours (point) | strict | S2+ADM | 3/3 | -0.11 ± 0.14 | 3.10 ± 0.19 | -0.37 ± 0.29 | 2.21 ± 0.23 | -0.78 / 3.93 | 0.07 / 2.84 (3D-ConvLSTM) | 0.14 / 1.76 (3D-ConvLSTM) |
| URG-S | Ours (point) | strict | S2+ADM | 3/3 | 0.30 ± 0.01 | 1.32 ± 0.01 | 0.58 ± 0.02 | 0.69 ± 0.02 | 0.34 / 1.28 | 0.36 / 1.26 (3D-ConvLSTM) | 0.68 / 0.61 (3D-ConvLSTM) |

## LOYO (leave one year out)

| Pair | Model | Policy | Inputs | Seeds complete | Pixel R² | Pixel RMSE | Field R² | Field RMSE | Paper LSTM pixel R² / RMSE (S2) | Paper best pixel R² / RMSE | Paper best field R² / RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ARG-C | Ours (point) | paper | S2 | 3/3 | 0.35 ± 0.04 | 3.01 ± 0.09 | 0.44 ± 0.05 | 2.10 ± 0.09 | 0.39 / 2.90 | 0.48 / 2.68 (3D-ConvLSTM) | 0.61 / 1.75 (3D-ConvLSTM) |
| ARG-S | Ours (point) | paper | S2 | 3/3 | 0.48 ± 0.03 | 1.10 ± 0.04 | 0.56 ± 0.05 | 0.80 ± 0.05 | 0.46 / 1.12 | 0.55 / 1.03 (3D-LSTM) | 0.66 / 0.70 (3D-LSTM) |
| ARG-W | Ours (point) | paper | S2 | 3/3 | 0.58 ± 0.03 | 1.54 ± 0.06 | 0.71 ± 0.03 | 1.16 ± 0.05 | 0.61 / 1.49 | 0.68 / 1.35 (3D-LSTM) | 0.78 / 1.00 (3D-ConvLSTM) |
| BRA-C | Ours (point) | paper | S2 | 3/3 | 0.08 ± 0.02 | 2.77 ± 0.03 | -0.02 ± 0.06 | 1.79 ± 0.05 | 0.15 / 2.66 | 0.21 / 2.56 (3D-LSTM) | 0.36 / 1.42 (3D-LSTM) |
| BRA-S | Ours (point) | paper | S2 | 3/3 | 0.05 ± 0.04 | 1.17 ± 0.02 | -0.10 ± 0.09 | 0.73 ± 0.03 | 0.04 / 1.18 | 0.19 / 1.08 (3D-ConvLSTM) | 0.36 / 0.55 (3D-ConvLSTM) |
| BRA-W | Ours (point) | paper | S2 | 3/3 | -0.07 ± 0.07 | 1.63 ± 0.05 | -0.47 ± 0.26 | 0.96 ± 0.09 | 0.08 / 1.50 | 0.08 / 1.50 (LSTM) | -0.05 / 0.82 (LSTM) |
| GER-R | Ours (point) | paper | S2 | 3/3 | -0.21 ± 0.17 | 1.83 ± 0.12 | -0.08 ± 0.18 | 1.39 ± 0.11 | -0.05 / 1.71 | 0.21 / 1.49 (3D-ConvLSTM) | 0.43 / 1.02 (3D-ConvLSTM) |
| GER-W | Ours (point) | paper | S2 | 3/3 | 0.02 ± 0.05 | 2.92 ± 0.07 | -0.11 ± 0.10 | 2.00 ± 0.09 | -0.68 / 3.82 | 0.15 / 2.71 (3D-ConvLSTM) | 0.23 / 1.67 (3D-ConvLSTM) |
| URG-S | Ours (point) | paper | S2 | 3/3 | 0.29 ± 0.01 | 1.33 ± 0.01 | 0.49 ± 0.02 | 0.76 ± 0.02 | 0.33 / 1.29 | 0.35 / 1.27 (3D-LSTM) | 0.63 / 0.65 (3D-ConvLSTM) |
| ARG-C | Ours (point) | paper | S2+ADM | 3/3 | 0.35 ± 0.03 | 3.01 ± 0.08 | 0.41 ± 0.05 | 2.16 ± 0.08 | 0.39 / 2.90 | 0.44 / 2.78 (LSTM) | 0.42 / 2.13 (LSTM) |
| ARG-S | Ours (point) | paper | S2+ADM | 3/3 | 0.49 ± 0.01 | 1.09 ± 0.01 | 0.58 ± 0.01 | 0.79 ± 0.01 | 0.46 / 1.12 | 0.66 / 0.89 (AFF) | 0.77 / 0.59 (AFF) |
| ARG-W | Ours (point) | paper | S2+ADM | 3/3 | 0.46 ± 0.06 | 1.75 ± 0.09 | 0.56 ± 0.10 | 1.40 ± 0.16 | 0.61 / 1.49 | 0.72 / 1.27 (AFF) | 0.81 / 0.93 (AFF) |
| BRA-C | Ours (point) | paper | S2+ADM | 3/3 | 0.12 ± 0.05 | 2.71 ± 0.08 | 0.11 ± 0.15 | 1.66 ± 0.14 | 0.15 / 2.66 | 0.29 / 2.43 (AFF) | 0.45 / 1.32 (3D-ConvLSTM) |
| BRA-S | Ours (point) | paper | S2+ADM | 3/3 | 0.12 ± 0.04 | 1.13 ± 0.03 | 0.03 ± 0.14 | 0.68 ± 0.05 | 0.04 / 1.18 | 0.29 / 1.02 (AFF) | 0.45 / 0.52 (3D-LSTM) |
| BRA-W | Ours (point) | paper | S2+ADM | 3/3 | -0.07 ± 0.06 | 1.62 ± 0.04 | -0.53 ± 0.17 | 0.98 ± 0.05 | 0.08 / 1.50 | 0.11 / 1.48 (AFF) | 0.14 / 0.74 (AFF) |
| GER-R | Ours (point) | paper | S2+ADM | 3/3 | 0.09 ± 0.05 | 1.59 ± 0.05 | 0.24 ± 0.08 | 1.17 ± 0.06 | -0.05 / 1.71 | 0.31 / 1.38 (LSTM) | 0.58 / 0.87 (LSTM) |
| GER-W | Ours (point) | paper | S2+ADM | 3/3 | 0.03 ± 0.06 | 2.90 ± 0.09 | -0.12 ± 0.13 | 2.01 ± 0.12 | -0.68 / 3.82 | 0.22 / 2.60 (3D-ConvLSTM) | 0.46 / 1.39 (3D-ConvLSTM) |
| URG-S | Ours (point) | paper | S2+ADM | 3/3 | 0.27 ± 0.01 | 1.35 ± 0.01 | 0.47 ± 0.01 | 0.78 ± 0.01 | 0.33 / 1.29 | 0.32 / 1.30 (AFF) | 0.56 / 0.71 (3D-ConvLSTM) |
| ARG-C | Ours (point) | strict | S2 | 3/3 | 0.33 ± 0.08 | 3.05 ± 0.18 | 0.38 ± 0.10 | 2.20 ± 0.18 | 0.39 / 2.90 | 0.48 / 2.68 (3D-ConvLSTM) | 0.61 / 1.75 (3D-ConvLSTM) |
| ARG-S | Ours (point) | strict | S2 | 3/3 | 0.46 ± 0.05 | 1.12 ± 0.05 | 0.52 ± 0.10 | 0.83 ± 0.09 | 0.46 / 1.12 | 0.55 / 1.03 (3D-LSTM) | 0.66 / 0.70 (3D-LSTM) |
| ARG-W | Ours (point) | strict | S2 | 3/3 | 0.48 ± 0.07 | 1.73 ± 0.12 | 0.62 ± 0.06 | 1.32 ± 0.10 | 0.61 / 1.49 | 0.68 / 1.35 (3D-LSTM) | 0.78 / 1.00 (3D-ConvLSTM) |
| BRA-C | Ours (point) | strict | S2 | 3/3 | -0.17 ± 0.14 | 3.12 ± 0.18 | -0.49 ± 0.28 | 2.15 ± 0.21 | 0.15 / 2.66 | 0.21 / 2.56 (3D-LSTM) | 0.36 / 1.42 (3D-LSTM) |
| BRA-S | Ours (point) | strict | S2 | 3/3 | -0.14 ± 0.08 | 1.28 ± 0.04 | -0.46 ± 0.11 | 0.84 ± 0.03 | 0.04 / 1.18 | 0.19 / 1.08 (3D-ConvLSTM) | 0.36 / 0.55 (3D-ConvLSTM) |
| BRA-W | Ours (point) | strict | S2 | 3/3 | -0.07 ± 0.05 | 1.63 ± 0.04 | -0.54 ± 0.15 | 0.99 ± 0.05 | 0.08 / 1.50 | 0.08 / 1.50 (LSTM) | -0.05 / 0.82 (LSTM) |
| GER-R | Ours (point) | strict | S2 | 3/3 | 0.01 ± 0.08 | 1.66 ± 0.07 | 0.08 ± 0.13 | 1.29 ± 0.09 | -0.05 / 1.71 | 0.21 / 1.49 (3D-ConvLSTM) | 0.43 / 1.02 (3D-ConvLSTM) |
| GER-W | Ours (point) | strict | S2 | 3/3 | -0.07 ± 0.03 | 3.04 ± 0.05 | -0.24 ± 0.06 | 2.12 ± 0.05 | -0.68 / 3.82 | 0.15 / 2.71 (3D-ConvLSTM) | 0.23 / 1.67 (3D-ConvLSTM) |
| URG-S | Ours (point) | strict | S2 | 3/3 | 0.28 ± 0.02 | 1.33 ± 0.02 | 0.51 ± 0.05 | 0.75 ± 0.04 | 0.33 / 1.29 | 0.35 / 1.27 (3D-LSTM) | 0.63 / 0.65 (3D-ConvLSTM) |
| ARG-C | Ours (point) | strict | S2+ADM | 3/3 | 0.33 ± 0.02 | 3.04 ± 0.06 | 0.39 ± 0.03 | 2.19 ± 0.06 | 0.39 / 2.90 | 0.44 / 2.78 (LSTM) | 0.42 / 2.13 (LSTM) |
| ARG-S | Ours (point) | strict | S2+ADM | 3/3 | 0.47 ± 0.01 | 1.11 ± 0.01 | 0.55 ± 0.04 | 0.81 ± 0.03 | 0.46 / 1.12 | 0.66 / 0.89 (AFF) | 0.77 / 0.59 (AFF) |
| ARG-W | Ours (point) | strict | S2+ADM | 3/3 | 0.46 ± 0.12 | 1.74 ± 0.20 | 0.60 ± 0.12 | 1.35 ± 0.19 | 0.61 / 1.49 | 0.72 / 1.27 (AFF) | 0.81 / 0.93 (AFF) |
| BRA-C | Ours (point) | strict | S2+ADM | 3/3 | -0.40 ± 0.22 | 3.40 ± 0.27 | -0.71 ± 0.52 | 2.30 ± 0.34 | 0.15 / 2.66 | 0.29 / 2.43 (AFF) | 0.45 / 1.32 (3D-ConvLSTM) |
| BRA-S | Ours (point) | strict | S2+ADM | 3/3 | -0.04 ± 0.03 | 1.23 ± 0.02 | -0.27 ± 0.13 | 0.78 ± 0.04 | 0.04 / 1.18 | 0.29 / 1.02 (AFF) | 0.45 / 0.52 (3D-LSTM) |
| BRA-W | Ours (point) | strict | S2+ADM | 3/3 | -0.08 ± 0.02 | 1.63 ± 0.02 | -0.43 ± 0.05 | 0.95 ± 0.02 | 0.08 / 1.50 | 0.11 / 1.48 (AFF) | 0.14 / 0.74 (AFF) |
| GER-R | Ours (point) | strict | S2+ADM | 3/3 | 0.08 ± 0.04 | 1.60 ± 0.03 | 0.20 ± 0.05 | 1.20 ± 0.04 | -0.05 / 1.71 | 0.31 / 1.38 (LSTM) | 0.58 / 0.87 (LSTM) |
| GER-W | Ours (point) | strict | S2+ADM | 3/3 | -0.03 ± 0.03 | 2.98 ± 0.05 | -0.19 ± 0.03 | 2.07 ± 0.02 | -0.68 / 3.82 | 0.22 / 2.60 (3D-ConvLSTM) | 0.46 / 1.39 (3D-ConvLSTM) |
| URG-S | Ours (point) | strict | S2+ADM | 3/3 | 0.28 ± 0.03 | 1.34 ± 0.03 | 0.48 ± 0.07 | 0.77 ± 0.05 | 0.33 / 1.29 | 0.32 / 1.30 (AFF) | 0.56 / 0.71 (3D-ConvLSTM) |
