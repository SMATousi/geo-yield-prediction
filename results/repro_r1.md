# Reproduction R1: input-fusion LSTM, GER-R CV10 (thesis configuration and ablations)

Generated 2026-10-05 20:42 UTC by `yieldsat_repro_report.py` (spec/yieldsat-paper-reproduction.md). Fold mean ± std over folds (the paper's stated metric) and pooled out-of-fold R², next to the paper's "S2+ADM / input / LSTM" rows.

> **Correction (2026-10-06):** the experiment table below scored the *last non-improving* epoch, not the validation-best epoch: `checkpoint_best.pth` was saved on non-improving epochs from 81e9fdd until this fix. The values are biased low. The H1 table further down is computed from the per-epoch history and is correct. See spec/yieldsat-paper-protocol.md §4.


| Experiment | Row | Folds | Pixel R² fold mean ± std | Pixel pooled | Paper pixel | Field R² fold mean ± std | Field pooled | Paper field |
|---|---|---|---|---|---|---|---|---|
| a1-s2-thesis_seed0 | GER-R CV10 | 10 | 0.32 ± 0.18 | 0.35 | 0.47 ± 0.10 | 0.60 ± 0.21 | 0.66 | 0.81 ± 0.09 |
| a2-adm-nocoords_seed0 | GER-R CV10 | 10 | 0.27 ± 0.14 | 0.29 | 0.47 ± 0.10 | 0.56 ± 0.26 | 0.60 | 0.81 ± 0.09 |
| a3-adm-firstmasked_seed0 | GER-R CV10 | 10 | 0.25 ± 0.13 | 0.27 | 0.47 ± 0.10 | 0.52 ± 0.22 | 0.56 | 0.81 ± 0.09 |
| a4-adm-tutorial_seed0 | GER-R CV10 | 10 | 0.23 ± 0.14 | 0.23 | 0.47 ± 0.10 | 0.45 ± 0.29 | 0.47 | 0.81 ± 0.09 |
| a5-adm-noes_seed0 | GER-R CV10 | 10 | 0.21 ± 0.14 | 0.24 | 0.47 ± 0.10 | 0.52 ± 0.22 | 0.54 | 0.81 ± 0.09 |
| b1-s2-weather_seed0 | GER-R CV10 | 10 | 0.26 ± 0.16 | 0.29 | 0.47 ± 0.10 | 0.49 ± 0.22 | 0.54 | 0.81 ± 0.09 |
| b2-s2-soil_seed0 | GER-R CV10 | 10 | 0.27 ± 0.13 | 0.29 | 0.47 ± 0.10 | 0.53 ± 0.24 | 0.57 | 0.81 ± 0.09 |
| b3-s2-dem_seed0 | GER-R CV10 | 10 | 0.37 ± 0.11 | 0.39 | 0.47 ± 0.10 | 0.67 ± 0.12 | 0.70 | 0.81 ± 0.09 |
| b4-s2-terrain_seed0 | GER-R CV10 | 10 | 0.32 ± 0.14 | 0.34 | 0.47 ± 0.10 | 0.62 ± 0.16 | 0.66 | 0.81 ± 0.09 |
| b5-s2-weather-soil-dem_seed0 | GER-R CV10 | 10 | 0.29 ± 0.17 | 0.31 | 0.47 ± 0.10 | 0.55 ± 0.33 | 0.59 | 0.81 ± 0.09 |
| r1-lstm-paperlike_seed0 | GER-R CV10 | 10 | 0.28 ± 0.12 | 0.30 | 0.47 ± 0.10 | 0.53 ± 0.21 | 0.57 | 0.81 ± 0.09 |

## H1: epoch selection on the test fold (diagnostic)

50 epochs without early stopping; the test fold is scored after every epoch (`--diag_test_each_epoch`, never used for selection). Values are fold mean ± std, pixel / field R².

| Run | Validation-selected epoch | Test-selected epoch (optimistic) | Last epoch |
|---|---|---|---|
| S2 only | 0.33 ± 0.15 / 0.63 ± 0.17 | 0.44 ± 0.10 / 0.78 ± 0.13 | 0.24 / 0.56 |
| S2 + weather + soil + DEM | 0.35 ± 0.08 / 0.64 ± 0.18 | 0.42 ± 0.06 / 0.75 ± 0.09 | 0.31 / 0.64 |

Paper: S2 0.36 ± 0.14 / 0.62 ± 0.25; S2+ADM input fusion 0.47 ± 0.10 / 0.81 ± 0.09.

Note: the paper-row columns in the table above use the S2+ADM row for every experiment; the S2-only run (a1) should be read against the S2 row (0.36 / 0.62).
