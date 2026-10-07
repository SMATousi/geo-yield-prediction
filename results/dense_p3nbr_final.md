# Dense raw S2 series: p3-nbr results (paper protocol)

Generated 2026-10-07 from all 217 per-pair folds (spec/yieldsat-dense-series.md). Pooled out-of-fold R² per pair (pixel / field), the paper's convention; selection on the test fold. Bold: dense p3-nbr at or above the paper's best model of that row. Source: results/dense_p3nbr_final.json.

#### CV10

| Pair | Paper LSTM S2+ADM | Paper best | Monthly p3-nbr | **Dense p3-nbr** |
|---|---|---|---|---|
| ARG-C | 0.58 / 0.76 | 0.70 / 0.84 | 0.68 / 0.85 | **0.72** / **0.88** |
| ARG-S | 0.59 / 0.72 | 0.73 / 0.84 | 0.69 / 0.80 | 0.72 / 0.83 |
| ARG-W | 0.75 / 0.85 | 0.84 / 0.92 | 0.80 / 0.89 | 0.82 / 0.92 |
| BRA-C | 0.43 / 0.81 | 0.46 / 0.84 | 0.49 / 0.85 | **0.50** / **0.90** |
| BRA-S | 0.33 / 0.64 | 0.44 / 0.80 | 0.41 / 0.78 | 0.43 / **0.83** |
| BRA-W | 0.21 / 0.58 | 0.24 / 0.73 | 0.22 / 0.74 | **0.25** / **0.77** |
| GER-R | 0.47 / 0.81 | 0.49 / 0.82 | 0.46 / 0.78 | **0.52** / **0.84** |
| GER-W | 0.35 / 0.63 | 0.44 / 0.77 | 0.44 / 0.76 | **0.47** / **0.79** |
| URG-S | 0.39 / 0.72 | 0.43 / 0.81 | 0.41 / 0.76 | 0.42 / 0.77 |
| **Mean** | 0.46 / 0.72 | 0.53 / 0.82 | 0.51 / 0.80 | 0.54 / 0.84 |

Dense ≥ paper best (pixel) on 5/9 pairs. Dense − monthly (mean over pairs, paired fold bootstrap 95% CI): pixel +0.029 [+0.022, +0.037], field +0.036 [+0.024, +0.047].

#### LOYO

| Pair | Paper LSTM S2+ADM | Paper best | Monthly p3-nbr | **Dense p3-nbr** |
|---|---|---|---|---|
| ARG-C | 0.44 / 0.42 | 0.48 / 0.61 | 0.48 / 0.58 | **0.54** / **0.62** |
| ARG-S | 0.49 / 0.62 | 0.66 / 0.77 | 0.58 / 0.69 | 0.63 / 0.73 |
| ARG-W | 0.61 / 0.74 | 0.72 / 0.81 | 0.65 / 0.76 | **0.74** / **0.86** |
| BRA-C | 0.19 / 0.29 | 0.29 / 0.45 | 0.28 / 0.43 | 0.12 / -0.02 |
| BRA-S | 0.17 / 0.26 | 0.29 / 0.45 | 0.23 / 0.30 | **0.31** / **0.55** |
| BRA-W | 0.09 / 0.12 | 0.11 / 0.14 | 0.04 / -0.10 | **0.15** / **0.39** |
| GER-R | 0.31 / 0.58 | 0.31 / 0.58 | 0.29 / 0.50 | 0.28 / 0.50 |
| GER-W | 0.18 / 0.33 | 0.22 / 0.46 | 0.22 / 0.27 | 0.08 / -0.00 |
| URG-S | 0.30 / 0.54 | 0.35 / 0.63 | 0.35 / 0.60 | **0.36** / **0.63** |
| **Mean** | 0.31 / 0.43 | 0.38 / 0.54 | 0.35 / 0.45 | 0.36 / 0.47 |

Dense ≥ paper best (pixel) on 5/9 pairs. Dense − monthly (mean over pairs, paired fold bootstrap 95% CI): pixel +0.010 [-0.032, +0.054], field +0.026 [-0.058, +0.103].

#### LORO

| Pair | Paper LSTM S2+ADM | Paper best | Monthly p3-nbr | **Dense p3-nbr** |
|---|---|---|---|---|
| ARG-C | 0.48 / 0.43 | 0.54 / 0.68 | 0.50 / 0.60 | **0.54** / 0.64 |
| ARG-S | 0.54 / 0.66 | 0.65 / 0.78 | 0.57 / 0.67 | **0.66** / 0.76 |
| ARG-W | 0.53 / 0.61 | 0.78 / 0.87 | 0.67 / 0.79 | 0.78 / **0.88** |
| BRA-C | 0.28 / 0.38 | 0.37 / 0.65 | 0.29 / 0.51 | 0.24 / 0.32 |
| BRA-S | 0.12 / 0.29 | 0.35 / 0.63 | 0.26 / 0.46 | 0.34 / 0.62 |
| BRA-W | 0.11 / 0.26 | 0.20 / 0.52 | 0.17 / 0.48 | 0.17 / 0.49 |
| GER-R | 0.05 / 0.20 | 0.17 / 0.26 | 0.05 / 0.14 | 0.15 / **0.38** |
| GER-W | -0.36 / -1.39 | 0.10 / 0.14 | 0.04 / -0.13 | 0.09 / 0.09 |
| URG-S | 0.32 / 0.56 | 0.36 / 0.68 | 0.35 / 0.66 | **0.36** / 0.66 |
| **Mean** | 0.23 / 0.22 | 0.39 / 0.58 | 0.32 / 0.46 | 0.37 / 0.54 |

Dense ≥ paper best (pixel) on 3/9 pairs. Dense − monthly (mean over pairs, paired fold bootstrap 95% CI): pixel +0.047 [+0.029, +0.071], field +0.075 [+0.031, +0.144].

