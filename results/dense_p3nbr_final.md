

## S2 observation dropout (min keep 0.1) on the dense series

Dense p3-nbr trained with `--s2_obs_dropout 0.1` (spec/yieldsat-dense-series.md §10) vs dense p3-nbr without it, same folds. Pooled out-of-fold R² (pixel / field). ARG-S LORO (6/8 folds) and BRA-S CV10 (8/10) were still running and are left out; all other rows are complete. Bold: best of the four columns per pair. Source: results/obsdrop_final.json.

#### CV10

| Pair | Paper baseline (LSTM S2+ADM) | Paper best | Dense | Dense + dropout |
|---|---|---|---|---|
| ARG-C | 0.58 / 0.76 | 0.70 / 0.84 | **0.72** / **0.88** | 0.72 / 0.86 |
| ARG-S | 0.59 / 0.72 | **0.73** / **0.84** | 0.72 / 0.83 | 0.72 / 0.83 |
| ARG-W | 0.75 / 0.85 | **0.84** / **0.92** | 0.82 / 0.92 | 0.82 / 0.91 |
| BRA-C | 0.43 / 0.81 | 0.46 / 0.84 | **0.50** / **0.90** | 0.48 / 0.84 |
| BRA-W | 0.21 / 0.58 | 0.24 / 0.73 | **0.25** / **0.77** | 0.22 / 0.70 |
| GER-R | 0.47 / 0.81 | 0.49 / 0.82 | 0.52 / **0.84** | **0.53** / 0.82 |
| GER-W | 0.35 / 0.63 | 0.44 / 0.77 | 0.47 / 0.79 | **0.48** / **0.79** |
| URG-S | 0.39 / 0.72 | **0.43** / **0.81** | 0.42 / 0.77 | 0.42 / 0.75 |
| **Mean (8 pairs)** | 0.47 / 0.73 | 0.54 / 0.82 | 0.55 / 0.84 | 0.55 / 0.82 |

#### LOYO

| Pair | Paper baseline (LSTM S2+ADM) | Paper best | Dense | Dense + dropout |
|---|---|---|---|---|
| ARG-C | 0.44 / 0.42 | 0.48 / 0.61 | **0.54** / **0.62** | 0.50 / 0.55 |
| ARG-S | 0.49 / 0.62 | **0.66** / **0.77** | 0.63 / 0.73 | 0.63 / 0.71 |
| ARG-W | 0.61 / 0.74 | 0.72 / 0.81 | **0.74** / **0.86** | 0.57 / 0.69 |
| BRA-C | 0.19 / 0.29 | **0.29** / **0.45** | 0.12 / -0.02 | 0.12 / -0.03 |
| BRA-S | 0.17 / 0.26 | 0.29 / 0.45 | 0.31 / 0.55 | **0.34** / **0.60** |
| BRA-W | 0.09 / 0.12 | 0.11 / 0.14 | 0.15 / 0.39 | **0.15** / **0.41** |
| GER-R | 0.31 / 0.58 | **0.31** / **0.58** | 0.28 / 0.50 | 0.31 / 0.40 |
| GER-W | 0.18 / 0.33 | 0.22 / **0.46** | 0.08 / -0.00 | **0.28** / 0.39 |
| URG-S | 0.30 / 0.54 | 0.35 / 0.63 | **0.36** / **0.63** | 0.31 / 0.57 |
| **Mean (9 pairs)** | 0.31 / 0.43 | 0.38 / 0.54 | 0.36 / 0.47 | 0.36 / 0.48 |

#### LORO

| Pair | Paper baseline (LSTM S2+ADM) | Paper best | Dense | Dense + dropout |
|---|---|---|---|---|
| ARG-C | 0.48 / 0.43 | 0.54 / 0.68 | 0.54 / 0.64 | **0.56** / **0.69** |
| ARG-W | 0.53 / 0.61 | **0.78** / 0.87 | 0.78 / **0.88** | 0.72 / 0.82 |
| BRA-C | 0.28 / 0.38 | **0.37** / **0.65** | 0.24 / 0.32 | 0.28 / 0.44 |
| BRA-S | 0.12 / 0.29 | 0.35 / 0.63 | 0.34 / 0.62 | **0.35** / **0.67** |
| BRA-W | 0.11 / 0.26 | **0.20** / **0.52** | 0.17 / 0.49 | 0.15 / 0.32 |
| GER-R | 0.05 / 0.20 | 0.17 / 0.26 | 0.15 / **0.38** | **0.19** / 0.34 |
| GER-W | -0.36 / -1.39 | 0.10 / **0.14** | 0.09 / 0.09 | **0.13** / 0.00 |
| URG-S | 0.32 / 0.56 | 0.36 / **0.68** | **0.36** / 0.66 | 0.35 / 0.61 |
| **Mean (8 pairs)** | 0.19 / 0.17 | 0.36 / 0.55 | 0.33 / 0.51 | 0.34 / 0.48 |

#### Dropout − dense (mean over pairs, paired fold bootstrap 95% CI)

| Protocol | Pixel | Field |
|---|---|---|
| CV10 | -0.006 [-0.012, +0.001] | -0.022 [-0.036, -0.006] |
| LOYO | +0.002 [-0.042, +0.048] | +0.004 [-0.078, +0.090] |
| LORO | +0.006 [-0.028, +0.034] | -0.026 [-0.075, +0.046] |
| ALL | +0.001 [-0.017, +0.021] | -0.014 [-0.045, +0.023] |

**Conclusion:** applied to every pair, observation dropout has no net effect on LOYO or LORO (CIs span 0; large gains on GER-W LOYO and BRA-C LORO are offset by losses on ARG-W, ARG-C and URG-S) and costs field-level accuracy on CV10 (−0.022, significant). It fixes the sparse-year failure it targets (GER-W 2016) but is not a general improvement.
