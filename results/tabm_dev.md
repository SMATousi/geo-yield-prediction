# TabM DEV results (round TM-1)

Generated 2026-10-05 16:39 UTC by `yieldsat_tabm_report.py`. spec/yieldsat-tabm.md. Pooled out-of-fold R² (the paper's computation) on the DEV subset (12 rows: ARG-W, BRA-C, GER-R, URG-S × CV10/LOYO/LORO, paper policy, S2+ADM, seed 0). Δ: paired fold bootstrap (2,000 draws), 95% CI. Success bar (improvement plan): paper best + 0.03 at pixel and field level.

Feature sets: F0 = month-aligned flat S2 (12 bands) + weather (4 sums) per slot, time features, DEM/terrain/soil, validity masks; F1 = F0 + 5×5 field-masked neighbourhood mean of the S2 bands. TabM: k = 32, 3 × 256, dropout 0.2, periodic embeddings d = 8, early stopping on validation seasons (TM-06 sweep).

## Summary

| Model | Rows | Pixel R² | Field R² | CV10 / LOYO / LORO pixel |
|---|---|---|---|---|
| TabM F1 (+ S6 neighbourhood) | 12 | 0.351 | 0.478 | 0.50 / 0.29 / 0.27 |
| TabM F0 | 12 | 0.342 | 0.462 | 0.51 / 0.25 / 0.27 |
| LightGBM F0 | 12 | 0.313 | 0.420 | 0.49 / 0.21 / 0.23 |
| MLP F0 (k = 1) | 12 | 0.288 | 0.392 | 0.47 / 0.14 / 0.26 |
| Point + S6 (p3-nbr, dev_r2) | 12 | 0.332 | 0.497 | 0.50 / 0.24 / 0.26 |
| Point model A0 (before_full seed 0) | 12 | 0.305 | 0.470 | 0.48 / 0.25 / 0.19 |
| **Paper best** | 12 | **0.460** | **0.686** | 0.56 / 0.41 / 0.41 |
| Success bar | | 0.490 | 0.716 | |

## Paired comparisons

| Comparison | Rows | Δ pixel [95% CI] | Δ field [95% CI] | Δ pixel CV10 / LOYO / LORO |
|---|---|---|---|---|
| TabM F0 − Point model A0 (before_full seed 0) | 12 | +0.037 [-0.003, +0.076] | -0.008 [-0.112, +0.057] | +0.028 / +0.004 / +0.081 |
| TabM F1 (+ S6 neighbourhood) − Point model A0 (before_full seed 0) | 12 | +0.046 [+0.008, +0.083] | +0.009 [-0.118, +0.081] | +0.018 / +0.042 / +0.078 |
| TabM F0 − Point + S6 (p3-nbr, dev_r2) | 12 | +0.010 [-0.043, +0.054] | -0.036 [-0.160, +0.036] | +0.009 / +0.012 / +0.010 |
| TabM F1 (+ S6 neighbourhood) − Point + S6 (p3-nbr, dev_r2) | 12 | +0.019 [-0.027, +0.052] | -0.019 [-0.140, +0.043] | -0.001 / +0.050 / +0.007 |
| TabM F1 (+ S6 neighbourhood) − TabM F0 | 12 | +0.009 [-0.014, +0.034] | +0.017 [-0.026, +0.065] | -0.009 / +0.038 / -0.003 |
| TabM F0 − LightGBM F0 | 12 | +0.029 [-0.009, +0.070] | +0.042 [-0.035, +0.109] | +0.014 / +0.037 / +0.037 |
| TabM F0 − MLP F0 (k = 1) | 12 | +0.054 [+0.028, +0.098] | +0.070 [+0.013, +0.176] | +0.036 / +0.113 / +0.013 |
| LightGBM F0 − Point model A0 (before_full seed 0) | 12 | +0.008 [-0.038, +0.044] | -0.050 [-0.165, +0.023] | +0.013 / -0.033 / +0.044 |

## Per row (pixel / field R²)

| Model | ARG-W CV10 | BRA-C CV10 | GER-R CV10 | URG-S CV10 | ARG-W LORO | BRA-C LORO | GER-R LORO | URG-S LORO | ARG-W LOYO | BRA-C LOYO | GER-R LOYO | URG-S LOYO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TabM F1 (+ S6 neighbourhood) | 0.75 / 0.84 | 0.43 / 0.77 | 0.40 / 0.70 | 0.41 / 0.75 | 0.52 / 0.64 | -0.01 / -0.24 | 0.21 / 0.35 | 0.34 / 0.60 | 0.59 / 0.69 | -0.01 / -0.26 | 0.25 / 0.35 | 0.32 / 0.54 |
| TabM F0 | 0.76 / 0.84 | 0.45 / 0.79 | 0.41 / 0.72 | 0.40 / 0.75 | 0.55 / 0.64 | -0.01 / -0.28 | 0.21 / 0.34 | 0.33 / 0.60 | 0.56 / 0.66 | -0.13 / -0.45 | 0.27 / 0.42 | 0.30 / 0.50 |
| LightGBM F0 | 0.72 / 0.80 | 0.45 / 0.79 | 0.41 / 0.69 | 0.40 / 0.75 | 0.57 / 0.65 | 0.07 / -0.03 | -0.02 / -0.19 | 0.31 / 0.54 | 0.37 / 0.45 | 0.02 / -0.14 | 0.18 / 0.28 | 0.27 / 0.45 |
| MLP F0 (k = 1) | 0.70 / 0.79 | 0.41 / 0.73 | 0.37 / 0.67 | 0.40 / 0.76 | 0.53 / 0.61 | 0.01 / -0.17 | 0.17 / 0.42 | 0.31 / 0.57 | 0.44 / 0.54 | -0.26 / -0.83 | 0.11 / 0.16 | 0.26 / 0.45 |
| Point + S6 (p3-nbr, dev_r2) | 0.76 / 0.86 | 0.46 / 0.83 | 0.39 / 0.68 | 0.39 / 0.72 | 0.61 / 0.75 | 0.17 / 0.22 | -0.05 / -0.02 | 0.31 / 0.60 | 0.55 / 0.67 | 0.08 / -0.03 | 0.03 / 0.18 | 0.30 / 0.49 |
| Point model A0 (before_full seed 0) | 0.74 / 0.85 | 0.45 / 0.80 | 0.37 / 0.65 | 0.36 / 0.70 | 0.53 / 0.69 | 0.18 / 0.24 | -0.24 / -0.36 | 0.29 / 0.54 | 0.51 / 0.61 | 0.18 / 0.28 | 0.04 / 0.19 | 0.26 / 0.46 |
| Paper best | 0.84 / 0.92 | 0.46 / 0.84 | 0.49 / 0.81 | 0.43 / 0.81 | 0.78 / 0.87 | 0.37 / 0.65 | 0.15 / 0.25 | 0.36 / 0.68 | 0.72 / 0.81 | 0.29 / 0.45 | 0.31 / 0.58 | 0.32 / 0.56 |

## Decisions (spec/yieldsat-tabm.md §4)

- **TabM F0 adopted as point backbone** (Δ pixel ≥ +0.01 vs A0 with CI > 0, field not worse): no
- **TabM F1 (+ S6 neighbourhood) adopted as point backbone** (Δ pixel ≥ +0.01 vs A0 with CI > 0, field not worse): yes
- TabM F0 passes the paper bar: no
- TabM F1 (+ S6 neighbourhood) passes the paper bar: no
