# Knowledge pretraining, DEV round (phase 1 (stopped 2026-10-05 at 84% of fine-tuning))

Generated 2026-10-05 15:50 UTC by `yieldsat_pretrain_report.py`.
Criteria and decision rules: spec/pretraining/success-criteria.md. Arms: spec/yieldsat-point-knowledge-pretraining.md §6.
Pooled out-of-fold R² on the DEV subset. Δ CIs: paired bootstrap over DEV folds (2000 draws). Failed criteria are reported as failures.

## Yield prediction (E)

| Arm | Rows | Pixel R² | Field R² | CV10 / LOYO / LORO pixel |
|---|---|---|---|---|
| A0 | 11/11 | 0.299 | 0.454 | 0.52 / 0.25 / 0.19 |
| A2 | 11/11 | 0.296 | 0.454 | 0.52 / 0.23 / 0.19 |
| A3 | 11/11 | 0.296 | 0.435 | 0.53 / 0.22 / 0.19 |
| A6 | 9/11 (incomplete) | 0.368 | 0.551 | 0.51 / 0.27 / 0.32 |
| A7 | 9/11 (incomplete) | 0.368 | 0.535 | 0.52 / 0.25 / 0.33 |
| **Paper best** | | **0.457** | **0.675** | |

Incomplete experiments (excluded): {'A2': 1, 'A3': 1, 'A6': 3, 'A7': 3}

| Comparison | Rows | Δ pixel [95% CI] | Δ field [95% CI] | Δ pixel CV10 / LOYO / LORO |
|---|---|---|---|---|
| A2 − A0 | 11 | -0.003 [-0.038, +0.033] | +0.000 [-0.049, +0.061] | -0.001 / -0.013 / +0.006 |
| A3 − A0 | 11 | -0.003 [-0.034, +0.026] | -0.019 [-0.072, +0.038] | +0.012 / -0.021 / +0.005 |
| A3 − A2 | 11 | +0.000 [-0.041, +0.040] | -0.019 [-0.093, +0.055] | +0.012 / -0.008 / -0.001 |
| A3 − A6 | 9 | +0.007 [-0.031, +0.056] | -0.023 [-0.093, +0.077] | +0.017 / -0.001 / +0.007 |
| A3 − A7 | 9 | +0.007 [-0.021, +0.059] | -0.007 [-0.071, +0.075] | +0.005 / +0.015 / +0.002 |

## Pretraining health (P)

| Arm | Units | P1 convergence | P2 no collapse | P3 coverage | P4 estimator sanity |
|---|---|---|---|---|---|
| A2 | 46 | 35/46 | 46/46 | n/a | n/a |
| A3 | 46 | 35/46 | 46/46 | 46/46 | 46/46 |
| A6 | 46 | 30/46 | 46/46 | n/a | n/a |
| A7 | 46 | 38/46 | 46/46 | 46/46 | 46/46 |

## Knowledge learned (I), held-out seasons

| Concept | A3 AUROC | A7 AUROC | I1 pass (≥0.70 and ≥ A7+0.10) |
|---|---|---|---|
| active_spectral_state | 0.937 | 0.509 | yes |
| clay_rich_surface | 0.979 | 0.487 | yes |
| cool_regime | 0.805 | 0.470 | yes |
| fine_surface_texture | 0.981 | 0.548 | yes |
| low_elevation_position | 0.575 | 0.562 | no |
| optical_growth | 0.919 | 0.517 | yes |
| organic_surface | 0.991 | 0.483 | yes |
| persistent_canopy | 0.916 | 0.500 | yes |
| pole_facing_aspect | 0.933 | 0.502 | yes |
| rain_supported | 0.931 | 0.464 | yes |
| spectral_moisture_state | 0.934 | 0.519 | yes |
| warm_regime | 0.818 | 0.470 | yes |

I5: mean I1 gain over A7, active-rule concepts +0.392 vs abstaining-rule concepts +0.383.

| Rule | Pairs (unit × season) | A3 − A7 alignment [95% CI] |
|---|---|---|
| ys_r01 | 1633 | +0.150 [+0.137, +0.164] |
| ys_r02 | 2077 | +0.092 [+0.078, +0.104] |
| ys_r03 | 1377 | +0.207 [+0.193, +0.222] |
| ys_r04 | 3104 | -0.022 [-0.023, -0.022] |
| ys_r05 | 0 | n/a (abstains) |
| ys_r06 | 2882 | +0.080 [+0.072, +0.089] |

I3/I4 use the 46 units finished by every arm.

| Probe (R², held-out) | A2 | A3 | A6 | A7 |
|---|---|---|---|---|
| clay030 | 0.210 | 0.502 | 0.386 | 0.405 |
| ndmi_mid | 0.367 | 0.442 | 0.375 | 0.244 |
| ndvi_rise | -0.042 | 0.094 | 0.047 | -0.015 |
| precip_mid_mm | 0.242 | 0.146 | 0.211 | 0.067 |
| rel_elev | -0.341 | -0.343 | -0.366 | -0.456 |
| soc030_gkg | 0.374 | 0.522 | 0.391 | 0.435 |
| tmean_mid_c | -0.770 | -0.941 | -1.001 | -0.789 |

I4 held-out SSL losses: A2 masked_observation 0.2673, forecast_last_valid 0.2415; A3 masked_observation 0.2612, forecast_last_valid 0.2348; A6 masked_observation 0.2653, forecast_last_valid 0.2466; A7 masked_observation 0.2671, forecast_last_valid 0.2576

## Criteria and decisions

| Criterion | Result |
|---|---|
| E1 | **fail** |
| E2 | **fail** |
| E4 | **fail** |
| E5 | **fail** |
| E6 (A3 ≥ paper best + 0.03) | **fail** |
| I1 | **fail** |
| I5 | pass |
| I2 | **fail** |
| I3 | **fail** |
| I4 | pass |

- **Pretraining is healthy** (A2, A3: P1–P4 on every unit): no
- **The knowledge is doing its job** (I1, I2, I4, E2 vs A7): no
- **Pretraining is helping** (E1, E2, E3 or E4, E5): no

## Compute (pretraining wall-hours per arm, concurrent runs share GPUs)

A2 33.8 h · A3 72.9 h · A6 50.2 h · A7 65.5 h
