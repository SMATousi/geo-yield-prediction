# YS-11 fusion ablation (2026-09-29)

These results compare four ways of fusing the per-stream encodings in the
YieldSAT point model. Every variant uses the same encoders, head, data, budget
and model selection. Settings:
- 20×500 steps of 512 cells;
- `before_harvest` 30-day cutoff;
- crop context on;
- best epoch by validation pixel RMSE;
- test on every held-out cell;
- seeds 0–2.

The code was frozen at commit `b34f1cd` and the commands are in
`run_ablation.sh`. The stress test removes all optical input for a
deterministic 50% of test cells (`--eval_drop_stream yieldsat_s2`).

`runs.csv` has one row per run; `summary.csv` has the mean ± std over seeds
for each split and variant.

| Split | Variant | Params | Pixel RMSE | Pixel R² | Field RMSE | Field R² | Macro-country pixel RMSE | Stress pixel RMSE |
|---|---|---|---|---|---|---|---|---|
| farm-held-out | `perceiver_summary` (current) | 1,127,937 | **1.643 ± 0.024** | **0.662** | 1.069 ± 0.052 | 0.799 | 1.662 | **1.868 ± 0.038** |
| farm-held-out | `perceiver_tokens` (16 latents) | 1,204,769 | 1.665 ± 0.033 | 0.653 | **1.067 ± 0.044** | **0.800** | **1.636** | 1.910 ± 0.006 |
| farm-held-out | `token_transformer` | 1,060,161 | 1.668 ± 0.057 | 0.652 | 1.078 ± 0.062 | 0.796 | 1.647 | 1.913 ± 0.084 |
| farm-held-out | `concat_mlp` | 893,761 | 1.693 ± 0.037 | 0.642 | 1.133 ± 0.044 | 0.775 | 1.739 | 1.874 ± 0.060 |
| block-held-out (20 km) | `perceiver_summary` (current) | 1,127,937 | **1.492 ± 0.030** | **0.643** | **0.954 ± 0.030** | **0.797** | 1.541 | **1.648 ± 0.025** |
| block-held-out (20 km) | `perceiver_tokens` (16 latents) | 1,204,769 | 1.525 ± 0.021 | 0.627 | 1.038 ± 0.035 | 0.760 | 1.605 | 1.664 ± 0.035 |
| block-held-out (20 km) | `token_transformer` | 1,060,161 | 1.526 ± 0.016 | 0.627 | 0.962 ± 0.031 | 0.794 | **1.526** | 1.678 ± 0.022 |
| block-held-out (20 km) | `concat_mlp` | 893,761 | 1.514 ± 0.025 | 0.632 | 0.979 ± 0.007 | 0.786 | 1.544 | 1.671 ± 0.016 |

Latent sweep for `perceiver_tokens` (farm split, seed 0 only), pixel / field
RMSE:

| Latents | Pixel RMSE | Field RMSE |
|---|---|---|
| 8 | 1.698 | 1.132 |
| 16 (seed 0) | 1.619 | 1.006 |
| 32 | 1.711 | 1.121 |

**Decision:** keep `perceiver_summary` as the default. See
`spec/yieldsat_data_contract.md` §8 for the full reasoning.
