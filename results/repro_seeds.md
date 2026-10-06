# Reproduction: IF-LSTM with 3 seeds, GER-R CV10

Generated 2026-10-06 (spec/yieldsat-paper-reproduction.md).
- **Model:** thesis IF-LSTM (2 × 128, MLP head), paper-like inputs
  (supplied `stats-*`, raw target, NaN → −1, first dated slot kept).
- **Training:** 50 epochs; the test fold is scored after every epoch
  (diagnostic only).
- **Columns:** fold mean of pixel / field R², averaged over seeds 0–2 (± std
  across seeds), for three epoch-selection rules. 118/120 runs (S2 seed 1:
  8/10 folds).

| Inputs | Validation-selected (honest) | Test-selected (optimistic) | Last epoch |
|---|---|---|---|
| S2 | 0.323 ± 0.009 / 0.628 ± 0.009 | 0.431 ± 0.015 / 0.772 ± 0.005 | 0.233 / 0.576 |
| S2 + DEM | 0.345 ± 0.006 / 0.621 ± 0.018 | **0.466 ± 0.004 / 0.795 ± 0.003** | 0.303 / 0.631 |
| S2 + weather + soil + DEM | 0.272 ± 0.016 / 0.543 ± 0.034 | 0.398 ± 0.012 / 0.713 ± 0.019 | 0.255 / 0.566 |
| All 120 bands (all streams + soil uncertainty + coordinates) | 0.276 ± 0.014 / 0.556 ± 0.030 | 0.369 ± 0.008 / 0.690 ± 0.008 | 0.204 / 0.536 |

Paper (LSTM): S2 **0.36 ± 0.14 / 0.62 ± 0.25**; S2+ADM input fusion
**0.47 ± 0.10 / 0.81 ± 0.09**.

## Reading

- **Seed noise is small:** ≤ 0.016 pixel and ≤ 0.034 field std across
  seeds. The differences between rows and selection rules are real. The
  earlier "±0.06 between runs" came from comparing differently configured
  runs (early stopping vs not), not from seeds.
- **S2:** honest selection reproduces the paper's S2 row (0.32 vs 0.36,
  0.63 vs 0.62). Test selection overshoots it (0.43 / 0.77).
- **The paper's S2+ADM row (0.47 / 0.81) is matched only by S2 + DEM with
  test-selected epochs (0.466 / 0.795).** No configuration reaches it with
  honest selection: the best is S2 + DEM at 0.345 / 0.621.
- **Weather and soil hurt under every selection rule.** S2 + weather + soil
  + DEM is 0.05–0.07 below S2 + DEM. All 120 bands is no better.
