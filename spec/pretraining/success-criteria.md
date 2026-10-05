# Success criteria: is pretraining helping, and is the knowledge doing its job?

**Status:** defined 2026-10-04 (user request); revised 2026-10-04 after the
PK-07 smoke evidence (see "Revision log" at the end). Applies to the point-model
pretraining in [../yieldsat-point-knowledge-pretraining.md](../yieldsat-point-knowledge-pretraining.md)
and later to the pooled foundation model and the US corpus stages. Arms
(A0, A2–A7) are defined in that spec. All held-out measurements use seasons
that were **excluded from the pretraining unit** (its DEV test seasons),
never pretraining or fine-tuning data.

The criteria answer three separate questions, in order. A "yes" to a later
question requires "yes" to the earlier ones.

1. **Is pretraining healthy?** (P) — it trains, does not collapse and covers
   the knowledge it is meant to learn.
2. **Does the model learn the knowledge?** (I) — intrinsic, on held-out
   seasons, without yield.
3. **Does it help yield prediction?** (E) — extrinsic, the DEV pooled
   metric, paired against matched controls.

## P — Pretraining health (every run; automatic)

| ID | Criterion | Pass |
|---|---|---|
| P1 | Convergence | Total, SSL, grounding and relation losses decrease; the last-epoch validation loss is within 5% of its minimum (no divergence) |
| P2 | No representation collapse | For every stream, the per-dimension variance of the summary embeddings over a held-out sample stays ≥ 0.5× its value at initialization, and the effective rank (exp of the entropy of normalized singular values) stays ≥ 0.5 × min(the same stream's effective rank at initialization, the stream's input width: S2 12, weather 4, DEM 1, terrain 4, soil 48). (Relative and capped: random initialization spreads a 4-channel stream over ~15 dimensions, and learning down to its true width is compression, not collapse.) |
| P3 | Knowledge coverage | Each non-abstaining concept has known targets for ≥ 20% of training field seasons. Each active rule (r01–r04, r06; r05 abstains by design) has ≥ 200 applicable training field seasons per pretraining unit. Concepts or rules below this are reported and excluded from I/E claims |
| P4 | Estimator sanity | Each concept's raw index has non-degenerate spread within its strata (IQR > 0) and < 50% missing among rows with the source stream present |

## I — Intrinsic: does the model learn the knowledge? (held-out seasons)

| ID | Criterion | Pass |
|---|---|---|
| I1 | **Concept grounding** | For each concept, AUROC of the model's concept score against the estimator target binarized at 0.5 is ≥ 0.70 on held-out seasons, and ≥ the `random_targets` control (A7) + 0.10. Reported per concept and per country. (Not against the text-shuffled A4: any distinct frozen prototypes are equally learnable, so A4 grounds as well as A3: measured 0.75–0.99 vs 0.82–0.98, excluding `low_elevation_position` at 0.58 in both.) |
| I2 | **Relational alignment** | On held-out applicable pairs of each active rule, mean `cos(normalize(z_B − z_A), r_AB)` is higher for A3 than for A7 (`random_targets`), paired by field season, bootstrap 95% CI above 0. The absolute value is not a criterion: a constant projector offset satisfies it (A7 measured 0.79–1.00 after 300 steps) |
| I3 | **No information loss** | Linear probes from frozen sensor embeddings to physical properties (mid-season precipitation and temperature, peak NDVI, NDMI, clay, SOC, relative elevation; ridge regression fitted on training seasons and scored on held-out seasons) reach R² for A3 ≥ R² for A2 − 0.02 on every property |
| I4 | **Sensor preservation** | The masked-observation and forecast losses of A3 on held-out seasons are ≤ 1.05× those of A2 |
| I5 | **Knowledge specificity** | The I1 gain over A7 is larger for concepts in an active rule than for concepts whose rules abstain (r05: `pole_facing_aspect`, `cool_regime`). Supporting evidence that the relational structure, not just more supervision, is learned |

## E — Extrinsic: does it help yield prediction? (DEV, pooled metric)

Paired comparisons use the same DEV rows (pair × protocol), folds, seed and
cells. "Δ" is the DEV-mean pooled R² difference. Confidence intervals come
from a paired bootstrap over DEV folds.

| ID | Criterion | Pass |
|---|---|---|
| E1 | **Pretraining helps** | A2 (SSL) ≥ A0 (from scratch) on DEV pixel and field R²: Δ ≥ 0, with no protocol worse by > 0.02 |
| E2 | **Knowledge helps beyond SSL and controls** | A3 > A2, A6 and A7, each with Δ ≥ +0.01 pixel R² and the 95% CI of Δ excluding 0; field R² not worse |
| E2b | **The language prior matters** (separate claim) | A3 > A4 (`shuffled` text) and A3 > A5 (`notext`) with the E2 margins. Failing E2b with E2 passing means the knowledge helps through the estimator targets and rules, not through the CLIP text |
| E3 | **Label efficiency** | At 10% and 25% of training labels, the A3 − A0 gain is ≥ the gain at 100% (pretraining matters more when labels are scarce) and positive |
| E4 | **Out-of-distribution benefit** | The A3 − A2 gain on LOYO and LORO is ≥ its gain on CV10 (knowledge should help most under year and region shift) |
| E5 | **No harm** | No DEV row is worse than A2 by more than 0.05 pixel R² |
| E6 | **Toward the paper** | The selected arm moves the DEV mean toward the paper's best; the final bar is the improvement plan's criterion (paper best + 0.03), checked on the full dataset |

## Decision rules

- **"Pretraining is healthy"** — P1–P4 pass for every pretraining unit of an
  arm. Otherwise the arm is debugged, not evaluated.
- **"The knowledge is doing its job"** — I1, I2, I4 pass, plus E2 against A7
  (`random_targets`). With only intrinsic success the knowledge is learned but
  not useful; with only E2 against A6 the benefit may be regularization or
  extra supervision of any kind.
- **"Language matters"** — E2b. Reported separately; not required for the
  knowledge to be useful.
- **"Pretraining is helping"** — E1 and E2 pass, plus E3 or E4, plus E5.
- **Scale-up gate for the US corpus** (stages 250k → 1M → 5M → 20M): a stage
  is kept only if, against the previous stage, E2-style Δ on the DEV mean is
  ≥ +0.005 pixel R² with a CI excluding 0, and I1/I3 do not regress.
  Otherwise scaling stops and the result is reported.

## Reporting

Every evaluation writes `results/pretrain_<round>.md` with:
- P/I/E tables per arm;
- the decision-rule verdicts;
- per-concept and per-rule coverage;
- the arms' compute (GPU-hours, steps).

Failed criteria are reported as failures, not omitted.

## Applied: point knowledge pretraining, DEV phase 1 (2026-10-05)

Source: `results/pretrain_dev.md` (round stopped at 84% of fine-tuning).

| Question | Verdict | Evidence |
|---|---|---|
| Is pretraining healthy? | **Yes, apart from P1** | P2–P4 pass on all 46 units of every arm. P1 fails on 11–16 units per arm because validation loss is dominated by a few extreme-input cells (S2 up to 70σ), not divergence. A trimmed validation metric is recommended |
| Is the knowledge doing its job? | **Learned, but not useful** | I1 11/12 concepts (A3 0.81–0.99 vs A7 ≈ 0.5); I2 4/5 rules (CIs > 0); I4 pass. E2 vs A7 fails: +0.007 [−0.021, +0.059] pixel |
| Is pretraining helping? | **No** | E1 fails: A2 − A0 −0.003 [−0.038, +0.033]. E2 fails vs A2, A6, A7. E4 and E5 fail. E3 not run |
| US scale-up gate | **Not met** | Stage 0 (YieldSAT inputs) shows no gain, so US data is not added beyond the pilot on this evidence |

## Revision log

- **2026-10-04 (PK-07 smoke, 300 pretraining steps on unit `cv_k00`).**
  - The text-`shuffled` control reached the same held-out grounding AUROC
    (0.75–0.99, excluding `low_elevation_position` at 0.58 in both) and relational alignment (0.83–0.98) as A3 (0.82–0.98 /
    0.85–0.98). With learnable projectors, any set of distinct frozen
    prototypes is equally learnable, so A4 tests only the language prior.
  - A new control, `random_targets` (A7), gives every field season another
    season's targets and gates within the same country × crop. Grounding AUROC
    dropped to 0.45–0.66, so it separates real grounding from chance.
  - Relational alignment under A7 stayed at 0.79–1.00, so the absolute value is
    uninformative.
  - Changes:
    - I1, I2, I5 and E2 compare against A7;
    - the language claim moved to the new E2b (A3 vs A4/A5);
    - P2's effective-rank threshold became relative to initialization (the DEM
      stream is one scalar; the absolute 25%-of-width rule failed every arm,
      including SSL-only).
- **2026-10-04 (pk_dev1r, first 8 A2 units, full budget).**
  - **P2:** terrain's effective rank went from 15.6 at init to 5.0 while its
    variance grew 50–68×. Terrain has 4 input channels, so the rank threshold
    is now capped at the stream's input width. Every other stream passed
    unchanged.
  - **I-criteria aggregation:** units with tiny held-out sets (e.g. a
    3-season LORO farm) gave probe R² of −400. I1–I4 are now aggregated as
    the median over units with ≥ 10 held-out seasons.
  - **P1 caveat:** on units `cv_k00`/`cv_k03` the validation batches are
    fixed, yet the masked-observation loss swings 0.6–3.6. A few extreme-input
    validation cells (S2 up to 70σ) dominate the MSE, while held-out I4 is
    normal (0.115). P1 is reported as measured, with this caveat. A trimmed
    validation metric is proposed for the next round.
