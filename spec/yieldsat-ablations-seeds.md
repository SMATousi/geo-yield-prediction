# Ablations and seed variance of the best model

**Status (2026-10-13): A1 done (from existing runs, §A1 results); A2–A4 and the seeds submitted to the cluster.**

Related:
- [yieldsat-paper-models.md](./yieldsat-paper-models.md) §8: the main comparison (paper protocol, seed 0).
- [yieldsat-field-context.md](./yieldsat-field-context.md) §4: the field-context streams that A4 takes apart.
- [yieldsat-dense-series.md](./yieldsat-dense-series.md): the dense series that A1 evaluates.
- Paper: `sec/4_method.tex` describes the full model; the existing ablation table (`tables/supp_ablation.tex`) covers
  only the relation loss and the field context as a whole.

## 1. Why

- **Seeds:** every result so far is seed 0. Our within-field margin over MMGF is 0.02–0.04, so the seed-to-seed
  spread of both models must be known.
- **Ablations:** the paper ablates only the last two components (relation loss, field context). The earlier design
  choices (dense series, neighbourhood stream, the auxiliary inputs, the parts of the field context) are not ablated.

## 2. Reference model

`ours-p3nbr-dense-fc-rel10`: dense p3-nbr (72 slots, 5×5 neighbourhood stream) + field context + relation loss
(λ = 1), S2+ADM, the paper protocol (noval splits, selection on the test fold), seed 0. Ablations remove one
component from this model (leave-one-out), with everything else unchanged.

## 3. Experiments

| ID | Variant (tag suffix) | Change | Scope | Runs |
|---|---|---|---|---|
| A1 | dense vs monthly series | `ours-p3nbr-dense` vs `ours-p3nbr` (both without field context / relation loss) | 9 pairs × 3 protocols | existing |
| A2 | `-abl-nonbr` | no 5×5 neighbourhood stream (`--neighbourhood` removed) | DEV | 65 |
| A3 | `-abl-noweather` | weather stream removed | DEV | 65 |
| A3 | `-abl-nosoil` | soil stream removed | DEV | 65 |
| A3 | `-abl-noterrain` | DEM and terrain streams removed | DEV | 65 |
| A4 | `-abl-nofstats` | field S2 mean/std stream removed (`--fc_streams` s2_dev field_rel) | DEV | 65 |
| A4 | `-abl-nofdev` | within-field S2 anomaly stream removed (`--fc_streams` s2_field field_rel) | DEV | 65 |
| A4 | `-abl-nofpos` | within-field position stream removed (`--fc_streams` s2_field s2_dev) | DEV | 65 |
| S | `ours-p3nbr-dense-fc-rel10`, seeds 1, 2 | `--seed`; output `…_seed{1,2}` | 9 pairs × 3 protocols | 434 |
| S | `paper_mmgf`, seeds 1, 2 | as above | 9 pairs × 3 protocols | 434 |

- **DEV** = the 4 development pairs (ARG-W, BRA-C, GER-R, URG-S) × CV10 + LOYO, as in FC-04 and the existing
  ablation table. The reference runs are the FC-04 units of the best model.
- **Notes on A3 / A4:**
  - `-abl-noterrain` removes the raw DEM/terrain streams. The within-field terrain z-scores of the field context
    (`yieldsat_field_rel`) stay; they are ablated in A4 (`-abl-nofpos`).
  - The field-context streams are always computed from S2 and terrain inputs; removing a raw stream does not change
    them.
- **Seed** changes the weight initialisation, the batch sampling and the stream dropout. The splits are fixed
  manifests and do not depend on it.

## 4. Implementation

- `--fc_streams` (trainer) → `field_context_streams` (`YieldSATPointDataset`): keeps a non-empty subset of
  `yieldsat_s2_field`, `yieldsat_s2_dev`, `yieldsat_field_rel`; the others are removed from the layout. Test:
  `tests/test_field_context.py::test_field_context_stream_subset` (kept streams identical to the full context; the
  model trains on the subset; unknown names are refused).
- `yieldsat_protocol_runs.py units --batch ablation-dev` → `cluster/tabm/protocol_ablation_dev_units.json` (455).
- `yieldsat_protocol_runs.py units --batch seeds` → `protocol_seeds_ours_units.json` (434),
  `protocol_seeds_mmgf_units.json` (434) and `protocol_abl_seeds_ours_units.json` (889 = ablations first, then our
  seeds, one pool).
- Jobs: `cluster/nautilus/protocol_abl_seeds_ours_job.yaml` (40 GPUs × 4 runs, 32 Gi, as RX-2) and
  `protocol_seeds_mmgf_job.yaml` (24 GPUs × 6 runs, 12 CPU / 36 Gi, as RX-2), with the current bad-node exclusions.

## 5. Analysis

- **A1:** `results/paper_models_table.py` with `PMT_OURS=ours-p3nbr-dense,ours-p3nbr PMT_PAPER=ours-p3nbr
  PMT_BEST=ours-p3nbr-dense` → `results/ablation_a1_dense.json`.
- **A2–A4:** Δ pooled / within-field R² of each variant vs the reference, paired field-season bootstrap, per DEV pair
  and protocol and as the mean over the DEV pairs (`results/rel_cmp.py`, `REL_BASE=ours-p3nbr-dense-fc-rel10`).
- **Seeds:**
  - per model, protocol and seed: the 9-pair mean pooled / within-field R²; reported as mean ± sd over 3 seeds;
  - ours − MMGF per seed (does the ranking hold in every seed?);
  - paired bootstrap of the seed-averaged difference: each resample of field seasons is scored for all 3 seeds and
    the 3 differences are averaged;
  - the seed sd of a single cell is compared with the ours − MMGF margins.

**Success criteria:** none (descriptive). The paper reports the seed spread next to the main table and an extended
ablation table in the supplementary.

## A1 results (2026-10-13)

From the existing runs (all 217 folds of both models), base model without field context / relation loss.

| Protocol | Dense (72 slots): pooled / within | Monthly (24 slots): pooled / within | Δ within (mean) | Δ pooled (mean) | Within: better / worse | Pooled: better / worse | Field R² dense / monthly |
|---|---|---|---|---|---|---|---|
| CV10 | 0.540 / 0.276 | 0.511 / 0.248 | +0.028 | +0.029 | 8 / 0 | 5 / 0 | 0.836 / 0.800 |
| LOYO | 0.356 / 0.195 | 0.346 / 0.167 | +0.028 | +0.010 | 6 / 1 | 5 / 2 | 0.473 / 0.447 |
| LORO | 0.371 / 0.208 | 0.324 / 0.166 | +0.041 | +0.047 | 5 / 0 | 4 / 0 | 0.538 / 0.463 |

(better / worse: number of the 9 pairs with a significant difference, 95% field-season cluster bootstrap.)

**Reading:**
1. **The dense series is the largest single gain so far:** +0.028 / +0.028 / +0.041 within-field R² (CV10 / LOYO /
   LORO), larger than field context + relation loss together (+0.016 / +0.024 / +0.028).
2. **Pooled and field-level R² rise too**, most out of distribution (LORO pooled +0.047, field R² +0.075).
3. **Exceptions:** LOYO BRA-C (within −0.054, pooled −0.165) and LOYO GER-W (pooled −0.142, within +0.020). These are
   the two LOYO rows that regressed with the dense series (yieldsat-dense-series.md §9–10). The cause is a sparse
   early year. S2 observation dropout fixes GER-W LOYO but costs as much on other rows, so it was not adopted.
