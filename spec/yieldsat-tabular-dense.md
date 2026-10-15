# Flat-feature baselines on the dense series

**Status (2026-10-15): done (§5 results).**

Related:
- [yieldsat-tabm.md](./yieldsat-tabm.md): the flat view (feature sets F0 / F1), TabM, MLP and LightGBM, run so far on
  the monthly release.
- [yieldsat-ablations-seeds.md](./yieldsat-ablations-seeds.md): every paper result uses the dense series.

## 1. Why

- **Question:** does a sequence model beat strong models on hand-made per-cell features? A reviewer will ask.
- The flat-feature baselines exist for all folds, but only on the monthly release (24 slots). The paper uses the
  dense series throughout (project lead, 2026-10-13), so they must be re-run on it.
- **On the monthly release** (9-pair means, pooled / within): TabM F0 0.510 / 0.268 (CV10), 0.297 / 0.199 (LORO).
  This is above our monthly base model within fields (0.248 CV10), so TabM is a serious baseline.
- F1 (with the neighbourhood stream) did not beat F0 on the monthly release (CV10 0.509 / 0.268 vs 0.510 / 0.268).
  Only F0 is run here; F1 on the dense series would need about 28 GB per matrix for ARG-S.

## 2. Runs

- **Models:** `tabm-f0-dense`, `mlp-f0-dense`, `lgbm-f0-dense`, with the monthly runs' arguments (`TAB` in
  `yieldsat_protocol_runs.py`). LightGBM uses 8 threads.
- **Features (F0, dense):** 72 slots × (12 S2 bands + 4 weather + 3 date features) and the static values (DEM, aspect
  sin / cos, curvature, slope, TWI, 48 soil), plus validity masks. About 1,420 values per cell. No neighbourhood, no
  field context.
- **Protocol:** the paper protocol (noval splits, selection on the test fold), 9 pairs × CV10 / LOYO / LORO, all
  217 folds per model, seed 0.
- **Flat matrices:** built once per pair on the PVC, `<artifact_root>/tabular/<Country>_<crop>_F0_cache_dense.npz`;
  the monthly matrices are untouched.

## 3. Implementation

- `standardize_stats` and `standardize` (`dataset/yieldsat_tabular.py`) are chunked:
  - stats in two float64 passes over the training rows, without copying them;
  - one float32 output copy.
  - Before, ARG-S on the dense series would have needed about 70 GB of RAM. Same result as nanmean / nanstd (test
    `tests/test_yieldsat_tabm.py::test_chunked_standardize_matches_nan_reference`).
- `train_lgbm` builds feature rows on demand (capped training rows, test rows in chunks of 50k) instead of a full
  concatenated copy of values and masks (~22 GB for ARG-S; the first LightGBM pods were OOM-killed at 40 Gi,
  2026-10-14). Same predictions (test `test_lgbm_rows_on_demand_matches_full_matrix`).
- `yieldsat_protocol_runs.py units --batch tab-dense` writes four unit files: GPU (TabM + MLP) and CPU-only
  (LightGBM), each split into big pairs (ARG-S, BRA-S, URG-S, ARG-C) and small pairs.
- **Jobs** (`cluster/nautilus/tab_dense_*_job.yaml`):

  | Job | Pods | Resources |
  |---|---|---|
  | gpu-big | 6 | 1 GPU, 6 CPU, 48 Gi (ARG-S peak ≈ 37 GB) |
  | gpu-small | 6 | 1 GPU, 4 CPU, 16 Gi |
  | lgbm-big | 4 | 8 CPU, 40 Gi, no GPU |
  | lgbm-small | 4 | 8 CPU, 16 Gi, no GPU (12 Gi OOM-killed on BRA-W) |

  - LightGBM runs in CPU-only pods, so no GPU sits idle.
  - The LightGBM jobs are submitted after the GPU jobs have built the flat matrices.

## 3b. TabM batch size (2026-10-14)

- At batch 4096, TabM on the dense F0 features runs out of GPU memory in the backward pass on a 24 GB GPU: it
  needs about 25.5 GB (BRA-W, both attempts).
- TabM runs instead as `tabm-f0-dense-bs2048`: batch 2048, 6,000 steps, evaluation every 200 steps. The same
  number of training cells and the same early-stopping horizon in cells as the monthly runs.
- Batch: `yieldsat_protocol_runs.py units --batch tab-dense-tabm`; jobs
  `cluster/nautilus/tab_dense_tabm_{big,small}_job.yaml`.
- The superseded `tabm-f0-dense` units are held (claim marker `held: superseded`). Any that finish on a larger GPU
  write under their own tag and are not mixed with the batch-2048 runs.
- The analysis uses `tabm-f0-dense-bs2048`.

## 4. Analysis

`results/paper_models_table.py` with `PMT_OURS=ours-p3nbr-dense-fc-rel10,ours-p3nbr-dense
PMT_PAPER=tabm-f0-dense,mlp-f0-dense,lgbm-f0-dense,paper_mmgf PMT_BEST=ours-p3nbr-dense-fc-rel10` → pooled and
within-field R², ours vs the best flat-feature model (paired field-season bootstrap).

## 5. Results (2026-10-15)

- **Data:** `results/tabular_dense.json` (`results/paper_models_table.py`, the env in §4).
- **Runs:** all 217 folds of each model (`tabm-f0-dense-bs2048`, `mlp-f0-dense`, `lgbm-f0-dense`) complete.
- **Test pixels:** identical to ours and MMGF.

Mean over the 9 pairs, pooled / within-field R²:

| Protocol | Ours | Ours (base) | TabM | MLP | LightGBM | MMGF |
|---|---|---|---|---|---|---|
| CV10 | **0.542 / 0.292** | 0.540 / 0.276 | 0.523 / 0.282 | 0.509 / 0.256 | 0.520 / 0.264 | 0.506 / 0.257 |
| LOYO | **0.370 / 0.219** | 0.356 / 0.195 | 0.253 / 0.189 | 0.237 / 0.147 | 0.287 / 0.139 | 0.320 / 0.203 |
| LORO | 0.368 / **0.236** | **0.371** / 0.208 | 0.303 / 0.214 | 0.292 / 0.186 | 0.307 / 0.170 | 0.288 / 0.207 |

**Ours vs the best baseline of each pair** (among TabM, MLP, LightGBM and MMGF, chosen per pair on the same test data;
paired field-season bootstrap). Pairs significantly better / worse:

| Protocol | best baseline by within-field (pairs) | within better / worse | pooled better / worse |
|---|---|---|---|
| CV10 | TabM 8, LightGBM 1 | 4 / 0 | 4 / 0 |
| LOYO | MMGF 6, TabM 3 | 5 / 0 | 4 / 1 |
| LORO | TabM 5, MMGF 3, MLP 1 | 5 / 0 | 6 / 0 |

**Reading:**
1. **TabM is the strongest baseline in distribution.**
   - On CV10 it beats MMGF on both metrics (0.523 / 0.282 vs 0.506 / 0.257) and comes within 0.010 of our
     within-field R² (0.292).
   - It is the best baseline in 8 of 9 pairs. Ours is still significantly better within fields in 4 of them and never
     worse.
2. **Out of distribution the flat-feature models fall behind the sequence models.**
   - LOYO pooled: TabM 0.253 and LightGBM 0.287, against MMGF 0.320 and ours 0.370.
   - LORO: TabM within 0.214, against ours 0.236.
   - Hand-made per-cell features generalise worse across years and regions.
3. **Ours is never significantly worse within fields** than the best baseline of any pair, in any protocol. Pooled, it
   is worse in 1 LOYO pair (by within criterion) and 2 (by pooled criterion).
4. **The dense series helps the flat models too:** TabM on CV10 goes from 0.510 / 0.268 (monthly) to 0.523 / 0.282
   (dense).
