# Flat-feature baselines on the dense series

**Status (2026-10-14): implemented; running on the cluster.**

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
- `yieldsat_protocol_runs.py units --batch tab-dense` writes four unit files: GPU (TabM + MLP) and CPU-only
  (LightGBM), each split into big pairs (ARG-S, BRA-S, URG-S, ARG-C) and small pairs.
- **Jobs** (`cluster/nautilus/tab_dense_*_job.yaml`):

  | Job | Pods | Resources |
  |---|---|---|
  | gpu-big | 6 | 1 GPU, 6 CPU, 48 Gi (ARG-S peak ≈ 37 GB) |
  | gpu-small | 6 | 1 GPU, 4 CPU, 16 Gi |
  | lgbm-big | 4 | 8 CPU, 40 Gi, no GPU |
  | lgbm-small | 4 | 8 CPU, 12 Gi, no GPU |

  - LightGBM runs in CPU-only pods, so no GPU sits idle.
  - The LightGBM jobs are submitted after the GPU jobs have built the flat matrices.

## 4. Analysis

`results/paper_models_table.py` with `PMT_OURS=ours-p3nbr-dense-fc-rel10,ours-p3nbr-dense
PMT_PAPER=tabm-f0-dense,mlp-f0-dense,lgbm-f0-dense,paper_mmgf PMT_BEST=ours-p3nbr-dense-fc-rel10` → pooled and
within-field R², ours vs the best flat-feature model (paired field-season bootstrap).
