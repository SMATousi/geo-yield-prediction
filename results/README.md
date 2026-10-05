# Results

Result tables of the YieldSAT cluster suites, generated from W&B and the
cluster's prediction files. Earlier single-machine pilot results are in
[`yieldsat/`](yieldsat/README.md).

| File | Contents |
|---|---|
| [point_suite.md](point_suite.md) | Point model (`before_full`): CV10, LORO, LOYO × paper/strict policy × S2, S2+ADM × 3 seeds, plus our re-run of the paper's pixel LSTM (CV10), next to the paper's numbers |
| [image_v1_suite.md](image_v1_suite.md) | Image model v1 (`image_full`): same matrix, metrics on the valid cells of the image test tiles |
| [image_vs_point.md](image_vs_point.md) | Image v1 vs point model scored on **identical cells** (fair comparison) |
| [image_v2_suite.md](image_v2_suite.md) | Image model v2 (`image_v2`): + per-pixel time series, level head, full coverage; metrics on **all** cells, directly comparable with the paper |
| [image_v2_vs_point.md](image_v2_vs_point.md) | Image v2 vs point model on the same (all) cells |
| [image_v2_vs_v1.md](image_v2_vs_v1.md) | Image v2 vs image v1 on v1's tiled cells |
| [negative_r2_investigation.md](negative_r2_investigation.md) | Why per-fold R² is negative in LOYO/LORO: metric mismatch with the paper (pooled vs per-fold), training and image-model causes, candidate solutions |
| [dev_r1.md](dev_r1.md), [dev_r2.md](dev_r2.md) | Improvement plan DEV rounds 1 (hybrid image models) and 2 (point-model variants: early fusion, season level, neighbourhood) |
| [pretrain_dev.md](pretrain_dev.md) | Knowledge pretraining DEV phase 1 (`pk_dev1r`, stopped at 84% of fine-tuning): P/I/E success criteria, paired fold-bootstrap comparisons of A0/A2/A3/A6/A7 |
| [tabm_dev.md](tabm_dev.md) | TabM DEV round TM-1: TabM F0/F1, LightGBM, MLP vs the point model, p3-nbr and the paper's best; paired fold-bootstrap Δ |


**How to read them**
- **All R² and RMSE values are pooled out-of-fold scores, the paper's
  computation.** For each experiment (pair × protocol × policy × inputs ×
  seed), the held-out predictions of all folds are pooled (every cell is held
  out exactly once) and R² = 1 − SSE/SST and RMSE (t/ha) are computed once.
  Averages of per-fold R² are not reported. The reasons are in
  [negative_r2_investigation.md](negative_r2_investigation.md), the only file
  that shows per-fold values, as evidence.
- Pixel metrics count every 10 m cell once. Field metrics compare each field
  season's mean prediction with its mean target.
- Values are means ± std of the pooled scores over seeds. A seed counts only
  when all its folds are finished; "Seeds complete" shows how complete each
  row is.
- Protocols and policies are defined in
  [spec/yieldsat_paper_comparison.md](../spec/yieldsat_paper_comparison.md).
  The models are described in [spec/architecture.md](../spec/architecture.md)
  (point) and
  [spec/yieldsat-image-training.md](../spec/yieldsat-image-training.md)
  (image).

**Regenerate**
```bash
# on the PVC (cluster/nautilus/results_pod.yaml): pooled metrics per experiment
python yieldsat_pooled_metrics.py --root /data/YieldSAT/yieldsat_results/before_full --out pooled_before_full.json
python yieldsat_pooled_metrics.py --root /data/YieldSAT/yieldsat_results/image_full --out pooled_image_full.json
# then locally, with the plans' runs.jsonl:
python yieldsat_results_tables.py --pooled pooled_before_full.json --runs <plan>/runs.jsonl \
    --title "Point model results (before_full suite)" --retire_arg_farm_loro --out results/point_suite.md
python yieldsat_results_tables.py --pooled pooled_image_full.json --runs <plan>/runs.jsonl \
    --title "Image model v1 results (image_full suite)" --out results/image_v1_suite.md
# image vs point on identical cells (pooled over matched folds), on the PVC, then locally:
python yieldsat_image_compare.py --image-root /data/YieldSAT/yieldsat_results/image_full \
    --point-root /data/YieldSAT/yieldsat_results/before_full --out <dir>
python yieldsat_image_compare.py --from_folds <dir>/folds.json --out <dir> --md results/image_vs_point.md
```
