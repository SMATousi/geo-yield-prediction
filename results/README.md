# Results

Result tables of the YieldSAT cluster suites, generated from W&B and the
cluster's prediction files. Earlier single-machine pilot results are in
[`yieldsat/`](yieldsat/README.md).

| File | Contents |
|---|---|
| [point_suite.md](point_suite.md) | Point model (`before_full`): CV10, LORO, LOYO × paper/strict policy × S2, S2+ADM × 3 seeds, plus our re-run of the paper's pixel LSTM (CV10), next to the paper's numbers |
| [image_v1_suite.md](image_v1_suite.md) | Image model v1 (`image_full`): same matrix, metrics on the valid cells of the image test tiles |
| [image_vs_point.md](image_vs_point.md) | Image v1 vs point model scored on **identical cells** (fair comparison) |

Image model v2 (per-pixel time series, level head, full coverage; suite
`image_v2`) is running; its tables will be added here.

**How to read them**
- All metrics are on held-out test data, in t/ha; R² is the coefficient of
  determination (1 − SSE/SST).
- Pixel metrics count every 10 m cell once. Field metrics compare each field
  season's mean prediction with its mean target.
- Values are means ± std over finished folds × seeds. "Runs done" shows how
  complete each row is.
- Protocols and policies are defined in
  [spec/yieldsat_paper_comparison.md](../spec/yieldsat_paper_comparison.md).
  The models are described in [spec/architecture.md](../spec/architecture.md)
  (point) and
  [spec/yieldsat-image-training.md](../spec/yieldsat-image-training.md)
  (image).

**Regenerate**
```bash
python yieldsat_results_tables.py --suite before_full yieldsat-cvpr27 <plan>/runs.jsonl \
    --title "Point model results (before_full suite)" --retire_arg_farm_loro --out results/point_suite.md
python yieldsat_results_tables.py --suite image_full yieldsat-cvpr27-image <plan>/runs.jsonl \
    --title "Image model v1 results (image_full suite)" --out results/image_v1_suite.md
# on the PVC (cluster/nautilus/results_pod.yaml), then locally from folds.json:
python yieldsat_image_compare.py --image-root /data/YieldSAT/yieldsat_results/image_full \
    --point-root /data/YieldSAT/yieldsat_results/before_full --out <dir>
python yieldsat_image_compare.py --from_folds <dir>/folds.json --out <dir> --md results/image_vs_point.md
```
