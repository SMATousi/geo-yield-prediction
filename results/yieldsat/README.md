# YieldSAT training results (2026-09-28)

These are the real-data runs of the `yieldsat_preprocessed_v1` point-mode
pipeline. Each run is one seed (seed 0) and the hyperparameters were not tuned.
Everything is **pilot evidence, not a benchmark**. Background, data audits and
caveats are in
[spec/yieldsat_data_contract.md §7](../../spec/yieldsat_data_contract.md#7-implementation-progress-log-2026-09-28).

All metrics are in **t/ha** and computed over **every held-out cell** of the test
partition. R² is the coefficient of determination.
- *Pixel* counts each 10 m cell once.
- *Field-balanced* averages the per-field RMSE, so each field season has equal
  weight.
- *Field level* compares each field season's mean prediction with its mean
  target.
- *Macro country* averages the per-country pixel metrics.

## Layout

```text
summary.csv                  one row per run: every parameter + headline metrics + resources
run_all.sh                   exact commands used to produce the runs
splits/<name>.json           split manifests (field seasons per partition, fingerprints, leakage-checked)
runs/<run>/params.json       all CLI arguments, streams, excluded sources, split label, label policy
runs/<run>/history.csv       per-epoch train loss, learning rate, validation RMSE/R² (val: ≤500 cells/field)
runs/<run>/test_metrics.json overall / per-country / per-crop / per-country-crop / macro metrics
runs/<run>/group_metrics.csv the same as a flat table
runs/<run>/report.json       full run report (history, objective routing, provenance, I/O, memory)
runs/<run>/maps/*.tif        predicted yield maps for 3 test fields (verified EPSG + affine)
```

Checkpoints (`checkpoint_best.pth`, `sensor_checkpoint*.pth`), per-cell predictions
(`test_predictions.npz`) and normalizers stay in
`$YIELDSAT_ARTIFACT_ROOT/runs/<run>/`, which is not versioned. To regenerate this
directory:
`python yieldsat_collect_results.py --runs_dir $YIELDSAT_ARTIFACT_ROOT/runs --splits_dir $YIELDSAT_ARTIFACT_ROOT/splits --out_dir results/yieldsat`.

## Results

| Run | Purpose | Test split (label) | Test cells / seasons | Pixel RMSE | Pixel R² | Bias | Field-bal. RMSE | Field RMSE | Field R² | Macro-country RMSE / R² |
|---|---|---|---|---|---|---|---|---|---|---|
| `pilot_germany_farm` | Germany only | farm-held-out | 137,917 / 36 | 1.81 | 0.40 | +1.27 | 1.71 | 1.74 | 0.40 | – |
| `pooled_farm` | 4 countries | farm-held-out | 2,811,333 / 425 | 1.65 | 0.66 | +0.05 | 1.49 | 1.08 | 0.79 | 1.64 / 0.48 |
| `pooled_block20` | 4 countries | geographic block (20 km) | 2,913,259 / 446 | 1.45 | 0.66 | −0.08 | 1.33 | 0.91 | 0.81 | 1.52 / 0.32 |
| `loco_uruguay` | train AR+BR+DE | leave Uruguay out | 2,177,206 / 572 | 1.51 | 0.08 | +0.74 | 1.49 | 1.05 | 0.04 | – |
| `uruguay_scratch_f0.1` | Uruguay, 10% labels | farm-held-out (Uruguay part of `pooled_farm_s0`) | 436,708 / 96 | 1.21 | 0.31 | +0.20 | 1.23 | 0.73 | 0.58 | – |
| `uruguay_pretrained_f0.1` | same + pretrained sensors | same | 436,708 / 96 | 1.22 | 0.29 | +0.12 | 1.21 | 0.76 | 0.54 | – |
| `uruguay_scratch_f1.0` | Uruguay, 100% labels | same | 436,708 / 96 | 1.20 | 0.32 | +0.36 | 1.21 | 0.71 | 0.61 | – |
| `uruguay_pretrained_f1.0` | same + pretrained sensors | same | 436,708 / 96 | 1.24 | 0.27 | +0.34 | 1.22 | 0.71 | 0.60 | – |
| `pretrain_pooled` | yield-free pretraining (no test) | trained on `pooled_farm_s0` train only | – | – | – | – | – | – | – | – |

For reference, the `pooled_farm` model scores 1.20 / 0.32 pixel and 0.65 / 0.67
field level on the same 96 Uruguay test seasons.

Per country and crop, see `runs/<run>/group_metrics.csv`. For example, in
`pooled_farm` the pixel R² is 0.65 for Argentine corn and 0.56 for Argentine
soybean. It is −0.14 for Argentine wheat and −0.33 for German wheat.

What these results show:
- The pooled models mostly learn crop and country yield levels. Within-crop
  skill varies strongly between crops.
- Transfer to an unseen country (Uruguay) keeps almost no within-country skill.
- Yield-free pretraining did not help Uruguay at either label budget.

## Parameters

Shared by every run:

| Group | Setting |
|---|---|
| Data | contract `yieldsat_preprocessed_v1`, point mode, cache backend, streams `yieldsat_s2` (T×12), `yieldsat_weather` (T×4), `yieldsat_dem` (1), `yieldsat_terrain` (4), `yieldsat_soil` (48) |
| Inputs excluded | soil uncertainty, `coord_x/y/z`, row/col/field/farm IDs, harvest date as input, supplied `stats-*`, field ground-truth attributes |
| Cutoff | `before_harvest`, 30 days (pre-harvest prediction); weather at each row's first dated slot masked |
| Normalization | train-fields-only, pooled over countries (`norm_pooling=pooled`); target standardized, metrics in t/ha |
| Context | crop token on (`no_crop_context=False`); aspect raw (`aspect_encoding=raw`) |
| Model | `embed_dim=128`, `num_latents=8`, fusion `depth=2`, `num_heads=4`, `modality_embed=32`, `modality_dropout=0.1`; ~1.13 M parameters |
| Optimizer | AdamW, `lr=1e-3`, `weight_decay=0.05`, 1 warmup epoch then cosine, grad-clip 1.0, bf16 autocast |
| Sampling | `batch_size=512`, field-balanced season draws (`field_alpha=0.5`), `block_size=1` |
| Selection / eval | best epoch by validation pixel RMSE (≤500 cells per validation field); test on all cells (`test_rows_per_field=None`) |
| Other | `seed=0`, `num_workers=6`, CUDA (RTX 3090) |

Differing per run:

| Run | Countries | Split manifest | Mode | Train fraction | Init | Epochs × steps | Best epoch | Train / val / test seasons | Final train loss | Wall time | Samples/s | Peak GPU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `pilot_germany_farm` | DE | `germany_farm_s0` | finetune | 1.0 | – | 15 × 300 | 1 | 220 / 43 / 36 | 0.249 | 195 s | 12.6 k | 0.39 GB |
| `pooled_farm` | AR BR DE UY | `pooled_farm_s0` | finetune | 1.0 | – | 20 × 500 | 11 | 1,472 / 276 / 425 | 0.169 | 413 s | 14.7 k | 0.38 GB |
| `pooled_block20` | AR BR DE UY | `pooled_block20_s0` | finetune | 1.0 | – | 20 × 500 | 9 | 1,412 / 315 / 446 | 0.181 | 353 s | 17.6 k | 0.39 GB |
| `loco_uruguay` | AR BR DE UY | `loco_uruguay_s0` | finetune | 1.0 | – | 20 × 500 | 6 | 1,408 / 193 / 572 | 0.194 | 411 s | 14.7 k | 0.38 GB |
| `pretrain_pooled` | AR BR DE UY | `pooled_farm_s0` (train only) | pretrain | 1.0 | – | 20 × 500 | – | 1,472 / – / – | 0.154 | 478 s | 10.8 k | 0.58 GB |
| `uruguay_scratch_f0.1` | UY | `pooled_farm_s0` | finetune | 0.1 | – | 20 × 500 | 6 | 42 / 59 / 96 | 0.357 | 311 s | 17.8 k | 0.38 GB |
| `uruguay_pretrained_f0.1` | UY | `pooled_farm_s0` | finetune | 0.1 | `pretrain_pooled` | 20 × 500 | 0 | 42 / 59 / 96 | 0.320 | 311 s | 17.8 k | 0.38 GB |
| `uruguay_scratch_f1.0` | UY | `pooled_farm_s0` | finetune | 1.0 | – | 20 × 500 | 0 | 417 / 59 / 96 | 0.452 | 313 s | 17.6 k | 0.38 GB |
| `uruguay_pretrained_f1.0` | UY | `pooled_farm_s0` | finetune | 1.0 | `pretrain_pooled` | 20 × 500 | 5 | 417 / 59 / 96 | 0.438 | 313 s | 17.6 k | 0.38 GB |

The pretraining objective is masked observation (30% of observed values hidden,
reconstructed in normalized input space) plus a forecast of each cell's last
valid optical observation with every temporal stream hidden from that slot on. It
uses no yield labels and no crop token. Training losses are normalized-target MSE
for fine-tuning and reconstruction/forecast MSE for pretraining. Throughput was
measured with up to three runs sharing the GPU.

## Caveats

- Single seed. Brazil's farm-held-out test partition is one farm cluster, and
  Uruguay's validation partition is one farm cluster, which makes early stopping
  noisy (best epochs 0–6).
- Slope, TWI, soil-uncertainty and coordinate semantics are unresolved.
  Same-year aliased field seasons are grouped in splits but not deduplicated.
- Not directly comparable with the YieldSAT paper's CV10/LOYO/LORO protocols.
