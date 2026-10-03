# YieldSAT: beating the paper's best models (improvement plan)

Started 2026-10-02 (user request). Context: `results/` and
`results/negative_r2_investigation.md`. With the paper's metric (pooled
out-of-fold R²), our point model ties the paper's LSTM but is 0.05–0.2 R²
below the paper's best models (3D-ConvLSTM/3D-LSTM spatial models and AFF
fusion). Image v1/v2 are below the point model.

**Goal:** beat the paper's best model by a good margin on a fast development
subset, then run the winning setting on the full dataset.

## 1. Development subset (DEV)

To avoid running everything for each idea, every candidate is evaluated on a
fixed subset of country-crop pairs:

| Pair | Country / crop | Why | CV10 / LOYO / LORO folds |
|---|---|---|---|
| ARG-W | Argentina, wheat | Largest LOYO gap (ours 0.46 vs best 0.72); province LORO | 10 / 7 / 6 (provinces) |
| BRA-C | Brazil, corn | Corn; weak LOYO/LORO | 10 / 6 / 7 (farms) |
| GER-R | Germany, rapeseed | Small pair; worst LORO (−0.25 vs 0.15) | 10 / 7 / 6 (farms) |
| URG-S | Uruguay, soybean | Soybean (the main crop); 10 farm regions | 10 / 5 / 10 (farms) |

- Every country and four crops; small to medium pairs (283–2,504 tiles).
- **Matrix:** paper policy, S2+ADM (the paper's best rows are S2+ADM),
  seed 0 → **94 runs per candidate**. The fold manifests are those of the
  point suite.
- **Metric:** pooled out-of-fold pixel and field R²/RMSE (the paper's
  computation), from `yieldsat_pooled_metrics.py`.
- **Reference values** (pooled pixel R², S2+ADM):

  | Pair | CV10 ours / best | LOYO ours / best | LORO ours / best |
  |---|---|---|---|
  | ARG-W | 0.74 / 0.84 | 0.46 / 0.72 | 0.57 / 0.78 |
  | BRA-C | 0.43 / 0.46 | 0.12 / 0.29 | 0.22 / 0.37 |
  | GER-R | 0.38 / 0.49 | 0.09 / 0.31 | −0.25 / 0.15 |
  | URG-S | 0.38 / 0.43 | 0.27 / 0.32 | 0.31 / 0.36 |

  The DEV mean is 0.31 for our point model vs 0.46 for the paper's best.

**Success criterion ("good margin"):**
- the DEV mean pooled pixel R² is ≥ the paper-best DEV mean + 0.03, and the
  same at field level;
- no protocol's DEV mean is below the paper's best for that protocol.

Only a candidate that meets it moves to the full dataset (all 9 pairs ×
CV10/LOYO/LORO × paper/strict × S2/S2+ADM × 3 seeds).

## 2. Solutions (user selection: S4, S5, S6, S7, S9, S10)

All solutions are options of one new **hybrid model**
(`models_yieldsat_hybrid.py`; `main_yieldsat_image.py --arch hybrid`). It
predicts every cell of a 64×64 tile (full-coverage build, every point cell)
and combines a **per-cell point branch** with optional **spatial context**,
so each solution is a switch and can be ablated:

| ID | Solution | Implementation (flag) |
|---|---|---|
| S4 | **Early fusion of ADM** (as the paper's input-fusion LSTM) | Per cell and per time step, the S2 bands, weather, and static layers (DEM, terrain, soil, broadcast over time) are concatenated with masks and date features. A temporal transformer encodes this one fused sequence. `--cell_fusion early` (vs `s2` = S2 series only) |
| S5 | **Year/region level term** | Prediction = tile level + cell residual. The level head reads only season-wide inputs (the tile's weather series, crop, seeding day of year). An auxiliary loss ties it to the tile's mean target. `--level_head` |
| S6 | **Local spatial context** | Three masked 3×3 conv layers over the map of per-cell embeddings (7×7-cell neighbourhood of all modalities). `--context local` |
| S10 | **S2 and elevation as 64×64 maps, everything else as points** | A tile-level image branch over S2 (cached DINO RGB tokens + nine-band maps of the K = 4 observations) and the DEM map (U-Net, 64×64 output). Its features are gathered at each cell and fused with the cell's point embedding (weather, terrain, soil, full S2 series). `--context maps` |
| S9 | **Cross-attention between the elevation map and the RGB image** | In the S10 image branch, DEM tokens (8×8) attend to the RGB/DINO tokens and vice versa, with residual updates, before decoding. `--dem_rgb_xattn` |
| S7 | **Augmentation** | Random flips and 90° rotations of the tile (all maps, target and masks together; the 4×4 DINO token grid is permuted accordingly), training only. `--augment` |

**Training** (shared):
- every tile of the training seasons; loss = MSE over valid cells,
  cell-weighted like the point model, + λ · level loss with S5;
- AdamW with cosine schedule; best epoch on the validation seasons (all
  cells, pooled);
- predictions are written in the point format, so the pooled tools apply
  unchanged.

## 3. Experiment plan

| Step | Candidates on DEV | Question |
|---|---|---|
| E1 | hybrid `--cell_fusion early` | Does early fusion (S4) beat our late-fusion point model? |
| E2 | E1 + `--level_head` | S5 |
| E3 | E2 + `--context local` | S6 |
| E4 | E2 + `--context maps` | S10 |
| E5 | E4 + `--dem_rgb_xattn` | S9 |
| E6 | best of E3–E5 + `--augment` | S7 |

E1–E6 run together as one DEV suite (6 × 94 runs). The next round combines
the winners and tunes them. Each step's results and decisions are logged
below.

## 4. Progress log

- 2026-10-02 — Plan written. DEV subset and success criterion fixed.
- 2026-10-02 — **Implementation** (commit `f5b7cd0`): `models_yieldsat_hybrid.py`
  (`CellEncoder` S4, `LevelHead` S5, `LocalContext` S6, map branch S10 =
  `YieldSATImageModel` restricted to DINO + nine bands + DEM, via
  `adm_streams` and `features()`, with S9 `dem_rgb_xattn`); `augment_tile`
  (S7: flips/rotations of every map, target, masks, grid row/col and the DINO
  token grid; aspect sin/cos rotated with the map); `main_yieldsat_image.py
  --arch hybrid --cell_fusion --context --level_head --dem_rgb_xattn
  --augment`.
  - Tests: 31 image + 36 point tests pass. They cover each variant's
    gradients, absent cells never predicted, augmentation identity and
    aspect rotation, and hybrid runs end to end. Image-model defaults are
    unchanged (same parameters and outputs).
  - Small spec deviation: S9's DEM tokens are on the 4×4 token grid (the
    same grid as the DINO tokens), not 8×8.
- 2026-10-02 — **Local calibration** (GER-R CV10 fold 0, S2+ADM, RTX 3090;
  test = the same 31,880 cells as the point run):

  | Setting | Test pixel R² | Pixel RMSE | Field R² | Wall |
  |---|---|---|---|---|
  | Point model (`before_full` recipe) | 0.226 | 1.485 | 0.440 | – |
  | Hybrid early fusion, batch 8, lr 5e-4, 40 ep | 0.026 | 1.665 | 0.097 | 3 min |
  | … batch 4, lr 1e-3, 40 ep | 0.242 | 1.469 | 0.598 | 6.5 min |
  | … **batch 8, lr 2e-3, 80 ep** | **0.293** | **1.419** | **0.649** | 10 min |
  | Maps + S9 + S5 + S7, batch 8, lr 5e-4, 8 ep (smoke) | 0.090 | 1.610 | 0.323 | 1 min |

  - The default image recipe (lr 5e-4, 60 epochs) under-trains the hybrid:
    validation R² was still rising, and the best epoch was 33 of 40.
  - With lr 2e-3 and 80 epochs, early fusion alone beats the point model on
    this fold (+0.07 pixel R², +0.21 field R²).
  - The DEV recipe is therefore batch 8 tiles, lr 2e-3, 80 epochs, with all
    tiles of the training seasons (`--train_min_valid 0`).
- 2026-10-02 — **Round 1 submitted** (`cluster/suites/dev_r1.yaml`,
  `cluster/nautilus/dev_r1_pools.yaml`; W&B project `yieldsat-cvpr27-improve`).
  - **Configurations:** h1-early (S4), h2-level (+S5), h3-local (+S6), h4-maps
    (+S10), h5-maps-xattn (+S9), h6a-local-aug (h3 + S7), h6b-maps-xattn-aug
    (h5 + S7).
  - **Size:** 7 × 94 = 658 runs, ~119 GPU-hours (≈ 4–5.5 h on 30 A10s);
    2 × 15 A10 pods (16 CPU, 48 GiB).
  - **Data:** `YieldSAT-Image-full` on the PVC; sparse per-job staging.
- 2026-10-02 — **DEV evaluator** `yieldsat_dev_eval.py`: per configuration
  and DEV row, pooled pixel and field R² next to the point model and the
  paper's best; DEV means, per-protocol means and the success test. Only
  complete experiments count.
  - Validated with the point model's own pooled results as a fake
    configuration: seed 0 gives 0.305 / 0.470 vs 0.309 / 0.472 over 3
    seeds.
  - **Paper-best DEV means (S2+ADM): pixel 0.460, field 0.686.** Targets
    with the 0.03 margin: pixel ≥ 0.490 and field ≥ 0.716, with each of
    CV10/LOYO/LORO at or above the paper's per-protocol means.
  - The paper-best per-protocol DEV means are CV10 0.56, LOYO 0.41,
    LORO 0.42 (pixel).
- 2026-10-02 — Final image v2 results committed (`results/image_v2_*.md`;
  324/324 experiments): v2 worse than point in 91/108 rows, marginally
  better than v1 (55/86 rows).
- 2026-10-03 — **Round 1, first attempt: CUDA out of memory.**
  - 42 runs failed within ~45 s, all with `torch.OutOfMemoryError`. 3–4
    concurrent hybrid runs per A10 do not fit: a run on dense tiles (BRA-C,
    URG-S) peaks at ~9 GB, vs ~3 GB on sparse GER-R tiles in the local test.
  - The pool's circuit breaker mistook these for GPU faults and retired 6
    pods.
  - **Fix:**
    - 2 runs per GPU (`runs_per_gpu = max_runs_per_gpu = 2`);
    - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set in
      `main_yieldsat_image.py`;
    - pools re-rendered as 2 × 15 A10 (12 CPU, 40 GiB, 4 loader workers per
      run) and resubmitted.
  - Finished runs keep their markers; failed runs are retried.
- 2026-10-03 — **Cluster blocked.** After the old round-1 jobs were deleted,
  the resubmission (and even the small CPU results pod) was refused by
  Nautilus's admission webhooks: "pods resources utilization is too low".
  - Likely cause: image/hybrid runs are loader-bound and keep A10s at only
    15–30% utilization.
  - State at the block (W&B): 98 of 658 round-1 runs finished (h1 16, h2 17,
    h3 14, h4 13, h5 13, h6a 14, h6b 11); the rest will run when jobs are
    accepted again.
  - Lesson: never delete running jobs before a replacement is accepted.
  - **Next:** make the hybrid pipeline GPU-bound so it uses what it
    requests, then resubmit when the flag clears.
- 2026-10-03 — **GPU data path** (`dataset/yieldsat_image_gpu.py`;
  `YieldSATImageDataset(gpu_prep=True)`; default for `--arch hybrid`,
  `--no_gpu_prep` to disable).
  - **Split of work:** a loader worker only reads a tile, picks the optical
    observations (skipped when no map branch is used), looks up DINO
    features and draws the augmentation; it ships raw arrays (S2, weather
    and static as float32, since some values exceed the float16 range;
    times as float16 days since seeding). `prepare_batch` builds every
    model input on the GPU with the CPU item's semantics, and
    `augment_batch` is the GPU twin of `augment_tile`.
  - **Equivalence on 288 real tiles** (4 countries; training + augmentation,
    evaluation, before-harvest cutoff):
    - identical masks, NaN patterns, slot choices, targets, grid row/col and
      DINO tokens;
    - values within 0.008, from the CPU path's float16 storage (the GPU path
      keeps float32).
    - A first attempt shipped float16 raw values; weather temperatures
      (~1.6e4) and rare S2 outliers overflowed. Caught by this check and
      fixed.
    - Unit test `test_gpu_prep_matches_cpu_items`.
  - **Throughput** (URG-S fold 0, hybrid h2, dev host, 1 run, uncompressed
    tiles as staged on the cluster):

    | Path | Tiles/s | GPU util |
    |---|---|---|
    | CPU path | 74 | 21% |
    | GPU path | 91 | 41% |
    | GPU path, observation selection skipped | **118** | **50%** |

  - **Pods:** 2 runs per A10, so ~100% GPU expected; requests sized to use:
    8 CPU, 32 GiB, 4 loader workers per run.
  - Runs from the first attempt (98 done) used the CPU path. The remaining
    runs use the GPU path; the inputs are equivalent within 0.008.
- 2026-10-03 — **Round 1 resubmitted.** A server-side dry run first
  confirmed that admission works again; then 2 × 15 A10 pods (8 CPU,
  32 GiB, 2 runs each) were submitted.
  - First runs succeed on the GPU path (GER-R runs 7–9 min; URG-S staging
    129–186 s).
  - Sampled GPU utilization: 70–100% on pods with 2 runs (45–53% while the
    second run starts), vs 15–30% before. GPU memory 8–9.5 GB for 2 runs.
- 2026-10-03 — **W&B storage (user decision: all three steps).** W&B was
  running out of space.
  - **Estimated usage** (25-artifact sample per group): image project model
    checkpoints ~113 GB, point project models ~20 GB, results ~6.5 GB.
  - **Where things were kept:** checkpoints existed only on W&B (except the
    image donors); results are on the PVC for every cluster run.
  - **(3) No more model uploads** (commit `34c0463`): cluster runs pass
    `--wandb_no_model_artifacts` (only the results artifact is uploaded), and
    `_finish` copies `checkpoint_best.pth` to the PVC results root. Applies
    to pods started after the commit.
  - **(2) Backup of the point checkpoints:** `yieldsat_wandb_backup.py`
    downloads every model artifact to
    `/data/YieldSAT/yieldsat_results/before_full_checkpoints/<rel_path>/<artifact>/`
    with a manifest and `--verify`; it resumes. Pod
    `cluster/nautilus/wandb_backup_pod.yaml` runs it. Tested on the smoke
    project: 43/43 artifacts verified. A first version put retried runs
    into one folder; now each artifact gets its own.
  - **(1) Deleting the image v1/v2 model artifacts** with
    `yieldsat_wandb_backup.py --delete --confirm yieldsat-cvpr27-image` runs
    after the backup is verified. The delete mode was tested on a
    throwaway project: model artifacts removed, results kept.
- 2026-10-03 — **Backup, attempt 1:** the PVC backup pod downloaded about
  1,400 of 3,090 point checkpoints, then was evicted ("ephemeral local
  storage usage exceeds … 10Gi"): W&B caches every download under
  `~/.cache/wandb`.
  - **Fix** (`bcb477c`): download with `skip_cache=True`, `WANDB_CACHE_DIR`
    on the PVC, 20 GiB disk.
  - The relaunch was refused: the low-utilization flag again blocks new
    pods. Round 1's running pods are unaffected (29 running).
  - **Fallback:** the backup continues to the dev machine's local disk
    (`/root/yieldsat_backups/before_full_checkpoints`, same tool, manifest
    and verify), to be copied to the PVC when the flag clears. The image
    model deletion runs only after this local backup verifies.
- 2026-10-03 — **W&B clean-up done.**
  - **Point checkpoints backed up locally:** 3,090/3,090 artifacts,
    22.20 GB, verified (0 bad files) in
    `/root/yieldsat_backups/before_full_checkpoints`. They remain on W&B
    too, and are to be copied to the PVC when the flag clears.
  - **Image v1/v2 models deleted:** all 5,070 model artifact versions in
    `yieldsat-cvpr27-image` (115.09 GB), 0 failures. Results artifacts and
    PVC results are kept.
- 2026-10-03 — **Round 1, interim (190/658 runs; W&B, since the PVC is
  unreachable while the flag is active).** Only 3 of 84 experiments are
  complete, too few for pooled DEV scores. Paired per-fold comparison with
  the point model on the same fold/inputs/seed (test pixel RMSE):

  | Config | Folds | Median ΔRMSE vs point | Better folds (pixel / field) |
  |---|---|---|---|
  | h1-early | 32 | +0.075 | 8 / 8 |
  | h2-level | 34 | +0.105 | 4 / 8 |
  | h3-local | 29 | +0.119 | 5 / 5 |
  | h4-maps | 26 | +0.098 | 6 / 6 |
  | h5-maps-xattn | 25 | +0.228 | 4 / 4 |
  | h6a-local-aug | 25 | +0.222 | 2 / 3 |
  | h6b-maps-xattn-aug | 19 | +0.096 | 2 / 5 |

  - Folds are mostly GER-R and URG-S so far. No configuration beats the
    point model; LOYO is the worst (median +0.07 to +0.34).
  - **The local calibration win was seed luck, not a GPU-path bug.** On
    GER-R CV10 fold 0, with otherwise identical settings: CPU path seed 0
    R² 0.293, CPU path seed 1 0.117, GPU path seed 0 0.223 (point model
    0.226). Single folds on small pairs cannot separate configurations.
  - **Likely structural cause:** a hybrid step sees only 8 tiles (≈ 8
    fields), while the point model's batches mix cells from many fields.
    The level/bias signal per step is therefore noisy, and best epochs vary
    widely (1–48).
- 2026-10-03 — **Round 1 results** (`results/dev_r1.md`): pooled, 657/658
  runs, scored from W&B results artifacts while the PVC was unreachable.

  | Config | DEV pixel R² | DEV field R² |
  |---|---|---|
  | h1-early (S4) | 0.212 | 0.310 |
  | h2-level (+S5) | 0.114 | 0.070 |
  | h3-local (+S6) | 0.179 | 0.251 |
  | h4-maps (+S10) | 0.159 | 0.213 |
  | h5-maps-xattn (+S9) | 0.125 | 0.150 |
  | h6a-local-aug (+S7; 11/12 rows) | 0.079 | 0.077 |
  | h6b-maps-xattn-aug (+S7) | 0.161 | 0.228 |
  | **Point model** | **0.309** | **0.472** |
  | **Paper best** | **0.460** | **0.686** |

  - **No configuration passes**, and all are below our point model. The
    point model wins every DEV row except GER-R LORO, where h3 (−0.19) and
    h6b (−0.10) are less negative than the point model (−0.25).
  - **The S5 level term hurts most** (field R² 0.07). A tile-level head
    trained on a few hundred tiles per fold mispredicts the level.
  - **Spatial context (S6/S10), cross-attention (S9) and augmentation (S7)
    do not compensate** for the weaker per-cell learning of tile-batched
    training.
  - **Conclusion:** the tile-batched hybrid is the wrong training regime.
    The strongest learner is the point model's field-balanced sampling of
    cells from many fields per step.
  - **Round 2 direction:** bring S4/S5/S6 into the point pipeline (early
    fusion, a season-level term, and neighbourhood features as extra
    per-cell streams). Train the spatial variants with many fields per step.
- 2026-10-03 — **Round 2(a) implemented** (user decision: point-model
  variants).
  - **S4** `--fusion early`: one `MaskedTemporalEncoder` (width 192, depth 3)
    over the per-slot concatenation of every temporal stream and every
    static stream (masked on undated slots), plus date features.
  - **S5** `--level_head`: a season-level scalar from the field's weather
    series and the crop, added to the prediction. The auxiliary loss ties it
    to the field-season mean target (`season_target`). Point batches mix
    many fields, so the term sees many seasons per step.
  - **S6** `--neighbourhood`: new temporal stream `yieldsat_s2_nbr`, the
    5×5 masked mean of the 12 S2 bands (centre excluded) per cache row,
    from `yieldsat_build_neighbourhood.py` → `<artifact_root>/neighbourhood/`
    (Germany locally: 12 s). Point jobs stage it with the cache;
    `cluster/nautilus/neighbourhood_build_job.yaml` builds it on the PVC.
  - **Checks:** 3 new tests (`tests/test_yieldsat_round2.py`): neighbourhood
    = brute force; level = prediction − head; static masked on undated
    slots. The existing fusions are unchanged (identical parameters and
    outputs); 36 point tests pass. Smoke runs of all variants on real
    GER-R data.
  - **Suite** `cluster/suites/dev_r2.yaml`: p1-early, p2-level, p3-nbr,
    p4-early-level, p5-early-nbr, p6-all; same DEV matrix; point-suite
    budget. 564 runs, ~215 GPU-hours (~9 h on 30 A10s).
  - **Pools** `cluster/nautilus/dev_r2_pools.yaml`: 2 × 15 A10 (7 CPU,
    16 GiB, 4 → 6 runs), the shape that ran at ~96% utilization in
    `before_full`.
