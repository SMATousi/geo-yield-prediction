# The paper's training protocol for all models (no validation set)

**Status (2026-10-06): point-level batch complete (§6).** The image models (v1/v2, hybrids) are the second batch. Results
go in §6.

Related:
- [yieldsat-paper-reproduction.md](./yieldsat-paper-reproduction.md): H1,
  test-fold epoch selection.
- [yieldsat-tabm.md](./yieldsat-tabm.md).
- [yieldsat-improvement.md](./yieldsat-improvement.md).

## 1. What the paper did

Request (project lead, 2026-10-06): drop the validation set, train and test
as the paper did, and also train on the whole dataset and report per pair;
apply this to every model so far.

Reading of the paper (Pathak et al., CVPR 2026), the thesis and the release
tutorial:

| Question | Answer | Evidence |
|---|---|---|
| Separate validation set? | **No.** Model selection and early stopping used the held-out CV fold | Paper §5.3: "in the LOYO scenario, a model is colored by the year **in the validation set**". The held-out year is the test fold. Thesis §3.3: the held-out fold is "used **solely for testing**". Thesis A.1.2: early stopping "if validation performance does not improve for 8 epochs". Thesis §4.5.2: attention averaged "across all samples in the validation split over all CV folds". Our H1 runs reproduce the paper's S2+ADM row only with test-fold selection (0.466 / 0.795 vs 0.47 / 0.81) |
| One model per country–crop pair, or one model for everything? | **Per pair** | Thesis §5.2.3: "The model is trained and evaluated on **countries and crop types independently**, using the stratified-grouped K-Fold CV". Thesis §6: "The Models are **trained for each dataset separately** using 10-fold CV". Thesis Table 4.1 defines the datasets as country–crop subsets. Paper §5: "Results are reported on subsets per country and crop … which is how the data will be used in practice". The release ships one file per country, and the tutorial filters `TARGET_CROP` |

Decision (project lead, 2026-10-06):
- The **paper protocol (per pair, no validation set)** is the main table.
- **Pooled training** (one model for all 9 pairs, reported per pair) is an
  additional arm.
- Point-level models run first, image models after.
- **The paper LSTM is deferred** (project lead, 2026-10-06): our models are
  trained and compared with the paper's published rows first; the LSTM runs
  only if that comparison needs it (`units --with_lstm`).

Selecting on the test fold is optimistic by construction (H1: +0.08 to
+0.12 pixel R² on GER-R). Numbers from this protocol are comparable with the
paper's tables, not with a held-out estimate. Our earlier
validation-selected results stay the honest reference.

## 2. Protocol

- **Folds.** The paper folds we already use, with the same test folds:
  - stratified grouped CV10 (season grouping, paper policy);
  - LOYO;
  - LORO: provinces for Argentina, farms elsewhere (the point-suite
    convention).

  This gives 217 folds over the 9 pairs.
- **No validation set.** The 10% validation carve-out goes back into
  training. Every selection decision uses the full test fold (every cell):
  - the best epoch and early stopping for the neural models (patience 10,
    thesis A.1.2: 8–10);
  - the best evaluation step for TabM/MLP (patience 8 × 100 steps);
  - the boosting round for LightGBM (early stopping 100).
- **Manifests** (`yieldsat_protocol_runs.py folds`, `dataset/yieldsat_splits.py`):
  - `noval_<paper prefix>_fold<ii>`: train = train ∪ val, val = test =
    the paper test fold, `selection: test_fold`.
  - `pooled_noval_{cv10,loyo,loro}_s0_fold<ii>`, which hold out:
    - CV10: every pair's fold k (10 folds);
    - LOYO: one harvest year across all pairs (9 folds, 2016–2024);
    - LORO: one region across all pairs (30 folds; Argentine province
      spellings merged).

    Every season is tested exactly once, on the same test fold as in the
    per-pair arm.
- **Metric.** As in the paper, R² averaged over folds (pixel and field
  level). Pooled out-of-fold R² is reported next to it. Pooled models are
  scored per pair on that pair's cells of each test fold.

## 3. Models (point-level batch)

| Tag | Model | Inputs | Training |
|---|---|---|---|
| `lstm-s2` (deferred) | Paper input-fusion LSTM, thesis configuration (hidden 128, 2 layers, MLP head, supplied stats, raw t/ha target, Adam 1e-3, batch 1024, full passes) | S2 | ≤ 50 epochs, early stop 10 |
| `lstm-s2adm` (deferred) | same | S2 + ADM (weather, DEM, terrain, soil) | same |
| `ours-p3nbr` | our point model, best configuration (perceiver summary + 5×5 neighbourhood S2) | S2 + ADM + neighbourhood | ≤ 60 epochs × 500–1,500 steps, early stop 10 |
| `tabm-f1` | TabM k = 32, d_block 256, dropout 0.2, lr 1e-3 (adopted backbone) | F1 = F0 + 5×5 neighbourhood | ≤ 3,000 steps |
| `tabm-f0` | same | F0 (flat month-aligned features) | same |
| `mlp-f0` | MLP | F0 | same |
| `lgbm-f0` | LightGBM | F0 | ≤ 3,000 rounds, 3,000 cells per season |

**Pooled arm:**
- The LSTM has no pair context.
- Our point model uses its crop token.
- TabM, MLP and LightGBM get country and crop indicator columns.
- The flat models train on at most 1,000 cells per season, because the full
  12.4M-cell matrix does not fit an A10. They are tested on every cell.

Seed 0 throughout (the paper reports fold spread, not seeds).

## 4. Fix found while preparing this (affects earlier results)

`main_yieldsat_finetune.py` saved `checkpoint_best.pth` on
**non-improving** epochs from commit 81e9fdd (2026-10-05 18:53 UTC) until
this change. The test score in `report.json` therefore came from the last
non-improving epoch (with early stopping, best epoch + patience), not the
validation-best epoch.

Affected:
- The R1 ablation table in `results/repro_r1.md` (top table). It is biased
  low.

Not affected:
- H1, seeds and DEV repro numbers: computed from the per-epoch history,
  verified to equal the validation-best epoch's test score.
- pk_dev1r fine-tunes: finished before 81e9fdd.
- TabM, LightGBM and the image models: separate scripts.

The DEV repro results gathered when the job was stopped (97/252 units), with
validation-best epochs taken from the history:

| Row | Inputs | Validation-picked pixel / field R² | Test-picked pixel / field R² |
|---|---|---|---|
| ARG-W CV10 | S2 / S2+DEM / S2+WSD | 0.746 / 0.875; 0.666 / 0.790; 0.667 / 0.745 | 0.761 / 0.882; 0.734 / 0.855; 0.719 / 0.805 |
| URG-S CV10 | S2 / S2+DEM / S2+WSD | 0.354 / 0.681; 0.343 / 0.676; 0.326 / 0.652 | 0.366 / 0.687; 0.368 / 0.699; 0.339 / 0.669 |
| URG-S LOYO | S2 / S2+DEM / S2+WSD | 0.059 / 0.130; 0.080 / 0.182; −0.109 / −0.275 | 0.133 / 0.357; 0.100 / 0.244; −0.031 / −0.040 |
| URG-S LORO | S2 / S2+DEM / S2+WSD | 0.072; 0.096; 0.078 (pixel) | 0.182; 0.164; 0.156 (pixel) |

ARG-W is partial: 9, 9 and 4 of 10 folds.

## 5. Implementation and cluster plan

| Piece | Where |
|---|---|
| Fold manifests | `yieldsat_protocol_runs.py folds`; `noval_split`, `pooled_noval_splits` in `dataset/yieldsat_splits.py` |
| Neural trainers | `main_yieldsat_finetune.py`: `--val_rows_per_field 0` = full selection fold; `selection` recorded in `report.json` |
| Flat models | `main_yieldsat_tabm.py`: `--fold_set {dev,noval,pooled}`, `load_pooled` (indicator columns, training-cell cap), `--build_only` |
| Work units | `yieldsat_protocol_runs.py units` → `cluster/tabm/protocol_nn_units.json` (266: our point model × (217 + 49) folds; 798 with `--with_lstm`) and `protocol_tab_units.json` (432, including 14 cache builds) |
| Report | `yieldsat_protocol_runs.py report` → `results/protocol_point.md` (per pair: paper LSTM S2, LSTM S2+ADM, paper best, then every model in both arms) |
| Jobs | `cluster/nautilus/protocol_nn_job.yaml` (24 pods, staged to scratch); `protocol_tab_job.yaml` (16 pods, PVC). Any GPU with ≥ 24 GB and bf16 |
| Output | `/data/YieldSAT/yieldsat_results/protocol/paper/<group>/<inputs>/<tag>_seed0/<pair or ALL>/fold<ii>` |

Order:
1. Cache builds.
2. Per-pair units, largest pairs first.
3. Pooled units.

Estimate without the LSTM: ≈ 100 GPU-hours for our point model and ≈ 100
for the flat models (the original ≈ 600 included the pooled LSTM), i.e.
well under a day once most of the 40 pods are admitted.

## 6. Results (point-level batch complete, 2026-10-06)

All 1,330 folds have finished (217 per-pair and 49 pooled folds × 5
models). Full per-pair tables: `results/protocol_point.md`.

Scoring: as the paper's tables and the authors' release tutorial do, the
test predictions of all folds of a row are pooled and R² is computed once
(pooled out-of-fold R²). Field level is the R² of field-season means. The
paper text says "average across the folds", but 31 of its LOYO/LORO rows
cannot be fold means (§1 of `results/negative_r2_investigation.md`); in CV10
both conventions agree.

Each cell is pixel / field R². Columns:
- per-pair models (one model per pair, as in the paper);
- paper columns: the paper's published values ("Paper best" = the best
  paper model of that pair and protocol).

Bottom row: mean over the 9 pairs. Source: `results/protocol_point_oof.json`
(also `results/protocol_point_oof.md`).

#### CV10

| Pair | Paper LSTM S2+ADM | Paper best | Ours p3-nbr | TabM F1 | TabM F0 | MLP | LightGBM |
|---|---|---|---|---|---|---|---|
| ARG-C | 0.58 / 0.76 | 0.70 / 0.84 | 0.68 / 0.85 | 0.66 / 0.81 | 0.66 / 0.81 | 0.66 / 0.81 | 0.61 / 0.74 |
| ARG-S | 0.59 / 0.72 | 0.73 / 0.84 | 0.69 / 0.80 | 0.69 / 0.79 | 0.69 / 0.79 | 0.67 / 0.78 | 0.67 / 0.79 |
| ARG-W | 0.75 / 0.85 | 0.84 / 0.92 | 0.80 / 0.89 | 0.78 / 0.86 | 0.78 / 0.86 | 0.78 / 0.86 | 0.75 / 0.83 |
| BRA-C | 0.43 / 0.81 | 0.46 / 0.84 | 0.49 / 0.85 | 0.48 / 0.83 | 0.48 / 0.82 | 0.46 / 0.81 | 0.49 / 0.85 |
| BRA-S | 0.33 / 0.64 | 0.44 / 0.80 | 0.41 / 0.78 | 0.43 / 0.79 | 0.43 / 0.78 | 0.41 / 0.75 | 0.40 / 0.74 |
| BRA-W | 0.21 / 0.58 | 0.24 / 0.73 | 0.22 / 0.74 | 0.24 / 0.71 | 0.23 / 0.73 | 0.21 / 0.67 | 0.21 / 0.61 |
| GER-R | 0.47 / 0.81 | 0.49 / 0.82 | 0.46 / 0.78 | 0.47 / 0.79 | 0.48 / 0.80 | 0.46 / 0.78 | 0.43 / 0.70 |
| GER-W | 0.35 / 0.63 | 0.44 / 0.77 | 0.44 / 0.76 | 0.43 / 0.72 | 0.43 / 0.73 | 0.39 / 0.64 | 0.39 / 0.72 |
| URG-S | 0.39 / 0.72 | 0.43 / 0.81 | 0.41 / 0.76 | 0.42 / 0.77 | 0.41 / 0.78 | 0.40 / 0.77 | 0.41 / 0.75 |
| **Mean** | **0.46 / 0.72** | **0.53 / 0.82** | **0.51 / 0.80** | **0.51 / 0.79** | **0.51 / 0.79** | **0.49 / 0.76** | **0.48 / 0.75** |

#### LOYO

| Pair | Paper LSTM S2+ADM | Paper best | Ours p3-nbr | TabM F1 | TabM F0 | MLP | LightGBM |
|---|---|---|---|---|---|---|---|
| ARG-C | 0.44 / 0.42 | 0.48 / 0.61 | 0.48 / 0.58 | 0.39 / 0.43 | 0.39 / 0.47 | 0.38 / 0.47 | 0.39 / 0.44 |
| ARG-S | 0.49 / 0.62 | 0.66 / 0.77 | 0.58 / 0.69 | 0.57 / 0.65 | 0.56 / 0.63 | 0.51 / 0.58 | 0.54 / 0.62 |
| ARG-W | 0.61 / 0.74 | 0.72 / 0.81 | 0.65 / 0.76 | 0.63 / 0.75 | 0.64 / 0.75 | 0.57 / 0.70 | 0.49 / 0.58 |
| BRA-C | 0.19 / 0.29 | 0.29 / 0.45 | 0.28 / 0.43 | 0.11 / 0.09 | 0.04 / -0.09 | -0.06 / -0.28 | 0.08 / 0.13 |
| BRA-S | 0.17 / 0.26 | 0.29 / 0.45 | 0.23 / 0.30 | 0.30 / 0.43 | 0.29 / 0.45 | 0.29 / 0.43 | 0.19 / 0.27 |
| BRA-W | 0.09 / 0.12 | 0.11 / 0.14 | 0.04 / -0.10 | 0.07 / -0.10 | 0.06 / -0.16 | -0.01 / -0.44 | 0.13 / 0.25 |
| GER-R | 0.31 / 0.58 | 0.31 / 0.58 | 0.29 / 0.50 | 0.31 / 0.48 | 0.31 / 0.52 | 0.30 / 0.51 | 0.22 / 0.31 |
| GER-W | 0.18 / 0.33 | 0.22 / 0.46 | 0.22 / 0.27 | 0.15 / 0.10 | 0.15 / 0.12 | 0.13 / 0.14 | 0.17 / 0.21 |
| URG-S | 0.30 / 0.54 | 0.35 / 0.63 | 0.35 / 0.60 | 0.33 / 0.55 | 0.32 / 0.53 | 0.31 / 0.52 | 0.28 / 0.48 |
| **Mean** | **0.31 / 0.43** | **0.38 / 0.54** | **0.35 / 0.45** | **0.32 / 0.37** | **0.31 / 0.36** | **0.27 / 0.29** | **0.28 / 0.37** |

#### LORO

| Pair | Paper LSTM S2+ADM | Paper best | Ours p3-nbr | TabM F1 | TabM F0 | MLP | LightGBM |
|---|---|---|---|---|---|---|---|
| ARG-C | 0.48 / 0.43 | 0.54 / 0.68 | 0.50 / 0.60 | 0.38 / 0.48 | 0.43 / 0.56 | 0.40 / 0.56 | 0.37 / 0.45 |
| ARG-S | 0.54 / 0.66 | 0.65 / 0.78 | 0.57 / 0.67 | 0.56 / 0.64 | 0.56 / 0.63 | 0.52 / 0.59 | 0.57 / 0.68 |
| ARG-W | 0.53 / 0.61 | 0.78 / 0.87 | 0.67 / 0.79 | 0.61 / 0.70 | 0.63 / 0.72 | 0.57 / 0.67 | 0.58 / 0.67 |
| BRA-C | 0.28 / 0.38 | 0.37 / 0.65 | 0.29 / 0.51 | 0.15 / 0.13 | 0.11 / 0.04 | 0.17 / 0.22 | 0.14 / 0.25 |
| BRA-S | 0.12 / 0.29 | 0.35 / 0.63 | 0.26 / 0.46 | 0.29 / 0.49 | 0.30 / 0.52 | 0.30 / 0.54 | 0.25 / 0.47 |
| BRA-W | 0.11 / 0.26 | 0.20 / 0.52 | 0.17 / 0.48 | 0.14 / 0.37 | 0.13 / 0.31 | 0.10 / 0.24 | 0.14 / 0.37 |
| GER-R | 0.05 / 0.20 | 0.17 / 0.26 | 0.05 / 0.14 | 0.26 / 0.48 | 0.27 / 0.52 | 0.27 / 0.48 | 0.08 / 0.03 |
| GER-W | -0.36 / -1.39 | 0.10 / 0.14 | 0.04 / -0.13 | -0.10 / -0.55 | -0.09 / -0.55 | -0.14 / -0.62 | 0.10 / 0.11 |
| URG-S | 0.32 / 0.56 | 0.36 / 0.68 | 0.35 / 0.66 | 0.36 / 0.65 | 0.35 / 0.63 | 0.35 / 0.64 | 0.31 / 0.56 |
| **Mean** | **0.23 / 0.22** | **0.39 / 0.58** | **0.32 / 0.46** | **0.29 / 0.38** | **0.30 / 0.38** | **0.28 / 0.37** | **0.28 / 0.40** |

Findings:
1. **Against the paper's LSTM (S2+ADM).** Under the paper's protocol and
   metric, our point model (p3-nbr) beats it in every protocol:
   - on 8/9 pairs in CV10 and 9/9 pairs in LORO (pixel level);
   - by +0.04 pixel in LOYO and +0.09 pixel / +0.24 field in LORO (means).
2. **Against the paper's best model per row.** We are still below it on
   average:
   - CV10: −0.02 pixel, −0.02 field;
   - LOYO: −0.03 pixel, −0.09 field;
   - LORO: −0.07 pixel, −0.12 field.

   Our point model beats the best paper model on field-level CV10 for 3/9
   pairs.
3. **Our models.** The point model (p3-nbr) is the strongest of ours
   overall, especially out of distribution (LOYO/LORO). TabM ties it on
   CV10 pixel level.
4. **Pooled training** (one model for all pairs) was worse than per-pair
   training in every protocol when scored per fold (CV10 −0.03 pixel,
   −0.05 field). Its pooled-scoring numbers are not computed yet. This is
   consistent with the paper's per-pair training.
5. Every number is selected on the test fold (optimistic by construction).
   It is comparable with the paper's tables only.

### 6.1 Image batch (submitted 2026-10-06)

- **Models:** image v1, image v2 (+ series, level head) and the round-1
  hybrid h1-early.
  - Inputs S2+ADM, seed 0.
  - All 9 pairs × CV10/LOYO/LORO per pair, plus pooled CV10: 681 units
    (`cluster/tabm/protocol_image_units.json`,
    `cluster/nautilus/protocol_image_job.yaml`, 24 pods).
- **Tiles and test cells:** tiles come from the full-coverage build, and
  test cells are every cell (the paper's coverage).
  - v1 trains on the v1 tile set (≥ 2,048 valid cells).
  - German v1/v2 runs warm-start from the existing donors, which were
    trained on non-German tiles only.
- **Selection:** `main_yieldsat_image.py` selects on the test fold
  (`selection: test_fold`).
- **Pooled arm:** CV10 only, because pooled image training costs ~3–5 h
  per fold. Pooled LOYO/LORO run only if wanted.
- **Smoke (GER-R CV fold 0):** all three passed.

  | Model | Pixel R² | Field R² | Note |
  |---|---|---|---|
  | v1 | 0.02 | −0.16 | warm start, 336 tensors |
  | v2 | −0.03 | −0.41 | best at epoch 0 |
  | h1 | 0.32 | 0.72 | |

### 6.2 Pending

- Image batch results.
- The paper LSTM (deferred; run only if needed).

## 7. Log

- 2026-10-06:
  - Reading of the paper and thesis (§1).
  - The DEV repro job was stopped at 97/252 units on the project lead's
    request; its results were gathered first (§4).
  - The checkpoint-selection bug was fixed (§4).
  - Manifests (217 per-pair folds and 49 pooled folds) were built and copied
    to the PVC. The paper fold partitions on the PVC and locally are
    identical (601 manifests).
  - Smoke tests passed: TabM and LightGBM per pair, TabM pooled (2 pairs),
    LSTM per pair and pooled (all 9 pairs). In every case the test score
    equals the score at the test-selected epoch.
  - The paper LSTM was deferred (project lead): its 532 units were marked
    skipped on the PVC (`gave_up` = "skipped") so pods holding the old unit
    list never claim them; the running LSTM unit was stopped.
- 2026-10-06, overnight monitoring (watcher every 10 min):
  - `Argentina_soybean_F0.npz` was written corrupt on the PVC (right size,
    unreadable zip). The six ARG-S MLP/LightGBM units gave up on it.
    - Fixed: the file was deleted and rebuilt; the cache code now verifies an
      archive before publishing it and rebuilds an unreadable one.
  - The GPU on nrp-01.laccd.edu failed mid-run ("unspecified launch failure").
    Its pod burned 14 units of our point model in minutes.
    - Fixed: the pod was deleted and the 14 units re-queued.
    - The pool now re-checks the GPU in a fresh process before every claim.
    - The node is excluded from future protocol jobs.
  - A second pod on the same node, started before the fix, burned 10 of the
    re-queued units. The node then went NotReady ("no healthy devices") and
    the pod was evicted. The units were re-queued again, and no protocol pod
    remains on that node.
  - 07:38 UTC: the scheduler kept placing pods on nrp-01.laccd.edu, whose
    GPU was unhealthy. 65 pods failed admission (UnexpectedAdmissionError),
    each counted against the jobs' backoffLimit.
    - `protocol-nn` reached it and failed; `protocol-tab` stopped at 28/40
      with 5 healthy pods running.
    - The NRP utilization flag then blocked every job change and
      resubmission. The plan is to resubmit with the node excluded (the job
      files already exclude it) once the flag clears; claims on the PVC make
      the resubmission resume where the runs stopped.
  - 08:09 UTC: `Argentina_soybean_F1.npz` had corrupt data (Bad CRC-32 in
    `masks.npy`; its zip directory read fine). Seven TabM F1 ARG-S units
    failed on it.
    - A full CRC test of all 18 caches found only this file bad. It was
      deleted and re-queued for rebuild, along with its units.
    - Both corrupt ARG-S caches were among the first built, probably on the
      failing node.
    - The cache code now CRC-checks every archive before publishing it, and
      rebuilds a cache whose data fails to load.
  - ~08:55 UTC: the utilization flag had cleared.
    - `protocol-tab` backoffLimit was raised to 200. Its template cannot
      exclude nrp-01, but the per-claim GPU check limits a pod there to one
      burned unit; this was observed once.
    - `protocol-nn` was resubmitted with the node excluded and resumes from
      the PVC claims.
  - ~11:50 UTC: a pooled TabM F1 unit (LOYO fold 5, 337 test seasons) was
    OOM-killed at 48 Gi.
    - Five large pooled F1 units (LOYO 3/5/6, LORO 0/21) are held
      (`gave_up` = "held") until pods with the memory-lean loader (pushed)
      can run them.
    - The utilization flag is back on: the results pod cannot be recreated,
      so the watcher reads the PVC through a running protocol pod.
- 2026-10-06, ~16:30 UTC: the image batch was stopped on the project lead's
  request (pretrained-encoder p3-nbr first).
  - Done at the stop: image v1 25, v2 19, hybrid 9 units (mostly ARG-S;
    interim ARG-S CV10 fold-mean pixel: v1 0.44, v2 0.53, hybrid 0.66 on
    3 folds).
  - The hybrid's loader workers had been killed by the 2 Gi `/dev/shm`. A
    follow-up job (`protocol_hybrid_job.yaml`, 8 Gi shm) is ready.
  - The image batch resumes from the PVC claims; in-flight units go stale
    and are retaken.
  - The utilization flag (image pods requested 8 CPU / 40 Gi, used
    2–4 CPU / 4–6 Gi) blocks submissions. The job files have been
    right-sized.
