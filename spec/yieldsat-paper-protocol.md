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

The table shows R² averaged over folds (the paper's metric), then averaged
over the 9 pairs. Paper rows are means of the paper's per-pair numbers. In
brackets: the number of pairs (of 9) above the paper's LSTM S2+ADM / above
the paper's best model.

| Protocol / level | Paper LSTM S2 | Paper LSTM S2+ADM | Paper best | Ours p3-nbr | TabM F1 | TabM F0 | MLP | LightGBM | TabM F1 pooled | Ours pooled |
|---|---|---|---|---|---|---|---|---|---|---|
| CV10 pixel | 0.37 | 0.46 | 0.53 | **0.50** [8 / 1] | 0.49 [8 / 0] | 0.49 [8 / 1] | 0.47 [7 / 0] | 0.47 [5 / 1] | 0.46 [5 / 0] | 0.45 [5 / 0] |
| CV10 field | 0.69 | 0.72 | 0.82 | **0.78** [8 / 0] | 0.77 [6 / 0] | 0.77 [7 / 0] | 0.74 [6 / 0] | 0.74 [6 / 0] | 0.72 [5 / 0] | 0.69 [3 / 0] |
| LOYO pixel | 0.15 | 0.31 | 0.38 | **0.17** [0 / 0] | 0.08 [1 / 0] | 0.06 [1 / 0] | −0.01 [1 / 0] | −0.03 [1 / 0] | 0.03 [0 / 0] | 0.08 [0 / 0] |
| LOYO field | 0.08 | 0.43 | 0.54 | **−0.07** [0 / 0] | −1.24 | −0.81 | −1.82 | −0.49 | −1.31 | −0.73 |
| LORO pixel | 0.18 | 0.23 | 0.39 | **0.23** [4 / 1] | 0.21 [4 / 0] | 0.21 [4 / 0] | 0.15 [4 / 0] | 0.17 [3 / 0] | 0.13 [3 / 0] | 0.17 [3 / 0] |
| LORO field | 0.06 | 0.22 | 0.58 | −0.12 [2 / 0] | **−0.01** [2 / 0] | −0.03 [2 / 0] | −0.09 [1 / 0] | −0.10 [2 / 0] | −0.18 [1 / 0] | −0.09 [2 / 0] |

**Metric correction.** The fold-mean table above uses the wrong convention
for LOYO/LORO. The paper's LOYO/LORO rows behave like R² pooled over all
folds' predictions: 31 paper rows have a mean ± std that is impossible as a
per-fold mean (`results/negative_r2_investigation.md`). The same rows
recomputed as pooled out-of-fold R² (per-pair arm;
`results/protocol_point_oof.md`):

| Protocol / level | Paper LSTM S2+ADM | Paper best | Ours p3-nbr | TabM F1 | TabM F0 | MLP | LightGBM |
|---|---|---|---|---|---|---|---|
| CV10 pixel | 0.46 | 0.53 | 0.51 [8 / 1] | 0.51 [9 / 1] | 0.51 [9 / 1] | 0.49 [7 / 1] | 0.48 [6 / 1] |
| CV10 field | 0.72 | 0.82 | **0.80** [8 / 3] | 0.79 [8 / 0] | 0.79 [8 / 0] | 0.76 [7 / 0] | 0.75 [6 / 1] |
| LOYO pixel | 0.31 | 0.38 | **0.35** [7 / 0] | 0.32 [4 / 1] | 0.31 [5 / 2] | 0.27 [3 / 1] | 0.28 [3 / 1] |
| LOYO field | 0.43 | 0.54 | **0.45** [6 / 0] | 0.37 [5 / 0] | 0.36 [4 / 0] | 0.29 [2 / 0] | 0.37 [4 / 1] |
| LORO pixel | 0.23 | 0.39 | **0.32** [9 / 0] | 0.29 [7 / 1] | 0.30 [7 / 1] | 0.28 [5 / 1] | 0.28 [6 / 0] |
| LORO field | 0.22 | 0.58 | **0.46** [8 / 0] | 0.38 [7 / 1] | 0.38 [7 / 1] | 0.37 [6 / 1] | 0.40 [7 / 0] |

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
4. **Pooled training** (one model for all pairs) is worse than per-pair
   training in every protocol (CV10 fold-mean −0.03 pixel, −0.05 field).
   This is consistent with the paper's per-pair training.
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
