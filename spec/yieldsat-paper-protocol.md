# The paper's training protocol for all models (no validation set)

**Status (2026-10-06): point-level batch implemented and submitted to the
cluster.** The image models (v1/v2, hybrids) are the second batch. Results
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

## 6. Results

Pending.

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
