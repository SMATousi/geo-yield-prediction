# TabM for YieldSAT point prediction

**Status:** proposed 2026-10-05 (user request). Runs in parallel with the
knowledge-pretraining DEV round. Model: TabM, from Gorishniy et al.,
*"TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"*,
ICLR 2025, reference code `yandex-research/tabm`.

## 1. Motivation

The point task is a tabular regression: one 10 m cell → one yield value.
Each cell is a fixed set of columns:
- 24 slots × 12 S2 bands;
- 24 slots × 4 weather sums;
- DEM, terrain (4) and soil (48);
- crop and dates.

Three facts favour a tabular model over our Perceiver/transformer:

1. **Slots are calendar-aligned.** Slot k is calendar month k from January of
   harvest year − 1 (spec/yieldsat_data_contract.md). "B08 in July" is the
   same column for every cell, so a flat model sees aligned features
   directly. Attention over slots must learn an alignment the data already
   has.
2. **Strong inductive bias, little data diversity.** The DEV pairs have
   about 100–700 field seasons, and the cells of a field are highly
   correlated. Tabular deep models and GBDTs are robust in this regime.
   TabM's parameter-efficient ensemble of k ≈ 32 MLP members sharing weights
   (BatchEnsemble-style) reduces variance at little cost.
3. **Simplicity and speed.** An MLP trains and evaluates many times faster
   than the Perceiver, which makes larger sweeps (seeds, label budgets,
   feature sets) affordable.

The YieldSAT paper reports **no tabular baselines**: its benchmark models are
AFF, 3D-ConvLSTM, 3D-LSTM, LSTM, MMGF and Transformer. On the 12 DEV rows
the paper's best is AFF (14 of 24 row×level entries), 3D-ConvLSTM (5),
3D-LSTM (3) and LSTM (2).

**Expected limits.** The paper-best models see image tiles (spatial
context). TabM on single cells does not, unless the S6 neighbourhood
features are added; they gave +0.023 pixel R² to the point model in DEV
round 2. TabM is therefore expected to compete with or beat our point model,
not necessarily the paper's image models.

## 2. Inputs (feature sets)

The same point batches as `YieldSATPointDataset`: train-only normalization,
the same masks, cutoff `all_slots` (paper protocols). Feature sets for the
ablation:

| ID | Features | Width (approx.) |
|---|---|---|
| F0 | **Raw flat:** 24 × 16 temporal values + 24 × 16 masks, static 53 + masks, 24 × 3 time features (days since seeding/365, day-of-year sin/cos), crop one-hot | ≈ 900 |
| F1 | F0 + **S6 neighbourhood** S2 (24 × 12 + masks) | ≈ 1,480 |
| F2 | F0 + **pretrained stream embeddings**: frozen sensor encoders of a pretraining checkpoint (A2/A3/A6/A7), the per-stream summary embedding (5 × 128) | ≈ 1,540 |
| F3 | F0 + **concept scores**: the 12 cosine similarities between projected stream embeddings and the concept text prototypes (A3's knowledge heads), plus the 6 rule alignment scores | ≈ 920 |
| F4 | F0 + F2 + F3 | ≈ 1,560 |
| F5 | Embeddings + concept scores only (no raw) | ≈ 660 |

- **Missing values:** invalid values are set to 0 after standardization (the
  point contract), with an explicit mask column per value. TabM sees both.
- **Numeric embeddings:** periodic or piecewise-linear embeddings (the
  `rtdl_num_embeddings` package, recommended with TabM) for raw numeric
  columns; plain linear for embeddings and mask columns.
- **F2–F5 and leakage:** pretrained features come from the checkpoint of
  the fold's pretraining unit (`splits/pretrain_units_dev_s0.json`), never a
  checkpoint whose unit saw the fold's test seasons. Encoders are frozen in
  the first round. Fine-tuning the encoders jointly with TabM ("TabM head
  on the point encoders") is a follow-up arm.

## 3. Models and baselines

| ID | Model |
|---|---|
| T1 | **TabM** (k = 32, 3 blocks × 512, dropout 0.1, AdamW, MSE on the standardized target), per DEV row (pair-wise like all DEV runs) |
| T2 | TabM-mini (shared MLP, per-member input adapters only) |
| G1 | **LightGBM** on F0. A cheap, strong reference; if it beats TabM, the gain comes from the tabular view, not from TabM itself |
| M1 | Plain MLP on F0 (k = 1): isolates the ensembling effect |
| A0 | Existing point model (before_full seed 0) |

- **Training:** the same field-balanced sampler (α = 0.5), epochs of 500–1,500
  steps × 4,096 cells, early stopping on the fold's validation seasons
  (pixel RMSE, as for the point model). Prediction = mean of the k members.
- **Hyperparameters:** one small sweep on the CV10 folds of a single DEV pair
  (GER-R), then fixed for all rows (no per-row tuning on test folds).

## 4. Experiment (DEV)

Same DEV subset, folds, pooled metric and success bar as the improvement
plan (spec/yieldsat-improvement.md): 12 rows, 94 fine-tuning runs per
configuration.

- **Round TM-1:** T1 on F0, F1; G1 on F0; M1 on F0. Compared with A0, p3-nbr
  and the paper best.
- **Round TM-2:** T1 on F2–F5 with the A2/A3/A7 unit checkpoints, once the
  knowledge-pretraining DEV round has finished. Questions:
  - Do pretrained embeddings help TabM (F2 vs F0)?
  - Do concept scores help (F3 vs F0, and A3 vs A7 concept scores)? Concept
    scores from A7 (random targets) are the control.
- **Success:**
  - **TabM is adopted as the point backbone** if its DEV mean beats A0 by
    ≥ +0.01 pixel R² (paired fold-bootstrap CI excluding 0) with field R²
    not worse.
  - **The paper bar** is paper best + 0.03 at both levels
    (spec/yieldsat-improvement.md).
  - Knowledge-derived features count as **helping** only if F3/F4 beat
    F0/F2 *and* beat the same features from A7 (criterion E2 logic,
    spec/pretraining/success-criteria.md).

## 5. Tasks

| ID | Deliverable | Acceptance |
|---|---|---|
| TM-01 | `models_yieldsat_tabm.py`: TabM/TabM-mini (`tabm` package or a self-contained BatchEnsemble MLP), numeric embeddings, mean-of-members prediction | Unit tests: member independence, masks, shapes, gradients |
| TM-02 | `dataset/yieldsat_tabular.py`: flat F0/F1 feature builder on `YieldSATPointDataset` batches; column schema recorded | Columns match channel names; no target or date-leaking columns |
| TM-03 | Feature extractors for F2/F3 from unit checkpoints (frozen encoders + knowledge heads), fold → unit mapping | Leakage test: the unit's excluded set covers the fold's test seasons |
| TM-04 | Entry point `main_yieldsat_tabm.py` (or `--model tabm` in `main_yieldsat_finetune.py`) with the same report/predictions format | Pooled-metric tools read its outputs unchanged |
| TM-05 | LightGBM baseline (`--model lgbm`) on F0 | Same split/metric plumbing |
| TM-06 | GER-R CV10 sweep → fixed hyperparameters | Logged here |
| TM-07 | Cluster suites `tm_dev1` (TM-1) and `tm_dev2` (TM-2), `results/tabm_dev.md` | DEV pooled tables, paired CIs vs A0 / p3-nbr / paper |

## 6. Progress log

- 2026-10-05 — Specified.
- 2026-10-05 — **TM-01, TM-02, TM-04, TM-05 implemented; local run**
  (decision: TabM trains on the local RTX 3090 with each pair's flat matrix
  resident on the GPU).
  - **Code:** `dataset/yieldsat_tabular.py` (flat builder, point-contract
    validity, train-only per-column standardization), `models_yieldsat_tabm.py`
    (`TabMRegressor`), `main_yieldsat_tabm.py` (DEV-matrix driver: TabM,
    TabM-mini, MLP, LightGBM; outputs in the point model's
    report/predictions format plus `runs.jsonl`). Tests:
    `tests/test_yieldsat_tabm.py` (4 pass).
  - **Width:** F0 is 510 value columns (24 × 16 temporal, 24 × 3 time, 54
    static incl. aspect sin/cos) plus 125 mask columns.
    - Per-slot-per-stream masks (S2, weather) replace the per-value masks:
      cloud masking invalidates a slot's bands together, and individually
      invalid values are 0 after standardization.
    - Periodic embeddings (d = 8, lite) on value columns; masks pass through
      raw.
  - **Throughput** (RTX 3090, batch 4,096, bf16): TabM k = 32 ≈ 36k
    cells/s, TabM-mini ≈ 44k, plain MLP ≈ 180k. The first version (per-value
    masks, d = 16) ran at 23k cells/s.
  - **TM-06 sweep started** on GER-R CV10 (pooled R², 10 folds), each
    capped at 2,000 steps with early stopping on validation pixel RMSE:
    TabM at lr 1e-3 and 3e-4, TabM-mini, TabM with d_block 256 and
    dropout 0.2, MLP, LightGBM.
- 2026-10-05 — **Sweep, first results (GER-R CV10, pooled):**
  - LightGBM (F0): pixel R² 0.407, field 0.690 (10/10 folds, ≈ 12 s per
    fold). That beats the point model (A0 0.38 / 0.67) on this row; the
    paper's best is 0.49 / 0.81.
  - TabM lr 1e-3: 0.388 / 0.664 on 5/10 folds so far (≈ 3 min per fold,
    ≈ 1,400 steps before early stopping).
  - LightGBM F0 was launched on all 12 DEV rows (`runs/tabm_dev1`, tag
    `tm-lgbm-f0`) while the GPU sweep continues.
- 2026-10-05 — **TM-06 sweep done** (GER-R CV10, pooled pixel / field R²,
  s per fold):

  | Config | Pixel | Field | s/fold |
  |---|---|---|---|
  | TabM lr 3e-4 | 0.414 | 0.709 | 173 |
  | **TabM d_block 256, dropout 0.2** | 0.412 | **0.720** | 113 |
  | LightGBM | 0.407 | 0.690 | 12 |
  | TabM-mini | 0.403 | 0.700 | 157 |
  | TabM lr 1e-3 | 0.403 | 0.694 | 183 |
  | Plain MLP (k = 1) | 0.369 | 0.673 | 28 |
  | Point model A0 | 0.38 | 0.67 | |

  - Every TabM variant beats the point model on this row (+0.02 to +0.03
    pixel). The ensemble matters: the same MLP without it scores 0.369.
  - **Fixed for all rows:** TabM (k = 32, 3 blocks × 256, dropout 0.2,
    lr 1e-3, d_emb 8). It has the best field R², is within 0.002 of the best
    pixel R² and is the fastest TabM; the gaps are below fold noise, so there
    is no further tuning.
  - **TM-1 launched** (`runs/tabm_dev1`): LightGBM F0 (finishing), TabM F0,
    then MLP F0. TabM F1 follows once the S6 neighbourhood arrays are built
    locally for Argentina, Brazil and Uruguay (only Germany's existed
    locally).
- 2026-10-05 — **TM-1 interim: LightGBM F0 on all 12 DEV rows** (pooled):
  pixel 0.313, field 0.420 (CV10 / LOYO / LORO pixel 0.49 / 0.21 / 0.23).
  - That is level with the point model on pixel R² (0.309) and **below it at
    field level** (0.472); GER-R CV10's gain does not generalize. The paper's
    best is 0.460 / 0.686.
  - Neighbourhood arrays were built locally for Argentina, Brazil and
    Uruguay.
  - TabM F0 is running, MLP F0 is queued, and TabM F1 is queued after it.
- 2026-10-05 — **TM-1 (F0) complete** (12 DEV rows, pooled pixel / field
  R²):

  | Config | Pixel | Field | CV10 / LOYO / LORO pixel |
  |---|---|---|---|
  | **TabM F0** | **0.342** | 0.462 | 0.51 / 0.25 / 0.27 |
  | LightGBM F0 | 0.313 | 0.420 | 0.49 / 0.21 / 0.23 |
  | MLP F0 (k = 1) | 0.288 | 0.392 | 0.47 / 0.14 / 0.26 |
  | Point model A0 | 0.309 | 0.472 | |
  | p3-nbr (point + S6) | 0.332 | 0.497 | 0.50 / 0.24 / 0.26 |
  | Paper best | 0.460 | 0.686 | 0.56 / 0.41 / 0.41 |

  - TabM is the best pixel-level DEV mean of all point variants so far:
    +0.033 vs A0 and +0.010 vs p3-nbr. Field R² is about level with A0
    (−0.010).
  - It beats or ties A0 on 9 of 12 rows. Largest gains: GER-R LORO 0.21 vs
    −0.25, GER-R LOYO 0.27 vs 0.09, ARG-W LOYO 0.56 vs 0.46.
  - Weak spot: BRA-C LORO / LOYO (−0.01 / −0.13 pixel; field −0.28 /
    −0.45), where A0 scores 0.22 / 0.12.
  - The ensemble is essential: the same network without it (MLP) loses
    0.054 pixel.
  - TabM F1 (+ neighbourhood) is running. The paired CI vs A0 is computed in
    the final report (`results/tabm_dev.md`).
- 2026-10-05 — **Correction:** TabM F1 had crashed with a CUDA OOM shortly
  after starting (prediction batch 16,384 × 32 members at F1's ≈ 800 value
  columns needed a 6.4 GB block); it had been reported as running. The
  prediction batch is now 4,096 with `expandable_segments`; restarted
  (16.4 GB, 100% GPU).
