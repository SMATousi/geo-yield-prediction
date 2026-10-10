# The paper's best models: implementation and within-field comparison

**Status (2026-10-10): PM-01..PM-03 done (§6). PM-04 is almost complete: 3D-LSTM and MMGF 217/217, AFF 216/217,
3D-ConvLSTM 130/217. A partial PM-05 table is in §8.**
- **Reproduction:** the reimplementations match the paper's CV10 R² (MMGF, 3D-LSTM) or come out slightly lower (AFF;
  3D-ConvLSTM so far).
- **Within-field:** our best model beats every paper model within fields on all protocols. It is significantly better
  than the best paper model of each row in 9/9 (CV10), 5/9 (LOYO) and 6/9 (LORO) rows, and never significantly worse.

Related:
- [yieldsat-paper-protocol.md](./yieldsat-paper-protocol.md): protocol (no validation set, test-fold
  selection, pooled scoring) and the paper's per-row numbers.
- [yieldsat-paper-reproduction.md](./yieldsat-paper-reproduction.md): the IF-LSTM reproduction (R1)
  and the plan for R2/R3, which this spec carries out.
- [yieldsat-field-context.md](./yieldsat-field-context.md): our current best model (dense p3-nbr +
  field context + relation loss) and the within-field metrics.

## 1. Goal

1. **Within-field R² of the paper's best models.**
   - The paper reports only pooled pixel and field R². Within-field accuracy, the main objective, is
     unknown for its models.
   - We run its best architectures under the same protocol and splits as ours. The full table then
     compares within-field R² for all models.
2. **Reusable implementations for our own dataset.**
   - The models read the same artifacts (index, cache, splits) as our pipeline.
   - Any dataset converted to that contract (e.g. our own yield maps) can be benchmarked against them
     unchanged.

## 2. Which models

The paper's Table 4 and appendix A.3, S2 + ADM:
- **AFF** (feature fusion, [32]; the thesis's MMAF) is best in most rows.
- **3D-ConvLSTM** ([13], input fusion) and **MMGF** ([25], gated feature fusion) win some rows.
- **3D-LSTM** ([32], input fusion) is the paper's spatial baseline. It is also the deep-ensemble
  model of its LOYO/LORO table.

The pixel LSTM and Transformer are never best; the IF-LSTM is already in the repo (`paper_lstm`, R1).

| ID | Model | Fusion | Spatial input | Time series |
|---|---|---|---|---|
| `paper_3dlstm` | 3D-LSTM | input (all streams concatenated per slot; static repeated) | 5×5 window | monthly, 24 slots (as in the thesis IF pipeline) |
| `paper_3dconvlstm` | 3D-ConvLSTM | input | 5×5 window | monthly, 24 slots |
| `paper_aff` | AFF / MMAF | feature (modality encoders + attention pooling) | 5×5 window (S2, statics); weather at the centre | dense |
| `paper_mmgf` | MMGF | feature (modality encoders + gated fusion) | centre pixel | dense |

## 3. Architectures (from the thesis, A.1.2 and §4.5, unless marked)

**3D-LSTM:**
- Input X ∈ R^{B×C×T×5×5}: all streams, input fusion; static streams repeated over T.
- Conv3d(C → 64, kernel (1, 5, 5)), then BatchNorm and LeakyReLU, giving (B, 64, T).
- LSTM: 2 layers, 64 hidden units; last-slot readout.
- Head: Linear(64, 128), BatchNorm, ReLU, Linear(128, 1). This is the thesis LSTM head.

**3D-ConvLSTM** (thesis §7, following Helber et al. [13]):
- Per slot, a 3D convolution to 64 features. Kernel (1, 3, 3), padding 1: *assumed*, since the size is
  not given.
- ConvLSTM: 1 layer, 64 channels, kernel 9 (padding 4, so the 5×5 window is kept).
- A 1×1 convolution to one value per cell; the centre cell is the prediction.

**AFF / MMAF:** one encoder per modality, each giving N × 64, with BatchNorm + ReLU in every block.
- **S2:** the 3D-LSTM block (Conv3d (1, 5, 5) → LSTM 2 × 64).
- **Weather** (temporal, no spatial extent): LSTM, 2 × 64, on the centre cell.
- **Each static modality** (DEM, terrain, soil, coordinates): Conv2d(C → 64, kernel 5 × 5) → Linear
  64 → 64.
- **Fusion:** scaled dot-product attention pooling. A learnable query attends over the M modality
  features; dropout 0.2 on the attention weights; a Linear layer predicts the yield.

**MMGF** (Mena et al. [25], adaptive gated fusion). *Approximation from the description:* the paper gives
no layer sizes.
- **Encoders:** LSTM 2 × 64 for each temporal modality (S2, weather); MLP (C → 64 → 64) for each static
  one; centre pixel only.
- **Gate:** g = softmax(W [h_1, …, h_M]) ∈ R^M; fused = Σ_m g_m h_m.
- **Head:** MLP 64 → 64 → 1.

**Inputs and preprocessing:**
- **Streams:** S2, weather, DEM, terrain and soil (our `s2_adm`) plus coordinates, which the thesis
  feeds to MMAF (`Xcoord`).
- **Normalization:** train-fold statistics, as in our pipeline.
- **Missing values:** invalid values and cells outside the field are set to −1, the thesis padding value.
  These models do not see masks.
- **Window source:** the 5×5 window comes from the field's grid (cache `grid_row` / `grid_col`); cells
  outside the field season are padding.

**Training** (thesis A.1.2: 3D-LSTM; MMAF "similar to the 3D-LSTM"):
- Adam, learning rate 0.006, batch 2048.
- Reduce-on-plateau: factor 0.1, patience 3 epochs. The patience is *assumed* (not given; it must be
  below the early-stopping patience to act).
- At most 50 epochs; early stopping after 10 epochs without improvement.
- Augmentation (training only):
  - random 90° rotations of the input window;
  - temporal dropout: each slot of the temporal streams dropped with p = 0.2.
- Epoch = one pass over the training cells, capped at 1,500 steps. Full passes are 147–1,530 steps per
  pair, so only ARG-S is capped slightly.
- MSE on the train-standardized target.
- **Protocol:** the paper protocol, the same as all our tables: no validation set, epoch selection on the
  test fold, pooled scoring (spec/yieldsat-paper-protocol.md).

**Known differences from the paper:**
- **Dense series:** our dense series is 72 ten-day slots, while the thesis uses the raw acquisitions
  (~118 per season, padded with −1). Weather in the dense cache is 10-day sums, not daily values.
- **One seed** per fold (as for our models). The paper's LOYO/LORO table uses 5-member deep ensembles;
  this spec uses single models throughout.
- **Assumed hyperparameters**, marked above: the ConvLSTM input kernel, the plateau patience and the MMGF
  sizes.

## 4. Implementation

- **`dataset/yieldsat_window.py`: `YieldSATWindowDataset(YieldSATPointDataset)`.**
  - For every item, the dataset indices of its 5×5 neighbours within the same field season (−1 outside
    the field, or not in the partition).
  - `get_batch` gathers the unique neighbours through the point dataset's own batch path, so
    normalization, cutoff, masks and fill are identical.
  - It returns `win_inputs[stream]`: (B, 25, T, C) temporal, (B, 25, C) static.
  - Train-time rotation (a permutation of the 25 positions) and temporal dropout.
  - Batches use field blocks (`--block_size`) so that neighbouring windows share rows.
- **`models_yieldsat_paper.py`:** the four models, sharing the `loss` / `forward` interface of
  `PaperLSTMBaseline`.
- **`main_yieldsat_finetune.py`:**
  - `--model paper_3dlstm|paper_3dconvlstm|paper_aff|paper_mmgf` selects the window dataset;
  - `--lr_schedule plateau` (`--plateau_patience`, `--plateau_factor`);
  - `--paper_aug` turns on rotation and temporal dropout.
  - Test reports carry `within_field` as for all models.
- **Units:** `yieldsat_protocol_runs.py --batch paper-models`. All 9 pairs × CV10/LOYO/LORO, i.e.
  217 folds per model.
  - IF models use the monthly cache; AFF and MMGF use the dense cache.

## 5. Plan

| ID | Deliverable | Acceptance |
|---|---|---|
| PM-01 | Window dataset | Unit tests: centre = the point item; neighbours = the grid cells of the same field; padding outside; rotation permutes positions consistently; no target leaks into inputs |
| PM-02 | The four models | Unit tests: forward/backward on window batches; parameter counts logged |
| PM-03 | Trainer integration + local smoke (GER-R CV10 fold 0) | Trains, writes `report.json` with `within_field`; speed measured to size the cluster job |
| PM-04 | Cluster runs: 4 models × 217 folds | All folds done |
| PM-05 | Full table: pooled and within-field R² per pair × protocol for the paper models, our dense p3-nbr and dense p3-nbr + field context + relation loss; paper's reported pooled R² alongside | Paired bootstrap of our best vs the best paper model per row |

## 6. Implementation notes and smoke (2026-10-09)

**Tests:** `tests/test_paper_models.py`, 8 tests, all passing.
- The window index matches the field grid; padding where no cell of the field season exists.
- The centre equals the point item.
- Rotation tables match `np.rot90`.
- Temporal dropout removes slots and leaves the targets alone.
- Each model trains a step.

**Local smoke** (RTX 3090, GER-R CV10 fold 0, monthly cache, 2 × 60 steps, batch 2048; too short for the
scores to mean anything):

| Model | Parameters | Samples/s | Peak GPU memory (training) |
|---|---|---|---|
| 3D-LSTM | 190,657 | 17.6k | 0.95 GiB |
| 3D-ConvLSTM (bf16) | 2,696,193 | 5.3k | 2.16 GiB |
| AFF | 248,449 | 17.1k | 0.95 GiB |
| MMGF | 131,463 | 17.6k | 0.70 GiB |

RSS per process is ≈ 1.7 GiB on GER-R.

**Cluster sizing:** 868 units (4 models × 217 folds).
- 6 runs per 24 GB GPU: ≈ 35–45% GPU memory; 12 CPU, 40 Gi.
- 12 runs per 48 GB GPU in the companion job.
- `cluster/nautilus/protocol_paper_models_job.yaml` and `protocol_paper_models_b_job.yaml`.

**Table script:** `results/paper_models_table.py`. It was checked on the local backup of the existing runs.

## 7. Cluster fixes during PM-04 (2026-10-09)

1. **Shared memory and RAM.**
   - **Problem:**
     - Dense AFF/MMGF window batches (float32, every stream, masks) exhausted the 4 Gi `/dev/shm`.
     - Pods at 6 runs reached 34–40 of 40 GiB.
   - **Fix:**
     - Window values are sent as float16.
     - Streams a model reads only at the centre are sent as centre-only (MMGF: all; AFF: weather).
     - Masks are optional; prefetch is 1 batch per worker.
     - 4 runs per pod (36 Gi, 8 Gi shm).
2. **Last-slot readout (a real bug).**
   - **Problem:** AFF/MMGF read their LSTMs at the last of the 72 dense slots. A season fills ~15–35 slots,
     so the LSTMs first saw ~40–55 padding steps. Train loss stayed ~0.7 and pooled R² ~0 on ARG/BRA/URG.
   - **Fix:** read at the last in-season slot.
     - ARG-W CV10 fold 0, 5 epochs: AFF val R² 0.79, MMGF 0.84.
     - The 3D models (monthly, 24 slots) were not affected and were not rerun.
   - **Cleanup:** stale outputs were moved to `*_stale_lastslot`; all AFF/MMGF units were rerun.
3. **GPU utilization below 40% (project lead: at least 40%).**
   - **Problem:** building the 5×5 windows on the CPU made AFF CPU-bound (workers at ~90% CPU, GPU
     9–35%).
   - **Fix:** the dataset now sends the batch's distinct cells (`win_src`) plus an index map, and the model
     gathers the windows on the GPU. AFF dense trains at 14.2k vs 4.7k samples/s, with the same accuracy.
   - **After deployment:** mean GPU utilization was 83–87%. AFF-heavy pods were still CPU-bound: ~0.17 s
     of CPU per batch, mostly the point dataset's normalization of the 72-slot series for ~5.3 distinct
     cells per sample.
   - **Remedy:** the 122 AFF units not yet started moved to `protocol_paper_aff_job.yaml`: 24 CPU, 6 data
     workers per run, ids `…__w6`, the same output paths.

## 8. Results (partial, 2026-10-10)

Data: `results/paper_models_table_partial.json`, made by `results/paper_models_table.py`.
- **Protocol:** all models on the same splits and the same test pixels, with pooled scoring.
- **Ours: FC + rel** = dense p3-nbr + field context + relation loss (spec/yieldsat-field-context.md §7.1).
- **Last column:** our best model minus the best paper model of the row, where "best" is chosen by within-field R² on
  the same test data, which favours the paper models. 95% field-season cluster bootstrap; **bold** = CI excludes 0.
- **Incomplete rows:** a paper model whose folds are not all done shows its fold count instead of scores. They will
  be filled in when PM-04 completes.

**CV10**, pooled R² / within-field R²:

| Pair | Ours: FC + rel | Ours: dense p3-nbr | MMGF | AFF | 3D-LSTM | 3D-ConvLSTM | Ours vs best paper model (within-field) |
|---|---|---|---|---|---|---|---|
| ARG-C | 0.724 / 0.455 | 0.724 / 0.448 | 0.680 / 0.427 | 0.655 / 0.357 | 0.655 / 0.338 | (7 folds) | MMGF: **+0.027** [+0.007, +0.047] |
| ARG-S | 0.738 / 0.541 | 0.724 / 0.522 | 0.710 / 0.524 | (9 folds) | 0.675 / 0.465 | (0 folds) | MMGF: **+0.017** [+0.006, +0.027] |
| ARG-W | 0.822 / 0.394 | 0.824 / 0.367 | 0.804 / 0.351 | 0.791 / 0.327 | 0.775 / 0.266 | (4 folds) | MMGF: **+0.044** [+0.011, +0.076] |
| BRA-C | 0.494 / 0.277 | 0.502 / 0.275 | 0.458 / 0.236 | 0.472 / 0.252 | 0.434 / 0.198 | (2 folds) | AFF: **+0.025** [+0.009, +0.043] |
| BRA-S | 0.440 / 0.249 | 0.434 / 0.229 | 0.422 / 0.221 | 0.422 / 0.219 | 0.393 / 0.188 | (2 folds) | MMGF: **+0.027** [+0.021, +0.034] |
| BRA-W | 0.247 / 0.112 | 0.246 / 0.095 | 0.190 / 0.061 | 0.210 / 0.088 | 0.206 / 0.069 | (7 folds) | AFF: **+0.025** [+0.014, +0.036] |
| GER-R | 0.517 / 0.254 | 0.521 / 0.229 | 0.457 / 0.219 | 0.477 / 0.194 | 0.420 / 0.186 | (9 folds) | MMGF: **+0.034** [+0.012, +0.056] |
| GER-W | 0.480 / 0.237 | 0.469 / 0.216 | 0.426 / 0.189 | 0.399 / 0.172 | 0.407 / 0.173 | 0.343 / 0.108 | MMGF: **+0.048** [+0.028, +0.066] |
| URG-S | 0.419 / 0.105 | 0.419 / 0.098 | 0.407 / 0.085 | 0.400 / 0.086 | 0.396 / 0.072 | (8 folds) | AFF: **+0.019** [+0.011, +0.027] |
| **Mean** | **0.542 / 0.292** | **0.540 / 0.276** | **0.506 / 0.257** | **0.478 / 0.212** (8 pairs) | **0.484 / 0.217** | **0.343 / 0.108** (1 pairs) | |

**LOYO**, pooled R² / within-field R²:

| Pair | Ours: FC + rel | Ours: dense p3-nbr | MMGF | AFF | 3D-LSTM | 3D-ConvLSTM | Ours vs best paper model (within-field) |
|---|---|---|---|---|---|---|---|
| ARG-C | 0.512 / 0.324 | 0.542 / 0.301 | 0.477 / 0.322 | 0.524 / 0.297 | 0.268 / 0.177 | (7 folds) | MMGF: +0.002 [-0.027, +0.030] |
| ARG-S | 0.660 / 0.492 | 0.627 / 0.467 | 0.597 / 0.473 | 0.550 / 0.440 | 0.515 / 0.381 | (3 folds) | MMGF: **+0.019** [+0.009, +0.030] |
| ARG-W | 0.725 / 0.181 | 0.742 / 0.151 | 0.590 / 0.217 | 0.360 / 0.155 | 0.626 / 0.110 | (4 folds) | MMGF: -0.036 [-0.104, +0.029] |
| BRA-C | 0.207 / 0.200 | 0.116 / 0.147 | 0.148 / 0.186 | 0.162 / 0.193 | 0.058 / 0.115 | (3 folds) | AFF: +0.007 [-0.013, +0.023] |
| BRA-S | 0.315 / 0.191 | 0.308 / 0.169 | 0.351 / 0.185 | 0.320 / 0.182 | 0.137 / 0.122 | (7 folds) | MMGF: +0.005 [-0.012, +0.023] |
| BRA-W | 0.142 / 0.095 | 0.153 / 0.089 | 0.050 / 0.071 | 0.104 / 0.085 | 0.063 / 0.066 | (5 folds) | AFF: **+0.009** [+0.001, +0.018] |
| GER-R | 0.291 / 0.208 | 0.285 / 0.168 | 0.325 / 0.163 | 0.328 / 0.173 | 0.251 / 0.092 | 0.178 / -0.024 | AFF: **+0.035** [+0.014, +0.058] |
| GER-W | 0.133 / 0.193 | 0.077 / 0.177 | 0.063 / 0.147 | 0.151 / 0.123 | 0.210 / 0.088 | (5 folds) | MMGF: **+0.046** [+0.025, +0.067] |
| URG-S | 0.345 / 0.086 | 0.355 / 0.086 | 0.274 / 0.061 | 0.230 / 0.056 | 0.219 / 0.041 | 0.187 / 0.032 | MMGF: **+0.024** [+0.015, +0.035] |
| **Mean** | **0.370 / 0.219** | **0.356 / 0.195** | **0.320 / 0.203** | **0.303 / 0.189** | **0.261 / 0.132** | **0.183 / 0.004** (2 pairs) | |

**LORO**, pooled R² / within-field R²:

| Pair | Ours: FC + rel | Ours: dense p3-nbr | MMGF | AFF | 3D-LSTM | 3D-ConvLSTM | Ours vs best paper model (within-field) |
|---|---|---|---|---|---|---|---|
| ARG-C | 0.548 / 0.377 | 0.543 / 0.335 | 0.542 / 0.324 | 0.499 / 0.303 | 0.346 / 0.178 | (2 folds) | MMGF: **+0.053** [+0.019, +0.090] |
| ARG-S | 0.671 / 0.494 | 0.662 / 0.480 | 0.612 / 0.472 | 0.604 / 0.450 | 0.485 / 0.337 | (5 folds) | MMGF: **+0.023** [+0.012, +0.034] |
| ARG-W | 0.756 / 0.269 | 0.777 / 0.276 | 0.644 / 0.274 | 0.554 / 0.142 | 0.613 / 0.181 | (3 folds) | MMGF: -0.005 [-0.060, +0.039] |
| BRA-C | 0.286 / 0.213 | 0.239 / 0.197 | 0.268 / 0.202 | 0.272 / 0.209 | 0.213 / 0.153 | (5 folds) | AFF: +0.004 [-0.018, +0.022] |
| BRA-S | 0.357 / 0.201 | 0.341 / 0.184 | 0.337 / 0.174 | 0.283 / 0.177 | 0.238 / 0.109 | (8 folds) | AFF: **+0.024** [+0.013, +0.036] |
| BRA-W | 0.170 / 0.087 | 0.172 / 0.078 | 0.098 / 0.054 | 0.126 / 0.081 | 0.144 / 0.043 | (2 folds) | AFF: +0.006 [-0.005, +0.019] |
| GER-R | 0.174 / 0.204 | 0.151 / 0.132 | -0.142 / 0.148 | -0.015 / 0.156 | 0.014 / 0.096 | (5 folds) | AFF: **+0.047** [+0.019, +0.080] |
| GER-W | -0.019 / 0.188 | 0.090 / 0.106 | -0.096 / 0.143 | -0.115 / 0.113 | -0.018 / 0.029 | -0.194 / -0.014 | MMGF: **+0.046** [+0.028, +0.067] |
| URG-S | 0.371 / 0.089 | 0.365 / 0.081 | 0.330 / 0.078 | 0.308 / 0.068 | 0.311 / 0.049 | (8 folds) | MMGF: **+0.012** [+0.005, +0.019] |
| **Mean** | **0.368 / 0.236** | **0.371 / 0.208** | **0.288 / 0.207** | **0.279 / 0.189** | **0.261 / 0.130** | **-0.194 / -0.014** (1 pairs) | |

**Means over the 9 pairs** (pooled R² / within-field R²):

| Protocol | Ours: FC + rel | Ours: dense p3-nbr | MMGF | AFF | 3D-LSTM |
|---|---|---|---|---|---|
| CV10 | **0.542 / 0.292** | 0.540 / 0.276 | 0.506 / 0.257 | 0.478 / 0.212 (8 pairs) | 0.484 / 0.217 |
| LOYO | **0.370 / 0.219** | 0.356 / 0.195 | 0.320 / 0.203 | 0.303 / 0.189 | 0.261 / 0.132 |
| LORO | **0.368 / 0.236** | 0.371 / 0.208 | 0.288 / 0.207 | 0.279 / 0.189 | 0.261 / 0.130 |

**Reading:**
1. **Our best model is ahead of every paper model on both metrics, on every protocol.**
   - Within fields it is significantly better than the best paper model of the row in 9/9 (CV10), 5/9 (LOYO) and
     6/9 (LORO) rows. The other rows are not significant; the worst is ARG-W LOYO, −0.036 [−0.104, +0.029].
   - It is never significantly worse.
2. **MMGF is the strongest paper model.**
   - Within fields it trails our best by ≈ 0.02–0.035 on average.
   - AFF and the 3D-LSTM are clearly weaker.
3. **The pooled-R² lead grows out of distribution:** +0.04 (CV10), +0.05 (LOYO) and +0.08 (LORO) over MMGF.
4. **Every model is far from the within-field noise ceiling** (≈ 0.72 on CV10, spec/yieldsat-field-context.md §3).

### 8.1 Reproduction check against the paper (CV10, fold-mean R² as the paper reports)

The paper's Table 4 averages R² over folds (spec/yieldsat-paper-protocol.md). The table above pools the folds, so
this check uses fold means.

**Pixel R², ours / paper** (S2 + ADM rows):

| Model | ARG-S | BRA-C | BRA-S | GER-R | GER-W | URG-S | Mean, ours vs paper |
|---|---|---|---|---|---|---|---|
| MMGF | 0.70 / 0.70 | 0.44 / 0.42 | 0.40 / 0.42 | 0.45 / 0.44 | 0.42 / 0.44 | 0.40 / 0.40 | 0.47 vs 0.47 |
| 3D-LSTM (input fusion) | 0.66 / 0.64 | 0.41 / 0.46 | 0.38 / 0.41 | 0.41 / 0.49 | 0.40 / 0.37 | 0.39 / 0.40 | 0.44 vs 0.46 |
| AFF | 0.69 / 0.73 | 0.45 / 0.46 | 0.40 / 0.44 | 0.46 / 0.49 | 0.40 / 0.44 | 0.40 / 0.43 | 0.47 vs 0.50 |
| 3D-ConvLSTM (input fusion; incomplete) | 0.59 / 0.68 (3 folds) | 0.36 / 0.46 | 0.33 / 0.38 | 0.37 / 0.42 | 0.33 / 0.39 | 0.38 / 0.41 | lower by 0.03–0.10 |

**Field R², ours / paper:**

| Model | ARG-S | BRA-C | GER-R | GER-W | URG-S |
|---|---|---|---|---|---|
| MMGF | 0.81 / 0.82 | 0.81 / 0.76 | 0.74 / 0.75 | 0.71 / 0.77 | 0.75 / 0.75 |
| 3D-LSTM | 0.80 / 0.76 | 0.78 / 0.84 | 0.67 / 0.81 | 0.71 / 0.62 | 0.74 / 0.76 |
| AFF | 0.82 / 0.84 | 0.80 / 0.84 | 0.76 / 0.80 | 0.70 / 0.74 | 0.73 / 0.81 |

**Reading:**
- **MMGF reproduces the paper:** within ±0.02 pixel and ±0.06 field.
- **3D-LSTM reproduces on average** (−0.02 pixel). The single-seed per-pair spread is larger: GER-R is −0.08 pixel and
  −0.14 field.
- **AFF is slightly lower everywhere:** −0.03 pixel and −0.04 field on average.
  - It is consistent with the known input difference (§3): 10-day dense slots and 10-day weather sums instead of raw
    acquisitions and daily weather.
- **3D-ConvLSTM is lower by 0.03–0.10 pixel so far.** Its input kernel is assumed (§3); to be judged when complete.
- **Fairness:** the reimplementations are close enough for a fair within-field comparison. If anything they are
  slightly favourable to us for AFF and the 3D-ConvLSTM. Our best model also beats MMGF, the paper model that
  reproduces exactly.
