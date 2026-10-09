# The paper's best models: implementation and within-field comparison

**Status (2026-10-09): PM-01..PM-03 done (§6); PM-04 running on the cluster.**

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
