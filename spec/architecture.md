# Architecture

Tree state: commit `c8be363`. ~9,700 lines of Python across 44 modules.

**Full-layer update (2026-09-28):** Current implementations are described below
with Phase 2 corrections. The target architecture for real data is specified in
[layer_integration.md](./layer_integration.md); it is planned, not implemented.
That contract covers downloader catalogs and external PlanetScope, native grids,
quality/time masks, source-configured encoders and separate unlabelled/labelled
datasets. The fixed five-modality synthetic configuration is a prototype subset.

## 0. The repo contains two systems

This is the single most important fact for anyone reading the code.

```
                 ┌──────────────────────────────────────────┐
  SYSTEM A       │ upstream MMST-ViT (county-level)          │
  (inherited)    │ main_pretrain_mmst_vit.py                 │
                 │ main_finetune_mmst_vit.py                 │
                 │ models_mmst_vit.py, models_pvt.py,        │
                 │ models_pvt_simclr.py, attention.py        │
                 │ dataset/{sentinel,hrrr,usda}_loader.py    │
                 │ config/build_config_soybean.py            │
                 └──────────────────────────────────────────┘
                                   │
                          shares only util/
                                   │
                 ┌──────────────────────────────────────────┐
  SYSTEM B       │ field-scale multimodal FM (this project)  │
  (new, 48       │ main_pretrain_multimodal.py               │
   commits)      │ main_multimodal_finetune.py               │
                 │ models_multimodal_encoder.py              │
                 │ models_latent_fusion.py                   │
                 │ models_heads.py, models_neck.py           │
                 │ models_multimodal_pretrain.py             │
                 │ models_multi_unified.py                   │
                 │ dataset/field_*.py, heterogeneous_*.py    │
                 └──────────────────────────────────────────┘
```

System B does **not** import System A's models or loaders. `models_mmst_vit.py` is
untouched by the new work. The README still documents only System A.

Everything below describes System B unless stated otherwise.

---

## 1. Data flow

```
  on-disk GeoTIFFs / .npy per field-year
            │
            │  ❌ NOT CONNECTED (see status.md, gap G1)
            ▼
  dataset/field_yield_dataset.py      FieldYieldDataset
  dataset/heterogeneous_modality_loader.py
            │   per-sample dict: {layer: tensor, 'yield': (1,H,W), 'available': {...}}
            ▼
  dataset/collate.py                  collate_field_samples
            │   {inputs: {mod: (B,...)}, available: {mod: (B,)}, targets: (B,1,H,W)}
            ▼
  models_multimodal_encoder.py        MultiModalEncoder.forward_with_missing
            │   {mod: (B, L_mod, D)}, {mod: (B,)}      ← per-modality native encoders
            ▼
  models_latent_fusion.py             LatentFusionTransformer
            │   (B, num_latents, D)                     ← Perceiver cross-attn + self-attn
            ▼
  [reshape latents → (B, D, g, g)]
            ▼
  models_heads.py                     DenseYieldFCNHead / FPN / DPT / MDN
            │
            ▼
  (B, 1, H_out, W_out) dense yield map
```

The dashed break at the top is the central architectural gap: the loaders exist and
are competently written, but no training script imports them. Both entry points call
`make_synthetic_batch()` instead.

The planned replacement reads catalog assets, indexes and sidecars into lazy
native-footprint datasets. Both training modes share a source registry and sensor
adapters; the supervised mode additionally loads yield and its output grid/mask.
Tokens carry physical grid/time metadata into fusion. Expert-approved concept
grounding and relationship constraints form an optional pretraining branch, with
an optional offline VLM/LLM applicability teacher. See tasks
LI-01–LI-15 rather than interpreting the demo data flow as the target handoff.

A second planned input branch, [YieldSAT](./yieldsat_data_contract.md),
reads merged NetCDF grid-cell histories lazily and separates temporal optical,
weather, static soil and terrain streams. Its point-mode adapter/scalar yield head
and optional later patch mode share encoder/fusion interfaces while preserving
branch-specific provenance. Stored merged grids do not recover original native
resolutions; YS tasks govern this branch independently of LI compliance.

Implemented data flow (2026-09-28):

```text
<Country>/merge_s2-soil-dem-weather-coords.nc   (read-only, HDF5 via h5py)
  └─ yieldsat_prepare.py index / cache / geometry / splits  → artifact root
       rows.npz, fields.json | temporal/static/times .npy + field_stats | geometry | split manifests
  └─ YieldSATPointDataset (cache or h5 backend; masks, cutoff, train-only normalizer)
       inputs/masks: s2 (T,12)  weather (T,4)  dem (1)  terrain (4)  soil (48)  + time features, crop id
  └─ YieldSATPointModel
       MultiModalEncoder: masked_temporal ×2, masked_static ×2, soil_profile  → one token each
       LatentFusionTransformer (+ optional crop token)  → scalar_cell_yield head  → t/ha per cell
  └─ util/yieldsat_eval.py: per-country/crop pixel, field-balanced, field-level metrics; grid scatter → GeoTIFF
```

---

## 2. Modality encoders — `models_multimodal_encoder.py`

`MultiModalEncoder` is a config-driven dispatcher. `encoders_cfg` maps a source name
to `{'type': ..., ...kwargs}`, and `_TYPES` (`:240`) resolves the type to a class.

| type | class | input shape | output tokens | temporal modelling |
|---|---|---|---|---|
| `raster` | `RasterCNNEncoder` `:33` | `(B,C,H,W)` | `(B, H·W, D)` | — |
| `patch` | `PatchEmbeddingEncoder` `:56` | `(B,C,H,W)` | `(B, N+1, D)` cls-prefixed | — |
| `sar` | `MultitemporalSAREncoder` `:129` | `(B,T,C,H,W)` | `(B, H·W, D)` | Conv3d over T, then mean-pool |
| `timeseries` | `TimeSeriesEncoder` `:156` | `(B,T,Dᵢ)` | `(B, 1, D)` | **mean-pool only** ⚠ |
| `tabular` | `TabularEncoder` `:177` | `(B,Dᵢ)` | `(B, 1, D)` | — |
| `categorical` | `CategoricalEncoder` `:197` | `(B,H,W)` int | `(B, H·W, D)` | — |
| `grouped_vit` | `GroupChannelsVisionTransformer` | `(B,C,H,W)` | ViT tokens | — |
| `module` | caller-supplied `nn.Module` | any | any | — |

⚠ **`TimeSeriesEncoder` has no temporal modelling.** It applies a per-timestep MLP
then `h.mean(dim=1)` (`:173`), which is permutation-invariant over time. A shuffled
growing season produces an identical embedding. It satisfies the *name* of the
"temporal encoder" requirement, not its intent. See status.md defect **D4**.

### Native resolution — what it actually means here

Each encoder consumes its source at whatever `H×W` that source arrives at; no
cross-modality resampling happens *inside the model*. But `FieldYieldDataset`
registers every layer onto one shared field template grid before the model sees it
(`field_yield_dataset.py:46`). So "native resolution" is preserved **relative to the
field grid**, not absolutely. These loaders do not satisfy the native-first
full-layer requirement. LI-03 and LI-08 replace shared-grid ingestion with native
footprint windows and physical position/resolution/time metadata for fusion. A
shared grid is chosen for supervised outputs, separately from predictor inputs.

### Missing-modality handling

`forward_with_missing` (`:331`) returns `(embeddings, mask)` covering *every*
configured modality. Three paths:
- present for the whole batch → run the encoder, mask = ones
- absent for the whole batch → emit the learned `missing_tokens[name]` expanded to
  `(B, 1, D)`, mask = zeros
- **mixed within the batch** → `_encode_mixed` (`:298`) encodes the present rows,
  repeats the missing token to the same token count for absent rows, then unsorts
  back to batch order

`modality_dropout` (`:360`) randomly ablates sources in training mode.

### Per-modality normalisation

`normalize_modality()` from `util/norm_stats.py` now accepts NumPy arrays and
PyTorch tensors. The encoder selects the channel axis for each modality type and
applies registered statistics before encoding. A config can set `norm_source`
to choose the sensor statistics; unknown sources retain their input values.

---

## 3. Fusion backbone — `models_latent_fusion.py`

`LatentFusionTransformer` (`:129`) is a Perceiver: `num_latents` learned queries
cross-attend to the concatenation of all modality tokens, then a stack of
`nn.TransformerEncoderLayer` blocks self-attends over the latents.

Per-token conditioning, all summed/concatenated into the `D`-dim token before fusion:

| signal | source | dims |
|---|---|---|
| spatial position | `SeparablePosEmbed.pos_embed_spatial` `:100` | `D − modality_embed` |
| acquisition time | `SeparablePosEmbed.pos_embed_temporal` `:101` | (added to spatial) |
| modality identity | `modality_embedding` `:167` | `modality_embed` (concatenated) |
| ground resolution | `util/pos_embed.py:83` `get_2d_sincos_pos_embed_with_resolution` | used by encoders |

`modalities` maps each source to `{'spatial': int, 'temporal': int}` declaring its
token layout. **These counts must match what the encoder actually emits.** The
fusion now allows a compact 1-token missing-modality embedding and raises a
`ValueError` for any other declared-layout mismatch.

`forward_multiscale` (`:287`) re-runs the same stack and harvests latents at four
intermediate block indices for the FPN/DPT heads. Both paths use the same token
assembly and mask logic.

### Attention masking

`_build_key_padding_mask` builds a boolean `(B, ΣL)` mask so absent-modality
tokens cannot contribute. If every modality is absent in a sample, one learned
missing token remains visible to attention to avoid an all-masked softmax row.

---

## 4. Task heads — `models_heads.py`

`HEAD_REGISTRY` (`:28`) + `@register_head(...)` (`:31`) with validated `category` and
`modality` vocabularies. This is the designated extension point: register once,
attach by name in a `heads_cfg`, never touch the encoders.

| name | class | line | output |
|---|---|---|---|
| `dense_yield` | `DenseYieldFCNHead` | `:200` | `(B,1,H,W)` per-pixel map |
| `dense_yield_fpn` | `DenseYieldFPNHead` | `:344` | multi-scale FPN + PPM decoder |
| `dense_yield_dpt` | `DenseYieldDPTHead` | `:439` | DPT-style reassembly decoder |
| `mdn_yield` | `MDNYieldHead` | `:575` | mixture-density, gives predictive uncertainty |

`DenseYieldFCNHead` also carries `reassemble()` and `stitch()` for scattering
per-pixel latents back to grid positions and bilinearly stitching tiled patches onto
an arbitrary field geometry — the path from model output to a georeferenced raster.

`models_neck.py` `MultiFusionNeck` (`:19`) provides the feature-pyramid neck that
lets high-resolution terrain tokens coexist with coarse soil/climate tokens.

`models_multi_unified.py` `MultiUnifiedModel` (`:66`) is the encoder + neck + multi-head
container that realises the "one backbone, many tasks" claim. **It has no callers.**

---

## 5. Self-supervised pretraining — `models_multimodal_pretrain.py`

`MultimodalSelfSupervisedPretrain` (`:59`) combines five objectives into a weighted sum:

| objective | method | what it teaches |
|---|---|---|
| masked spatial reconstruction | `_masked_objectives` | local spatial structure |
| masked-modality modelling | `_masked_objectives` | fill in a dropped source |
| temporal forecasting | `_temporal_forecast` `:221` | predict frame T from 1..T−1 |
| cross-modal prediction | `_cross_modal_prediction` `:266` | leave-one-out: reconstruct a held-out modality from the rest |
| contrastive alignment | `_contrastive_alignment` `:290` | InfoNCE — two sensors over the same field share a latent |

**Planned extension:** [Expert-validated relationship pretraining v2](./knowledge_pretraining.md)
supersedes the previous statement-conditioned reconstruction proposal. Frozen text
references anchor sensor concept projections; reviewed applicability policies and
relation-specific scorers provide soft grounding/relationship losses. An optional
frozen VLM/LLM estimates applicability offline and must be validated. Both sensor
endpoints can receive gradients; this branch does not directly train fusion. Yield
fine-tuning and inference discard all knowledge branches and remain sensor-only.
This extension is specified, not implemented; see KP-01–KP-08.

`set_encoder_trainable()` (`:351`) supports the frozen / partial / full fine-tuning
comparison.

**Cost warning.** `_cross_modal_prediction` runs the full fusion transformer once per
modality, and `_temporal_forecast` once per temporal modality. With the default
5-modality config, one pretraining step invokes the fusion stack roughly 8–12 times.
At the default `input_size=64` that is ~12,288 modality tokens per fusion call. This
is not a bug, but it makes the pretraining loop far more expensive than it looks and
should be budgeted for. See roadmap Phase 4.

---

## 6. Dataset layer — `dataset/`

The genuinely strong, genuinely disconnected part of the codebase.

| module | role | wired in? |
|---|---|---|
| `field_roi.py` | builds the canonical field template grid (crs/transform/w/h) from a boundary mask at a fixed ground resolution | ✅ used by 4 modules |
| `field_yield_dataset.py` | field-year `Dataset`; registers boundary + yield-monitor raster + each modality layer into the template grid with per-kind resampling (nearest for categorical, bilinear for continuous) | ⚠ only by `heterogeneous_modality_loader` |
| `heterogeneous_modality_loader.py` | shapes each registered layer to its encoder's expected tensor rank (`(1,H,W)`, `(T,C,H,W)`, `(D,)`, `(H,W)`) | ❌ no callers |
| `collate.py` | batches samples with differing modality keys into dense tensors + availability mask | ❌ no callers |
| `field_pipeline.py` | checkpointed multi-step preprocessing pipeline | ❌ no callers |
| `nearest_reproject.py` | kd-tree nearest-neighbour reprojection | ⚠ only by `field_pipeline` |
| `block_retiler.py` | mmap block retiling for large rasters | ❌ no callers |
| `soilgrids_loader.py` | SoilGrids field-level loader | ❌ no callers |
| `mapping_dataset.py` | h5 multimodal loader | ❌ no callers |
| `spectral_indices.py` | NDVI/EVI computation | ⚠ only by `field_pipeline` |
| `field_patch_dataset.py` | patch-level dataset | ⚠ only by `field_pipeline`, `spectral_indices` |
| `sentinel_loader.py`, `hrrr_loader.py`, `usda_loader.py`, `data_wrapper.py`, `sentinel_wrapper.py` | **System A** county-level loaders | ✅ by System A entry points |

Read [data_contract.md](./data_contract.md) for the on-disk layout these expect.

---

## 7. Utilities — `util/`

| module | role | wired in? |
|---|---|---|
| `pos_embed.py` | sincos embeddings incl. resolution-conditioned `:83` | ✅ widely |
| `metrics.py` | `RMSE`/`R2`/`PCC`, `SpatialCorrelation`, `ZonePreservation`, `evaluate_regression`, `SpatialYieldMetric`, `ConfusionMatrix` | ✅ |
| `misc.py`, `lr_sched.py`, `lr_decay.py`, `lars.py` | training scaffolding from MAE | ✅ |
| `norm_stats.py` | per-modality mean/std registry (S2/S1a/S1d/Landsat) | ✅ encoder tensor and NumPy paths |
| `spatial_split.py` | `cross_location_split` — disjoint state groups, prevents spatial leakage | ⚠ System A only |
| `modality_robustness.py` | `evaluate_modality_robustness` — runs every modality subset, reports dense metrics per subset | ❌ no callers |
| `spatial_viz.py` | quantile colour-class renderer, SVD/PCA pseudocolour | ❌ no callers |
| `sar_normalize.py` | S1 VV/VH backscatter normalisation | ✅ by `sentinel_loader` |

---

## 8. Entry points

| script | system | data source | status |
|---|---|---|---|
| `main_multimodal_finetune.py` | B | `make_synthetic_batch` (`:185`) — `torch.randn` | runs only on CUDA; trains on noise |
| `main_pretrain_multimodal.py` | B | `make_synthetic_batch` (`:130`) — `torch.randn` | runs only on CUDA; trains on noise |
| `main_finetune_mmst_vit.py` | A | real county loaders | needs Tiny-CropNet data + JSON config |
| `main_pretrain_mmst_vit.py` | A | real county loaders | needs Tiny-CropNet data + JSON config |
| `config/build_config_soybean.py` | A | `input/county_info_*.csv` | **broken from repo root** — hardcoded `./../` paths (`:7`, `:16`) require cwd=`config/`; also `open(path,"x")` fails if the file exists and `./../data/` is never created |
| `primae_vandv_entry.py` | — | delegates to `.primae/vandv/probe.py` | **`.primae/vandv/` does not exist** → always returns `ok: False` |

Both System B scripts accept `--log_dir` but never construct a `SummaryWriter`
(System A's scripts do, at `main_finetune_mmst_vit.py:211` and
`main_pretrain_mmst_vit.py:133`). The flag is accepted and silently ignored, and
System B produces no TensorBoard output at all.

---

## 9. YieldSAT point model — exact architecture

`models_yieldsat.YieldSATPointModel` is the model used for every run in
`results/yieldsat/` (2026-09-28). The values below are the defaults of
`main_yieldsat_finetune.py` and are the configuration actually trained:
`embed_dim=128`, `num_latents=8`, `depth=2`, `num_heads=4`, `modality_embed=32`,
`modality_dropout=0.1`, crop context on. Shapes are per batch of B grid cells;
T = 24 time slots. Data preparation is specified in
[yieldsat_data_contract.md](./yieldsat_data_contract.md) §7 (YS-04/05).

### 9.1 Inputs (from `YieldSATPointDataset`)

| Stream | Values | Mask | Extra |
|---|---|---|---|
| `yieldsat_s2` | (B, 24, 12): B01–B12 incl. B8A, standardized, 0 where invalid | (B, 24, 12) bool | time features (B, 24, 3) |
| `yieldsat_weather` | (B, 24, 4): temp_max, temp_mean, temp_min, total_prec (interval sums) | (B, 24, 4) | same time features |
| `yieldsat_dem` | (B, 1) | (B, 1) | – |
| `yieldsat_terrain` | (B, 4): aspect, curvature, slope, twi (5 with cyclic aspect) | (B, 4) | – |
| `yieldsat_soil` | (B, 48): 8 properties × 6 depths, property-major (96 with uncertainty) | (B, 48) | – |
| crop | (B,) id in {corn, rapeseed, soybean, wheat} | – | – |

Time features per slot are (days since seeding)/365, sin(day of year) and cos(day
of year), zeroed for ineligible slots. `models_yieldsat.pack_inputs` packs each
stream into one tensor for `MultiModalEncoder`: temporal streams as
`[values·mask, mask, time]` = (B, 24, 2C+3); static streams as
`[values·mask, mask]` = (B, 2C). Per-stream availability is 1 if any value of the
stream is valid.

### 9.2 Per-stream encoders (`models_multimodal_encoder.py`)

Every encoder outputs **one token (B, 1, 128)**.

**`MaskedTemporalEncoder`** (`yieldsat_s2`, `yieldsat_weather`; separate weights):
```text
x_t = Linear(2C→128)([v·m, m]_t) + Linear(3→128)(time_t) + slot_embed[t]   slot_embed (1,24,128), trunc-normal 0.02
seq = [CLS; x_1..x_24]                                                      CLS (1,1,128), trunc-normal 0.02
key_padding_mask: slot t masked if no channel valid at t (CLS never masked)
2 × TransformerEncoderLayer(d=128, heads=4, ff=256, GELU, dropout 0, pre-norm)
out = Linear(128→128)(LayerNorm(seq[CLS]))
```
Parameters: s2 288,640; weather 286,592.

**`MaskedStaticEncoder`** (`yieldsat_dem` C=1, `yieldsat_terrain` C=4):
```text
out = Linear(64→128)(GELU(Linear(64→64)(GELU(Linear(2C→64)([v·m, m])))))
```
Parameters: dem 12,672; terrain 13,056.

**`SoilProfileEncoder`** (`yieldsat_soil`, P=8 properties, D=6 depths):
```text
depth token d = [values of 8 props at d, masks of 8 props at d]   (16 features; 32 with uncertainty)
x_d = Linear(16→64)(token_d) + depth_embed[d]                     depth_embed (1,6,64)
seq = [CLS; x_1..x_6], depths with no valid property masked
1 × TransformerEncoderLayer(d=64, heads=4, ff=128, GELU, dropout 0, pre-norm)
out = Linear(64→128)(LayerNorm(seq[CLS]))
```
Parameters: 43,456.

**Missing streams.** `MultiModalEncoder.forward_with_missing` substitutes a
learned per-stream token (128, zero-init; 640 parameters total) for samples whose
stream is unavailable. During training, **modality dropout 0.1** removes each
stream per sample with probability 0.1. Absent streams are also masked out of the
fusion cross-attention (below).

### 9.3 Fusion (`LatentFusionTransformer`, `models_latent_fusion.py`)

```text
tokens: 5 stream tokens + 1 crop token = Embedding(4, 128)(crop)       (6, 128) per cell
each token += concat(pos_embed (96), modality_embed (32))
    pos_embed: learnable SeparablePosEmbed(spatial=1, temporal=1, 96), sin-cos init
    modality_embed: learnable (1,1,32) per stream, sin-cos init of the stream index
latents: learnable (1, 8, 128), randn init
cross-attention: MultiheadAttention(128, 4 heads)(query=latents, key/value=tokens,
                 key_padding_mask = unavailable streams)  → latents = LayerNorm(latents + attn)
2 × TransformerEncoderLayer(d=128, heads=4, ff=512, GELU, dropout 0, post-norm) over the 8 latents
LayerNorm → fused latents (B, 8, 128)
```
Parameters: 465,472 (crop embedding 512). Linear layers are trunc-normal 0.02
initialized, LayerNorms to (1, 0).

### 9.4 Head (`ScalarCellYieldHead`, registry name `scalar_cell_yield`)

```text
y = Linear(128→1)(Dropout(0)(GELU(Linear(128→128)(LayerNorm(mean over 8 latents)))))   → (B,)
```
Parameters: 16,897. Output is the standardized target; predictions are inverted
with the training mean/std to t/ha.

### 9.5 Totals, loss and training

- **Total 1,127,937 parameters** (encoders 644,416 + missing tokens 640 + fusion
  465,472 + crop 512 + head 16,897).
- **Loss:** MSE on the standardized per-cell target over valid targets.
- **Optimization:** AdamW (lr 1e-3, weight decay 0.05), 1 warmup epoch then
  cosine decay, gradient clipping at 1.0, bf16 autocast, batch 512 cells drawn
  field-balanced (season probability ∝ n_rows^0.5).
- **Model selection:** best epoch by validation pixel RMSE.

**Pretraining variant** (`yieldsat_objectives.YieldSATPointPretrainer`, run
`pretrain_pooled`). It uses the same encoders and fusion, with the crop token
disabled. Heads are one Linear(128→24·C) per temporal stream and Linear(128→C)
per static stream for masked-observation reconstruction (30% of observed values
hidden), plus Linear(128→12) for the last-valid optical forecast. Loss = masked
MSE + forecast MSE. Only encoders and fusion (156 tensors) are transferred via
`sensor_state_dict`.

### 9.6 What this model is not

- Its Perceiver fusion is under-used: it reads 6 summary tokens through 8
  latents with constant positional embeddings. The planned upgrade to
  per-slot/per-depth tokens with per-token masks and date-aware positions is task
  **YS-11** ([yieldsat_data_contract.md §8](./yieldsat_data_contract.md#8-model-improvement-tasks)).

- Each cell is predicted independently. There is no spatial context between
  cells: patch mode is not implemented.
- It is not a native-grid model. It consumes the preprocessed common 10 m grid.
- It is not one of the paper's baselines. For the paper-comparison plan see
  [yieldsat_paper_comparison.md](./yieldsat_paper_comparison.md).
