# Architecture

Tree state: commit `c8be363`. ~9,700 lines of Python across 44 modules.

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
field grid**, not absolutely. This is a defensible design — the alternative demands
per-modality positional encodings in projected world coordinates — but it should be
stated plainly rather than implied away.

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

### Per-modality normalisation is effectively dead

`normalize_modality()` from `util/norm_stats.py` is only invoked when the input is a
`np.ndarray` (`:293`, `:309`, `:378`, `:394`). Every caller in the repo passes
`torch.Tensor`, so normalisation never runs on the live path. See defect **D5**.

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
mismatch guard at `:267` silently truncates `pos` to the embedding's token count —
which is correct for the 1-token missing-modality case it was written for, but
silently absorbs genuine layout bugs. See defect **D3**.

`forward_multiscale` (`:287`) re-runs the same stack and harvests latents at four
intermediate block indices for the FPN/DPT heads. It duplicates ~40 lines of
`forward`; the two must be kept in sync by hand.

### Attention masking

`_build_key_padding_mask` (`:209`) builds a `(B, ΣL)` mask so absent-modality tokens
do not contribute. **It is currently broken** — the mask is built as a float tensor,
which PyTorch treats as *additive bias*, not exclusion. This is defect **D1** and is
the highest-severity finding in this review.

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
| `norm_stats.py` | per-modality mean/std registry (S2/S1a/S1d/Landsat) | ⚠ imported but never reached — defect D5 |
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
