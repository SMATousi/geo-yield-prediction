# Mission

## Core statement

Build a **field-scale multimodal geospatial foundation model** for crop yield
prediction that encodes each data source at its own native resolution and in its
own scientific units, fuses them in a shared latent space, predicts **dense
within-field yield maps** rather than a single county average, and keeps working
when input modalities are absent.

## Why this exists

The upstream MMST-ViT predicts one yield number per county-year. That is the right
granularity for USDA reporting and the wrong granularity for agronomy: the decisions
that matter — variable-rate seeding, fertiliser prescription, drainage, zone
management — are made *within* a field, at sub-hectare scale.

Three limitations of the upstream design motivated this work:

**1. Resolution loss through common-grid resampling.** Forcing DEM, SAR, optical,
soil, and categorical land-cover onto one raster grid destroys exactly the
high-frequency spatial structure that within-field variability lives in. A 1 m
LiDAR DEM resampled to a 9 km county cell carries no terrain signal at all.

**2. Fragility to missing inputs.** Real agricultural monitoring is incomplete by
default: cloud cover kills optical overpasses, weather stations fail, soil surveys
go stale, SAR revisit is irregular. A model whose forward pass requires all inputs
is unusable in production.

**3. Single-task architecture.** Yield is one of several questions asked of the same
inputs (crop stress, soil property inference, management-zone segmentation). Baking
the task into the encoders means re-architecting for every new question.

## Objectives

### Primary
1. Predict field-scale yield with **within-field spatial resolution** (dense per-pixel
   output, not a scalar).
2. Encode every modality with a **dedicated encoder matched to its structure**:
   convolutional for rasters, 3D-convolutional for multitemporal SAR, temporal for
   weather series, MLP for tabular soil, embedding for categorical crop history.
3. Provide **graceful degradation** — any subset of modalities produces a valid
   forward pass, with measured rather than assumed degradation.
4. Support an **extensible head registry** so new tasks attach to the shared backbone
   without touching the encoders.
5. Support **self-supervised pretraining** on unlabelled fields, then supervised
   fine-tuning on the much scarcer yield-monitor labels.

### Secondary
6. Preserve geographic honesty in evaluation: **spatial splits**, not random splits.
7. Quantify each modality's marginal contribution via systematic ablation.
8. Produce interpretable spatial outputs (yield maps, management zones, uncertainty).

## Success criteria

These are the bar this project is measured against. **None are currently met** — see
[status.md](./status.md). They are written here as targets, not claims.

### Functional
- [ ] One training run completes end to end on **real georeferenced field data**.
- [ ] Dense yield-map RMSE reported against held-out yield-monitor rasters.
- [ ] Ablation table showing RMSE as each modality is dropped, on real data.
- [ ] Spatial (cross-location) split used for the headline metric; random-split
      numbers reported separately and labelled as optimistic.

### Architectural
- [x] Six modality types encoded by dedicated native-resolution encoders.
- [x] Shared fusion backbone with spatial + temporal + modality-identity encodings.
- [x] `HEAD_REGISTRY` supporting ≥3 registered task heads (currently 4).
- [ ] Missing-modality path verified correct under test (currently defective — see
      [status.md](./status.md), defect **D1**).
- [ ] Pretrained backbone demonstrably transfers: fine-tuning from pretrained weights
      beats training from scratch on the same split.

### Engineering
- [ ] `pip install -r requirements.txt` succeeds in a clean container.
- [ ] `--device cpu` path runs, so the model is debuggable without a GPU.
- [ ] Test suite covering encoder shape contracts, fusion masking, head output
      shapes, and dataset registration.
- [ ] CI running that suite on every commit.

## Scope

### In scope
- Field-scale (not county-scale) dense yield prediction.
- Heterogeneous multimodal fusion at native resolution.
- Missing-modality design **and its evaluation**.
- Extensible multi-head architecture on a shared backbone.
- Self-supervised pretraining + supervised fine-tuning.
- Geospatial registration and field-boundary alignment.

### Out of scope
- Real-time / low-latency inference. This is a batch system.
- Data acquisition infrastructure (satellite APIs, weather procurement, yield-monitor
  ingest hardware). The spec assumes data arrives as GeoTIFFs.
- Crop insurance underwriting or any regulated decision. Advisory output only.
- Variable-rate controller integration. The system emits rasters; translating those
  into machine prescriptions is downstream.
- Replacing or maintaining the upstream county-level MMST-ViT path. It is retained
  as a reference baseline, not developed.

## Design principles

1. **Native-first.** Resample for visualisation, never before feature extraction.
2. **Modality-optional.** No input is required. Degradation must be measured, not
   assumed.
3. **Foundation-first.** Encoders serve many tasks; resist task-specific leakage
   into the encoder layer.
4. **Geospatial-explicit.** Position, ground resolution, CRS, and acquisition time
   are first-class tensor metadata, not implicit in array order.
5. **Honest evaluation.** Spatial splits by default. A metric that cannot be
   reproduced from a committed script does not exist.
6. **Synthetic data is a smoke test, never a result.** Any number produced from
   `torch.randn` inputs must be labelled as such at the point it is printed.
