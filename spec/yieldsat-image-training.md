# YieldSAT 64×64 Image Dataset and Training

**Status:** Dataset built and verified, 2026-09-30; image-model training is
specified below but not implemented. The image dataset is
derived from the implemented [YieldSAT point contract](./yieldsat_data_contract.md).
Its source is the four read-only preprocessed country NetCDF files; its output root
is `/home1/pupil/SMATousi/YieldSAT-Image` by default. This dataset does not restore
original native-resolution rasters: the YieldSAT inputs were merged onto 10 m grids
upstream.

## Sample and tiling rules

One image is a **single field-season** (same `field_shared_name`, country, crop and
year) and one non-overlapping 64×64 window on that field's verified DEM grid. Use
the decoded integer `grid_row` and `grid_col` from the prepared index and the
field's recovered width, height, EPSG and affine transform. Window origins are
`(64 × tile_row, 64 × tile_col)`, anchored at grid origin `(0,0)`; stride is 64 in
both directions. The edge of a field grid is padded, never wrapped or filled with
cells from another field. A stored tile must contain **at least 2,048 valid pixels**
(50% of 4,096). Valid means a unique, in-bounds source row with finite per-cell
yield and at least one usable sensor value. This definition makes each tile usable
for both supervised image training and pretraining. Occupied rows failing the
validity rule do not count toward coverage or a training target.

Do not concatenate the corpus's arbitrary row order into a square. Retain a
`source_row` map (`-1` for holes) to prove which NetCDF row occupies each pixel.
Reject duplicate `(field-season,row,col)` entries; validate year/crop against the
index and geometry. No tile may contain pixels from different field-seasons.
Different years on the same physical ground are separate images and must remain
in the same train/test group when splitting.

Using the existing index and geometry, a preliminary count at this fixed origin
found **2,437 candidate windows** with at least 2,048 finite-target rows:
Argentina 1,220; Brazil 920; Germany 68; Uruguay 229. The final count also checks
sensor validity during materialization. These are fewer than the simple
12,372,920/4,096 arithmetic bound because fields have boundaries, holes and
partially filled windows. Changing origin or introducing overlapping windows
would create a different dataset version.

## Stored tensors and provenance

Store one HDF5 file per country and a JSONL patch table, with a top-level manifest.
HDF5 datasets use channel-last spatial axes and raw decoded feature values:

| Dataset per patch | Shape | Meaning |
|---|---|---|
| `temporal` | `(64,64,24,16)` float32 | 12 S2 and 4 weather channels in the canonical registry order; NaN where missing. |
| `static` | `(64,64,104)` float32 | DEM, terrain, 48 soil properties, 48 uncertainties and 3 coordinate channels; NaN where missing. |
| `times` | `(64,64,24)` float32 | Days since 1970-01-01; NaN for undated slots. |
| `target` | `(64,64)` float32 | Per-cell yield, t/ha; NaN where absent. |
| `valid_pixel` | `(64,64)` bool | The above eligibility definition. |
| `source_row` | `(64,64)` int32 | Original country-local NetCDF row, `-1` for holes. |

The file's first dimension is the number of patches. A patch-table record includes
index, country, field-season, crop, year, physical-field ID, source-row range,
window origin and actual height/width, occupancy/valid count, EPSG, tile affine,
source fingerprint and cache version. The affine is shifted from the field affine
by `(col0,row0)`; padded edge cells remain invalid. Store canonical channel-name
lists and units in the manifest. In particular, `coord_*` and soil uncertainty
are preserved for reproducibility but are **not** default model streams.

Missing features remain NaN in storage. Training loaders must derive independent
per-feature/time masks, apply a selected observation cutoff, fit normalization
only on training groups, and fill invalid normalized values only after masking.
No train/test assignment or normalization is baked into images. A pre-harvest
model must discard post-cutoff observations; the stored tiles retain the full
24-slot histories for future retrospective and forecasting protocols. Yield and
field-level yield metadata are target-side only, never sensor inputs. The first
weather slot's unrecorded interval remains invalid under the point contract.

The dataset is **image-shaped**, but each pixel keeps its own temporal history.
An image trainer must explicitly choose how to encode `(H,W,T,C)` and static
layers, preserve pixel/feature/time masks and prevent masked cells from affecting
losses. The existing point model is not silently a 64×64 image model.

## Image training contract (planned)

**Task.** Predict yield in t/ha for each eligible 10 m cell of a 64×64
field-season tile, using sensor observations available by a declared cutoff
and optional known-in-advance crop identity. Output is a 64×64 map, not a
single tile value. These are design requirements, not a completed experiment.

### Inputs and separate encoders

The default satellite backbone is provisionally [Meta DINOv3 ViT-L/16
pretrained on SAT-493M](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m),
using [Meta's official implementation and preprocessing
contract](https://github.com/facebookresearch/dinov3). Pin the exact checkpoint
revision/hash, loader version, license and preprocessing in each run manifest.
Use spatial patch tokens, never only the global/class token. This checkpoint
expects three optical channels; it is not a pretrained 12-band Sentinel-2
encoder. Keep the model choice configurable so a different satellite DINO
checkpoint can be adopted with an explicit channel contract.

| Stream | Stored channels | Planned encoder and output |
|---|---|---|
| RGB optical | S2 B04/B03/B02 from `temporal`, in red/green/blue order | DINOv3 SAT ViT-L/16 on selected observations; retain spatial patch tokens and project to fusion width. |
| Additional S2 bands | B01, B05, B06, B07, B08, B8A, B09, B11, B12 | Separate learned spectral encoder with missing-band masks; fuse date-aligned spatial features with DINO tokens. This preserves all 12 bands without altering the pretrained RGB input layer. |
| Weather | `temp_max`, `temp_mean`, `temp_min`, `total_prec` across 24 slots | Separate masked temporal encoder with interval/date features; broadcast or cross-attend its summary to spatial cells. Audit whether coarse upstream weather varies meaningfully within a tile before using a spatial weather CNN. |
| Elevation and terrain | DEM (1); aspect, curvature, slope, TWI (4) | Independent DEM and terrain masked spatial encoders; encode aspect cyclically. Track coarse native resolution despite the stored 10 m grid. |
| Soil | Eight properties × six depth intervals (48) | Depth-aware soil encoder per cell, then a lightweight spatial encoder; preserve depth/property masks. Soil uncertainty (48) is an optional reliability input/ablation, not a default feature. Track its coarse native resolution. |

Trainable non-DINO encoders should emit a common fusion width (initial design:
128–256). The input API must carry independent per-feature, per-cell and
per-time validity. Crop identity may be used only when known at the forecast
cutoff. Coordinate channels, `source_row`, target values, field-level yields
and harvest outcome metadata are excluded. Stored `valid_pixel` depends on
finite yield and is **loss/evaluation-side only**; construct model sensor masks
from feature finiteness and dates, and mask padded/outside-field cells
separately. Audit whether source-row occupancy reveals target availability
before claiming whole-field deployment.

### Optical observations and spatial fusion

Use raw decoded S2 bands from HDF5, never percentile-stretched preview TIFFs.
Establish how YieldSAT's decoded S2 values map to the checkpoint's expected
RGB `[0,1]` scale before training; do not silently clip or treat raw reflectance
as display bytes. Meta's SAT-493M transform uses RGB mean
`(0.430, 0.411, 0.296)` and standard deviation `(0.213, 0.156, 0.143)` after
scaling to float. Persist the exact conversion and fit any extra statistics
only on training groups. Fill missing optical channels after creating masks, with
a documented neutral value compatible with checkpoint normalization. DINO
itself has no nodata mask, so carry optical coverage as a token/cell mask
through fusion and assess partially observed patches. Do not treat upscaled
source pixels as independent high-resolution observations.

Choose at most four eligible optical observations per tile as an initial memory
budget, spread through the pre-cutoff season using coverage and dates, never
yield. Training can sample within fixed seasonal bins; validation/test choice
must be deterministic. Group cells by actual acquisition date or explicitly
verify slot-date coherence: one slot index does not guarantee that neighbors
were observed on one day. Keep per-pixel time masks and date features (days
since seeding and cyclical day of year). If no optical observation is usable,
emit a missing-optical token while other streams remain available.

A 64×64 image yields only a 4×4 token grid with a `/16` DINO backbone. Fuse
date-aligned DINO and extra-band features with weather, DEM, terrain and soil
features at a spatial token grid, retaining modality and missingness masks.
Use a trainable multiscale decoder (for example FPN/U-Net style) with genuine
64×64 optical/static skip features to predict each cell. Upsampling DINO's
4×4 tokens alone does not recover 10 m detail. Ablate fusion/decoder choice;
the point Perceiver informs the design, but its scalar head and point-only
stream summaries are not an image model.

### Training and evaluation

Use YieldSAT's grouped physical-field/farm/block/country split artifacts;
keep all years and tiles from the same held-out physical ground in one split.
Check neighboring tiles and cross-farm aliases. Fit input and target
normalization on training groups only. Record the forecast cutoff and exclude
post-cutoff observations. Apply supervised loss only to finite target cells.
Report pixel RMSE/MAE in t/ha plus field-balanced, per-country and per-crop
metrics. Aggregate loss per tile/field so large fields do not dominate merely
by pixel count. Inference output coverage must not depend on the target mask.
Preserve the point contract's rule that each row's first dated weather interval
is invalid because its interval start is unknown.

Start with frozen DINO features while training other encoders, fusion and
decoder. Cache frozen features only with a key containing checkpoint,
preprocessing, date selection and split, and never holdout labels. If a memory
pilot supports it, compare selectively unfrozen final DINO blocks with a
lower backbone learning rate, mixed precision and activation checkpointing.
First smoke-test checkpoint loading, 64×64 positional handling, variable
missingness, forward/backward and peak memory on the available GPU. Choose
batch size and observation count from that measurement, not point-model values.

Compare with the point model on the **same held-out physical fields and cells**,
with matching cutoffs and label budgets. Required ablations: pretrained versus
randomly initialized optical backbone; frozen versus partial fine-tuning;
RGB versus RGB plus nine-band branch; optical/weather/terrain/soil removals;
spatial context versus per-cell prediction. Distinguish DINO's external optical
pretraining from optional YieldSAT sensor-only or [expert relationship
pretraining](./knowledge_pretraining.md). The yield head never receives text
at inference. Report size, GPU memory, throughput, seeds, split IDs and
uncertainty in improvements.

### Implementation tasks and acceptance

1. **YI-01 — Data audit and loader:** implement image HDF5 loader with grouped
   splits, cutoff, exact band maps, independent sensor/target masks, train-only
   normalization and deterministic evaluation dates. Verify source manifest
   and date coherence across the corpus.
2. **YI-02 — Optical backbone:** pin/load satellite DINO; reproduce RGB
   preprocessing and validate 64×64 patch-token output on real tiles. Add the
   nine-band spectral branch and missingness handling. Reject a run if the
   checkpoint/channel contract mismatches.
3. **YI-03 — Other encoders and dense fusion:** add masked weather, DEM,
   terrain and depth-aware soil encoders, date fusion, spatial decoder and
   64×64 yield head. Verify shapes and gradients under missing modalities and
   padded cells.
4. **YI-04 — Training:** implement frozen-backbone supervised training,
   optional partial fine-tuning, mixed precision, field-balanced sampling,
   checkpoint provenance and resumable configuration. Profile GPU and I/O.
5. **YI-05 — Evaluation:** enforce group-leakage checks and matched point/image
   holdouts; publish t/ha metrics, map readback and the ablations above.
   Acceptance requires an end-to-end run on all four countries with declared
   holdouts, not only a loader or visually plausible map.

## Dataset construction plan and acceptance

1. **Validate sources:** compare prepared index/cache fingerprints against each
   original NetCDF; use the verified geometry table; check source-size, field
   contiguity, duplicate positions and row/col bounds. Keep source/cache read-only.
2. **Plan tiles:** for each field-season independently, partition its grid into
   disjoint 64×64 windows; count finite-target rows and prefilter below 2,048.
   Record candidate and excluded counts by country.
3. **Materialize:** gather rows from the canonical prepared cache, verify actual
   sensor/target validity, and write dense patch arrays, masks, row maps and
   provenance to the requested output root. Publish each country atomically from
   temporary files; never publish a partial country as complete.
4. **Verify:** read back every patch's shape, masks, row mapping, non-overlap,
   field/year consistency and ≥2,048 valid pixels. Check a bounded sample against
   the original cache and the shifted geometry. Publish a final manifest with
   totals and source fingerprints only after all selected countries pass.
5. **Train later:** implement YI-01–YI-05 above. Image-model training was not
   part of this dataset construction task.

Build command (using the existing prepared artifacts):

```bash
.conda/phase0/bin/python yieldsat_build_images.py \
  --source-root /home1/pupil/SMATousi/YieldSAT/Preprocessed \
  --artifact-root /root/yieldsat_artifacts \
  --output-root /home1/pupil/SMATousi/YieldSAT-Image
```

The output lives outside the Git repository. Commit the spec, builder and tests,
not the multi-gigabyte dataset. The builder may create the explicitly requested
output directory but must not alter the YieldSAT source or prepared artifacts.

## Build record — 2026-09-30

The command above completed for all four countries. The published output includes
one compressed `images.h5` and `patches.jsonl` per country, plus `manifest.json`
at the root. The payload is 2,007,061,803 bytes (~2.01 GB decimal). Counts:

| Country | Images | Valid pixels |
|---|---:|---:|
| Argentina | 1,220 | 3,374,971 |
| Brazil | 920 | 2,550,502 |
| Germany | 68 | 174,698 |
| Uruguay | 229 | 566,176 |
| **Total** | **2,437** | **6,666,347** |

The builder verified every patch's minimum coverage, source-row mapping and
field-season bounds before publishing. An independent readback checked each
patch's tensor dimensions, mask counts, finite valid targets, aligned window
origins and globally unique source rows within each country. Bounded tensor
readbacks matched the prepared cache. The new tiling tests and existing YieldSAT
tests passed (38 total). The output is an image *dataset*; image-model training
is the remaining step 5 above.

## Random GeoTIFF previews — 2026-09-30

Twelve patches (three per country) were selected reproducibly from the image
dataset with seed `20260930` and exported to the sibling directory
`/home1/pupil/SMATousi/YieldSAT-Image-Samples`. Each selected patch has:

- a 64×64, four-band RGBA GeoTIFF using S2 B04/B03/B02 at the temporal slot with
  the most dated, valid RGB pixels. The RGB channels are stretched to the 2nd–98th
  percentile **for display only**; transparent pixels lack valid RGB. The slot
  index is stored in tags and `samples.json`. Different cells at one slot can have
  different acquisition dates, so the TIFF is not necessarily a single-date scene;
- a matching one-band float32 yield GeoTIFF in t/ha with internal validity mask
  and `-9999` nodata. Both files retain the original patch CRS and affine.

The previews are not the full 24-slot or 120-channel training tensors; that history
remains in the HDF5 dataset. The visual previews apply no pre-harvest cutoff and
must not be used as training input for a forecasting experiment.

```bash
.conda/phase0/bin/python -B yieldsat_export_tif_samples.py \
  --dataset-root /home1/pupil/SMATousi/YieldSAT-Image \
  --output-root /home1/pupil/SMATousi/YieldSAT-Image-Samples \
  --per-country 3 --seed 20260930
```

The published `samples.json` lists exact country/patch IDs, chosen slots and TIFF
paths. Readback verified all 24 TIFFs against the image dataset, including 64×64
shape, CRS/affine, RGB alpha, yield mask and target pixel values. TIFFs are stored
outside Git; the exporter and this documentation are committed.

## Full temporal preview sequences — 2026-09-30

`yieldsat_export_temporal_sequences.py` selects one of the three existing
sample tiles per country, prioritizing the number of nonempty optical slots
then total dated RGB coverage. It writes all 24 slot positions as georeferenced
RGBA GeoTIFFs and a labeled `contact_sheet.png` under
`YieldSAT-Image-Samples/TemporalSequences/<country>/`. RGB uses one display-only
2nd–98th percentile stretch per band across that tile's full sequence, allowing
visual comparison between its dates. Each TIFF retains the slot index, date
range, valid RGB count, field and year as tags. `sequences.json` records the
selection and metadata. If a slot has no dated RGB values, its TIFF is entirely
transparent and the sheet explicitly labels it “No RGB observation”; no image
is synthesized. A slot can contain more than one acquisition date across cells.

| Country | Patch | Slots with dated RGB |
|---|---:|---:|
| Argentina | 508 | 9 of 24 |
| Brazil | 810 | 8 of 24 |
| Germany | 64 | 9 of 24 |
| Uruguay | 75 | 7 of 24 |

```bash
.conda/phase0/bin/python -B yieldsat_export_temporal_sequences.py
```

The published output has 96 slot GeoTIFFs and four contact sheets. Readback
verified file counts, 64×64 shape, alpha coverage and date tags against
`sequences.json`. These are previews of the RGB part of the stored time series;
the HDF5 retains all 12 optical bands, weather and per-cell timestamps.

## Implementation plan v1 — 2026-10-01

This section turns the planned contract above into concrete decisions for the
first trainable version (tasks YI-01 to YI-08). Every step below is implemented,
documented here and committed separately. Status per task is in the progress
log at the end.

### Comparability with the point model

- **Same folds, same field seasons.** Image runs use the *point suite's fold
  manifests* (`paper_<cv10|loro|loyo>_<pair>_<group>_<policy>_s0_foldNN`), not
  new splits. A tile inherits the partition (train/val/test/excluded) of its
  field season (`<Country>/<field_shared_name>`). Tiles never straddle seasons,
  so this is exact. LORO uses province regions for ARG-C/S/W and farm regions
  elsewhere, exactly as the point suite after 2026-09-30.
- **Same matrix.** CV10, LORO and LOYO; `paper` and `strict` leakage
  policies; seeds 0–2; the retrospective `all_slots` cutoff; train-fold
  normalization.
- **Inputs.**
  - Image **S2** = RGB through DINO plus the nine-band spectral branch (all 12
    S2 bands).
  - Image **S2+ADM** additionally uses the weather, DEM, terrain and soil
    encoders.
  - Excluded, as in the point model: `coord_*`, soil uncertainty, `source_row`
    and all target-side metadata.
- **Same cells.** Image test metrics are computed on the valid pixels of test
  tiles. `yieldsat_image_compare.py` re-scores the point model's
  `test_predictions.npz` on exactly those cells (matched by field season,
  grid row and grid col via `source_row`). Image-versus-point tables therefore
  compare identical held-out cells; the point model's whole-field scores are
  reported separately.
- **Output format.** Image runs write `test_predictions.npz` in the point
  format (`pred, target, season, grid_row, grid_col, season_names`) and the
  same `report.json` fields, so `aggregate_folds`, `yieldsat_paper_compare.py`
  and the cluster driver work unchanged.

### Model (YI-02/YI-03): one encoder per modality, Perceiver aggregation

Width `D = 192`. Tile `64×64`, `T = 24` slots, up to `K = 4` optical
observations.

| Stream | Encoder | Tokens to the Perceiver | Dense skip features |
|---|---|---|---|
| RGB (B04/B03/B02) | **Frozen** DINOv3 ViT-L/16 SAT-493M (pinned revision), run on each selected observation; 4×4 patch tokens (1024-d) → Linear → D, plus observation-date encoding | K × 16 | – |
| 9 extra S2 bands | Per observation: `[value·mask, mask]` → 1×1 conv → 3×3 convs (32 ch at 64×64) → patchify(16) → D, plus date encoding | K × 16 | 64→8 pyramid (mean over observations) |
| Weather (4 ch × 24 slots) | Point `MaskedTemporalEncoder` on the tile's weather history (constant within a field), first-dated-slot weather invalid | 1 | – |
| DEM | Masked conv encoder (2 ch: value, mask) | 16 | pyramid |
| Terrain (aspect as sin/cos, curvature, slope, TWI) | Masked conv encoder | 16 | pyramid |
| Soil (8 properties × 6 depths) | Per cell depth-aware MLP (properties × depths + masks) → convs | 16 | pyramid |
| Crop identity | Embedding (known at planting) | 1 | – |

- **Fusion.** The point model's `LatentFusionTransformer` (Perceiver;
  required) reads **all** tokens: up to 32 optical plus 50 others, with
  per-token masks. Missing observations, padded or empty patches, and absent
  streams are masked. It also gets modality-identity embeddings and a 4×4
  spatial position per patch token. It has 32 latents, 2 cross-attention
  reads, depth 4 and pre-norm.
- **Decoder.** Perceiver-IO style: 16 learned spatial queries (4×4 grid)
  cross-attend to the latents → 4×4×D map → four ×2 upsampling stages to 64×64,
  each concatenating the pyramid skip features at matching resolution → 1×1
  conv → yield. Outputs cover every cell; loss and metrics use `valid_pixel`
  only.
- **Loss.** MSE on the standardized target over valid cells, averaged per tile
  and then over tiles, so large tiles don't dominate. Batches are sampled
  field-balanced, by field season.

### Data handling (YI-01)

- **Masks** are built from feature finiteness and dates only. `valid_pixel` is
  used only for the loss and metrics. Padded cells (`source_row == -1`) are
  invalid inputs.
- **Normalization** uses per-channel mean/std from the training tiles of each
  run (valid values only) and target mean/std from the training tiles' valid
  cells. Never the supplied `stats-*`.
- **Observation selection.** Slots with ≥ 25% of the tile's cells carrying
  finite RGB qualify. The season (seeding → harvest) is split into K equal
  bins. Evaluation takes the slot with the most RGB coverage per bin,
  deterministically, with fallback to the best remaining slots. Training
  samples one qualifying slot per bin at random.
- **Date coherence** is audited per (tile, slot): the spread of `times` among
  observed cells must be ≤ 1 day for the slot to qualify, otherwise the slot
  is dropped and counted in the audit.
- **RGB scaling for DINO.** S2 L2A digital numbers are converted to
  reflectance (÷10,000), mapped to `[0,1]` with the common true-colour scaling
  reflectance/0.3, and clipped. The clip fraction is recorded per tile in the
  feature cache. The checkpoint's mean/std are then applied, and missing RGB
  pixels are filled with the mean (zero after normalization) plus a
  pixel/token mask.

### Frozen DINO feature cache (YI-02)

Because DINO is frozen, its patch tokens depend only on the checkpoint and the
RGB preprocessing, not on splits or labels. A one-off GPU job computes them for
**every** (tile, qualifying slot): ≤ 2,437 × 24 × 16 × 1024 × fp16 ≈ 1.9 GB.
They're stored per country in `dino_<rev>_<prep-hash>/<Country>.h5` next to the
image dataset. Training reads cached tokens; evaluation reads the same cache
deterministically. The cache key holds the checkpoint revision and the
preprocessing hash; a mismatch refuses to load.

### Small-country warm start (YI-06)

- **Rule.** A pair with < 100 tiles is warm-started: GER-R (41 tiles) and
  GER-W (27).
- **Donors** (user decision, 2026-10-01):
  - **GER-W** ← **BRA-W** donor: the same crop, from the wheat country with
    the most tiles (242).
  - **GER-R** ← **pooled multi-crop donor**: all non-German pairs, crop
    token on. No other country grows rapeseed.
- **Donor training.** Donors are trained per seed on all tiles of their donor
  pairs, with a 10% validation carve-out for epoch selection. Donor countries
  differ from the target, so no target-fold data is seen.
- **Fine-tuning.** It starts from the donor's weights, except the crop
  embedding and any shape-mismatched tensors. DINO stays frozen; lr ×0.3.
  Reports record the donor checkpoint and seed.

### Cluster execution (YI-07)

- **Infrastructure.** The same `yieldsat_cluster.py` plans, pools, W&B logging
  and completion markers. The model preset `image` calls
  `main_yieldsat_image.py`.
- **Order** on Nautilus:
  1. a CPU pod builds the image dataset on the PVC with
     `yieldsat_build_images.py`, from the PVC's index, cache and geometry;
  2. a GPU job builds the DINO feature cache;
  3. the donor plan;
  4. the main image plan (German runs reference donor checkpoints on the PVC).
- **Access.** W&B project `yieldsat-cvpr27-image`. The Hugging Face token is
  the namespace secret `smatousi-hf` (key `HF_TOKEN`). The checkpoint is
  downloaded once to the PVC and pinned by revision `f692fa42`.
- **Image.** The container is rebuilt with `transformers` (DINOv3 support) and
  `huggingface_hub`.

### Tasks

| ID | Deliverable |
|---|---|
| YI-01 | `dataset/yieldsat_image_dataset.py`: loader, fold mapping from point manifests, masks, train-only normalization, observation selection, date-coherence audit; tests |
| YI-02 | `models_yieldsat_image.py` DINO wrapper (pinned, frozen) + `yieldsat_image_dino_cache.py`; RGB preprocessing; tests with a stub backbone |
| YI-03 | Spectral, weather, DEM, terrain and soil encoders; Perceiver aggregation; spatial decoder; dense head; shape/gradient/missingness tests |
| YI-04 | `main_yieldsat_image.py`: training/eval entry (W&B, reports, point-format predictions, donor init) |
| YI-05 | `yieldsat_image_compare.py`: matched-cell image-vs-point tables per protocol/pair |
| YI-06 | Donor suite and warm-start wiring |
| YI-07 | Cluster: suites, presets, dataset/DINO-cache pod and job manifests, image rebuild |
| YI-08 | Nautilus smoke run (one GPU), then submission |

### Progress log

- 2026-10-01 — Plan v1 written (this section). Decisions: Perceiver aggregates
  all modality tokens; point fold manifests reused; GER-R ← pooled donor and
  GER-W ← BRA-W donor; DINO access via the user's HF token.
- 2026-10-01 — **YI-01 done.** `dataset/yieldsat_image_dataset.py` implements:
  - the tile table and `tiles_for_split` (partition inherited from a point
    fold manifest by season ID; excluded seasons and other countries/crops
    dropped);
  - per-worker HDF5 reading;
  - sensor masks from finiteness and dates, with first-dated-slot weather
    invalid;
  - cutoff modes `all_slots`, `before_harvest` and `harvest`;
  - a train-tile-only `ImageNormalizer` (≤ 200 training tiles);
  - optical observation selection (≥ 25% RGB coverage, date spread ≤ 1 day,
    K = 4 seasonal bins; deterministic for evaluation, random within bins for
    training);
  - tile-level weather (mean over observed cells), cyclic aspect,
    `cell_present` and `target_valid`;
  - per-cell grid row/col for point-format outputs, a field-season-balanced
    sampler and `audit_date_coherence`.

  **Audit on the full dataset:** 2,437 tiles, 16,939 (tile, slot) pairs with
  RGB, **0 date-incoherent** (all observed cells of a slot share one
  acquisition date). On real data, a GER-R CV fold maps to 33/4/4
  train/val/test tiles; one item loads in ~0.03 s. Tests:
  `tests/test_yieldsat_image_model.py` (4).
- 2026-10-01 — **DINO access.** The HF token is stored as the namespace secret
  `smatousi-hf`, but download of the gated checkpoint returns HTTP 403: the
  account still needs access approval on the model page. Until then the
  pipeline is developed against a stub backbone with the same interface.
- 2026-10-01 — **YI-02 done (pending checkpoint access).**
  - `models_yieldsat_image.py`: `rgb_to_dino_input` (DN/10000 → reflectance →
    /0.3 true-colour, clipped to [0, 1] with clip fraction → SAT-493M
    mean/std; missing pixels → mean → 0), `preprocessing_spec`/`_hash`,
    `FrozenDino` (pinned revision, no gradients, always eval, returns the 4×4
    patch tokens after the class and register tokens) and `StubDino` (same
    interface, fixed random patchify; tests and development only).
  - `yieldsat_image_dino_cache.py`: caches tokens for every (tile, slot) with
    any dated finite RGB to `dino_cache/<rev8>_<prep-hash>/<Country>.h5`
    (float16 `features` (M, 16, 1024), `patch_index`, `slot`, `coverage`,
    `clip_fraction`) plus `manifest.json`; atomic per country.
  - `DinoFeatureCache` in the dataset: refuses a cache with a different
    revision or preprocessing hash; `YieldSATImageDataset(dino_cache=...)`
    adds `dino` (K, 16, 1024) and `dino_valid` per selected observation.
  - `transformers`, `huggingface_hub` and `safetensors` added to
    `requirements.txt` and `environment.yml` (the Docker image must be
    rebuilt, YI-07).
  - Tests: 6 pass (stub cache build, lookup, mismatch refusal, dataset items).
  - Real cache: ~16.9k forward passes of ViT-L at 64×64 (a few minutes on
    one GPU); blocked only by the HF access approval.
- 2026-10-01 — **YI-03 done.** `YieldSATImageModel` in `models_yieldsat_image.py`:
  - **Encoders**, one per modality, each giving D-wide tokens on the 4×4
    patch grid:
    - `dino`: cached frozen tokens → LayerNorm → Linear.
    - `spec`: masked conv pyramid over `[bands·mask, mask]` per observation.
    - `weather`: the point model's `MaskedTemporalEncoder`, one summary token.
    - `dem`, `terrain`: masked conv pyramids.
    - `soil`: depth-aware per-cell MLP (shared over depths, plus a learned
      depth embedding, depths concatenated), then a conv pyramid.
    - `crop`: an embedding.
    Optical tokens get the point model's `DateEncoding` of the observation
    date. Dense streams also emit 64/32/16/8 skip maps, zeroed where no input
    cell is valid; spectral skips are averaged over valid observations.
  - **Fusion:** `LatentFusionTransformer` reads all tokens (K×16 DINO + K×16
    spectral + 1 weather + 3×16 static + 1 crop = 178 at K = 4) with per-token
    masks: invalid observation, patch with no valid cell, stream dropped.
    32 latents, 2 reads, depth 4, pre-norm, input norm.
  - **Training-time modality dropout:** p = 0.1 per stream and sample, never
    dropping all of a sample's streams.
  - **Decoder:** Perceiver-IO read-out (16 learned queries, 2 cross-attention
    + MLP layers) → 4×4×D, then four ×2 U-Net stages concatenating the skips
    → 1×1 conv. Output covers every cell.
  - **Loss:** `tile_balanced_mse` (per-tile mean over valid cells, then mean
    over tiles). NaN-safe on invalid cells; a test caught `0·NaN`.
  - **Size:** S2 4.4M and S2+ADM 6.3M trainable parameters (DINO excluded,
    cached). On an RTX 3090 at batch 16 with bf16: ~30 ms per step, ~1 GB
    peak.
  - **Tests (12 pass):** gradients reach every encoder, the fusion, decoder
    and head; S2-only builds no ADM encoders; no optical observation, padded
    rows, and all streams masked stay finite; masked observations have no
    effect on the output while valid ones do; tile balancing; end-to-end on
    a dataset batch with the stub cache.
- 2026-10-01 — **DINO access granted; checkpoint validated (YI-02 closed).**
  - Loaded `facebook/dinov3-vitl16-pretrain-sat493m` at revision `f692fa42`
    with the token from the `smatousi-hf` secret:
    - `DINOv3ViTModel`: 303M parameters, 0 trainable, hidden 1024, patch 16,
      4 register tokens. Stays in eval mode after `train()`.
    - A 64×64 input gives 21 tokens (1 class + 4 register + 16 patch); the
      wrapper returns exactly the 4×4 patch tokens.
  - **Real cache** built on the local RTX 3090 for all four countries in a
    few minutes: `f692fa42_b764aff546ec`, 531 MB.

    | Country | Tiles | (tile, slot) entries | Mean clip fraction |
    |---|---:|---:|---:|
    | Argentina | 1,220 | 8,628 | 0.05% |
    | Brazil | 920 | 6,275 | 0.83% |
    | Germany | 68 | 625 | 0.27% |
    | Uruguay | 229 | 1,411 | 0.45% |

    The low clip fractions confirm the reflectance/0.3 scaling. Tokens are
    finite (std 0.25). Mean-token cosine is 0.78 for the same tile on
    another date vs 0.68 across tiles.
- 2026-10-01 — **YI-04 done.** `main_yieldsat_image.py`:
  - **Cluster-compatible CLI:** accepts the point entry's flags
    (`--data_contract yieldsat_preprocessed_v1`, `--streams` S2 or all five,
    `--split`, `--countries`/`--crops`, budget, `--save_maps`, W&B; reuses
    `WandbLogger`).
  - **Training:** tiles inherit their season's partition from the point
    manifest; normalization from training tiles; DINO cache checked against
    revision and preprocessing hash. Season-balanced sampling (the sampler
    carries the epoch, so persistent loader workers resample deterministically
    per (seed, epoch, tile)); AdamW, cosine with warm-up, bf16; best
    checkpoint by validation pixel RMSE.
  - **Outputs:**
    - `report.json` with the point fields plus tiles, DINO provenance and
      transfer;
    - `test_predictions.npz` in the point format: one row per valid test
      cell, keyed by season + field-grid row/col. The builder's tiles don't
      overlap and use the point grid, so cells match point predictions
      exactly.
  - **Donor and warm start:** `--donor` trains on all tiles of the
    countries/crops with a 10% season validation split and no test.
    `--init_ckpt` warm-starts from a donor, skipping the crop embedding and
    shape mismatches.
  - **Default budget:** 60 epochs × max(20, one pass) steps of 16 tiles,
    lr 5e-4.
  - **Real-data smoke** (GER-R CV fold 0, S2+ADM, real DINO cache): 33/4/4
    tiles; 30 epochs ≈ 4 min on a 3090; 1.0 GB GPU memory. Test pixel RMSE
    1.54 t/ha, R² 0.03; train loss 0.13 means it overfits with 33 tiles,
    which is the case the donor warm start targets (YI-06).
  - **Open (I/O):** training reads ~45 tiles/s. One item costs 28 ms
    single-process and the standalone loader does ~105 tiles/s, but items
    take ~85 ms inside training workers. Being investigated before the
    cluster estimate (YI-07).
  - Tests: 14 pass, including an end-to-end donor → warm-start fold run
    and an S2-only run on the synthetic dataset.
- 2026-10-01 — **Loader I/O investigated and improved.**
  - **Finding:** a pure-NumPy multi-process benchmark on the development host
    stops scaling at ~2 processes (memory-bandwidth bound). The ~45 tiles/s
    cap is not caused by NFS (local copy: same), HDF5 handles, pinned memory
    or threads. An earlier 105 tiles/s reading was taken under different
    host load.
  - **Changes:**
    - an item now validates and normalizes only the channel groups it uses
      (RGB validity, the nine bands at the ≤ K chosen slots, weather) instead
      of the whole (64,64,24,16) array: 29 → 20 ms per item;
    - dense values travel as float16 and masks as bool (`cast_batch`
      restores float32 on the GPU; `target_valid` stays bool), ~3.5× less
      worker→trainer traffic;
    - the normalizer's HDF5 handles are closed before workers fork.
  - **Check vs the previous item on 450 real tiles** (Germany, Uruguay,
    Brazil; training, evaluation and cutoff modes):
    - every mask, slot choice, date, static layer and target is identical;
    - spectral values differ by ≤ 1 float16 step (2e-3) and weather by ≤ 6e-5
      (float32 vs float64 arithmetic).
  - **Throughput** on this host: 45 → 59 tiles/s. Cluster rates will be
    measured in the Nautilus smoke run (YI-08).
- 2026-10-01 — **YI-05 done.** `yieldsat_image_compare.py`:
  - **Matching:** pairs each image fold with the point fold at the same
    relative results path (`paper/<group>/<inputs>/<tag>_seed<s>/<pair>/foldNN`,
    tags `image` vs `ours`). Cells are joined on (field season name, grid row,
    grid col), independent of season index order. The tool refuses
    duplicates, and refuses matched cells whose targets disagree.
  - **Scoring:** both models on exactly those cells (pixel R²/RMSE;
    field-level R² only for folds with ≥ 3 seasons). Fold means per (protocol
    group, inputs, pair) go to `summary.csv`; per-fold rows to `folds.json`;
    point folds still missing to `missing_point_folds.json`.
  - **Test:** 15 pass.
  - **Real check** (GER-R CV10 fold 0, S2+ADM): image smoke run vs the
    earlier local point run.
    - 10,420 cells matched; 100% of image test cells, all targets equal.
    - Pixel RMSE: image 1.54, point 1.58 (one fold, early point run; not a
      result).
  - **Open point — coverage.** The image dataset keeps only grid-aligned,
    non-overlapping 64×64 windows with ≥ 2,048 valid cells. It holds 63%
    (Argentina), 60% (Brazil), 29% (Germany) and 26% (Uruguay) of the point
    cells; GER-R fold 0 test: 33%. Image-vs-point tables are therefore on
    the tiled subset, with the point model's whole-field scores reported
    separately. Raising coverage (lower threshold, overlapping or edge tiles
    for inference) would mean changing the dataset builder; the user decides.
- 2026-10-01 — **YI-06 done; YI-07 driver and suites done.**
  - **`yieldsat_cluster.py`:**
    - Model preset `image` launches `main_yieldsat_image.py` with
      `--image_root`. Point runs are unchanged.
    - Image jobs stage only the country's tiles, its DINO cache file and
      manifest, `index/<country>/fields.json` and the runs' splits. Copies are
      atomic (`.partial` → rename), skipped when already present.
    - `donors` experiments create one run per donor × inputs × seed under
      `donors/<name>/<inputs>/seed<s>`, keeping `checkpoint_best.pth` in the
      results root.
    - `warm_start: {pair: donor}` adds `--init_ckpt
      <donor_root>/donors/<donor>/<inputs>/seed<s>/checkpoint_best.pth` and
      `--lr` × 0.3 (same seed and input set as the target run).
    - Tile-based time estimates (59 tiles/s per run, dev host).
    - Runs whose fold has no train or no test tiles are left out at plan
      time and listed in `plan.json` (`suite.unrunnable`).
    - `YIELDSAT_IMAGE_ROOT` overrides the planned image root.
  - **Suites:**
    - `cluster/suites/image_donors.yaml`: `bra-w` (BRA-W tiles) and `pooled`
      (Argentina + Brazil + Uruguay, all crops) × S2, S2+ADM × seeds 0–2 →
      12 runs.
    - `cluster/suites/image_full.yaml`: CV10 + LOYO for all pairs, farm LORO
      outside Argentina, province LORO for ARG-C/S/W, paper and strict
      policies, S2 and S2+ADM, seeds 0–2. GER-R ← pooled and GER-W ← bra-w.
      W&B project `yieldsat-cvpr27-image`.
  - **Local dry plan against the real artifacts:**
    - The (split, inputs, seed) set equals the point suite's `ours` runs
      exactly (2,346 outside ARG LORO), plus 240 province-LORO runs.
    - No fold manifest was created; all 667 are reused.
    - 126 runs left out for lack of train/test tiles (e.g. BRA/URG farm-LORO
      folds and GER-W LOYO folds whose held-out fields have no qualifying
      window). Their comparison falls back to the point model only.
    - 2,460 runs in 64 jobs, ~176 GPU-hours at the dev-host loader speed
      (~5.5 h on 32 GPUs). To be recalibrated in YI-08.
  - **Local end-to-end through `yieldsat_cluster.py run`:**
    - pooled donor (S2, 1 epoch): staged Argentina+Brazil+Uruguay in 14 s,
      ran in 50 s, checkpoint kept;
    - warm-started GER-R CV fold 0: loaded 175 donor tensors at lr 1.5e-4,
      results copied.
    - GPU utilization was 10–14%: image runs are loader-bound, so pods need
      CPU rather than more runs per GPU.
  - Tests: image 17, point 35, all pass.
- 2026-10-01 — **YI-07 done: image, manifests and operating order.**
  - **Loader:** image jobs stage tiles uncompressed (tile by tile from the
    PVC, `stage_uncompressed: true`). Training on the dev host goes from
    45 to 118 tiles/s per run; LZF decompression dominated. Argentina needs
    ~10 GB of pod scratch. `REF_THROUGHPUT['image'] = 118`.
  - **Donor ordering:** pools skip warm-start jobs until their donor
    checkpoints exist and wait (5-minute polls, ≤ 24 h) when only those
    remain, so German runs never start early or trip the circuit breaker.
  - **Pod environment:** pods get `YIELDSAT_IMAGE_ROOT=/data/YieldSAT/YieldSAT-Image`
    and `YIELDSAT_DONOR_ROOT=/data/YieldSAT/yieldsat_results/image_donors`.
  - **Container image** `gitlab-registry.nrp-nautilus.io/smatous/yieldsat:image-v1`
    (pushed; digest `sha256:97b7036c…`):
    - adds `transformers` and the DINOv3 class; the build runs the point and
      image tests;
    - a separate tag, so the running point pools (`:latest`) are unaffected;
    - `.dockerignore` now excludes the gitignored credential YAML (it was
      missing there). The pushed `:latest` was checked and does not contain
      it (built before the file existed).
  - **Manifests** (`cluster/nautilus/`, no credentials inside):
    1. `image_build_job.yaml`: CPU Job (4 CPU, 16 GiB). Builds the tiles on
       the PVC with `yieldsat_build_images.py` from the PVC's index, cache and
       geometry; skips if the manifest exists; prints per-country counts to
       compare with the development copy (1,220 / 920 / 68 / 229 tiles).
    2. `image_dino_cache_job.yaml`: one A10/3090. `HF_TOKEN` from the
       `smatousi-hf` secret; the checkpoint is downloaded once to
       `/data/YieldSAT/hf_cache` and the cache written to
       `YieldSAT-Image/dino_cache/f692fa42_b764aff546ec/`.
    3. `image_smoke_pool.yaml` + `cluster/suites/image_smoke.yaml`: 1 A10,
       9 runs. Pooled donor (3 epochs) → GER-R folds 0–1 warm-started, plus
       ARG-S folds 0–1 for throughput; W&B `yieldsat-cvpr27-image-smoke`.
    4. `image_donors_pool.yaml`: 2 A10 pods (16 CPU, 32 GiB, 3→4 runs);
       12 donor runs.
    5. `image_full_pools.yaml`: provisional, revised after the smoke run.
       8 A10 (16 CPU, 32 GiB, 3→5 runs) + 8 RTX 3090 (12 CPU, 24 GiB,
       2→4 runs). Image runs use ~1 GB GPU each and are loader-bound, so
       utilization comes from concurrent runs with CPU-rich pods.
  - **Order:** 1 → 2 → 3 (check results, calibrate) → 4 → 5. Steps 4 and 5
    may overlap: German jobs wait for the donors. Plans are made by the
    first pod (`run_in_pod.sh`).
- 2026-10-01 — **YI-08 step 1: image dataset built on the PVC.**
  `image_build_job` (image-v1) wrote `/data/YieldSAT/YieldSAT-Image` in ~14 min.
  It is identical in counts to the development copy: 2,437 tiles, 6,666,347
  valid cells; Argentina 1,220 / 3,374,971, Brazil 920 / 2,550,502, Germany
  68 / 174,698, Uruguay 229 / 566,176. The DINO cache job was then submitted.
- 2026-10-01 — **YI-08 step 2: DINO cache on the PVC.**
  `/data/YieldSAT/YieldSAT-Image/dino_cache/f692fa42_b764aff546ec` matches
  the development cache exactly: entries 8,628 / 6,275 / 625 / 1,411, and
  the same clip fractions. It was built on an RTX 3090 in ~30 min (CephFS
  reads dominate).
  - The first submission waited 2 h for an A10/3090, all taken by the point
    suite. The one-off job now accepts any common GPU type.
  - Smoke pool re-rendered for one A10 or 3090 (12 CPU, 32 GiB) and
    submitted.
