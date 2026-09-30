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
