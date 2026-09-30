# YieldSAT 64×64 Image Dataset

**Status:** Dataset built and verified, 2026-09-30. The image dataset is
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

## Action plan and acceptance

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
5. **Train later:** add an explicit image loader and image/spatial model path with
   grouped splits, pre-harvest cutoff, train-only stats and no target leakage.
   Compare point and image modes on the same held-out physical fields. This
   construction task does not claim image-model training is already implemented.

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
