# Data Contract

This is the on-disk layout the field-level loaders already expect. It is written
down here because **nothing else in the repository states it**, and it is the single
piece of missing information standing between the current prototype and a real
training run.

Everything below was read out of `dataset/field_yield_dataset.py`,
`dataset/heterogeneous_modality_loader.py`, and `dataset/field_roi.py`. It describes
what the code accepts today, not an aspiration.

---

## 1. Directory layout

One flat directory per dataset. One sample = one **field-year**, identified by a name.

```
<root_dir>/
  train.txt                      # one field-year name per line
  test.txt
  <field>_boundary.tif           # REQUIRED — defines the template grid
  <field>_yield.tif              # REQUIRED for supervised training — dense target
  <field>_DEM.tif                # optional modality layer
  <field>_SAR.tif                # optional modality layer
  <field>_crop.tif               # optional modality layer
  <field>_SOIL.npy               # optional tabular layer
  ...
```

Field-year names are opaque strings. The convention used in the module demos is
`<field>_<year>`, e.g. `field_2021`. Whatever you choose, it must not contain the
layer separator `_` in a position that makes `<field>_<layer>.tif` ambiguous — prefer
a name with no trailing numeric segment that collides with a layer name.

A layer file that is **absent is not an error**. The loader flags it unavailable and
the model substitutes a learned missing-modality token. This is the mechanism by
which the whole missing-modality story is exercised on real data.

---

## 2. The boundary raster defines everything

`<field>_boundary.tif` is the anchor. `field_roi.load_field_roi` reads it and builds
the canonical **template grid** that every other layer is registered into:

| property | source |
|---|---|
| CRS | the boundary raster's CRS — **must be set, a `None` CRS raises** (`field_roi.py:42`) |
| extent | the boundary raster's bounds |
| resolution | `out_resolution_m`, a constructor argument (default **10.0 m**) — *not* the boundary's own resolution |
| width / height | derived from extent ÷ resolution |

The mask itself is `src.read(1) > 0` — any positive value is inside the field.

Use a **projected** CRS in metres (e.g. UTM, `EPSG:326xx` / `EPSG:327xx`). The
`out_resolution_m` parameter is interpreted directly in CRS units, so a geographic
CRS in degrees will produce a grid roughly 10⁵× too coarse with no error raised.

---

## 3. Layer specifications

### `FieldYieldDataset` — all layers as single-band rasters

Constructor: `layers=("DEM", "S2", "SOIL", "crop")`, each read from
`<field>_<layer>.tif`, registered into the template grid, emitted as `(1, H, W)`
float32.

Resampling is chosen by name (`field_yield_dataset.py:93`):

| layer name | resampling | rationale |
|---|---|---|
| `crop`, `landcover`, `crop_history` | **nearest** | class indices must not be interpolated |
| anything else | **bilinear** | continuous surfaces |

⚠ This list is a hardcoded tuple, `CATEGORICAL_LAYERS`. A categorical layer under any
other name will be **bilinearly interpolated into meaningless fractional class
indices, silently**. Add your name to that tuple or rename your file.

### `HeterogeneousModalityLoader` — per-kind tensor shapes

Constructor takes `(name, kind)` pairs and shapes each layer for its encoder:

| kind | file | shape emitted | encoder |
|---|---|---|---|
| `raster` / `dem` / `terrain` / `soil_raster` | `<field>_<name>.tif` | `(1, H, W)` | `RasterCNNEncoder` |
| `sar` | `<field>_<name>.tif`, `T×C` bands **frame-major** | `(T, C, H, W)` | `MultitemporalSAREncoder` |
| `categorical` | `<field>_<name>.tif` | `(H, W)` int | `CategoricalEncoder` |
| `tabular` | `<field>_<name>.npy` | `(D,)` float32 | `TabularEncoder` |

**SAR band order is frame-major**: for T=6 frames of VV/VH, the 12 bands are
`[t0_VV, t0_VH, t1_VV, t1_VH, ...]`. Getting this wrong transposes time and
polarisation with no error — a silent correctness failure. Verify band order against
your writer before training.

### Not yet supported by any loader

`TimeSeriesEncoder` expects `(B, T, D)` weather/precipitation/soil-moisture series,
but **neither loader emits a `timeseries` kind**. Weather is the one modality in the
model config with no ingest path. Adding it is Phase 2 work: either a
`<field>_<name>.npy` of shape `(T, D)`, or a per-field CSV joined by field-year.

---

## 4. The yield target

`<field>_yield.tif`, single band, registered bilinearly into the template grid,
emitted as a dense `(H, W)` float32 target.

Requirements and cautions:

- **Units must be consistent across all fields.** The model regresses raw values;
  nothing normalises the target. Mixing bu/ac with t/ha across fields will not be
  caught.
- **Nodata is filled with `-9999.0`** (`DEFAULT_NODATA`, `field_yield_dataset.py:39`).
  The dense yield heads now exclude this value and nonfinite targets from L1/MSE
  loss, and accept an optional `valid_mask` for field-boundary pixels. An all-invalid
  target raises a clear error. Apply the field-boundary mask when wiring real data
  in Phase 3.
- Yield-monitor data needs the usual agronomic cleaning (flow-delay correction, start/
  stop pass artefacts, speed outliers) *before* it reaches this directory. The loader
  does no cleaning.
- If you use `DenseYieldFCNHead(log_target=True)`, invalid cells are removed before
  the valid targets are log-transformed.

---

## 5. Split files

`train.txt` / `test.txt`, one field-year name per line, blanks ignored.

The mission requires **spatial** splits. `util/spatial_split.py:11`
`cross_location_split` implements disjoint state-group partitioning, but it operates
on the county-level System A schema and is not wired to these files. Until that is
connected, whoever writes `train.txt` is responsible for the split's geographic
honesty.

Two failure modes to avoid explicitly:
- **Same field, different years, split across train and test.** Field identity leaks
  soil, topography, and management — the dominant yield signal. Split by *field*,
  then assign all its years to one side.
- **Adjacent fields split across train and test.** Spatial autocorrelation at field
  scale is strong. Split by farm, or by a spatial block considerably larger than a
  field.

---

## 6. Known loader limitations

Recorded here so they are not rediscovered at 3 a.m.

| # | Issue | Location | Impact |
|---|---|---|---|
| L1 | **Eager full-dataset registration in `__init__`.** Every field is reprojected and held in memory at construction. | `field_yield_dataset.py:159`, `heterogeneous_modality_loader.py:142` | O(dataset) memory and startup time; all work happens in the parent process, so `num_workers` buys nothing. Fine for the 2-field demos, fatal at 10k fields. Must become lazy + cached before any real run. |
| L2 | `HeterogeneousModalityLoader` **does not subclass `torch.utils.data.Dataset`** (`:79`). | `heterogeneous_modality_loader.py:79` | Cannot be passed to a `DataLoader` as-is despite having `__len__`/`__getitem__`. `FieldYieldDataset` does subclass correctly. |
| L3 | The per-sample `available` array is positional over `self.layers`, while `collate_field_samples` derives availability from key presence and ignores it. | `field_yield_dataset.py:211`, `collate.py:44` | Harmless today (the two agree), but two sources of truth for the same fact. |
| L4 | Layer files are resolved by string interpolation with no manifest. | both loaders | A typo in a layer name is indistinguishable from a legitimately missing modality — it silently trains with that source absent. A manifest or a strict mode would catch it. |
| L5 | No band-count validation on SAR. | `heterogeneous_modality_loader.py:_load_raster` | A `T×C` reshape mismatch surfaces as a confusing shape error or, worse, a silent mis-grouping. |

---

## 7. Minimal worked example

`dataset/field_yield_dataset.py:219` contains a runnable end-to-end example that
writes synthetic GeoTIFFs to a temp directory (EPSG:32615, 10 m, 20×20), builds a
two-field dataset where `field_2022` deliberately lacks the `SOIL` layer, and prints
the resulting shapes and availability vector.

It is the fastest way to confirm your own data is shaped correctly: point the same
constructor at your directory and compare the printed shapes. It needs `rasterio`,
`numpy`, and `torch` installed — see [tech_stack.md](./tech_stack.md).
