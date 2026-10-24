# Training on our own dataset: does PlanetScope's resolution raise within-field R²?

**Status (2026-10-24): dataset built and checked; units written; cluster run starting.**

Related:
- [own-dataset-ground-truth.md](./own-dataset-ground-truth.md): the yield maps (65 clean field-seasons, 10 m target).
- [yieldsat-field-context.md](./yieldsat-field-context.md), [yieldsat-relational-loss.md](./yieldsat-relational-loss.md):
  the best model (dense p3-nbr + field context + relation loss, `ours-p3nbr-dense-fc-rel10`).
- geo-data-downloader `specs/phase-3b-planetscope`: the PlanetScope download (75 field-seasons, 3,645 scenes).

## 1. Question

The paper's contribution 5 (high-resolution dataset). The input-information study showed within-field R² falls
when S2 is coarsened (20/30/60 m: −0.012/−0.024/−0.055) and that half the within-field variance sits at the
20–50 m scale. **Does finer imagery (PlanetScope, 3 m) raise within-field R² on our data?**

## 2. Decisions (project lead, 2026-10-24)

| ID | Decision |
|---|---|
| D1 | **Same model and protocol as YieldSAT.** The best model `ours-p3nbr-dense-fc-rel10`, unchanged; the paper's protocol (no validation set, selection on the test fold), per pair |
| D2 | **Pairs and protocols: per crop, CV10 + LOYO.** MO-S (soybean, 49 field-seasons, 2018–2024) and MO-C (corn, 15, 2021–2025). CV10 grouped by field season, stratified by county; LOYO one fold per harvest year. No LORO (one state). The single "other" crop field is dropped |
| D3 | **Three arms** (separates "more data" from "finer resolution"): S2 only; S2 + PlanetScope averaged to the 10 m cell (`-ps10`); S2 + PlanetScope on a 3×3 sub-grid of the cell (`-ps3x3`) |
| D4 | **Target and cell grid:** the 10 m yield raster on each field's S2 grid (own-dataset-ground-truth.md). DEM and terrain on the same 10 m grid (project lead): the 1 m DEM averaged per cell, terrain derived from that 10 m DEM |
| D5 | **QC-clean field-seasons only** (65; 64 after D2) |

## 3. Dataset (`yieldsat_build_owndata.py`, country `Missouri`)

Built in the YieldSAT artifact contract, so the dense-series code reads it unchanged
(`YIELDSAT_NUM_SLOTS=72 YIELDSAT_CACHE_NAME=cache_dense YIELDSAT_NBR_NAME=neighbourhood_dense`, `--skip_source_check`).

| Input | Source | Processing |
|---|---|---|
| Slots | — | 72 ten-day bins from January of harvest year − 1, masked outside [planting, harvest] (dense rule D1) |
| Season dates | catalog `PlantingDate`, ground-truth harvest start | 56 reported; 6 imputed from the crop × year median, 2 from the crop's median day of year. Free-text dates parsed to the first day ('4/6,7,9/2023', 'Early May' → the 5th) |
| S2 | Planetary Computer L2A (downloader) | Per cell and bin the clear acquisition (SCL 4/5/6) nearest the bin centre; 20 m bands nearest-upsampled; **BOA offset removed** (−1000 DN for products processed from 2022-01-25, i.e. baseline ≥ 04.00, reprocessed archive included: Planetary Computer serves them unharmonized). B01, B09 not downloaded (NaN) |
| Weather | ERA5-Land daily (Open-Meteo) | Sum over the bin's in-season days; °C → K, mm → m (YieldSAT units) |
| DEM, terrain | USGS 3DEP 1 m; 5 m TWI | DEM averaged to 10 m; Horn slope (°), downslope aspect (°), Zevenbergen–Thorne profile curvature (×100) from the 10 m DEM; TWI = mean of the 5 m TWI |
| Soil | POLARIS 30 m (clay, sand, silt, pH, OM → `soc` slot) per cell; SoilGrids centroid for the rest | The SoilGrids centroid file has no CEC, coarse fragments or nitrogen: those 18 channels are empty (masked) |
| PlanetScope | 8-band SR (4-band 2018–2020), UDM2 | Per scene, area-weighted onto the 10 m grid and onto the 3×3 sub-grid (3.33 m); per cell and bin the scene nearest the bin centre whose cell is ≥ 90% clear (UDM2); sub-cells ≥ 50% clear. 4-band scenes fill blue/green/red/NIR of the 8-band order |
| Neighbourhood, field context | existing builders | 5×5 S2 neighbourhood mean; field S2 mean/std, within-field z-scores, within-field terrain z-scores, edge distance |

**Contents:** 64 field-seasons, **190,124 cells**, 55 physical fields (shared ground across years: 8 fields with
2–3 seasons). Clear slots per cell: S2 10–15, PlanetScope 13–17. Build 18 s on 12 workers.

**Checks (2026-10-24):**
- Same-day S2 vs PlanetScope over the cells, per year: median NDVI correlation 0.87–0.99; red and NIR levels
  agree (e.g. 2023: red 677 vs 614, NIR 3,310 vs 3,735). Before the BOA fix the 2022+ S2 bands were ~1,000 DN
  too high.
- **Registration against NAIP (0.6 m), 7 fields with same-season NAIP:** PlanetScope best aligned at zero shift
  on 6 (one column on 1); S2 is one 10 m pixel off on 3 of 7. S2's own geolocation error (≤ 1 pixel) is a source
  of target–input mismatch that PlanetScope reduces; not corrected (YieldSAT has the same).

## 4. Code

- `dataset/yieldsat_schema.py`: `OWN_COUNTRIES = ('Missouri',)`; trainer and builders accept it.
- `dataset/yieldsat_dataset.py`: PlanetScope streams `planetscope_10m` (8 channels) and `planetscope_3x3`
  (72 = 9 sub-cells × 8 bands), per-band train-only normalization (`ps_normalization`, from
  `ps_field_sums.npz`); masked where not clear or outside the slot window. They share the S2 slot times.
- `main_yieldsat_finetune.py --planetscope 10m|3x3`.
- `yieldsat_protocol_runs.py units --batch own`: 96 units (3 arms × 32 folds), `cluster/tabm/protocol_own_units.json`.

## 5. Evaluation

Within-field R² (the main objective) and pooled pixel / field R², pooled out of fold per pair and protocol;
paired fold bootstrap of each PlanetScope arm against S2 only. Also the noise ceiling of our maps (semivariogram
nugget, as yieldsat-field-context.md §3).

## 6. Log

- 2026-10-24: dataset built and checked (§3); CPU smoke of the S2 and 3×3 arms (2 × 30 steps) runs end to end.
