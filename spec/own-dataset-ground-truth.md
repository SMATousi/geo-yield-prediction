# Our yield dataset: ground-truth generation

**Status (2026-10-15):** ground truth regenerated for every catalog field-season with one pipeline. 65 field-seasons
pass QC: 1,870 ha harvested, 2,002 ha of unique ground, Missouri, 2018–2025. This document explains the pipeline and
the dataset for the paper's dataset section.

Related:
- `geo-data-downloader/specs/phase-5b-ground-truth/requirements.md`: the batch run, its inputs, every deviation and
  every failure in detail (operational record).
- `vigir_remote_sensing/yield_project/spec/`: the pipeline's own specs (Phase 1.5 preprocessing, Phase 2 RITAS,
  Phase 5 web portal).
- `geo-data-downloader/specs/phase-3b-planetscope/requirements.md`: imagery for these fields.

## 1. Source data

- **Trials:** on-farm trials in Missouri, run by the TG group with cooperating farmers.
  - 81 strip trials (`StripTrial`, mostly fungicide and planting comparisons);
  - 15 on-farm nitrogen trials (`OnFarmN`, `OnFarmBioN`).
- **Catalog:** every trial-year is one row of the group's yield-map catalog, identified by `VGIR_ID` (e.g.
  `SoyBean2018_89`). The catalog records crop, year, county, the raw yield file, a trial-layout shapefile and
  agronomic context (variety, planting date, row spacing, tillage, irrigation).
- **Raw data:** the combine's yield-monitor log, one point per measurement interval (usually 1 s), with GPS position,
  grain mass flow, interval duration, swath width, travel distance and grain moisture.
  - Formats differ by monitor brand and export path (Ag Leader / SMS, John Deere Operations Center, CNH).
  - Column names differ accordingly, and some exports are curated merges of several harvest files.

| | count |
|---|---|
| catalog field-seasons | 96 (2018–2025) |
| with a raw yield file | 90 |
| processed by the pipeline | 75 |
| **pass QC (the dataset)** | **65**: 49 soybean, 15 corn, 1 other |

## 2. Pipeline

All field-seasons go through the same pipeline version: the TG web portal's (`yield_project` commit `0bd49ee`),
driven through the portal's own backend functions. A re-run of one field reproduces the portal's earlier output of
it bit for bit.

### 2.1 Preprocessing (point level)

1. **Header unification.**
   - Raw columns are mapped to a canonical schema (latitude, longitude, timestamp, mass flow [lb/s], duration [s],
     swath width [in], travel distance [in], moisture [%], elevation).
   - Mapping uses ordered regular-expression rules covering common monitor formats. A small language model
     (`phi3:mini` via Ollama) answers a one-word multiple-choice question for any required column the rules leave
     unmatched.
   - Unit heuristics: a median swath below 20 is taken as metres; a distance label with "ft" is converted from feet;
     a flow label with "kg" is converted from kilograms.
2. **Point order.** Time order and file order are compared by their spatial jump rate (the share of consecutive
   points more than N × the median step apart); the order with fewer jumps is used.
3. **Pass detection.** A new pass starts at:
   - a spatial jump (more than 20 × the median step);
   - a time gap (more than 30 s);
   - a heading change (more than 45°).

   Passes with fewer than 20 points (turns, noise) are dropped.
4. **Flow-delay correction.** Grain reaches the mass-flow sensor several seconds after it is cut, so the logged yield
   is displaced along the pass.
   - Adjacent passes driven in opposite directions are paired: bounding boxes within one header width, separation
     within ±15% of the header width.
   - For each pair, the delay maximising the normalised cross-correlation of the two flow profiles is found, within
     −4 to +17 intervals (the Yield Editor 2 defaults).
   - The modal delay over all pairs (up to 5,000) is applied: each point takes the flow measured D intervals later in
     its pass.
   - Points shifted past a pass end are flagged and removed.
   - With fewer than 10 valid pairs, no correction is applied.

### 2.2 RITAS (polygon level)

The point log is turned into a polygon yield map in the manner of the RITAS method (rectangles, intersection,
tessellation and aggregation, smoothing). TODO: verify the citation, probably the `ritas` R package by Damiano and
Niemi.

1. **Vehicle polygons.** Each point becomes a rectangle of the swath width by the travel distance, oriented along the
   travel direction: the ground the combine covered during that interval.
2. **Overlap resolution.** Overlapping rectangles are resolved geometrically. Earlier rectangles take precedence, and
   later ones keep only their non-overlapping part (the effective area).
3. **Yield.**
   - Dry-matter mass = wet mass × (100 − moisture) / 100.
   - Yield = dry-matter mass / effective area, in **Mg/ha of dry matter**, not at standard moisture.
4. **Cleaning.** Slivers with an effective area under 1% of the median polygon area are dropped. Polygons whose
   yield lies outside [Q1 − 3·IQR, Q3 + 3·IQR] of the field are removed (median 3.2% of polygons).
5. **Outputs.**
   - the resolved polygons: the ground-truth geometry, one polygon per measurement with its own yield and
     footprint;
   - a field raster from aggregating the polygons onto a 10,000-cell grid (about 5 m cells);
   - a Gaussian-smoothed version (σ = 1 cell).

### 2.3 Our additions

- **One raw file per field-season:** the catalog's file. Three rows whose catalog folder is empty use the same
  trial's original export from the archive.
- **Crop fallback:** the catalog crop where the pipeline cannot detect the crop (62 of 75 runs); otherwise those corn
  fields would be processed with soybean defaults.
- **Format normalisation of 11 inputs:** reprojection of 10 curated exports from UTM to geographic coordinates, and
  renaming of one time column. No values are changed.
- **Quality control** (not part of the pipeline). A field-season is excluded if either holds:
  - its area-weighted mean yield is outside 4–18 Mg/ha (corn) or 1.5–7.5 Mg/ha (soybean);
  - its resolved polygons cover under 40% of the field outline.

  This excludes 10 of 75 (§4).
- **Target raster (2026-10-15, `geo-data-downloader rasterize-yield`).** The pipeline's 10,000-cell grid has
  field-specific, non-square cells (5.00–5.21 m), so it is not used.
  - The target is rasterised from the resolved polygons onto each field's Sentinel-2 10 m grid, the same CRS,
    transform and shape as its S2 bands, as in YieldSAT.
  - A cell's yield is the mean polygon yield weighted by exact polygon-cell intersection area. Cells with less than
    50% polygon coverage are masked; the coverage band is kept.
  - Checks: the intersected area equals the polygon area, and the raster mean equals the polygon mean, in every
    clean field.
  - The 65 clean field-seasons give **192,325 target pixels** (median 2,545 per field, 141–9,216).
  - Files: `layers/<id>/yield/<id>_yield_s2_10m.tif`. Details are in geo-data-downloader
    `specs/phase-5b-ground-truth/requirements.md` §5.

## 3. The dataset (65 field-seasons after QC)

| | |
|---|---|
| field-seasons | 65 (49 soybean, 15 corn, 1 other) |
| harvest years | 2018 (7), 2019 (11), 2020 (6), 2021 (7), 2022 (5), 2023 (13), 2024 (11), 2025 (5) |
| harvested area | 1,870 ha (median field 24.7 ha, 1.4–88.3 ha) |
| unique ground | 2,002 ha (some fields recur in several years) |
| raw yield-monitor points | 1.33 M → 1.27 M resolved yield polygons |
| field-mean yield (dry matter) | corn 6.9–15.4 Mg/ha (median 9.7); soybean 2.1–5.0 Mg/ha (median 3.6) |
| harvest | median 1 day per field (range up to several weeks) |
| trial layouts | merged as separate polygons for 73 of the 75 processed |

## 4. Known limitations (for the paper's limitations paragraph)

- **Flow delay is rarely estimated with confidence:** "reasonable" in 8 runs, "questionable" in 51, not estimated in
  16 (0 applied). 12 estimates sit at a bound of the search range. Residual along-pass displacement of up to a few
  metres is therefore likely in many fields.
- **Not processed:**
  - 12 field-seasons exported from John Deere Operations Center. They carry the monitor's area-normalised wet yield
    (lb/ac) rather than a mass-flow log and no interval duration, so the pipeline cannot rebuild them.
  - 2 field-seasons that hit a polygon-writer bug.
  - 1 with no yield data.
- **Excluded by QC:** 10 field-seasons with unit errors in the export (implausible yields, or swath polygons covering
  1–7% of the field). The earlier portal outputs of the same fields have the same defects, so these are input
  problems.

## 5. Draft for the paper (dataset section)

> **Yield maps.** We use yield-monitor data from 65 on-farm trial field-seasons in Missouri (49 soybean, 15 corn, one
> other crop; 2018–2025; 1,870 ha harvested). All raw logs are processed with one pipeline. Columns from different
> monitor brands are mapped to a common schema, points are ordered and split into passes, and the grain-flow delay is
> estimated by cross-correlating adjacent opposite-direction passes and corrected. Following RITAS [cite], every
> measurement is then turned into a rectangle of the swath width by the travel distance; overlaps between rectangles
> are resolved geometrically, and yield (Mg/ha of dry matter) is the moisture-corrected mass divided by the
> rectangle's effective area. Polygons outside three interquartile ranges of their field are removed, as are
> field-seasons with implausible yields or coverage. The resolved polygons, about 1.3 M in total, are rasterised onto
> each field's Sentinel-2 10 m grid by exact area weighting; cells less than half covered are masked, leaving
> 192,325 target pixels.

## 6. Open items

1. The 12 John Deere exports: rebuild per-record mass from wet yield × swath × distance, or use the monitor yield.
2. The writer bug (2 field-seasons): a one-line fix in the pipeline, which needs the project lead's go-ahead.
3. The 10 QC-flagged field-seasons: exclude them (current choice), or fix the units. For 4, the original exports
   are available.
4. Target raster: done on the S2 10 m grid. Open: whether 0.5 is the right minimum coverage, and a 3 m version if
   PlanetScope is bought.
5. Verify the RITAS citation.
