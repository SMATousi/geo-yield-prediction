# Own dataset: the field-seasons that did not make it into the 65

Status: REPORT (2026-10-28). No data changed. Companion to `spec/own-dataset-ground-truth.md`; the per-run numbers
are in `gdd-workspace/ground_truth/_summary.csv` and each run's `job.log` (on pupil,
`/home1/pupil/SMATousi/gdd-workspace`).

## 1. Overview

| stage | count | lost here |
|---|---|---|
| catalog field-seasons (`VGIR_ID`) | 96 | |
| with a raw yield file | 90 | 6 (§2) |
| processed by the pipeline | 75 | 15 (§3) |
| pass QC (the dataset) | 65 | 10 (§4) |

So 31 catalog rows are missing from the dataset. They fall into seven causes:

| cause | n | fixable? |
|---|---|---|
| no raw file in the catalog folder | 6 | only if the files exist elsewhere |
| John Deere Operations Center export (area-normalised yield, no mass flow) | 12 | yes, needs a reconstruction rule |
| pipeline writer bug (degenerate polygon) | 2 | yes, one-line fix in the pipeline |
| boundary polygon only, no yield points | 1 | no |
| rows sorted by northing (curated GIS merge), so pass detection drops 95-98% of points | 5 | yes for 4, by restoring the log order |
| distance in inches converted as if feet (×12) | 4 | yes, by fixing the unit heuristic |
| harvest log covers only part of the field outline | 1 | maybe, by lowering the coverage limit |

The earlier note in `own-dataset-ground-truth.md` §4 put all 10 QC failures down to "unit errors in the export" and
called them input problems. That is only partly right: 4 of the 10 are a pipeline unit heuristic (§4.2), and the
same heuristic also affects 9 of the 65 clean field-seasons (§5).

## 2. No raw file (6)

| VGIR_ID | catalog `YldMapUpload` |
|---|---|
| Corn2025_10 | Y, but the folder is empty |
| SoyBean2024_24, SoyBean2023_39, SoyBean2021_59, SoyBean2019_82, SoyBean2018_93 | N (never uploaded) |

Source: `ground_truth/_jobs.json`, key `missing`.

## 3. Not processed by the pipeline (15)

### 3.1 John Deere Operations Center exports (12)

Corn2023_17, SoyBean2018_89, SoyBean2018_95, SoyBean2020_70, SoyBean2021_52, SoyBean2021_54, SoyBean2022_44,
SoyBean2022_46, SoyBean2023_38, SoyBean2023_40, SoyBean2024_21, SoyBean2024_26.

- These are exports from John Deere Operations Center (file names `<farm>_<field>_Harvest_<date>_00.shp`), not raw monitor logs.
- `WetMass` is the monitor's wet yield per area (lb/ac), not the mass of each record, and there is no `Duration`.
  The pipeline needs mass flow and duration, so it cannot build the RITAS polygons.
- Checked against the monitor's own yield: for SoyBean2024_21, `VRYIELDVOL` 23.45 bu/ac gives 1,362 lb/ac against
  `WetMass` 1,409; for Corn2023_17, 9,166 against 9,264. `DISTANCE` is ft per 1-s record.
- **Possible fix:** rebuild each record's wet mass as `WetMass` × swath × distance (1-s duration) and run RITAS as
  usual, or use the monitor's yield points directly. For most of these trials there is no other raw file.

### 3.2 Pipeline writer bug (2)

SoyBean2021_51, Corn2019_85.

- During overlap resolution, a resolved polygon degenerates to a POINT, and the polygon-shapefile writer fails.
- **Fix:** drop non-polygonal geometries in `process_yield.write_polygon_shapefile` (one line). This changes the
  pipeline repository, so it needs the project lead's go-ahead.

### 3.3 No yield data (1)

SoyBean2024_23: the file in the drive (`2024willottsoybeanfungicide_poly.shp`) is a single field boundary polygon,
with no yield points.

## 4. Processed but excluded by QC (10)

QC rule: area-weighted field-mean yield outside 4-18 t/ha (corn) or 1.5-7.5 t/ha (soybean), or resolved polygons
covering under 40% of the field outline.

| VGIR_ID | raw file | raw pts | pts kept | coverage | mean yield (t/ha) | cause |
|---|---|---|---|---|---|---|
| SoyBean2020_64 | FT2002 | 20,444 | 391 | 1.9% | 3.02 | sorted rows (§4.1) |
| SoyBean2020_66 | FT2004 | 27,288 | 1,203 | 7.1% | 1.46 | sorted rows (§4.1) |
| SoyBean2022_42 | combinedyield2022 | 6,216 | 169 | 7.4% | 4.99 | sorted rows (§4.1) |
| SoyBean2022_45 | HR_Yield_2022 | 26,127 | 481 | 1.4% | 5.63 | sorted rows (§4.1) |
| SoyBean2022_50 | HR_Yield2022_teasley | 31,330 | 775 | 6.8% | 1.76 | sorted rows (§4.1) and ×12 distance (§4.2) |
| SoyBean2020_68 | a2020_FT2006_NR_middle39 (set 2) | 8,762 | 8,601 | 87% | 90.99 | ×12 distance (§4.2) |
| SoyBean2021_58 | 2021buschfungicideyield | 54,850 | 51,161 | 49% | 50.42 | ×12 distance (§4.2) |
| SoyBean2024_4 | teasleybeanfungicide2024 | 18,412 | 18,362 | 63% | 57.69 | ×12 distance (§4.2) |
| SoyBean2023_33 | 2023willottbeanfungicideyield | 10,463 | 10,092 | 38% | 1.08 | ×12 distance (§4.2) |
| SoyBean2018_92 | FT1807_SMSmin | 1,828 | 1,801 | 37.7% | 2.58 | partial log (§4.3) |

### 4.1 Rows sorted by northing (5)

- These five are curated GIS products, not monitor logs: they carry ArcGIS spatial-join columns (`FID_SMSmin`,
  `Join_Count`, `TARGET_FID`), and several merge more than one harvest dataset (`Dataset` has 2-4 values).
- Their rows are sorted by northing: 99.8-100% of consecutive rows have non-decreasing y. Consecutive rows are 83-166 m
  apart (median), against 1-2 m in a monitor log.
- The `Date` column holds only the day, so every timestamp is identical and the pipeline falls back to file order.
  Pass detection then sees a heading change or a jump at almost every point and discards 95-98% of them
  (`pass_id = -1` in `job.log`). The few points left cover 1-7% of the field.
- **The log order can be recovered.** Sorting by (`Field`, `Dataset`, `Date`, `Product`, `Obj__Id`) brings the median
  step back to 0.3-1.7 m, with 0-0.1% of steps over 20 m, for FT2002, FT2004, HR_Yield_2022 and HR_Yield2022_teasley.
  For combinedyield2022 (3 products) 40% of steps stay over 20 m, so it needs a closer look. Checked on the portal's
  staging copies of the same files, under `yield_project/web_portal/portal_outputs/.staging/`.
- For SoyBean2022_45 the trial's set-2 originals (Morrow1 / Morrow2) have standard raw columns and are an
  alternative.
- SoyBean2020_66 has a second uncertainty: its swath median is 19.25, below the heuristic's 20, so it was read as
  metres (a 19 m header). If it is in feet (5.9 m), the yield would be about 3× higher than computed.

### 4.2 Distance in inches converted as if feet (4, plus SoyBean2022_50)

- Ag Leader / SMS exports name the distance column `Distance_in` or `Distance_ft`. A shapefile truncates field names
  to 10 characters, so they arrive as `Distance_i` and `Distance_f`.
- The pipeline's unit heuristic converts `Distance_f` correctly (the label says feet). For `Distance_i` it falls back
  to the value: a median in [10, 100) is "assumed feet, converted to inches (×12)". A combine logging once per second
  at 1.5-2 m/s covers 59-79 in, which lands inside that range, so inch values are multiplied by 12.
- The implied combine speed after the conversion gives this away: 16.5-22.1 m/s (60-80 km/h) for these fields, against
  1.2-2.4 m/s for the correctly converted ones. The raw file of SoyBean2020_68 also logs `Speed_mph_` 3.69
  (65 in/s) next to a `Distance_i` of 65.
- **Effect:** each vehicle rectangle is 12× too long. Consecutive rectangles overlap, and rectangles from neighbouring
  passes overlap too. Overlap resolution then leaves later rectangles with slivers, and mass over a sliver gives
  absurd yields: median polygon yields of 31 to 2·10^10 t/ha, and twice as many resolved polygons as records (1.9-2.2×,
  against 0.9-1.0 normally). SoyBean2023_33 went the other way (mean 1.08 t/ha).
- **Fix:** read the `_i` suffix as inches, or check the implied speed (distance / duration should be 0.5-3 m/s), then
  re-run.

### 4.3 Partial harvest log (1)

SoyBean2018_92 (`FT1807_SMSmin.shp`) is in proper log order (median step 1.9 m) and has a plausible yield
(2.58 t/ha). But its 1,828 points cover only 2.9 ha of the 7.8 ha outline (38%), just under the 40% limit. The file
name ("SMSmin") suggests a reduced export of the trial area. The yield map itself looks valid, so it could be kept
with only the covered cells labelled.

## 5. The same distance bug in the 65 clean field-seasons

The ×12 conversion also hit fields that passed QC, because their yields stayed plausible. Ranking all 75 runs by
the implied combine speed (median distance / duration of the preprocessed points):

| VGIR_ID | distance column | implied speed (m/s) | resolved / vehicle polygons | mean yield (t/ha) |
|---|---|---|---|---|
| Corn2024_13 | `Distance_i`, ×12 | 14.5 | 0.94 | 13.05 |
| SoyBean2019_77 | `Distance_i`, ×12 | 17.6 | 0.97 | 4.08 |
| SoyBean2018_96 | `Distance_i`, ×12 | 19.8 | 0.99 | 2.61 |
| Other2021_62 | `Distance_i`, ×12 | 20.6 | 1.00 | 2.30 |
| SoyBean2020_69 | `Distance_i`, ×12 | 20.7 | 0.97 | 2.83 |
| SoyBean2018_90 | `Distance`, ×12 (median 70) | 21.3 | 0.96 | 2.74 |
| SoyBean2024_22 | `Distance_i`, ×12 | 22.4 | 0.95 | 3.71 |
| Corn2024_14 | `Distance_i`, ×12 | 26.2 | 0.97 | 11.05 |
| SoyBean2019_83 | `Distance`, read as metres (median 4.72) | 4.8 | 0.96 | 3.33 |

- Every other clean field-season implies 1.2-3.2 m/s.
- For SoyBean2019_83, a median of 4.72 per 1-s record is far more likely feet (1.4 m/s) than metres. Read as metres,
  its rectangles are 3.3× too long.
- Why the yields still look plausible: when the passes do not overlap sideways, each over-long rectangle keeps about
  the strip it adds beyond its predecessor, so the effective area, and with it the field-mean yield, comes out roughly
  right. But each record's footprint ends up displaced along the pass, up to about 5× the true step ahead of the point
  (around 10 m), and the first rectangle of each pass is too large. **The within-field pattern, which is what the
  paper evaluates, is therefore likely distorted in these 9 field-seasons.** Not yet quantified.
- These 9 hold 9 of the 65 field-seasons in every Missouri result in the paper.

## 6. Recommendations

In order of value for the paper:

1. **Fix the distance heuristic and re-run the 9 affected clean fields** (§5). Compare the old and new rasters
   (within-field correlation) to see whether the Missouri results need re-running.
2. **Recover the 5 sorted files** (§4.1) by reordering and the **4 ×12 QC failures** (§4.2) by the same fix: up to
   9 more field-seasons.
3. **Fix the writer bug** (2 field-seasons, §3.2).
4. **The 12 John Deere exports** (§3.1): the largest group, but they need a reconstruction rule we would have to
   validate.
5. **SoyBean2018_92** (§4.3): keep it with only the covered cells labelled, or accept the loss.

Items 1-3 change the pipeline's behaviour, which is in the project lead's repository (`yield_project`). They need
the project lead's go-ahead, or a documented pre-processing step on our side.
