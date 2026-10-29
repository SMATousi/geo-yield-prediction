# Own dataset v2: fixed ground truth, test- vs validation-selected epochs

Status: RUNNING (2026-10-29). Diagnosis of v1's losses: `spec/own-dataset-unprocessed.md`.

## 1. Ground truth v2

The unchanged portal pipeline (`yield_project` 0bd49ee), run on staged raw copies through
`geo-data-downloader/scripts/ground_truth/run_portal_pipeline.py --fixes`. The fixes are value rewrites of the staged
copy (`raw_fixes.py`), each applied only when the raw data contradict the pipeline's assumption, so a file the
pipeline already read correctly is byte-identical:

| fix | rule | field-seasons |
|---|---|---|
| John Deere export | one record per machine and timestamp, mass = `WetMass` × width × distance, 1-s duration; several machines: logs kept machine by machine, date-only time | 12 |
| row order | sort by `FID_SMSmin` / `TARGET_FID` / (`Field`, `Dataset`, `Date`, `Product`, `Obj__Id`) when the file-order GPS step is > 10 m; accepted if median step < 5 m and < 5% of steps > 20 m | 5 |
| distance unit | the unit (in / ft / m) whose length matches the GPS step within 30% | 10 |
| swath unit | from the monitor's productivity, `Prod_ac_h_` × 8.25 / `Speed_mph_` ft | 3 |
| writer bug | non-polygon rows dropped when writing polygon layers | 2 |

The raw drive is not mounted; every raw file has a hash-identical copy in the portal staging area or in
`gdd-workspace/raw/set2`.

Results (`gdd-workspace/ground_truth_v2/_summary.csv`):

- 89 of 90 raw files processed (SoyBean2024_23 is a boundary polygon only).
- 55 field-seasons bit-identical to v1 (polygons and 10 m target), 20 changed (exactly the diagnosed set), 14 new.
- QC: 85 pass (17 corn, 67 soybean, 1 other; the dataset uses the 84 corn and soybean). Still excluded: SoyBean2023_33
  (low yield and coverage, cause unknown), SoyBean2022_44 (1.19 t/ha) and SoyBean2024_21 (1.47 t/ha), whose low
  yields agree with the monitor's own yield column.
- The previously clean fields with a fix: within-field correlation of the v1 and v2 10 m targets 0.93–0.97 for the
  distance-only fields, 0.36–0.82 for SoyBean2021_57, SoyBean2020_69, SoyBean2020_67, Other2021_62 and
  SoyBean2018_90 (swath or distance with overlapping passes).

## 2. Layers and dataset

- Layers of the 14 new field-seasons downloaded with the same config (`config.groundtruth_v2.yaml`, data root
  `layers_v2dl`); PlanetScope also for the 4 field-seasons whose outline grew (SoyBean2020_64, SoyBean2022_42,
  SoyBean2022_45, SoyBean2022_50). 18/18 PlanetScope orders; SoilGrids timed out for SoyBean2020_70 (its 18
  SoilGrids channels are masked, POLARIS channels present).
- Workspace `gdd-workspace-v2` assembled from links (`scripts/own_v2_workspace.py`); targets by `rasterize-yield`.
- Artifacts `/root/owndata_artifacts_v2` (PVC `/data/YieldSAT/owndata_artifacts_v2`): 84 field-seasons, 252,136
  cells (v1: 64, 190,124). LOYO corn gains 2023 (6 folds).

## 3. Epoch selection

The paper protocol (`noval_` splits) selects the epoch on the test fold. The `val_` splits carve 15% of the training
field-seasons (grouped as the test folds) out for selection; their test folds are identical to the `noval_` ones
(asserted in `write_splits`).

Runs (`yieldsat_protocol_runs.py units --batch own-v2`): the eight paper arms × 3 seeds × 33 folds × 2 selections.

| job | units | pods |
|---|---|---|
| `protocol_own_v2_job.yaml` | 1,188 (S2-only, ps10, ps3x3, ps3x3sp, dem10, ps3x3sp-dem10) | 20 × 2 |
| `protocol_own_v2_win_job.yaml` | 396 (pswin, ps3x3-pswin) | 12 × 2 |

Outputs: `protocol/own_v2/<group>_noval_s0/...` (test-selected) and `<group>_val_s0/...` (validation-selected).

## 4. Results

Pending.
