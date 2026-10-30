# Own dataset v2: fixed ground truth, test- vs validation-selected epochs

Status: DONE (2026-10-30); paper on v2, test-selected. Diagnosis of v1's losses: `spec/own-dataset-unprocessed.md`.

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

All 1,584 units (2026-10-30). Within-field / pooled R2 of S2 only, seeds 0-2 pooled (`results/own_cmp_all.json`):

| | v1 test-sel. | v2 test-sel. | v2 val-sel. |
|---|---|---|---|
| MO-S CV10 | 0.211 / 0.293 | 0.273 / 0.435 | 0.244 / 0.179 |
| MO-C CV10 | 0.440 / 0.431 | 0.462 / 0.448 | 0.382 / 0.159 |
| MO-S LOYO | 0.193 / 0.245 | 0.241 / 0.395 | 0.228 / 0.203 |
| MO-C LOYO | 0.381 / 0.219 | 0.412 / 0.266 | 0.329 / 0.054 |

Mean within-field gain over S2 only, test-selected (val-selected in brackets): ps10 +0.033 / +0.055 (+0.048 / +0.057),
ps3x3 +0.040 / +0.019 n.s. (+0.054 / +0.048), pswin +0.036 / +0.041 (+0.048 / +0.063), ps3x3-pswin +0.056 / +0.062
(+0.050 / +0.094), ps3x3sp -0.004 / +0.010 n.s., dem10 -0.003 / -0.011 n.s., ps3x3sp-dem10 +0.010 / +0.012 n.s.
(CV10 / LOYO). ps3x3-pswin vs ps3x3: +0.015 n.s. / +0.043; vs ps10: +0.023 [+0.006, +0.047] / +0.007 n.s.

- The data fix raises every S2-only number; the PS gain is at least v1's (+0.044 / +0.058).
- Validation selection costs within-field R2 (0.01-0.08) and most of pooled R2: the carve-out (about two corn fields
  per fold) picks the epoch poorly for field levels. The PS gain holds under it.
- Decision (project lead, 2026-10-30): test selection stays the default (as for YieldSAT); val selection is not
  reported for now.
- Per field (`results/field_cmp.json`, `FC_OWN_SUB=own_v2`): PS improves 70% (CV10) / 75% (LOYO) of 79 field
  seasons, median gain +0.027 / +0.046. Noise ceiling 0.82 (soybean) / 0.92 (corn). Paper updated to v2.

## 5. Baselines on our data (2026-10-30, project lead: "run the papers and our added baselines on our dataset")

The four YieldSAT models and the two strong baselines, with their YieldSAT arguments unchanged (no search), on the v2
data: 3 seeds, test- and val-selected (as §3), CV10 and LOYO. TabM and LightGBM also with the PlanetScope 3×3
sub-grid as flat features (`--features F0P`: F0 + `ps3x3.npy`, 72 slots × 72 values, one validity flag per slot),
to test whether the PS gain is specific to our model.

| Model | Input | Tag |
|---|---|---|
| 3D-LSTM, 3D-ConvLSTM | monthly series (`cache/Missouri`, below) | `paper_3dlstm`, `paper_3dconvlstm` |
| AFF, MMGF | dense series | `paper_aff`, `paper_mmgf` |
| TabM, LightGBM | dense F0 / F0P | `tabm-f0-dense-bs2048`, `lgbm-f0-dense`, `tabm-f0p-dense-bs2048`, `lgbm-f0p-dense` |

**Monthly cache** (`yieldsat_build_owndata.py --monthly_only`), the release's format for the two input-fusion
models: 24 calendar months from January of harvest year − 1; per field and month the in-season S2 acquisition with
the most clear cells (SCL 4/5/6; ties nearest mid-month), cells not clear in it NaN, slot time = that acquisition's
date; weather summed over the month's in-season days; months outside [seeding, harvest] NaN (as the dense series;
the release keeps some out-of-season months in Argentina and Brazil). 5.62 clear S2 months per cell (dense: ~11
clear slots).

**Runs** (local RTX 3090, `yieldsat_protocol_runs.py units --batch own-baselines`): 840 GPU units
(`cluster/tabm/protocol_own_baselines_gpu_units.json`, one fold per paper-model unit, one row per tabular unit) and
48 LightGBM units (`..._lgbm_units.json`), claim pool under `/root/owndata_results_v2/_claims`, outputs
`/root/owndata_results_v2/own_v2/<group>/s2_adm/<tag>_seed<k>/<pair>/foldNN`. The paper models are CPU-bound
(data loading; GPU ~25%): 5 concurrent runs.
