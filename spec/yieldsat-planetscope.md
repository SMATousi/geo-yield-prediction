# PlanetScope on a YieldSAT subset: does 3 m imagery raise within-field R² there too?

**Status (2026-10-26): done. PS 3×3 + 30 m window raises within-field R² on all three pairs under both protocols
(mean +0.019 CV10, +0.019 LOYO; every paired CI above zero). Results §4, `results/ysat_ps_cmp.json`.**

Related:
- [own-dataset-training.md](./own-dataset-training.md) §8: on our Missouri data the flat PS 3×3 sub-grid + 30 m window
  is the best arm (+0.044 CV10, +0.058 LOYO within-field R², significant in all four crop–protocol cells).
- [yieldsat-input-information.md](./yieldsat-input-information.md): half of YieldSAT's within-field variance sits at
  20–50 m scales; coarsening S2 lowers within-field R² (the resolution experiment, supplementary).

## 1. Question and design (project lead, 2026-10-25)

Missouri is one dataset with clean yield maps. This tests the same claim on YieldSAT: **adding PlanetScope to
the paper's best model raises within-field R²**.

| Decision | Choice |
|---|---|
| D1 Arm | **Fixed before any YieldSAT result**: the Missouri-best arm, `--planetscope 3x3 win` (flat 3×3 sub-grid + 30 m window, conv encoder) on `ours-p3nbr-dense-fc-rel10`. A confirmatory test, no further arm search. |
| D2 Pairs | ARG-S (440 field seasons), ARG-C (185), BRA-C (118): the most within-field room (noise ceiling 0.82–0.83; 80 m oracle well above our model). URG-S (ceiling 0.48) and the German pairs (0.66; 42% of S2 slots cloud-masked) left out. BRA-C was a development pair of the model, not of the PS arm. |
| D3 Protocols | CV10 and LOYO as in the paper; LORO needs all pairs. |
| D4 Baseline | The paper's own S2-only runs of the same model, seeds 0–2 (identical arguments, splits and seeds; since they ran, the training code changed only by adding PS / DEM / spatial streams). |
| D5 Seeds | 0–2, pooled as in `results/rel_cmp.py` (REL_SEEDS). |

## 2. Data

**Footprints.** Per field season the exact YieldSAT grid (geometry JSON: UTM EPSG, 10 m affine, width, height) + 30 m,
so the 30 m window of edge cells is covered (`/root/yieldsat_artifacts/planetscope/aois_ARG-S_ARG-C_BRA-C.geojson`,
743 field seasons, 11,996,340 grid cells). The raw S2 rasters sit on this grid exactly (transform, size, EPSG equal on
12 of 12 sampled subset fields), and the index `grid_row` / `grid_col` index it.

**Download** (`yieldsat_ps_download.py`; geo-data-downloader's selection and order code, second Planet account):
- window seeding − 10 d to harvest + 10 d (YieldSAT masks slots outside [seeding, harvest]);
- as Missouri: one scene per 7-day bin (full AOI coverage, then 8-band, then clear %, then nearest the bin centre),
  cloud cover ≤ 0.1, 8-band SR where offered, else 4-band; clipped to the AOI;
- areas in South America Albers (ESRI:102033; the downloader's CONUS Albers would distort them).

Selection (search only, 2026-10-25):

| Pair | Field seasons | Scenes | 8-band | 7-day bins filled | Clipped km² |
|---|---|---|---|---|---|
| ARG-S | 440 | 10,722 | 68.5% | 95.2% | 19,605 |
| ARG-C | 185 | 5,284 | 80.3% | 91.4% | 8,558 |
| BRA-C | 118 | 2,441 | 59.4% | 77.8% | 4,339 |
| All | 743 | 18,447 | 70.7% | 91.4% | 32,502 |

By year: no 8-band before 2020 (4-band only), 8-band ≥ 92% from 2021; 2017 bins 34–69% filled (ARG-C, ARG-S,
BRA-C), every other year ≥ 69%. The charge (clipped area, measured on this account for Missouri) is 0.1% of the
30 M km² access-period quota (55,570 km² used before this run).

**Artifacts** (`yieldsat_build_ps_subset.py`, into `neighbourhood_dense/<Country>/`): `ps_win/<field_code>.npy`,
`ps_win_scene.npy` (−1 outside the subset), `ps_field_sums.npz`, `ps_subset_manifest.json`. Same per-cell, per-slot
scene rule and 3.33 m padded grid as Missouri (`yieldsat_build_owndata._ps_window`). No `ps3x3.npy` (it would be
~100 GB for the two countries): the dataset cuts the 3×3 sub-grid from the window's centre
(`ps3x3_from_window`; on Missouri the cut equals the stored 3×3 exactly, inputs and masks, 2,048 cells × 72 slots).
Normalization per band from the cell mean (centre 3×3) over the training field seasons.

Delivered and built (2026-10-25): 18,418 of 18,447 selected scenes delivered (29 not in the order results), 31 GB
of clipped scenes, uploaded to `/data/YieldSAT/yieldsat_artifacts/planetscope/layers` (per-file sizes checked) and
built on the PVC: 11,720 scenes used in the windows (one per cell and dense slot at most), 23 GB (Argentina) + 3.8 GB
(Brazil) of window stacks. Every one of the 5,260,617 cells of the 743 field seasons has PlanetScope in at least one
slot; PS slots per cell median 15 (p10 11, p90 19) of the in-season dense slots (Missouri: 13–14).

**Registration** (`results/ps_registration_qc.py` → `results/ps_registration_qc.json`): per field season, the mean
over slots of the across-cell correlation of S2 NDVI with PlanetScope NDVI (cell mean of the 3×3), PS shifted by
up to ±3 sub-pixels (±10 m). 740 of 743 field seasons (3 have no slot with ≥ 500 cells seen by both):

| Pair | Fields | r at 0 (median) | r at best shift | Median gain | Mean best dy, dx (m) | Best shift ≥ 6.6 m | At the ±10 m edge |
|---|---|---|---|---|---|---|---|
| ARG-S | 438 | 0.824 | 0.849 | 0.014 | +1.1, −0.5 | 23.5% | 1.4% |
| ARG-C | 184 | 0.802 | 0.834 | 0.018 | +2.2, +1.1 | 25.0% | 0.5% |
| BRA-C | 118 | 0.763 | 0.792 | 0.020 | +3.4, −1.2 | 17.8% | 0.8% |
| All | 740 | 0.809 | 0.836 | 0.016 | +1.7, −0.2 | 23.0% | 1.1% |

PlanetScope sits on the yield / S2 grid to within about one 3.3 m sub-pixel for most fields (median best offset
3.3 m; 77% ≤ 4.7 m), with no systematic shift beyond ~2 m; correcting it would raise the PS–S2 agreement by only
~0.02. The data are used as delivered (no co-registration), as on Missouri; the 30 m window lets the encoder
absorb a residual sub-cell offset.

## 3. Runs

Jobs `smatousi-yieldsat-protocol-ysat-ps` (20 pods: A10 / 3090 / 4090) and `-b` (6 L4 pods) share the units' claims.

`yieldsat_protocol_runs.py units --batch ysat-ps` → `cluster/tabm/protocol_ysat_ps_units.json`: 156 units
(3 pairs × CV10 10 folds + LOYO 8/8/6 folds × 3 seeds), outputs next to the paper runs
(`paper/<group>/s2_adm/ours-p3nbr-dense-fc-rel10-ps3x3-pswin_seed<s>/<pair>/foldNN`).
Job `cluster/nautilus/protocol_ysat_ps_job.yaml`: 20 pods × 2 runs, 32 Gi.

Comparison: `REL_SUB=paper REL_SEEDS=0,1,2 REL_BASE=ours-p3nbr-dense-fc-rel10 python results/rel_cmp.py
ours-p3nbr-dense-fc-rel10-ps3x3-pswin ARG-S,ARG-C,BRA-C ysat_ps_cmp`.

## 4. Results

`results/ysat_ps_cmp.json`. Our model (`ours-p3nbr-dense-fc-rel10`, S2 only, the paper runs) vs the same model with
`--planetscope 3x3 win`; seeds 0–2 pooled, Δ with a 95% paired field-season bootstrap CI.

| Protocol | Pair | Fields | Within S2 | Within +PS | Δ within [CI] | Pooled S2 | Pooled +PS | Δ pooled [CI] | Median field R² |
|---|---|---|---|---|---|---|---|---|---|
| CV10 | ARG-S | 440 | 0.543 | 0.555 | +0.012 [+0.005, +0.019] | 0.737 | 0.739 | +0.001 [-0.007, +0.010] | 0.501 → 0.528 |
| CV10 | ARG-C | 185 | 0.456 | 0.481 | +0.025 [+0.011, +0.042] | 0.726 | 0.744 | +0.018 [+0.005, +0.031] | 0.437 → 0.457 |
| CV10 | BRA-C | 118 | 0.280 | 0.299 | +0.019 [+0.010, +0.028] | 0.490 | 0.504 | +0.015 [+0.003, +0.026] | 0.309 → 0.321 |
| LOYO | ARG-S | 440 | 0.491 | 0.502 | +0.011 [+0.004, +0.018] | 0.654 | 0.655 | +0.002 [-0.014, +0.016] | 0.441 → 0.454 |
| LOYO | ARG-C | 185 | 0.345 | 0.376 | +0.032 [+0.008, +0.061] | 0.501 | 0.602 | +0.101 [+0.062, +0.148] | 0.326 → 0.349 |
| LOYO | BRA-C | 118 | 0.194 | 0.208 | +0.014 [+0.005, +0.023] | 0.197 | 0.244 | +0.047 [+0.015, +0.076] | 0.203 → 0.224 |

Mean over the three pairs (Δ [CI]): CV10 within +0.019 [+0.013, +0.025], pooled +0.011 [+0.005, +0.017];
LOYO within +0.019 [+0.010, +0.029], pooled +0.050 [+0.033, +0.068].

- The answer to §1 is yes: the gain is not specific to our Missouri maps. Within-field R² rises in all 6 pair–protocol
  cells, each CI above zero, and the three seeds of the PS arm all beat all three S2 seeds in every cell.
- The gain is about half of Missouri's (+0.044 CV10, +0.058 LOYO). Untested candidate reasons: the yield maps'
  own processing and resolution, the PS-to-grid registration (§2: best shift ≥ 6.6 m for 23% of fields), and a
  stronger S2 baseline (ARG-S within 0.54 vs Missouri 0.33).
- Pooled R²: no change on ARG-S, gains on the corn pairs; the large LOYO ARG-C pooled gain (+0.10) comes from
  better between-field levels in held-out years, with a wide CI.

## 5. Log

- 2026-10-25: design fixed (§1); AOIs and selection (§2); orders started (10 workers, then 60: one order takes > 8 min).
- 2026-10-25: 60 parallel orders hit HTTP 429 on order creation (20 fields failed); creation serialized 2 s apart with
  a long backoff, restarted: 743/743 field seasons, 0 failures, ~4.5 h. Upload in 10-field tar chunks (50-field
  chunks timed out). Windows built on the PVC (results pod, 7 workers, ~20 min). CPU smoke of a BRA-C unit (1 epoch,
  5 steps) loads the window and the cut 3×3 and validates; job submitted.
- 2026-10-26: 156/156 units done (~22 h, 0 failed); rel_cmp → §4.
