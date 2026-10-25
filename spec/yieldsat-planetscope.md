# PlanetScope on a YieldSAT subset: does 3 m imagery raise within-field R² there too?

**Status (2026-10-25): data download running; builder, units and job ready.**

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

## 3. Runs

`yieldsat_protocol_runs.py units --batch ysat-ps` → `cluster/tabm/protocol_ysat_ps_units.json`: 156 units
(3 pairs × CV10 10 folds + LOYO 8/8/6 folds × 3 seeds), outputs next to the paper runs
(`paper/<group>/s2_adm/ours-p3nbr-dense-fc-rel10-ps3x3-pswin_seed<s>/<pair>/foldNN`).
Job `cluster/nautilus/protocol_ysat_ps_job.yaml`: 20 pods × 2 runs, 32 Gi.

Comparison: `REL_SUB=paper REL_SEEDS=0,1,2 REL_BASE=ours-p3nbr-dense-fc-rel10 python results/rel_cmp.py
ours-p3nbr-dense-fc-rel10-ps3x3-pswin ARG-S,ARG-C,BRA-C ysat_ps_cmp`.

## 4. Results

Pending.

## 5. Log

- 2026-10-25: design fixed (§1); AOIs and selection (§2); orders started (10 workers, then 60: one order takes > 8 min).
