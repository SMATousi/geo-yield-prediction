# US pilot pretraining dataset (YieldSAT-aligned, 250k points)

**Status (2026-10-06): extracted and audited.** Assembly into the YieldSAT
cache format, loader integration with a crop filter, and the parity test
(NP-04a) are next.
- Location: `/home1/pupil/SMATousi/YieldSAT-US-Pilot/`.
- Design and decisions: [yieldsat-us-national-pretraining.md](./yieldsat-us-national-pretraining.md),
  "YieldSAT-aligned acquisition design" (D13–D19).
- Builder: `yieldsat_us/` and `yieldsat_us_pilot.py` (stages `frame`,
  `extract`).

This dataset is unrelated to the Missouri AOI build described in
[yieldsat-us-pretraining-point-dataset.md](./yieldsat-us-pretraining-point-dataset.md).
That build is not YieldSAT-aligned.

## 1. Contents

| Item | Value |
|---|---|
| Points | **249,952** 10 m cropland cells, no yield or target |
| Harvest years | 2021 (51,991), 2022 (49,999), 2023 (47,993), 2024 (49,980), 2025 (49,989) |
| Clusters | **125** cluster cells of 10.23 km × 10.23 km (EPSG:5070), ≈ 2,000 points each, in 31 states |
| Regions (NOAA, D17) | Upper Midwest 15, Ohio Valley 20, Northern Rockies & Plains 25, South 25, Southeast 10, Northwest 12, Northeast 8, Southwest 5, West 5 |
| Cluster strata | corn 46, soybean 37, wheat 17, canola 10, winter wheat 10, spring wheat 5 |
| Point classes (D18: all CDL cropland) | 87 CDL classes. Largest: corn 70,539; soybean 67,537; winter wheat 36,038; fallow/idle 15,352; spring wheat 12,141; canola 7,547; alfalfa 5,587; sorghum 5,302; other hay 5,169; double-crop winter wheat/soybean 4,925 |
| Size on disk | 108 MB (`clusters/*.npz`) + 6.6 MB of weather cache |

## 2. Sampling (NP-04b)

- **Frame:** the national CDL 30 m of each harvest year, downloaded one year
  at a time and deleted after use (D19). Pixel counts per crop group per
  cluster cell; NOAA region from the Census 2022 state boundaries at the
  cell centre.
- **Cluster selection:**
  - The region's clusters are split evenly over the 5 years (rotated per
    region), then across the region's crop strata in proportion to stratum
    cropland area.
  - Within a stratum, cells with ≥ 30% cropland and a stratum share ≥ 25%
    are drawn with probability ∝ stratum area. The share threshold relaxes to
    10%, then 2%, if a stratum has no candidates.
  - Each cell is used once across years.
  - Seeds are deterministic (SHA-256 of year / region / stratum).
- **Points:** one random 10 m cell centre per 100 m block inside the
  cluster, cropland only (CDL 1–61, 66–80, 200–255), then stratified by CDL
  class (largest remainder) to ≈ 2,000.
- **Recorded per cluster** (`frame_<year>_clusters.json`): region, state,
  stratum, cropland fraction, stratum share, selection weight and group
  counts. Per point (`frame_<year>_points.npz`): EPSG:5070 x/y, lon/lat, CDL
  code, cluster.

## 3. Per-point features (NP-04c/d), YieldSAT conventions

Layout per point: `temporal (24, 16)` = 12 S2 bands + 4 weather;
`static (104)` = DEM, terrain (4), soil (48), soil uncertainty (48, NaN),
coords (3); `times (24)` = days since 1970. These are the YieldSAT cache
channel orders.

| Family | Source | Processing |
|---|---|---|
| Season dates | **NASS Crop Progress** (QuickStats API): the first week-ending date at which the state × crop × harvest year reaches 50% planted and 50% harvested. Progress is filed under the harvest (crop) year, including winter wheat's autumn planting | 192,146 points (77%) on Crop Progress dates. 57,806 (23%) on documented fallback windows: the classes NASS does not publish (89 keys), plus a few state × crop gaps (corn 3, soybean 1, winter wheat 7, spring wheat 8, canola 11 keys) |
| Slots | Calendar month k from January of (harvest year − 1), one acquisition per month from the seeding month to the harvest month | Dated slots per point p10/p50/p90 = 5/7/9 |
| Sentinel-2 L2A | AWS Earth Search `sentinel-2-l2a` COGs, **pixel-only HTTP range reads** (D14); no scene stored | Per point and month: prefer clear at the point (SCL 4/5/6), then lowest scene cloud cover, latest, item ID. Native pixel containing the point. Harmonized DN: Earth Search applies the BOA offset (`earthsearch:boa_offset_applied`). Items without public COGs (requester-pays JP2) are skipped, and unreadable assets count as missing. **98.4% of dated slots clear at the point** |
| Weather | Open-Meteo archive: temperatures from ERA5-Land (0.1°); precipitation from ERA5 (0.25°), because Open-Meteo serves no ERA5-Land precipitation | **YieldSAT operator:** sum of daily values over the inclusive interval [t_prev, t] between consecutive dated slots, K·days and metres; first dated slot NaN. Verified on `Raw.zip` that YieldSAT uses exactly this operator |
| Soil | ISRIC SoilGrids 2.0 VRTs (Homolosine) | 8 properties × 6 depths, mean, mapped units (clay g/kg, SOC dg/kg, pH × 10, …), cubic interpolation at the point. Uncertainty bands NaN (excluded by default in YieldSAT). 0.8% of points missing |
| DEM / terrain | NASADEM (reprocessed SRTM, 1″) via Planetary Computer | Derivatives on the native lat/lon grid as recovered from YieldSAT (`Raw.zip`): slope = Horn rise/run per degree (YieldSAT's ≈ 84,000× scale), aspect = downslope azimuth, curvature = Zevenbergen–Thorne (1/100 m), TWI NaN; cubic interpolation at the point |
| Coordinates | Point lon/lat | Unit-sphere x/y/z, as YieldSAT's `coord_*` (metadata) |

Per-slot provenance is stored: S2 item ID, SCL class, clear flag. Per point:
calendar source, seeding/harvest day, CDL code, coordinates.

## 4. Storage

```text
YieldSAT-US-Pilot/
  frame_<year>_clusters.json   cluster selection records
  frame_<year>_points.npz      sampled points (x, y, lon, lat, cdl, cluster)
  frame_<year>_cellstats.npz   per-cell crop-group counts and NOAA region (frame audit)
  calendar_cache.json          state|group|year -> seeding, harvest, source
  clusters/<cluster_id>.npz    temporal, static, times, seeding_day, harvest_day,
                               calendar_source, cdl, lon, lat, x, y, item_id, scl,
                               clear, items (S2 provenance), cluster (record)
  logs/extract_<year>.log      per-cluster timing and read counts
  _cache/era5land/             Open-Meteo responses per weather cell
```

No credentials are stored. The NASS key is supplied through the environment
(`NASS_API_KEY`).

## 5. Audit vs YieldSAT (2026-10-06)

All channels fall in YieldSAT's ranges and units, with wider spread as
expected for 31 states:

| Channel (p5 / p50 / p95) | US pilot | Argentina | Germany |
|---|---|---|---|
| B04 (DN) | 229 / 1,044 / 2,314 | 195 / 869 / 2,032 | 243 / 696 / 1,840 |
| B08 (DN) | 1,530 / 2,916 / 5,090 | 1,428 / 2,820 / 5,392 | 1,716 / 3,376 / 5,561 |
| temp_mean (K·day per slot) | 3,232 / 9,199 / 15,250 | 0 / 8,946 / 11,870 | 1,755 / 8,444 / 14,090 |
| total_prec (m per slot) | 0.0008 / 0.056 / 0.196 | 0 / 0.049 / 0.168 | 0.004 / 0.045 / 0.125 |
| DEM (m) | 18 / 371 / 1,270 | 77 / 122 / 573 | 68 / 123 / 270 |
| Slope (YieldSAT scale) | 879 / 3,222 / 10,110 | 1,143 / 2,559 / 5,536 | 865 / 2,682 / 8,501 |
| Curvature | −0.45 / 0.00 / 0.44 | −0.68 / 0.00 / 0.68 | −0.51 / 0.01 / 0.53 |
| Clay 0–5 (g/kg) | 137 / 265 / 390 | 152 / 246 / 351 | 109 / 145 / 261 |
| SOC 0–5 (dg/kg) | 136 / 338 / 772 | 200 / 245 / 321 | 419 / 521 / 672 |
| pH 0–5 (× 10) | 50 / 66 / 74 | 60 / 64 / 73 | 58 / 63 / 67 |

## 6. Known differences from YieldSAT

1. **Clear-sky selection per point** (YieldSAT: one acquisition per field
   and month). As a result, 98.4% of the US dated slots are clear, vs 58–94%
   in YieldSAT (6–42% of its dated slots are cloud-masked).
2. **Precipitation from ERA5 (0.25°)** instead of ERA5-Land, the forcing
   ERA5-Land interpolates; temperatures are ERA5-Land.
3. **Season dates are state-level calendar estimates** (Crop Progress 50%
   weeks or fallback windows), not farmer-reported dates.
4. **Soil uncertainty and TWI are NaN** (both unused or mostly missing in
   YieldSAT).
5. **The DEM is NASADEM** (reprocessed SRTM) rather than the original SRTM
   product. Terrain is derived with YieldSAT's recovered conventions, and
   curvature's scale is still to be confirmed in NP-04a.
6. **87 CDL classes** are present; YieldSAT has 4 crops. Training applies a
   crop filter (D18).

## 7. Build log

- 2026-10-04 — Frame built: 125 clusters, 249,952 points. National CDL
  downloaded per year and deleted.
- 2026-10-06 — NASS key received; calendar prefilled (371 keys).
  - The QuickStats client was fixed for exact series names per crop, the
    crop-year convention, throttling (paced requests; failures never cached)
    and HTTP 400 meaning "no series".
- 2026-10-06 — Extraction of all 125 clusters (≈ 130 s per cluster on
  average; median 184 S2 acquisitions scanned and 252 band reads per
  cluster; 5 harvest years in parallel).
  - Fixed during the run: skip non-COG Earth Search items; retry
    Open-Meteo network timeouts.
- 2026-10-06 — Audit (§5) passed: value ranges and units consistent with
  YieldSAT.

## 8. Next tasks

| ID | Deliverable | Acceptance |
|---|---|---|
| NP-04e | Assemble into the YieldSAT cache format (`cache/US/{temporal,static,times}.npy`, field stats v3, `index/US/{rows.npz,fields.json}`; season = cluster × crop group) | `YieldSATPointDataset` loads it with no target |
| NP-04f | Pretraining loader integration + crop filter (corn, soybean, wheat, canola → rapeseed; other classes excluded or kept by flag) | Unit tests; the pretraining entry point accepts the US corpus next to YieldSAT |
| NP-04a | Parity test: re-extract S2 for ≥ 20 YieldSAT fields with this pipeline and compare with the cache | Same monthly slot dates for ≥ 90%; median band difference ≤ 2% |
