# YieldSAT knowledge pretraining on the point model

**Status:** specified 2026-10-04 (user decisions below). Implementation log in §9.
This is the point-model realization of [knowledge_pretraining.md](./knowledge_pretraining.md)
v2 and the YieldSAT library in `spec/yieldsat_knowledge_assets/`. The image-model
package `yieldsat_knowledge/` stays as is; we return to it later. Success is judged
by the separate criteria in [pretraining/success-criteria.md](./pretraining/success-criteria.md).

## 1. User decisions (2026-10-04)

| ID | Decision |
|---|---|
| K1 | **The 6 relationships (`ys_r01`–`ys_r06`) and their 12 concepts are treated as expert-verified** (reviewer: the project lead). Gate G1 of [yieldsat-foundation-model.md](./yieldsat-foundation-model.md) is satisfied for these entries. The review is recorded in the library JSON (`review.status = approved`). |
| K2 | **Concept estimators are simple, informed rule-based estimators first** (§3). No VLM/LLM teacher for now. |
| K3 | **Everything is implemented, tested and trained on the point model from now on.** The image model comes back later. |
| K4 | **Pretraining corpus, first stage: YieldSAT inputs of all 9 country–crop pairs**, labels never used (§5). The US national corpus ([yieldsat-us-national-pretraining.md](./yieldsat-us-national-pretraining.md)) is added later in measured stages. |

## 2. What is pretrained

- **The point model's sensor side:** the per-stream encoders (S2 series,
  weather, DEM, terrain, soil), the Perceiver fusion and the date encoding,
  i.e. what `YieldSATPointModel.sensor_state_dict()` exports.
  - The backbone is the default point model (`perceiver_summary`). It is the
    best DEV configuration so far (round 2 found no better variant; the
    early-fusion + neighbourhood variant is within 0.01).
- **Losses** (`KnowledgePointPretrainer`):
  1. **Sensor preservation (SSL):** the existing point objectives, masked
     observation reconstruction and forecast of the last valid observation
     (`YieldSATPointPretrainer`), always on.
  2. **Concept grounding:** a linear projector per stream maps the stream's
     summary embedding into the frozen CLIP text space, normalized. The
     concept logit is `cos(z_stream, t_concept) / τ` (τ = 0.1); loss is binary
     cross-entropy against the estimator's soft target. Unknown targets are
     masked.
  3. **Relational distillation (`directed_displacement_v1`):** for a rule
     A → B, on samples where both concept targets are ≥ the rule's
     `minimum_presence` (0.5) and the rule gate is open, the loss is
     `1 − cos(normalize(z_B − z_A), normalize(t_B − t_A))`, weighted by
     `p_A · p_B · gate`.
  - Same formulation as the image package (`yieldsat_knowledge/model.py`),
    so the two branches stay comparable.
- **Total:** `L = L_ssl + λ_g · L_ground + λ_r · L_rel`, with λ_g = λ_r = 1 by
  default.
- **Yield and the crop token are never used in pretraining.** The heads are
  discarded; only the sensor state transfers to fine-tuning
  (`--init_sensor_ckpt`).

## 3. Rule-based concept estimators (K2)

Raw indices are computed once per point-cache row, from inputs only
(`yieldsat_build_knowledge.py` → `<artifact_root>/knowledge/<Country>/concept_raw.npz`).
They are fold-independent. **Soft targets are the percentile rank of a raw
index in a reference population fitted on the fold's training rows only**,
within a declared stratum (`KnowledgeReference`, stored with each
checkpoint). This implements "relative to a reviewed local reference" without
fixed global thresholds.

Units:
- S2: L2A digital numbers / 10,000 = reflectance.
- Weather: interval sums. Temperatures are Kelvin-day sums and precipitation
  is metre sums. A slot's interval runs from the previous dated slot
  (exclusive) to this slot (inclusive), so a daily mean = sum / interval
  days. Each row's first dated slot has no interval and is excluded (point
  contract).
- Season: seeding → harvest day of the row. Mid-season = 30–80% and late
  season = 60–100% of that window.

| Concept | Stream | Raw index | Reference stratum | Soft target |
|---|---|---|---|---|
| `rain_supported` | weather | mid-season precipitation (mm) | country × crop | rank |
| `warm_regime` | weather | mid-season mean daily mean temperature (°C) | country × crop | rank, set to 0 when the mid-season mean daily max exceeds the crop's non-stress limit (corn 35, soybean 35, wheat 32, rapeseed 30 °C) |
| `cool_regime` | weather | as `warm_regime` | country × crop | 1 − rank |
| `optical_growth` | S2 | NDVI rise: max NDVI in season − min NDVI in the first 30% of the season | country × crop | rank |
| `active_spectral_state` | S2 | mean NDRE (B8A, B05) in mid-season | country × crop | rank |
| `persistent_canopy` | S2 | mean late-season NDVI / season max NDVI | country × crop | rank |
| `spectral_moisture_state` | S2 | mean NDMI (B08, B11) in mid-season | country × crop | rank |
| `clay_rich_surface` | soil | clay, depth-weighted 0–30 cm (5/10/15 cm) | country | rank |
| `fine_surface_texture` | soil | (clay + silt) / (clay + silt + sand), 0–30 cm | country | rank |
| `organic_surface` | soil | SOC, depth-weighted 0–30 cm | country | rank |
| `low_elevation_position` | DEM | cell elevation − field-season mean elevation (m) | within the field season | 1 − within-field rank; **unknown** when field relief (p95 − p5) < 1 m |
| `pole_facing_aspect` | terrain | cos(aspect − pole azimuth); pole = 180° in Argentina, Brazil, Uruguay, 0° in Germany | — | max(0, cos); **unknown** when slope is below its country 25th percentile (flat) |

**Rule gates** (applicability context only; never the consequence's value):

| Rule | Gate |
|---|---|
| `ys_r01` rain → optical growth | open for rainfed summer crops (corn, soybean); **abstain** for wheat and rapeseed |
| `ys_r02` warm → active spectral state | open when the mid-season mean daily max is within the crop's non-stress limit; closed otherwise |
| `ys_r03` clay → persistent canopy | open in limited-rainfall seasons (mid-season precipitation rank < 0.5 in the fold reference); closed otherwise (waterlogging can reverse the association) |
| `ys_r04` low elevation → fine texture | open when field relief ≥ 1 m (a position is defined); abstain otherwise |
| `ys_r05` pole-facing → cool regime | **always abstain**: field-centroid ERA5 cannot resolve within-field thermal differences, as the rule itself states. The rule contributes no loss until weather at matching support exists |
| `ys_r06` organic → spectral moisture | open for mineral soils (0–30 cm SOC below the organic-soil limit of 120 g/kg); closed otherwise |

- **Supports:** weather concepts are field-season properties (the weather
  series is identical for all cells of a field season). Soil comes from
  250 m SoilGrids, so cells of one field share values. Grounding and
  relation losses are therefore **averaged per field season** before
  averaging over the batch: thousands of cells sharing one weather series
  are not counted as thousands of independent judgments.
- **Estimators are diagnostic, not ground truth.** Their agreement with
  sensor-side predictions is audited (success criterion I1); a concept whose
  estimator is degenerate (near-constant or mostly unknown) is reported and
  excluded.

## 4. Controls

| Control | Change | Tests |
|---|---|---|
| `shuffled` | Concept→text assignments permuted (a fixed derangement) and relation directions recomputed from the shuffled prototypes; same losses and compute | knowledge content vs regularization |
| `notext` | Learned, randomly initialized concept prototypes (trainable) instead of frozen CLIP vectors | role of the language prior |
| `ssl_long` | SSL only, with as many extra steps as the knowledge losses add | compute-matched |

## 5. Pretraining corpus and units (K4)

- **Corpus:** every YieldSAT field season of all 9 pairs (4 countries),
  inputs only. For each DEV evaluation fold, the seasons that would leak it
  are excluded:

| Protocol | Pretraining unit | Excluded |
|---|---|---|
| CV10 | fold *k* (one unit per *k*) | the fold-*k* test seasons of every DEV pair |
| LOYO | harvest year *Y* (one unit per year present in DEV test folds) | every season of year *Y*, all countries and crops |
| LORO | region *R* of a DEV pair's country | every season in *R*, all crops of that country |

- Each DEV fine-tuning run starts from the checkpoint of its unit. That is
  fewer pretraining runs than fine-tuning runs (10 + ~8 + 29 units vs 94
  runs per arm on DEV).
- **Split manifests** for the units (`yieldsat_pretrain_units.py`):
  - the same contract and format as the fold manifests;
  - all four countries;
  - partitions train (eligible seasons) / val (10%, grouped by physical
    field);
  - a `check_unit` audit proves no excluded season is present.
- **Normalization:** per-country train statistics (`--norm_pooling
  per_country`). Fine-tuning on one pair uses that country's training
  statistics, consistent with pretraining.

## 6. DEV experiment

The same DEV subset, matrix and pooled metric as the improvement plan
([yieldsat-improvement.md](./yieldsat-improvement.md)), 94 fine-tuning runs per arm:

| Arm | Pretraining | Fine-tuning |
|---|---|---|
| A0 | none (existing `before_full` point results, seed 0) | supervised |
| A2 | SSL | from the A2 unit checkpoint |
| **A3** | **SSL + knowledge** | from the A3 unit checkpoint |
| A4 | SSL + knowledge, `shuffled` | from the A4 checkpoint |
| A5 | SSL + knowledge, `notext` | from the A5 checkpoint |
| A6 | `ssl_long` | from the A6 checkpoint |

- **Label efficiency** (criterion E3): A0, A2 and A3 fine-tuned with 10% and
  25% of training seasons on the DEV CV10 folds.
- **Cluster:** a two-stage plan.
  1. Pretraining units are run as plan entries that keep their sensor
     checkpoints in the results root (like the image donors).
  2. Fine-tuning runs reference them via `--init_sensor_ckpt`. Pools wait
     for the checkpoints to exist (the existing donor gating).

## 7. Tasks

| ID | Deliverable | Acceptance |
|---|---|---|
| PK-01 | Library review record (K1) | 6 rules + 12 concepts `approved`, reviewer and date; validator passes |
| PK-02 | `yieldsat_build_knowledge.py`: raw indices + gate contexts per cache row | Unit tests on synthetic rows (units, intervals, abstentions); built for all countries |
| PK-03 | `KnowledgeReference` (fold-fitted percentile tables) and `KnowledgePointPretrainer` (SSL + grounding + relation, per-field-season averaging, controls) | Unit tests: gradients reach every stream; masked unknowns; gate semantics; shuffled derangement |
| PK-04 | Text references: pinned CLIP text model (`text_cache.py`), frozen vectors | Manifest with model revision and hashes |
| PK-05 | `yieldsat_pretrain_units.py` and `check_unit` | Audit passes for every DEV unit |
| PK-06 | Entry point: `main_yieldsat_finetune.py --mode pretrain --knowledge …`, then fine-tuning with `--init_sensor_ckpt` | Local smoke: pretrain → transfer → fine-tune |
| PK-07 | Diagnostics for the success criteria (`yieldsat_pretrain_diagnostics.py`) | Criteria I1–I5 computed on a pretrained checkpoint |
| PK-08 | Cluster suites (pretraining units + fine-tuning arms), DEV run, `results/pretrain_dev.md` | Success criteria evaluated and logged |

## 8. Open points

- The non-stress temperature limits and the organic-soil limit are informed
  defaults from crop physiology and soil classification; the project lead
  may revise them.
- SoilGrids units in the YieldSAT cache are verified during PK-02 before
  limits apply (g/kg vs dg/kg).
- `ys_r05` stays abstaining until a within-field thermal source exists.

## 9. Progress log

- 2026-10-04 — Specified.
