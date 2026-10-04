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

Units (verified on the cache 2026-10-04, §9):
- S2: L2A digital numbers / 10,000 = reflectance.
- Weather: interval sums. Temperatures are Kelvin-day sums and precipitation
  is metre sums. A slot's interval is the **inclusive** span from the
  previous dated slot to this one (dt + 1 days; the data contract's
  provenance), so a daily mean = sum / (dt + 1). A 31-day slot measured
  8,937 K-days = 288 K/day. Each row's first dated slot has no interval and
  is excluded (point contract). Window statistics weight each interval by
  its day overlap with the window. They are unknown when the dated intervals
  cover < 50% of the window.
- Soil: SoilGrids mapped units. Clay/silt/sand are g/kg (they sum to
  ≈ 1,000). SOC is **dg/kg** (Germany 0–5 cm median 525 → 52.5 g/kg), so the
  builder divides by 10.
- Terrain: aspect in degrees. Slope's scale is unresolved (values
  1e3–1e4), so it is used only through a country percentile. TWI is
  missing for all of Germany and 67% of Argentina and is not used.
- Season: seeding → harvest day of the row. Mid-season = 30–80% and late
  season = 60–100% of that window.
- **Input cutoff:** estimators read only the slots the model receives, so no
  concept is grounded in data the model never sees.
  - The builder writes one file per cutoff, `concept_raw_c<days>.npz`. The
    entry point picks the file that matches the run (`estimator_cutoff`) and
    refuses a mismatch.
  - **c0** (slots ≤ harvest) matches `--cutoff_mode all_slots` / `harvest`,
    which the paper protocols and DEV use. All estimator windows end at
    harvest anyway.
  - **c30** matches the pre-harvest mode (`before_harvest`, 30 d). Late
    season is then 60% → cutoff.
  - Pretraining uses the same input settings as the fine-tuning it serves
    (DEV: `s2_adm` streams, `all_slots`), so sensor transfer is exact.

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
| `shuffled` | Concept→text assignments permuted (a fixed derangement) and relation directions recomputed from the shuffled prototypes; same losses and compute | role of the language prior (the text semantics) |
| `notext` | Learned, randomly initialized concept prototypes (trainable) instead of frozen CLIP vectors | role of the language prior |
| `ssl_long` | SSL only, with 1.55× the steps (knowledge steps measured 7.6 s vs 4.9 s per 150 steps on a 3090) | compute-matched |
| `random_targets` | Every field season receives the concept targets and rule gates of another season of the same country × crop (fixed derangement; rows matched by position); real text | **knowledge content**: same supervision, marginals and rule structure, no link to the season's inputs |

The smoke evidence for `random_targets` is in §9 and in the criteria's
revision log. With learnable projectors, a text shuffle is just as learnable
as the real assignment, so `shuffled` cannot test the knowledge content.

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
  fewer pretraining runs than fine-tuning runs: **46 units** (10 CV10 +
  9 LOYO years + 27 LORO regions) for the 94 DEV runs per arm.
  `splits/pretrain_units_dev_s0.json` maps each DEV fold to its unit.
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
| A7 | SSL + knowledge, `random_targets` | from the A7 checkpoint |

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

- **Corrupt terrain curvature in one Argentina field season** (found
  2026-10-04 from the phase-1 validation losses; decision for the project
  lead).
  - All 1,981 cells of `Argentina_DUP1_farm45_field596_corn_2020` have
    curvature 1e8–3e9 (every other cell: p1–p99 ±0.97).
  - When the season is in a training set, Argentina's curvature std becomes
    2.5e7, which silences the channel for every Argentina cell. Pretraining
    unit `cv_k00` measured mean 33,727 / std 2.5e7, vs 0.003 / 0.42 without
    the season.
  - When it is held out, its standardized inputs reach ~1e10. The phase-1
    unit `cv_k09` validation loss is 1.3e14.
  - Affected:
    - every pretraining unit, through normalization or validation;
    - ARG-C rows of the paper suite (`before_full`), where this season is a
      test or training season.
  - Not affected: the DEV fine-tuning rows, since ARG-W uses only wheat
    seasons and per-pair statistics.
  - Proposed fix: validity rule v+1, i.e. |curvature| > 1,000 is invalid in
    the dataset and the field statistics. Then recompute Argentina's field
    statistics and restart phase 1. This must not change the PVC statistics
    while phase-1 pods are still staging, or pools would mix normalizations.
  - Other extremes found by a full cache scan are plausible, not fixed:
    - S2 bands up to 70σ in Argentina (≈ 10k cells per band) and up to 239σ
      in Uruguay (B01, 46 cells);
    - soil pH up to 58σ in Brazil.

- **ARG-W LORO folds split one province under two spellings.** The source
  metadata spells some provinces two ways ("Buenos Aires" / "Buenos_Aires",
  "Santa Fe" / "Santa_Fe"). The existing paper LORO folds treat them as
  different regions: fold 0 holds out "Buenos Aires" while training on
  "Buenos_Aires" fields of the same province, and likewise folds 4/5.
  - Pretraining units merge the spellings (one unit per physical province),
    so pretraining never sees the held-out province.
  - The fine-tuning folds themselves are unchanged for now, to stay
    comparable with all existing results. Merging them changes ARG-W LORO
    from 6 to 4 folds. **Decision for the project lead.**
- **CLIP text vectors are anisotropic:** pairwise concept cosine 0.62–0.95
  (mean 0.77). Concept discrimination lives in the residual directions;
  centering the prototypes is an option if grounding saturates.
- **Relational loss with linear projectors** can be partly satisfied by a
  constant offset between stream projections (A7 alignment 0.79–1.00).
  Criterion I2 is therefore relative to A7. If A3 ≯ A7, a contrastive
  relational variant (applicable vs non-applicable pairs) is the next
  option.
- The non-stress temperature limits and the organic-soil limit are informed
  defaults from crop physiology and soil classification; the project lead
  may revise them.
- `ys_r05` stays abstaining until a within-field thermal source exists.

## 9. Progress log

- 2026-10-04 — Specified.
- 2026-10-04 — **PK-02 done.** `yieldsat_build_knowledge.py`:
  - per-row raw indices, from pure functions with unit tests (weather
    interval overlap and the first-slot exclusion, low-coverage unknowns, S2
    windows, soil depth weighting and units, within-field DEM rank,
    hemisphere-aware aspect, input cutoff);
  - built for all 4 countries locally in ~1 min:
    `<artifact_root>/knowledge/<Country>/concept_raw.npz` (format
    `yieldsat_concept_raw_v2`, cutoff 30 d) plus `concept_raw_summary.json`.
  - Argentina sanity values (medians): mid-season precipitation 200 mm,
    mean temperature 22.7 °C, max temperature 28.8 °C, NDVI rise 0.67, SOC
    0–30 cm 15.7 g/kg, field relief 4.0 m. Missing values < 2.5% for every
    index.
- 2026-10-04 — **PK-03 done.** `yieldsat_knowledge_point.py`:
  - `KnowledgeReference`: fold-train weighted percentile tables (each field
    season counts once), mid-rank ties, strata with < 20 seasons → unknown;
    serialized with each run;
  - the gates of §3;
  - `attach_knowledge`;
  - `KnowledgePointPretrainer`: SSL + grounding + relation on the per-stream
    summary embeddings, per-field-season averaging, controls
    shuffled/notext/ssl_long/random_targets.
  - Tests: gradients reach every stream encoder; unknown targets, closed
    gates, sub-threshold presence and absent streams contribute nothing;
    derangement; season averaging; serialization.
- 2026-10-04 — **PK-04 done.** CLIP text vectors:
  - `openai/clip-vit-base-patch32` at commit
    `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` (official repo; only
    `.bin` weights). Built with `yieldsat_knowledge/text_cache.py` in a
    throwaway CPU env with torch 2.14, because transformers refuses `.bin`
    loading under torch < 2.6 (CVE-2025-32434). The project images have
    torch 2.5.1.
  - Output: `<artifact_root>/knowledge/text_clip_b32/` (512-d,
    `vectors_hash` 2df53f2e…, `library_hash` 99681a5c…). The trainer
    verifies both hashes and refuses mock vectors (`load_text_vectors`).
- 2026-10-04 — **PK-05 done.** `yieldsat_pretrain_units.py`:
  - 46 unit manifests (`pretrain_unit_<unit>_s0`) and the fold → unit index;
  - every unit passes `check_unit` (no mapped fold test season and no
    held-out year/region in train/val; every season assigned exactly once);
  - a leak test covers both failure modes.
  - Units train on 1,584–2,018 seasons, excluding 1–410.
  - Found the ARG-W province-spelling issue (§8).
- 2026-10-04 — **PK-06 done.** `main_yieldsat_finetune.py --mode pretrain --knowledge --knowledge_control …`:
  - fits the reference on the unit's training rows, attaches targets, logs
    coverage and rule applicability, and runs a per-epoch pretraining
    validation loss (criterion P1);
  - saves `sensor_checkpoint.pth`, `knowledge_reference.json` and
    `pretrainer_heads.pth`;
  - `--skip_source_check` (cache backend) allows local runs without the
    source NetCDF.
  - Local smoke on unit `cv_k00` (all 4 countries, 300 steps):
    - coverage ≥ 89% of training seasons for every concept;
    - applicable seasons per active rule: r01 612, r02 806, r03 479,
      r04 1,206, r06 1,271; r05 0 by design;
    - transfer → fine-tune GER-R fold 0 loaded 153 tensors and completed.
    - All five pretraining variants (A2, A3, shuffled, notext, ssl_long) ran.
- 2026-10-04 — **Cutoff fix.** The paper protocols (DEV included) feed the
  model all slots (`MODELS['ours']`: `--cutoff_mode all_slots`), so the first
  build's fixed 30-day estimator cutoff under-read the late season.
  - Estimators are now built per cutoff (c0 and c30, §3) and matched to each
    run's input cutoff.
  - The PK-07 smoke below ran with c30 inputs (`before_harvest`), consistent
    with itself.
- 2026-10-04 — **PK-07 done.** `yieldsat_pretrain_diagnostics.py` computes
  P2, I1, I2 (with per-season values for pairing, and at initialization),
  I3 and I4 on the unit's excluded seasons. Smoke results (300 steps; not
  evidence about the method):

  | Arm | I1 AUROC (12 concepts) | I2 alignment (5 active rules) |
  |---|---|---|
  | A3 | 0.58–0.98 (11 of 12 ≥ 0.80; low-elevation 0.58) | 0.85–0.98 |
  | shuffled | 0.58–0.99 | 0.83–0.98 |
  | random_targets | 0.45–0.66, except fine texture 0.88 (shared 250 m soil across seasons of a ground) | 0.79–1.00 |

  - Consequences:
    - the A7 `random_targets` control was added;
    - the criteria were revised (I1/I2/E2 vs A7; new E2b for the language
      prior; P2 relative to initialization; see the criteria's revision log).
  - P2 passes for every stream under A2 and A3.
- 2026-10-04 — **PK-08 started: DEV phase 1** (`cluster/suites/pk_dev1.yaml`).
  - The cluster driver gained a two-stage plan:
    - `pretrain:` expands one run per (arm, unit) over all 4 countries. Each
      keeps `sensor_checkpoint.pth`, `knowledge_reference.json` and
      `pretrainer_heads.pth` in the results root.
    - Fine-tuning experiments with `init_from: <arm>` start from their fold's
      unit checkpoint (`splits/pretrain_units_dev_s0.json`) and wait for it
      (`_job_ready`).
    - Jobs are grouped per (pair, arm), so each fine-tuning job depends only
      on its own arm's units.
    - Staging copies every country of a job, plus the knowledge assets for
      knowledge runs.
  - Phase 1 arms: A2, A3, A6 (1.55× steps) and A7. Pretraining budget: 30 ×
    500 steps of 512 cells. Fine-tuning: the dev_r2/before_full budget with
    `--norm_pooling per_country`.
  - Plan: 560 runs in 71 jobs, ≈ 186 A10 GPU-hours (≈ 13 h on 16 GPUs).
  - Phase 1 decides E1, E2, E4 and E5 ("knowledge does its job" vs A7,
    "pretraining helps").
  - Phase 2 (A4 shuffled, A5 notext for E2b; label efficiency for E3) follows
    if phase 1 is healthy (P1–P4).
  - Local end-to-end test of the pool path:
    - a mini plan with one A3 pretraining run (Germany, 20 steps) and its
      gated fine-tuning run (GER-R fold 0) through `run_pool` with staging;
    - the pretraining checkpoint, reference and heads landed in the results
      root, then the fine-tuning job ran;
    - transfer loaded 153 tensors with no sensor key missing (only the
      crop-context fusion entries are skipped: fine-tuning a single crop
      turns the crop token off).
  - Staging copies the neighbourhood arrays only for runs that use
    `--neighbourhood`.
  - Pools (`cluster/nautilus/pk_dev1_pool_{a,b}.yaml`): 12 CPU / 40 Gi / A10.
    The node without the CephFS driver (`hcc-nrp-shor-c5825.unl.edu`) is
    excluded. Pool a (16 pods) starts first and takes the 16 pretraining
    jobs, then fine-tuning. Pool b (14 pods) is applied once checkpoints
    exist, so no pod idles waiting for them (utilization flag).
- 2026-10-04 — **PK-08 phase 1 launched.**
  - Prepared on the PVC with the results pod:
    - concept indices c0/c30 for all 4 countries;
    - the 46 unit manifests, audit passed with the real source fingerprint
      check;
    - the CLIP text vectors (copied, hash verified).
  - Plan `/data/YieldSAT/yieldsat_artifacts/cluster/pk_dev1`: 560 runs (184
    pretraining, 376 fine-tuning gated on unit checkpoints).
  - Pool a (16 × A10) submitted, results in
    `/data/YieldSAT/yieldsat_results/pk_dev1`. Pool b follows once
    checkpoints exist.
- 2026-10-04 — **Phase 1, first results (8 A2 units, ≈ 41 min per run at
  6 concurrent runs per A10).**
  - Most validation curves decrease smoothly, e.g. masked observation 0.33
    → 0.11 on `cv_k01`.
  - The noisy or exploding ones trace to the corrupt Argentina curvature
    season (§8).
  - Diagnostics failed for the SSL arms because pods staged the concept
    indices only for knowledge runs, and the I3 probes need them for every
    arm. Fixed in `stage_local` (764b43e) for new pods; A2/A6 diagnostics
    from pool a are backfilled after pretraining.
