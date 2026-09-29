# Roadmap

Ordered by **dependency**, not by ambition. Each phase unblocks the next; doing them
out of order mostly wastes effort.

The governing principle: **this project's bottleneck is not model capability, it is
that nothing runs on real data.** Every phase before 4 exists to fix that. Resist
adding architecture until Phase 3 produces a number you believe.

---

## Phase 0 — Make it run at all

**Blocks everything.** Nothing else in this roadmap is verifiable until this is done.

- [x] Unpin the yanked `torch==1.13.0` / `torchvision==0.14.0`; move to `torch>=2.1`.
      Drop the `argparse` dependency (stdlib). See [tech_stack.md](./tech_stack.md).
- [x] Verify `pip install -r requirements.txt` succeeds in a fresh Conda environment.
- [x] **Add a working `--device cpu` path.** Both System B scripts allocate directly
      on the device; make the synthetic-batch helpers and model construction
      device-agnostic. Without this the model cannot be debugged or tested without a
      GPU, which is what makes Phase 1 possible.
- [x] Fix `config/build_config_soybean.py`: resolve paths relative to the module
      (`Path(__file__).parent`), create `data/` if absent, replace `open(path,"x")`
      with an explicit overwrite guard.
- [x] Add an `environment.yml` pinning Python 3.10–3.12 plus a
      CUDA-matched torch build. This is the highest-leverage single artefact in the
      phase — it makes the environment reproducible instead of described.
- [x] Add `.gitignore` (`__pycache__/`, `output_dir/`, `*.pth`).

**Exit:** `python main_multimodal_finetune.py --device cpu --epochs 1` completes on a
clean container.

**Phase 0 verification (2026-09-23):** The exit command completed in a new Python
3.11 Conda environment. The GPU was also verified on an RTX 3090. These runs use
synthetic batches; real-data training remains Phase 3 work. See [RUNNING.md](../RUNNING.md).

---

## Phase 1 — Lock the contracts with tests

Do this **before** fixing the defects in Phase 2, so the fixes are provably correct
and the regressions are caught.

- [x] Add `pytest` + `pyproject.toml`. Make the repo installable.
- [ ] **Convert the 36 `__main__` demo blocks into test cases.** They already
      exercise the right shape contracts; this is mechanical work with high payoff
      and is why Phase 1 is cheaper here than in most codebases.
- [x] Targeted tests for the contracts most likely to silently drift:
  - [x] every encoder type: input rank → output token count
  - [x] `forward_with_missing`: all-present / all-absent / mixed-batch, per modality
  - [x] **masking: assert a masked modality provably cannot influence the output** —
        perturb an absent modality's tensor, assert the fused latent is bit-identical.
        This test fails today and is the regression test for defect D1.
  - [x] `LatentFusionTransformer`: declared token layout vs. actual encoder output
  - [x] every head: output shape and loss finiteness
  - [x] `FieldYieldDataset` against the synthetic-GeoTIFF fixture at
        `field_yield_dataset.py:219`
  - [x] `collate_field_samples` with heterogeneous modality keys across the batch
- [x] CI running the suite on push.

**Progress (2026-09-23):** 37 tests pass. Five strict expected failures capture D1,
D3, D6, the `grouped_vit` output-rank mismatch, and the disconnected unified
container. Strict `xfail` keeps CI green while making an unexpected pass fail the
suite, so a fix requires updating the assertion. The 36 demo blocks have not all
been converted; targeted System B contract tests and several geospatial utility
tests are in place.

**Exit:** finish converting the useful demo contracts, with known defects explicitly
tracked until Phase 2 fixes them.

---

## Phase 2 — Fix the defects

Full detail and file:line for each in [status.md](./status.md).

**Correctness — do these together:**
- [x] **D1** — `key_padding_mask` is float, so PyTorch treats it as an *additive
      bias* and absent modalities are up-weighted rather than excluded. Build it as
      `bool`. This is the highest-severity finding in the review: the missing-modality
      masking currently does not work.
- [x] **D2** — a sample with every modality dropped yields an all-masked attention
      row → `NaN` → `sys.exit(1)`. Currently latent *because* D1 masks nothing;
      **fixing D1 exposes it.** Keep one learned missing token available to attention
      when a sample has no real modality.
- [x] **D5** — `normalize_modality` is gated on `isinstance(x, np.ndarray)` and never
      runs on the tensor path. Either normalise tensors, or move normalisation into
      the dataset and delete it from the model. Unnormalised SAR dB alongside DEM
      metres will not train.
- [x] **Loss masking for nodata.** `-9999.0` nodata cells currently enter the L1/MSE
      loss directly (see [data_contract.md](./data_contract.md) §4). This is not in
      the defect list because it is not a code bug — it is a missing feature that will
      silently wreck the first real training run. **Do not skip it.**

**Robustness:**
- [x] **D3** — replace the silent `pos` truncation with an explicit 1-token
      special case plus an assertion on layout mismatch.
- [x] **D6** — `PatchEmbeddingEncoder` creates an `nn.Parameter` inside `forward`;
      require `num_patches` at construction.
- [x] **D7** — `int(round(sqrt(n)))` can round down and produce a negative pad; use
      `math.ceil`.

**Cleanup:**
- [x] **D8** — extract the ~40 duplicated lines shared by `forward` and
      `forward_multiscale`.
- [x] **D9** — remove the unused `dpr` variable.

**Verification (2026-09-23):** The D1, D3, and D6 regression tests now pass;
all listed Phase 2 fixes are implemented. The full suite has 52 passing tests and
two strict expected failures for separate `grouped_vit` and unified-container
integration gaps. Both System B training scripts complete a small CPU smoke run.
The Phase 1 demo-block conversion remains open, and real-data training is Phase 3.

**Exit:** listed Phase 2 defects closed; Phase 1 masking test green.

---

## Phase 3 — Connect the real data pipeline

**This is the phase that changes what the project *is*.** Everything before it is
maintenance; everything after depends on it.

**Updated scope (2026-09-28):** Consume the downloader's catalogs, native grids,
daily tables, soil vectors and irregular imagery, plus external PlanetScope.
Pretraining uses unlabelled AOI tiles; fine-tuning uses the separate yield-labelled
field corpus. The old flat-file demo loaders do not implement this handoff.
[Full-layer integration](./layer_integration.md) is the authoritative requirement
and task list, including dependencies and acceptance checks.

- [ ] **LI-01–03:** Inspect the actual handoff, implement the source registry, and
      build lazy datasets over shared geographic footprints with separate native
      predictor grids. Resolve assets relative to `data_root` and read catalog
      geometry instead of requiring a boundary TIFF.
- [ ] **LI-04–06:** Add static/soil/CDL, daily weather/SMAP, and imagery adapters.
      Preserve depth/band order, sensor units, native S2 groups, SAR tracks, dates
      and ancillary quality masks. Report unavailable AOI extensions explicitly.
- [ ] **LI-07–08:** Implement train-only normalization and unit conversion; extend
      collation/encoders/fusion for variable sizes, physical token metadata, and
      pixel/time/padding masks. Do not resize all sources to one input grid.
- [ ] **LI-09–10:** Index co-occurrence and objective eligibility; enforce geographic
      splits and separated contrastive negatives. Audit all five losses for
      missing targets, invalid observations and temporal leakage.
- [ ] **LI-11:** Read exported yield raster/polygons/metadata, choose an explicit
      output grid, and apply field/target masks to loss and metrics in Mg/ha.
- [ ] **LI-12:** Wire both real-data entry points and explicit pretrained
      encoder/fusion checkpoint transfer. Keep synthetic input behind
      `--smoke-test` with a visible banner.
- [ ] **LI-13:** Run a documented real-data pilot with coverage/exclusion reports,
      CPU verification and spatially held-out yield metrics. A declared subset is
      a pilot, not evidence that all source adapters are compliant.

### Alternative Phase 3 branch — YieldSAT

The [YieldSAT data contract](./yieldsat_data_contract.md) adds selectable
`yieldsat_preprocessed_v1` alongside `downloader_native_layers_v1`.
It covers Argentina, Brazil, Germany and Uruguay with explicit country selection,
per-file channel/date/category decoding and country-qualified split identities.
Its already merged grids use separate requirements; this is not LI completion.

- [x] **YS-01–03** (2026-09-28): Whole-file CRC32 matches the recorded archive
      CRC for all four files. Full-corpus value audit done. Semantics recovered:
      target is dry t/ha, verified cell by cell against raw masks; weather is
      inclusive-interval sums. Grids come from `Raw.zip`: 1,059 physical fields and
      146 cross-farm aliases. Grouped farm-cluster/field/block/country/year splits
      carry leakage checks. Slope/TWI/soil-uncertainty/coordinate semantics remain
      open.
- [x] **YS-04–06** (2026-09-28): Cache and direct-HDF5 backends (identical
      output), per-feature/time/target masks, first-slot weather invalidation,
      cutoff modes, train-only field-sum normalization. One encoder per stream
      (`masked_temporal` ×2, `masked_static` ×2, `soil_profile`), Perceiver
      fusion, registered `scalar_cell_yield` head, checked sensor-checkpoint
      transfer.
- [x] **YS-07–08** (2026-09-28): Objective routing recorded and enforced
      (spatial MAE inactive in point mode; masked-observation and last-valid
      forecast variants; knowledge blocked). `main_yieldsat_finetune.py
      --data_contract yieldsat_preprocessed_v1` runs GPU pilots with I/O/memory
      reports and labelled per-country/per-crop/field-balanced metrics.
- [x] **YS-09** (point outputs, 2026-09-28): Test predictions are scattered to
      each field's verified grid and written as georeferenced GeoTIFFs; holes stay
      NaN, and bounds/duplicates raise. Patch mode is **not pursued** in this
      iteration.

YieldSAT pilot acceptance and limitations are in the YieldSAT contract. Retain the
original downloader/native-grid exit criteria below independently.

**Exit:** real pretraining and supervised fine-tuning run through the intended
handoff, with an honest spatially held-out yield RMSE. Full-layer compliance also
requires adapter/contract coverage for every listed source. Missing upstream AOI
assets remain external dependencies, not permission to fabricate samples.

---

## Phase 4 — Make pretraining affordable

Only worth doing once Phase 3 gives real unlabelled fields to pretrain on.

- [ ] **LI-14:** Profile the full-layer pipeline and measure transfer on identical
      spatial splits/budgets, including modality subsets and validation-tuned loss
      weights. Profile actual token counts; the five-modality prototype's cost is
      not an estimate for the expanded inventory.

- [ ] **LI-15:** Implement and evaluate the optional
      [expert-validated relationship pretraining v2](./knowledge_pretraining.md),
      following **KP-01–KP-08**. Ground sensor concepts using frozen text references
      and accepted annotations; apply reviewed soft relationship scorers to both
      observed endpoints. Pilot and validate an optional offline VLM/LLM applicability
      estimator with abstention and calibration. Compare grounding-only, relationship,
      shuffled/no-text and teacher controls; preserve sensor-only transfer/inference.
      This supersedes the earlier text-conditioned held-out prediction proposal.

- [~] **YS-10:** Evaluate YieldSAT scratch/pretrained transfer on matching grouped
      splits, with point/patch modes distinguished. Knowledge evaluation follows
      YS-07 and KP-01–KP-08 with country-specific tags, units and evidence policies.
      *2026-09-28:* a first single-seed point-mode comparison (yield-free
      pretraining on pooled training farms → Uruguay fine-tuning at 10%/100% label
      budgets) is in §7 of the contract. Multi-seed runs, patch mode and knowledge
      controls remain open (knowledge is blocked on KP tasks).

- [ ] **YS-11:** Improve the fusion: feed the Perceiver per-slot optical/weather
      and per-depth soil tokens with per-token masks and date-aware positions,
      repeated cross-attention and pre-norm latents, and ablate it against
      concat-MLP, token-transformer and the current summary Perceiver. Specified in
      [yieldsat_data_contract.md §8](./yieldsat_data_contract.md#8-model-improvement-tasks).

- [ ] **PC-01–PC-08:** Paper-compatible evaluation (per country–crop CV10, LORO,
      LOYO; S2 vs S2+ADM; LSTM protocol check), specified in
      [yieldsat_paper_comparison.md](./yieldsat_paper_comparison.md).

- [ ] Profile the five objectives. `_cross_modal_prediction` runs the fusion stack
      once per modality and `_temporal_forecast` once per temporal modality — roughly
      8–12 fusion invocations per step at ~12k tokens each.
- [ ] Sample a subset of leave-one-out pairs per step instead of exhausting them, or
      share one fusion forward across objectives.
- [ ] Tune the loss weights. Five objectives summed with fixed weights is a lot of
      unexamined hyperparameter surface; ablate which actually help.
- [ ] **Demonstrate transfer:** fine-tuning from pretrained weights must beat training
      from scratch on the same split. Until that is measured, the pretraining
      framework is unjustified complexity. Use `set_encoder_trainable()` for the
      frozen / partial / full comparison.

**Exit:** a measured transfer benefit, or a decision to drop objectives that don't earn
their cost.

---

## Phase 5 — Evaluation the mission actually requires

- [ ] Wire `util/modality_robustness.py::evaluate_modality_robustness` into the eval
      path. It already computes every modality subset with full dense metrics and has
      **no callers** — it is the single highest-value orphan in the repo, because it
      produces the ablation table the mission depends on.
- [ ] Publish the ablation table: RMSE per modality subset on real data.
- [ ] Report spatial-split metrics as the headline. Report random-split numbers
      separately and label them optimistic.
- [ ] Report the spatial metrics beyond RMSE that `util/metrics.py` already provides —
      `SpatialCorrelation`, `ZonePreservation` — since within-field *pattern* is the
      point, and a model can win on RMSE while getting every zone backwards.
- [ ] Baselines, or the numbers are uninterpretable: field-mean predictor, single-
      modality models, and a common-grid-resampled version of this model (which is the
      direct test of the native-resolution claim).

**Exit:** a defensible results table.

---

## Phase 6 — Fix the temporal encoder

Deliberately placed after Phase 5 so the change can be **measured** rather than
assumed to help.

Phase 3 must already preserve actual timestamps, missing observations and forecast
cutoffs. This phase upgrades the order-sensitive temporal architecture; it does
not defer ingestion, masks, or prevention of future-information leakage.

- [ ] **D4** — `TimeSeriesEncoder` is an MLP followed by `mean(dim=1)`, which is
      permutation-invariant over time: a shuffled growing season produces an identical
      embedding. Phenology cannot be represented. Replace with a temporal conv, GRU,
      or small temporal transformer.
- [ ] Emit `T` tokens instead of 1 so the fusion backbone's temporal positional
      embedding has something to act on, and update the `modalities` layout from
      `weather: {'spatial': 1, 'temporal': 1}` accordingly.
- [ ] Measure against the Phase 5 baseline. If it doesn't help, that is a finding
      worth writing down.

---

## Phase 7 — Deliverables and interpretability

The four `undetermined` requirements in `.primae/COMPLETENESS_TODO.md` land here.

- [ ] A script that emits a **georeferenced yield-map GeoTIFF** for a field, using the
      existing `DenseYieldFCNHead.reassemble()` / `.stitch()` path. Today nothing
      produces a georeferenced product.
- [ ] Wire `util/spatial_viz.py` (quantile colour classes, PCA pseudocolour) — also
      currently orphaned.
- [ ] Surface the `MDNYieldHead` predictive uncertainty as a per-pixel confidence
      raster. Agronomic users need to know where the model is guessing.
- [ ] Management-zone output via `ZonePreservation`.
- [ ] Commit a V&V battery at `.primae/vandv/probe.py`; `primae_vandv_entry.py`
      already delegates to it and currently always returns `ok: False` because the
      directory does not exist.

---

## Phase 8 — Documentation and extensibility proof

- [ ] **Rewrite `README.md`.** It still documents only the upstream county-level
      system and does not mention System B at all — a new reader cannot discover that
      the field-scale model exists.
- [ ] Keep `RUNNING.md`'s candour when updating it. "No runnable step succeeded and
      here is exactly why" is a genuinely good artefact; don't replace it with
      optimism.
- [ ] Prove the head registry's extensibility claim by registering a genuinely new
      task head (crop stress or soil property) and training it on the frozen backbone
      without touching the encoders. The claim is currently architectural, not
      demonstrated.
- [ ] Wire `models_multi_unified.py::MultiUnifiedModel` — the multi-head container
      that embodies the foundation-model claim and has no callers.
- [ ] Decide the fate of the remaining orphans (`field_pipeline`, `block_retiler`,
      `soilgrids_loader`, `mapping_dataset`, `models_mae_group_channels`): wire them or
      delete them. Code that no one calls is code no one maintains.

---

## Sequencing summary

```
Phase 0  make it run          ──┐
Phase 1  tests                ──┤ prerequisites — weeks, not months
Phase 2  fix defects          ──┘
Phase 3  REAL DATA            ──── the milestone that matters
Phase 4  pretraining cost     ──┐
Phase 5  evaluation           ──┤ turns it into a result
Phase 6  temporal encoder     ──┘
Phase 7  deliverables         ──┐ turns it into a product
Phase 8  docs + extensibility ──┘
```

## Risks

| Risk | Mitigation |
|---|---|
| **Real-data integration is missing.** The downloader documents a field corpus and separate PlanetScope source; AOI-wide companion layers have additional upstream requirements. | Inspect actual catalogs/assets in LI-01, build native-grid adapters, and report missing sources instead of assuming full-layer coverage. |
| Yield-monitor data is noisy and needs agronomic cleaning the loaders do not perform. | Treat cleaning as an explicit upstream stage with its own validation; do not let raw monitor output reach `<field>_yield.tif`. |
| Architecture keeps growing while nothing is validated. 48 commits added ~9,700 lines with zero tests. | Freeze new architecture until Phase 3 produces a believable number. |
| The five undetermined `.primae` requirements get read as "done". | They were probe timeouts, not verdicts. [status.md](./status.md) re-reads all five as partial. |
| Whole-modality masks pass, but clouds, missing dates and padding are not handled by that test. | Extend pixel/time/token masks and audit every objective in LI-08–10. |
