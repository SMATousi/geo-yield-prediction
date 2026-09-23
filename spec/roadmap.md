# Roadmap

Ordered by **dependency**, not by ambition. Each phase unblocks the next; doing them
out of order mostly wastes effort.

The governing principle: **this project's bottleneck is not model capability, it is
that nothing runs on real data.** Every phase before 4 exists to fix that. Resist
adding architecture until Phase 3 produces a number you believe.

---

## Phase 0 — Make it run at all

**Blocks everything.** Nothing else in this roadmap is verifiable until this is done.

- [ ] Unpin the yanked `torch==1.13.0` / `torchvision==0.14.0`; move to `torch>=2.1`.
      Drop the `argparse` dependency (stdlib). See [tech_stack.md](./tech_stack.md).
- [ ] Verify `pip install -r requirements.txt` succeeds in a clean container.
- [ ] **Add a working `--device cpu` path.** Both System B scripts allocate directly
      on the device; make the synthetic-batch helpers and model construction
      device-agnostic. Without this the model cannot be debugged or tested without a
      GPU, which is what makes Phase 1 possible.
- [ ] Fix `config/build_config_soybean.py`: resolve paths relative to the module
      (`Path(__file__).parent`), create `data/` if absent, replace `open(path,"x")`
      with an explicit overwrite guard.
- [ ] Commit a `Dockerfile` or `environment.yml` pinning Python 3.10–3.12 plus a
      CUDA-matched torch build. This is the highest-leverage single artefact in the
      phase — it makes the environment reproducible instead of described.
- [ ] Add `.gitignore` (`__pycache__/`, `output_dir/`, `*.pth`).

**Exit:** `python main_multimodal_finetune.py --device cpu --epochs 1` completes on a
clean container.

---

## Phase 1 — Lock the contracts with tests

Do this **before** fixing the defects in Phase 2, so the fixes are provably correct
and the regressions are caught.

- [ ] Add `pytest` + `pyproject.toml`. Make the repo installable.
- [ ] **Convert the 36 `__main__` demo blocks into test cases.** They already
      exercise the right shape contracts; this is mechanical work with high payoff
      and is why Phase 1 is cheaper here than in most codebases.
- [ ] Targeted tests for the contracts most likely to silently drift:
  - [ ] every encoder type: input rank → output token count
  - [ ] `forward_with_missing`: all-present / all-absent / mixed-batch, per modality
  - [ ] **masking: assert a masked modality provably cannot influence the output** —
        perturb an absent modality's tensor, assert the fused latent is bit-identical.
        This test fails today and is the regression test for defect D1.
  - [ ] `LatentFusionTransformer`: declared token layout vs. actual encoder output
  - [ ] every head: output shape and loss finiteness
  - [ ] `FieldYieldDataset` against the synthetic-GeoTIFF fixture at
        `field_yield_dataset.py:219`
  - [ ] `collate_field_samples` with heterogeneous modality keys across the batch
- [ ] CI running the suite on push.

**Exit:** green suite; the D1 masking test is present and failing (red for the right
reason).

---

## Phase 2 — Fix the defects

Full detail and file:line for each in [status.md](./status.md).

**Correctness — do these together:**
- [ ] **D1** — `key_padding_mask` is float, so PyTorch treats it as an *additive
      bias* and absent modalities are up-weighted rather than excluded. Build it as
      `bool`. This is the highest-severity finding in the review: the missing-modality
      masking currently does not work.
- [ ] **D2** — a sample with every modality dropped yields an all-masked attention
      row → `NaN` → `sys.exit(1)`. Currently latent *because* D1 masks nothing;
      **fixing D1 exposes it.** Guarantee ≥1 surviving modality per sample.
- [ ] **D5** — `normalize_modality` is gated on `isinstance(x, np.ndarray)` and never
      runs on the tensor path. Either normalise tensors, or move normalisation into
      the dataset and delete it from the model. Unnormalised SAR dB alongside DEM
      metres will not train.
- [ ] **Loss masking for nodata.** `-9999.0` nodata cells currently enter the L1/MSE
      loss directly (see [data_contract.md](./data_contract.md) §4). This is not in
      the defect list because it is not a code bug — it is a missing feature that will
      silently wreck the first real training run. **Do not skip it.**

**Robustness:**
- [ ] **D3** — replace the silent `pos` truncation with an explicit 1-token
      special case plus an assertion on layout mismatch.
- [ ] **D6** — `PatchEmbeddingEncoder` creates an `nn.Parameter` inside `forward`;
      require `num_patches` at construction.
- [ ] **D7** — `int(round(sqrt(n)))` can round down and produce a negative pad; use
      `math.ceil`.

**Cleanup:**
- [ ] **D8** — extract the ~40 duplicated lines shared by `forward` and
      `forward_multiscale`.
- [ ] **D9** — remove the unused `dpr` variable.

**Exit:** the Phase 1 masking test goes green; all defects closed.

---

## Phase 3 — Connect the real data pipeline

**This is the phase that changes what the project *is*.** Everything before it is
maintenance; everything after depends on it.

- [ ] Replace `make_synthetic_batch` in `main_multimodal_finetune.py` with a real
      `DataLoader` over `FieldYieldDataset` / `HeterogeneousModalityLoader` +
      `collate_field_samples`. Keep the synthetic path behind an explicit
      `--smoke-test` flag, and **print a loud banner whenever it is active** so no
      synthetic number is ever mistaken for a result.
- [ ] Same for `main_pretrain_multimodal.py`.
- [ ] Fix loader scalability (**L1** in [data_contract.md](./data_contract.md)):
      registration is eager in `__init__`, so the whole dataset is reprojected into
      memory in the parent process and `num_workers` buys nothing. Make it lazy per
      `__getitem__` with an on-disk cache of registered arrays.
- [ ] Make `HeterogeneousModalityLoader` subclass `torch.utils.data.Dataset` (**L2**).
- [ ] **Add a weather/timeseries ingest path.** `TimeSeriesEncoder` is configured in
      both entry points but no loader emits a `timeseries` kind — weather is the one
      modality in the model with no route from disk.
- [ ] Wire `util/spatial_split.py` to field-level splits so the geographic-honesty
      requirement is enforced by code rather than by whoever writes `train.txt`.
- [ ] Assemble a real pilot dataset per [data_contract.md](./data_contract.md) —
      even 50 field-years is enough to make the pipeline honest.

**Exit:** a training run on real georeferenced fields producing an RMSE that means
something. **This is the project's first genuine milestone.**

---

## Phase 4 — Make pretraining affordable

Only worth doing once Phase 3 gives real unlabelled fields to pretrain on.

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
| **No real data exists yet.** Phase 3 is gated on assembling a field-year corpus with yield-monitor rasters; that is a data-acquisition problem this repo cannot solve on its own. | Start sourcing in parallel with Phase 0. Fifty field-years is enough to make the pipeline honest — do not wait for thousands. |
| Yield-monitor data is noisy and needs agronomic cleaning the loaders do not perform. | Treat cleaning as an explicit upstream stage with its own validation; do not let raw monitor output reach `<field>_yield.tif`. |
| Architecture keeps growing while nothing is validated. 48 commits added ~9,700 lines with zero tests. | Freeze new architecture until Phase 3 produces a believable number. |
| The five undetermined `.primae` requirements get read as "done". | They were probe timeouts, not verdicts. [status.md](./status.md) re-reads all five as partial. |
| Fixing D1 exposes D2 and training starts crashing with `NaN`. | Fix them in the same change; the Phase 1 test covers both. |
