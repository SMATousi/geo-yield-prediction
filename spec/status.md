# Status — Verified Review

Reviewed at commit `c8be363`, 2026-09-23. Method: full source read of System B plus
static import-graph analysis. **No code was executed** — `torch` is not installed in
this environment (Python 3.14, no torch/timm/rasterio/h5py), so dynamic verification
was not possible. Findings below are from source reading and are marked accordingly.

---

## Headline

> The architecture is real. The pipeline is not.

Every capability in `.primae/COMPLETENESS_TODO.md` has genuine code behind it — this
is not scaffolding or stubs. The modality encoders, Perceiver fusion, head registry,
five-objective pretraining framework, and the `rasterio` field-registration loaders
are all substantively implemented and readable.

What does not exist is a working system. The two flagship entry points train on
random noise; the real data loaders they were written for have no callers; there are
no tests; and no command in the repository has ever been observed to succeed.

The `.primae` receipts are consistent with this: `completeness.json` reports **83%
requirement coverage** while `executability.json` reports **`executable: false`, 0 of
14 commands ran, 14 failed**. Those two numbers are not in tension — the first
measures presence of code, the second measures whether any of it runs.

---

## What is genuinely built and sound

| Area | Assessment |
|---|---|
| Per-modality encoder dispatch | **Solid.** Clean config-driven registry, sensible per-type tensor contracts, honest docstrings. |
| Perceiver latent fusion | **Solid in shape**, defective in masking (D1). Separable spatial/temporal pos-embeds plus concatenated modality identity is a reasonable, well-executed design. |
| Head registry | **Solid.** Validated vocabularies, 4 registered heads including an MDN head that gives real predictive uncertainty. Genuinely extensible as claimed. |
| Dense decoders (FCN/FPN/DPT) | **Solid**, with `reassemble`/`stitch` providing the model-output → georeferenced-raster path. |
| Field registration loaders | **The best code in the repo.** `field_roi` + `field_yield_dataset` do correct CRS-aware reprojection with per-layer resampling mode (nearest for categorical, bilinear for continuous) and honest nodata handling. |
| Metrics | **Good coverage** beyond RMSE/R²: spatial correlation, zone preservation, confusion matrix. Appropriate for dense spatial outputs. |
| Self-supervised framework | **Ambitious and coherent.** Five objectives, correctly composed into a weighted sum, with freeze/unfreeze support. |
| Commit hygiene | Each commit names its gap and its provenance (which upstream repo a pattern was adapted from). Unusually traceable. |

---

## Integration gaps

These are not bugs in any single file. They are things that were built and then never
connected.

### G1 — Training runs on synthetic noise (critical)

Both System B entry points generate their batches with `torch.randn`:

- `main_multimodal_finetune.py:185` `make_synthetic_batch()`, called at `:311` (train)
  and `:355` (eval). Targets are `torch.randn(b, 1, 8, 8)` — pure noise, `:201`.
- `main_pretrain_multimodal.py:130` `make_synthetic_batch()`, called at `:250`.

The loop is otherwise complete and correct: LR schedule, grad accumulation, AMP
scaler, checkpointing, DDP, metric logging. It is a well-built harness attached to no
data. The `evaluate()` function at `main_multimodal_finetune.py:349` prints
`RMSE / R_Squared / Corr` computed from noise-against-noise — numbers that look like
results and mean nothing. **Any metric this repo has produced so far is meaningless.**

### G2 — The real data pipeline is orphaned

Modules with **zero importers anywhere in the tree**:

```
dataset/heterogeneous_modality_loader.py    dataset/collate.py
dataset/field_pipeline.py                   dataset/block_retiler.py
dataset/soilgrids_loader.py                 dataset/mapping_dataset.py
models_multi_unified.py                     util/modality_robustness.py
util/spatial_viz.py                         models_mae_group_channels.py
```

`dataset/field_yield_dataset.py` is imported only by
`heterogeneous_modality_loader.py`, which itself has no callers — so the entire
field-data ingest chain is dead from the root.

This matters most for `util/modality_robustness.py`: `evaluate_modality_robustness`
is the function that would produce the ablation table the mission depends on, and
nothing calls it.

### G3 — No tests, no CI, no packaging

No `tests/`, no `test_*.py`, no `conftest.py`, no `pytest.ini`, no `pyproject.toml`,
no `setup.py`, no `.github/`, no `Dockerfile`, no `Makefile`.

Every module does carry an `if __name__ == "__main__"` demo block exercising its
shapes (36 of 44 modules). These are the de facto test suite and are genuinely
useful — they are the right raw material to convert into `pytest` cases, which makes
Phase 1 of the roadmap much cheaper than it would otherwise be.

### G4 — Nothing has ever run

`.primae/executability.json`: `executable: false`, 14 of 14 commands failed, 0 of 7
documented steps verified. Two independent blockers:

1. **Dependencies won't install.** `torch==1.13.0` / `torchvision==0.14.0` are yanked
   from PyPI; `pip install -r requirements.txt` exits 1.
2. **CPU is not a supported path.** Both System B scripts default to `--device cuda`
   and construct tensors directly on the device, so on a CPU-only box they fail at
   startup with `AssertionError: Torch not compiled with CUDA enabled`. The `--device`
   flag exists but `cpu` was never exercised.

Additionally `config/build_config_soybean.py` cannot run from the repo root as the
README instructs: paths are hardcoded `./../input/...` and `./../data/...` (`:7`,
`:16`, `:29`, `:38`), it writes with `open(path, "x")` which raises if the file
exists, and `./../data/` is never created.

### G5 — The V&V entry point has no battery

`primae_vandv_entry.py` delegates to `.primae/vandv/probe.py`. That directory does not
exist — `.primae/` contains only the three JSON receipts and the TODO. So `run()`
returns `{"ok": False, "reason": "no verification battery committed under
.primae/vandv", "checks": []}` unconditionally. To the module's credit it reports the
absence honestly rather than fabricating a pass.

### G6 — Documentation describes only the inherited system

`README.md` is unchanged from upstream: it documents Tiny-CropNet, county-level
prediction, and the two MMST-ViT scripts. It contains no mention of the multimodal
encoders, the fusion backbone, the head registry, the field-level loaders, or either
new entry point. A new reader would not learn that System B exists.

`RUNNING.md` is accurate and unusually candid — it states plainly that no step
succeeded and why. Keep that tone.

---

## Defects

Ordered by severity. All found by source reading; none executed.

### D1 — Attention masking is inert (high)

`models_latent_fusion.py:226` builds the key-padding mask as a **float** tensor:

```python
avail = torch.as_tensor(avail, device=emb.device).float()
...
m = (1 - avail).unsqueeze(1).expand(emb.shape[0], emb.shape[1])
```

and passes it to `nn.MultiheadAttention` at `:280` and `:326`.

PyTorch treats a float `key_padding_mask` as an **additive bias on the attention
logits**, and a bool mask as exclusion. So absent modalities receive `+1.0` added to
their attention scores — they are not masked out, they are *slightly up-weighted*.
The missing-modality masking that the whole robustness story rests on does not do
what its docstring says.

**Fix:** build the mask as `bool` (`True` = ignore), i.e. `m = (avail < 0.5)`.

### D2 — All-modalities-absent produces NaN (high)

If every modality is masked for a given sample, that row of `key_padding_mask` is
entirely "ignore", the attention softmax has no valid keys, and the output is `NaN`.

This is reachable. `main_multimodal_finetune.py:320` drops each of 5 modalities
independently with `p = args.modality_dropout = 0.3`, so P(all five dropped) =
`0.3⁵ ≈ 0.24%` per sample. At batch 4 × 20 steps × 50 epochs that is ~4,000 sample
draws per run — the event is near-certain over a full run. The loop then hits the
`math.isfinite` guard at `:327` and calls `sys.exit(1)`, killing training.

Note this is currently *latent*, because D1 means the mask is additive and no row is
ever fully excluded. **Fixing D1 will expose D2.** Fix both together: guarantee at
least one modality survives dropout, or force-unmask a row whose mask is all-True.

### D3 — Positional-embedding mismatch is silently swallowed (medium)

`models_latent_fusion.py:268` (and `:318` in `forward_multiscale`):

```python
if pos.shape[1] != emb.shape[1]:
    pos = pos[:, :emb.shape[1], :]
```

This was written for the legitimate case of a 1-token missing-modality embedding. But
it also silently truncates whenever a declared `modalities[name]['spatial'] ×
['temporal']` layout disagrees with what the encoder really emits — a class of bug
that should be loud. And it only truncates: if the encoder emits *more* tokens than
declared, `pos` is too short and the subsequent `emb + pos` raises a broadcast error
far from the cause.

**Fix:** special-case `emb.shape[1] == 1` explicitly; assert equality otherwise.

### D4 — `TimeSeriesEncoder` has no temporal modelling (medium)

`models_multimodal_encoder.py:156–174`: a per-timestep MLP followed by
`h.mean(dim=1)`. Mean-pooling is permutation-invariant, so a shuffled growing season
yields an identical embedding. Phenology — the thing weather-driven yield modelling
is *about* — cannot be represented.

The mission requires "time-series encoders" for weather/precipitation/soil-moisture;
this satisfies the label, not the requirement. Compare `MultitemporalSAREncoder`
(`:129`), which does apply `Conv3d` over time before pooling and is the right model
for what this should be.

**Fix:** 1D temporal conv, GRU, or a small temporal transformer, and emit `T` tokens
rather than 1 so the fusion backbone's temporal positional embedding has something to
act on. Note the fusion config already declares `weather: {'spatial': 1, 'temporal': 1}`
— that `temporal: 1` is a direct consequence of this collapse.

### D5 — Per-modality normalisation never executes (medium)

`normalize_modality()` is guarded by `isinstance(x, np.ndarray)` at
`models_multimodal_encoder.py:293`, `:309`, `:378`, `:394`. Every live caller passes a
`torch.Tensor`, so normalisation is skipped on every path the training scripts use.
The `util/norm_stats.py` registry — real per-sensor S2/S1a/S1d/Landsat statistics —
is imported and never reached.

Feeding unnormalised SAR backscatter (dB, typically −25..0) alongside a DEM (metres,
0..3000) into a shared latent space will not train well.

**Fix:** normalise tensors too, or move normalisation into the dataset where it
belongs and drop it from the model.

### D6 — Lazy `nn.Parameter` creation inside `forward` (medium)

`models_multimodal_encoder.py:92` sets `self.pos_embedding = None`; `:122` creates it
as an `nn.Parameter` on the **first forward pass** via `_init_pos_embedding`.

Consequences: the parameter is absent when the optimizer is constructed, so it is
never trained; it is missing from `state_dict()` until a forward has run, so
checkpoint save/load is order-dependent; and it breaks DDP, which requires a fixed
parameter set at wrap time.

Currently latent — `PatchEmbeddingEncoder` is registered as type `patch` but no config
in the repo uses it. It will bite the first person who does.

**Fix:** require `num_patches` at construction, or use a non-parameter buffer.

### D7 — Non-square `num_latents` crashes the reshape (low)

`main_multimodal_finetune.py:174–181`:

```python
grid = int(round(num_latents ** 0.5))
if grid * grid != num_latents:
    pad = grid * grid - num_latents
    latents = torch.cat([latents, latents.new_zeros(b, pad, d)], dim=1)
```

`round()` can round *down* (e.g. `num_latents=65` → `grid=8` → `pad=-1`), and
`new_zeros(b, -1, d)` raises. Works for the default `64`; breaks on most other values
of `--num_latents`.

**Fix:** `grid = math.ceil(num_latents ** 0.5)`.

### D8 — `forward` / `forward_multiscale` duplication (low)

`models_latent_fusion.py:237–285` and `:287–335` share ~40 lines of token assembly
verbatim. D1 and D3 each had to be noted in both places; the next fix will too.

**Fix:** extract a `_assemble_tokens(embeddings, ids_keep)` helper.

### D9 — Dead variable (trivial)

`models_latent_fusion.py:186`: `dpr = [drop_rate] * depth` is computed and never used
(a vestige of a stochastic-depth schedule that was not carried over).

---

## Requirement coverage, re-read

`.primae/completeness.json` reports 24/29 implemented, 0 partial, 0 missing, **5
undetermined**. The 5 undetermined were not judged as absent — the audit probe timed
out after 180 s on lane 4. They are:

| id | requirement | my read |
|---|---|---|
| `cap-b797f016` | Training Protocol | **Partial.** Loop scaffolding is complete and correct; it has no data. |
| `cap-5c9780e5` | Evaluation | **Partial.** Metrics and `evaluate_modality_robustness` exist; nothing calls them, and no split protocol is wired into System B. |
| `cap-145c78df` | Interpretability | **Partial.** `util/spatial_viz.py` (quantile classes, PCA pseudocolour) and the MDN head's uncertainty exist; both orphaned. |
| `cap-9066043a` | Model Output and Deliverables | **Partial.** `reassemble`/`stitch` give the raster-output path; no script emits a georeferenced product. |
| `cap-72da356e` | Native-structure encoding + shared-space fusion | **Implemented.** This is the strongest part of the repo. |

So the honest coverage is closer to **25 implemented / 4 partial / 0 missing** at the
level of *code presence* — and **0 verified** at the level of *working behaviour*.

---

## Housekeeping

A stray directory `root/yield-prediction-primae/` is untracked at the repo root. It
holds an earlier draft spec (`README.md`, `mission.md`, `tech_stack.md`,
`roadmap.md`) plus a duplicate `mission.md` under a nested copy of this repo's own
directory name — the signature of a previous session writing to a relative path
instead of an absolute one.

That draft's substance has been folded into this `spec/` directory, with its
inaccurate claims corrected (it stated "Total Commits: 63 (January 2024 – September
2026)"; the real figure is 48 new commits, all on 2026-09-17/18, on top of 26
inherited upstream commits — and it did not mention that training runs on synthetic
data). The stray tree is now redundant:

```bash
rm -rf root/          # from the repo root — untracked, nothing else lives there
```

Left in place for you to confirm rather than deleted, since it is the only copy of
that draft.
