# Tech Stack & Environment

## Framework

Plain PyTorch. This is a deliberate and load-bearing choice: several components were
adapted from source projects built on `mmengine` / `mmsegmentation` registries
(AgriFM, SpectralGPT, geoai) and were **rewritten as plain `nn.Module`s** so the repo
carries no OpenMMLab dependency. Preserve this — reintroducing a registry framework
would fork the codebase's idioms.

No Lightning, no Hydra. Configuration is argparse plus inline dicts in each entry
point's `build_model()`.

## Declared dependencies — `requirements.txt`

```
torch == 1.13.0          ← YANKED from PyPI, install fails
torchvision == 0.14.0    ← YANKED from PyPI, install fails
timm == 0.5.4
numpy == 1.24.4
pandas == 2.0.3
h5py == 3.9.0
einops == 0.6.1
Pillow == 10.0.0
argparse == 1.4.0        ← stdlib since Python 3.2; remove
tqdm == 4.65.0
scikit-learn == 1.3.0
tensorboard == 2.13.0
rasterio == 1.4.4        ← added by this project (field registration)
requests == 2.31.0       ← added by this project
```

Everything is `==`-pinned. `rasterio` and `requests` were added for System B; the
rest is inherited from upstream MMST-ViT.

### The install is broken

`pip install -r requirements.txt` exits 1. `torch==1.13.0` and `torchvision==0.14.0`
have been yanked from PyPI, and the resolver aborts:

```
ERROR: Ignored the following yanked versions: 0.1.6, ... 0.15.0
```

Both attempted install commands failed in the `.primae` executability probe. This is
blocker #1 for every other piece of work.

### Recommended resolution

Relax the two yanked pins and let the rest float within compatible ranges:

```
torch >= 2.1
torchvision >= 0.16
timm >= 0.9
numpy >= 1.24, < 2.0     # rasterio 1.4.x and older h5py wheels are happier here
pandas >= 2.0
h5py >= 3.9
einops >= 0.6
Pillow >= 10.0
tqdm >= 4.65
scikit-learn >= 1.3
tensorboard >= 2.13
rasterio >= 1.3
requests >= 2.31
```

and drop `argparse` entirely. Install torch from the PyTorch index for a specific
CUDA build rather than pinning it here:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Two API changes to check when moving to torch ≥ 2.1, both of which touch code this
project actually uses:

- `torch.cuda.amp.autocast` / `GradScaler` are deprecated in favour of
  `torch.amp.*` with an explicit device string. `util/misc.py`'s `NativeScaler` uses
  the old form.
- `nn.MultiheadAttention` is stricter about mask dtypes and will now warn on a float
  `key_padding_mask`. That warning is exactly defect **D1** in
  [status.md](./status.md) — treat it as a free diagnostic, not noise to silence.

## Compute

### Training
- **CUDA GPU is currently mandatory.** Both System B entry points default to
  `--device cuda` and allocate directly on the device; there is no working CPU path
  despite the flag existing. Adding one is Phase 0 work and is what makes the model
  debuggable at all.
- DDP is supported (`util/misc.py::init_distributed_mode`,
  `find_unused_parameters=True` — necessary because modality dropout leaves encoders
  unused on some steps).
- AMP via `NativeScalerWithGradNormCount`.

### Memory budget

The default config is heavier than it looks. At `--input_size 64`:

| modality | tokens |
|---|---|
| `dem` | 64 × 64 = 4,096 |
| `sar` | 4,096 |
| `crop` | 4,096 |
| `weather` | 1 |
| `soil` | 1 |
| **total per sample** | **12,290** |

Cross-attention is `num_latents=64` queries against 12,290 keys — linear in tokens,
so tractable. But `main_pretrain_multimodal.py` invokes the fusion stack ~8–12 times
per step (leave-one-out cross-modal prediction plus temporal forecasting), so
pretraining costs roughly an order of magnitude more per step than fine-tuning.
Budget accordingly, and see roadmap Phase 4 for the fix.

### Inference
Batch, offline. Not designed for low latency, and the mission explicitly excludes it.

## Environment as tested

The review environment had **Python 3.14.7 with no ML dependencies installed** — no
`torch`, `timm`, `rasterio`, or `h5py`. All findings in [status.md](./status.md) are
therefore from source reading, not execution.

Python 3.14 is also ahead of what several pinned dependencies publish wheels for.
**Target Python 3.10–3.12** for a reproducible environment.

## Missing infrastructure

| | present? | note |
|---|---|---|
| `pyproject.toml` / `setup.py` | ❌ | not installable as a package; imports rely on cwd being the repo root |
| `tests/`, `conftest.py`, `pytest.ini` | ❌ | 36 `__main__` demo blocks are the de facto suite |
| CI (`.github/workflows`) | ❌ | |
| `Dockerfile` / `environment.yml` | ❌ | the most direct fix for the whole dependency problem |
| `.gitignore` | ❌ | `__pycache__/` directories are present in the tree |
| lockfile | ❌ | `==` pins are not a lockfile — no transitive dependency pinning |
| linter / formatter config | ❌ | style is consistent by hand, which will not survive more contributors |

## Repository size

~9,700 lines of Python, 44 modules. 74 commits total: 26 inherited from upstream
MMST-ViT (through 2024-03-26), 48 new (2026-09-17 → 2026-09-18).
