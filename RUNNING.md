# Running the field-scale prototype

Phase 0 was verified on 2026-09-23 with Python 3.11, PyTorch 2.5.1, CUDA 12.1,
and an NVIDIA RTX 3090 (driver 595.84). The training entry points still use
synthetic tensors. Their losses and metrics are smoke-test results, not crop-yield
results.

## Environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate geo-yield-phase0
python -m pip install -r requirements.txt
python -m pip check
```

The Conda file pins Python, the CUDA-matched PyTorch build, NumPy, and the MKL
runtime required by this PyTorch package. It includes the other dependencies from
`requirements.txt`. The separate pip command checks that the declared dependency
list remains installable. For a local environment under the repository instead of
Conda's default environment directory, use
`conda env create --prefix .conda/phase0 -f environment.yml` and activate that
prefix.

## Phase 1 tests

Install the project in editable mode, then run the contract suite:

```bash
python -m pip install -e .
python -m pytest
```

As of 2026-09-23, 52 tests pass and two strict expected failures remain for the
`grouped_vit` and unified-container integration gaps. The tests use small synthetic
tensors and temporary GeoTIFF files; they do not validate crop-yield accuracy.
GitHub Actions runs the CPU suite on pushes and pull requests. Both entry points
also completed a small CPU smoke run with the Phase 2 changes.
The CUDA smoke run could not be repeated in this session because no NVIDIA device
is exposed (`torch.cuda.device_count() == 0`).

## Smoke runs

```bash
python main_multimodal_finetune.py --device cpu --epochs 1
python main_pretrain_multimodal.py --device cpu --epochs 1 \
  --input_size 8 --batch_size 1 --embed_dim 48 \
  --modality_embed 16 --num_heads 3 --depth 1 --output_dir ''
```

Both commands completed in the Phase 0 environment. A small GPU fine-tuning run
also completed using `--device cuda` and the reduced settings above. PyTorch
reported `torch.cuda.is_available() == True` and ran a tensor operation on the
RTX 3090. The default fine-tuning command writes a checkpoint and log under
`output_dir/mmst_multimodal/`.

`config/build_config_soybean.py` now reads `input/` relative to the repository,
creates `data/` if needed, and refuses to replace existing JSON unless invoked
with `--overwrite`.

The field-level GeoTIFF loaders are still disconnected from these training scripts.
See `spec/roadmap.md` for the remaining integration and correctness work.
