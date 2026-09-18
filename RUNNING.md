# RUNNING

## Building and running this artifact

This repository implements a multimodal geospatial foundation model for
field-scale crop-yield prediction. It contains several entry-point scripts
(`main_*.py`) that are runnable from the repo root with optional argparse flags
(`--output_dir`, `--resume`, `--eval`, `--device`, etc.), plus a config builder
(`config/build_config_soybean.py`). The declared build system is
`requirements.txt` (pip).

### Installing dependencies

The dependency list is declared in `requirements.txt`. Installing it was
attempted but **failed** in this environment. The pip resolver rejected the
pinned `torch == 1.13.0` / `torchvision == 0.14.0` versions because the
corresponding releases have been yanked from PyPI, so the install aborts with
`ERROR: Ignored the following yanked versions: 0.1.6, ... 0.15.0` and exits 1.
Both of the following commands were attempted and both failed with that error:

```bash
pip install -r requirements.txt
```

```bash
pip3 install -r requirements.txt --target /tmp/primae-exec-xsqr_vzc
```

A reader who wants to build this must first resolve the yanked `torch` /
`torchvision` pins (for example by installing a non-yanked CUDA build of torch
from the PyTorch index, or relaxing the version pins) before the rest of the
requirements can be installed.

### Running the training / evaluation scripts

The primary scripts are `main_multimodal_finetune.py` (dense yield-map
fine-tuning of the multimodal transformer) and `main_pretrain_multimodal.py`
(self-supervised multimodal pretraining). Both default to `--device cuda` and
construct tensors directly on that device, so they require a CUDA-enabled build
of PyTorch. In this container the installed torch is **not compiled with CUDA**,
so both commands were attempted and both failed immediately with:

```
AssertionError: Torch not compiled with CUDA enabled
```

```bash
python main_multimodal_finetune.py
```

```bash
python main_pretrain_multimodal.py
```

The MMST-ViT scripts (`main_finetune_mmst_vit.py`, `main_pretrain_mmst_vit.py`)
and the config builder (`config/build_config_soybean.py`) were also attempted and
each failed with a `Traceback (most recent call last):` during startup (an
environment issue, before any training ran):

```bash
python main_finetune_mmst_vit.py
```

```bash
python main_pretrain_mmst_vit.py
```

```bash
python config/build_config_soybean.py
```

### What the artifact produces

When the scripts run successfully they write checkpoints, `log.txt` logs, and
TensorBoard event files under the `--output_dir` (defaults:
`./output_dir/mmst_multimodal` for fine-tuning and
`./output_dir/mmst_multimodal_pretrain` for pretraining). The fine-tuning script
also supports `--eval` for evaluation-only runs and `--resume` to continue from a
checkpoint. None of these outputs were produced in this container because every
command above failed before training began.

### Summary of the current state

No runnable step succeeded in this environment. The blocker is twofold: (1) the
pinned `torch`/`torchvision` versions in `requirements.txt` are yanked and cannot
be installed, and (2) the installed PyTorch lacks CUDA while the scripts default
to `--device cuda`. Building and running this artifact requires a working,
CUDA-enabled PyTorch installation and a resolvable dependency set.
