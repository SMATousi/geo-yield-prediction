# YieldSAT training image for the Nautilus (NRP) cluster.
#   docker build -t gitlab-registry.nrp-nautilus.io/smatous/yieldsat .
#   docker push gitlab-registry.nrp-nautilus.io/smatous/yieldsat
# Image suites use the tag image-v1 (adds transformers for DINOv3), built from
# the same file: docker build -t gitlab-registry.nrp-nautilus.io/smatous/yieldsat:image-v1 .
# Matches the geo-yield-phase0 environment: Python 3.11, PyTorch 2.5.1,
# CUDA 12.1, cuDNN 9. Data and the W&B key are provided at run time
# (spec/yieldsat_cluster_runs.md); nothing secret or data-bearing is baked in.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends rsync ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/geo-yield-prediction

# dependencies first (cached layer); torch/torchvision come from the base image
COPY requirements.txt .
RUN grep -vE '^\s*(#|$)|^torch(vision)?\b' requirements.txt > /tmp/requirements-image.txt \
    && pip install -r /tmp/requirements-image.txt \
    && python -c "import torch, h5py, rasterio, sklearn, wandb, yaml, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__); from transformers import DINOv3ViTModel"

# project code
COPY . .
RUN pip install --no-deps -e . \
    && python -m pytest -q tests/test_yieldsat.py tests/test_yieldsat_image_model.py -x -p no:cacheprovider

CMD ["python", "yieldsat_cluster.py", "--help"]
