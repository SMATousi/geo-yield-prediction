#!/bin/bash
# Entry point inside a Nautilus pod, run from a fresh clone of this repository
# (the job clones YIELDSAT_GIT_REF first, so code changes need no image
# rebuild): check the GPU, plan the suite if its plan does not exist yet, then
# run yieldsat_cluster.py with the given arguments.
#   env: YIELDSAT_SUITE (e.g. cluster/suites/smoke.yaml), YIELDSAT_PLAN_DIR,
#        YIELDSAT_SOURCE_ROOT, YIELDSAT_ARTIFACT_ROOT, WANDB_API_KEY, ...
set -euo pipefail
cd "$(dirname "$0")/../.."
echo "== host $(hostname) $(date -Is) code $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, 'gpu', torch.cuda.get_device_name(0))"
if [ -n "${YIELDSAT_SUITE:-}" ] && [ ! -f "${YIELDSAT_PLAN_DIR}/plan.json" ]; then
  # first pod plans; others wait for plan.json (fold manifests are deterministic)
  mkdir -p "$(dirname "$YIELDSAT_PLAN_DIR")"
  if mkdir "${YIELDSAT_PLAN_DIR}.lock" 2>/dev/null; then
    python yieldsat_cluster.py plan --suite "$YIELDSAT_SUITE" --out "$YIELDSAT_PLAN_DIR"
    rmdir "${YIELDSAT_PLAN_DIR}.lock"
  else
    until [ -f "${YIELDSAT_PLAN_DIR}/plan.json" ]; do sleep 10; done
  fi
fi
exec python yieldsat_cluster.py "$@"
