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

## YieldSAT preprocessed branch (`yieldsat_preprocessed_v1`)

Implemented 2026-09-28; see `spec/yieldsat_data_contract.md` §7 for evidence and
results. Source NetCDF files are only read. All derived artifacts go under a
separate writable root (≈2 KB per row for the optional cache, ~25 GB for all four
countries on local disk):

```bash
export YIELDSAT_SOURCE_ROOT=/home1/pupil/SMATousi/YieldSAT/Preprocessed
export YIELDSAT_ARTIFACT_ROOT=/root/yieldsat_artifacts
C="Argentina Brazil Germany Uruguay"

python yieldsat_prepare.py index    --countries $C       # YS-01/03 row index, audit (~20 s)
python yieldsat_prepare.py geometry --countries $C \
    --raw_zip /home1/pupil/SMATousi/YieldSAT/Raw/Raw.zip  # YS-02 grids, centroids, cell checks
python yieldsat_prepare.py cache    --countries Germany  # YS-04 audit pass + cache + field stats
python yieldsat_prepare.py snapshot --countries Germany  # YS-01 integrity (add --full_crc to recompute)
python yieldsat_prepare.py semantics --countries Germany # YS-02 unit/aggregation evidence
python yieldsat_prepare.py splits   --countries Germany --scheme farm --name germany_farm_s0
```

`cache` reads each file once (~120 MB/s aggregate over the NFS mount; Argentina
≈63 GB). Run one process per country in parallel. Split schemes: `farm` (farm
clusters, including aliased farms), `field` (geometry-derived physical fields),
`block` (`--block_km` geographic blocks), `country` (`--holdout_country`), `year`
(`--holdout_year`).

Training and evaluation (point mode, one cell history → t/ha):

```bash
python main_yieldsat_finetune.py --data_contract yieldsat_preprocessed_v1 \
    --countries Germany --split germany_farm_s0 --device cuda \
    --output_dir output_dir/yieldsat/germany_farm
# yield-free pretraining on the training partition, then transfer:
python main_yieldsat_finetune.py --data_contract yieldsat_preprocessed_v1 --mode pretrain \
    --countries Argentina Brazil Germany Uruguay --split pooled_farm_s0 --output_dir output_dir/yieldsat/pt
python main_yieldsat_finetune.py --data_contract yieldsat_preprocessed_v1 --countries Uruguay \
    --split pooled_farm_s0 --init_sensor_ckpt output_dir/yieldsat/pt/sensor_checkpoint.pth
```

Useful flags: `--backend h5` reads the NetCDF directly with contiguous-block
sampling (no cache; about 60× slower per batch over NFS). `--cutoff_mode
{before_harvest,after_seeding,harvest,all_slots}` and `--cutoff_days` set the
prediction cutoff (the last two are labelled retrospective). The ablation flags
are `--streams`, `--soil_uncertainty ancillary`, `--aspect_encoding cyclic`,
`--no_crop_context`, `--norm_pooling per_country` and `--train_fraction`.
`report.json` records the split label, cutoff policy, excluded sources, objective
routing, I/O rates, memory and metrics: pixel, field-balanced and field-level,
per country, per crop, and macro-country. Tests: `python -m pytest
tests/test_yieldsat.py` (synthetic NetCDF layout; no real data needed).

Versioned results (parameters, per-epoch history, per-group test metrics, sample
maps) for the 2026-09-28 runs are in `results/yieldsat/`. Rebuild that directory
after new runs with
`python yieldsat_collect_results.py --runs_dir $YIELDSAT_ARTIFACT_ROOT/runs --splits_dir $YIELDSAT_ARTIFACT_ROOT/splits`.

### YieldSAT paper-protocol comparison

The paper's protocols are specified in `spec/yieldsat_paper_comparison.md`: one
model per country–crop pair; CV10 / LORO / LOYO; S2 vs S2+ADM; fold mean ± std.

```bash
# fold manifests (created automatically by the runner if missing)
python yieldsat_prepare.py folds --countries Germany --pairs GER-R --protocols cv loro loyo
# run matrix (resumable): our model and/or the paper's LSTM baseline
python yieldsat_paper_runs.py --pairs GER-R ARG-S --protocols cv --models ours paper_lstm \
    --inputs s2 s2_adm --fusion perceiver_summary --concurrency 3
# long training under a separate tag, extra flags after "--"
python yieldsat_paper_runs.py --protocols cv loro loyo --models ours --epochs 60 --tag e60 -- --lr 5e-4
# leakage-safe sensitivity (PC-08)
python yieldsat_paper_runs.py --pairs GER-R --protocols cv --group physical --policy strict
# side-by-side tables against the paper
python yieldsat_paper_compare.py --runs_root $YIELDSAT_ARTIFACT_ROOT/runs
```

### YieldSAT on a GPU cluster

See `spec/yieldsat_cluster_runs.md`. In short:

```bash
# once, with YIELDSAT_* pointing at the shared volume:
python yieldsat_cluster.py plan --suite cluster/suites/before_full.yaml \
    --out $YIELDSAT_ARTIFACT_ROOT/cluster/before_full          # prints the runtime estimate
# per pod (Kubernetes Indexed Job sets JOB_COMPLETION_INDEX; WANDB_API_KEY from the job YAML):
python yieldsat_cluster.py run --plan $YIELDSAT_ARTIFACT_ROOT/cluster/before_full \
    --local_root /scratch/yieldsat --results_root <DATA_DIR>/yieldsat_results/before_full
python yieldsat_cluster.py status --plan $YIELDSAT_ARTIFACT_ROOT/cluster/before_full
python yieldsat_cluster.py aggregate --plan ... --results_root ... --compare_out results/yieldsat/paper_comparison
```
Start with `cluster/suites/smoke.yaml` (2 jobs). An example Indexed Job manifest
is in `cluster/k8s_indexed_job.example.yaml`.
