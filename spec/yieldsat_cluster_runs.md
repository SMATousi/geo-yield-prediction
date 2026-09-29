# YieldSAT Cluster Runs — Distributed "Before" Suite

**Status:** Implemented and tested locally on 2026-09-29; not yet run on the
cluster. The cluster and pod setup is out of scope here and will be added when
the cluster is chosen. This document specifies how the full YieldSAT
paper-comparison suite is split across GPU pods, how results reach W&B, and the
runtime estimate.

Related documents:
- [yieldsat_paper_comparison.md](./yieldsat_paper_comparison.md): protocols and
  reporting;
- [yieldsat_data_contract.md](./yieldsat_data_contract.md): data pipeline;
- [architecture.md §9](./architecture.md#9-yieldsat-point-model--exact-architecture):
  the model.

## 1. Goal

The "before knowledge pretraining" suite is the complete baseline table for the
CVPR 2027 submission:
- every paper country–crop pair;
- CV10, LORO and LOYO;
- the paper-compatible and the leakage-safe (strict) policy;
- S2 and S2+ADM inputs;
- 3 seeds for our model;
- the paper-LSTM re-run as a protocol anchor.

The knowledge-pretraining ("after") suite will reuse the same plans, folds,
seeds and budgets with a pretrained initialization added. §8 covers it.

## 2. Terminology

| Term | Meaning |
|---|---|
| **Run** | One training + evaluation: one fold × seed × input set × model × policy. It produces one W&B run. |
| **Job** | What a pod executes: a set of runs of **one country–crop pair**. `job_unit: pair` gives exactly one job per pair. `pair_shard` (default) splits a pair's runs into balanced shards of about `target_job_hours`. A job stages its country's data once and executes its runs `runs_per_gpu` at a time. |
| **Suite** | A YAML file (`cluster/suites/*.yaml`) declaring the experiment matrix, training budget, W&B project and data paths. |
| **Plan** | The expansion of a suite into `runs.jsonl` (every run with its exact arguments, stable 12-character `run_id` and runtime estimate) and `plan.json` (jobs). It is written by `yieldsat_cluster.py plan` into a directory on the shared volume, together with `state/` completion markers. |

One job per pair would be simpler, but the pairs are very unequal. ARG-S alone
has 3.1 M cells, 44 LORO folds and about 30% of the suite's GPU time. With
`job_unit: pair`, the 16-GPU wall time is set by ARG-S at about 319 h, because
7 GPUs idle and one runs ARG-S alone. With `pair_shard`, jobs still contain one
pair only, so staging and page cache stay per country, but the load balances.

## 3. The `before_full` suite (`cluster/suites/before_full.yaml`)

| Experiment | Protocols | Policies | Inputs | Seeds | Budget per run |
|---|---|---|---|---|---|
| `ours`: point model, `perceiver_summary` fusion (YS-11 decision) | CV10, LORO, LOYO | paper (season grouping), strict (physical grouping, shared ground excluded) | S2, S2+ADM | 0, 1, 2 | **60 epochs** of one pass over the fold's training cells, clamped to 500–1,500 steps of 512 (previously 20×500) |
| `paper_lstm`: paper LSTM re-run (PC-05 preset) | CV10 | paper | S2, S2+ADM | 0 | 15 full-pass epochs, batch 1,028, `stats-*` normalization (paper-faithful) |

Totals: **3,546 runs in 343 jobs**. The table gives runs per protocol, policy
and fold count.

| Experiment | Protocol | Policy | Runs |
|---|---|---|---|
| ours | CV10 | paper | 540 (9 pairs × 10 folds × 2 inputs × 3 seeds) |
| ours | CV10 | strict | 540 |
| ours | LORO | paper | 828 (138 farm folds) |
| ours | LORO | strict | 702 (117 farm-cluster folds) |
| ours | LOYO | paper | 378 (63 year folds) |
| ours | LOYO | strict | 378 |
| paper_lstm | CV10 | paper | 180 |

**Effective training length.** The clamp gives small pairs many passes and
caps the large ones:

| Pair size | Example | Steps × epochs | Passes over training cells |
|---|---|---|---|
| Small | GER-R, ~0.24 M training cells per CV fold | 500 × 60 | ~63 |
| Medium | URG-S, ~1.7 M | 1,500 × 60 | ~26 |
| Large | ARG-S, ~2.5 M | 1,500 × 60 | ~18 |

The best epoch is chosen on the 10% validation carve-out, so extra epochs on
small pairs cannot leak test information.

Other evaluation settings: test on every held-out cell, all 24 slots
(retrospective, as in the paper), and one GeoTIFF map per run
(`save_maps: 1`).

## 4. Shared data volume

All pods mount one data directory, written below as `<DATA_DIR>`. The path is
provided at submission.

```text
<DATA_DIR>/Preprocessed/<Country>/merge_s2-soil-dem-weather-coords.nc   read-only source (146 GB)
<DATA_DIR>/yieldsat_artifacts/                                           prepared once (~27 GB)
    index/<Country>/         rows.npz, fields.json, index_manifest.json (uuids.npy optional)
    cache/<Country>/         temporal/static/times/flags .npy, field_stats.npz (v2), cache_manifest.json
    geometry/                fields_geometry_*.json (physical fields, grids)
    splits/                  all fold manifests (created by `plan`)
    cluster/<suite>/         plan.json, runs.jsonl, suite.yaml, state/, wandb_offline/
<DATA_DIR>/yieldsat_results/<suite>/paper/...                            report.json + predictions per run
```

**Preparation**, done once, from a machine with write access. Either:
- (a) copy `/root/yieldsat_artifacts` from the preparation host with `rsync -a`
  (index, cache, geometry and splits are all portable), or
- (b) rebuild on the cluster with `yieldsat_prepare.py index/geometry/cache/stats`
  (~35 min for the cache, which needs `Raw.zip` for geometry).

Artifacts identify the source snapshot by **content** (size plus SHA-256 of the
first and last 4 MiB; changed 2026-09-29). Copying the source without
preserving timestamps therefore does not invalidate them; changed bytes still
do.

**Planning.** Then run
`yieldsat_cluster.py plan --suite cluster/suites/before_full.yaml --out <DATA_DIR>/yieldsat_artifacts/cluster/before_full`
with `YIELDSAT_SOURCE_ROOT`/`YIELDSAT_ARTIFACT_ROOT` pointing at the shared
paths. Planning takes seconds. It creates every fold manifest before submission,
so pods never write shared inputs and cannot race. Fold assignment is
deterministic (seeded), so re-planning reproduces the same folds and `run_id`s.
At run time the pods' `YIELDSAT_*` variables override the paths recorded in the
plan.

## 5. Job execution (`yieldsat_cluster.py run`)

1. **Job index.** It comes from `--job_index`, or from `JOB_COMPLETION_INDEX`
   (Kubernetes Indexed Jobs). `worker --worker_index i --num_workers N` instead
   loops over jobs i, i+N, … on long-lived pods.
2. **Skip finished runs.** Runs with a `state/<run_id>.done.json` marker are
   skipped. A resubmitted or retried job only does the remaining work.
3. **Stage data.** With `--local_root` (pod scratch), the job copies the
   country's index, cache and field statistics, the geometry and its runs' split
   manifests to local disk. That is ≤ 11 GB (Argentina), and the manifests are
   copied last, so a partial copy is never used. Concurrent runs then share the
   OS page cache for random row reads instead of hitting the network volume.
   Without `--local_root`, runs read the shared artifact root directly.
4. **Launch runs.** Runs start as subprocesses of `main_yieldsat_finetune.py`,
   `runs_per_gpu` (default 2) at a time. One run needs < 1 GB of GPU memory and
   ~2 GB RSS.
5. **Record each run.**
   - **W&B** (`--wandb`):
     - `project` comes from the suite, `entity` from the suite or
       `WANDB_ENTITY`;
     - `group = suite|experiment|protocol|policy|inputs|pair`, so the folds and
       seeds of one table cell sit together;
     - `name = suite|experiment|protocol|policy|inputs|pair|foldNN|seedS`;
     - tags: suite, experiment, protocol, policy, inputs, pair, seed;
     - `config` = all CLI arguments plus `config.job` (run_id, rel_path, pair,
       fold, split, …).
   - **Logged content.** Per-epoch train loss, learning rate and validation
     metrics. The summary gets test metrics (overall and per country×crop;
     pixel, field-balanced and field level), the stress-test metrics if
     enabled, I/O and resources.
   - **Artifacts.** `model-<name>` holds `checkpoint_best.pth`,
     `sensor_checkpoint_last.pth` and `normalizer.json`. `results-<name>` holds
     `report.json`, `test_predictions.npz` and the GeoTIFF maps.
   - **Offline fallback.** If W&B cannot be reached at start-up, the run logs
     offline instead of failing. Offline run directories are preserved under
     `cluster/<suite>/wandb_offline/<run_id>/` for `wandb sync`.
6. **Completion.** On success, `report.json`, `test_predictions.npz`,
   `normalizer.json` and `train.log` are copied to `--results_root` (shared,
   used by `aggregate`). The `done` marker is written (wall time, host, headline
   metrics) and the local outputs are deleted. On failure, a
   `state/<run_id>.failed.json` marker with the log tail is written. The job
   continues with its other runs and exits non-zero at the end, so the scheduler
   can retry it, and the retry repeats only the failed runs.

**The API key** is never written to files or plans. It is read from the
`WANDB_API_KEY` environment variable, which the job YAML provides (plain value
or secret).

**W&B storage.** A run uploads about 9 MB of models (best checkpoint and last
sensor checkpoint) plus ≤ 2 MB of results (per-fold predictions ≤ 0.4 M cells,
report, map). The full suite is roughly **35 GB** of artifacts; check the W&B
storage quota. If
needed, `--wandb_no_artifacts` (via `extra_args`) keeps metrics only, and the
shared `results_root` still receives reports and predictions.

## 6. Pod requirements and submission

**Image (Nautilus/NRP):** `gitlab-registry.nrp-nautilus.io/smatous/yieldsat`,
built from the repository `Dockerfile` (2026-09-29).
- Base: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`. It carries Python
  3.11, PyTorch 2.5.1+cu121 and cuDNN 9.1, the same stack as `geo-yield-phase0`.
- Additions: `requirements.txt` minus torch/torchvision (wandb 0.30, rasterio,
  h5py, scikit-learn, pyyaml, …), plus rsync/git.
- The code sits at `/workspace/geo-yield-prediction`, installed editable.
- The build runs `tests/test_yieldsat.py` (31 passed).
- Size: 3.4 GB compressed, 10.3 GB unpacked. No data or credentials are baked
  in.
- Verified on the build host: imports, and a real GER-R fold trained and
  evaluated on CPU from mounted data. The GPU path is untested there (no NVIDIA
  container toolkit); CUDA 12.1 supports the A10 (sm_86).
- Rebuild after code changes with
  `docker build -t gitlab-registry.nrp-nautilus.io/smatous/yieldsat . && docker push …`,
  and tag builds (e.g. `:<git-sha>`) so a running suite keeps a fixed code
  version.
- On the build host, `docker run` needs `--network host`: runc cannot set a
  network sysctl there. This does not affect Kubernetes pods.

The container image needs the repository and the `geo-yield-phase0`
environment (`environment.yml`; `wandb` and `pyyaml` were added 2026-09-29).
Per pod:
- 1 GPU (A10, 24 GB);
- ≥ 8 vCPU (2 runs × 4 loader workers);
- 32–48 GiB RAM;
- ≥ 40 GiB local scratch.

Environment variables:
- `YIELDSAT_SOURCE_ROOT` and `YIELDSAT_ARTIFACT_ROOT` (shared paths);
- `WANDB_API_KEY`, and optionally `WANDB_ENTITY`;
- optionally `YIELDSAT_LOCAL_ROOT` and `YIELDSAT_RESULTS_ROOT` (equivalent to
  the flags).

Submission options:
- **Indexed Job:** `completions = n_jobs` (343), `parallelism = 16`, one pod
  per index. `cluster/k8s_indexed_job.example.yaml` is an untested template
  with placeholders for image, volume and key.
- **16 long-lived workers:** pod *i* runs
  `yieldsat_cluster.py worker --worker_index i --num_workers 16`.

**Smoke test first.** `cluster/suites/smoke.yaml` is 2 jobs (GER-R and URG-S,
LOYO, 2 epochs). Plan it into its own directory and submit it with
`completions: 2`. It checks the mounts, staging, W&B credentials and
artifacts, markers and aggregation, and it measures the real A10 throughput
(W&B summary `io/samples_per_second`) to recalibrate `gpu_speed_factor`
before the full suite.

## 7. Monitoring and aggregation

```bash
python yieldsat_cluster.py status --plan <plan>          # done/failed/pending per job, failed log tails
python yieldsat_cluster.py aggregate --plan <plan> --results_root <DATA_DIR>/yieldsat_results/before_full \
    --compare_out results/yieldsat/paper_comparison      # fold mean±std per cell + paper tables
python yieldsat_cluster.py aggregate --plan <plan> --results_root <dir> --from_wandb   # rebuild from W&B artifacts
```

`aggregate` refuses a cell that appears in two test folds, and it marks each
experiment `complete` only when all of its folds are done. The comparison
merges seeds: the reported value is the mean over seeds of the fold mean ± the
mean fold std (the paper's statistic), with the std over seeds as `seed_std_*`.

## 8. Runtime estimate

**Method.** Each run is estimated as start-up (20 s) plus epochs × steps ×
batch / throughput plus validation (~2 s/epoch) plus test cells / inference
rate. The reference numbers were measured on the RTX 3090 on 2026-09-29 with
ARG-S and S2+ADM, one run alone:
- our model: 30.6 k samples/s;
- the LSTM: 141.8 k samples/s;
- inference: 158 k cells/s.

These are scaled by `gpu_speed_factor = 0.7` for an A10. That factor is an
**assumption**; the A10 has not been measured. Two concurrent runs per GPU are
assumed to deliver 1.35× one run (the 3090 gave 1.45× with three). Jobs are
packed onto GPUs longest-first.

| Suite variant | Runs | Jobs | GPU-hours (A10) | **Wall clock, 16 A10** | Longest job |
|---|---|---|---|---|---|
| **A `before_full` as specified** (60 epochs, 500–1,500 steps, 3 seeds, both policies) | 3,546 | 343 | 1,351 | **≈ 87 h (3.6 days)** | 4.2 h |
| B budget 40 epochs, 500–1,000 steps | 3,546 | 173 | 670 | ≈ 43 h | 4.0 h |
| C full budget; strict policy with 1 seed | 2,466 | 240 | 943 | ≈ 59 h | 4.2 h |
| D = B + C | 2,466 | 121 | 464 | ≈ 31 h | 3.9 h |
| A with 3 runs per GPU | 3,546 | 319 | 1,258 | ≈ 79 h | 4.1 h |
| A with `job_unit: pair` (9 jobs) | 3,546 | 9 | 1,351 | ≈ 319 h | 319 h |

GPU-hours of variant A by component:

| Component | GPU-h |
|---|---|
| CV10, paper | 216 |
| CV10, strict | 217 |
| LORO, paper | 365 |
| LORO, strict | 282 |
| LOYO, paper | 150 |
| LOYO, strict | 114 |
| LSTM re-run | 8 |

**Uncertainty ±30–50%.** It depends on the true A10 speed, pod CPU count (data
loading), the shared-volume read rate during staging (an 11 GB Argentina copy
per job; about 1–2 min at ≥ 100 MB/s, not included), and W&B upload time.
Recalibrate after the smoke run. The planner prints the updated estimate for
any suite (`plan … --gpus 16`).

## 9. "After" suite

The knowledge-pretraining suite reuses this machinery unchanged:
- the same pairs, protocols, folds (same seeds give the same manifests and
  `run_id` scheme) and budgets;
- new `experiments` entries that pass the pretraining checkpoint through
  `extra_args` (`--init_sensor_ckpt …`);
- generic-SSL and shuffled-knowledge controls as separate experiments.

Pretraining must not see the fold's test data. Either pretrain per fold, which
needs a pretraining job type in the plan, or pretrain on data disjoint from
every test fold. This will be specified when KP-01–KP-08 land.

## 10. Open items

- Measure the A10 throughput (smoke run) and pod CPU count; update
  `gpu_speed_factor`.
- Choose the budget and seed variant (A–D) against the deadline.
- Confirm the W&B storage quota for about 35 GB of artifacts, or switch large
  predictions to `results_root` only.
- Provide `<DATA_DIR>` and the image; fill in the example manifest; copy or
  rebuild `yieldsat_artifacts` on the shared volume and plan there.
