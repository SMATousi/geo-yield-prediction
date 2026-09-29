# --------------------------------------------------------
# YieldSAT cluster driver: plan a suite into jobs, run jobs on GPU pods,
# track completion on the shared data volume, aggregate. Specified in
# spec/yieldsat_cluster_runs.md.
#
#   # once, on a machine with write access to the shared artifact root:
#   python yieldsat_cluster.py plan --suite cluster/suites/before_full.yaml \
#       --out $SHARED/cluster/before_full
#   # on each pod (job index from JOB_COMPLETION_INDEX or --job_index):
#   python yieldsat_cluster.py run --plan $SHARED/cluster/before_full
#   # progress, then aggregation + paper tables:
#   python yieldsat_cluster.py status --plan $SHARED/cluster/before_full
#   python yieldsat_cluster.py aggregate --plan $SHARED/cluster/before_full
#
# A *run* is one training (one fold x seed x config). A *job* is the unit a
# pod executes: all runs of one country-crop pair (job_unit: pair) or a
# balanced shard of them (job_unit: pair_shard). A job stages its country's
# cache to local disk once and executes its runs (runs_per_gpu at a time),
# each logging to W&B and writing a completion marker to the shared state
# directory, so re-submitted jobs skip finished runs.
# --------------------------------------------------------

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from dataset.yieldsat_splits import (
    PAPER_PAIRS, load_field_table, make_paper_folds, parse_pair, save_folds,
)
from yieldsat_paper_runs import INPUTS, MODELS
from yieldsat_prepare import fold_prefix

PLAN_VERSION = 1
# RTX 3090 single-run throughput measured 2026-09-29 (ARG-S, S2+ADM):
# ours 30.6 k samples/s, paper LSTM 141.8 k samples/s, test 158 k cells/s,
# ~4 s fixed start-up. Scaled by ``gpu_speed_factor`` for the target GPU.
REF_THROUGHPUT = {'ours': 30600.0, 'paper_lstm': 141800.0}
REF_TEST_ROWS_PER_S = 158000.0
REF_STARTUP_S = 20.0            # index/cache open, normalizer, loaders (conservative)
# aggregate speed-up of k concurrent runs on one GPU (3090: 3 runs -> ~1.45x)
CONCURRENCY_SPEEDUP = {1: 1.0, 2: 1.35, 3: 1.45, 4: 1.5}
POLICY = {'paper': ('season', 'paper'), 'strict': ('physical', 'strict')}


def _expand_env(value):
    return os.path.expandvars(value) if isinstance(value, str) else value


def load_suite(path):
    suite = yaml.safe_load(Path(path).read_text())
    suite['data'] = {k: _expand_env(v) for k, v in suite.get('data', {}).items()}
    suite.setdefault('pairs', 'all')
    if suite['pairs'] == 'all':
        suite['pairs'] = list(PAPER_PAIRS)
    suite.setdefault('job_unit', 'pair_shard')
    suite.setdefault('target_job_hours', 3.0)
    suite.setdefault('runs_per_gpu', 2)
    suite.setdefault('gpu_speed_factor', 0.7)
    suite.setdefault('wandb', {})
    return suite


def _budget_args(budget):
    args = []
    for key in ('epochs', 'steps_per_epoch', 'min_steps_per_epoch', 'max_steps_per_epoch',
                'batch_size', 'lr'):
        if key in budget:
            args += ['--{}'.format(key), str(budget[key])]
    return args


def _override(args, extra):
    """Append ``extra`` flags; argparse keeps the last occurrence."""
    return list(args) + list(extra)


def expand_runs(suite, table, artifact_root):
    """All runs of the suite, with fold manifests created where missing."""
    runs = []
    by_pair = {}
    for f in table['fields']:
        by_pair.setdefault((f['country'], f['crop']), []).append(f)
    for exp in suite['experiments']:
        model = exp['model']
        for protocol in exp['protocols']:
            for policy_name in exp.get('policies', ['paper']):
                group, policy = POLICY[policy_name]
                k = exp.get('k', 10)
                for pair in suite['pairs']:
                    prefix = fold_prefix(pair, protocol, group, policy, exp.get('fold_seed', 0), k)
                    index = Path(artifact_root) / 'splits' / '{}.folds.json'.format(prefix)
                    if not index.exists():
                        save_folds(make_paper_folds(table, pair, protocol, k=k, group=group,
                                                    policy=policy, seed=exp.get('fold_seed', 0),
                                                    val_frac=exp.get('val_frac', 0.1)),
                                   artifact_root, prefix)
                    folds = json.loads(index.read_text())['folds']
                    if exp.get('max_folds'):
                        folds = folds[:exp['max_folds']]      # calibration / smoke suites
                    country, crop = parse_pair(pair)
                    for inputs in exp['inputs']:
                        for seed in exp.get('seeds', [0]):
                            for i, split_name in enumerate(folds):
                                split = json.loads((Path(artifact_root) / 'splits' /
                                                    '{}.json'.format(split_name)).read_text())
                                runs.append(_make_run(suite, exp, model, protocol, policy_name, group,
                                                      policy, k, inputs, seed, pair, country, crop,
                                                      i, split_name, split))
    return runs


def _make_run(suite, exp, model, protocol, policy_name, group, policy, k, inputs, seed, pair,
              country, crop, fold, split_name, split):
    proto = {'cv': 'cv{}'.format(k), 'loro': 'loro', 'loyo': 'loyo'}[protocol]
    exp_group = '{}_{}_{}_s{}'.format(proto, group if protocol == 'cv' else 'na', policy,
                                      exp.get('fold_seed', 0))
    tag = exp['name']
    rel = Path('paper') / exp_group / inputs / '{}_seed{}'.format(tag, seed) / pair / 'fold{:02d}'.format(fold)
    readable = '{}|{}|{}|{}|{}|{}|fold{:02d}|seed{}'.format(
        suite['suite'], tag, proto, policy_name, inputs, pair, fold, seed)
    run_id = hashlib.sha1(readable.encode()).hexdigest()[:12]
    args = ['--data_contract', 'yieldsat_preprocessed_v1', '--countries', country,
            '--crops', crop, '--split', split_name, '--save_maps', str(exp.get('save_maps', 1)),
            '--seed', str(seed)]
    args += INPUTS[inputs] + MODELS[model]
    if model == 'ours':
        args += ['--fusion', exp.get('fusion', 'perceiver_summary')]
    args = _override(args, _budget_args(exp.get('budget', {})))
    args = _override(args, [str(a) for a in exp.get('extra_args', [])])
    n_train = split['summary']['train']['rows']
    n_test = split['summary']['test']['rows']
    return {'run_id': run_id, 'name': readable, 'rel_path': str(rel), 'pair': pair,
            'country': country, 'crop': crop, 'experiment': tag, 'model': model,
            'protocol': proto, 'policy': policy_name, 'inputs': inputs, 'seed': seed,
            'fold': fold, 'split': split_name, 'n_train': n_train, 'n_test': n_test,
            'budget': exp.get('budget', {}), 'args': args,
            'wandb_group': '{}|{}|{}|{}|{}|{}'.format(suite['suite'], tag, proto, policy_name,
                                                      inputs, pair),
            'wandb_tags': [suite['suite'], tag, proto, policy_name, inputs, pair, 'seed{}'.format(seed)]}


def _factor(speed, model):
    """``gpu_speed_factor`` may be one number or ``{model: factor, 'test': f}``."""
    if isinstance(speed, dict):
        return float(speed.get(model, speed.get('default', 0.7)))
    return float(speed)


def estimate_seconds(run, speed_factor):
    """Single-run wall seconds on the target GPU (see REF_* constants).
    Factors for concurrent runs are the per-run speed at ``runs_per_gpu``
    divided by CONCURRENCY_SPEEDUP, so measured per-run rates can be entered
    directly via ``measured_per_run``."""
    b = run['budget']
    model = run['model']
    batch = b.get('batch_size', 1028 if model == 'paper_lstm' else 512)
    epochs = b.get('epochs', 15 if model == 'paper_lstm' else 20)
    steps = b.get('steps_per_epoch', 0 if model == 'paper_lstm' else 500)
    if steps <= 0:
        steps = max(b.get('min_steps_per_epoch', 1), math.ceil(run['n_train'] / batch))
        if b.get('max_steps_per_epoch', 0) > 0:
            steps = min(steps, b['max_steps_per_epoch'])
    train = epochs * steps * batch / (REF_THROUGHPUT[model] * _factor(speed_factor, model))
    val = epochs * 2.0                       # ~500 cells per validation field
    test = run['n_test'] / (REF_TEST_ROWS_PER_S * _factor(speed_factor, 'test')) * (
        2 if '--eval_drop_stream' in run['args'] else 1)
    return REF_STARTUP_S + train + val + test


STAGE_GB = {'Argentina': 11.0, 'Brazil': 8.7, 'Germany': 1.2, 'Uruguay': 4.4}   # cache + index


def make_jobs(suite, runs):
    """Group runs into jobs: one per pair, or balanced shards within a pair.
    Each job pays a one-off staging cost (its country's cache copied to pod
    scratch at ``stage_mb_per_s``; Nautilus CephFS measured ~42 MB/s)."""
    speed = suite['gpu_speed_factor']
    rpg = suite['runs_per_gpu']
    eff = suite.get('concurrency_speedup', CONCURRENCY_SPEEDUP.get(rpg, 1.5))
    stage_rate = float(suite.get('stage_mb_per_s', 42.0))
    for r in runs:
        r['est_seconds'] = round(estimate_seconds(r, speed), 1)
    jobs = []
    for pair in suite['pairs']:
        pr = sorted([r for r in runs if r['pair'] == pair], key=lambda r: -r['est_seconds'])
        if not pr:
            continue
        total = sum(r['est_seconds'] for r in pr) / eff
        n = 1 if suite['job_unit'] == 'pair' else max(1, math.ceil(total / 3600 / suite['target_job_hours']))
        n = min(n, len(pr))                  # never an empty shard
        shards = [[] for _ in range(n)]
        load = [0.0] * n
        for r in pr:                         # longest-processing-time packing
            i = min(range(n), key=load.__getitem__)
            shards[i].append(r)
            load[i] += r['est_seconds']
        stage_h = STAGE_GB.get(pr[0]['country'], 5.0) * 1024 / stage_rate / 3600
        for i, shard in enumerate(shards):
            jobs.append({'job_index': len(jobs), 'pair': pair, 'shard': i, 'n_shards': n,
                         'country': shard[0]['country'],
                         'run_ids': [r['run_id'] for r in shard],
                         'est_hours': round(sum(r['est_seconds'] for r in shard) / eff / 3600
                                            + stage_h, 2)})
    return jobs


def simulate(jobs, gpus):
    """Wall-clock hours for ``gpus`` workers taking jobs longest-first."""
    free = [0.0] * gpus
    for j in sorted(jobs, key=lambda j: -j['est_hours']):
        i = min(range(gpus), key=free.__getitem__)
        free[i] += j['est_hours']
    return max(free) if free else 0.0


# ---- plan ---------------------------------------------------------------------

def cmd_plan(a):
    suite = load_suite(a.suite)
    data = suite['data']
    for key in ('source_root', 'artifact_root'):
        if not data.get(key) or '$' in data[key]:
            sys.exit('suite data.{} is not set (export the env var it references)'.format(key))
    countries = sorted({parse_pair(p)[0] for p in suite['pairs']})
    needs_geo = any(pol == 'strict' for e in suite['experiments'] for pol in e.get('policies', []))
    table = load_field_table(data['artifact_root'], data['source_root'], countries,
                             require_geometry=needs_geo)
    runs = expand_runs(suite, table, data['artifact_root'])
    jobs = make_jobs(suite, runs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'state').mkdir(exist_ok=True)
    plan = {'plan_version': PLAN_VERSION, 'suite': suite, 'suite_file': str(a.suite),
            'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'n_runs': len(runs), 'n_jobs': len(jobs),
            'jobs': jobs}
    (out / 'plan.json').write_text(json.dumps(plan, indent=1))
    with open(out / 'runs.jsonl', 'w') as f:
        for r in runs:
            f.write(json.dumps(r) + '\n')
    shutil.copy(a.suite, out / 'suite.yaml')
    print_estimate(plan, runs, a.gpus)


def print_estimate(plan, runs, gpus):
    jobs = plan['jobs']
    gpu_h = sum(j['est_hours'] for j in jobs)
    print('{} runs in {} jobs; estimated {:.1f} GPU-hours (A10-equivalent, {} runs/GPU)'.format(
        len(runs), len(jobs), gpu_h, plan['suite']['runs_per_gpu']))
    by = {}
    for r in runs:
        key = (r['experiment'], r['protocol'], r['policy'])
        d = by.setdefault(key, [0, 0.0])
        d[0] += 1
        d[1] += r['est_seconds']
    eff = plan['suite'].get('concurrency_speedup',
                            CONCURRENCY_SPEEDUP.get(plan['suite']['runs_per_gpu'], 1.5))
    for key, (n, s) in sorted(by.items()):
        print('  {:<12} {:<6} {:<7} {:>5} runs  {:7.1f} GPU-h'.format(*key, n, s / eff / 3600))
    longest = max(jobs, key=lambda j: j['est_hours'])
    print('longest job: {:.2f} h ({} shard {}/{})'.format(longest['est_hours'], longest['pair'],
                                                           longest['shard'] + 1, longest['n_shards']))
    for g in sorted({gpus, 8, 16, 32}):
        print('  wall-clock on {:>2} GPUs: {:.1f} h'.format(g, simulate(jobs, g)))


# ---- run ----------------------------------------------------------------------

def _load_plan(plan_dir):
    """Load a plan; YIELDSAT_SOURCE_ROOT / YIELDSAT_ARTIFACT_ROOT in the
    environment override the paths recorded at planning time, so a plan made
    on one machine runs where the shared volume is mounted elsewhere."""
    plan_dir = Path(plan_dir)
    plan = json.loads((plan_dir / 'plan.json').read_text())
    for key, env in (('source_root', 'YIELDSAT_SOURCE_ROOT'), ('artifact_root', 'YIELDSAT_ARTIFACT_ROOT')):
        if os.environ.get(env):
            plan['suite']['data'][key] = os.environ[env]
    runs = {}
    with open(plan_dir / 'runs.jsonl') as f:
        for line in f:
            r = json.loads(line)
            runs[r['run_id']] = r
    return plan, runs


def stage_local(plan, job, runs, local_root):
    """Copy what the job's runs read from the artifact root to local disk:
    the country's index, cache and field statistics, geometry and the runs'
    split manifests. The source NetCDF stays on the shared volume (only its
    fingerprint is read)."""
    src = Path(plan['suite']['data']['artifact_root'])
    dst = Path(local_root) / 'artifacts'
    c = job['country']
    t0 = time.time()
    for rel in ('index/{}'.format(c), 'cache/{}'.format(c)):
        target = dst / rel
        if (target / ('cache_manifest.json' if rel.startswith('cache') else 'index_manifest.json')).exists():
            continue                          # already staged by an earlier job on this pod
        target.mkdir(parents=True, exist_ok=True)
        for f in (src / rel).iterdir():
            if f.name == 'uuids.npy':
                continue                      # not read by training
            if f.name.endswith('_manifest.json'):
                continue                      # copied last: marks a complete copy
            shutil.copy2(f, target / f.name)
        for f in (src / rel).glob('*_manifest.json'):
            shutil.copy2(f, target / f.name)
    (dst / 'geometry').mkdir(parents=True, exist_ok=True)
    for f in (src / 'geometry').glob('fields_geometry_*.json'):
        shutil.copy2(f, dst / 'geometry' / f.name)
    (dst / 'splits').mkdir(parents=True, exist_ok=True)
    for rid in job['run_ids']:
        name = runs[rid]['split']
        shutil.copy2(src / 'splits' / '{}.json'.format(name), dst / 'splits' / '{}.json'.format(name))
    print('staged {} to {} in {:.0f} s'.format(c, dst, time.time() - t0), flush=True)
    return dst


def _state_paths(plan_dir, run_id):
    state = Path(plan_dir) / 'state'
    return state / '{}.done.json'.format(run_id), state / '{}.failed.json'.format(run_id)


def _launch(run, plan, artifact_root, work_dir, use_wandb):
    out = Path(work_dir) / run['rel_path']
    out.mkdir(parents=True, exist_ok=True)
    wb = plan['suite']['wandb']
    cmd = [sys.executable, 'main_yieldsat_finetune.py', '--source_root',
           plan['suite']['data']['source_root'], '--artifact_root', str(artifact_root),
           '--output_dir', str(out), '--num_workers', str(plan['suite'].get('num_workers', 4))]
    cmd += run['args']
    if use_wandb:
        cmd += ['--wandb', '--wandb_project', str(wb.get('project', 'yieldsat')),
                '--wandb_group', run['wandb_group'], '--wandb_name', run['name'],
                '--wandb_job_type', 'finetune', '--wandb_tags', *run['wandb_tags'],
                '--wandb_meta', json.dumps({k: run[k] for k in (
                    'run_id', 'name', 'rel_path', 'pair', 'experiment', 'protocol', 'policy',
                    'inputs', 'seed', 'fold', 'split')})]
        if wb.get('entity'):
            cmd += ['--wandb_entity', str(wb['entity'])]
    log = open(out / 'train.log', 'w')
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log, out


def _finish(run, proc, log, out, plan_dir, results_root, keep_local):
    log.close()
    done, failed = _state_paths(plan_dir, run['run_id'])
    if proc.returncode == 0 and (out / 'report.json').exists():
        rep = json.loads((out / 'report.json').read_text())
        o = rep.get('test', {}).get('overall', {})
        if results_root:
            dst = Path(results_root) / run['rel_path']
            dst.mkdir(parents=True, exist_ok=True)
            for f in ('report.json', 'test_predictions.npz', 'normalizer.json', 'train.log'):
                if (out / f).exists():
                    shutil.copy2(out / f, dst / f)
        marker = {'run_id': run['run_id'], 'name': run['name'], 'finished': time.strftime('%Y-%m-%dT%H:%M:%S'),
                  'wall_seconds': rep['resources']['wall_seconds'], 'host': os.uname().nodename,
                  'pixel_rmse': o.get('pixel', {}).get('rmse'), 'field_rmse': o.get('field_level', {}).get('rmse')}
        done.write_text(json.dumps(marker))
        if failed.exists():
            failed.unlink()
        offline = out / 'wandb'
        if offline.exists() and any(offline.glob('offline-run-*')):
            # offline W&B runs survive on the shared volume for `wandb sync`
            keep = Path(plan_dir) / 'wandb_offline' / run['run_id']
            keep.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(offline, keep, dirs_exist_ok=True)
            marker['wandb_offline_dir'] = str(keep)
            done.write_text(json.dumps(marker))
        if not keep_local:
            shutil.rmtree(out, ignore_errors=True)
        return True
    tail = (out / 'train.log').read_text()[-4000:] if (out / 'train.log').exists() else ''
    failed.write_text(json.dumps({'run_id': run['run_id'], 'name': run['name'],
                                  'returncode': proc.returncode, 'host': os.uname().nodename,
                                  'log_tail': tail}))
    return False


def run_job(plan_dir, job_index, local_root=None, work_dir=None, results_root=None,
            runs_per_gpu=None, use_wandb=True, keep_local=False, dry_run=False):
    plan, runs = _load_plan(plan_dir)
    if not 0 <= job_index < plan['n_jobs']:
        raise SystemExit('job index {} outside the plan (n_jobs={})'.format(job_index, plan['n_jobs']))
    job = plan['jobs'][job_index]
    todo = [runs[r] for r in job['run_ids'] if not _state_paths(plan_dir, r)[0].exists()]
    print('job {} ({} shard {}/{}): {} runs, {} to do, est {:.2f} h'.format(
        job_index, job['pair'], job['shard'] + 1, job['n_shards'], len(job['run_ids']), len(todo),
        job['est_hours']), flush=True)
    if dry_run or not todo:
        return 0
    artifact_root = plan['suite']['data']['artifact_root']
    if local_root:
        artifact_root = stage_local(plan, job, runs, local_root)
    work_dir = work_dir or (Path(local_root) / 'work' if local_root else Path(plan_dir) / 'work')
    k = runs_per_gpu or plan['suite']['runs_per_gpu']
    pending, running, failures = list(todo), [], 0
    while pending or running:
        while pending and len(running) < k:
            r = pending.pop(0)
            proc, log, out = _launch(r, plan, artifact_root, work_dir, use_wandb)
            running.append((r, proc, log, out, time.time()))
        time.sleep(5)
        for item in list(running):
            r, proc, log, out, t0 = item
            if proc.poll() is None:
                continue
            running.remove(item)
            ok = _finish(r, proc, log, out, plan_dir, results_root, keep_local)
            failures += 0 if ok else 1
            print('{} {} ({:.0f} s)'.format('ok' if ok else 'FAILED', r['name'], time.time() - t0),
                  flush=True)
    return 1 if failures else 0


def cmd_run(a):
    idx = a.job_index
    if idx is None:
        idx = int(os.environ.get('JOB_COMPLETION_INDEX', os.environ.get('JOB_INDEX', '-1')))
    if idx < 0:
        sys.exit('no job index: pass --job_index or set JOB_COMPLETION_INDEX')
    sys.exit(run_job(a.plan, idx, a.local_root, a.work_dir, a.results_root, a.runs_per_gpu,
                     not a.no_wandb, a.keep_local, a.dry_run))


def cmd_worker(a):
    plan, _ = _load_plan(a.plan)
    rc = 0
    for idx in range(a.worker_index, plan['n_jobs'], a.num_workers):
        rc |= run_job(a.plan, idx, a.local_root, a.work_dir, a.results_root, a.runs_per_gpu,
                      not a.no_wandb, a.keep_local, a.dry_run)
    sys.exit(rc)


def cmd_status(a):
    plan, runs = _load_plan(a.plan)
    state = Path(a.plan) / 'state'
    done = {p.name.split('.')[0] for p in state.glob('*.done.json')}
    failed = {p.name.split('.')[0] for p in state.glob('*.failed.json')} - done
    print('runs: {} done, {} failed, {} pending of {}'.format(
        len(done), len(failed), len(runs) - len(done) - len(failed), len(runs)))
    for j in plan['jobs']:
        d = sum(r in done for r in j['run_ids'])
        f = sum(r in failed for r in j['run_ids'])
        print('  job {:>3} {} shard {}/{}: {}/{} done{}'.format(
            j['job_index'], j['pair'], j['shard'] + 1, j['n_shards'], d, len(j['run_ids']),
            ', {} failed'.format(f) if f else ''))
    for rid in sorted(failed)[:20]:
        info = json.loads((state / '{}.failed.json'.format(rid)).read_text())
        print('FAILED', info['name'], 'rc', info['returncode'], 'on', info['host'])


def cmd_aggregate(a):
    from yieldsat_collect_results import aggregate_folds
    plan, runs = _load_plan(a.plan)
    results_root = Path(a.results_root)
    if a.from_wandb:
        download_from_wandb(plan, runs, results_root)
    exp_dirs = sorted({str(Path(r['rel_path']).parent) for r in runs.values()})
    for rel in exp_dirs:
        d = results_root / rel
        if not d.exists() or not any(d.glob('fold*/report.json')):
            continue
        expected = sum(1 for r in runs.values() if str(Path(r['rel_path']).parent) == rel)
        agg = aggregate_folds(d)
        agg['complete'] = agg['folds_done'] == expected
        (d / 'aggregate.json').write_text(json.dumps(agg, indent=1))
        print('{} [{}/{}] pixel R2 {:.3f} field R2 {:.3f}'.format(
            rel, agg['folds_done'], expected, agg['fold_mean_std']['pixel_r2']['mean'],
            agg['fold_mean_std']['field_r2']['mean']))
    if a.compare_out:
        subprocess.check_call([sys.executable, 'yieldsat_paper_compare.py', '--runs_root',
                               str(results_root), '--out_dir', a.compare_out])


def download_from_wandb(plan, runs, results_root):
    """Fetch each finished run's results artifact (report + predictions)
    into ``results_root/<rel_path>`` using the W&B public API."""
    import wandb
    api = wandb.Api()
    wb = plan['suite']['wandb']
    path = '{}/{}'.format(wb['entity'], wb['project']) if wb.get('entity') else wb['project']
    for run in api.runs(path, filters={'tags': plan['suite']['suite']}):
        meta = run.config.get('job', {})
        rel = meta.get('rel_path')
        if not rel or run.state != 'finished':
            continue
        dst = results_root / rel
        if (dst / 'report.json').exists():
            continue
        for art in run.logged_artifacts():
            if art.type == 'results':
                art.download(root=str(dst))
        print('downloaded', rel, flush=True)


# ---- Kubernetes (Nautilus/NRP) manifest ------------------------------------------

def render_k8s(suite_path, plan_name, n_jobs, parallelism, image, pvc, secret, gpus,
               data_mount='/data', data_subdir='YieldSAT', git_ref='main', cpu=8, memory_gi=32,
               name_prefix='smatousi-yieldsat'):
    """Indexed Job running every job of a plan on the shared PVC."""
    base = '{}/{}'.format(data_mount, data_subdir)
    plan_dir = '{}/yieldsat_artifacts/cluster/{}'.format(base, plan_name)
    script = ('git clone --quiet https://github.com/SMATousi/geo-yield-prediction.git /workspace/code && '
              'cd /workspace/code && git checkout --quiet "$YIELDSAT_GIT_REF" && '
              'exec cluster/nautilus/run_in_pod.sh run --plan {} --local_root /scratch/yieldsat '
              '--results_root {}/yieldsat_results/{}'.format(plan_dir, base, plan_name))
    res = {'nvidia.com/gpu': 1, 'cpu': str(cpu), 'memory': '{}Gi'.format(memory_gi),
           'ephemeral-storage': '50Gi'}
    return {
        'apiVersion': 'batch/v1', 'kind': 'Job',
        'metadata': {'name': '{}-{}'.format(name_prefix, plan_name.replace('_', '-'))},
        'spec': {
            'completionMode': 'Indexed', 'completions': n_jobs, 'parallelism': parallelism,
            'backoffLimitPerIndex': 2, 'maxFailedIndexes': max(1, n_jobs // 10),
            'ttlSecondsAfterFinished': 7 * 86400,
            'template': {'spec': {
                'restartPolicy': 'Never',
                'affinity': {'nodeAffinity': {'requiredDuringSchedulingIgnoredDuringExecution': {
                    'nodeSelectorTerms': [{'matchExpressions': [{
                        'key': 'nvidia.com/gpu.product', 'operator': 'In', 'values': list(gpus)}]}]}}},
                'containers': [{
                    'name': 'yieldsat', 'image': image, 'imagePullPolicy': 'Always',
                    'command': ['/bin/bash', '-c'], 'args': [script],
                    'env': [
                        {'name': 'YIELDSAT_GIT_REF', 'value': git_ref},
                        {'name': 'YIELDSAT_SUITE', 'value': suite_path},
                        {'name': 'YIELDSAT_PLAN_DIR', 'value': plan_dir},
                        {'name': 'YIELDSAT_SOURCE_ROOT', 'value': base + '/preprocessed'},
                        {'name': 'YIELDSAT_ARTIFACT_ROOT', 'value': base + '/yieldsat_artifacts'},
                        {'name': 'WANDB_API_KEY', 'valueFrom': {'secretKeyRef': {
                            'name': secret, 'key': 'WANDB_API_KEY'}}}],
                    'resources': {'limits': dict(res), 'requests': dict(res)},
                    'volumeMounts': [{'name': 'data', 'mountPath': data_mount},
                                     {'name': 'scratch', 'mountPath': '/scratch'},
                                     {'name': 'dshm', 'mountPath': '/dev/shm'}]}],
                'volumes': [{'name': 'data', 'persistentVolumeClaim': {'claimName': pvc}},
                            {'name': 'scratch', 'emptyDir': {'sizeLimit': '40Gi'}},
                            {'name': 'dshm', 'emptyDir': {'medium': 'Memory', 'sizeLimit': '8Gi'}}]}}}}


def cmd_k8s(a):
    plan, _ = _load_plan(a.plan)
    manifest = render_k8s(a.suite, a.plan_name or Path(a.plan).name, plan['n_jobs'], a.parallelism,
                          a.image, a.pvc, a.secret, a.gpu, git_ref=a.git_ref, cpu=a.cpu,
                          memory_gi=a.memory_gi)
    text = '# generated by yieldsat_cluster.py k8s; no credentials inside\n' + yaml.safe_dump(
        manifest, sort_keys=False, width=200)
    if a.out:
        Path(a.out).write_text(text)
        print('wrote', a.out, '({} jobs, parallelism {})'.format(plan['n_jobs'], a.parallelism))
    else:
        print(text)


def main():
    p = argparse.ArgumentParser('YieldSAT cluster driver')
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('plan')
    s.add_argument('--suite', required=True)
    s.add_argument('--out', required=True, help='plan directory on the shared volume')
    s.add_argument('--gpus', type=int, default=16)
    s.set_defaults(func=cmd_plan)
    for name, fn in (('run', cmd_run), ('worker', cmd_worker)):
        s = sub.add_parser(name)
        s.add_argument('--plan', required=True)
        if name == 'run':
            s.add_argument('--job_index', type=int, default=None)
        else:
            s.add_argument('--worker_index', type=int, required=True)
            s.add_argument('--num_workers', type=int, required=True)
        s.add_argument('--local_root', default=os.environ.get('YIELDSAT_LOCAL_ROOT'),
                       help='pod-local scratch; the country cache is staged here')
        s.add_argument('--work_dir', default=None)
        s.add_argument('--results_root', default=os.environ.get('YIELDSAT_RESULTS_ROOT'),
                       help='shared dir receiving report/predictions for aggregation')
        s.add_argument('--runs_per_gpu', type=int, default=None)
        s.add_argument('--no_wandb', action='store_true')
        s.add_argument('--keep_local', action='store_true')
        s.add_argument('--dry_run', action='store_true')
        s.set_defaults(func=fn)
    s = sub.add_parser('k8s', help='render a Nautilus Indexed Job for a plan')
    s.add_argument('--plan', required=True, help='a local plan of the same suite (for n_jobs)')
    s.add_argument('--suite', required=True, help='suite path inside the repository')
    s.add_argument('--plan_name', default=None, help='plan directory name on the PVC')
    s.add_argument('--parallelism', type=int, default=16)
    s.add_argument('--image', default='gitlab-registry.nrp-nautilus.io/smatous/yieldsat:latest')
    s.add_argument('--pvc', default='ali-vol-1tera')
    s.add_argument('--secret', default='smatousi-wandb')
    s.add_argument('--gpu', nargs='+', default=['NVIDIA-A10'])
    s.add_argument('--git_ref', default='main')
    s.add_argument('--cpu', type=int, default=8)
    s.add_argument('--memory_gi', type=int, default=32)
    s.add_argument('--out', default=None)
    s.set_defaults(func=cmd_k8s)
    s = sub.add_parser('status')
    s.add_argument('--plan', required=True)
    s.set_defaults(func=cmd_status)
    s = sub.add_parser('aggregate')
    s.add_argument('--plan', required=True)
    s.add_argument('--results_root', required=True)
    s.add_argument('--from_wandb', action='store_true')
    s.add_argument('--compare_out', default=None)
    s.set_defaults(func=cmd_aggregate)
    a = p.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
