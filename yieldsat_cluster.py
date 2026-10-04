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
#
# Image suites (model ``image``, spec/yieldsat-image-training.md) run
# main_yieldsat_image.py on the same fold manifests. Their jobs stage the
# country's 64x64 tiles, DINO feature cache, index fields and splits instead
# of the point cache. ``donors`` experiments train warm-start donors (one run
# per donor x inputs x seed, checkpoint kept in the results root);
# ``warm_start`` maps target pairs to a donor whose checkpoint they start from.
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
REF_THROUGHPUT = {'ours': 30600.0, 'paper_lstm': 141800.0,
                  # image model: 64x64 tiles/s per run, loader-bound; measured in
                  # training on the development host (RTX 3090, 3 loader workers,
                  # uncompressed staged tiles), 2026-10-01
                  'image': 118.0,
                  # point pretraining over all 4 countries (random cell access),
                  # RTX 3090, 6 loader workers, 2026-10-04 smoke (PK-06)
                  'pretrain_ssl': 15700.0, 'pretrain_knowledge': 10100.0}
SCRIPTS = {'image': 'main_yieldsat_image.py'}           # default: main_yieldsat_finetune.py
IMAGE_DEFAULTS = {'epochs': 60, 'batch_size': 16, 'min_steps_per_epoch': 20, 'lr': 5e-4}
WARM_START_LR_SCALE = 0.3
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


def _image_tiles(suite):
    """Tile table of the suite's image dataset (planning needs tile counts)."""
    from dataset.yieldsat_image_dataset import load_tile_table
    countries = sorted({parse_pair(p)[0] for p in suite['pairs']}
                       | {c for e in suite['experiments'] for d in e.get('donors', [])
                          for c in d['countries']})
    return load_tile_table(suite['data']['image_root'], countries)


def _tile_counts(tiles, split, crop, train_min_valid=0):
    """Tiles per partition; training tiles need ``train_min_valid`` valid
    cells (full-coverage builds, plan v2), validation/test use all."""
    from dataset.yieldsat_image_dataset import tiles_for_split
    parts = tiles_for_split(tiles, split, crops=[crop])
    full = [t for t in parts['train'] if t.get('valid_pixels', 0) >= train_min_valid]
    parts['train'] = full or parts['train']          # same fallback as main_yieldsat_image.py
    return {k: len(v) for k, v in parts.items()}


def expand_runs(suite, table, artifact_root):
    """All runs of the suite, with fold manifests created where missing."""
    runs = []
    tiles = None
    if any(e['model'] == 'image' for e in suite['experiments']):
        if not suite['data'].get('image_root') or '$' in suite['data']['image_root']:
            sys.exit('image suites need data.image_root (tile counts are planned from it)')
        tiles = _image_tiles(suite)
    if suite.get('pretrain'):
        runs += _make_pretrain_runs(suite, artifact_root)
    for exp in suite['experiments']:
        model = exp['model']
        if exp.get('donors'):
            for donor in exp['donors']:
                for inputs in exp['inputs']:
                    for seed in exp.get('seeds', [0]):
                        runs.append(_make_donor_run(suite, exp, donor, inputs, seed, tiles))
            continue
        for protocol in exp['protocols']:
            for policy_name in exp.get('policies', ['paper']):
                group, policy = POLICY[policy_name]
                k = exp.get('k', 10)
                region = exp.get('loro_region', 'farm') if protocol == 'loro' else 'farm'
                for pair in exp.get('pairs', suite['pairs']):
                    prefix = fold_prefix(pair, protocol, group, policy, exp.get('fold_seed', 0), k,
                                         region=region)
                    index = Path(artifact_root) / 'splits' / '{}.folds.json'.format(prefix)
                    if not index.exists():
                        save_folds(make_paper_folds(table, pair, protocol, k=k, group=group,
                                                    policy=policy, seed=exp.get('fold_seed', 0),
                                                    val_frac=exp.get('val_frac', 0.1),
                                                    region=region),
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
                                run = _make_run(suite, exp, model, protocol, policy_name, group,
                                                policy, k, inputs, seed, pair, country, crop,
                                                i, split_name, split, region)
                                if model == 'image':
                                    n = _tile_counts(tiles, split, crop, exp.get('train_min_valid', 0))
                                    run['n_train'], run['n_test'] = n['train'], n['test']
                                    run['tiles'] = n
                                    if not n['train'] or not n['test']:
                                        # held-out fields without a qualifying 64x64
                                        # window: no image metric possible
                                        suite.setdefault('unrunnable', []).append(
                                            {'name': run['name'], 'tiles': n})
                                        continue
                                runs.append(run)
    return runs


def _make_run(suite, exp, model, protocol, policy_name, group, policy, k, inputs, seed, pair,
              country, crop, fold, split_name, split, region='farm'):
    proto = {'cv': 'cv{}'.format(k), 'loro': 'loro', 'loyo': 'loyo'}[protocol]
    region_tag = 'province' if protocol == 'loro' and region == 'province' else 'na'
    exp_group = '{}_{}_{}_s{}'.format(proto, group if protocol == 'cv' else region_tag, policy,
                                      exp.get('fold_seed', 0))
    if region_tag == 'province':
        proto = 'loro-province'           # distinct run names/ids from farm-level LORO
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
    if model == 'image' and exp.get('train_min_valid'):
        args = _override(args, ['--train_min_valid', str(exp['train_min_valid'])])
    donor = exp.get('warm_start', {}).get(pair) if model == 'image' else None
    init_ckpt = None
    if donor:
        root = suite['data'].get('donor_root')
        if not root:
            sys.exit('warm_start needs data.donor_root (results root of the donor plan)')
        init_ckpt = str(Path(root) / _donor_rel(donor, inputs, seed) / 'checkpoint_best.pth')
        lr = float(exp.get('budget', {}).get('lr', IMAGE_DEFAULTS['lr'])) * WARM_START_LR_SCALE
        args = _override(args, ['--init_ckpt', init_ckpt, '--lr', '{:g}'.format(lr)])
    if exp.get('init_from'):
        # knowledge pretraining DEV arms: start from the sensor checkpoint of the
        # fold's pretraining unit (spec/yieldsat-point-knowledge-pretraining.md §6)
        init_ckpt = pretrain_ckpt(suite, exp['init_from'], split_name, artifact_root=suite['data']['artifact_root'])
        args = _override(args, ['--init_sensor_ckpt', init_ckpt])
    args = _override(args, [str(a) for a in exp.get('extra_args', [])])
    n_train = split['summary']['train']['rows']
    n_test = split['summary']['test']['rows']
    return {'run_id': run_id, 'name': readable, 'rel_path': str(rel), 'pair': pair,
            'country': country, 'crop': crop, 'experiment': tag, 'model': model,
            'protocol': proto, 'policy': policy_name, 'inputs': inputs, 'seed': seed,
            'fold': fold, 'split': split_name, 'n_train': n_train, 'n_test': n_test,
            'budget': exp.get('budget', {}), 'args': args, 'donor': donor, 'init_ckpt': init_ckpt,
            'job_group': '{}|{}'.format(pair, exp['init_from']) if exp.get('init_from') else pair,
            'wandb_group': '{}|{}|{}|{}|{}|{}'.format(suite['suite'], tag, proto, policy_name,
                                                      inputs, pair),
            'wandb_tags': [suite['suite'], tag, proto, policy_name, inputs, pair, 'seed{}'.format(seed)]}


def _pretrain_rel(arm, unit_manifest):
    return 'pretrain/{}/{}'.format(arm, unit_manifest)


def pretrain_ckpt(suite, arm, split_name, artifact_root):
    """Sensor checkpoint of the pretraining unit that serves DEV fold ``split_name``."""
    pt = suite['pretrain']
    if arm not in pt['arms']:
        sys.exit('init_from {} is not a pretraining arm ({})'.format(arm, sorted(pt['arms'])))
    index = json.loads((Path(artifact_root) / 'splits' / '{}.json'.format(pt['unit_index'])).read_text())
    unit = index['fold_to_unit'].get(split_name)
    if unit is None:
        sys.exit('fold {} has no pretraining unit in {}'.format(split_name, pt['unit_index']))
    return str(Path(pt['results_root']) / _pretrain_rel(arm, unit) / 'sensor_checkpoint.pth')


def _make_pretrain_runs(suite, artifact_root):
    """One yield-free pretraining run per (arm, unit) over all countries; the
    sensor checkpoint, knowledge reference and heads are kept in the results
    root for the fine-tuning arms and the diagnostics."""
    pt = suite['pretrain']
    if not pt.get('results_root') or '$' in str(pt['results_root']):
        sys.exit('pretrain.results_root must be the pools\' results root')
    index = json.loads((Path(artifact_root) / 'splits' / '{}.json'.format(pt['unit_index'])).read_text())
    countries = list(pt.get('countries', ['Argentina', 'Brazil', 'Germany', 'Uruguay']))
    runs = []
    for arm, spec in pt['arms'].items():
        spec = spec if isinstance(spec, dict) else {'args': spec}
        budget = dict(pt.get('budget', {}))
        factor = float(spec.get('steps_factor', 1.0))
        if factor != 1.0:
            budget['epochs'] = int(round(budget.get('epochs', 20) * factor))
        knowledge = '--knowledge' in spec.get('args', [])
        for unit in sorted(index['units'].values(), key=lambda u: u['manifest']):
            readable = '{}|pretrain|{}|{}'.format(suite['suite'], arm, unit['manifest'])
            args = ['--data_contract', 'yieldsat_preprocessed_v1', '--mode', 'pretrain',
                    '--countries', *countries, '--split', unit['manifest'], '--norm_pooling', 'per_country',
                    '--seed', str(pt.get('seed', 0))]
            args += INPUTS[pt.get('inputs', 's2_adm')] + MODELS['ours']
            args += ['--fusion', pt.get('fusion', 'perceiver_summary')]
            args = _override(args, _budget_args(budget))
            args = _override(args, [str(a) for a in spec.get('args', [])])
            if pt.get('diagnostics', True):
                args.append('--pretrain_diagnostics')
            runs.append({'run_id': hashlib.sha1(readable.encode()).hexdigest()[:12], 'name': readable,
                         'rel_path': _pretrain_rel(arm, unit['manifest']), 'pair': 'pretrain',
                         'job_group': 'pretrain|{}'.format(arm),
                         'country': countries[0], 'countries': countries, 'crop': 'all',
                         'experiment': 'pretrain-' + arm, 'model': 'ours',
                         'throughput_key': 'pretrain_knowledge' if knowledge else 'pretrain_ssl',
                         'protocol': 'pretrain', 'policy': 'na', 'inputs': pt.get('inputs', 's2_adm'),
                         'seed': pt.get('seed', 0), 'fold': 0, 'split': unit['manifest'],
                         'n_train': 0, 'n_test': 0, 'budget': budget, 'args': args, 'knowledge': knowledge,
                         'keep_files': ['sensor_checkpoint.pth', 'knowledge_reference.json',
                                        'pretrainer_heads.pth', 'diagnostics.json',
                                        'diagnostics_error.txt'],
                         'wandb_group': '{}|pretrain|{}'.format(suite['suite'], arm),
                         'wandb_tags': [suite['suite'], 'pretrain', arm, unit['manifest']]})
    return runs


def _donor_rel(name, inputs, seed):
    return 'donors/{}/{}/seed{}'.format(name, inputs, seed)


def _make_donor_run(suite, exp, donor, inputs, seed, tiles):
    """A warm-start donor: all tiles of ``countries`` (optionally ``crops``),
    10% season validation, no test; keeps its checkpoint."""
    name, countries, crops = donor['name'], list(donor['countries']), donor.get('crops')
    readable = '{}|{}|donor|{}|{}|seed{}'.format(suite['suite'], exp['name'], name, inputs, seed)
    args = ['--data_contract', 'yieldsat_preprocessed_v1', '--donor', '--countries', *countries,
            '--seed', str(seed)] + (['--crops', *crops] if crops else [])
    args += INPUTS[inputs] + MODELS['image']
    args = _override(args, _budget_args(exp.get('budget', {})))
    args = _override(args, [str(a) for a in exp.get('extra_args', [])])
    n = sum(1 for t in tiles if t['country'] in countries and (not crops or t['crop'] in crops)
            and t.get('valid_pixels', 0) >= exp.get('train_min_valid', 0))
    if exp.get('train_min_valid'):
        args = _override(args, ['--train_min_valid', str(exp['train_min_valid'])])
    return {'run_id': hashlib.sha1(readable.encode()).hexdigest()[:12], 'name': readable,
            'rel_path': _donor_rel(name, inputs, seed), 'pair': 'donor-' + name,
            'country': countries[0], 'countries': countries, 'crop': ','.join(crops or ['all']),
            'experiment': exp['name'], 'model': 'image', 'protocol': 'donor', 'policy': 'na',
            'inputs': inputs, 'seed': seed, 'fold': 0, 'split': None,
            'n_train': int(round(0.9 * n)), 'n_test': 0, 'budget': exp.get('budget', {}),
            'args': args, 'keep_files': ['checkpoint_best.pth'],
            'wandb_group': '{}|{}|donor|{}|{}'.format(suite['suite'], exp['name'], name, inputs),
            'wandb_tags': [suite['suite'], exp['name'], 'donor', name, inputs, 'seed{}'.format(seed)]}


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
    if model == 'image':                  # n_train in tiles; test/val inference is cheap
        b = dict(IMAGE_DEFAULTS, **b)
        steps = b.get('steps_per_epoch', 0) or max(b['min_steps_per_epoch'],
                                                   math.ceil(run['n_train'] / b['batch_size']))
        if b.get('max_steps_per_epoch', 0) > 0:
            steps = min(steps, b['max_steps_per_epoch'])
        return 30.0 + b['epochs'] * (steps * b['batch_size'] / (REF_THROUGHPUT['image'] *
                                                              _factor(speed_factor, 'image')) + 2.0)
    batch = b.get('batch_size', 1028 if model == 'paper_lstm' else 512)
    epochs = b.get('epochs', 15 if model == 'paper_lstm' else 20)
    steps = b.get('steps_per_epoch', 0 if model == 'paper_lstm' else 500)
    if steps <= 0:
        steps = max(b.get('min_steps_per_epoch', 1), math.ceil(run['n_train'] / batch))
        if b.get('max_steps_per_epoch', 0) > 0:
            steps = min(steps, b['max_steps_per_epoch'])
    key = run.get('throughput_key', model)
    train = epochs * steps * batch / (REF_THROUGHPUT[key] * _factor(speed_factor, key))
    val = epochs * 2.0                       # ~500 cells per validation field
    test = run['n_test'] / (REF_TEST_ROWS_PER_S * _factor(speed_factor, 'test')) * (
        2 if '--eval_drop_stream' in run['args'] else 1)
    return REF_STARTUP_S + train + val + test


STAGE_GB = {'Argentina': 11.0, 'Brazil': 8.7, 'Germany': 1.2, 'Uruguay': 4.4}   # cache + index
IMAGE_STAGE_GB = {'Argentina': 1.2, 'Brazil': 0.95, 'Germany': 0.08, 'Uruguay': 0.25}  # read: tiles + DINO


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
    group = lambda r: r.get('job_group', r['pair'])  # noqa: E731
    pairs = list(suite['pairs']) + sorted({group(r) for r in runs} - set(suite['pairs']))
    for pair in pairs:
        pr = sorted([r for r in runs if group(r) == pair], key=lambda r: -r['est_seconds'])
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
        if pr[0]['model'] == 'image':
            stage_gb = sum(IMAGE_STAGE_GB.get(c, 1.0) for c in pr[0].get('countries', [pr[0]['country']]))
        else:
            stage_gb = sum(STAGE_GB.get(c, 5.0) for c in pr[0].get('countries', [pr[0]['country']]))
        stage_h = stage_gb * 1024 / stage_rate / 3600
        for i, shard in enumerate(shards):
            jobs.append({'job_index': len(jobs), 'pair': pair, 'shard': i, 'n_shards': n,
                         'country': shard[0]['country'],
                         'countries': shard[0].get('countries', [shard[0]['country']]),
                         'model': shard[0]['model'],
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

def check_warm_starts(suite, runs, suite_path=None):
    """Every warm-started run needs a planned donor run (same donor, inputs
    and seed): in this suite, or in ``donor_suite`` (path relative to the
    repository). A missing donor would make the target job wait forever."""
    have = {(r['pair'][len('donor-'):], r['inputs'], r['seed']) for r in runs if r['protocol'] == 'donor'}
    if suite.get('donor_suite'):
        other = load_suite(suite['donor_suite'])
        for e in other['experiments']:
            for d in e.get('donors', []):
                for inputs in e['inputs']:
                    for seed in e.get('seeds', [0]):
                        have.add((d['name'], inputs, seed))
    missing = sorted({(r['donor'], r['inputs'], r['seed']) for r in runs if r.get('donor')} - have)
    if missing:
        sys.exit('warm starts without a planned donor run (donor, inputs, seed): {}'.format(missing))


def cmd_plan(a):
    suite = load_suite(a.suite)
    data = suite['data']
    for key in ('source_root', 'artifact_root'):
        if not data.get(key) or '$' in data[key]:
            sys.exit('suite data.{} is not set (export the env var it references)'.format(key))
    countries = sorted({parse_pair(p)[0] for p in suite['pairs']}
                       | {c for e in suite['experiments'] for d in e.get('donors', []) for c in d['countries']})
    needs_geo = any(pol == 'strict' for e in suite['experiments'] for pol in e.get('policies', []))
    table = load_field_table(data['artifact_root'], data['source_root'], countries,
                             require_geometry=needs_geo)
    runs = expand_runs(suite, table, data['artifact_root'])
    check_warm_starts(suite, runs)
    jobs = make_jobs(suite, runs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'state').mkdir(exist_ok=True)
    if suite.get('unrunnable'):
        print('{} runs left out: no train or test tiles in their fold (listed in plan.json '
              'suite.unrunnable)'.format(len(suite['unrunnable'])))
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
    for key, env in (('source_root', 'YIELDSAT_SOURCE_ROOT'), ('artifact_root', 'YIELDSAT_ARTIFACT_ROOT'),
                     ('image_root', 'YIELDSAT_IMAGE_ROOT')):
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
    countries = job.get('countries', [job['country']])
    t0 = time.time()
    rels = []
    nbr = any('--neighbourhood' in runs[r]['args'] for r in job['run_ids'])
    for c in countries:
        rels += ['index/{}'.format(c), 'cache/{}'.format(c)]
        if nbr and (src / 'neighbourhood' / c / 'neighbourhood_manifest.json').exists():
            rels.append('neighbourhood/{}'.format(c))          # S6 stream (round 2), only when used
    if any(runs[r].get('knowledge') for r in job['run_ids']):
        # knowledge pretraining: raw concept indices + frozen text vectors
        for c in countries:
            (dst / 'knowledge' / c).mkdir(parents=True, exist_ok=True)
            for f in (src / 'knowledge' / c).glob('concept_raw_c*.npz'):
                _copy_once(f, dst / 'knowledge' / c / f.name)
        (dst / 'knowledge' / 'text_clip_b32').mkdir(parents=True, exist_ok=True)
        for f in (src / 'knowledge' / 'text_clip_b32').iterdir():
            shutil.copy2(f, dst / 'knowledge' / 'text_clip_b32' / f.name)
    manifests = {'cache': 'cache_manifest.json', 'index': 'index_manifest.json',
                 'neighbourhood': 'neighbourhood_manifest.json'}
    for rel in rels:
        target = dst / rel
        if (target / manifests[rel.split('/')[0]]).exists():
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
    print('staged {} to {} in {:.0f} s'.format(','.join(countries), dst, time.time() - t0), flush=True)
    return dst


def _copy_once(src, dst):
    """Copy unless an identical-size copy exists; atomic via a .partial name."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    tmp = dst.with_name(dst.name + '.partial')
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _stage_tiles_uncompressed(src, dst):
    """Copy an image HDF5 tile by tile without LZF compression: loading a
    tile drops from ~20 to ~8 ms (decompression dominated), for ~10x the
    disk (Argentina ~10 GB on pod scratch). Atomic via a .partial name."""
    import h5py
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.partial')
    with h5py.File(src, 'r') as fs, h5py.File(tmp, 'w') as fd:
        for k, v in fs.attrs.items():
            fd.attrs[k] = v
        for k, ds in fs.items():
            out = fd.create_dataset(k, ds.shape, ds.dtype, chunks=ds.chunks)
            for a, v in ds.attrs.items():
                out.attrs[a] = v
            for i in range(ds.shape[0]):
                out[i] = ds[i]
    os.replace(tmp, dst)


def _stage_tiles_sparse(src, dst, keep):
    """Uncompressed copy holding only the tiles in ``keep`` (patch indices);
    unwritten HDF5 chunks use no disk, so indices stay valid (plan v2: full-
    coverage builds are ~4x larger than v1). Atomic via a .partial name."""
    import h5py
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.partial')
    with h5py.File(src, 'r') as fs, h5py.File(tmp, 'w') as fd:
        for k, v in fs.attrs.items():
            fd.attrs[k] = v
        for k, ds in fs.items():
            out = fd.create_dataset(k, ds.shape, ds.dtype, chunks=ds.chunks)
            for a, v in ds.attrs.items():
                out.attrs[a] = v
            for i in sorted(keep):
                out[i] = ds[i]
    os.replace(tmp, dst)


def _job_crops(job, runs):
    crops = set()
    for rid in job['run_ids']:
        crop = runs[rid]['crop']
        if crop == 'all':
            return None
        crops.update(crop.split(','))
    return crops


def stage_image_local(plan, job, runs, local_root):
    """Image jobs: copy the countries' tiles, DINO features, index fields and
    the runs' splits to local disk. Returns (artifact_root, image_root)."""
    from models_yieldsat_image import DINO_REVISION, preprocessing_hash
    from yieldsat_image_dino_cache import cache_tag
    data = plan['suite']['data']
    src_art, src_img = Path(data['artifact_root']), Path(data['image_root'])
    dst_art, dst_img = Path(local_root) / 'artifacts', Path(local_root) / 'images'
    mode = data.get('stage_mode', 'uncompressed')      # v1 default; v2 suites: sparse
    if mode == 'sparse':
        # one staging root per job (a pod runs jobs one at a time): drop the
        # previous job's tiles, keep only this job's crops, uncompressed;
        # multi-crop donor jobs copy the compressed file instead
        base = Path(local_root) / 'images_by_job'
        dst_img = base / job['pair']
        if base.exists():
            for d in base.iterdir():
                if d != dst_img:
                    shutil.rmtree(d, ignore_errors=True)
        crops = _job_crops(job, runs)
    dino = Path('dino_cache') / cache_tag(data.get('dino_revision', DINO_REVISION), preprocessing_hash())
    t0 = time.time()
    _copy_once(src_img / 'manifest.json', dst_img / 'manifest.json')
    _copy_once(src_img / dino / 'manifest.json', dst_img / dino / 'manifest.json')
    for c in job.get('countries', [job['country']]):
        for rel in (Path(c) / 'patches.jsonl', dino / '{}.h5'.format(c)):
            _copy_once(src_img / rel, dst_img / rel)
        if mode == 'sparse' and crops is not None:
            keep = [json.loads(line)['patch_index'] for line in open(src_img / c / 'patches.jsonl')
                    if json.loads(line)['crop'] in crops]
            _stage_tiles_sparse(src_img / c / 'images.h5', dst_img / c / 'images.h5', keep)
        elif mode == 'uncompressed' and data.get('stage_uncompressed', True):
            _stage_tiles_uncompressed(src_img / c / 'images.h5', dst_img / c / 'images.h5')
        else:
            _copy_once(src_img / c / 'images.h5', dst_img / c / 'images.h5')
        _copy_once(src_art / 'index' / c / 'fields.json', dst_art / 'index' / c / 'fields.json')
    for rid in job['run_ids']:
        name = runs[rid]['split']
        if name:
            _copy_once(src_art / 'splits' / '{}.json'.format(name), dst_art / 'splits' / '{}.json'.format(name))
    print('staged image data for {} to {} in {:.0f} s'.format(job.get('countries'), local_root,
                                                               time.time() - t0), flush=True)
    return dst_art, dst_img


def _state_paths(plan_dir, run_id):
    state = Path(plan_dir) / 'state'
    return state / '{}.done.json'.format(run_id), state / '{}.failed.json'.format(run_id)


def _launch(run, plan, artifact_root, work_dir, use_wandb, image_root=None):
    out = Path(work_dir) / run['rel_path']
    out.mkdir(parents=True, exist_ok=True)
    wb = plan['suite']['wandb']
    cmd = [sys.executable, SCRIPTS.get(run['model'], 'main_yieldsat_finetune.py'), '--source_root',
           plan['suite']['data']['source_root'], '--artifact_root', str(artifact_root),
           '--output_dir', str(out), '--num_workers', str(plan['suite'].get('num_workers', 4))]
    if run['model'] == 'image':
        cmd += ['--image_root', str(image_root or plan['suite']['data']['image_root'])]
        if plan['suite']['data'].get('dino_revision'):
            cmd += ['--dino_revision', plan['suite']['data']['dino_revision']]
    cmd += run['args']
    if use_wandb:
        cmd += ['--wandb', '--wandb_no_model_artifacts', '--wandb_project', str(wb.get('project', 'yieldsat')),
                '--wandb_group', run['wandb_group'], '--wandb_name', run['name'],
                '--wandb_job_type', 'pretrain' if run['protocol'] == 'pretrain' else 'finetune',
                '--wandb_tags', *run['wandb_tags'],
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
            # checkpoints are kept on the PVC, not uploaded to W&B (storage, 2026-10-03)
            for f in ('report.json', 'test_predictions.npz', 'normalizer.json', 'train.log',
                      'checkpoint_best.pth', *run.get('keep_files', [])):
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


class GpuMonitor:
    """Samples ``nvidia-smi`` every ``interval`` s in a background process
    (utilization %, memory used/total MiB) into ``path``."""

    def __init__(self, path, interval=10):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = None
        if shutil.which('nvidia-smi'):
            self.fh = open(self.path, 'w')
            self.proc = subprocess.Popen(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                 '--format=csv,noheader,nounits', '-l', str(interval)],
                stdout=self.fh, stderr=subprocess.DEVNULL)

    def samples(self):
        if self.proc is None or not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            try:
                u, used, total = (float(x) for x in line.split(',')[:3])
                out.append((u, used, total))
            except ValueError:
                continue
        return out

    def recent_util(self, n):
        s = self.samples()[-n:]
        return sum(x[0] for x in s) / len(s) if s else None

    def memory_fraction(self):
        s = self.samples()
        return s[-1][1] / s[-1][2] if s else 0.0

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            self.proc.wait(timeout=10)
            self.fh.close()


def run_job(plan_dir, job_index, local_root=None, work_dir=None, results_root=None,
            runs_per_gpu=None, use_wandb=True, keep_local=False, dry_run=False,
            max_runs_per_gpu=None, num_workers=None, heartbeat=None):
    """Run one planned job. Concurrency starts at ``runs_per_gpu`` and, if the
    suite sets ``target_gpu_util`` (%), grows by one run whenever the mean GPU
    utilization over the last ``adapt_window`` samples is below target, up to
    ``max_runs_per_gpu`` and while GPU memory stays below 85%. Utilization is
    recorded per job in ``state/job_<index>.gpu.json``."""
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
    image_root = plan['suite']['data'].get('image_root')
    if local_root and job.get('model') == 'image':
        artifact_root, image_root = stage_image_local(plan, job, runs, local_root)
    elif local_root:
        artifact_root = stage_local(plan, job, runs, local_root)
    work_dir = work_dir or (Path(local_root) / 'work' if local_root else Path(plan_dir) / 'work')
    suite = plan['suite']
    if num_workers is not None:                  # pod-shape overrides (worker pools)
        suite['num_workers'] = num_workers
    k = runs_per_gpu or suite['runs_per_gpu']
    target = suite.get('target_gpu_util')
    k_max = max(k, max_runs_per_gpu or suite.get('max_runs_per_gpu', k))
    window = int(suite.get('adapt_window', 6))              # samples of 10 s
    monitor = GpuMonitor(Path(work_dir) / 'gpu_util_job{}.csv'.format(job_index))
    last_adapt = time.time()
    t_start = time.time()
    pending, running, failures = list(todo), [], 0
    fast_fail_streak = 0
    trace = []
    try:
        while pending or running:
            while pending and len(running) < k:
                r = pending.pop(0)
                proc, log, out = _launch(r, plan, artifact_root, work_dir, use_wandb, image_root)
                running.append((r, proc, log, out, time.time()))
            time.sleep(5)
            if heartbeat is not None:
                heartbeat()
            for item in list(running):
                r, proc, log, out, t0 = item
                if proc.poll() is None:
                    continue
                running.remove(item)
                ok = _finish(r, proc, log, out, plan_dir, results_root, keep_local)
                failures += 0 if ok else 1
                elapsed = time.time() - t0
                print('{} {} ({:.0f} s)'.format('ok' if ok else 'FAILED', r['name'], elapsed),
                      flush=True)
                # circuit breaker: consecutive fast failures mean a broken GPU/pod,
                # not a bad run; stop instead of draining the queue
                fast_fail_streak = 0 if ok or elapsed > 120 else fast_fail_streak + 1
                if fast_fail_streak >= 3:
                    for item2 in running:
                        item2[1].kill()
                    monitor.stop()
                    raise GpuFault('3 consecutive runs failed within 2 min each (last: {})'.format(r['name']))
            # adaptive concurrency: only after runs are past start-up
            if target and pending and k < k_max and time.time() - last_adapt > 10 * window + 60:
                util = monitor.recent_util(window)
                if util is not None and util < target and monitor.memory_fraction() < 0.85:
                    k += 1
                    last_adapt = time.time()
                    print('GPU util {:.0f}% < {}%: concurrency -> {}'.format(util, target, k), flush=True)
                    trace.append({'t': round(time.time() - t_start), 'util': round(util, 1), 'k': k})
    finally:
        monitor.stop()
    samples = monitor.samples()
    if samples:
        # ignore the first minute (staging/start-up) when the job is long enough
        steady = samples[6:] if len(samples) > 12 else samples
        util = [x[0] for x in steady]
        summary = {'job_index': job_index, 'pair': job['pair'], 'samples': len(util),
                   'mean_util': round(sum(util) / len(util), 1),
                   'frac_ge_80': round(sum(u >= 80 for u in util) / len(util), 3),
                   'max_mem_frac': round(max(x[1] / x[2] for x in samples), 3),
                   'final_concurrency': k, 'adaptations': trace, 'host': os.uname().nodename,
                   'wall_seconds': round(time.time() - t_start)}
        (Path(plan_dir) / 'state' / 'job_{}.gpu.json'.format(job_index)).write_text(json.dumps(summary))
        print('GPU utilization: mean {:.0f}%, {:.0%} of samples >= 80% (concurrency {})'.format(
            summary['mean_util'], summary['frac_ge_80'], k), flush=True)
    return 1 if failures else 0


def cmd_run(a):
    idx = a.job_index
    if idx is None:
        idx = int(os.environ.get('JOB_COMPLETION_INDEX', os.environ.get('JOB_INDEX', '-1')))
    if idx < 0:
        sys.exit('no job index: pass --job_index or set JOB_COMPLETION_INDEX')
    sys.exit(run_job(a.plan, idx, a.local_root, a.work_dir, a.results_root, a.runs_per_gpu,
                     not a.no_wandb, a.keep_local, a.dry_run))


# ---- GPU health ---------------------------------------------------------------------

class GpuFault(RuntimeError):
    pass


def gpu_health_check():
    """One real forward/backward pass of the point model on the GPU. A GPU
    that fails it (e.g. 'illegal memory access' on a faulty device) must not
    take jobs. Runs in a subprocess so a CUDA fault cannot poison this one."""
    code = (
        "import numpy\n"          # before torch: avoids the MKL/libgomp load-order error
        "import torch\n"
        "from dataset.yieldsat_dataset import stream_layout\n"
        "from dataset.yieldsat_schema import STREAMS\n"
        "from models_yieldsat import YieldSATPointModel\n"
        "d = torch.device('cuda'); torch.manual_seed(0)\n"
        "L = stream_layout(list(STREAMS)); m = YieldSATPointModel(L).to(d)\n"
        "B, T = 512, 24\n"
        "b = {'inputs': {}, 'masks': {}, 'available': {}}\n"
        "for n, l in L.items():\n"
        "    C = len(l['out_channels']); sh = (B, T, C) if l['temporal'] else (B, C)\n"
        "    b['inputs'][n] = torch.randn(sh, device=d); b['masks'][n] = torch.rand(sh, device=d) > 0.3\n"
        "    b['available'][n] = (torch.rand(B, device=d) > 0.2).float()\n"
        "b['time_features'] = torch.randn(B, T, 3, device=d); b['time_valid'] = torch.rand(B, T, device=d) > 0.3\n"
        "b['crop'] = torch.randint(0, 4, (B,), device=d); b['target'] = torch.randn(B, device=d)\n"
        "b['target_valid'] = torch.ones(B, dtype=torch.bool, device=d)\n"
        "with torch.autocast('cuda', dtype=torch.bfloat16):\n"
        "    _, loss = m.loss(b)\n"
        "loss.float().backward(); torch.cuda.synchronize()\n"
        "assert torch.isfinite(loss), 'non-finite loss'\n"
        "print('gpu ok', torch.cuda.get_device_name(0))\n")
    try:
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        raise GpuFault('GPU health check timed out')
    if out.returncode != 0:
        raise GpuFault('GPU health check failed: ' + (out.stderr or out.stdout)[-600:])
    return out.stdout.strip()


def record_bad_gpu(plan_dir, reason):
    """Remember a faulty GPU (node + device UUID) on the shared volume."""
    uuid = ''
    try:
        uuid = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        pass
    d = Path(plan_dir) / 'state' / 'bad_gpus'
    d.mkdir(parents=True, exist_ok=True)
    node = os.environ.get('NODE_NAME', os.uname().nodename)
    (d / '{}_{}.json'.format(os.environ.get('HOSTNAME', node), int(time.time()))).write_text(json.dumps(
        {'pod': os.environ.get('HOSTNAME'), 'node': node, 'gpu_uuid': uuid, 'reason': reason[-1000:],
         'time': time.strftime('%Y-%m-%dT%H:%M:%S')}))


# ---- worker pools: pods claim jobs from a shared queue ------------------------------

class JobClaims:
    """Atomic job claims on the shared volume: ``state/claims/job_<i>/`` is
    created with ``mkdir`` (atomic on CephFS/NFS); the owner refreshes
    ``heartbeat`` and writes ``done`` when finished. A claim whose heartbeat is
    older than ``stale_minutes`` and has no ``done`` is taken over by exactly
    one pod (atomic ``mkdir`` of ``takeover_<n>``); runs already finished are
    skipped through their completion markers."""

    def __init__(self, plan_dir, owner, stale_minutes=30):
        self.root = Path(plan_dir) / 'state' / 'claims'
        self.root.mkdir(parents=True, exist_ok=True)
        self.owner = owner
        self.stale = stale_minutes * 60
        self._last_beat = 0.0
        self.current = None

    def _dir(self, idx):
        return self.root / 'job_{}'.format(idx)

    MAX_ATTEMPTS = 3

    def is_done(self, idx):
        """Finished for good: succeeded, or failed MAX_ATTEMPTS times."""
        return (self._dir(idx) / 'done').exists()

    def try_claim(self, idx):
        d = self._dir(idx)
        try:
            d.mkdir()
        except FileExistsError:
            if (d / 'done').exists():
                return False
            hb = d / 'heartbeat'
            age = time.time() - (hb.stat().st_mtime if hb.exists() else d.stat().st_mtime)
            if age < self.stale:
                return False
            n = len(list(d.glob('takeover_*')))
            try:
                (d / 'takeover_{}'.format(n)).mkdir()
            except FileExistsError:
                return False
        (d / 'owner').write_text('{} {}'.format(self.owner, time.strftime('%Y-%m-%dT%H:%M:%S')))
        self.current = idx
        self._last_beat = 0.0
        self.beat()
        return True

    def beat(self):
        if self.current is not None and time.time() - self._last_beat > 60:
            (self._dir(self.current) / 'heartbeat').write_text(str(time.time()))
            self._last_beat = time.time()

    def finish(self, idx, rc):
        """rc == 0: mark done. Otherwise count the attempt and release the
        claim so another pod retries the job's unfinished runs; after
        MAX_ATTEMPTS the job is closed with its failures recorded."""
        d = self._dir(idx)
        attempts_file = self.root / 'job_{}.attempts'.format(idx)
        attempts = int(attempts_file.read_text()) + 1 if attempts_file.exists() else 1
        if rc == 0 or attempts >= self.MAX_ATTEMPTS:
            (d / 'done').write_text(json.dumps({'owner': self.owner, 'rc': rc, 'attempts': attempts,
                                                'finished': time.strftime('%Y-%m-%dT%H:%M:%S')}))
        else:
            attempts_file.write_text(str(attempts))
            self.release(idx)
        self.current = None

    def release(self, idx):
        """Give a claim back to the queue (e.g. this pod's GPU is faulty)."""
        shutil.rmtree(self._dir(idx), ignore_errors=True)
        self.current = None

    def reopen_failed(self):
        """Re-open jobs closed as done with rc != 0 before attempt counting
        existed (one-off repair; returns reopened job indices)."""
        reopened = []
        for d in self.root.glob('job_*'):
            done = d / 'done'
            if d.is_dir() and done.exists():
                info = json.loads(done.read_text())
                if info.get('rc', 0) != 0 and 'attempts' not in info:
                    shutil.rmtree(d, ignore_errors=True)
                    reopened.append(int(d.name.split('_')[1]))
        return sorted(reopened)


def _job_matches(job, runs, only):
    """``only`` = {run field: value}; every run of the job must match."""
    return all(all(str(runs[r].get(k)) == v for k, v in only.items()) for r in job['run_ids'])


def _job_ready(job, runs):
    """A warm-started job waits until its donor checkpoints exist."""
    return all(not runs[r].get('init_ckpt') or Path(runs[r]['init_ckpt']).exists()
               for r in job['run_ids'])


def run_pool(plan_dir, owner, only=None, donor_wait_hours=24.0, **kw):
    """Claim and run jobs (longest first) until the queue is exhausted.
    ``only`` restricts the pool to jobs whose runs all match, e.g.
    ``{'protocol': 'loro-province'}``. Jobs whose donor checkpoints do not
    exist yet are skipped; when only such jobs remain, the worker waits for
    them (up to ``donor_wait_hours``)."""
    plan, _ = _load_plan(plan_dir)
    claims = JobClaims(plan_dir, owner, kw.pop('stale_minutes', 30))
    check_gpu = kw.pop('health_check', True)
    reopened = claims.reopen_failed()
    if reopened:
        print('re-opened {} jobs closed with failures: {}'.format(len(reopened), reopened), flush=True)
    rc, ran = 0, 0
    wait_start = None
    while True:
        # re-read the plan between jobs: `extend` may have appended jobs
        plan, plan_runs = _load_plan(plan_dir)
        order = sorted(range(plan['n_jobs']), key=lambda i: -plan['jobs'][i]['est_hours'])
        if only:
            order = [i for i in order if _job_matches(plan['jobs'][i], plan_runs, only)]
        if check_gpu:
            try:
                print(gpu_health_check(), flush=True)
            except GpuFault as exc:
                record_bad_gpu(plan_dir, str(exc))
                print('GPU FAULT, leaving the pool: {}'.format(exc), flush=True)
                return 3
        claimed, waiting = None, 0
        for idx in order:
            if claims.is_done(idx):
                continue
            if not _job_ready(plan['jobs'][idx], plan_runs):
                waiting += 1
                continue
            if claims.try_claim(idx):
                claimed = idx
                break
        if claimed is None:
            if not waiting:
                break
            wait_start = wait_start or time.time()
            if time.time() - wait_start > donor_wait_hours * 3600:
                print('{} jobs still wait for donor checkpoints; giving up'.format(waiting), flush=True)
                return rc | 4
            print('{} jobs wait for donor checkpoints; sleeping 5 min'.format(waiting), flush=True)
            time.sleep(300)
            continue
        wait_start = None
        try:
            code = run_job(plan_dir, claimed, heartbeat=claims.beat, **kw)
        except GpuFault as exc:
            claims.release(claimed)
            record_bad_gpu(plan_dir, str(exc))
            print('GPU FAULT during job {}, released it and leaving the pool: {}'.format(claimed, exc),
                  flush=True)
            return 3
        claims.finish(claimed, code)
        rc |= code
        ran += 1
    print('pool worker {} finished: {} jobs run, queue empty'.format(owner, ran), flush=True)
    return rc


def cmd_pool(a):
    owner = os.environ.get('HOSTNAME', os.uname().nodename)
    only = dict(kv.split('=', 1) for kv in (a.only or []))
    sys.exit(run_pool(a.plan, owner, only=only or None, local_root=a.local_root, work_dir=a.work_dir,
                      results_root=a.results_root, runs_per_gpu=a.runs_per_gpu,
                      max_runs_per_gpu=a.max_runs_per_gpu, num_workers=a.num_workers,
                      use_wandb=not a.no_wandb, keep_local=a.keep_local,
                      stale_minutes=a.stale_minutes))


def cmd_worker(a):
    plan, _ = _load_plan(a.plan)
    rc = 0
    for idx in range(a.worker_index, plan['n_jobs'], a.num_workers):
        rc |= run_job(a.plan, idx, a.local_root, a.work_dir, a.results_root, a.runs_per_gpu,
                      not a.no_wandb, a.keep_local, a.dry_run)
    sys.exit(rc)


def _write_atomic(path, text):
    tmp = Path(str(path) + '.tmp{}'.format(os.getpid()))
    tmp.write_text(text)
    os.replace(tmp, path)


def extend_plan(plan_dir, suite_path):
    """Append the runs of ``suite_path`` that the plan does not contain yet
    (same suite name, so W&B names/tags stay consistent) as new jobs. Pods
    pick them up between jobs (pools re-read the plan)."""
    plan, runs = _load_plan(plan_dir)
    suite = load_suite(suite_path)
    if suite['suite'] != plan['suite']['suite']:
        raise SystemExit('extension suite name {} != plan suite {}'.format(suite['suite'], plan['suite']['suite']))
    data = plan['suite']['data']
    countries = sorted({parse_pair(p)[0] for e in suite['experiments']
                        for p in e.get('pairs', suite['pairs'])})
    table = load_field_table(data['artifact_root'], data['source_root'], countries, require_geometry=True)
    new = [r for r in expand_runs(suite, table, data['artifact_root']) if r['run_id'] not in runs]
    if not new:
        print('nothing new to add')
        return []
    jobs = make_jobs(suite, new)
    for j in jobs:
        j['job_index'] += plan['n_jobs']
    plan['jobs'] += jobs
    plan['n_jobs'] = len(plan['jobs'])
    plan['n_runs'] += len(new)
    plan.setdefault('extensions', []).append({'suite_file': str(suite_path), 'runs': len(new),
                                              'jobs': len(jobs), 'time': time.strftime('%Y-%m-%dT%H:%M:%S')})
    with open(Path(plan_dir) / 'runs.jsonl', 'a') as f:
        for r in new:
            f.write(json.dumps(r) + '\n')
    _write_atomic(Path(plan_dir) / 'plan.json', json.dumps(plan, indent=1))
    print('added {} runs in {} jobs ({:.1f} GPU-h); plan now {} runs / {} jobs'.format(
        len(new), len(jobs), sum(j['est_hours'] for j in jobs), plan['n_runs'], plan['n_jobs']))
    return new


def skip_runs(plan_dir, pair_prefix=None, protocol=None, reason=''):
    """Retire runs that are not done yet: a done marker with
    ``skipped: true`` (running pods treat it as done and never start them)."""
    _, runs = _load_plan(plan_dir)
    n = 0
    for r in runs.values():
        if pair_prefix and not r['pair'].startswith(pair_prefix):
            continue
        if protocol and r['protocol'] != protocol:
            continue
        done, _ = _state_paths(plan_dir, r['run_id'])
        if done.exists():
            continue
        done.write_text(json.dumps({'run_id': r['run_id'], 'name': r['name'], 'skipped': True,
                                    'reason': reason, 'time': time.strftime('%Y-%m-%dT%H:%M:%S')}))
        n += 1
    print('skipped {} runs'.format(n))
    return n


def cmd_extend(a):
    extend_plan(a.plan, a.suite)


def cmd_skip(a):
    skip_runs(a.plan, a.pair_prefix, a.protocol, a.reason)


def cmd_status(a):
    plan, runs = _load_plan(a.plan)
    state = Path(a.plan) / 'state'
    done = {p.name.split('.')[0] for p in state.glob('*.done.json')}
    skipped = {p.name.split('.')[0] for p in state.glob('*.done.json')
               if '"skipped": true' in p.read_text()}
    print('skipped (retired) runs: {}'.format(len(skipped)))
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
    gpu = [json.loads(p.read_text()) for p in state.glob('job_*.gpu.json')]
    if gpu:
        mean = sum(g['mean_util'] for g in gpu) / len(gpu)
        low = [g for g in gpu if g['mean_util'] < a.min_util]
        print('GPU utilization over {} finished jobs: mean {:.0f}%; {} job(s) below {}%'.format(
            len(gpu), mean, len(low), a.min_util))
        for g in sorted(low, key=lambda g: g['mean_util'])[:10]:
            print('  job {} {} on {}: {:.0f}% (concurrency {})'.format(
                g['job_index'], g['pair'], g['host'], g['mean_util'], g['final_concurrency']))


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
               name_prefix='smatousi-yieldsat', exclude_nodes=(), image_dir='YieldSAT-Image',
               donor_plan='image_donors'):
    """Indexed Job running every job of a plan on the shared PVC.
    ``image_dir``/``donor_plan`` set the image suites' YIELDSAT_IMAGE_ROOT
    and YIELDSAT_DONOR_ROOT (which override a plan's recorded paths)."""
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
            # pods rejected by nodes with failed GPUs (UnexpectedAdmissionError) count
            # as failures before any work starts; allow generous retries per index
            'backoffLimitPerIndex': 8, 'maxFailedIndexes': max(1, n_jobs // 10),
            'ttlSecondsAfterFinished': 7 * 86400,
            'template': {'spec': {
                'restartPolicy': 'Never',
                'affinity': {'nodeAffinity': {'requiredDuringSchedulingIgnoredDuringExecution': {
                    'nodeSelectorTerms': [{'matchExpressions': [{
                        'key': 'nvidia.com/gpu.product', 'operator': 'In', 'values': list(gpus)}]
                        + ([{'key': 'kubernetes.io/hostname', 'operator': 'NotIn',
                             'values': list(exclude_nodes)}] if exclude_nodes else [])}]}}},
                'containers': [{
                    'name': 'yieldsat', 'image': image, 'imagePullPolicy': 'Always',
                    'command': ['/bin/bash', '-c'], 'args': [script],
                    'env': [
                        {'name': 'YIELDSAT_GIT_REF', 'value': git_ref},
                        {'name': 'YIELDSAT_SUITE', 'value': suite_path},
                        {'name': 'YIELDSAT_PLAN_DIR', 'value': plan_dir},
                        {'name': 'YIELDSAT_SOURCE_ROOT', 'value': base + '/preprocessed'},
                        {'name': 'YIELDSAT_ARTIFACT_ROOT', 'value': base + '/yieldsat_artifacts'},
                        # image suites: tiles + DINO cache, donor checkpoints
                        {'name': 'YIELDSAT_IMAGE_ROOT', 'value': base + '/' + image_dir},
                        {'name': 'YIELDSAT_DONOR_ROOT', 'value': base + '/yieldsat_results/' + donor_plan},
                        {'name': 'WANDB_API_KEY', 'valueFrom': {'secretKeyRef': {
                            'name': secret, 'key': 'WANDB_API_KEY'}}}],
                    'resources': {'limits': dict(res), 'requests': dict(res)},
                    'volumeMounts': [{'name': 'data', 'mountPath': data_mount},
                                     {'name': 'scratch', 'mountPath': '/scratch'},
                                     {'name': 'dshm', 'mountPath': '/dev/shm'}]}],
                'volumes': [{'name': 'data', 'persistentVolumeClaim': {'claimName': pvc}},
                            {'name': 'scratch', 'emptyDir': {'sizeLimit': '40Gi'}},
                            {'name': 'dshm', 'emptyDir': {'medium': 'Memory', 'sizeLimit': '8Gi'}}]}}}}


def render_pool(suite_path, plan_name, pool_name, pods, gpus, cpu, memory_gi, runs_per_gpu,
                max_runs_per_gpu, num_workers, exclude_nodes=(), only=None, **kw):
    """A worker pool: ``pods`` identical pods of one GPU type, each claiming
    jobs from the plan's shared queue until it is empty."""
    m = render_k8s(suite_path, plan_name, pods, pods, kw.pop('image'), kw.pop('pvc'), kw.pop('secret'),
                   gpus, cpu=cpu, memory_gi=memory_gi, exclude_nodes=exclude_nodes, **kw)
    m['metadata']['name'] = '{}-{}'.format(m['metadata']['name'], pool_name)
    spec = m['spec']
    for key in ('completionMode', 'backoffLimitPerIndex', 'maxFailedIndexes'):
        spec.pop(key, None)
    # admission rejections and node losses on faulty nodes count as failures;
    # the rtx3090 pool died at 8 per pod (2026-10-01), so allow many
    spec['backoffLimit'] = pods * 50
    c = spec['template']['spec']['containers'][0]
    c['args'] = [c['args'][0].replace(
        'run_in_pod.sh run --plan', 'run_in_pod.sh pool --plan').replace(
        '--local_root', '--runs_per_gpu {} --max_runs_per_gpu {} --num_workers {}{} --local_root'.format(
            runs_per_gpu, max_runs_per_gpu, num_workers,
            ' --only ' + ' '.join(only) if only else ''))]
    return m


def cmd_pools(a):
    """Render one Job per worker pool (e.g. A10 and RTX 3090 shapes)."""
    docs = []
    for spec in a.pool:
        name, gpu, pods, cpu, mem, k, kmax, workers = spec.split(':')
        docs.append(render_pool(a.suite, a.plan_name, name, int(pods), gpu.split(','), int(cpu),
                                int(mem), int(k), int(kmax), int(workers),
                                exclude_nodes=a.exclude_nodes, only=a.only, image=a.image, pvc=a.pvc,
                                secret=a.secret, git_ref=a.git_ref, image_dir=a.image_dir,
                                donor_plan=a.donor_plan))
    text = '# generated by yieldsat_cluster.py pools; no credentials inside\n' + '---\n'.join(
        yaml.safe_dump(d, sort_keys=False, width=200) for d in docs)
    Path(a.out).write_text(text)
    print('wrote', a.out, 'with', len(docs), 'pools:',
          ', '.join('{} x {}'.format(d['spec']['parallelism'], d['metadata']['name']) for d in docs))


def cmd_k8s(a):
    plan, _ = _load_plan(a.plan)
    manifest = render_k8s(a.suite, a.plan_name or Path(a.plan).name, plan['n_jobs'], a.parallelism,
                          a.image, a.pvc, a.secret, a.gpu, git_ref=a.git_ref, cpu=a.cpu,
                          memory_gi=a.memory_gi, exclude_nodes=a.exclude_nodes)
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
    for name, fn in (('run', cmd_run), ('worker', cmd_worker), ('pool', cmd_pool)):
        s = sub.add_parser(name)
        s.add_argument('--plan', required=True)
        if name == 'run':
            s.add_argument('--job_index', type=int, default=None)
        elif name == 'worker':
            s.add_argument('--worker_index', type=int, required=True)
            s.add_argument('--num_workers', type=int, required=True)
        else:
            s.add_argument('--max_runs_per_gpu', type=int, default=None)
            s.add_argument('--num_workers', type=int, default=None, help='loader workers per run')
            s.add_argument('--stale_minutes', type=float, default=30)
            s.add_argument('--only', nargs='*', default=None,
                           help='claim only jobs whose runs all match field=value (e.g. protocol=loro-province)')
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
    s.add_argument('--exclude_nodes', nargs='*', default=[],
                   help='hostnames to avoid (e.g. nodes with failed GPUs)')
    s.add_argument('--out', default=None)
    s.set_defaults(func=cmd_k8s)
    s = sub.add_parser('pools', help='render worker-pool Jobs sharing the plan queue')
    s.add_argument('--suite', required=True)
    s.add_argument('--plan_name', required=True)
    s.add_argument('--pool', nargs='+', required=True,
                   help='name:GPU[,GPU]:pods:cpu:memGi:runs:max_runs:loader_workers')
    s.add_argument('--image', default='gitlab-registry.nrp-nautilus.io/smatous/yieldsat:latest')
    s.add_argument('--pvc', default='ali-vol-1tera')
    s.add_argument('--secret', default='smatousi-wandb')
    s.add_argument('--git_ref', default='main')
    s.add_argument('--exclude_nodes', nargs='*', default=[])
    s.add_argument('--only', nargs='*', default=None, help='job filter passed to the pool pods')
    s.add_argument('--image_dir', default='YieldSAT-Image', help='image dataset dir under /data/YieldSAT')
    s.add_argument('--donor_plan', default='image_donors', help='donor results dir under yieldsat_results')
    s.add_argument('--out', required=True)
    s.set_defaults(func=cmd_pools)
    s = sub.add_parser('extend', help='append a suite\'s new runs to an existing plan')
    s.add_argument('--plan', required=True)
    s.add_argument('--suite', required=True)
    s.set_defaults(func=cmd_extend)
    s = sub.add_parser('skip', help='retire not-yet-done runs (marked skipped)')
    s.add_argument('--plan', required=True)
    s.add_argument('--pair_prefix', default=None, help='e.g. ARG')
    s.add_argument('--protocol', default=None, help='e.g. loro (farm-level only; province is loro-province)')
    s.add_argument('--reason', default='')
    s.set_defaults(func=cmd_skip)
    s = sub.add_parser('status')
    s.add_argument('--plan', required=True)
    s.add_argument('--min_util', type=float, default=80.0)
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
