# --------------------------------------------------------
# YieldSAT paper-protocol run matrix (PC-06).
#
# For every (protocol, input set, model, country-crop pair, fold) this builds
# the fold manifests if missing, runs main_yieldsat_finetune.py, and
# aggregates each finished experiment (PC-04). Finished folds are skipped, so
# the matrix can be resumed or extended (e.g. more epochs under a new tag).
#
#   python yieldsat_paper_runs.py --protocols cv --pairs GER-R \
#       --models paper_lstm --inputs s2 --concurrency 3
#   python yieldsat_paper_runs.py --protocols cv loro loyo --models ours \
#       --inputs s2 s2_adm --epochs 50 --tag long50
#
# Layout: <runs_root>/paper/<protocol>_<group>_<policy>_s<seed>/<inputs>/
#         <model>[_<tag>]/<pair>/fold<ii>/  (+ aggregate.json per pair)
# --------------------------------------------------------

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dataset.yieldsat_splits import (
    PAPER_PAIRS, load_field_table, make_paper_folds, parse_pair, save_folds,
)
from yieldsat_collect_results import aggregate_folds
from yieldsat_prepare import fold_prefix

INPUTS = {
    # paper "Sentinel-2" rows
    's2': ['--streams', 'yieldsat_s2'],
    # paper "Sentinel-2 + ADM" rows (all auxiliary layers)
    's2_adm': ['--streams', 'yieldsat_s2', 'yieldsat_weather', 'yieldsat_dem',
               'yieldsat_terrain', 'yieldsat_soil'],
}

# Model presets. Paper-protocol runs always use all 24 slots (retrospective).
MODELS = {
    # our point model; fusion chosen with --fusion. Default budget per fold:
    # 20 x 500 steps of 512 cells (override with --epochs/--steps_per_epoch)
    'ours': ['--model', 'yieldsat_point', '--cutoff_mode', 'all_slots',
             '--epochs', '20', '--steps_per_epoch', '500'],
    # the paper's pixel LSTM (PC-05): release-tutorial training (NaN -> -1,
    # raw t/ha target, Adam 1e-3, batch 1028, full passes, 15 epochs) with the
    # file's stats-* input normalization, which reproduces the paper's GER-R
    # CV10 numbers (raw inputs, as in the tutorial, do not; see spec §5)
    'paper_lstm': ['--model', 'paper_lstm', '--cutoff_mode', 'all_slots',
                   '--normalization', 'supplied', '--target_normalization', 'none',
                   '--fill_value', '-1', '--optimizer', 'adam', '--lr_schedule', 'constant',
                   '--weight_decay', '0', '--grad_clip', '0', '--field_alpha', '1.0',
                   '--batch_size', '1028', '--steps_per_epoch', '0', '--epochs', '15'],
}


def build_jobs(args, table):
    jobs = []
    for protocol in args.protocols:
        exp_group = '{}_{}_{}_s{}'.format(
            {'cv': 'cv{}'.format(args.k), 'loro': 'loro', 'loyo': 'loyo'}[protocol],
            args.group if protocol == 'cv' else 'na', args.policy, args.seed)
        for pair in args.pairs:
            prefix = fold_prefix(pair, protocol, args.group, args.policy, args.seed, args.k)
            index = Path(args.artifact_root) / 'splits' / '{}.folds.json'.format(prefix)
            if not index.exists():
                splits = make_paper_folds(table, pair, protocol, k=args.k, group=args.group,
                                          policy=args.policy, seed=args.seed,
                                          val_frac=args.val_frac)
                save_folds(splits, args.artifact_root, prefix)
            folds = json.loads(index.read_text())['folds']
            if args.max_folds:
                folds = folds[:args.max_folds]
            country, crop = parse_pair(pair)
            for inputs in args.inputs:
                for model in args.models:
                    tag = model if model != 'ours' else 'ours-{}'.format(args.fusion)
                    if args.tag:
                        tag = '{}_{}'.format(tag, args.tag)
                    exp_dir = Path(args.runs_root) / 'paper' / exp_group / inputs / tag / pair
                    for i, split_name in enumerate(folds):
                        out = exp_dir / 'fold{:02d}'.format(i)
                        cmd = [sys.executable, 'main_yieldsat_finetune.py',
                               '--data_contract', 'yieldsat_preprocessed_v1',
                               '--source_root', args.source_root, '--artifact_root', args.artifact_root,
                               '--countries', country, '--crops', crop, '--split', split_name,
                               '--output_dir', str(out), '--save_maps', '0',
                               '--num_workers', str(args.num_workers), '--seed', str(args.run_seed)]
                        cmd += INPUTS[inputs] + MODELS[model]
                        if model == 'ours':
                            cmd += ['--fusion', args.fusion]
                            if args.fusion == 'perceiver_tokens':
                                cmd += ['--num_latents', str(args.num_latents)]
                        if args.epochs is not None:
                            cmd += ['--epochs', str(args.epochs)]
                        if args.steps_per_epoch is not None:
                            cmd += ['--steps_per_epoch', str(args.steps_per_epoch)]
                        cmd += args.extra
                        jobs.append({'exp_dir': exp_dir, 'out': out, 'cmd': cmd,
                                     'expected_folds': len(folds)})
    return jobs


def run_jobs(jobs, concurrency, dry_run=False):
    pending = [j for j in jobs if not (j['out'] / 'report.json').exists()]
    print('{} jobs, {} to run'.format(len(jobs), len(pending)), flush=True)
    if dry_run:
        for j in pending:
            print(' '.join(j['cmd']))
        return
    running = []
    while pending or running:
        while pending and len(running) < concurrency:
            j = pending.pop(0)
            j['out'].mkdir(parents=True, exist_ok=True)
            log = open(j['out'] / 'train.log', 'w')
            j['proc'] = subprocess.Popen(j['cmd'], stdout=log, stderr=subprocess.STDOUT)
            j['log'] = log
            running.append(j)
        time.sleep(5)
        for j in list(running):
            if j['proc'].poll() is not None:
                j['log'].close()
                running.remove(j)
                status = 'ok' if j['proc'].returncode == 0 else 'FAILED ({})'.format(j['proc'].returncode)
                print('{} {}'.format(status, j['out']), flush=True)


def aggregate(jobs):
    summary = []
    for exp_dir in sorted({j['exp_dir'] for j in jobs}, key=str):
        done = len([d for d in exp_dir.glob('fold*') if (d / 'report.json').exists()])
        expected = next(j['expected_folds'] for j in jobs if j['exp_dir'] == exp_dir)
        if done == 0:
            continue
        agg = aggregate_folds(exp_dir)
        agg['complete'] = done == expected
        (exp_dir / 'aggregate.json').write_text(json.dumps(agg, indent=1))
        fm = agg['fold_mean_std']
        summary.append(exp_dir)
        print('{} [{}/{} folds] pixel R2 {:.2f}±{:.2f} RMSE {:.2f}±{:.2f} | field R2 {:.2f}±{:.2f} '
              'RMSE {:.2f}±{:.2f}'.format(
                  exp_dir, done, expected, fm['pixel_r2']['mean'], fm['pixel_r2']['std'],
                  fm['pixel_rmse']['mean'], fm['pixel_rmse']['std'], fm['field_r2']['mean'],
                  fm['field_r2']['std'], fm['field_rmse']['mean'], fm['field_rmse']['std']))
    return summary


def main():
    p = argparse.ArgumentParser('YieldSAT paper-protocol runs')
    p.add_argument('--source_root', default=os.environ.get('YIELDSAT_SOURCE_ROOT'))
    p.add_argument('--artifact_root', default=os.environ.get('YIELDSAT_ARTIFACT_ROOT'))
    p.add_argument('--runs_root', default=None, help='default <artifact_root>/runs')
    p.add_argument('--pairs', nargs='+', default=list(PAPER_PAIRS), choices=list(PAPER_PAIRS))
    p.add_argument('--protocols', nargs='+', default=['cv'], choices=['cv', 'loro', 'loyo'])
    p.add_argument('--inputs', nargs='+', default=['s2', 's2_adm'], choices=list(INPUTS))
    p.add_argument('--models', nargs='+', default=['ours'], choices=list(MODELS))
    p.add_argument('--fusion', default='perceiver_summary')
    p.add_argument('--num_latents', type=int, default=16)
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--group', default='season', choices=['season', 'physical'])
    p.add_argument('--policy', default='paper', choices=['paper', 'strict'])
    p.add_argument('--seed', type=int, default=0, help='fold assignment seed')
    p.add_argument('--run_seed', type=int, default=0, help='training seed')
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=None, help='override the preset epochs')
    p.add_argument('--steps_per_epoch', type=int, default=None)
    p.add_argument('--max_folds', type=int, default=None, help='run only the first N folds')
    p.add_argument('--tag', default='', help='suffix for the model directory (e.g. long50)')
    p.add_argument('--concurrency', type=int, default=3)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--aggregate_only', action='store_true')
    p.add_argument('extra', nargs=argparse.REMAINDER,
                   help='after "--": extra arguments passed to main_yieldsat_finetune.py')
    args = p.parse_args()
    args.extra = [a for a in args.extra if a != '--']
    if not args.source_root or not args.artifact_root:
        sys.exit('--source_root and --artifact_root (or YIELDSAT_* env vars) are required')
    args.runs_root = args.runs_root or str(Path(args.artifact_root) / 'runs')
    countries = sorted({parse_pair(pp)[0] for pp in args.pairs})
    table = load_field_table(args.artifact_root, args.source_root, countries,
                             require_geometry=args.policy == 'strict' or args.group == 'physical')
    jobs = build_jobs(args, table)
    if not args.aggregate_only:
        run_jobs(jobs, args.concurrency, dry_run=args.dry_run)
    if not args.dry_run:
        aggregate(jobs)


if __name__ == '__main__':
    main()
