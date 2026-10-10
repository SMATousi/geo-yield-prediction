"""The paper's training protocol for every model (spec/yieldsat-paper-protocol.md).

No validation set: each model trains on all non-test seasons of the fold and
selects its epoch / step / boosting round on the held-out test fold, as the
YieldSAT paper did. Two arms:

  per-pair  one model per country-crop pair and fold (the paper's protocol)
  pooled    one model for all 9 pairs per fold, results reported per pair

  folds   write the fold manifests (noval_<paper prefix>_fold<ii>, pooled_noval_<protocol>_s0_fold<ii>)
  units   write the cluster work units (cluster/tabm/protocol_<batch>_units.json)
  report  aggregate per pair: fold-mean R2 (the paper's metric) and pooled out-of-fold R2

    python yieldsat_protocol_runs.py folds --artifact_root /root/yieldsat_artifacts
    python yieldsat_protocol_runs.py units --batch point
    python yieldsat_protocol_runs.py report --results_root <dir> --out results/protocol_point.md
"""
import argparse
import json
from pathlib import Path

from dataset.yieldsat_splits import (PAPER_PAIRS, noval_split, paper_row_prefix, parse_pair,
                                     pooled_noval_splits, save_folds)

PROTOCOLS = ('cv', 'loyo', 'loro')


def noval_prefix(pair, protocol):
    return 'noval_' + paper_row_prefix(pair, protocol)


def pooled_prefix(protocol):
    return 'pooled_noval_{}_s0'.format({'cv': 'cv10', 'loyo': 'loyo', 'loro': 'loro'}[protocol])


def cmd_folds(a):
    root = Path(a.artifact_root) / 'splits'
    for protocol in PROTOCOLS:
        for pair in PAPER_PAIRS:
            src = paper_row_prefix(pair, protocol)
            names = json.loads((root / '{}.folds.json'.format(src)).read_text())['folds']
            splits = [noval_split(json.loads((root / '{}.json'.format(n)).read_text()), n) for n in names]
            save_folds(splits, a.artifact_root, noval_prefix(pair, protocol))
            print(noval_prefix(pair, protocol), len(splits))
        splits = pooled_noval_splits(a.artifact_root, protocol)
        save_folds(splits, a.artifact_root, pooled_prefix(protocol))
        print(pooled_prefix(protocol), len(splits), [len(s['partitions']['test']) for s in splits])


COUNTRIES = ('Argentina', 'Brazil', 'Germany', 'Uruguay')
GROUP = {'cv': 'cv10_season', 'loyo': 'loyo_na', 'loro': 'loro_na'}
# relative size of a pair (cells), for longest-first ordering
SIZE = {'ARG-S': 3.1, 'BRA-S': 2.3, 'URG-S': 2.2, 'ARG-C': 1.3, 'BRA-W': 1.1, 'ARG-W': 0.9, 'BRA-C': 0.9,
        'GER-W': 0.3, 'GER-R': 0.3, 'ALL': 12.4}

STREAMS = {'s2': ['yieldsat_s2'],
           's2_adm': ['yieldsat_s2', 'yieldsat_weather', 'yieldsat_dem', 'yieldsat_terrain', 'yieldsat_soil']}
COMMON = ['--data_contract', 'yieldsat_preprocessed_v1', '--artifact_root', '{artifact_root}',
          '--source_root', '{source_root}', '--cutoff_mode', 'all_slots', '--save_maps', '0',
          '--num_workers', '3', '--seed', '0',
          # the paper's protocol: select on the whole test fold, early stopping as in the thesis (A.1.2)
          '--val_rows_per_field', '0']
# point-level models of this project so far (spec/yieldsat-paper-protocol.md §3)
NN = {
    # the paper's input-fusion LSTM, thesis configuration (spec/yieldsat-paper-reproduction.md)
    'lstm-s2': ('s2', ['--model', 'paper_lstm', '--normalization', 'supplied', '--target_normalization', 'none',
                       '--fill_value', '-1', '--optimizer', 'adam', '--lr', '1e-3', '--lr_schedule', 'constant',
                       '--weight_decay', '0', '--grad_clip', '0', '--steps_per_epoch', '0', '--field_alpha', '1.0',
                       '--lstm_hidden', '128', '--lstm_layers', '2', '--lstm_head', 'mlp', '--batch_size', '1024',
                       '--weather_first_slot', 'keep', '--epochs', '50', '--early_stop_patience', '10'],
                False),
    'lstm-s2adm': ('s2_adm', None, False),
    # our point model, best configuration so far (p3-nbr: perceiver summary + 5x5 neighbourhood S2)
    'ours-p3nbr': ('s2_adm', ['--model', 'yieldsat_point', '--fusion', 'perceiver_summary', '--neighbourhood',
                              '--epochs', '60', '--steps_per_epoch', '0', '--min_steps_per_epoch', '500',
                              '--max_steps_per_epoch', '1500', '--early_stop_patience', '10'], True),
}
NN['lstm-s2adm'] = ('s2_adm', NN['lstm-s2'][1], False)
TAB = {
    'tabm-f1': ['--model', 'tabm', '--features', 'F1', '--d_block', '256', '--dropout', '0.2', '--lr', '1e-3'],
    'tabm-f0': ['--model', 'tabm', '--features', 'F0', '--d_block', '256', '--dropout', '0.2', '--lr', '1e-3'],
    'mlp-f0': ['--model', 'mlp', '--features', 'F0'],
    'lgbm-f0': ['--model', 'lgbm', '--features', 'F0', '--lgbm_threads', '4'],
}


# image models (spec/yieldsat-image-training.md; spec/yieldsat-improvement.md round 1), S2+ADM, seed 0;
# all are tested on every cell of the full-coverage tile build
IMG_COMMON = ['--artifact_root', '{artifact_root}', '--image_root', '{image_root}', '--cutoff_mode', 'all_slots',
              '--seed', '0', '--num_workers', '6', '--save_maps', '0']
IMAGE = {
    # v1: spatial image model (64x64 tiles, frozen DINOv3 + modality encoders), trained on the v1 tile set
    'image-v1': (['--epochs', '60', '--batch_size', '16', '--min_steps_per_epoch', '20', '--lr', '5e-4',
                  '--train_min_valid', '2048'], 'image_donors'),
    # v2: + per-pixel S2 time series, level + residual head, present-cell slot coverage
    'image-v2': (['--epochs', '60', '--batch_size', '16', '--min_steps_per_epoch', '20', '--lr', '5e-4',
                  '--train_min_valid', '2048', '--series', '--level_head', '--slot_coverage', 'present'],
                 'image_v2_donors'),
    # round-1 hybrid h1-early (S4 early cell fusion), all tiles, no warm start (as in dev_r1)
    'hybrid-h1': (['--arch', 'hybrid', '--slot_coverage', 'present', '--cell_fusion', 'early', '--epochs', '80',
                   '--batch_size', '8', '--min_steps_per_epoch', '20', '--lr', '2e-3'], None),
}
WARM = {'GER-R': 'pooled', 'GER-W': 'bra-w'}       # donors trained on non-German tiles only (YI-06)


def image_units(artifact_root, pooled_protocols=('cv',)):
    units = []
    for pair, proto, group, folds in _rows(artifact_root):
        if pair == 'ALL' and proto not in pooled_protocols:
            continue
        countries = list(COUNTRIES) if pair == 'ALL' else [parse_pair(pair)[0]]
        crop_args = [] if pair == 'ALL' else ['--crops', parse_pair(pair)[1]]
        prefix = folds[0].rsplit('_fold', 1)[0]
        for tag, (margs, donors) in IMAGE.items():
            for i, name in enumerate(folds):
                args = IMG_COMMON + margs + ['--countries', *countries] + crop_args + [
                    '--split', name, '--streams', *STREAMS['s2_adm'],
                    '--output_dir', '{{out_root}}/paper/{}/s2_adm/{}_seed0/{}/fold{:02d}'.format(group, tag, pair, i)]
                if donors and pair in WARM:
                    args += ['--init_ckpt', '/data/YieldSAT/yieldsat_results/{}/donors/{}/s2_adm/seed0/'
                             'checkpoint_best.pth'.format(donors, WARM[pair]),
                             '--lr', '{:g}'.format(float(margs[margs.index('--lr') + 1]) * 0.3)]
                units.append({'id': '{}__{}__{}__f{}'.format(tag, pair, proto, i), 'pair': pair, 'n_folds': 1,
                              'script': 'main_yieldsat_image.py',
                              'stage': ['index/' + c for c in countries] + ['splits/{}*'.format(prefix)],
                              'stage_image': {'key': pair, 'countries': countries,
                                              'crops': None if pair == 'ALL' else [parse_pair(pair)[1]]},
                              'args': args})
    # per pair first, grouped by pair (one staged tile root per pod), largest first; pooled last
    units.sort(key=lambda u: (u['pair'] == 'ALL', -SIZE[u['pair']], u['pair']))
    return units


PK_ROOT = '/data/YieldSAT/yieldsat_results/pk_dev1r'
# knowledge-pretrained sensor encoders (spec/yieldsat-point-knowledge-pretraining.md) under p3-nbr:
# a3 = knowledge pretraining, a7 = random-target control; the neighbourhood encoder starts fresh
PK_ARMS = ('a3', 'a7')


def pretrain_unit_for(artifact_root, fold_name):
    """Leakage-safe pretraining unit for a fold: one whose corpus (train + val) holds none of the
    fold's test seasons; the DEV index's unit when it qualifies, else the unit of the same protocol
    with the largest corpus. None when no unit qualifies."""
    S = Path(artifact_root) / 'splits'
    if not hasattr(pretrain_unit_for, 'units'):
        pretrain_unit_for.units = {}
        for f in sorted(S.glob('pretrain_unit_*_s0.json')):
            d = json.loads(f.read_text())
            pretrain_unit_for.units[f.stem] = (d['protocol'], set(d['partitions']['train']) | set(d['partitions']['val']))
        pretrain_unit_for.index = json.loads((S / 'pretrain_units_dev_s0.json').read_text())['fold_to_unit']
    split = json.loads((S / '{}.json'.format(fold_name)).read_text())
    test = set(split['partitions']['test'])
    ok = {u: proto for u, (proto, seen) in pretrain_unit_for.units.items() if not (test & seen)}
    pref = pretrain_unit_for.index.get(split.get('source_split', fold_name))
    if pref in ok:
        return pref
    if not ok:
        return None
    return sorted(ok, key=lambda u: (ok[u] != split['scheme'], -len(pretrain_unit_for.units[u][1]), u))[0]


def pk_units(artifact_root):
    units = []
    inputs, margs, _ = NN['ours-p3nbr']
    for pair, proto, group, folds in _rows(artifact_root):
        if pair == 'ALL':
            continue
        country, crop = parse_pair(pair)
        prefix = folds[0].rsplit('_fold', 1)[0]
        for i, name in enumerate(folds):
            unit = pretrain_unit_for(artifact_root, name)
            if unit is None:
                continue
            for arm in PK_ARMS:
                tag = 'ours-p3nbr-{}'.format(arm)
                out = '{{out_root}}/paper/{}/{}/{}_seed0/{}/fold{:02d}'.format(group, inputs, tag, pair, i)
                units.append({'id': '{}__{}__{}__f{}'.format(tag, pair, proto, i), 'pair': pair, 'n_folds': 1,
                              'script': 'main_yieldsat_finetune.py', 'pretrain_unit': unit,
                              'stage': ['cache/' + country, 'index/' + country, 'neighbourhood/' + country,
                                        'splits/{}*'.format(prefix), 'geometry/fields_geometry_*.json'],
                              'args': COMMON + margs + ['--norm_pooling', 'per_country',
                                       '--init_sensor_ckpt', '{}/pretrain/{}/{}/sensor_checkpoint.pth'.format(
                                           PK_ROOT, arm, unit),
                                       '--encoders_only_transfer', '--init_nonstrict_streams',
                                       '--countries', country, '--crops', crop, '--split', name,
                                       '--output_dir', out, '--streams', *STREAMS[inputs]]})
    units.sort(key=lambda u: -SIZE[u['pair']])
    return units


DENSE_ENV = {'YIELDSAT_NUM_SLOTS': '72', 'YIELDSAT_CACHE_NAME': 'cache_dense', 'YIELDSAT_NBR_NAME': 'neighbourhood_dense'}


def dense_units(artifact_root):
    """p3-nbr on the dense raw S2 series (spec/yieldsat-dense-series.md), per pair, all 217 folds."""
    units = []
    inputs, margs, _ = NN['ours-p3nbr']
    for pair, proto, group, folds in _rows(artifact_root):
        if pair == 'ALL':
            continue
        country, crop = parse_pair(pair)
        prefix = folds[0].rsplit('_fold', 1)[0]
        for i, name in enumerate(folds):
            tag = 'ours-p3nbr-dense'
            out = '{{out_root}}/paper/{}/{}/{}_seed0/{}/fold{:02d}'.format(group, inputs, tag, pair, i)
            units.append({'id': '{}__{}__{}__f{}'.format(tag, pair, proto, i), 'pair': pair, 'n_folds': 1,
                          'script': 'main_yieldsat_finetune.py', 'env': DENSE_ENV,
                          'stage': ['cache_dense/' + country, 'neighbourhood_dense/' + country, 'index/' + country,
                                    'splits/{}*'.format(prefix), 'geometry/fields_geometry_*.json'],
                          'args': COMMON + margs + ['--weather_first_slot', 'keep',
                                   '--countries', country, '--crops', crop, '--split', name,
                                   '--output_dir', out, '--streams', *STREAMS[inputs]]})
    units.sort(key=lambda u: -SIZE[u['pair']])
    return units


# The paper's best models (spec/yieldsat-paper-models.md): thesis A.1.2 training (Adam 0.006, batch 2048,
# reduce-on-plateau, 50 epochs, early stop 10, rot90 + temporal dropout), -1 padding, 5x5 windows. Input fusion
# (3D-LSTM, 3D-ConvLSTM) on the monthly series as in the thesis IF pipeline; feature fusion (AFF, MMGF) dense.
PAPER_MODEL_ARGS = ['--fill_value', '-1', '--optimizer', 'adam', '--weight_decay', '0', '--lr', '0.006',
                    '--lr_schedule', 'plateau', '--batch_size', '2048', '--eval_batch_size', '2048', '--epochs', '50',
                    '--early_stop_patience', '10', '--steps_per_epoch', '0', '--min_steps_per_epoch', '50',
                    '--max_steps_per_epoch', '1500', '--paper_aug', '--field_alpha', '1.0', '--block_size', '32',
                    '--weather_first_slot', 'keep']
PAPER_MODEL_STREAMS = list(STREAMS['s2_adm']) + ['yieldsat_coords']
PAPER_MODELS_DENSE = {'paper_3dlstm': False, 'paper_3dconvlstm': False, 'paper_aff': True, 'paper_mmgf': True}


def paper_model_units(artifact_root, models=tuple(PAPER_MODELS_DENSE)):
    units = []
    for pair, proto, group, folds in _rows(artifact_root):
        if pair == 'ALL':
            continue
        country, crop = parse_pair(pair)
        prefix = folds[0].rsplit('_fold', 1)[0]
        for m in models:
            dense = PAPER_MODELS_DENSE[m]
            for i, name in enumerate(folds):
                out = '{{out_root}}/paper/{}/s2_adm/{}_seed0/{}/fold{:02d}'.format(group, m, pair, i)
                units.append({'id': '{}__{}__{}__f{}'.format(m, pair, proto, i), 'pair': pair, 'n_folds': 1,
                              'script': 'main_yieldsat_finetune.py', 'env': DENSE_ENV if dense else {},
                              'stage': ['cache_dense/' + country if dense else 'cache/' + country, 'index/' + country,
                                        'splits/{}*'.format(prefix), 'geometry/fields_geometry_*.json'],
                              'args': COMMON + PAPER_MODEL_ARGS + ['--model', m, '--countries', country, '--crops', crop,
                                                                  '--split', name, '--output_dir', out,
                                                                  '--streams', *PAPER_MODEL_STREAMS]})
    units.sort(key=lambda u: -SIZE[u['pair']])
    return units


PK_DENSE_ROOT = '/data/YieldSAT/yieldsat_results/pk_dense'


def pk_dense_pretrain_units(artifact_root):
    """Knowledge pretraining (A3) and its random-target control (A7) at 72 slots on the 46 units
    (spec/yieldsat-dense-series.md §9); arguments as pk_dev1r (yieldsat_cluster._make_pretrain_runs)."""
    index = json.loads((Path(artifact_root) / 'splits' / 'pretrain_units_dev_s0.json').read_text())
    arms = {'a3': ['--knowledge'], 'a7': ['--knowledge', '--knowledge_control', 'random_targets']}
    units = []
    for arm, extra in arms.items():
        for u in sorted(index['units'].values(), key=lambda u: u['manifest']):
            out = '{{out_root}}/pretrain/{}/{}'.format(arm, u['manifest'])
            units.append({'id': 'pretrain-dense__{}__{}'.format(arm, u['manifest']), 'pair': 'ALL', 'n_folds': 1,
                          'script': 'main_yieldsat_finetune.py', 'env': DENSE_ENV,
                          'stage': ['cache_dense/' + c for c in COUNTRIES] + ['index/' + c for c in COUNTRIES]
                                   + ['knowledge', 'splits/{}.json'.format(u['manifest'])],
                          'args': ['--data_contract', 'yieldsat_preprocessed_v1', '--artifact_root', '{artifact_root}',
                                   '--source_root', '{source_root}', '--mode', 'pretrain', '--countries', *COUNTRIES,
                                   '--split', u['manifest'], '--norm_pooling', 'per_country', '--seed', '0',
                                   '--streams', *STREAMS['s2_adm'], '--model', 'yieldsat_point',
                                   '--cutoff_mode', 'all_slots', '--fusion', 'perceiver_summary',
                                   '--epochs', '30', '--steps_per_epoch', '500', '--batch_size', '512',
                                   '--weather_first_slot', 'keep', '--num_workers', '3', '--pretrain_diagnostics',
                                   '--output_dir', out, *extra]})
    return units


def pk_dense_finetune_units(artifact_root):
    """Dense p3-nbr from the dense A3/A7 unit checkpoints on the leakage-safe folds."""
    units = []
    for u in pk_units(artifact_root):
        arm = u['id'].split('__')[0].rsplit('-', 1)[1]
        args = list(u['args'])
        i = args.index('--init_sensor_ckpt')
        args[i + 1] = '{}/pretrain/{}/{}/sensor_checkpoint.pth'.format(PK_DENSE_ROOT, arm, u['pretrain_unit'])
        tag_old = 'ours-p3nbr-{}'.format(arm)
        tag_new = 'ours-p3nbr-dense-{}'.format(arm)
        j = args.index('--output_dir')
        args[j + 1] = args[j + 1].replace(tag_old + '_seed0', tag_new + '_seed0')
        args += ['--weather_first_slot', 'keep']
        stage = [x.replace('cache/', 'cache_dense/').replace('neighbourhood/', 'neighbourhood_dense/') for x in u['stage']]
        units.append(dict(u, id=u['id'].replace(tag_old, tag_new), env=DENSE_ENV, stage=stage, args=args))
    return units


def _rows(artifact_root):
    """(pair, protocol, group, prefix) of every per-pair row and pooled row."""
    out = []
    for proto in PROTOCOLS:
        for pair in PAPER_PAIRS:
            group = GROUP[proto] + ('_province' if proto == 'loro' and pair.startswith('ARG') else '')
            out.append((pair, proto, '{}_noval_s0'.format(group), noval_prefix(pair, proto)))
        out.append(('ALL', proto, 'pooled_{}_noval_s0'.format(GROUP[proto].split('_')[0]), pooled_prefix(proto)))
    for pair, proto, group, prefix in out:
        folds = json.loads((Path(artifact_root) / 'splits' / '{}.folds.json'.format(prefix)).read_text())['folds']
        yield pair, proto, group, folds


def cmd_units(a):
    nn, tab = [], []
    nn_tags = [t for t in NN if a.with_lstm or not t.startswith('lstm-')]
    for pair, proto, group, folds in _rows(a.artifact_root):
        countries = list(COUNTRIES) if pair == 'ALL' else [parse_pair(pair)[0]]
        crop_args = [] if pair == 'ALL' else ['--crops', parse_pair(pair)[1]]
        prefix = folds[0].rsplit('_fold', 1)[0]
        for tag in nn_tags:
            inputs, margs, nbr = NN[tag]
            for i, name in enumerate(folds):
                stage = (['cache/' + c for c in countries] + ['index/' + c for c in countries]
                         + (['neighbourhood/' + c for c in countries] if nbr else [])
                         + ['splits/{}*'.format(prefix), 'geometry/fields_geometry_*.json'])
                out = '{{out_root}}/paper/{}/{}/{}_seed0/{}/fold{:02d}'.format(group, inputs, tag, pair, i)
                nn.append({'id': '{}__{}__{}__f{}'.format(tag, pair, proto, i), 'pair': pair, 'n_folds': 1,
                           'script': 'main_yieldsat_finetune.py', 'stage': stage,
                           'args': COMMON + margs + ['--countries', *countries] + crop_args
                           + ['--split', name, '--output_dir', out, '--streams', *STREAMS[inputs]]})
        for tag, targs in TAB.items():
            chunk = 1 if pair == 'ALL' else (len(folds) if tag in ('mlp-f0', 'lgbm-f0') else 3)
            for c in range(0, len(folds), chunk):
                fl = list(range(c, min(c + chunk, len(folds))))
                args = targs + ['--tag', tag, '--fold_set', 'pooled' if pair == 'ALL' else 'noval',
                                '--protocols', proto, '--folds', *map(str, fl), '--seed', '0',
                                '--val_rows_per_field', '0']
                if pair != 'ALL':
                    args += ['--pairs', pair]
                else:
                    args += ['--lgbm_rows_per_field', '1000']
                tab.append({'id': '{}__{}__{}__f{}'.format(tag, pair, proto, '-'.join(map(str, fl))),
                            'pair': pair, 'n_folds': len(fl), 'args': args})
    # per-pair rows first (the paper's protocol, the main table), then pooled; longest first within each
    key = lambda u: (u['pair'] == 'ALL', -SIZE[u['pair']] * u['n_folds'])
    nn.sort(key=key)
    tab.sort(key=key)
    # flat-matrix caches missing on the PVC are built first, one unit each
    cached = {('ARG-W', 'F1'), ('BRA-C', 'F1'), ('GER-R', 'F1'), ('URG-S', 'F1')}
    builds = [{'id': 'build__{}__{}'.format(f, pair), 'pair': pair, 'n_folds': 1,
               'args': ['--build_only', '--features', f, '--pairs', pair, '--tag', 'build']}
              for pair in sorted(PAPER_PAIRS, key=lambda x: -SIZE[x]) for f in ('F1', 'F0')
              if (pair, f) not in cached]
    tab = builds + tab
    if a.batch in ('rel-dev', 'fc-dev'):
        # RL-04 (spec/yieldsat-relational-loss.md): relation-matching loss; FC-04
        # (spec/yieldsat-field-context.md): field-context streams with and without it. 4 DEV pairs x CV10/LOYO
        arms = ({'ours-p3nbr-dense-rel05': ['--rel_weight', '0.5'], 'ours-p3nbr-dense-rel10': ['--rel_weight', '1.0']}
                if a.batch == 'rel-dev' else
                {'ours-p3nbr-dense-fc': ['--field_context'],
                 'ours-p3nbr-dense-fc-rel10': ['--field_context', '--rel_weight', '1.0']})
        units = []
        for u in dense_units(a.artifact_root):
            pair, proto = u['id'].split('__')[1:3]
            if pair not in ('ARG-W', 'BRA-C', 'GER-R', 'URG-S') or proto not in ('cv', 'loyo'):
                continue
            for tag, extra in arms.items():
                args = list(u['args'])
                j = args.index('--output_dir')
                args[j + 1] = args[j + 1].replace('ours-p3nbr-dense_seed0', tag + '_seed0')
                units.append(dict(u, id=u['id'].replace('ours-p3nbr-dense', tag), args=args + extra))
        suite = 'protocol_{}'.format(a.batch.replace('-', '_'))
        Path('cluster/tabm/{}_units.json'.format(suite)).write_text(
            json.dumps({'suite': suite, 'out_root': a.out_root, 'units': units}, indent=1))
        print('{}: {} units'.format(suite, len(units)))
        return
    if a.batch == 'paper-models':
        units = paper_model_units(a.artifact_root)
        Path('cluster/tabm/protocol_paper_models_units.json').write_text(
            json.dumps({'suite': 'protocol_paper_models', 'out_root': a.out_root, 'units': units}, indent=1))
        print('protocol_paper_models: {} units'.format(len(units)))
        return
    if a.batch in ('fe-dev', 'fe-all'):
        # FE-04 / FE-05 (spec/yieldsat-field-encoder.md): field encoder on top of the best configuration
        # (dense p3-nbr + field context + relation loss); fe-dev = 4 DEV pairs x CV10/LOYO, fe-all = the rest
        tag = 'ours-p3nbr-dense-fe'
        dev = {'ARG-W', 'BRA-C', 'GER-R', 'URG-S'}
        units = []
        for u in dense_units(a.artifact_root):
            pair, proto = u['id'].split('__')[1:3]
            in_dev = pair in dev and proto in ('cv', 'loyo')
            if in_dev != (a.batch == 'fe-dev'):
                continue
            args = list(u['args'])
            j = args.index('--output_dir')
            args[j + 1] = args[j + 1].replace('ours-p3nbr-dense_seed0', tag + '_seed0')
            units.append(dict(u, id=u['id'].replace('ours-p3nbr-dense', tag),
                              args=args + ['--field_context', '--rel_weight', '1.0', '--model', 'field_encoder']))
        suite = 'protocol_{}'.format(a.batch.replace('-', '_'))
        Path('cluster/tabm/{}_units.json'.format(suite)).write_text(
            json.dumps({'suite': suite, 'out_root': a.out_root, 'units': units}, indent=1))
        print('{}: {} units'.format(suite, len(units)))
        return
    if a.batch == 'fc-all':
        # FC-05 (spec/yieldsat-field-context.md): field context + relation loss (lambda 1) on every pair and
        # protocol; the DEV pairs' CV10/LOYO folds already ran in FC-04 (same ids, outputs) and are left out
        tag = 'ours-p3nbr-dense-fc-rel10'
        units = []
        for u in dense_units(a.artifact_root):
            pair, proto = u['id'].split('__')[1:3]
            if pair in ('ARG-W', 'BRA-C', 'GER-R', 'URG-S') and proto in ('cv', 'loyo'):
                continue
            args = list(u['args'])
            j = args.index('--output_dir')
            args[j + 1] = args[j + 1].replace('ours-p3nbr-dense_seed0', tag + '_seed0')
            units.append(dict(u, id=u['id'].replace('ours-p3nbr-dense', tag),
                              args=args + ['--field_context', '--rel_weight', '1.0']))
        Path('cluster/tabm/protocol_fc_all_units.json').write_text(
            json.dumps({'suite': 'protocol_fc_all', 'out_root': a.out_root, 'units': units}, indent=1))
        print('protocol_fc_all: {} units'.format(len(units)))
        return
    if a.batch == 'obsdrop-all':
        # S2 observation dropout (min keep 0.1) on every pair and protocol (spec §10); same ids/paths as the
        # BRA-C/GER-W LOYO units of 'obsdrop', so those are not run twice
        units = []
        tag = 'ours-p3nbr-dense-obsdrop01'
        for u in dense_units(a.artifact_root):
            args = list(u['args'])
            j = args.index('--output_dir')
            args[j + 1] = args[j + 1].replace('ours-p3nbr-dense_seed0', tag + '_seed0')
            units.append(dict(u, id=u['id'].replace('ours-p3nbr-dense', tag), args=args + ['--s2_obs_dropout', '0.1']))
        Path('cluster/tabm/protocol_obsdrop_all_units.json').write_text(
            json.dumps({'suite': 'protocol_obsdrop_all', 'out_root': a.out_root, 'units': units}, indent=1))
        print('protocol_obsdrop_all: {} units'.format(len(units)))
        return
    if a.batch == 'obsdrop':
        # S2 observation dropout on the two LOYO rows that regressed with the dense series (spec §10)
        units = []
        for u in dense_units(a.artifact_root):
            pair, proto = u['id'].split('__')[1:3]
            if proto != 'loyo' or pair not in ('BRA-C', 'GER-W'):
                continue
            for k in ('0.3', '0.1'):
                tag = 'ours-p3nbr-dense-obsdrop{}'.format(k.replace('.', ''))
                args = list(u['args'])
                j = args.index('--output_dir')
                args[j + 1] = args[j + 1].replace('ours-p3nbr-dense_seed0', tag + '_seed0')
                units.append(dict(u, id=u['id'].replace('ours-p3nbr-dense', tag), args=args + ['--s2_obs_dropout', k]))
        Path('cluster/tabm/protocol_obsdrop_units.json').write_text(
            json.dumps({'suite': 'protocol_obsdrop', 'out_root': a.out_root, 'units': units}, indent=1))
        print('protocol_obsdrop: {} units'.format(len(units)))
        return
    if a.batch == 'pk-dense':
        pre = pk_dense_pretrain_units(a.artifact_root)
        ft = pk_dense_finetune_units(a.artifact_root)
        smoke = [u for u in pre if u['id'] in ('pretrain-dense__a3__pretrain_unit_loyo_2017_s0',
                                                'pretrain-dense__a7__pretrain_unit_loyo_2017_s0')]
        for name, us, root in (('pk_dense_pretrain', pre, PK_DENSE_ROOT), ('pk_dense_pretrain_smoke', smoke, PK_DENSE_ROOT),
                               ('pk_dense_finetune', ft, a.out_root)):
            Path('cluster/tabm/{}_units.json'.format(name)).write_text(
                json.dumps({'suite': name, 'out_root': root, 'units': us}, indent=1))
            print('{}: {} units'.format(name, len(us)))
        return
    if a.batch == 'dense':
        units = dense_units(a.artifact_root)
        smoke = [u for u in units if u['id'] in ('ours-p3nbr-dense__GER-R__cv__f0', 'ours-p3nbr-dense__ARG-S__loyo__f3',
                                                  'ours-p3nbr-dense__URG-S__cv__f0')]
        for name, us in (('protocol_dense', units), ('protocol_dense_smoke', smoke)):
            Path('cluster/tabm/{}_units.json'.format(name)).write_text(
                json.dumps({'suite': name, 'out_root': a.out_root, 'units': us}, indent=1))
            print('{}: {} units'.format(name, len(us)))
        return
    if a.batch == 'pk':
        units = pk_units(a.artifact_root)
        smoke = [u for u in units if u['pair'] == 'GER-R' and u['id'].endswith('__loyo__f0')]
        for name, us in (('protocol_pk', units), ('protocol_pk_smoke', smoke)):
            Path('cluster/tabm/{}_units.json'.format(name)).write_text(
                json.dumps({'suite': name, 'out_root': a.out_root, 'units': us}, indent=1))
            print('{}: {} units'.format(name, len(us)))
        return
    if a.batch == 'hybrid-shm':
        # hybrid follow-up (the first image job's 2 Gi /dev/shm killed its loader workers): 3 workers,
        # larger shm in the job, new claim ids; units whose report already exists are skipped
        units = []
        for u in image_units(a.artifact_root):
            if not u['id'].startswith('hybrid-h1'):
                continue
            out = u['args'][u['args'].index('--output_dir') + 1]
            units.append(dict(u, id=u['id'] + '__shm', args=u['args'] + ['--num_workers', '3'],
                              skip_if_exists=out + '/report.json'))
        Path('cluster/tabm/protocol_hybrid_shm_units.json').write_text(
            json.dumps({'suite': 'protocol_hybrid_shm', 'out_root': a.out_root, 'units': units}, indent=1))
        print('protocol_hybrid_shm: {} units'.format(len(units)))
        return
    if a.batch == 'image':
        units = image_units(a.artifact_root)
        smoke = [u for u in units if u['pair'] == 'GER-R' and u['id'].endswith('__cv__f0')]
        for name, us in (('protocol_image', units), ('protocol_image_smoke', smoke)):
            Path('cluster/tabm/{}_units.json'.format(name)).write_text(
                json.dumps({'suite': name, 'out_root': a.out_root, 'units': us}, indent=1))
            print('{}: {} units'.format(name, len(us)))
        return
    for name, units in (('protocol_nn', nn), ('protocol_tab', tab)):
        out = Path('cluster/tabm/{}_units.json'.format(name))
        out.write_text(json.dumps({'suite': name, 'out_root': a.out_root, 'units': units}, indent=1))
        print('{}: {} units -> {}'.format(name, len(units), out))


def _protocol_of(group):
    g = group.replace('pooled_', '')
    return 'CV10' if g.startswith('cv10') else 'LOYO' if g.startswith('loyo') else 'LORO'


def collect(results_root):
    """{(protocol, arm, tag, inputs, pair): [(pixel R2, field R2) per fold]}; pooled runs are split per pair."""
    import numpy as np
    from dataset.yieldsat_splits import pair_code
    out = {}
    if str(results_root).endswith('.json'):
        # compact export: [{'path': 'paper/<group>/<inputs>/<tag>_seed0/<pair>/fold<ii>/report.json', ...}]
        items = [(Path(r['path']), r) for r in json.loads(Path(results_root).read_text())]
    else:
        items = ((f, json.loads(f.read_text())) for f in
                 sorted(Path(results_root).glob('paper/*_noval_s0/*/*_seed0/*/fold*/report.json')))
    for f, r in items:
        fold_dir = f.parent
        pair, tagdir, inputs, group = fold_dir.parent.name, fold_dir.parents[1].name, \
            fold_dir.parents[2].name, fold_dir.parents[3].name
        if r.get('selection') != 'test_fold':
            continue
        tag = tagdir.rsplit('_seed', 1)[0]
        t = r['test']
        if pair == 'ALL':
            items = [(pair_code(*k.split('/')), v) for k, v in t.get('per_country_crop', {}).items()]
            arm = 'pooled'
        else:
            items, arm = [(pair, t['overall'])], 'per-pair'
        for code, m in items:
            out.setdefault((_protocol_of(group), arm, tag, inputs, code), []).append(
                (m['pixel']['r2'], m['field_level']['r2'] if m['field_level']['r2'] is not None else np.nan))
    return out


def _paper(path):
    import csv
    rows = list(csv.DictReader(open(path)))
    ref = {}
    for r in rows:
        k = (r['protocol'], r['level'], r['pair'])
        v = float(r['r2_mean'])
        if r['model'] == 'LSTM':
            ref[k + ('LSTM ' + r['modalities'],)] = v
        best = ref.get(k + ('best',))
        if best is None or v > best:
            ref[k + ('best',)] = v
    return ref


def cmd_report(a):
    import numpy as np
    res = collect(a.results_root)
    paper = _paper(a.paper_csv)
    cols = sorted({(arm, tag, inputs) for (_, arm, tag, inputs, _) in res}, key=lambda c: (c[0] != 'per-pair', c[1], c[2]))
    lines = ['# The paper\'s protocol: results', '',
             'Generated by `yieldsat_protocol_runs.py report` (spec/yieldsat-paper-protocol.md). R\u00b2 averaged over '
             'folds (the paper\'s metric); no validation set, selection on the test fold. *per-pair*: one model '
             'per country-crop pair (the paper\'s protocol); *pooled*: one model for all pairs, scored per pair. '
             'Cells: R\u00b2 (folds done / folds). Paper: LSTM S2, LSTM S2+ADM input fusion, best model of the row.', '']
    summary = {}
    for proto in ('CV10', 'LOYO', 'LORO'):
        for level, li in (('pixel', 0), ('field', 1)):
            lines += ['## {} — {} level'.format(proto, level), '',
                      '| Pair | Paper LSTM S2 | Paper LSTM S2+ADM | Paper best | ' +
                      ' | '.join('{} {} ({})'.format(t, i, arm) for arm, t, i in cols) + ' |',
                      '|---' * (4 + len(cols)) + '|']
            for pair in PAPER_PAIRS:
                cells = [pair] + ['{:.2f}'.format(paper[(proto, level, pair, k)]) if (proto, level, pair, k) in paper
                                  else '—' for k in ('LSTM S2', 'LSTM S2+ADM', 'best')]
                for arm, tag, inputs in cols:
                    v = res.get((proto, arm, tag, inputs, pair))
                    if not v:
                        cells.append('—')
                        continue
                    m = float(np.nanmean([x[li] for x in v]))
                    summary.setdefault((proto, level, arm, tag, inputs), []).append(m)
                    cells.append('{:.2f} ({})'.format(m, len(v)))
                lines.append('| ' + ' | '.join(cells) + ' |')
            mean_row = ['**mean over pairs**', '', '', '']
            for arm, tag, inputs in cols:
                v = summary.get((proto, level, arm, tag, inputs), [])
                mean_row.append('{:.2f} ({} pairs)'.format(np.mean(v), len(v)) if v else '—')
            lines += ['| ' + ' | '.join(mean_row) + ' |', '']
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('folds')
    f.add_argument('--artifact_root', default='/root/yieldsat_artifacts')
    f.set_defaults(func=cmd_folds)
    u = sub.add_parser('units')
    u.add_argument('--artifact_root', default='/root/yieldsat_artifacts')
    u.add_argument('--out_root', default='/data/YieldSAT/yieldsat_results/protocol')
    u.add_argument('--batch', default='point', choices=['point', 'image', 'pk', 'hybrid-shm', 'dense', 'pk-dense', 'obsdrop', 'obsdrop-all', 'rel-dev', 'fc-dev', 'fc-all', 'paper-models', 'fe-dev', 'fe-all'])
    u.add_argument('--with_lstm', action='store_true',
                   help='include the paper LSTM (deferred 2026-10-06: our models first, LSTM only if needed)')
    u.set_defaults(func=cmd_units)
    r = sub.add_parser('report')
    r.add_argument('--results_root', default='/data/YieldSAT/yieldsat_results/protocol')
    r.add_argument('--paper_csv', default='results/yieldsat/paper_benchmark.csv')
    r.add_argument('--out', default='results/protocol_point.md')
    r.set_defaults(func=cmd_report)
    a = p.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
