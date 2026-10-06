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
    if a.batch == 'pk':
        units = pk_units(a.artifact_root)
        smoke = [u for u in units if u['pair'] == 'GER-R' and u['id'].endswith('__loyo__f0')]
        for name, us in (('protocol_pk', units), ('protocol_pk_smoke', smoke)):
            Path('cluster/tabm/{}_units.json'.format(name)).write_text(
                json.dumps({'suite': name, 'out_root': a.out_root, 'units': us}, indent=1))
            print('{}: {} units'.format(name, len(us)))
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
    u.add_argument('--batch', default='point', choices=['point', 'image', 'pk'])
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
