"""Markdown result tables for the cluster suites (results/*.md).

Test metrics come from W&B (every finished run logs ``test/*`` to its suite's
project); expected run counts come from the plan's ``runs.jsonl``. One table
per protocol: fold/seed means (± std over runs) of pixel and field-level R²
and RMSE (t/ha) per pair, policy and input set, next to the paper's pixel LSTM
and best model (results/yieldsat/paper_benchmark.csv). Field-level R² is
averaged only over runs whose test fold has >= 3 field seasons.

    python yieldsat_results_tables.py --suite before_full yieldsat-cvpr27 plans/before_full/runs.jsonl \
        --title "Point model (before_full)" --out results/point_suite.md
"""

import argparse
import collections
import csv
import json
import math
import time
from pathlib import Path

PROTO_NAME = {'cv10': 'CV10', 'loro': 'LORO', 'loro-province': 'LORO', 'loyo': 'LOYO'}
PROTO_TITLE = {'cv10': 'CV10 (10-fold, grouped by field season)',
               'loro-province': 'LORO, provinces (Argentina; the paper\'s regions)',
               'loro': 'LORO, farm regions',
               'loyo': 'LOYO (leave one year out)'}
INPUT_NAME = {'s2': 'S2', 's2_adm': 'S2+ADM'}
MODEL_NAME = {'ours': 'Ours (point)', 'paper_lstm': 'Paper LSTM (our re-run)', 'image': 'Image v1',
              'image2': 'Image v2'}
METRICS = ('test/pixel_r2', 'test/pixel_rmse', 'test/field_level_r2', 'test/field_level_rmse',
           'test/field_level_n')


def load_paper(path):
    lstm, best = {}, {}
    for p in csv.DictReader(open(path)):
        key = (p['protocol'], p['level'], p['modalities'], p['pair'])
        r2, rmse = float(p['r2_mean']), float(p['rmse_mean'])
        if p['model'] == 'LSTM' and p['modalities'] == 'S2':
            lstm[key] = (r2, rmse)
        if key not in best or r2 > best[key][0]:
            best[key] = (r2, rmse, p['model'])
    return lstm, best


def fetch(project, suite):
    import wandb
    api = wandb.Api(timeout=300)
    out = {}
    for r in api.runs('{}/{}'.format(api.default_entity, project),
                      filters={'display_name': {'$regex': '^{}\\|'.format(suite)}, 'state': 'finished'}):
        s = r.summary
        if s.get('test/pixel_rmse') is None:
            continue
        out[r.name] = [s.get(k) for k in METRICS]
    return out


def _stats(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return m, sd


def _fmt(st, digits=2):
    return '–' if st is None else '{:.{d}f} ± {:.{d}f}'.format(st[0], st[1], d=digits)


def build_tables(suite, runs_path, finished, paper, retired=None):
    lstm, best = paper
    expected = collections.defaultdict(int)
    for line in open(runs_path):
        r = json.loads(line)
        if r['protocol'] == 'donor':
            continue
        if retired and retired(r):
            continue
        expected[(r['experiment'], r['protocol'], r['policy'], r['inputs'], r['pair'])] += 1
    cells = collections.defaultdict(list)
    for name, vals in finished.items():
        parts = name.split('|')
        key = (parts[1], parts[2], parts[3], parts[4], parts[5])
        if key in expected:
            cells[key].append(vals)
    sections = []
    for proto in ('cv10', 'loro-province', 'loro', 'loyo'):
        keys = sorted(k for k in expected if k[1] == proto)
        if not keys:
            continue
        rows = []
        for k in sorted(keys, key=lambda k: (k[2], k[3], k[4], k[0] != 'ours', k[0])):
            exp, _, policy, inputs, pair = k
            v = cells.get(k, [])
            px_r2, px_rmse = _stats([x[0] for x in v]), _stats([x[1] for x in v])
            f_r2 = _stats([x[2] for x in v if (x[4] or 0) >= 3])
            f_rmse = _stats([x[3] for x in v])
            P, mods = PROTO_NAME[proto], INPUT_NAME[inputs]
            pl = lstm.get((P, 'pixel', 'S2', pair))
            pb = best.get((P, 'pixel', mods, pair))
            fb = best.get((P, 'field', mods, pair))
            rows.append('| {} | {} | {} | {} | {}/{} | {} | {} | {} | {} | {} | {} | {} |'.format(
                pair, MODEL_NAME.get(exp, exp), policy, mods, len(v), expected[k], _fmt(px_r2), _fmt(px_rmse), _fmt(f_r2), _fmt(f_rmse),
                '{:.2f} / {:.2f}'.format(*pl) if pl else '–',
                '{:.2f} / {:.2f} ({})'.format(*pb) if pb else '–',
                '{:.2f} / {:.2f} ({})'.format(*fb) if fb else '–'))
        sections.append('\n'.join([
            '## {}'.format(PROTO_TITLE[proto]), '',
            '| Pair | Model | Policy | Inputs | Runs done | Pixel R² | Pixel RMSE | Field R² | Field RMSE '
            '| Paper LSTM pixel R² / RMSE (S2) | Paper best pixel R² / RMSE | Paper best field R² / RMSE |',
            '|---|---|---|---|---|---|---|---|---|---|---|---|'] + rows))
    return sections, sum(len(v) for v in cells.values()), sum(expected.values())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--suite', nargs=3, metavar=('SUITE', 'WANDB_PROJECT', 'RUNS_JSONL'), required=True)
    p.add_argument('--title', required=True)
    p.add_argument('--note', default='')
    p.add_argument('--retire_arg_farm_loro', action='store_true',
                   help='leave out the farm-level ARG LORO runs (retired; replaced by provinces)')
    p.add_argument('--paper', default='results/yieldsat/paper_benchmark.csv')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    suite, project, runs_path = a.suite
    retired = ((lambda r: r['protocol'] == 'loro' and r['pair'].startswith('ARG'))
               if a.retire_arg_farm_loro else None)
    finished = fetch(project, suite)
    sections, done, expected = build_tables(suite, runs_path, finished, load_paper(a.paper), retired)
    head = [
        '# {}'.format(a.title), '',
        'Generated {} from W&B project `{}` (suite `{}`) by `yieldsat_results_tables.py`.'.format(
            time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()), project, suite),
        '**{} of {} planned runs finished.** Values are means ± std over finished folds × seeds.'.format(
            done, expected), '',
        '- **Pixel**: every held-out cell counts once. **Field**: per field season, mean prediction '
        'vs mean target; field R² only over folds with ≥ 3 test field seasons.',
        '- **RMSE** in t/ha. **Policy** `paper` = the paper\'s grouping; `strict` additionally '
        'removes training seasons sharing a physical field with the test fold.',
        '- **Paper** numbers are fold means from Pathak et al. (CVPR 2026, '
        '`results/yieldsat/paper_benchmark.csv`); "best" is the best model per row and modality set.',
    ]
    if a.note:
        head += ['', a.note]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text('\n'.join(head + [''] + ['\n\n'.join(sections), '']))
    print('wrote {} ({} / {} runs)'.format(a.out, done, expected))


if __name__ == '__main__':
    main()
