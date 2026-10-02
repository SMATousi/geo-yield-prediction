"""Markdown result tables for the cluster suites (results/*.md), with the
paper's metric: pooled out-of-fold R²/RMSE.

Input: the pooled metrics JSON written by ``yieldsat_pooled_metrics.py`` (on
the PVC) and the plan's ``runs.jsonl`` (expected folds per experiment). For
each experiment (protocol, policy, inputs, pair, seed), all folds' held-out
predictions are pooled and R²/RMSE computed once, at pixel level (every cell)
and field level (season mean prediction vs season mean target). A seed counts
only when all its folds are finished. Rows show mean ± std over seeds, next to
the paper's pixel LSTM and best model (results/yieldsat/paper_benchmark.csv).

    python yieldsat_results_tables.py --pooled pooled_before_full.json --runs plans/before_full/runs.jsonl \
        --title "Point model results (before_full suite)" --retire_arg_farm_loro --out results/point_suite.md
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


def _stats(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return m, sd


def _fmt(st):
    if st is None:
        return '–'
    return '{:.2f} ± {:.2f}'.format(*st) if st[1] > 0 else '{:.2f}'.format(st[0])


def build_tables(runs_path, pooled, paper, retired=None):
    lstm, best = paper
    folds = collections.defaultdict(int)           # experiment dir -> expected folds
    meta = {}
    for line in open(runs_path):
        r = json.loads(line)
        if r['protocol'] == 'donor' or (retired and retired(r)):
            continue
        exp_dir = r['rel_path'].rsplit('/', 1)[0]
        folds[exp_dir] += 1
        meta[exp_dir] = (r['experiment'], r['protocol'], r['policy'], r['inputs'], r['pair'])
    cells = collections.defaultdict(lambda: {'seeds': 0, 'complete': [], 'partial': 0})
    for exp_dir, n in folds.items():
        key = meta[exp_dir]
        c = cells[key]
        c['seeds'] += 1
        got = pooled.get(exp_dir)
        if got and len(got['folds']) >= n:
            c['complete'].append(got)
        elif got:
            c['partial'] += 1
    sections, n_complete, n_seeds = [], 0, 0
    for proto in ('cv10', 'loro-province', 'loro', 'loyo'):
        keys = sorted((k for k in cells if k[1] == proto), key=lambda k: (k[2], k[3], k[4], k[0] != 'ours', k[0]))
        if not keys:
            continue
        rows = []
        for k in keys:
            exp, _, policy, inputs, pair = k
            c = cells[k]
            v = c['complete']
            n_complete += len(v)
            n_seeds += c['seeds']
            P, mods = PROTO_NAME[proto], INPUT_NAME[inputs]
            pl = lstm.get((P, 'pixel', 'S2', pair))
            pb = best.get((P, 'pixel', mods, pair))
            fb = best.get((P, 'field', mods, pair))
            seeds = '{}/{}'.format(len(v), c['seeds']) + (' (+{} running)'.format(c['partial']) if c['partial'] else '')
            rows.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                pair, MODEL_NAME.get(exp, exp), policy, mods, seeds,
                _fmt(_stats([x['pixel_r2'] for x in v])), _fmt(_stats([x['pixel_rmse'] for x in v])),
                _fmt(_stats([x['field_r2'] for x in v])), _fmt(_stats([x['field_rmse'] for x in v])),
                '{:.2f} / {:.2f}'.format(*pl) if pl else '–',
                '{:.2f} / {:.2f} ({})'.format(*pb) if pb else '–',
                '{:.2f} / {:.2f} ({})'.format(*fb) if fb else '–'))
        sections.append('\n'.join([
            '## {}'.format(PROTO_TITLE[proto]), '',
            '| Pair | Model | Policy | Inputs | Seeds complete | Pixel R² | Pixel RMSE | Field R² | Field RMSE '
            '| Paper LSTM pixel R² / RMSE (S2) | Paper best pixel R² / RMSE | Paper best field R² / RMSE |',
            '|---|---|---|---|---|---|---|---|---|---|---|---|'] + rows))
    return sections, n_complete, n_seeds


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pooled', required=True, help='yieldsat_pooled_metrics.py output')
    p.add_argument('--runs', required=True, help="the plan's runs.jsonl")
    p.add_argument('--title', required=True)
    p.add_argument('--source', default='', help='where the predictions came from (shown in the header)')
    p.add_argument('--note', default='')
    p.add_argument('--retire_arg_farm_loro', action='store_true',
                   help='leave out the farm-level ARG LORO runs (retired; replaced by provinces)')
    p.add_argument('--paper', default='results/yieldsat/paper_benchmark.csv')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    retired = ((lambda r: r['protocol'] == 'loro' and r['pair'].startswith('ARG'))
               if a.retire_arg_farm_loro else None)
    pooled = json.loads(Path(a.pooled).read_text())
    sections, done, total = build_tables(a.runs, pooled, load_paper(a.paper), retired)
    head = [
        '# {}'.format(a.title), '',
        'Generated {} by `yieldsat_results_tables.py` from pooled out-of-fold predictions{}.'.format(
            time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()), ' ({})'.format(a.source) if a.source else ''),
        '**{} of {} experiments (pair × protocol × policy × inputs × seed) complete.**'.format(done, total), '',
        '- **Metric = the paper\'s computation:** for each experiment, the held-out predictions of all folds '
        'are pooled (each cell is held out exactly once) and R²/RMSE are computed once. Values are mean ± '
        'std of these pooled scores over seeds; a seed counts only when all its folds are finished.',
        '- **Pixel**: every held-out 10 m cell. **Field**: per field season, mean prediction vs mean target. '
        'R² = 1 − SSE/SST; RMSE in t/ha.',
        '- **Policy** `paper` = the paper\'s grouping; `strict` additionally removes training seasons that '
        'share a physical field with the test fold.',
        '- **Paper** numbers are from Pathak et al. (CVPR 2026, `results/yieldsat/paper_benchmark.csv`); '
        '"best" is the best model per row and modality set. Why pooled: '
        '[negative_r2_investigation.md](negative_r2_investigation.md).',
    ]
    if a.note:
        head += ['', a.note]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text('\n'.join(head + [''] + ['\n\n'.join(sections), '']))
    print('wrote {} ({} / {} experiments complete)'.format(a.out, done, total))


if __name__ == '__main__':
    main()
