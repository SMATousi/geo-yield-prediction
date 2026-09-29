# --------------------------------------------------------
# Side-by-side comparison with the YieldSAT paper (PC-07).
#
#   python yieldsat_paper_compare.py --runs_root $YIELDSAT_ARTIFACT_ROOT/runs \
#       --out_dir results/yieldsat/paper_comparison
#
# Reads results/yieldsat/paper_benchmark.csv (paper Tables 13-18, mean±std
# over folds) and every aggregate.json under <runs_root>/paper/ (written by
# yieldsat_paper_runs.py). For each protocol, level and country-crop pair it
# lists: the paper's LSTM (S2), the paper's best model per input set, and
# each of our experiments (fold mean±std, the paper's statistic). Experiments
# run with the strict leakage policy or physical-field grouping are listed as
# sensitivity rows, not as like-for-like comparisons.
# --------------------------------------------------------

import argparse
import csv
import json
from pathlib import Path

PAIRS = ('ARG-C', 'ARG-S', 'ARG-W', 'BRA-C', 'BRA-S', 'BRA-W', 'GER-R', 'GER-W', 'URG-S')
PROTOCOLS = {'cv10': 'CV10', 'loro': 'LORO', 'loyo': 'LOYO'}
INPUT_LABEL = {'s2': 'S2', 's2_adm': 'S2+ADM'}


def load_paper(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in ('r2_mean', 'r2_std', 'rmse_mean', 'rmse_std'):
            r[k] = float(r[k])
        r['sources_agree'] = r.get('sources_agree', 'True') == 'True'
    return rows


def load_ours(runs_root):
    """Our experiments; ``<model>_seed<N>`` directories of one experiment are
    merged: values are the mean over seeds of the fold mean (± the mean fold
    std, the paper's statistic), with the std over seeds as ``seed_std``."""
    import re
    import numpy as np
    raw = _load_ours_raw(runs_root)
    groups = {}
    for r in raw:
        base = re.sub(r'_seed\d+$', '', r['model_dir'])
        key = (r['protocol'], r['level'], r['pair'], r['modalities'], r['entry_base'].replace(
            r['model_dir'], base), r['like_for_like'])
        groups.setdefault(key, []).append(r)
    out = []
    for (proto, level, pair, mods, entry, lfl), rs in groups.items():
        m = lambda k: float(np.mean([x[k] for x in rs]))
        out.append({'protocol': proto, 'level': level, 'pair': pair, 'modalities': mods,
                    'entry': entry + (' ({} seeds)'.format(len(rs)) if len(rs) > 1 else ''),
                    'like_for_like': lfl, 'r2_mean': m('r2_mean'), 'r2_std': m('r2_std'),
                    'rmse_mean': m('rmse_mean'), 'rmse_std': m('rmse_std'),
                    'seed_std_r2': float(np.std([x['r2_mean'] for x in rs])),
                    'seed_std_rmse': float(np.std([x['rmse_mean'] for x in rs])),
                    'folds': '/'.join(x['folds'] for x in rs), 'pooled_r2': m('pooled_r2'),
                    'source': '; '.join(x['source'] for x in rs)})
    return out


def _load_ours_raw(runs_root):
    out = []
    for agg in sorted(Path(runs_root).glob('paper/*/*/*/*/aggregate.json')):
        pair_dir = agg.parent
        model, inputs, exp = pair_dir.parent.name, pair_dir.parent.parent.name, pair_dir.parent.parent.parent.name
        proto_key, group, policy = exp.split('_')[:3]
        if proto_key not in PROTOCOLS:        # e.g. cv5: no paper counterpart
            continue
        proto = PROTOCOLS[proto_key]
        a = json.loads(agg.read_text())
        fm = a['fold_mean_std']
        like_for_like = policy == 'paper' and group in ('season', 'na')
        for level, key in (('field', 'field'), ('pixel', 'pixel')):
            out.append({'protocol': proto, 'level': level, 'pair': pair_dir.name,
                        'modalities': INPUT_LABEL[inputs],
                        'model_dir': model,
                        'entry_base': ('re-run: ' if model.startswith('paper_') else 'ours: ') + model
                                 + ('' if like_for_like else
                                                            ' [{} grouping, {} policy]'.format(group, policy)),
                        'like_for_like': like_for_like,
                        'r2_mean': fm['{}_r2'.format(key)]['mean'], 'r2_std': fm['{}_r2'.format(key)]['std'],
                        'rmse_mean': fm['{}_rmse'.format(key)]['mean'],
                        'rmse_std': fm['{}_rmse'.format(key)]['std'],
                        'folds': '{}{}'.format(a['folds_done'], '' if a.get('complete', True) else ' (incomplete)'),
                        'pooled_r2': a['pooled_oof']['pixel' if level == 'pixel' else 'field_level']['r2'],
                        'source': str(pair_dir)})
    return out


def build(paper, ours):
    table = []
    for proto in ('CV10', 'LORO', 'LOYO'):
        for level in ('field', 'pixel'):
            for pair in PAIRS:
                cand = [r for r in paper if r['protocol'] == proto and r['level'] == level
                        and r['pair'] == pair]
                for r in cand:
                    if r['model'] == 'LSTM' and r['modalities'] == 'S2':
                        table.append(dict(r, entry='paper: LSTM (S2)', like_for_like=True, folds=''))
                for mods in ('S2', 'S2+ADM'):
                    sub = [r for r in cand if r['modalities'] == mods]
                    if sub:
                        best = max(sub, key=lambda r: r['r2_mean'])
                        table.append(dict(best, entry='paper best ({})'.format(mods),
                                          note='{}{}'.format(best['model'], '' if best['fusion'] == 'none'
                                                             else ', ' + best['fusion'].split('_')[0]),
                                          like_for_like=True, folds=''))
                for o in ours:
                    if o['protocol'] == proto and o['level'] == level and o['pair'] == pair:
                        table.append(o)
    return table


def write(table, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ('protocol', 'level', 'pair', 'entry', 'note', 'modalities', 'r2_mean', 'r2_std', 'rmse_mean',
            'rmse_std', 'seed_std_r2', 'seed_std_rmse', 'folds', 'like_for_like', 'sources_agree',
            'pooled_r2')
    with open(out_dir / 'comparison.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in table:
            w.writerow({k: (round(r[k], 3) if isinstance(r.get(k), float) else r.get(k, ''))
                        for k in cols})
    lines = ['# YieldSAT paper comparison', '',
             'Generated by `yieldsat_paper_compare.py`. Values are R² / RMSE (t/ha) as mean ± std',
             'over folds (the paper\'s statistic). Paper rows come from arXiv:2604.00940 Tables',
             '13–18; `*` marks cells where the PDF and the project results page disagree.',
             'Rows tagged with a grouping/policy in brackets are leakage-sensitivity runs, not',
             'like-for-like comparisons. "re-run" rows are our re-implementations of the paper\'s',
             'baselines (PC-05); "ours" rows are our models.', '']
    for proto in ('CV10', 'LORO', 'LOYO'):
        for level in ('field', 'pixel'):
            rows = [r for r in table if r['protocol'] == proto and r['level'] == level]
            if not any(not r['entry'].startswith('paper') for r in rows):
                continue
            entries = []
            for r in rows:
                key = (r['entry'], r['modalities'])
                if key not in entries:
                    entries.append(key)
            lines += ['## {} — {} level'.format(proto, level), '',
                      '| Entry | Inputs | ' + ' | '.join(PAIRS) + ' |',
                      '|---|---|' + '---|' * len(PAIRS)]
            for entry, mods in entries:
                cells = []
                for pair in PAIRS:
                    m = [r for r in rows if r['entry'] == entry and r['modalities'] == mods
                         and r['pair'] == pair]
                    if not m:
                        cells.append('–')
                        continue
                    r = m[0]
                    flag = '' if r.get('sources_agree', True) else '*'
                    note = ' ({})'.format(r['note']) if r.get('note') else ''
                    cells.append('{:.2f}±{:.2f} / {:.2f}{}{}'.format(r['r2_mean'], r['r2_std'],
                                                                   r['rmse_mean'], flag, note))
                lines.append('| {} | {} | {} |'.format(entry, mods, ' | '.join(cells)))
            lines.append('')
    (out_dir / 'comparison.md').write_text('\n'.join(lines))
    return out_dir


def main():
    p = argparse.ArgumentParser('Compare with the YieldSAT paper')
    p.add_argument('--runs_root', required=True)
    p.add_argument('--paper_csv', default='results/yieldsat/paper_benchmark.csv')
    p.add_argument('--out_dir', default='results/yieldsat/paper_comparison')
    a = p.parse_args()
    table = build(load_paper(a.paper_csv), load_ours(a.runs_root))
    out = write(table, a.out_dir)
    # archive the per-experiment aggregates next to the tables
    import shutil
    for agg in Path(a.runs_root).glob('paper/*/*/*/*/aggregate.json'):
        rel = agg.parent.relative_to(Path(a.runs_root) / 'paper')
        dst = out / 'aggregates' / rel.parent / '{}.json'.format(rel.name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(agg, dst)
    print('wrote', out)


if __name__ == '__main__':
    main()
