"""Reproduction report (spec/yieldsat-paper-reproduction.md RP-01).

For every experiment directory paper/<group>/<inputs>/<tag>_seed<s>/<pair>
under a run root: fold mean ± std of pixel and field R² (the paper's stated
metric) and pooled out-of-fold R², next to the paper's row for a chosen
paper model / modality / fusion.

    python yieldsat_repro_report.py --root runs/repro_r1 --paper_model LSTM \
        --paper_modalities S2+ADM --paper_fusion input --out results/repro_r1.md
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from yieldsat_pooled_metrics import pooled

PROTO = {'cv10': 'CV10', 'loyo': 'LOYO', 'loro': 'LORO'}


def paper_rows(path, model, modalities, fusion):
    out = {}
    for r in csv.DictReader(open(path)):
        if r['model'] != model or r['modalities'] != modalities:
            continue
        if fusion and fusion not in r['fusion'].lower():
            continue
        out[(r['protocol'], r['level'], r['pair'])] = (float(r['r2_mean']), float(r['r2_std'] or 'nan'))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', required=True)
    p.add_argument('--paper', default='results/yieldsat/paper_benchmark.csv')
    p.add_argument('--paper_model', default='LSTM')
    p.add_argument('--paper_modalities', default='S2+ADM')
    p.add_argument('--paper_fusion', default='input')
    p.add_argument('--title', default='Reproduction')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    ref = paper_rows(a.paper, a.paper_model, a.paper_modalities, a.paper_fusion)
    L = ['# {}'.format(a.title), '',
         'Generated {} by `yieldsat_repro_report.py` (spec/yieldsat-paper-reproduction.md). Fold mean ± std over '
         'folds (the paper\'s stated metric) and pooled out-of-fold R², next to the paper\'s "{} / {} / {}" rows.'.format(
             time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()), a.paper_modalities, a.paper_fusion, a.paper_model), '',
         '| Experiment | Row | Folds | Pixel R² fold mean ± std | Pixel pooled | Paper pixel | Field R² fold mean ± std | Field pooled | Paper field |',
         '|---|---|---|---|---|---|---|---|---|']
    for exp in sorted(Path(a.root).glob('paper/*/*/*/*')):
        reps = [json.loads(f.read_text()) for f in sorted(exp.glob('fold*/report.json'))]
        if not reps:
            continue
        pix = [r['test']['overall']['pixel']['r2'] for r in reps]
        fld = [r['test']['overall']['field_level']['r2'] for r in reps]
        po = pooled(exp)
        group, tag, pair = exp.parts[-4], exp.parts[-2], exp.parts[-1]
        proto = PROTO[group.split('_')[0]]
        rp = ref.get((proto, 'pixel', pair), (np.nan, np.nan))
        rf = ref.get((proto, 'field', pair), (np.nan, np.nan))
        L.append('| {} | {} {} | {} | {:.2f} ± {:.2f} | {:.2f} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | {:.2f} | {:.2f} ± {:.2f} |'.format(
            tag, pair, proto, len(reps), np.mean(pix), np.std(pix), po['pixel_r2'], rp[0], rp[1],
            np.mean(fld), np.std(fld), po['field_r2'], rf[0], rf[1]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
