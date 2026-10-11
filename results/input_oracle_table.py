"""IC-A table (spec/yieldsat-input-information.md §2): within-field oracle vs models vs noise ceiling, per pair,
on the same field seasons (those the oracle could fit, >= 200 cells).

    python results/input_oracle_table.py <oracle_fields.jsonl> <results root with paper/> results/input_oracle.json
"""
import glob
import json
import sys

import numpy as np

ORACLE, ROOT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PAIRS = ['ARG-C', 'ARG-S', 'ARG-W', 'BRA-C', 'BRA-S', 'BRA-W', 'GER-R', 'GER-W', 'URG-S']
MODELS = ['ours-p3nbr-dense-fc-rel10', 'ours-p3nbr-dense', 'paper_mmgf', 'lgbm-f0', 'tabm-f1']
NOISE = json.load(open('results/noise_ceiling.json'))

rows = [json.loads(l) for l in open(ORACLE)]
fields = {}
for r in rows:
    fields.setdefault(r['pair'], set()).add(r['field'])


def model_within(tag, pair, keep):
    sse = sst = 0.0
    n = 0
    for f in glob.glob('%s/paper/cv10_season_noval_s0/s2_adm/%s_seed0/%s/fold*/test_predictions.npz' % (ROOT, tag, pair)):
        z = np.load(f)
        names = z['season_names'].astype(str)
        for si in np.unique(z['season']):
            if names[si] not in keep:
                continue
            sel = (z['season'] == si) & np.isfinite(z['pred']) & np.isfinite(z['target'])
            p, y = z['pred'][sel].astype(float), z['target'][sel].astype(float)
            pc, yc = p - p.mean(), y - y.mean()
            sse += np.sum((pc - yc) ** 2); sst += np.sum(yc ** 2); n += 1
    return (1 - sse / sst if sst > 0 else None), n


res = {}
for pair in PAIRS:
    pr = [r for r in rows if r['pair'] == pair]
    if not pr:
        continue
    row = {'fields': len(pr), 'cells': int(sum(r['n'] for r in pr)),
           'noise_ceiling': NOISE[pair]['ceiling_within_r2']}
    for B in ('B4', 'B8', 'B16'):
        sub = [r[B] for r in pr if B in r]
        for m in ('gbm', 'ridge', 'coord'):
            row['%s_%s' % (m, B)] = 1 - sum(x[m][0] for x in sub) / sum(x[m][1] for x in sub)
        row['fields_' + B] = len(sub)
    for tag in MODELS:
        row[tag], row[tag + '_fields'] = model_within(tag, pair, fields[pair])
    res[pair] = row
    print(pair, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
