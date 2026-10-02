"""Matched-cell comparison of image runs against point runs (YI-05).

Image and point runs share fold manifests and the results layout
``paper/<protocol group>/<inputs>/<tag>_seed<s>/<pair>/foldNN/``. For every
image fold with a point counterpart, both ``test_predictions.npz`` files are
joined on (field season, grid row, grid col): the cells of test tiles. Both
models are scored on exactly those cells (pixel and field level). Point
coverage (the share of the point model's test cells that lie in tiles) is
reported, because the point model's own whole-field scores are on more cells.

    python yieldsat_image_compare.py --image-root /data/YieldSAT/yieldsat_results/image_full \
        --point-root /data/YieldSAT/yieldsat_results/before_full --out results/yieldsat/image_vs_point
"""

import argparse
import collections
import csv
import json
import re
from pathlib import Path

import numpy as np

from util.yieldsat_eval import _field_level, regression_metrics

FOLD_RE = re.compile(r'paper/(?P<group>[^/]+)/(?P<inputs>[^/]+)/(?P<tag>.+)_seed(?P<seed>\d+)/'
                     r'(?P<pair>[^/]+)/fold(?P<fold>\d+)$')


def _keys(npz, codes):
    """Integer cell keys: shared season-name code, grid row and col packed."""
    names = npz['season_names'].astype(str)
    season_code = np.array([codes.setdefault(n, len(codes)) for n in names], np.int64)
    return (season_code[npz['season']] << 40) | (npz['grid_row'].astype(np.int64) << 20) \
        | npz['grid_col'].astype(np.int64)


def match_fold(image_npz, point_npz):
    """-> dict of matched arrays and coverage, or None if nothing matches."""
    codes = {}
    ik, pk = _keys(image_npz, codes), _keys(point_npz, codes)
    if len(np.unique(ik)) != len(ik) or len(np.unique(pk)) != len(pk):
        raise ValueError('duplicate cells in a prediction file')
    common, ii, pi = np.intersect1d(ik, pk, return_indices=True)
    if not len(common):
        return None
    season = np.unique(common >> 40, return_inverse=True)[1]
    inv = {v: k for k, v in codes.items()}
    season_name = np.array([inv[int(c)] for c in (common >> 40)])
    y_img, y_pt = image_npz['target'][ii], point_npz['target'][pi]
    if not np.allclose(y_img, y_pt, atol=1e-4, equal_nan=True):
        raise ValueError('targets of matched cells disagree: not the same cells')
    return {'target': y_img, 'image': image_npz['pred'][ii], 'point': point_npz['pred'][pi],
            'season': season, 'season_name': season_name, 'n_image': len(ik), 'n_point': len(pk),
            'n_matched': len(common)}


def score(m):
    out = {}
    for model in ('image', 'point'):
        px = regression_metrics(m['target'], m[model])
        fl = _field_level(m['target'], m[model], m['season'])
        out[model] = {'pixel_r2': px['r2'], 'pixel_rmse': px['rmse'], 'field_r2': fl.get('r2'),
                      'field_rmse': fl.get('rmse'), 'fields': fl.get('n', 0)}
    out['n_matched'] = m['n_matched']
    out['point_coverage'] = m['n_matched'] / m['n_point']
    out['image_coverage'] = m['n_matched'] / m['n_image']
    return out


def pool(matches):
    """Pool matched cells of all folds of one experiment (the paper's metric:
    one R²/RMSE over every held-out cell; field level over field seasons)."""
    names = np.concatenate([m['season_name'] for m in matches])
    season = np.unique(names, return_inverse=True)[1]
    cat = lambda k: np.concatenate([m[k] for m in matches])
    m = {'target': cat('target'), 'image': cat('image'), 'point': cat('point'), 'season': season,
         'n_matched': sum(x['n_matched'] for x in matches), 'n_point': sum(x['n_point'] for x in matches),
         'n_image': sum(x['n_image'] for x in matches)}
    return score(m)


def collect(image_root, point_root, image_tag='image', point_tag='ours'):
    """One row per experiment (group, inputs, seed, pair), pooled over the
    folds where both an image and a point prediction exist. Experiments are
    processed one at a time, so memory holds one experiment's cells."""
    groups, missing, rows, n = {}, [], [], 0
    for f in sorted(Path(image_root).glob('paper/*/*/*/*/fold*/test_predictions.npz')):
        rel = f.parent.relative_to(image_root).as_posix()
        g = FOLD_RE.match(rel)
        if not g or g['tag'] != image_tag:
            continue
        key = (g['group'], g['inputs'], int(g['seed']), g['pair'])
        groups.setdefault(key, []).append((f, rel))
    for key, files in sorted(groups.items()):
        matches = []
        for f, rel in files:
            prel = rel.replace('/{}_seed'.format(image_tag), '/{}_seed'.format(point_tag), 1)
            pf = Path(point_root) / prel / 'test_predictions.npz'
            if not pf.exists():
                missing.append(prel)
                continue
            m = match_fold(np.load(f), np.load(pf))
            if m is None:
                missing.append(prel + ' (no common cells)')
                continue
            matches.append(m)
            n += 1
        if matches:
            rows.append(dict(group=key[0], inputs=key[1], seed=key[2], pair=key[3], folds=len(matches),
                             **pool(matches)))
        if len(rows) % 50 == 0:
            print('experiments', len(rows), 'folds', n, flush=True)
    return rows, missing


def summarize(rows):
    """Mean over seeds of the pooled scores, per (protocol group, inputs, pair)."""
    by = collections.defaultdict(list)
    for r in rows:
        by[(r['group'], r['inputs'], r['pair'])].append(r)
    out = []
    for (group, inputs, pair), rs in sorted(by.items()):
        rec = {'group': group, 'inputs': inputs, 'pair': pair, 'seeds': len(rs),
               'point_coverage': float(np.mean([r['point_coverage'] for r in rs]))}
        for model in ('image', 'point'):
            for k in ('pixel_r2', 'pixel_rmse', 'field_r2', 'field_rmse'):
                vals = [r[model][k] for r in rs if r[model][k] is not None and np.isfinite(r[model][k])]
                rec['{}_{}'.format(model, k)] = float(np.mean(vals)) if vals else float('nan')
        rec['delta_pixel_rmse'] = rec['image_pixel_rmse'] - rec['point_pixel_rmse']
        out.append(rec)
    return out


PROTO_TITLE = (('cv10', 'CV10'), ('loro_province', 'LORO, provinces (Argentina)'), ('loro_na', 'LORO, farm regions'),
               ('loyo', 'LOYO'))


def write_markdown(summary, path, title, note=''):
    """One table per protocol group: image vs point on identical cells."""
    lines = ['# {}'.format(title), '', note, '',
             '- Both models are scored on **exactly the same cells**: the valid cells of the image '
             'test tiles, matched by field season and grid row/col. "Coverage" is the share of the '
             'point model\'s test cells that lie in image tiles.',
             '- **Metric = the paper\'s computation:** per experiment (pair × protocol × policy × inputs '
             '× seed) the matched cells of all folds are pooled and R²/RMSE computed once (pixel: every '
             'cell; field: season means). Values are means over seeds. RMSE in t/ha. **Δ RMSE** = image '
             '− point (negative: image better).', '']
    for prefix, name in PROTO_TITLE:
        rows = [r for r in summary if r['group'].startswith(prefix)]
        if not rows:
            continue
        lines += ['## {}'.format(name), '',
                  '| Pair | Policy | Inputs | Seeds | Coverage | Pixel R² image / point | '
                  'Pixel RMSE image / point | Δ RMSE | Field R² image / point | Field RMSE image / point |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for r in sorted(rows, key=lambda r: (r['group'].split('_')[-2], r['inputs'], r['pair'])):
            policy = r['group'].split('_')[-2]
            lines.append('| {} | {} | {} | {} | {:.0f}% | {:.2f} / {:.2f} | {:.2f} / {:.2f} | {:+.2f} | '
                         '{:.2f} / {:.2f} | {:.2f} / {:.2f} |'.format(
                             r['pair'], policy, {'s2': 'S2', 's2_adm': 'S2+ADM'}[r['inputs']], r['seeds'],
                             100 * r['point_coverage'], r['image_pixel_r2'], r['point_pixel_r2'],
                             r['image_pixel_rmse'], r['point_pixel_rmse'], r['delta_pixel_rmse'],
                             r['image_field_r2'], r['point_field_r2'], r['image_field_rmse'],
                             r['point_field_rmse']))
        lines.append('')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(lines).replace('nan', '–'))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--image-root', default=None)
    p.add_argument('--point-root', default=None)
    p.add_argument('--image-tag', default='image')
    p.add_argument('--point-tag', default='ours')
    p.add_argument('--out', required=True)
    p.add_argument('--from_folds', default=None, help='rebuild tables from a saved folds.json')
    p.add_argument('--md', default=None, help='also write a markdown report here')
    p.add_argument('--title', default='Image vs point on identical cells')
    p.add_argument('--note', default='')
    a = p.parse_args()
    if a.from_folds:
        rows, missing = json.loads(Path(a.from_folds).read_text()), []
    else:
        rows, missing = collect(a.image_root, a.point_root, a.image_tag, a.point_tag)
    summary = summarize(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'folds.json').write_text(json.dumps(rows, indent=1))
    (out / 'missing_point_folds.json').write_text(json.dumps(missing, indent=1))
    if summary:
        with open(out / 'summary.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
    if a.md:
        write_markdown(summary, a.md, a.title, a.note)
    print('{} experiments (pooled over matched folds), {} image folds without a point counterpart'.format(
        len(rows), len(missing)))
    print('%-28s %-7s %-6s %5s %6s | %-13s | %-13s | %-13s' % (
        'protocol group', 'inputs', 'pair', 'seeds', 'cover', 'pixel R2 i/p', 'pixel RMSE i/p',
        'field R2 i/p'))
    for r in summary:
        print('%-28s %-7s %-6s %5d %5.0f%% | %5.2f / %5.2f | %5.2f / %5.2f | %5.2f / %5.2f' % (
            r['group'], r['inputs'], r['pair'], r['seeds'], 100 * r['point_coverage'],
            r['image_pixel_r2'], r['point_pixel_r2'], r['image_pixel_rmse'], r['point_pixel_rmse'],
            r['image_field_r2'], r['point_field_r2']))


if __name__ == '__main__':
    main()
