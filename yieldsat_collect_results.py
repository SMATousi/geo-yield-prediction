# --------------------------------------------------------
# Collect YieldSAT run outputs into a versionable results directory.
#
#   python yieldsat_collect_results.py --runs_dir $YIELDSAT_ARTIFACT_ROOT/runs \
#       --out_dir results/yieldsat
#
# Per run: params.json (all CLI arguments + model config + data policy),
# history.csv (per-epoch loss/validation), test_metrics.json, group_metrics.csv
# (overall / per country / per crop / per country-crop), report.json and the
# georeferenced prediction maps. Across runs: summary.csv and README.md.
# Checkpoints and per-cell prediction arrays stay in the artifact root.
# --------------------------------------------------------

import argparse
import csv
import json
import shutil
from pathlib import Path

# purpose of each run in the 2026-09-28 pilot set (run_all.sh)
PURPOSE = {
    'pilot_germany_farm': 'Single-country pilot (Germany)',
    'pooled_farm': 'Pooled 4-country supervised, farm-held-out',
    'pooled_block20': 'Pooled 4-country supervised, geographic-block-held-out',
    'loco_uruguay': 'Leave-one-country-out: train AR/BR/DE, test Uruguay',
    'pretrain_pooled': 'Yield-free pretraining (masked observation + last-valid forecast)',
    'uruguay_scratch_f0.1': 'YS-10 transfer: Uruguay from scratch, 10% labels',
    'uruguay_pretrained_f0.1': 'YS-10 transfer: Uruguay from pretrained sensors, 10% labels',
    'uruguay_scratch_f1.0': 'YS-10 transfer: Uruguay from scratch, 100% labels',
    'uruguay_pretrained_f1.0': 'YS-10 transfer: Uruguay from pretrained sensors, 100% labels',
}
ORDER = list(PURPOSE)

PARAM_KEYS = ('countries', 'split', 'mode', 'train_fraction', 'init_sensor_ckpt', 'cutoff_mode',
              'cutoff_days', 'streams', 'soil_uncertainty', 'aspect_encoding', 'norm_pooling',
              'no_crop_context', 'embed_dim', 'num_latents', 'depth', 'num_heads', 'modality_embed',
              'modality_dropout', 'epochs', 'steps_per_epoch', 'batch_size', 'lr', 'weight_decay',
              'warmup_epochs', 'field_alpha', 'val_rows_per_field', 'test_rows_per_field',
              'backend', 'seed')


def _fmt(v, nd=3):
    if isinstance(v, float):
        return round(v, nd)
    if isinstance(v, list):
        return ' '.join(str(x) for x in v)
    return v


def _group_rows(test):
    rows = []
    groups = [('overall', 'all', test['overall'])]
    groups += [('country', k, v) for k, v in test['per_country'].items()]
    groups += [('crop', k, v) for k, v in test['per_crop'].items()]
    groups += [('country_crop', k, v) for k, v in test['per_country_crop'].items()]
    for level, name, b in groups:
        rows.append({
            'level': level, 'group': name,
            'test_cells': b['pixel']['n'], 'test_seasons': b['field_level']['n'],
            'pixel_rmse': _fmt(b['pixel']['rmse']), 'pixel_mae': _fmt(b['pixel']['mae']),
            'pixel_r2': _fmt(b['pixel']['r2']), 'pixel_pearson_r2': _fmt(b['pixel']['pearson_r2']),
            'pixel_bias': _fmt(b['pixel']['bias']),
            'field_balanced_rmse': _fmt(b['field_balanced']['rmse']),
            'field_level_rmse': _fmt(b['field_level']['rmse']),
            'field_level_r2': _fmt(b['field_level']['r2']),
        })
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def collect(runs_dir, out_dir, splits_dir=None):
    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted((p.parent.name for p in runs_dir.glob('*/report.json')),
                   key=lambda n: (ORDER.index(n) if n in ORDER else len(ORDER), n))
    summary = []
    for name in names:
        src = runs_dir / name
        r = json.loads((src / 'report.json').read_text())
        dst = out_dir / 'runs' / name
        (dst / 'maps').mkdir(parents=True, exist_ok=True)
        args = r['args']
        params = {
            'run': name, 'purpose': PURPOSE.get(name, ''),
            'contract': r['contract'], 'data_mode': r['data_mode'],
            'split': r['split'], 'label_policy': r['label_policy'],
            'seasons': {'train': r['train_seasons'], 'val': r['val_seasons'], 'test': r['test_seasons']},
            'streams': r['streams'], 'excluded_sources': r['excluded_sources'],
            'args': {k: v for k, v in args.items() if k not in ('source_root', 'artifact_root', 'output_dir')},
            'init_from': Path(args['init_sensor_ckpt']).parent.name if args.get('init_sensor_ckpt') else None,
        }
        (dst / 'params.json').write_text(json.dumps(params, indent=1))
        _write_csv(dst / 'history.csv', [{k: _fmt(v, 5) for k, v in h.items()} for h in r['history']])
        shutil.copy(src / 'report.json', dst / 'report.json')
        for tif in src.glob('pred_*.tif'):
            shutil.copy(tif, dst / 'maps' / tif.name)
        row = {'run': name, 'purpose': PURPOSE.get(name, '')}
        row.update({k: _fmt(args.get(k)) for k in PARAM_KEYS})
        row['init_sensor_ckpt'] = params['init_from'] or ''
        row['split_label'] = r['split']['label']
        row['train_seasons'], row['val_seasons'], row['test_seasons'] = (
            r['train_seasons'], r['val_seasons'], r['test_seasons'])
        row['best_epoch'] = (r.get('best_val') or {}).get('epoch', '')
        row['final_train_loss'] = _fmt(r['history'][-1]['train_loss'])
        test = r.get('test')
        if test:
            (dst / 'test_metrics.json').write_text(json.dumps(test, indent=1))
            _write_csv(dst / 'group_metrics.csv', _group_rows(test))
            o = test['overall']
            row.update({
                'test_cells': test['rows_evaluated'],
                'pixel_rmse': _fmt(o['pixel']['rmse']), 'pixel_r2': _fmt(o['pixel']['r2']),
                'pixel_bias': _fmt(o['pixel']['bias']),
                'field_balanced_rmse': _fmt(o['field_balanced']['rmse']),
                'field_level_rmse': _fmt(o['field_level']['rmse']),
                'field_level_r2': _fmt(o['field_level']['r2']),
                'macro_country_pixel_rmse': _fmt(test['macro_country']['pixel_rmse']),
                'macro_country_pixel_r2': _fmt(test['macro_country']['pixel_r2']),
            })
        row.update({'samples_per_second': r['io']['samples_per_second'],
                    'wall_seconds': r['resources']['wall_seconds'],
                    'peak_gpu_gb': r['resources']['peak_gpu_gb']})
        summary.append(row)
    keys = []
    for row in summary:
        keys += [k for k in row if k not in keys]
    _write_csv(out_dir / 'summary.csv', [{k: row.get(k, '') for k in keys} for row in summary])
    if splits_dir is not None:
        (out_dir / 'splits').mkdir(exist_ok=True)
        for p in Path(splits_dir).glob('*.json'):
            shutil.copy(p, out_dir / 'splits' / p.name)
    if (runs_dir / 'run_all.sh').exists():
        shutil.copy(runs_dir / 'run_all.sh', out_dir / 'run_all.sh')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser('Collect YieldSAT results')
    p.add_argument('--runs_dir', required=True)
    p.add_argument('--out_dir', default='results/yieldsat')
    p.add_argument('--splits_dir', default=None)
    a = p.parse_args()
    for row in collect(a.runs_dir, a.out_dir, a.splits_dir):
        print(row['run'], row.get('pixel_rmse', '-'), row.get('field_level_rmse', '-'))
