"""Test-time input re-normalization for saved fold checkpoints (AdaBN at the input level).

For every fold of a finished run, the saved ``checkpoint_best.pth`` predicts the test fold twice:
  train  inputs standardized with the training fold's statistics (as trained; reproduces report.json)
  test   inputs standardized with the test fold's own feature statistics (temporal and static means /
         stds from the test seasons' inputs only); the target scaling stays the training fold's, so
         no test label is used. Transductive: it uses the held-out seasons' inputs, which exist at
         prediction time.
Writes <out_root>/<pair>/fold<ii>.npz (pred_train, pred_test, target, season) and fold<ii>.json.

    YIELDSAT_NUM_SLOTS=72 YIELDSAT_CACHE_NAME=cache_dense YIELDSAT_NBR_NAME=neighbourhood_dense \\
    python yieldsat_testnorm_eval.py --artifact_root A --run_root R/paper/loyo_na_noval_s0/s2_adm/ours-p3nbr-dense_seed0 \\
        --pairs GER-W --out_root O
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset.yieldsat_dataset import NormalizerView, YieldSATNormalizer, YieldSATPointDataset
from dataset.yieldsat_schema import NBR_NAME, NUM_TIME_SLOTS
from dataset.yieldsat_source import load_country_index
from dataset.yieldsat_splits import load_field_table, load_split
from main_yieldsat_finetune import predict
from models_yieldsat import YieldSATPointModel


class TestFeatureNormalizer:
    """Feature statistics from the test fold, target statistics from the training fold."""

    def __init__(self, train, test):
        self.train, self.test = train, test

    def get(self, country):
        st = dict(self.train.get(country))
        t = self.test.get(country)
        for k in ('temporal_mean', 'temporal_std', 'static_mean', 'static_std'):
            st[k] = t[k]
        return st


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def run_fold(fold_dir, artifact_root, device, out_dir):
    ck = torch.load(fold_dir / 'checkpoint_best.pth', map_location='cpu', weights_only=False)
    a = argparse.Namespace(**ck['args'])
    table = load_field_table(artifact_root, None, a.countries, check_source=False)
    split = load_split(artifact_root, a.split)
    by_season = {f['season_id']: f for f in table['fields']}
    parts = {p: [by_season[s] for s in split['partitions'][p] if s in by_season] for p in ('train', 'test')}
    if a.crops:
        parts = {p: [f for f in v if f['crop'] in a.crops] for p, v in parts.items()}
    targets = {c: load_country_index(artifact_root, None, c, False)['rows']['target'] for c in a.countries}
    fields = lambda part: [(f['country'], f['field_code'], targets[f['country']][f['row_start']:f['row_end']])
                           for f in parts[part]]
    train_norm = YieldSATNormalizer.fit(artifact_root, fields('train'), pooling=a.norm_pooling)
    test_norm = YieldSATNormalizer.fit(artifact_root, fields('test'), pooling=a.norm_pooling)
    views = {'train': NormalizerView(train_norm, features=a.normalization, target=a.target_normalization),
             'test': NormalizerView(TestFeatureNormalizer(train_norm, test_norm), features=a.normalization,
                                    target=a.target_normalization)}
    common = dict(streams=a.streams, backend='cache', cutoff_mode=a.cutoff_mode, cutoff_days=a.cutoff_days,
                  soil_uncertainty=a.soil_uncertainty, aspect_encoding=a.aspect_encoding, seed=a.seed,
                  fill_value=a.fill_value, check_source=False, weather_first_slot=a.weather_first_slot,
                  neighbourhood_root=str(Path(artifact_root) / NBR_NAME) if a.neighbourhood else None)
    out = {}
    model = None
    for name, view in views.items():
        ds = YieldSATPointDataset(None, artifact_root, parts['test'], view, max_rows_per_field=None, **common)
        if model is None:
            single = a.crops is not None and len(a.crops) == 1
            model = YieldSATPointModel(ds.layout, embed_dim=a.embed_dim, num_latents=a.num_latents, depth=a.depth,
                                       num_heads=a.num_heads, modality_embed=a.modality_embed,
                                       modality_dropout=a.modality_dropout,
                                       use_crop_context=not a.no_crop_context and not single, fusion=a.fusion,
                                       cross_attn_layers=a.cross_attn_layers, level_head=a.level_head,
                                       level_weight=a.level_weight, early_hidden=a.early_hidden,
                                       early_depth=a.early_depth, num_slots=NUM_TIME_SLOTS).to(device)
            model.load_state_dict(ck['model'])
        a.num_workers = min(a.num_workers, 3)
        p, y, s, _, _ = predict(model, ds, a, device)
        out[name] = (p, y, s)
    p_tr, y, s = out['train']
    p_te = out['test'][0]
    ok = np.isfinite(p_tr) & np.isfinite(y)
    res = {'split': a.split, 'holdout': split.get('holdout'), 'n': int(ok.sum()),
           'reported_r2': json.loads((fold_dir / 'report.json').read_text())['test']['overall']['pixel']['r2']}
    for name, p in (('train', p_tr), ('test', p_te)):
        res[name] = {'r2': r2(y[ok], p[ok]), 'bias': float(np.mean(p[ok] - y[ok]))}
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / '{}.npz'.format(fold_dir.name), pred_train=p_tr[ok], pred_test=p_te[ok],
                        target=y[ok], season=s[ok])
    (out_dir / '{}.json'.format(fold_dir.name)).write_text(json.dumps(res))
    print(fold_dir.parent.name, fold_dir.name, json.dumps(res), flush=True)


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--artifact_root', required=True)
    pa.add_argument('--run_root', required=True, help='.../paper/<group>/s2_adm/<tag>_seed0')
    pa.add_argument('--pairs', nargs='+', required=True)
    pa.add_argument('--out_root', required=True)
    a = pa.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for pair in a.pairs:
        for fold_dir in sorted(Path(a.run_root, pair).glob('fold*')):
            if (Path(a.out_root) / pair / '{}.json'.format(fold_dir.name)).exists():
                continue
            run_fold(fold_dir, a.artifact_root, device, Path(a.out_root) / pair)


if __name__ == '__main__':
    main()
