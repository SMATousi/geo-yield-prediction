# --------------------------------------------------------
# Real-data training entry point for the YieldSAT preprocessed contract
# (YS-08). Point mode: one grid-cell history -> one per-cell yield (t/ha).
#
#   python main_yieldsat_finetune.py --data_contract yieldsat_preprocessed_v1 \
#       --source_root /path/Preprocessed --artifact_root /path/artifacts \
#       --countries Germany --split germany_farm_s0 --device cuda
#
# --mode pretrain runs yield-free point-mode pretraining (masked observation
# + last-valid forecast) on the *training* partition only and saves a sensor
# checkpoint; --init_sensor_ckpt loads one (encoders + fusion) before
# fine-tuning. Prerequisites (index, cache, geometry, split) come from
# yieldsat_prepare.py.
# --------------------------------------------------------

import argparse
import json
import math
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.yieldsat_dataset import (
    CUTOFF_MODES, FieldBalancedBatchSampler, SequentialBatchSampler, YieldSATNormalizer,
    YieldSATPointDataset, collate_point_batch,
)
from dataset.yieldsat_schema import CONTRACT_KEY, COUNTRIES, PROVENANCE, STREAMS
from dataset.yieldsat_splits import load_field_table, load_split
from dataset.yieldsat_source import load_country_index
from models_yieldsat import YieldSATPointModel
from util.yieldsat_eval import evaluate_predictions, reconstruct_field, write_geotiff
from yieldsat_objectives import OBJECTIVE_ROUTING, YieldSATPointPretrainer

CONTRACTS = (CONTRACT_KEY, 'downloader_native_layers_v1')


def get_args_parser():
    p = argparse.ArgumentParser('YieldSAT point-mode training', add_help=False)
    p.add_argument('--data_contract', required=True, choices=CONTRACTS)
    p.add_argument('--source_root', default=os.environ.get('YIELDSAT_SOURCE_ROOT'))
    p.add_argument('--artifact_root', default=os.environ.get('YIELDSAT_ARTIFACT_ROOT'))
    p.add_argument('--countries', nargs='+', choices=COUNTRIES, required=True)
    p.add_argument('--split', required=True, help='split manifest name under artifact_root/splits')
    p.add_argument('--mode', default='finetune', choices=['finetune', 'pretrain'])
    p.add_argument('--backend', default='cache', choices=['cache', 'h5'])
    p.add_argument('--streams', nargs='+', default=list(STREAMS), choices=list(STREAMS))
    p.add_argument('--cutoff_mode', default='before_harvest', choices=CUTOFF_MODES)
    p.add_argument('--cutoff_days', type=int, default=30)
    p.add_argument('--soil_uncertainty', default='none', choices=['none', 'ancillary'])
    p.add_argument('--aspect_encoding', default='raw', choices=['raw', 'cyclic'])
    p.add_argument('--norm_pooling', default='pooled', choices=['pooled', 'per_country'])
    p.add_argument('--no_crop_context', action='store_true')
    p.add_argument('--train_fraction', type=float, default=1.0,
                   help='keep this fraction of training field seasons (label budget)')
    # model
    p.add_argument('--embed_dim', type=int, default=128)
    p.add_argument('--num_latents', type=int, default=8)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--modality_embed', type=int, default=32)
    p.add_argument('--modality_dropout', type=float, default=0.1)
    p.add_argument('--init_sensor_ckpt', default='')
    # optimisation
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--steps_per_epoch', type=int, default=500)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=float, default=1.0)
    p.add_argument('--field_alpha', type=float, default=0.5,
                   help='season sampling weight n_rows**alpha (0 field-balanced, 1 pixel)')
    p.add_argument('--block_size', type=int, default=None,
                   help='contiguous rows per draw (default 1 for cache, 64 for h5)')
    p.add_argument('--val_rows_per_field', type=int, default=500)
    p.add_argument('--test_rows_per_field', type=int, default=None,
                   help='None evaluates every held-out cell')
    p.add_argument('--eval_batch_size', type=int, default=4096)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output_dir', default='./output_dir/yieldsat')
    p.add_argument('--save_maps', type=int, default=3, help='test fields written as GeoTIFF')
    return p


def _loader(ds, batch_sampler, workers):
    return DataLoader(ds, batch_sampler=batch_sampler, collate_fn=collate_point_batch,
                      num_workers=workers, pin_memory=True, persistent_workers=False)


def _to(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
        else:
            out[k] = v.to(device, non_blocking=True)
    return out


@torch.no_grad()
def predict(model, ds, args, device):
    model.eval()
    loader = _loader(ds, SequentialBatchSampler(len(ds), args.eval_batch_size), args.num_workers)
    preds, targets, seasons, rows, cols = [], [], [], [], []
    for batch in loader:
        b = _to(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            p = model(b, apply_dropout=False).float()
        preds.append((p * b['target_std'] + b['target_mean']).cpu().numpy())
        targets.append(batch['target_raw'].numpy())
        seasons.append(batch['season'].numpy())
        rows.append(batch['grid_row'].numpy())
        cols.append(batch['grid_col'].numpy())
    cat = np.concatenate
    return cat(preds), cat(targets), cat(seasons), cat(rows), cat(cols)


def main(args):
    if args.data_contract != CONTRACT_KEY:
        raise SystemExit('{} ingestion is not implemented; this entry point serves {}'.format(
            args.data_contract, CONTRACT_KEY))
    if not args.source_root or not args.artifact_root:
        raise SystemExit('--source_root and --artifact_root are required')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    table = load_field_table(args.artifact_root, args.source_root, args.countries)
    split = load_split(args.artifact_root, args.split, fingerprints=table['fingerprints'])
    if not set(args.countries) <= set(split['countries']):
        raise SystemExit('split {} covers {}, not the requested {}'.format(
            args.split, split['countries'], args.countries))
    # a country subset of a pooled split keeps that split's partitions, so
    # single-country runs are comparable with pooled/pretrained ones
    by_season = {f['season_id']: f for f in table['fields']}
    parts = {p: [by_season[s] for s in split['partitions'][p] if s in by_season]
             for p in ('train', 'val', 'test')}
    rng = np.random.default_rng(args.seed)
    if args.train_fraction < 1.0:
        keep = max(1, int(round(args.train_fraction * len(parts['train']))))
        sel = rng.choice(len(parts['train']), keep, replace=False)
        parts['train'] = [parts['train'][i] for i in sorted(sel)]

    # train-only normalization
    targets = {c: load_country_index(args.artifact_root, args.source_root, c)['rows']['target']
               for c in args.countries}
    train_fields = [(f['country'], f['field_code'], targets[f['country']][f['row_start']:f['row_end']])
                    for f in parts['train']]
    normalizer = YieldSATNormalizer.fit(args.artifact_root, train_fields, pooling=args.norm_pooling)
    (out_dir / 'normalizer.json').write_text(json.dumps(normalizer.to_json()))

    common = dict(streams=args.streams, backend=args.backend, cutoff_mode=args.cutoff_mode,
                  cutoff_days=args.cutoff_days, soil_uncertainty=args.soil_uncertainty,
                  aspect_encoding=args.aspect_encoding, seed=args.seed)
    train_ds = YieldSATPointDataset(args.source_root, args.artifact_root, parts['train'], normalizer, **common)
    val_ds = YieldSATPointDataset(args.source_root, args.artifact_root, parts['val'], normalizer,
                                  max_rows_per_field=args.val_rows_per_field, **common) if parts['val'] else None
    block = args.block_size or (1 if args.backend == 'cache' else 64)

    model = YieldSATPointModel(train_ds.layout, embed_dim=args.embed_dim, num_latents=args.num_latents,
                               depth=args.depth, num_heads=args.num_heads,
                               modality_embed=args.modality_embed,
                               modality_dropout=args.modality_dropout,
                               use_crop_context=not args.no_crop_context).to(device)
    transfer = None
    if args.init_sensor_ckpt:
        transfer = model.load_sensor_state_dict(torch.load(args.init_sensor_ckpt, map_location='cpu', weights_only=True))
        print('loaded sensor checkpoint:', transfer['loaded'], 'tensors')
    trainable = YieldSATPointPretrainer(model).to(device) if args.mode == 'pretrain' else model
    opt = torch.optim.AdamW(trainable.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * args.steps_per_epoch
    warm = int(args.warmup_epochs * args.steps_per_epoch)

    def lr_at(step):
        if step < warm:
            return args.lr * (step + 1) / warm
        return args.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total_steps - warm)))

    sampler = FieldBalancedBatchSampler(train_ds.season_ranges, args.batch_size, args.steps_per_epoch,
                                        alpha=args.field_alpha, block_size=block, seed=args.seed)
    history, best, step = [], None, 0
    samples_seen, data_time = 0, 0.0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        trainable.train()
        t0 = time.time()
        tl = time.time()
        losses = []
        for batch in _loader(train_ds, sampler, args.num_workers):
            data_time += time.time() - tl
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            b = _to(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                if args.mode == 'pretrain':
                    loss, parts_loss = trainable(b)
                else:
                    _, loss = model.loss(b)
            opt.zero_grad(set_to_none=True)
            loss.float().backward()
            torch.nn.utils.clip_grad_norm_(trainable.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            samples_seen += batch['target'].shape[0]
            step += 1
            tl = time.time()
        rec = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'lr': lr_at(step - 1),
               'seconds': round(time.time() - t0, 1)}
        if args.mode == 'finetune' and val_ds is not None:
            p, y, s, _, _ = predict(model, val_ds, args, device)
            m = evaluate_predictions(p, y, s, val_ds.seasons)
            rec['val_pixel_rmse'] = m['overall']['pixel']['rmse']
            rec['val_pixel_r2'] = m['overall']['pixel']['r2']
            rec['val_field_rmse'] = m['overall']['field_level']['rmse']
            if best is None or rec['val_pixel_rmse'] < best['val_pixel_rmse']:
                best = dict(rec)
                torch.save({'model': model.state_dict(), 'descriptor': model.descriptor(), 'args': vars(args)},
                           out_dir / 'checkpoint_best.pth')
        history.append(rec)
        print(json.dumps(rec), flush=True)

    report = {
        'contract': CONTRACT_KEY, 'mode': args.mode, 'data_mode': 'point_timeseries',
        'args': vars(args), 'split': {'name': args.split, 'label': split['label'],
                                      'scheme': split['scheme'], 'hash': split['partition_hash']},
        'label_policy': ('retrospective ({})'.format(args.cutoff_mode) if train_ds.retrospective
                         else 'pre-harvest: cutoff {} {} d'.format(args.cutoff_mode, args.cutoff_days)),
        'streams': {n: l['out_channels'] for n, l in train_ds.layout.items()},
        'excluded_sources': ['coord_x/y/z (unverified, metadata only)', 'row/col/field/farm ids',
                             'harvest date as input', 'supplied stats-*', 'yield_ground_truth attrs']
                            + ([] if args.soil_uncertainty == 'ancillary' else ['soil uncertainty']),
        'objectives': {k: v['point_timeseries'] for k, v in OBJECTIVE_ROUTING.items()},
        'provenance': PROVENANCE,
        'train_seasons': len(parts['train']), 'val_seasons': len(parts['val']),
        'test_seasons': len(parts['test']),
        'history': history, 'best_val': best, 'transfer': transfer,
        'io': {'train_samples': samples_seen, 'data_wait_seconds': round(data_time, 1),
               'samples_per_second': round(samples_seen / max(1e-6, sum(h['seconds'] for h in history)), 1)},
    }
    if args.mode == 'pretrain':
        torch.save(model.sensor_state_dict(), out_dir / 'sensor_checkpoint.pth')
    else:
        torch.save(model.sensor_state_dict(), out_dir / 'sensor_checkpoint_last.pth')
        if best is not None:
            model.load_state_dict(torch.load(out_dir / 'checkpoint_best.pth', map_location=device, weights_only=True)['model'])
        test_ds = YieldSATPointDataset(args.source_root, args.artifact_root, parts['test'], normalizer,
                                       max_rows_per_field=args.test_rows_per_field, **common)
        te = time.time()
        p, y, s, r, c = predict(model, test_ds, args, device)
        report['test'] = evaluate_predictions(p, y, s, test_ds.seasons)
        report['test']['label'] = '{}; {}'.format(split['label'], report['label_policy'])
        report['test']['rows_evaluated'] = int(len(p))
        report['io']['test_rows_per_second'] = round(len(p) / max(1e-6, time.time() - te), 1)
        np.savez_compressed(out_dir / 'test_predictions.npz', pred=p, target=y, season=s, grid_row=r,
                            grid_col=c, season_names=np.array([x['field_shared_name'] for x in test_ds.seasons]))
        # spatial reconstruction of a few held-out fields on verified grids
        geo = {}
        for gp in (Path(args.artifact_root) / 'geometry').glob('fields_geometry_*.json'):
            geo.update({g['field_shared_name']: g for g in json.loads(gp.read_text())})
        maps = []
        for si in np.unique(s)[:args.save_maps]:
            sel = s == si
            rec_s = test_ds.seasons[si]
            g = geo.get(rec_s['field_shared_name'])
            grid, georef = reconstruct_field(rec_s, r[sel], c[sel], p[sel], g)
            path = out_dir / 'pred_{}.tif'.format(rec_s['field_shared_name'])
            if georef is not None:
                write_geotiff(path, grid, georef)
            maps.append({'field': rec_s['field_shared_name'], 'georeferenced': georef is not None,
                         'grid': list(grid.shape), 'cells': int(sel.sum())})
        report['maps'] = maps
    report['resources'] = {
        'wall_seconds': round(time.time() - t_start, 1),
        'peak_rss_gb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
        'peak_gpu_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == 'cuda' else None,
    }
    (out_dir / 'report.json').write_text(json.dumps(report, indent=1, default=str))
    if 'test' in report:
        t = report['test']['overall']
        print('TEST', report['test']['label'], json.dumps({
            'pixel_rmse': t['pixel']['rmse'], 'pixel_r2': t['pixel']['r2'],
            'field_rmse': t['field_level']['rmse'], 'field_r2': t['field_level']['r2']}))
    return report


if __name__ == '__main__':
    main(get_args_parser().parse_args())
