# --------------------------------------------------------
# YieldSAT image-mode training entry point (YI-04; spec/yieldsat-image-training.md).
# One 64x64 tile -> dense per-cell yield (t/ha), YieldSATImageModel on frozen,
# cached DINOv3 tokens plus the other modality encoders.
#
#   python main_yieldsat_image.py --artifact_root /data/YieldSAT/yieldsat_artifacts \
#       --image_root /data/YieldSAT/YieldSAT-Image --countries Germany --crops rapeseed \
#       --split paper_cv10_GER-R_season_paper_s0_fold00 --streams yieldsat_s2 ...
#
# It takes the same split manifests, stream flags, budget flags and W&B flags
# as main_yieldsat_finetune.py, so yieldsat_cluster.py drives both. A tile
# inherits its field season's partition. Outputs: report.json (point fields),
# test_predictions.npz in the point format (one row per valid test cell,
# matched to point runs by season + grid row/col), normalizer.json,
# checkpoint_best.pth. --donor trains on every tile of the countries/crops
# with a 10% season validation carve-out and no test (YI-06); --init_ckpt
# warm-starts from such a donor (crop embedding and shape-mismatched tensors
# are left at init).
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

from dataset.yieldsat_image_dataset import (
    DinoFeatureCache, ImageNormalizer, SeasonBalancedSampler, TileReader, YieldSATImageDataset, cast_batch,
    collate_tiles, load_tile_table, tiles_for_split,
)
from dataset.yieldsat_schema import COUNTRIES, CROPS
from main_yieldsat_finetune import WandbLogger
from models_yieldsat_image import (
    DINO_REVISION, YieldSATImageModel, preprocessing_hash, preprocessing_spec,
)
from util.yieldsat_eval import evaluate_predictions
from yieldsat_image_dino_cache import cache_tag

IMAGE_CONTRACT = 'yieldsat_image_v1'
S2 = ['yieldsat_s2']
S2_ADM = ['yieldsat_s2', 'yieldsat_weather', 'yieldsat_dem', 'yieldsat_terrain', 'yieldsat_soil']


def get_args_parser():
    p = argparse.ArgumentParser('YieldSAT image training', add_help=True)
    p.add_argument('--data_contract', default=IMAGE_CONTRACT,
                   choices=[IMAGE_CONTRACT, 'yieldsat_preprocessed_v1'],
                   help='the point contract name is accepted: splits come from it')
    p.add_argument('--source_root', default=os.environ.get('YIELDSAT_SOURCE_ROOT'),
                   help='unused (accepted for cluster compatibility)')
    p.add_argument('--artifact_root', default=os.environ.get('YIELDSAT_ARTIFACT_ROOT'))
    p.add_argument('--image_root', default=os.environ.get('YIELDSAT_IMAGE_ROOT'))
    p.add_argument('--dino_cache', default=None, help='default <image_root>/dino_cache/<rev>_<prep>')
    p.add_argument('--dino_revision', default=DINO_REVISION)
    p.add_argument('--no_dino', action='store_true', help='ablation: no DINO stream')
    p.add_argument('--countries', nargs='+', choices=COUNTRIES, required=True)
    p.add_argument('--crops', nargs='+', default=None, choices=list(CROPS))
    p.add_argument('--split', default=None, help='point fold manifest under artifact_root/splits')
    p.add_argument('--donor', action='store_true', help='train on all tiles (10%% val), no test')
    p.add_argument('--donor_val_frac', type=float, default=0.1)
    p.add_argument('--streams', nargs='+', default=S2_ADM, choices=S2_ADM,
                   help='yieldsat_s2 alone (S2) or all five (S2+ADM)')
    p.add_argument('--cutoff_mode', default='all_slots', choices=['all_slots', 'before_harvest', 'harvest'])
    p.add_argument('--cutoff_days', type=int, default=30)
    p.add_argument('--k_obs', type=int, default=4)
    p.add_argument('--no_crop_context', action='store_true')
    p.add_argument('--embed_dim', type=int, default=192)
    p.add_argument('--num_latents', type=int, default=32)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--cross_attn_layers', type=int, default=2)
    p.add_argument('--modality_dropout', type=float, default=0.1)
    p.add_argument('--init_ckpt', default='', help='donor checkpoint_best.pth (warm start)')
    # v2 (spec plan v2; defaults reproduce v1)
    p.add_argument('--series', action='store_true', help='YI-09: per-pixel full S2 time-series branch')
    p.add_argument('--level_head', action='store_true', help='YI-10: tile level + centered residual')
    p.add_argument('--level_weight', type=float, default=1.0, help='weight of the level loss')
    p.add_argument('--slot_coverage', default='tile', choices=['tile', 'present'],
                   help='optical slot coverage relative to the tile (v1) or its present cells (v2)')
    p.add_argument('--train_min_valid', type=int, default=0,
                   help='YI-11: training tiles need >= this many valid cells (val/test use all)')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--steps_per_epoch', type=int, default=0,
                   help='0 = one pass over the training tiles, clamped by min/max')
    p.add_argument('--min_steps_per_epoch', type=int, default=20)
    p.add_argument('--max_steps_per_epoch', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--eval_batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=float, default=2.0)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--norm_tiles', type=int, default=200, help='training tiles used for normalization')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output_dir', default='./output_dir/yieldsat_image')
    p.add_argument('--save_maps', type=int, default=0, help='accepted for cluster compatibility')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb_project', default=os.environ.get('WANDB_PROJECT', 'yieldsat-image'))
    p.add_argument('--wandb_entity', default=os.environ.get('WANDB_ENTITY'))
    p.add_argument('--wandb_group', default=None)
    p.add_argument('--wandb_name', default=None)
    p.add_argument('--wandb_tags', nargs='*', default=[])
    p.add_argument('--wandb_job_type', default=None)
    p.add_argument('--wandb_meta', default='{}')
    p.add_argument('--wandb_no_artifacts', action='store_true')
    return p


def _inputs(streams):
    if sorted(streams) == sorted(S2):
        return 's2'
    if sorted(streams) == sorted(S2_ADM):
        return 's2_adm'
    raise SystemExit('image runs take --streams yieldsat_s2 (S2) or all five streams (S2+ADM)')


def _season_table(artifact_root, countries):
    out = {}
    for c in countries:
        for f in json.loads((Path(artifact_root) / 'index' / c / 'fields.json').read_text()):
            out[f['season_id']] = f
    return out


def _donor_split(tiles, countries, crops, frac, seed):
    seasons = sorted({t['season_id'] for t in tiles if t['crop'] in (crops or CROPS)})
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(frac * len(seasons))))
    val = set(rng.choice(len(seasons), n_val, replace=False).tolist())
    return {'name': 'donor', 'label': 'donor: all tiles of {} {}, {:.0%} season val'.format(
                countries, crops or 'all crops', frac),
            'scheme': 'donor', 'partition_hash': None, 'countries': countries,
            'partitions': {'train': [s for i, s in enumerate(seasons) if i not in val],
                           'val': [s for i, s in enumerate(seasons) if i in val], 'test': []}}


def _to(batch, device):
    return cast_batch({k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                       for k, v in batch.items()})


def _loader(ds, args, sampler=None, batch_size=None):
    """Persistent workers: the sampler carries the epoch in each index."""
    return DataLoader(ds, batch_size=batch_size or args.batch_size, sampler=sampler,
                      num_workers=args.num_workers, collate_fn=collate_tiles,
                      persistent_workers=args.num_workers > 0,
                      pin_memory=args.device.startswith('cuda'), drop_last=False)


@torch.no_grad()
def predict(model, loader, args, device):
    """Valid cells of every tile -> point-format arrays (t/ha)."""
    model.eval()
    amp = device.type == 'cuda' and not args.no_amp
    pred, target, season, row, col = [], [], [], [], []
    for batch in loader:
        b = _to(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            p = model(b, apply_dropout=False).float()
        p = (p * b['target_std'].view(-1, 1, 1) + b['target_mean'].view(-1, 1, 1)).cpu().numpy()
        v = batch['target_valid'].numpy()
        s = np.broadcast_to(batch['season'].numpy()[:, None, None], v.shape)
        pred.append(p[v])
        target.append(batch['target_raw'].numpy()[v])
        season.append(s[v])
        row.append(batch['grid_row'].numpy()[v])
        col.append(batch['grid_col'].numpy()[v])
    cat = np.concatenate
    return cat(pred), cat(target), cat(season), cat(row), cat(col)


def load_donor(model, path):
    """Warm start: every tensor except the crop embedding and shape mismatches."""
    state = torch.load(path, map_location='cpu', weights_only=False)['model']
    own = model.state_dict()
    keep = {k: v for k, v in state.items()
            if k in own and own[k].shape == v.shape and not k.startswith('crop_embed')}
    model.load_state_dict(keep, strict=False)
    return {'checkpoint': str(path), 'loaded': len(keep),
            'skipped': sorted(set(own) - set(keep))}


def main(args):
    if not args.artifact_root or not args.image_root:
        raise SystemExit('--artifact_root and --image_root are required')
    if not args.donor and not args.split:
        raise SystemExit('--split is required (or --donor)')
    args.mode = 'finetune'                   # WandbLogger job type default
    inputs = _inputs(args.streams)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    seasons = _season_table(args.artifact_root, args.countries)
    tiles = load_tile_table(args.image_root, args.countries)
    if args.donor:
        split = _donor_split(tiles, args.countries, args.crops, args.donor_val_frac, args.seed)
    else:
        split = json.loads((Path(args.artifact_root) / 'splits' / '{}.json'.format(args.split)).read_text())
        if not set(args.countries) <= set(split['countries']):
            raise SystemExit('split {} covers {}, not {}'.format(args.split, split['countries'], args.countries))
    parts = tiles_for_split(tiles, split, crops=args.crops)
    train_filter = None
    if args.train_min_valid > 0:
        full = [t for t in parts['train'] if t['valid_pixels'] >= args.train_min_valid]
        # a fold whose training seasons have no full tile keeps all its tiles
        train_filter = 'min_valid {}'.format(args.train_min_valid) if full else 'fallback: all tiles'
        parts['train'] = full or parts['train']
    if not parts['train']:
        raise SystemExit('no training tiles for this split')
    single_crop = args.crops is not None and len(args.crops) == 1
    use_crop = not args.no_crop_context and not single_crop
    season_days = {s: (f['seeding_day'], f['harvest_day']) for s, f in seasons.items()}

    reader = TileReader(args.image_root)
    normalizer = ImageNormalizer.fit(reader, parts['train'], max_tiles=args.norm_tiles, seed=args.seed)
    reader.close()                           # no HDF5 handles inherited by loader workers
    (out_dir / 'normalizer.json').write_text(json.dumps(normalizer.to_json()))
    dino = None
    if not args.no_dino:
        prep_hash = preprocessing_hash(preprocessing_spec())
        cache_dir = args.dino_cache or str(Path(args.image_root) / 'dino_cache' /
                                           cache_tag(args.dino_revision, prep_hash))
        dino = DinoFeatureCache(cache_dir, expected_revision=args.dino_revision,
                                expected_prep_hash=prep_hash)
    common = dict(k_obs=args.k_obs, cutoff_mode=args.cutoff_mode, cutoff_days=args.cutoff_days,
                  seed=args.seed, dino_cache=dino, with_series=args.series,
                  slot_coverage=args.slot_coverage)
    train_ds = YieldSATImageDataset(args.image_root, parts['train'], normalizer, season_days,
                                    train=True, **common)
    val_ds = (YieldSATImageDataset(args.image_root, parts['val'], normalizer, season_days, **common)
              if parts['val'] else None)
    val_loader = _loader(val_ds, args, batch_size=args.eval_batch_size) if val_ds else None

    model = YieldSATImageModel(inputs=inputs, embed_dim=args.embed_dim, k_obs=args.k_obs,
                               dino_dim=dino.hidden_size if dino else 1024,
                               num_latents=args.num_latents, depth=args.depth, num_heads=args.num_heads,
                               cross_attn_layers=args.cross_attn_layers,
                               modality_dropout=args.modality_dropout, use_dino=dino is not None,
                               use_crop=use_crop, use_series=args.series, level_head=args.level_head,
                               level_weight=args.level_weight).to(device)
    transfer = load_donor(model, args.init_ckpt) if args.init_ckpt else None
    if transfer:
        print('warm start from {}: {} tensors'.format(args.init_ckpt, transfer['loaded']), flush=True)
    if args.steps_per_epoch <= 0:
        steps = max(args.min_steps_per_epoch, math.ceil(len(train_ds) / args.batch_size))
        if args.max_steps_per_epoch > 0:
            steps = min(steps, args.max_steps_per_epoch)
        args.steps_per_epoch = steps
    wb = WandbLogger(args, out_dir)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * args.steps_per_epoch
    warm = int(args.warmup_epochs * args.steps_per_epoch)

    def lr_at(step):
        if step < warm:
            return args.lr * (step + 1) / warm
        return args.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    amp = device.type == 'cuda' and not args.no_amp
    sampler = SeasonBalancedSampler(train_ds.tiles, args.steps_per_epoch * args.batch_size, seed=args.seed)
    train_loader = _loader(train_ds, args, sampler=sampler)
    history, best, step, seen, data_time = [], None, 0, 0, 0.0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        t0 = tl = time.time()
        losses = []
        for batch in train_loader:
            data_time += time.time() - tl
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            b = _to(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                _, loss = model.loss(b)
            opt.zero_grad(set_to_none=True)
            loss.float().backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(loss.item())
            seen += batch['target'].shape[0]
            step += 1
            tl = time.time()
        rec = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'lr': lr_at(step - 1),
               'seconds': round(time.time() - t0, 1)}
        if val_ds is not None:
            p, y, s, _, _ = predict(model, val_loader, args, device)
            m = evaluate_predictions(p, y, s, [seasons[x] for x in val_ds.seasons])
            rec['val_pixel_rmse'] = m['overall']['pixel']['rmse']
            rec['val_pixel_r2'] = m['overall']['pixel']['r2']
            rec['val_field_rmse'] = m['overall']['field_level']['rmse']
            if best is None or rec['val_pixel_rmse'] < best['val_pixel_rmse']:
                best = dict(rec)
                torch.save({'model': model.state_dict(), 'config': model.config, 'args': vars(args)},
                           out_dir / 'checkpoint_best.pth')
        history.append(rec)
        print(json.dumps(rec), flush=True)
        wb.log_epoch(rec)
    if best is None:
        torch.save({'model': model.state_dict(), 'config': model.config, 'args': vars(args)},
                   out_dir / 'checkpoint_best.pth')

    report = {
        'contract': IMAGE_CONTRACT, 'mode': 'donor' if args.donor else 'finetune',
        'data_mode': 'image_64', 'args': vars(args),
        'split': {'name': split.get('name', args.split), 'label': split['label'],
                  'scheme': split.get('scheme'), 'hash': split.get('partition_hash')},
        'label_policy': 'retrospective ({})'.format(args.cutoff_mode) if args.cutoff_mode == 'all_slots'
                        else 'pre-harvest: cutoff {} {} d'.format(args.cutoff_mode, args.cutoff_days),
        'inputs': inputs, 'streams': list(model.streams),
        'model': {'descriptor': model.config,
                  'parameters': int(sum(p.numel() for p in model.parameters()))},
        'dino': ({'cache': str(dino.dir), 'revision': dino.manifest['revision'],
                  'prep_hash': dino.manifest['prep_hash'], 'frozen': True} if dino else None),
        'excluded_sources': ['coord_x/y/z', 'soil uncertainty', 'source_row', 'valid_pixel as input',
                             'target-side metadata', 'supplied stats-*'],
        'pair_filter': {'countries': args.countries, 'crops': args.crops, 'crop_context': use_crop},
        'normalization': {'features': 'train tiles', 'target': 'train tiles'},
        'train_seasons': len({t['season_id'] for t in parts['train']}),
        'val_seasons': len({t['season_id'] for t in parts['val']}),
        'test_seasons': len({t['season_id'] for t in parts['test']}),
        'tiles': {k: len(v) for k, v in parts.items()},
        'train_tile_filter': train_filter,
        'history': history, 'best_val': best, 'transfer': transfer,
        'io': {'train_samples': seen, 'data_wait_seconds': round(data_time, 1),
               'samples_per_second': round(seen / max(1e-6, sum(h['seconds'] for h in history)), 1)},
    }
    if parts['test']:
        model.load_state_dict(torch.load(out_dir / 'checkpoint_best.pth', map_location=device,
                                         weights_only=False)['model'])
        test_ds = YieldSATImageDataset(args.image_root, parts['test'], normalizer, season_days, **common)
        te = time.time()
        p, y, s, r, c = predict(model, _loader(test_ds, args, batch_size=args.eval_batch_size), args, device)
        meta = [seasons[x] for x in test_ds.seasons]
        report['test'] = evaluate_predictions(p, y, s, meta)
        report['test']['label'] = '{}; {}; valid cells of test tiles'.format(split['label'],
                                                                            report['label_policy'])
        report['test']['rows_evaluated'] = int(len(p))
        report['io']['test_rows_per_second'] = round(len(p) / max(1e-6, time.time() - te), 1)
        np.savez_compressed(out_dir / 'test_predictions.npz', pred=p, target=y, season=s, grid_row=r,
                            grid_col=c, season_names=np.array([m['field_shared_name'] for m in meta]))
    report['resources'] = {
        'wall_seconds': round(time.time() - t_start, 1),
        'peak_rss_gb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
        'peak_gpu_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == 'cuda' else None,
    }
    (out_dir / 'report.json').write_text(json.dumps(report, indent=1, default=str))
    wb.finish(report)
    if 'test' in report:
        t = report['test']['overall']
        print('TEST', report['test']['label'], json.dumps({
            'pixel_rmse': t['pixel']['rmse'], 'pixel_r2': t['pixel']['r2'],
            'field_rmse': t['field_level']['rmse'], 'field_r2': t['field_level']['r2']}))
    return report


if __name__ == '__main__':
    main(get_args_parser().parse_args())
