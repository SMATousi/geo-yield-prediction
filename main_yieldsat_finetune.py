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
# fine-tuning. --knowledge adds concept grounding + relational distillation
# (spec/yieldsat-point-knowledge-pretraining.md). Prerequisites (index, cache, geometry, split) come from
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
    CUTOFF_MODES, FieldBalancedBatchSampler, NormalizerView, SequentialBatchSampler,
    YieldSATNormalizer, YieldSATPointDataset, collate_point_batch, load_supplied_stats,
)
from dataset.yieldsat_schema import CONTRACT_KEY, COUNTRIES, CROPS, DEFAULT_STREAMS, OPTIONAL_STREAMS, PROVENANCE, STREAMS
from dataset.yieldsat_splits import load_field_table, load_split
from dataset.yieldsat_source import load_country_index
from models_yieldsat import FUSIONS, PaperLSTMBaseline, YieldSATPointModel
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
    p.add_argument('--skip_source_check', action='store_true',
                   help='cache backend only: do not fingerprint the source NetCDF (local smoke without it)')
    p.add_argument('--streams', nargs='+', default=list(DEFAULT_STREAMS), choices=list(STREAMS) + list(OPTIONAL_STREAMS))
    p.add_argument('--weather_first_slot', default='mask', choices=['mask', 'keep'],
                   help="'keep' uses the first dated slot's weather sum, as the paper (repro D1)")
    p.add_argument('--lstm_hidden', type=int, default=64)
    p.add_argument('--lstm_layers', type=int, default=1)
    p.add_argument('--lstm_head', default='fc', choices=['fc', 'mlp'])
    p.add_argument('--diag_test_each_epoch', action='store_true',
                   help='DIAGNOSTIC ONLY: also score the test fold after every epoch (report.history), to '
                        'measure the effect of selecting the epoch on the test fold (repro hypothesis H1); '
                        'never used for model selection')
    p.add_argument('--early_stop_patience', type=int, default=0,
                   help='finetune: stop after this many epochs without validation improvement (0 = off)')
    p.add_argument('--cutoff_mode', default='before_harvest', choices=CUTOFF_MODES)
    p.add_argument('--cutoff_days', type=int, default=30)
    p.add_argument('--soil_uncertainty', default='none', choices=['none', 'ancillary'])
    p.add_argument('--aspect_encoding', default='raw', choices=['raw', 'cyclic'])
    p.add_argument('--norm_pooling', default='pooled', choices=['pooled', 'per_country'])
    p.add_argument('--no_crop_context', action='store_true')
    p.add_argument('--crops', nargs='+', default=None, choices=list(CROPS),
                   help='restrict every partition to these crops (paper: one country-crop pair)')
    p.add_argument('--model', default='yieldsat_point', choices=['yieldsat_point', 'paper_lstm'],
                   help='paper_lstm = the paper/tutorial pixel LSTM baseline (PC-05)')
    p.add_argument('--normalization', default='train', choices=['train', 'supplied', 'none'],
                   help='feature normalization: train-fold stats, the file stats-*, or raw')
    p.add_argument('--target_normalization', default='train', choices=['train', 'none'])
    p.add_argument('--fill_value', type=float, default=0.0,
                   help='value for invalid inputs after normalization (tutorial: -1 on raw values)')
    p.add_argument('--train_fraction', type=float, default=1.0,
                   help='keep this fraction of training field seasons (label budget)')
    # model
    p.add_argument('--embed_dim', type=int, default=128)
    p.add_argument('--num_latents', type=int, default=8)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--modality_embed', type=int, default=32)
    p.add_argument('--modality_dropout', type=float, default=0.1)
    p.add_argument('--fusion', default='perceiver_summary', choices=FUSIONS,
                   help='how per-stream encodings are fused (YS-11)')
    p.add_argument('--cross_attn_layers', type=int, default=None,
                   help='Perceiver reads (perceiver_tokens only; default 2)')
    # improvement plan round 2 (spec/yieldsat-improvement.md)
    p.add_argument('--level_head', action='store_true', help='S5: season-level term + field-mean loss')
    p.add_argument('--level_weight', type=float, default=1.0)
    p.add_argument('--early_hidden', type=int, default=192, help='S4 early-fusion encoder width')
    p.add_argument('--early_depth', type=int, default=3)
    p.add_argument('--neighbourhood', action='store_true',
                   help='S6: add the 5x5 neighbourhood S2 stream (<artifact_root>/neighbourhood)')
    # knowledge pretraining (spec/yieldsat-point-knowledge-pretraining.md)
    p.add_argument('--knowledge', action='store_true',
                   help='pretrain mode: add concept grounding + relational distillation')
    p.add_argument('--knowledge_text_dir', default='',
                   help='frozen CLIP text vectors (default <artifact_root>/knowledge/text_clip_b32)')
    p.add_argument('--knowledge_control', default='none', choices=['none', 'shuffled', 'notext', 'ssl_long', 'random_targets'])
    p.add_argument('--ground_weight', type=float, default=1.0)
    p.add_argument('--relation_weight', type=float, default=1.0)
    p.add_argument('--pretrain_diagnostics', action='store_true',
                   help='pretrain mode: compute the success-criteria diagnostics on the unit\'s held-out '
                        'seasons at the end (yieldsat_pretrain_diagnostics.py -> diagnostics.json)')
    p.add_argument('--pretrain_val_batches', type=int, default=20,
                   help='pretrain mode: validation batches per epoch (success criterion P1)')
    p.add_argument('--resume_dir', default='',
                   help='pretrain mode: save resumable state here every epoch (e.g. on the shared volume) '
                        'and continue from it if present')
    p.add_argument('--init_sensor_ckpt', default='')
    p.add_argument('--encoders_only_transfer', action='store_true',
                   help='load only encoder weights from --init_sensor_ckpt (fusion may differ)')
    # optimisation
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--steps_per_epoch', type=int, default=500,
                   help='0 = one full pass over the training cells per epoch')
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--optimizer', default='adamw', choices=['adamw', 'adam'])
    p.add_argument('--lr_schedule', default='cosine', choices=['cosine', 'constant'])
    p.add_argument('--grad_clip', type=float, default=1.0, help='0 disables clipping')
    p.add_argument('--no_amp', action='store_true', help='disable bf16 autocast')
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
    p.add_argument('--eval_drop_stream', default=None, choices=list(STREAMS),
                   help='stress test: remove this stream for --eval_drop_frac of test cells')
    p.add_argument('--eval_drop_frac', type=float, default=0.5)
    p.add_argument('--min_steps_per_epoch', type=int, default=1,
                   help='lower clamp when --steps_per_epoch 0 (full pass)')
    p.add_argument('--max_steps_per_epoch', type=int, default=0,
                   help='upper clamp when --steps_per_epoch 0 (0 = no clamp)')
    # Weights & Biases (optional; the API key comes from WANDB_API_KEY)
    p.add_argument('--wandb', action='store_true', help='log metrics and artifacts to W&B')
    p.add_argument('--wandb_project', default=os.environ.get('WANDB_PROJECT', 'yieldsat'))
    p.add_argument('--wandb_entity', default=os.environ.get('WANDB_ENTITY'))
    p.add_argument('--wandb_group', default=None)
    p.add_argument('--wandb_name', default=None)
    p.add_argument('--wandb_tags', nargs='*', default=[])
    p.add_argument('--wandb_job_type', default=None)
    p.add_argument('--wandb_meta', default='{}',
                   help='JSON dict of job metadata added to the W&B config')
    p.add_argument('--wandb_no_model_artifacts', action='store_true',
                   help='upload only the results artifact (checkpoints stay local / on the PVC)')
    p.add_argument('--wandb_no_artifacts', action='store_true',
                   help='log metrics only, do not upload models/results')
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


def build_knowledge_pretrainer(args, model, parts, train_ds, val_ds, out_dir):
    """Fit the fold-train concept reference, attach targets to the datasets
    and wrap the model in a KnowledgePointPretrainer."""
    from yieldsat_build_knowledge import estimator_cutoff, load_raw
    from yieldsat_knowledge_point import (KnowledgePointPretrainer, KnowledgeReference, attach_knowledge,
                                          load_library, load_text_vectors)
    lib = load_library()
    text_dir = args.knowledge_text_dir or str(Path(args.artifact_root) / 'knowledge' / 'text_clip_b32')
    text, text_manifest = load_text_vectors(text_dir, lib)
    cutoff = estimator_cutoff(args.cutoff_mode, args.cutoff_days)       # estimators see what the model sees
    raw = {c: load_raw(args.artifact_root, c, cutoff) for c in train_ds.countries}
    train_rows = {c: train_ds.row[train_ds.country_of == i] for i, c in enumerate(train_ds.countries)}
    ref = KnowledgeReference.fit({c: (raw[c], r) for c, r in train_rows.items() if len(r)}, lib)
    (out_dir / 'knowledge_reference.json').write_text(json.dumps(ref.to_json()))
    attach_knowledge(train_ds, ref, args.artifact_root, raw)
    if val_ds is not None:
        attach_knowledge(val_ds, ref, args.artifact_root, raw)
    control = None if args.knowledge_control == 'none' else args.knowledge_control
    if control == 'random_targets':
        from yieldsat_knowledge_point import randomize_targets
        randomize_targets(train_ds, args.seed)
        if val_ds is not None:
            randomize_targets(val_ds, args.seed)
    pt = KnowledgePointPretrainer(model, text, lib, control=control, ground_weight=args.ground_weight,
                                  relation_weight=args.relation_weight, shuffle_seed=args.seed)
    kt, kg = train_ds.knowledge['concept_target'], train_ds.knowledge['rule_gate']
    seasons = train_ds.season_of
    coverage = {}
    for i, cid in enumerate(ref.concepts):
        known = np.isfinite(kt[:, i])
        coverage[cid] = float(len(np.unique(seasons[known])) / max(1, len(train_ds.seasons)))
    rules = {}
    for j, r in enumerate(ref.rules):
        ia, ib = ref.concepts.index(r['concept_a']), ref.concepts.index(r['concept_b'])
        ok = (kg[:, j] > 0) & (kt[:, ia] >= r['minimum_presence']) & (kt[:, ib] >= r['minimum_presence'])
        rules[r['id']] = {'gate_open_seasons': int(len(np.unique(seasons[kg[:, j] > 0]))),
                          'applicable_seasons': int(len(np.unique(seasons[ok])))}
    info = {'control': args.knowledge_control, 'estimator_cutoff_days': cutoff, 'text': {k: text_manifest[k] for k in
                                                         ('model', 'revision', 'vectors_hash', 'library_hash')},
            'concept_season_coverage': coverage, 'rules': rules,
            'reference_seasons': ref.meta['reference_seasons']}
    print('knowledge:', json.dumps({'coverage': coverage, 'rules': rules}), flush=True)
    return pt, info


@torch.no_grad()
def pretrain_validation(trainable, ds, args, device, amp):
    """Mean pretraining losses over fixed validation batches (masking is random
    but seeded, so epochs are comparable)."""
    trainable.eval()
    sampler = FieldBalancedBatchSampler(ds.season_ranges, args.batch_size, args.pretrain_val_batches,
                                        alpha=args.field_alpha, block_size=1, seed=args.seed + 7919)
    state = torch.random.get_rng_state()
    torch.manual_seed(args.seed + 7919)
    tot, comps = [], {}
    for batch in _loader(ds, sampler, args.num_workers):
        b = _to(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            loss, parts = trainable(b)
        tot.append(float(loss))
        for k, v in parts.items():
            comps.setdefault(k, []).append(float(v))
    torch.random.set_rng_state(state)
    trainable.train()
    out = {'val_loss': float(np.mean(tot))}
    out.update({'val_' + k: float(np.mean(v)) for k, v in comps.items()})
    return out


def drop_stream(batch, stream, frac):
    """Remove ``stream`` for a deterministic ``frac`` of cells (by item id)."""
    item = batch['item']
    drop = ((item * 2654435761) % 1000) < int(round(frac * 1000))
    batch['available'][stream] = batch['available'][stream] * (~drop).float()
    keep = (~drop).view(-1, *([1] * (batch['masks'][stream].ndim - 1)))
    batch['masks'][stream] = batch['masks'][stream] & keep
    batch['inputs'][stream] = batch['inputs'][stream] * keep
    return drop


@torch.no_grad()
def predict(model, ds, args, device, drop=None):
    model.eval()
    loader = _loader(ds, SequentialBatchSampler(len(ds), args.eval_batch_size), args.num_workers)
    preds, targets, seasons, rows, cols = [], [], [], [], []
    for batch in loader:
        if drop is not None:
            drop_stream(batch, drop[0], drop[1])
        b = _to(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == 'cuda' and not args.no_amp
                            and args.model != 'paper_lstm'):
            p = model(b, apply_dropout=False).float()
        preds.append((p * b['target_std'] + b['target_mean']).cpu().numpy())
        targets.append(batch['target_raw'].numpy())
        seasons.append(batch['season'].numpy())
        rows.append(batch['grid_row'].numpy())
        cols.append(batch['grid_col'].numpy())
    cat = np.concatenate
    return cat(preds), cat(targets), cat(seasons), cat(rows), cat(cols)


class WandbLogger:
    """Optional W&B logging: config, per-epoch history, flattened test
    metrics as summary, and the run's models/results as artifacts. A no-op
    unless ``--wandb``; ``WANDB_MODE=offline`` works without network."""

    MODEL_FILES = ('checkpoint_best.pth', 'sensor_checkpoint.pth', 'sensor_checkpoint_last.pth',
                   'normalizer.json')
    RESULT_FILES = ('report.json', 'test_predictions.npz')

    def __init__(self, args, out_dir):
        self.run = None
        self.args = args
        self.out_dir = Path(out_dir)
        if not args.wandb:
            return
        import wandb
        config = {k: v for k, v in vars(args).items() if not k.startswith('wandb')}
        config['job'] = json.loads(args.wandb_meta)
        self.wandb = wandb
        kwargs = dict(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
                      name=args.wandb_name, tags=list(args.wandb_tags),
                      job_type=args.wandb_job_type or args.mode, config=config,
                      dir=str(self.out_dir))
        try:
            self.run = wandb.init(**kwargs)
        except Exception as exc:              # no network / bad key: keep training, sync later
            print('W&B init failed ({}); logging offline'.format(exc), flush=True)
            self.run = wandb.init(mode='offline', **kwargs)

    def log_epoch(self, rec):
        if self.run is not None:
            self.run.log({k: v for k, v in rec.items() if k != 'epoch'}, step=rec['epoch'])

    @staticmethod
    def _flatten(prefix, block, out):
        for level in ('pixel', 'field_level', 'field_balanced'):
            for k, v in block.get(level, {}).items():
                if isinstance(v, (int, float)):
                    out['{}/{}_{}'.format(prefix, level, k)] = v

    def finish(self, report):
        if self.run is None:
            return
        summary = {'train_seasons': report['train_seasons'], 'val_seasons': report['val_seasons'],
                   'test_seasons': report['test_seasons'],
                   'parameters': report['model']['parameters'],
                   'best_epoch': (report.get('best_val') or {}).get('epoch'),
                   'split_label': report['split']['label'], 'label_policy': report['label_policy']}
        for key in ('test', 'test_stress'):
            if key in report:
                self._flatten(key, report[key]['overall'], summary)
                for c, block in report[key].get('per_country_crop', {}).items():
                    self._flatten('{}/{}'.format(key, c), block, summary)
        summary.update({'io/' + k: v for k, v in report['io'].items()})
        summary.update({'resources/' + k: v for k, v in report['resources'].items()})
        self.run.summary.update(summary)
        if not self.args.wandb_no_artifacts:
            safe = ''.join(ch if ch.isalnum() or ch in '-_.' else '-' for ch in (self.run.name or self.run.id))
            kinds = (() if getattr(self.args, 'wandb_no_model_artifacts', False)
                     else (('model', self.MODEL_FILES, None),)) + (('results', self.RESULT_FILES, 'pred_*.tif'),)
            for kind, files, pattern in kinds:
                paths = [self.out_dir / f for f in files if (self.out_dir / f).exists()]
                if pattern:
                    paths += sorted(self.out_dir.glob(pattern))
                if not paths:
                    continue
                art = self.wandb.Artifact('{}-{}'.format(kind, safe)[:120], type=kind,
                                          metadata={'split': report['split'], 'job': self.run.config.get('job')})
                for path in paths:
                    art.add_file(str(path))
                self.run.log_artifact(art)
        self.run.finish()


def main(args):
    if args.data_contract != CONTRACT_KEY:
        raise SystemExit('{} ingestion is not implemented; this entry point serves {}'.format(
            args.data_contract, CONTRACT_KEY))
    if (not args.source_root and not args.skip_source_check) or not args.artifact_root:
        raise SystemExit('--source_root and --artifact_root are required')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    check = not args.skip_source_check
    if not check and args.backend != 'cache':
        raise SystemExit('--skip_source_check needs the cache backend')
    table = load_field_table(args.artifact_root, args.source_root, args.countries, check_source=check)
    split = load_split(args.artifact_root, args.split, fingerprints=table['fingerprints'])
    if not set(args.countries) <= set(split['countries']):
        raise SystemExit('split {} covers {}, not the requested {}'.format(
            args.split, split['countries'], args.countries))
    # a country subset of a pooled split keeps that split's partitions, so
    # single-country runs are comparable with pooled/pretrained ones
    by_season = {f['season_id']: f for f in table['fields']}
    parts = {p: [by_season[s] for s in split['partitions'][p] if s in by_season]
             for p in ('train', 'val', 'test')}
    if args.crops:
        parts = {p: [f for f in v if f['crop'] in args.crops] for p, v in parts.items()}
    # a single crop makes the crop token constant: switch it off (PC-02)
    single_crop = args.crops is not None and len(args.crops) == 1
    use_crop = not args.no_crop_context and not single_crop
    rng = np.random.default_rng(args.seed)
    if args.train_fraction < 1.0:
        keep = max(1, int(round(args.train_fraction * len(parts['train']))))
        sel = rng.choice(len(parts['train']), keep, replace=False)
        parts['train'] = [parts['train'][i] for i in sorted(sel)]

    # train-only normalization
    targets = {c: load_country_index(args.artifact_root, args.source_root, c, check)['rows']['target']
               for c in args.countries}
    train_fields = [(f['country'], f['field_code'], targets[f['country']][f['row_start']:f['row_end']])
                    for f in parts['train']]
    normalizer = YieldSATNormalizer.fit(args.artifact_root, train_fields, pooling=args.norm_pooling)
    def _supplied(c):
        # exported copy (<artifact_root>/supplied_stats/<c>.json) when the source NetCDF is absent
        cached = Path(args.artifact_root) / 'supplied_stats' / '{}.json'.format(c)
        if args.skip_source_check and cached.exists():
            return {k: np.asarray(v) for k, v in json.loads(cached.read_text()).items()}
        return load_supplied_stats(args.source_root, c)
    supplied = ({c: _supplied(c) for c in args.countries} if args.normalization == 'supplied' else None)
    normalizer = NormalizerView(normalizer, features=args.normalization,
                                target=args.target_normalization, supplied=supplied)
    (out_dir / 'normalizer.json').write_text(json.dumps(normalizer.to_json()))

    common = dict(streams=args.streams, backend=args.backend, cutoff_mode=args.cutoff_mode,
                  cutoff_days=args.cutoff_days, soil_uncertainty=args.soil_uncertainty,
                  aspect_encoding=args.aspect_encoding, seed=args.seed, fill_value=args.fill_value,
                  check_source=check, weather_first_slot=args.weather_first_slot,
                  neighbourhood_root=str(Path(args.artifact_root) / 'neighbourhood') if args.neighbourhood else None)
    train_ds = YieldSATPointDataset(args.source_root, args.artifact_root, parts['train'], normalizer, **common)
    val_ds = YieldSATPointDataset(args.source_root, args.artifact_root, parts['val'], normalizer,
                                  max_rows_per_field=args.val_rows_per_field, **common) if parts['val'] else None
    block = args.block_size or (1 if args.backend == 'cache' else 64)

    if args.model == 'paper_lstm':
        if args.mode == 'pretrain':
            raise SystemExit('the paper LSTM baseline has no pretraining mode')
        model = PaperLSTMBaseline(train_ds.layout, hidden_dim=args.lstm_hidden, num_layers=args.lstm_layers,
                                  head=args.lstm_head).to(device)
    else:
        model = YieldSATPointModel(train_ds.layout, embed_dim=args.embed_dim,
                                   num_latents=args.num_latents, depth=args.depth,
                                   num_heads=args.num_heads, modality_embed=args.modality_embed,
                                   modality_dropout=args.modality_dropout,
                                   use_crop_context=use_crop, fusion=args.fusion,
                                   cross_attn_layers=args.cross_attn_layers, level_head=args.level_head,
                                   level_weight=args.level_weight, early_hidden=args.early_hidden,
                                   early_depth=args.early_depth).to(device)
    # cuDNN LSTMs do not run under bf16 autocast
    amp = device.type == 'cuda' and not args.no_amp and args.model != 'paper_lstm'
    if args.steps_per_epoch <= 0:
        steps = max(args.min_steps_per_epoch, math.ceil(len(train_ds) / args.batch_size))
        if args.max_steps_per_epoch > 0:
            steps = min(steps, args.max_steps_per_epoch)
        args.steps_per_epoch = max(1, steps)
    wb = WandbLogger(args, out_dir)
    transfer = None
    if args.init_sensor_ckpt:
        transfer = model.load_sensor_state_dict(
            torch.load(args.init_sensor_ckpt, map_location='cpu', weights_only=True),
            encoders_only=args.encoders_only_transfer)
        print('loaded sensor checkpoint:', transfer['loaded'], 'tensors')
    knowledge_ref = None
    if args.mode == 'pretrain' and args.knowledge:
        trainable, knowledge_ref = build_knowledge_pretrainer(args, model, parts, train_ds, val_ds, out_dir)
        trainable = trainable.to(device)
    elif args.mode == 'pretrain':
        trainable = YieldSATPointPretrainer(model).to(device)
    else:
        trainable = model
    if args.optimizer == 'adam':
        opt = torch.optim.Adam(trainable.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(trainable.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * args.steps_per_epoch
    warm = int(args.warmup_epochs * args.steps_per_epoch)

    def lr_at(step):
        if args.lr_schedule == 'constant':
            return args.lr
        if step < warm:
            return args.lr * (step + 1) / warm
        return args.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total_steps - warm)))

    sampler = FieldBalancedBatchSampler(train_ds.season_ranges, args.batch_size, args.steps_per_epoch,
                                        alpha=args.field_alpha, block_size=block, seed=args.seed)
    history, best, step = [], None, 0
    bad_epochs = 0
    diag_test_ds = None
    samples_seen, data_time = 0, 0.0
    start_epoch = 0
    resume_path = Path(args.resume_dir) / 'resume.pth' if args.resume_dir and args.mode == 'pretrain' else None
    if resume_path is not None and resume_path.exists():
        st = torch.load(resume_path, map_location=device, weights_only=False)
        trainable.load_state_dict(st['trainable'])
        opt.load_state_dict(st['optimizer'])
        start_epoch, step, history = st['epoch'] + 1, st['step'], st['history']
        samples_seen = st.get('samples_seen', 0)
        print('resumed from {} at epoch {}'.format(resume_path, start_epoch), flush=True)
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        trainable.train()
        t0 = time.time()
        tl = time.time()
        losses, comps = [], {}
        for batch in _loader(train_ds, sampler, args.num_workers):
            data_time += time.time() - tl
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            b = _to(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                if args.mode == 'pretrain':
                    loss, parts_loss = trainable(b)
                else:
                    _, loss = model.loss(b)
            opt.zero_grad(set_to_none=True)
            loss.float().backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable.parameters(), args.grad_clip)
            opt.step()
            losses.append(loss.item())
            if args.mode == 'pretrain':
                for k, v in parts_loss.items():
                    comps.setdefault(k, []).append(float(v))
            samples_seen += batch['target'].shape[0]
            step += 1
            tl = time.time()
        rec = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'lr': lr_at(step - 1),
               'seconds': round(time.time() - t0, 1)}
        rec.update({'train_' + k: float(np.mean(v)) for k, v in comps.items()})
        if args.mode == 'pretrain' and val_ds is not None and args.pretrain_val_batches > 0:
            rec.update(pretrain_validation(trainable, val_ds, args, device, amp))
        if args.mode == 'finetune' and args.diag_test_each_epoch:
            if diag_test_ds is None:
                diag_test_ds = YieldSATPointDataset(args.source_root, args.artifact_root, parts['test'], normalizer,
                                                    max_rows_per_field=args.test_rows_per_field, **common)
            pt_, yt_, st_, _, _ = predict(model, diag_test_ds, args, device)
            mt_ = evaluate_predictions(pt_, yt_, st_, diag_test_ds.seasons)
            rec['diag_test_pixel_r2'] = mt_['overall']['pixel']['r2']
            rec['diag_test_field_r2'] = mt_['overall']['field_level']['r2']
        if args.mode == 'finetune' and val_ds is not None:
            p, y, s, _, _ = predict(model, val_ds, args, device)
            m = evaluate_predictions(p, y, s, val_ds.seasons)
            rec['val_pixel_rmse'] = m['overall']['pixel']['rmse']
            rec['val_pixel_r2'] = m['overall']['pixel']['r2']
            rec['val_field_rmse'] = m['overall']['field_level']['rmse']
            if best is None or rec['val_pixel_rmse'] < best['val_pixel_rmse']:
                best = dict(rec)
                bad_epochs = 0
            else:
                bad_epochs += 1
                torch.save({'model': model.state_dict(), 'descriptor': model.descriptor(), 'args': vars(args)},
                           out_dir / 'checkpoint_best.pth')
        history.append(rec)
        print(json.dumps(rec), flush=True)
        wb.log_epoch(rec)
        if args.early_stop_patience and args.mode == 'finetune' and bad_epochs >= args.early_stop_patience:
            print('early stopping at epoch', epoch, flush=True)
            break
        if resume_path is not None:
            resume_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = resume_path.with_suffix('.tmp')
            torch.save({'trainable': trainable.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch,
                        'step': step, 'history': history, 'samples_seen': samples_seen}, tmp)
            os.replace(tmp, resume_path)

    report = {
        'contract': CONTRACT_KEY, 'mode': args.mode, 'data_mode': 'point_timeseries',
        'args': vars(args), 'split': {'name': args.split, 'label': split['label'],
                                      'scheme': split['scheme'], 'hash': split['partition_hash']},
        'label_policy': ('retrospective ({})'.format(args.cutoff_mode) if train_ds.retrospective
                         else 'pre-harvest: cutoff {} {} d'.format(args.cutoff_mode, args.cutoff_days)),
        'streams': {n: l['out_channels'] for n, l in train_ds.layout.items()},
        'model': {'descriptor': model.descriptor()['config'],
                  'parameters': int(sum(p.numel() for p in model.parameters()))},
        'excluded_sources': ['coord_x/y/z (unverified, metadata only)', 'row/col/field/farm ids',
                             'harvest date as input', 'yield_ground_truth attrs']
                            + ([] if args.normalization == 'supplied' else ['supplied stats-*'])
                            + ([] if args.soil_uncertainty == 'ancillary' else ['soil uncertainty']),
        'objectives': {k: v['point_timeseries'] for k, v in OBJECTIVE_ROUTING.items()},
        'provenance': PROVENANCE,
        'pair_filter': {'countries': args.countries, 'crops': args.crops,
                        'crop_context': getattr(model, 'use_crop_context', False)},
        'normalization': {'features': args.normalization, 'target': args.target_normalization,
                          'fill_value': args.fill_value},
        'train_seasons': len(parts['train']), 'val_seasons': len(parts['val']),
        'test_seasons': len(parts['test']),
        'history': history, 'best_val': best, 'transfer': transfer,
        'io': {'train_samples': samples_seen, 'data_wait_seconds': round(data_time, 1),
               'samples_per_second': round(samples_seen / max(1e-6, sum(h['seconds'] for h in history)), 1)},
    }
    if knowledge_ref is not None:
        report['knowledge'] = knowledge_ref
    if args.mode == 'pretrain':
        torch.save(model.sensor_state_dict(), out_dir / 'sensor_checkpoint.pth')
        # disposable pretraining heads (SSL decoders; knowledge projectors and
        # prototypes), kept for the success-criteria diagnostics (I1, I2, I4)
        heads = {k: v.detach().cpu() for k, v in trainable.state_dict().items()
                 if not k.startswith(('model.', 'ssl.model.'))}
        torch.save({'state_dict': heads, 'knowledge': bool(args.knowledge),
                    'control': args.knowledge_control if args.knowledge else None,
                    'permutation': (None if getattr(trainable, 'permutation', None) is None
                                    else trainable.permutation.tolist())},
                   out_dir / 'pretrainer_heads.pth')
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
        if args.eval_drop_stream:
            ps, ys, ss, _, _ = predict(model, test_ds, args, device,
                                       drop=(args.eval_drop_stream, args.eval_drop_frac))
            report['test_stress'] = evaluate_predictions(ps, ys, ss, test_ds.seasons)
            report['test_stress']['label'] = '{} removed for {:.0%} of test cells'.format(
                args.eval_drop_stream, args.eval_drop_frac)
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
    if args.mode == 'pretrain' and args.pretrain_diagnostics:
        # never fails the run: the checkpoint is the deliverable, diagnostics can be redone
        try:
            from yieldsat_pretrain_diagnostics import run as run_diagnostics
            run_diagnostics(out_dir, device=str(device), artifact_root=args.artifact_root,
                            source_root=args.source_root, skip_source_check=args.skip_source_check)
            print('diagnostics written', flush=True)
        except Exception as exc:  # noqa: BLE001
            (out_dir / 'diagnostics_error.txt').write_text(repr(exc))
            print('diagnostics failed:', repr(exc), flush=True)
    wb.finish(report)
    if 'test' in report:
        t = report['test']['overall']
        print('TEST', report['test']['label'], json.dumps({
            'pixel_rmse': t['pixel']['rmse'], 'pixel_r2': t['pixel']['r2'],
            'field_rmse': t['field_level']['rmse'], 'field_r2': t['field_level']['r2']}))
    return report


if __name__ == '__main__':
    main(get_args_parser().parse_args())
