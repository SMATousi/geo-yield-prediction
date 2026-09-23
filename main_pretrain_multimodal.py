# --------------------------------------------------------
# Multimodal self-supervised pretraining entry point.
#
# Pretrains the shared multimodal backbone (native-resolution heterogeneous
# encoders + Perceiver latent fusion transformer) on unlabelled fields using
# the foundation-scale self-supervised objectives in
# models_multimodal_pretrain.py: masked spatial-token reconstruction, masked-
# modality modeling, temporal forecasting and cross-modal prediction. Unlike
# the PVT SimCLR pretraining (which is confined to contrastive learning
# between augmented Sentinel-2 views paired with weather), this loop trains
# the full multimodal backbone on arbitrary subsets of sources without any
# yield labels, so the learned field representation can later be transferred
# to the supervised yield-map head.
# --------------------------------------------------------

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn

import util.misc as misc
import util.lr_sched as lr_sched
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models_multimodal_pretrain import MultimodalSelfSupervisedPretrain

torch.manual_seed(0)
np.random.seed(0)


def get_args_parser():
    parser = argparse.ArgumentParser('MMST-ViT multimodal self-supervised pretraining', add_help=False)

    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--embed_dim', default=192, type=int, help='embed dimensions')
    parser.add_argument('--num_latents', default=64, type=int,
                        help='number of Perceiver latent bottleneck queries')
    parser.add_argument('--depth', default=4, type=int,
                        help='fusion transformer depth')
    parser.add_argument('--num_heads', default=3, type=int,
                        help='fusion transformer heads')
    parser.add_argument('--modality_embed', default=64, type=int,
                        help='modality-identity embedding dimension')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='fraction of spatial tokens masked for reconstruction')
    parser.add_argument('--modality_mask_ratio', default=0.5, type=float,
                        help='probability of dropping a whole modality (masked-modality modeling)')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations')
    parser.add_argument('--input_size', default=64, type=int, help='raster input size')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N')

    parser.add_argument('--output_dir', default='./output_dir/mmst_multimodal_pretrain',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir/mmst_multimodal_pretrain',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    return parser


def build_model(args):
    """Build the multimodal self-supervised pretraining framework.

    Each input source gets its own dedicated encoder at its native resolution
    (raster CNN for DEM/terrain/soil rasters, 3D conv for multitemporal SAR,
    time-series encoder for weather, tabular MLP for point soil attributes,
    categorical embedding for crop-history). All encoder outputs are projected
    to a shared latent dimension and fused by the Perceiver-style latent fusion
    transformer, which is pretrained with the four self-supervised objectives.
    """
    embed_dim = args.embed_dim
    encoders_cfg = {
        'dem': {'type': 'raster', 'in_channels': 1, 'embed_dim': embed_dim},
        'sar': {'type': 'sar', 'in_channels': 2, 'embed_dim': embed_dim},
        'weather': {'type': 'timeseries', 'in_dim': 9, 'embed_dim': embed_dim},
        'soil': {'type': 'tabular', 'in_dim': 12, 'embed_dim': embed_dim},
        'crop': {'type': 'categorical', 'num_classes': 20, 'embed_dim': embed_dim},
    }

    grid = args.input_size * args.input_size
    modalities = {
        'dem': {'spatial': grid, 'temporal': 1},
        'sar': {'spatial': grid, 'temporal': 1},
        'weather': {'spatial': 1, 'temporal': 1},
        'soil': {'spatial': 1, 'temporal': 1},
        'crop': {'spatial': grid, 'temporal': 1},
    }

    return MultimodalSelfSupervisedPretrain(
        encoders_cfg, embed_dim=embed_dim, num_latents=args.num_latents,
        depth=args.depth, num_heads=args.num_heads, modalities=modalities,
        modality_embed=args.modality_embed, mask_ratio=args.mask_ratio,
        modality_mask_ratio=args.modality_mask_ratio)


def make_synthetic_batch(args, device):
    """Build a synthetic multimodal batch of unlabelled fields.

    Each modality is generated at its own native resolution: DEM/terrain and
    crop-history as (B, C, H, W) rasters, SAR as a (B, T, C, H, W) stack,
    weather as a (B, T, D) series, and soil as (B, D) tabular attributes. No
    yield labels are provided -- the batch is used purely for self-supervised
    pretraining.
    """
    b = args.batch_size
    h = w = args.input_size
    return {
        'dem': torch.randn(b, 1, h, w, device=device),
        'sar': torch.randn(b, 6, 2, h, w, device=device),
        'weather': torch.randn(b, 12, 9, device=device),
        'soil': torch.randn(b, 12, device=device),
        'crop': torch.randint(0, 20, (b, h, w), device=device),
    }


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """Group parameters for AdamW: no weight decay for bias / norm layers.

    Mirrors timm's ``optim_factory.add_weight_decay`` (used by the base
    finetune script) without pulling in the timm dependency, so this entry
    point stays self-contained.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith('.bias') or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay},
    ]


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = device.type == 'cuda'
    print('SYNTHETIC DATA ONLY: pretraining losses are smoke-test results')

    model = build_model(args)
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    param_groups = add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler(device)

    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start pretraining for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, optimizer, device, epoch, loss_scaler, args=args)

        if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a",
                      encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Pretraining time {}'.format(total_time_str))


def train_one_epoch(model, optimizer, device, epoch, loss_scaler, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    total_step = 20
    for data_iter_step in range(total_step):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / total_step + epoch, args)

        inputs = make_synthetic_batch(args, device)

        # randomly drop a subset of modalities during training so the backbone
        # learns to run on arbitrary subsets of sources (missing-modality
        # robustness). The framework's masked-modality modeling also drops
        # whole modalities internally.
        available = {name: True for name in inputs}
        if model.training:
            for name in list(available):
                if torch.rand(1).item() < 0.3:
                    available[name] = False

        total_loss, losses = model(inputs, available=available)
        loss = total_loss
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        for k, v in losses.items():
            metric_logger.update(**{k: v.item()})
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
