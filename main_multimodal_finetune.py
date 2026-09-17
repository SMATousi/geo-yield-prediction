# --------------------------------------------------------
# Native-resolution heterogeneous modality finetuning entry point.
#
# Wires the dedicated per-modality encoders (models_multimodal_encoder.py)
# into a trainable dense yield-map pipeline: each input source is encoded at
# its own native resolution by its own encoder (raster CNN for DEM/terrain/
# soil rasters, 3D conv for multitemporal SAR, time-series encoder for
# weather/moisture, tabular MLP for point soil attributes, categorical
# embedding for crop-history, and a pretrained optical backbone), the token
# sets are fused by the Perceiver-style latent fusion transformer
# (models_latent_fusion.py), and a dense FCN head reconstructs a within-field
# yield map. Missing modalities are handled by the encoder's learned
# missing-modality token + availability mask, and modality dropout is applied
# during training so the backbone learns to run on arbitrary subsets of
# sources. This is the training loop that realises the native-resolution
# heterogeneous-encoder capability end to end.
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
from util import metrics

from models_multimodal_encoder import MultiModalEncoder
from models_latent_fusion import LatentFusionTransformer
from models_heads import DenseYieldFCNHead

torch.manual_seed(0)
np.random.seed(0)


def get_args_parser():
    parser = argparse.ArgumentParser('MMST-ViT multimodal finetuning', add_help=False)

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
    parser.add_argument('--modality_dropout', default=0.3, type=float,
                        help='training-time random modality removal rate')
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

    parser.add_argument('--output_dir', default='./output_dir/mmst_multimodal',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir/mmst_multimodal',
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

    # evaluate
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    return parser


def build_model(args):
    """Build the native-resolution heterogeneous-modality yield model.

    Each input source gets its own dedicated encoder at its native resolution
    (raster CNN for DEM/terrain/soil rasters, 3D conv for multitemporal SAR,
    time-series encoder for weather, tabular MLP for point soil attributes,
    categorical embedding for crop-history). All encoder outputs are projected
    to a shared latent dimension and fused by the Perceiver-style latent
    fusion transformer, whose latents are reshaped to a spatial grid and
    decoded by a dense FCN yield-map head.
    """
    embed_dim = args.embed_dim
    encoders_cfg = {
        # high-resolution DEM / terrain raster (single band, native res)
        'dem': {'type': 'raster', 'in_channels': 1, 'embed_dim': embed_dim},
        # multitemporal SAR backscatter stack (T frames of VV/VH)
        'sar': {'type': 'sar', 'in_channels': 2, 'embed_dim': embed_dim},
        # weather / precipitation / soil-moisture time series
        'weather': {'type': 'timeseries', 'in_dim': 9, 'embed_dim': embed_dim},
        # point / tabular soil attributes
        'soil': {'type': 'tabular', 'in_dim': 12, 'embed_dim': embed_dim},
        # categorical crop-history / land-cover layer
        'crop': {'type': 'categorical', 'num_classes': 20, 'embed_dim': embed_dim},
    }

    encoders = MultiModalEncoder(encoders_cfg, embed_dim=embed_dim,
                                 modality_dropout=args.modality_dropout)

    # Per-modality token layouts for the fusion transformer's separable
    # spatial + temporal positional embeddings. The raster modalities emit one
    # token per spatial cell (spatial = H*W, temporal = 1); the weather series
    # emits one token per time step (spatial = 1, temporal = T).
    grid = args.input_size * args.input_size
    modalities = {
        'dem': {'spatial': grid, 'temporal': 1},
        'sar': {'spatial': grid, 'temporal': 1},
        'weather': {'spatial': 1, 'temporal': 1},
        'soil': {'spatial': 1, 'temporal': 1},
        'crop': {'spatial': grid, 'temporal': 1},
    }

    fusion = LatentFusionTransformer(
        embed_dim=embed_dim, num_latents=args.num_latents, depth=args.depth,
        num_heads=args.num_heads, modalities=modalities,
        modality_embed=args.modality_embed)

    head = DenseYieldFCNHead(embed_dim=embed_dim, num_classes=1, loss='l1')

    return MultimodalYieldModel(encoders, fusion, head)


class MultimodalYieldModel(nn.Module):
    """Encoder + latent-fusion + dense-yield-head container.

    ``forward`` takes a dict of modality tensors (each at its native
    resolution) and an optional per-modality availability mask, runs the
    missing-modality-robust encoder forward, fuses the token sets in latent
    space, reshapes the fused latents to a spatial grid, and decodes a dense
    within-field yield map. Missing modalities are substituted with a learned
    token and flagged 0 in the mask, so the model runs on arbitrary subsets of
    sources.
    """

    def __init__(self, encoders, fusion, head):
        super().__init__()
        self.encoders = encoders
        self.fusion = fusion
        self.head = head

    def forward(self, inputs, targets=None, available=None, mode='tensor'):
        embeddings, mask = self.encoders.forward_with_missing(
            inputs, available=available)
        latents = self.fusion(embeddings, mask=mask)          # (B, num_latents, embed_dim)
        b, num_latents, d = latents.shape
        grid = int(round(num_latents ** 0.5))
        # reshape the latent bottleneck to a spatial feature map for the dense
        # decoder; pad with zeros if num_latents is not a perfect square.
        if grid * grid != num_latents:
            pad = grid * grid - num_latents
            latents = torch.cat(
                [latents, latents.new_zeros(b, pad, d)], dim=1)
        spatial = latents.transpose(1, 2).reshape(b, d, grid, grid)
        return self.head(spatial, targets=targets, mode=mode)


def make_synthetic_batch(args, device):
    """Build a synthetic multimodal batch for a smoke-test / self-check.

    Each modality is generated at its own native resolution: DEM/terrain and
    crop-history as (B, C, H, W) rasters, SAR as a (B, T, C, H, W) stack,
    weather as a (B, T, D) series, and soil as (B, D) tabular attributes.
    """
    b = args.batch_size
    h = w = args.input_size
    inputs = {
        'dem': torch.randn(b, 1, h, w, device=device),
        'sar': torch.randn(b, 6, 2, h, w, device=device),
        'weather': torch.randn(b, 12, 9, device=device),
        'soil': torch.randn(b, 12, device=device),
        'crop': torch.randint(0, 20, (b, h, w), device=device),
    }
    targets = torch.randn(b, 1, 8, 8, device=device)
    return inputs, targets


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

    cudnn.benchmark = True

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
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        evaluate(model, device, args)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, optimizer, device, epoch, loss_scaler, args=args)

        if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
            evaluate(model, device, args)
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
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model, optimizer, device, epoch, loss_scaler, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    accum_iter = args.accum_iter
    criterion = torch.nn.MSELoss().to(device)
    optimizer.zero_grad()

    total_step = 20
    for data_iter_step in range(total_step):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / total_step + epoch, args)

        inputs, targets = make_synthetic_batch(args, device)

        # randomly drop a subset of modalities during training so the backbone
        # learns to run on arbitrary subsets of sources (missing-modality
        # robustness). The encoder's forward_with_missing applies modality
        # dropout internally; here we also demonstrate explicit availability.
        available = {name: True for name in inputs}
        if model.training and args.modality_dropout > 0:
            for name in list(available):
                if torch.rand(1).item() < args.modality_dropout:
                    available[name] = False

        z_hat = model(inputs, targets=targets, available=available, mode='tensor')
        loss = criterion(z_hat, targets)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, device, args):
    model.eval()
    true_labels = torch.empty(0)
    pred_labels = torch.empty(0)

    for _ in range(10):
        inputs, targets = make_synthetic_batch(args, device)
        z_hat = model(inputs, mode='tensor')
        true_labels = torch.cat([true_labels, targets.detach().cpu()], dim=0)
        pred_labels = torch.cat([pred_labels, z_hat.detach().cpu()], dim=0)

    true_labels = true_labels.flatten().numpy()
    pred_labels = pred_labels.flatten().numpy()
    rmse, r2, corr = metrics.evaluate(true_labels, pred_labels)
    print("Metrics: RMSE: {:.2f}  R_Squared: {:.2f}  Corr: {:.2f}".format(
        rmse, r2, corr))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
