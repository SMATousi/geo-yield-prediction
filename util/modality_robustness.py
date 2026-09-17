"""Modality-robustness evaluation: progressively remove modalities.

The missing-modality robustness requirement asks that the model operate on
arbitrary subsets of inputs and that performance be evaluated as progressively
more modalities are removed, to determine which combinations provide the most
predictive information. This module provides ``evaluate_modality_robustness``,
which runs a trained multimodal yield model on a batch under every subset of
available modalities (from full input down to each single modality) and reports
dense yield metrics (RMSE, MAE, R2, spatial correlation, zone preservation) for
each subset, so the marginal contribution of each source can be measured.

It reuses the repo's dense regression evaluator (``util.metrics.evaluate_regression``)
and the spatially-aware ``SpatialYieldMetric``, so no new dependency is
introduced.
"""

import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.metrics import evaluate_regression, SpatialYieldMetric


def _subset_available(all_names, subset):
    """Build a per-modality availability dict for a kept subset."""
    return {name: (name in subset) for name in all_names}


def evaluate_modality_robustness(model, inputs, targets, modality_names=None,
                                 device='cpu', min_modalities=1):
    """Evaluate a trained yield model under every subset of available modalities.

    Args:
        model: a ``MultimodalYieldModel`` (or any callable accepting
            ``(inputs, available=...)`` and returning dense yield maps).
        inputs: dict of modality-name -> ``(B, ...)`` tensors.
        targets: ``(B, 1, H, W)`` observed yield maps.
        modality_names: optional list of modality names to ablate; defaults to
            the keys of ``inputs``.
        device: torch device.
        min_modalities: minimum number of modalities to keep in a subset.

    Returns:
        A list of dicts, one per subset, each with ``modalities`` (the kept
        subset), ``num_modalities``, and the dense yield metrics (mse, rmse,
        mae, r2, pcc, spatial_corr, zone_preservation).
    """
    if modality_names is None:
        modality_names = list(inputs.keys())
    model.eval()
    results = []
    for r in range(len(modality_names), min_modalities - 1, -1):
        for subset in itertools.combinations(modality_names, r):
            available = _subset_available(modality_names, subset)
            with torch.no_grad():
                pred = model(inputs, available=available, mode='tensor')
            pred = pred.detach().cpu().numpy()
            true = targets.detach().cpu().numpy()

            metrics = evaluate_regression(true, pred)
            spatial = SpatialYieldMetric()
            spatial.process(pred, true)
            spatial_summary = spatial.compute()
            metrics['spatial_corr'] = spatial_summary.get('spatial_corr', float('nan'))
            metrics['zone_preservation'] = spatial_summary.get(
                'zone_preservation', float('nan'))
            metrics['modalities'] = list(subset)
            metrics['num_modalities'] = r
            results.append(metrics)
    return results


def summarize_robustness(results):
    """Summarise a robustness sweep by number of kept modalities.

    Groups the per-subset results from :func:`evaluate_modality_robustness` by
    ``num_modalities`` and returns, for each group, the mean RMSE / R2 and the
    best single-subset RMSE, so the effect of progressively removing modalities
    is visible at a glance.
    """
    by_count = {}
    for res in results:
        n = res['num_modalities']
        by_count.setdefault(n, []).append(res)
    summary = []
    for n in sorted(by_count, reverse=True):
        group = by_count[n]
        rmse = [g['rmse'] for g in group if np.isfinite(g['rmse'])]
        r2 = [g['r2'] for g in group if np.isfinite(g['r2'])]
        best = min(group, key=lambda g: g['rmse'])
        summary.append({
            'num_modalities': n,
            'num_subsets': len(group),
            'mean_rmse': float(np.mean(rmse)) if rmse else float('nan'),
            'mean_r2': float(np.mean(r2)) if r2 else float('nan'),
            'best_rmse': float(best['rmse']),
            'best_subset': best['modalities'],
        })
    return summary


if __name__ == "__main__":
    # Build a tiny multimodal yield model and run a robustness sweep on a
    # synthetic batch to exercise the progressive-modality-removal evaluation.
    from models_multimodal_encoder import MultiModalEncoder
    from models_latent_fusion import LatentFusionTransformer
    from models_heads import DenseYieldFCNHead

    class TinyYieldModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoders = MultiModalEncoder(
                {'dem': {'type': 'raster', 'in_channels': 1},
                 'sar': {'type': 'sar', 'in_channels': 2},
                 'weather': {'type': 'timeseries', 'in_dim': 9}},
                embed_dim=64, modality_dropout=0.0)
            self.fusion = LatentFusionTransformer(
                embed_dim=64, num_latents=16, depth=2, num_heads=2,
                modalities={'dem': {'spatial': 16, 'temporal': 1},
                            'sar': {'spatial': 16, 'temporal': 1},
                            'weather': {'spatial': 1, 'temporal': 1}},
                modality_embed=16)
            self.head = DenseYieldFCNHead(embed_dim=64, num_classes=1, loss='l1')

        def forward(self, inputs, targets=None, available=None, mode='tensor'):
            embeddings, mask = self.encoders.forward_with_missing(
                inputs, available=available)
            latents = self.fusion(embeddings, mask=mask)
            b, num_latents, d = latents.shape
            grid = int(round(num_latents ** 0.5))
            if grid * grid != num_latents:
                pad = grid * grid - num_latents
                latents = torch.cat([latents, latents.new_zeros(b, pad, d)], dim=1)
            spatial = latents.transpose(1, 2).reshape(b, d, grid, grid)
            return self.head(spatial, targets=targets, mode=mode)

    model = TinyYieldModel()
    inputs = {
        'dem': torch.randn(2, 1, 4, 4),
        'sar': torch.randn(2, 6, 2, 4, 4),
        'weather': torch.randn(2, 12, 9),
    }
    targets = torch.randn(2, 1, 4, 4)

    results = evaluate_modality_robustness(model, inputs, targets)
    print('n_subsets:', len(results))
    for res in results:
        print('  mods=%d %-22s rmse=%.3f r2=%.3f' % (
            res['num_modalities'], '+'.join(res['modalities']),
            res['rmse'], res['r2']))
    print('summary:')
    for s in summarize_robustness(results):
        print('  ', s)
