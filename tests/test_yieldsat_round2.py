"""Improvement plan round 2(a): point-model early fusion (S4), season level (S5),
neighbourhood stream (S6)."""

import numpy as np
import pytest
import torch

from dataset.yieldsat_dataset import stream_layout
from models_yieldsat import YieldSATPointModel
from yieldsat_build_neighbourhood import field_neighbourhood

STREAMS = ['yieldsat_s2', 'yieldsat_weather', 'yieldsat_dem', 'yieldsat_terrain', 'yieldsat_soil']


def _batch(layout, B=16, T=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    b = {'inputs': {}, 'masks': {}, 'available': {}}
    for n, l in layout.items():
        C = len(l['out_channels'])
        sh = (B, T, C) if l['temporal'] else (B, C)
        b['inputs'][n] = torch.randn(sh, generator=g)
        b['masks'][n] = torch.rand(sh, generator=g) > 0.2
        b['available'][n] = torch.ones(B)
    b['time_features'] = torch.randn(B, T, 3, generator=g)
    b['time_valid'] = torch.rand(B, T, generator=g) > 0.3
    b['crop'] = torch.randint(0, 3, (B,), generator=g)
    b['target'] = torch.randn(B, generator=g)
    b['target_valid'] = torch.ones(B, dtype=torch.bool)
    b['season_target'] = torch.randn(B, generator=g)
    return b


def test_early_fusion_trains_and_ignores_static_on_undated_slots():
    layout = stream_layout(STREAMS, aspect_encoding='cyclic')
    torch.manual_seed(0)
    m = YieldSATPointModel(layout, fusion='early', early_hidden=32, early_depth=1).eval()
    b = _batch(layout)
    x = m._early_input(b)
    n_static = sum(len(l['out_channels']) for l in layout.values() if not l['temporal'])
    assert x.shape[-1] == 2 * sum(len(l['out_channels']) for l in layout.values()) + 3
    # static masks are zero on undated slots
    undated = ~b['time_valid']
    sm = x[..., -3 - n_static:-3]
    assert torch.all(sm[undated] == 0)
    m.train()
    pred, loss = m.loss(b)
    loss.backward()
    assert pred.shape == (16,) and m.early_enc.value_proj.weight.grad.abs().sum() > 0


def test_level_head_adds_season_term_and_its_loss():
    layout = stream_layout(STREAMS, aspect_encoding='cyclic')
    torch.manual_seed(0)
    m = YieldSATPointModel(layout, level_head=True)
    b = _batch(layout)
    m.eval()                                       # no modality dropout: deterministic passes
    with torch.no_grad():
        pred, level = m(b, return_level=True)
        head_only = m.head(m.forward_features(b, apply_dropout=False)[0])
    assert torch.allclose(pred, head_only + level, atol=1e-6)
    m.train()
    _, loss = m.loss(b)
    loss.backward()
    assert m.level_mlp[1].weight.grad.abs().sum() > 0 and m.level_weather.value_proj.weight.grad.abs().sum() > 0


def test_neighbourhood_mean_matches_brute_force():
    rng = np.random.default_rng(1)
    H, W = 10, 8
    grid = rng.normal(size=(H, W, 2, 3))
    grid[rng.random(grid.shape) < 0.3] = np.nan
    present = rng.random((H, W)) < 0.7
    rr, cc = np.nonzero(present)
    got = field_neighbourhood(grid[rr, cc], rr, cc)
    for k, (r, c) in enumerate(zip(rr, cc)):
        acc = [grid[r + dr, c + dc] for dr in range(-2, 3) for dc in range(-2, 3)
               if (dr or dc) and 0 <= r + dr < H and 0 <= c + dc < W and present[r + dr, c + dc]]
        exp = np.nanmean(np.stack(acc), 0) if acc else np.full((2, 3), np.nan)
        with np.errstate(invalid='ignore'):
            assert np.allclose(got[k], exp, equal_nan=True)
