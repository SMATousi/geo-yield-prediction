"""Relation-matching loss and within-field metrics (spec/yieldsat-relational-loss.md, RL-02/RL-03).

relation_loss: for pairs (i, j) of the same field season within ``radius`` grid cells,
    Huber( (p_i - p_j) - (t_i - t_j) ), weighted ``near_weight`` for pairs at distance <= 1
    (adjacent cells carry the most harvester noise) and 1 otherwise; averaged over the
    weighted pairs. Invariant to adding a constant to a field's predictions: it supervises
    the within-field pattern and its amplitude, not the level, and never pulls pixels
    towards equal values unless their targets are equal.
variance_loss: (std(p) - std(t))^2 per cluster of the same field season (against shrinkage).
within_field_metrics: pooled within-field R^2, median per-field R^2 / Pearson r /
    variability ratio, and the within-field share of pixel variance.
"""
import numpy as np
import torch
import torch.nn.functional as F


def relation_loss(pred, target, valid, season, grid_row, grid_col, radius=5, near_weight=0.5, delta=1.0):
    pred, target = pred.float().reshape(-1), target.float().reshape(-1)
    v = valid.reshape(-1) & torch.isfinite(target)
    same = season[:, None] == season[None, :]
    d = torch.maximum((grid_row[:, None] - grid_row[None, :]).abs(), (grid_col[:, None] - grid_col[None, :]).abs())
    upper = torch.triu(torch.ones_like(same), diagonal=1)
    pairs = same & upper & (d <= radius) & (d > 0) & v[:, None] & v[None, :]
    if not pairs.any():
        return pred.sum() * 0.0
    w = torch.where(d <= 1, torch.full_like(pred[:1], near_weight), torch.ones_like(pred[:1])).expand_as(d)
    w = torch.where(pairs, w, torch.zeros_like(w))
    t = torch.where(v, target, torch.zeros_like(target))
    diff = (pred[:, None] - pred[None, :]) - (t[:, None] - t[None, :])
    h = F.huber_loss(diff, torch.zeros_like(diff), delta=delta, reduction='none')
    return (w * h).sum() / w.sum().clamp_min(1e-8)


def variance_loss(pred, target, valid, season):
    pred, target = pred.float().reshape(-1), target.float().reshape(-1)
    v = valid.reshape(-1) & torch.isfinite(target)
    out = []
    for s in torch.unique(season[v]):
        m = v & (season == s)
        if m.sum() >= 3:
            out.append((pred[m].std() - target[m].std()) ** 2)
    return torch.stack(out).mean() if out else pred.sum() * 0.0


def within_field_metrics(pred, target, season, min_pixels=50):
    pred, target, season = np.asarray(pred, np.float64), np.asarray(target, np.float64), np.asarray(season)
    ok = np.isfinite(pred) & np.isfinite(target)
    pred, target, season = pred[ok], target[ok], season[ok]
    order = np.argsort(season, kind='stable')
    pred, target, season = pred[order], target[order], season[order]
    cuts = np.flatnonzero(np.diff(season)) + 1
    sse = sst = 0.0
    r2s, rs, ratios = [], [], []
    for p, t in zip(np.split(pred, cuts), np.split(target, cuts)):
        pc, tc = p - p.mean(), t - t.mean()
        e, s2 = float(np.sum((pc - tc) ** 2)), float(np.sum(tc ** 2))
        sse += e
        sst += s2
        if len(t) >= min_pixels and s2 > 0 and pc.std() > 0:
            r2s.append(1 - e / s2)
            rs.append(float(np.corrcoef(pc, tc)[0, 1]))
            ratios.append(float(pc.std() / tc.std()))
    total = float(np.sum((target - target.mean()) ** 2))
    return {'within_r2': 1 - sse / sst if sst > 0 else float('nan'),
            'median_field_r2': float(np.median(r2s)) if r2s else float('nan'),
            'median_field_r': float(np.median(rs)) if rs else float('nan'),
            'median_var_ratio': float(np.median(ratios)) if ratios else float('nan'),
            'within_share': sst / total if total > 0 else float('nan'), 'fields': len(r2s)}
