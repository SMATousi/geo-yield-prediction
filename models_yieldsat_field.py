# --------------------------------------------------------
# Field encoder (spec/yieldsat-field-encoder.md, FE-02): learned field context for within-field yield.
#
#   pixel embedding   h = mean of YieldSATPointModel's fused latents (targets and context pixels alike)
#   option 1          context tokens c_k = h_k + phi_abs(p_k - centroid); SAB over the K tokens; PMA with Q
#                     seeds -> field tokens F (Q, D)
#   option 2          the target's h cross-attends to c_k + phi_rel(p_target - p_k) -> r
#   option 3          level l = MLP(mean F) (field yield level), deviation d = MLP([h, r, mean F]); yhat = l + d
#   loss              MSE(yhat, y) + fe_level_weight * MSE(l, field-season mean target)   (+ relation loss outside)
# Inputs only: the context pixels are the target field season's own cells (FieldContextDataset).
# --------------------------------------------------------

import math

import torch
import torch.nn as nn

from models_heads import masked_yield_loss
from models_yieldsat import YieldSATPointModel

POS_SCALE = 50.0                      # grid cells (500 m) per unit of position for the Fourier features


class FourierPos(nn.Module):
    """2-D offsets (cells) -> D: sin/cos at geometric frequencies, then an MLP."""

    def __init__(self, dim, n_freq=8):
        super().__init__()
        self.register_buffer('freq', 2.0 ** torch.arange(n_freq, dtype=torch.float32) * math.pi)
        self.mlp = nn.Sequential(nn.Linear(4 * n_freq + 2, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, xy):                                              # (..., 2) in cells
        x = xy / POS_SCALE
        a = x.unsqueeze(-1) * self.freq                                 # (..., 2, F)
        return self.mlp(torch.cat([x, torch.sin(a).flatten(-2), torch.cos(a).flatten(-2)], dim=-1))


class Block(nn.Module):
    """Pre-norm attention block (queries attend to keys/values) + MLP."""

    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        self.nq, self.nkv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.nm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, q, kv):
        q = q + self.attn(self.nq(q), self.nkv(kv), self.nkv(kv), need_weights=False)[0]
        return q + self.mlp(self.nm(q))


class FieldEncoderModel(nn.Module):

    def __init__(self, layout, seeds=4, heads=4, fe_level_weight=1.0, **point_kw):
        super().__init__()
        self.base = YieldSATPointModel(layout, **point_kw)
        D = self.base.config['embed_dim']
        self.layout = layout
        self.streams = tuple(layout)
        self.use_crop_context = self.base.use_crop_context
        self.pos_abs, self.pos_rel = FourierPos(D), FourierPos(D)
        self.sab = Block(D, heads)                                      # option 1: set self-attention
        self.seeds = nn.Parameter(torch.randn(1, seeds, D) / math.sqrt(D))
        self.pma = Block(D, heads)                                      # option 1: pooling by attention
        self.cross = Block(D, heads)                                    # option 2: pixel -> field context
        self.level = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Linear(D, 1))
        self.dev = nn.Sequential(nn.LayerNorm(3 * D), nn.Linear(3 * D, D), nn.GELU(), nn.Linear(D, 1))
        self.fe_level_weight = fe_level_weight
        self.config = dict(self.base.config, model='field_encoder', seeds=seeds, fe_heads=heads,
                           fe_level_weight=fe_level_weight)

    def _embed(self, batch, apply_dropout):
        latents, _ = self.base.forward_features(batch, apply_dropout=apply_dropout)
        return latents.mean(1)                                          # (N, D)

    def forward(self, batch, apply_dropout=True, return_level=False):
        h = self._embed(batch, apply_dropout)                           # (B, D) targets
        ctx = {'inputs': batch['ctx_inputs'], 'masks': batch['ctx_masks'], 'available': batch['ctx_available'],
               'crop': batch['ctx_crop'], 'time_features': batch['ctx_time_features'],
               'time_valid': batch['ctx_time_valid']}
        hc = self._embed(ctx, apply_dropout)                            # (G*K, D)
        G = int(batch['ctx_group'].max()) + 1
        K = hc.shape[0] // G
        D = hc.shape[1]
        hc = hc.view(G, K, D)
        pc = torch.stack([batch['ctx_grid_row'], batch['ctx_grid_col']], -1).float().view(G, K, 2)
        cent = pc.mean(1, keepdim=True)
        tok = hc + self.pos_abs(pc - cent)
        tok = self.sab(tok, tok)                                        # (G, K, D)
        F = self.pma(self.seeds.expand(G, -1, -1), tok)                 # (G, Q, D)
        f = F.mean(1)                                                   # (G, D)
        g = batch['tgt_group']
        pt = torch.stack([batch['grid_row'], batch['grid_col']], -1).float()            # (B, 2)
        kv = tok[g] + self.pos_rel(pt.unsqueeze(1) - pc[g])            # (B, K, D) relative offsets
        r = self.cross(h.unsqueeze(1), kv).squeeze(1)                   # (B, D)
        level = self.level(f).squeeze(-1)[g]                            # (B,)
        pred = level + self.dev(torch.cat([h, r, f[g]], -1)).squeeze(-1)
        return (pred, level) if return_level else pred

    def loss(self, batch, apply_dropout=True):
        pred, level = self.forward(batch, apply_dropout=apply_dropout, return_level=True)
        target = torch.where(batch['target_valid'], batch['target'], torch.full_like(batch['target'], float('nan')))
        loss = masked_yield_loss(pred, target, loss='mse')
        ok = batch['target_valid'] & torch.isfinite(batch['season_target'])
        if self.fe_level_weight > 0 and ok.any():
            loss = loss + self.fe_level_weight * ((level.float() - batch['season_target'])[ok] ** 2).mean()
        return pred, loss

    def descriptor(self):
        d = self.base.descriptor()
        d['model'] = 'field_encoder'
        d['config'] = self.config
        return d

    def sensor_state_dict(self):
        return {'descriptor': self.descriptor(), 'state_dict': self.state_dict()}

    def load_sensor_state_dict(self, payload, **kwargs):
        raise ValueError('the field encoder does not support sensor transfer')
