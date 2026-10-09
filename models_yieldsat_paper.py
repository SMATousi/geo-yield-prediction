# --------------------------------------------------------
# The YieldSAT paper's best models (spec/yieldsat-paper-models.md, PM-02), on 5x5 window batches from
# dataset/yieldsat_window.py:
#   Paper3DLSTM      3D-LSTM, input fusion (Miranda et al. [32]; thesis A.1.2)
#   Paper3DConvLSTM  3D-ConvLSTM, input fusion (Helber et al. [13]; thesis ch. 7)
#   PaperAFF         AFF / MMAF, feature fusion with attention pooling (Miranda et al. [32]; thesis 4.5, A.1.2)
#   PaperMMGF        MMGF, gated feature fusion (Mena et al. [25]; sizes assumed, see the spec)
# Values are used as delivered (fill_value where invalid), masks are not used, as in the paper. Interface as
# models_yieldsat.PaperLSTMBaseline: forward(batch) -> standardized yield, loss(batch) -> (pred, MSE).
# --------------------------------------------------------

import math

import torch
import torch.nn as nn

from dataset.yieldsat_window import CENTRE, K, R
from dataset.yieldsat_schema import CONTRACT_KEY
from models_heads import masked_yield_loss

S = 2 * R + 1
TEMPORAL_FUSION_STREAMS = ('yieldsat_s2', 'yieldsat_weather')


def _centre(v):
    """Centre cell of a window stream, (B, 25, ...) or centre-only (B, 1, ...), as float32."""
    return (v[:, CENTRE] if v.shape[1] == K else v[:, 0]).float()


def _window_cube(batch, layout, names=None):
    """Input fusion: (B, C, T, 5, 5) from the window streams; static streams repeated over T."""
    win = batch['win_inputs']
    names = names or list(layout)
    T = next(win[n].shape[2] for n in names if layout[n]['temporal'])
    parts = []
    for n in names:
        v = win[n].float()
        if layout[n]['temporal']:                                   # (B, K, T, C)
            parts.append(v.permute(0, 3, 2, 1))
        else:                                                       # (B, K, C)
            parts.append(v.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, T, -1))
    x = torch.cat(parts, dim=1)                                      # (B, C, T, K)
    return x.reshape(x.shape[0], x.shape[1], T, S, S)


def _last_valid(out, batch):
    """LSTM output at each sample's last in-season slot (B, T, H) -> (B, H). On the dense series a season fills
    ~15-35 of 72 slots; reading the last slot instead means ~40-55 padding steps after the season, which washed
    the signal out for the feature-fusion models (2026-10-09, cluster AFF/MMGF train loss ~0.7 on ARG/BRA/URG)."""
    tv = batch['time_valid']
    T = tv.shape[1]
    last = (T - 1) - torch.flip(tv, dims=[1]).float().argmax(dim=1)
    return out[torch.arange(out.shape[0], device=out.device), last]


def _mse(pred, batch):
    target = torch.where(batch['target_valid'], batch['target'], torch.full_like(batch['target'], float('nan')))
    return masked_yield_loss(pred, target, loss='mse')


class _PaperModel(nn.Module):
    name = 'paper'
    window = 'all'                  # streams the model reads over the 5x5 window (YieldSATWindowDataset ``window``)

    def loss(self, batch, apply_dropout=True):
        pred = self.forward(batch)
        return pred, _mse(pred, batch)

    def descriptor(self):
        return {'contract': CONTRACT_KEY, 'mode': 'point_window5', 'model': self.name,
                'streams': {n: list(l['out_channels']) for n, l in self.layout.items()}, 'config': self.config}

    def sensor_state_dict(self):
        return {'descriptor': self.descriptor(), 'state_dict': self.state_dict()}

    def load_sensor_state_dict(self, payload, **kwargs):
        raise ValueError('{} does not support sensor transfer'.format(self.name))


class Block3DLSTM(nn.Module):
    """Conv3d (1, 5, 5) over the window -> BatchNorm -> activation -> LSTM over slots; last-slot hidden."""

    def __init__(self, in_ch, hidden=64, layers=2, act='leaky'):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_ch, 64, kernel_size=(1, S, S)), nn.BatchNorm3d(64),
                                  nn.LeakyReLU() if act == 'leaky' else nn.ReLU())
        self.lstm = nn.LSTM(64, hidden, num_layers=layers, batch_first=True)

    def forward(self, x, batch=None):                                # (B, C, T, 5, 5)
        h = self.conv(x).flatten(2).transpose(1, 2)                   # (B, T, 64)
        out, _ = self.lstm(h)
        return out[:, -1] if batch is None else _last_valid(out, batch)


def _head(d):
    return nn.Sequential(nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 1))


class Paper3DLSTM(_PaperModel):
    name = 'paper_3dlstm'

    def __init__(self, layout, hidden=64, layers=2):
        super().__init__()
        self.layout = layout
        self.streams = tuple(layout)
        c = sum(len(l['out_channels']) for l in layout.values())
        self.block = Block3DLSTM(c, hidden, layers, act='leaky')
        self.head = _head(hidden)
        self.use_crop_context = False
        self.config = {'model': self.name, 'in_channels': c, 'hidden': hidden, 'layers': layers}

    def forward(self, batch, apply_dropout=True):
        return self.head(self.block(_window_cube(batch, self.layout))).squeeze(-1)


class ConvLSTMCell(nn.Module):
    def __init__(self, ch, hidden, kernel):
        super().__init__()
        self.hidden = hidden
        self.gates = nn.Conv2d(ch + hidden, 4 * hidden, kernel, padding=kernel // 2)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class Paper3DConvLSTM(_PaperModel):
    name = 'paper_3dconvlstm'

    def __init__(self, layout, hidden=64, kernel=9):
        super().__init__()
        self.layout = layout
        self.streams = tuple(layout)
        c = sum(len(l['out_channels']) for l in layout.values())
        self.embed = nn.Sequential(nn.Conv3d(c, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)), nn.BatchNorm3d(64),
                                   nn.ReLU())
        self.cell = ConvLSTMCell(64, hidden, kernel)
        self.out = nn.Conv2d(hidden, 1, 1)
        self.use_crop_context = False
        self.config = {'model': self.name, 'in_channels': c, 'hidden': hidden, 'kernel': kernel}

    def forward(self, batch, apply_dropout=True):
        x = self.embed(_window_cube(batch, self.layout))              # (B, 64, T, 5, 5)
        B, _, T = x.shape[:3]
        h = x.new_zeros(B, self.cell.hidden, S, S)
        c = torch.zeros_like(h)
        for t in range(T):
            h, c = self.cell(x[:, :, t], (h, c))
        return self.out(h)[:, 0, R, R]


class PaperAFF(_PaperModel):
    """AFF / MMAF: S2 by the 3D-LSTM block, weather by an LSTM on the centre cell, each static modality by
    Conv2d(5x5) + Linear; scaled dot-product attention pooling with a learnable query."""
    name = 'paper_aff'
    window = 's2_static'

    def __init__(self, layout, dim=64, attn_dropout=0.2):
        super().__init__()
        self.layout = layout
        self.streams = tuple(layout)
        enc = {}
        for n, l in layout.items():
            c = len(l['out_channels'])
            if n == 'yieldsat_s2':
                enc[n] = Block3DLSTM(c, dim, 2, act='relu')
            elif l['temporal']:
                enc[n] = nn.LSTM(c, dim, num_layers=2, batch_first=True)
            else:
                enc[n] = nn.Sequential(nn.Conv2d(c, dim, S), nn.BatchNorm2d(dim), nn.ReLU(), nn.Flatten(),
                                       nn.Linear(dim, dim), nn.ReLU())
        self.enc = nn.ModuleDict(enc)
        self.query = nn.Parameter(torch.randn(1, 1, dim) / math.sqrt(dim))
        self.key = nn.Linear(dim, dim)
        self.drop = nn.Dropout(attn_dropout)
        self.out = nn.Linear(dim, 1)
        self.use_crop_context = False
        self.config = {'model': self.name, 'dim': dim, 'modalities': list(layout)}
        self.last_attention = None

    def _features(self, batch):
        win = batch['win_inputs']
        feats = []
        for n, l in self.layout.items():
            v = win[n]
            if n == 'yieldsat_s2':
                v = v.float()
                x = v.permute(0, 3, 2, 1).reshape(v.shape[0], v.shape[3], v.shape[2], S, S)
                feats.append(self.enc[n](x, batch))
            elif l['temporal']:
                out, _ = self.enc[n](_centre(v))                         # (B, T, C) centre cell
                feats.append(_last_valid(out, batch))
            else:
                v = v.float()
                feats.append(self.enc[n](v.permute(0, 2, 1).reshape(v.shape[0], v.shape[2], S, S)))
        return torch.stack(feats, dim=1)                                 # (B, M, dim)

    def forward(self, batch, apply_dropout=True):
        f = self._features(batch)
        scores = (self.query * self.key(f)).sum(-1) / math.sqrt(f.shape[-1])       # (B, M)
        w = torch.softmax(scores, dim=-1)
        self.last_attention = w.detach()
        w = self.drop(w) if apply_dropout else w
        return self.out((w.unsqueeze(-1) * f).sum(1)).squeeze(-1)


class PaperMMGF(_PaperModel):
    """MMGF: per-modality encoders on the centre pixel (LSTM for temporal, MLP for static) and a softmax gate
    over modalities computed from their concatenated features."""
    name = 'paper_mmgf'
    window = 'none'

    def __init__(self, layout, dim=64):
        super().__init__()
        self.layout = layout
        self.streams = tuple(layout)
        enc = {}
        for n, l in layout.items():
            c = len(l['out_channels'])
            enc[n] = (nn.LSTM(c, dim, num_layers=2, batch_first=True) if l['temporal'] else
                      nn.Sequential(nn.Linear(c, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU()))
        self.enc = nn.ModuleDict(enc)
        M = len(layout)
        self.gate = nn.Linear(M * dim, M)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.use_crop_context = False
        self.config = {'model': self.name, 'dim': dim, 'modalities': list(layout)}
        self.last_gate = None

    def forward(self, batch, apply_dropout=True):
        win = batch['win_inputs']
        feats = []
        for n, l in self.layout.items():
            v = _centre(win[n])
            if l['temporal']:
                out, _ = self.enc[n](v)
                feats.append(_last_valid(out, batch))
            else:
                feats.append(self.enc[n](v))
        f = torch.stack(feats, dim=1)                                    # (B, M, dim)
        g = torch.softmax(self.gate(f.flatten(1)), dim=-1)
        self.last_gate = g.detach()
        return self.head((g.unsqueeze(-1) * f).sum(1)).squeeze(-1)


PAPER_MODELS = {'paper_3dlstm': Paper3DLSTM, 'paper_3dconvlstm': Paper3DConvLSTM, 'paper_aff': PaperAFF,
                'paper_mmgf': PaperMMGF}
