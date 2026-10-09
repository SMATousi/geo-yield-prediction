# --------------------------------------------------------
# 5x5 window dataset for the YieldSAT paper's spatial models (spec/yieldsat-paper-models.md, PM-01).
#
# One item is still one grid cell; its batch additionally carries the cell's 5x5 neighbourhood of the same
# field season, gathered through the point dataset's own batch path (identical normalization, cutoff, masks
# and fill):
#   win_inputs[stream] - (B, 25, T, C) temporal / (B, 25, C) static, float16, fill_value where invalid or outside;
#                        (B, 1, ...) for streams a model reads only at the centre (``window``)
#   win_masks[stream]  - validity, same shapes (only with ``return_masks``; the paper models ignore masks)
#   win_valid          - (B, 25) the neighbour exists (inside the field season and in this partition)
# Window batches are large (dense S2: 2048 x 25 x 72 x 12 values), hence float16 and centre-only streams: the
# DataLoader passes them through /dev/shm (2026-10-09: float32 + masks for every stream exhausted a 4 Gi shm).
# Position k = (dr + 2) * 5 + (dc + 2); the centre is k = 12 and equals the point item.
# Training augmentation: random 90-degree rotations of the window and temporal dropout of slots.
# --------------------------------------------------------

import numpy as np
import torch

from dataset.yieldsat_dataset import YieldSATPointDataset

R = 2                                    # window radius (5x5)
K = (2 * R + 1) ** 2
CENTRE = K // 2
_DR, _DC = np.meshgrid(np.arange(-R, R + 1), np.arange(-R, R + 1), indexing='ij')
_DR, _DC = _DR.ravel(), _DC.ravel()
# ROT[k][j] = source position of new position j after k quarter turns (np.rot90 convention)
ROT = np.stack([np.rot90(np.arange(K).reshape(2 * R + 1, 2 * R + 1), k).ravel() for k in range(4)])


class YieldSATWindowDataset(YieldSATPointDataset):
    """``augment=True`` (training only): random rot90 per sample and temporal dropout with probability
    ``temporal_dropout`` per slot (all temporal streams, all window cells)."""

    def __init__(self, *args, augment=False, temporal_dropout=0.2, window='all', return_masks=False, **kw):
        super().__init__(*args, **kw)
        # which streams get the full window: 'all', 's2_static' (S2 + static streams; other temporal streams at
        # the centre only) or 'none' (every stream at the centre)
        if window == 'all':
            self.window_streams = set(self.layout)
        elif window == 's2_static':
            self.window_streams = {n for n, l in self.layout.items() if n == 'yieldsat_s2' or not l['temporal']}
        elif window == 'none':
            self.window_streams = set()
        else:
            raise ValueError('window must be all, s2_static or none')
        self.return_masks = return_masks
        self.augment = augment
        self.temporal_dropout = float(temporal_dropout)
        self._win_rng = None
        self.win_items = self._build_window_index()

    def _build_window_index(self):
        """(N, 25) dataset item index of each neighbour, -1 if outside the field season or absent."""
        out = np.full((len(self.row), K), -1, dtype=np.int64)
        ranges = {int(self.season_of[a]): (a, b) for a, b in self.season_ranges}
        for ci, country in enumerate(self.countries):
            pos = np.flatnonzero(self.country_of == ci)
            if not len(pos):
                continue
            rows_c = self.row[pos]                               # ascending within a country
            idx_rows = self.index[country]['rows']
            for si in np.unique(self.season_of[pos]):
                s = self.seasons[si]
                ia, ib = ranges[si]
                a, b = s['row_start'], s['row_end']
                gr, gc = idx_rows['grid_row'][a:b], idx_rows['grid_col'][a:b]
                r0, c0 = gr.min() - R, gc.min() - R
                grid = np.full((gr.max() - r0 + R + 1, gc.max() - c0 + R + 1), -1, dtype=np.int64)
                grid[gr - r0, gc - c0] = np.arange(a, b)
                items = np.arange(ia, ib)
                rr = idx_rows['grid_row'][self.row[items]] - r0
                cc = idx_rows['grid_col'][self.row[items]] - c0
                nb = grid[rr[:, None] + _DR[None, :], cc[:, None] + _DC[None, :]]        # cache rows
                j = np.searchsorted(rows_c, np.maximum(nb, 0))
                j = np.minimum(j, len(rows_c) - 1)
                hit = (nb >= 0) & (rows_c[j] == nb)
                out[items] = np.where(hit, pos[j], -1)
        assert (out[:, CENTRE] == np.arange(len(self.row))).all()
        return out

    def get_batch(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        B = len(idx)
        W = self.win_items[idx].copy()
        if self.augment:
            if self._win_rng is None:
                import os
                self._win_rng = np.random.default_rng([os.getpid(), 7])
            W = W[np.arange(B)[:, None], ROT[self._win_rng.integers(0, 4, size=B)]]
        flat = W.ravel()
        present = flat >= 0
        uniq, inv = np.unique(flat[present], return_inverse=True)
        sub = super().get_batch(uniq)
        where = np.full(flat.shape, -1, dtype=np.int64)
        where[present] = inv
        centre = np.searchsorted(uniq, idx)
        out = {}
        for k, v in sub.items():
            if isinstance(v, dict):
                out[k] = {n: t[centre] for n, t in v.items()}
            else:
                out[k] = v[centre]
        safe = torch.from_numpy(np.maximum(where, 0))
        pres = torch.from_numpy(present).view(B, K)
        win_inputs, win_masks = {}, {}
        drop = None
        if self.augment and self.temporal_dropout > 0:
            T = out['time_valid'].shape[1]
            drop = torch.from_numpy(self._win_rng.random((B, T)) < self.temporal_dropout)
        rot_centre = torch.from_numpy(where.reshape(B, K)[:, CENTRE])
        for name, v in sub['inputs'].items():
            m = sub['masks'][name]
            if name in self.window_streams:
                gv = v[safe].view(B, K, *v.shape[1:])
                gm = m[safe].view(B, K, *m.shape[1:]) & pres.view((B, K) + (1,) * (v.dim() - 1))
            else:
                gv, gm = v[rot_centre].unsqueeze(1), m[rot_centre].unsqueeze(1)
            if drop is not None and gv.dim() == 4:                      # temporal (B, K or 1, T, C)
                gm = gm & ~drop[:, None, :, None]
            win_inputs[name] = torch.where(gm, gv, torch.full_like(gv, self.fill_value)).to(torch.float16)
            if self.return_masks:
                win_masks[name] = gm
        out['win_inputs'], out['win_valid'] = win_inputs, pres
        if self.return_masks:
            out['win_masks'] = win_masks
        return out
