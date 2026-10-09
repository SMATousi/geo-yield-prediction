# --------------------------------------------------------
# 5x5 window dataset for the YieldSAT paper's spatial models (spec/yieldsat-paper-models.md, PM-01).
#
# One item is still one grid cell; its batch additionally carries the cell's 5x5 neighbourhood of the same
# field season, gathered through the point dataset's own batch path (identical normalization, cutoff, masks
# and fill):
#   win_inputs[stream] - (B, 25, T, C) temporal / (B, 25, C) static, fill_value where invalid or outside
#   win_masks[stream]  - validity, same shapes
#   win_valid          - (B, 25) the neighbour exists (inside the field season and in this partition)
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

    def __init__(self, *args, augment=False, temporal_dropout=0.2, **kw):
        super().__init__(*args, **kw)
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
        for name, v in sub['inputs'].items():
            m = sub['masks'][name]
            gv = v[safe].view(B, K, *v.shape[1:])
            gm = m[safe].view(B, K, *m.shape[1:])
            shape = (B, K) + (1,) * (gv.dim() - 2)
            gm = gm & pres.view(shape)
            if drop is not None and gv.dim() == 4:                      # temporal (B, K, T, C)
                gm = gm & ~drop[:, None, :, None]
            gv = torch.where(gm, gv, torch.full_like(gv, self.fill_value))
            win_inputs[name], win_masks[name] = gv, gm
        out['win_inputs'], out['win_masks'], out['win_valid'] = win_inputs, win_masks, pres
        return out
