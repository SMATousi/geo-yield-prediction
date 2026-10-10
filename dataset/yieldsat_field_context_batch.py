# --------------------------------------------------------
# Field-context batches for the field encoder (spec/yieldsat-field-encoder.md, FE-01).
#
# A batch is the usual point batch of target items plus, for each distinct field season among the targets,
# K context items of the same field season, produced by the point dataset's own batch path (identical inputs):
#   ctx_inputs / ctx_masks / ctx_available   per stream, (G*K, ...)
#   ctx_time_features / ctx_time_valid / ctx_crop / ctx_grid_row / ctx_grid_col
#   ctx_group (G*K,) and tgt_group (B,)      group index in [0, G) of each context / target item
# Training draws a fresh random context; evaluation uses a fixed sample per field season (seeded by the season),
# so every pixel of a field sees the same context. Context items carry inputs only (no targets).
# --------------------------------------------------------

import numpy as np
import torch

from dataset.yieldsat_dataset import YieldSATPointDataset

CTX_KEYS = ('time_features', 'time_valid', 'crop', 'grid_row', 'grid_col')


class FieldContextDataset(YieldSATPointDataset):

    def __init__(self, *args, context_k=64, train=False, context_seed=0, **kw):
        super().__init__(*args, **kw)
        self.context_k = int(context_k)
        self.train_context = train
        self.context_seed = context_seed
        self._ctx_rng = None
        self._range_of = {int(self.season_of[a]): (a, b) for a, b in self.season_ranges}
        self._fixed = {}

    def context_items(self, season):
        a, b = self._range_of[season]
        n = b - a
        if self.train_context:
            if self._ctx_rng is None:
                import os
                self._ctx_rng = np.random.default_rng([os.getpid(), self.context_seed, 11])
            rng = self._ctx_rng
        else:
            if season in self._fixed:
                return self._fixed[season]
            rng = np.random.default_rng([self.context_seed, season])
        pick = a + rng.choice(n, size=self.context_k, replace=n < self.context_k)
        if not self.train_context:
            self._fixed[season] = pick
        return pick

    def get_batch(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        out = super().get_batch(idx)
        seasons, tgt_group = np.unique(self.season_of[idx], return_inverse=True)
        ctx = np.concatenate([self.context_items(int(s)) for s in seasons])
        sub = super().get_batch(ctx)
        out['ctx_inputs'], out['ctx_masks'], out['ctx_available'] = sub['inputs'], sub['masks'], sub['available']
        for k in CTX_KEYS:
            out['ctx_' + k] = sub[k]
        out['ctx_group'] = torch.from_numpy(np.repeat(np.arange(len(seasons)), self.context_k))
        out['tgt_group'] = torch.from_numpy(tgt_group.astype(np.int64))
        return out
