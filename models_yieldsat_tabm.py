"""TabM point regressor for YieldSAT (TM-01, spec/yieldsat-tabm.md).

Value columns go through periodic numeric embeddings (shared by all members,
no fold-specific bins); validity-mask columns are passed through as 0/1
features. The backbone is ``tabm.TabM`` (Gorishniy et al., ICLR 2025):
k parameter-efficient MLP members (BatchEnsemble, ``arch_type='tabm'``) or
shared-MLP with per-member adapters (``'tabm-mini'``). k = 1 with a plain MLP
gives the M1 baseline. Training loss is the mean of the members' MSE;
prediction is the mean over members.
"""
import torch
from torch import nn


class TabMRegressor(nn.Module):
    def __init__(self, n_values, n_masks, k=32, n_blocks=3, d_block=512, dropout=0.1, d_embedding=8,
                 arch_type='tabm', embeddings='periodic'):
        super().__init__()
        import tabm
        from rtdl_num_embeddings import PeriodicEmbeddings
        self.n_values, self.n_masks, self.k = n_values, n_masks, k
        if embeddings == 'periodic':
            self.emb = PeriodicEmbeddings(n_values, d_embedding=d_embedding, lite=True)
            d_in_values = n_values * d_embedding
        else:
            self.emb = None
            d_in_values = n_values
        if arch_type == 'mlp':
            self.backbone = tabm.TabM.make(n_num_features=d_in_values + n_masks, d_out=1, k=1, arch_type='tabm-packed',
                                           n_blocks=n_blocks, d_block=d_block, dropout=dropout)
        else:
            self.backbone = tabm.TabM.make(n_num_features=d_in_values + n_masks, d_out=1, k=k, arch_type=arch_type,
                                           n_blocks=n_blocks, d_block=d_block, dropout=dropout)
        self.config = dict(n_values=n_values, n_masks=n_masks, k=k if arch_type != 'mlp' else 1, n_blocks=n_blocks,
                           d_block=d_block, dropout=dropout, d_embedding=d_embedding, arch_type=arch_type,
                           embeddings=embeddings)

    def members(self, values, masks):
        """(B, k) member predictions (standardized target units)."""
        x = values if self.emb is None else self.emb(values).flatten(1)
        x = torch.cat([x, masks.to(x.dtype)], 1)
        return self.backbone(x).squeeze(-1)

    def forward(self, values, masks):
        return self.members(values, masks).mean(1)

    def loss(self, values, masks, target):
        p = self.members(values, masks)
        return (p - target[:, None]).square().mean()
