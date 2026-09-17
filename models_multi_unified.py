# --------------------------------------------------------
# Extensible multi-head foundation-model container.
# Adapted from flyakon/AgriFM (models/multi_unified_model.py) to the MMST-ViT
# plain-PyTorch layout (no mmengine registry). MultiUnifiedModel is a modular
# encoder + neck + heads container that decouples the shared backbone from
# task heads: the encoders produce per-modality embeddings, an optional neck
# fuses them into a shared latent, and a ModuleList of task heads consumes
# that same fused latent. New agricultural heads (yield map, field-average
# yield, stress detection, soil inference, ...) attach by appending to the
# heads config without modifying the modality encoders. A mode dispatch
# ('tensor' / 'predict' / 'loss') mirrors the source template.
# --------------------------------------------------------

import inspect

import torch
from torch import nn

from models_multimodal_encoder import MultiModalEncoder
from models_neck import MultiFusionNeck
from models_heads import DenseYieldFCNHead, DenseYieldFPNHead


class FieldAverageHead(nn.Module):
    """Scalar field-average yield head.

    Pools the fused latent field representation to a single field-level value.
    Demonstrates that a second task head can be attached alongside the dense
    yield-map head without touching the modality encoders.
    """

    def __init__(self, embed_dim, out_dim=1, loss='l1'):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.loss = loss
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, out_dim),
        )

    def compute_loss(self, logits, targets):
        if self.loss == 'l1':
            return nn.functional.l1_loss(logits, targets)
        return nn.functional.mse_loss(logits, targets)

    def forward(self, inputs, targets=None, mode='tensor'):
        """Dispatch: 'tensor' returns logits, 'loss' returns (logits, loss)."""
        logits = self.head(inputs)
        if mode == 'loss':
            loss = self.compute_loss(logits, targets)
            return logits, loss
        return logits


class MultiUnifiedModel(nn.Module):
    """Modular encoder + neck + heads container with mode dispatch.

    ``encoders_cfg`` builds the per-modality encoders (MultiModalEncoder),
    ``neck_cfg`` optionally builds the latent fusion neck (MultiFusionNeck),
    and ``heads_cfg`` is a list of task-head configs, each built into a
    ModuleList entry. All heads consume the same fused latent produced by the
    shared backbone, so new heads attach without redesigning the encoders.

    ``forward`` dispatches on ``mode``:
      - 'tensor' / 'predict': return a list of per-head logits;
      - 'loss': return a list of (logits, loss) tuples.
    """

    _HEAD_TYPES = {
        'dense_yield': DenseYieldFCNHead,
        'dense_yield_fpn': DenseYieldFPNHead,
        'field_average': FieldAverageHead,
    }

    def __init__(self, encoders_cfg, heads_cfg, neck_cfg=None, embed_dim=192):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoders = MultiModalEncoder(encoders_cfg, embed_dim=embed_dim)
        self.neck = MultiFusionNeck(**neck_cfg) if neck_cfg is not None else None
        self.heads = nn.ModuleList()
        for cfg in heads_cfg:
            self.heads.append(self._build_head(cfg))

    def _build_head(self, cfg):
        head_type = cfg.get('type')
        if head_type not in self._HEAD_TYPES:
            raise ValueError('Unknown task head type: {}'.format(head_type))
        cls = self._HEAD_TYPES[head_type]
        kwargs = {k: v for k, v in cfg.items() if k != 'type'}
        if 'embed_dim' in inspect.signature(cls.__init__).parameters:
            kwargs.setdefault('embed_dim', self.embed_dim)
        return cls(**kwargs)

    def forward(self, inputs, targets=None, mode='tensor'):
        """Dispatch: 'tensor'/'predict' return per-head logits, 'loss' returns
        per-head (logits, loss) tuples. Missing modalities are simply absent
        from the encoder output dict, so the backbone runs on arbitrary
        subsets of sources."""
        outputs = self.encoders(inputs)
        if self.neck is not None:
            outputs = self.neck(outputs)
        if mode == 'tensor' or mode == 'predict':
            return [head(outputs, mode=mode) for head in self.heads]
        return [head(outputs, targets=targets, mode='loss') for head in self.heads]


if __name__ == "__main__":
    # fused latent field representation shared by every task head:
    # B, embed_dim, H, W
    latent = torch.randn((2, 192, 16, 16))

    # two heads consuming the same fused latent: dense yield map + field average
    model = MultiUnifiedModel(
        encoders_cfg={},
        heads_cfg=[
            {'type': 'dense_yield', 'num_classes': 1, 'loss': 'l1'},
            {'type': 'field_average', 'out_dim': 1, 'loss': 'l1'},
        ],
        embed_dim=192,
    )

    # 'tensor' dispatch: one logits tensor per head
    logits_list = [head(latent, mode='tensor') for head in model.heads]
    for logits in logits_list:
        print('tensor', tuple(logits.shape))

    # 'loss' dispatch: (logits, loss) per head
    dense_targets = torch.randn((2, 1, 16, 16))
    avg_targets = torch.randn((2, 1))
    for head, tgt in zip(model.heads, (dense_targets, avg_targets)):
        logits, loss = head(latent, targets=tgt, mode='loss')
        print('loss', tuple(logits.shape), loss.item())
