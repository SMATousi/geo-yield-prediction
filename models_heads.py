import torch
from torch import nn


class DenseYieldFCNHead(nn.Module):
    """Dense per-pixel FCN spatial decoder head.

    Maps a fused latent field representation (B, embed_dim, H, W) to a
    full-resolution per-pixel output map (B, num_classes, H, W). This is the
    dense yield-map head that replaces the classification-token + linear MLP
    pooling used for scalar county-level yield: it reconstructs within-field
    spatial variability instead of a single field-average value.

    For continuous yield regression set num_classes=1 and use an L1/MSE loss
    (the default 'l1' loss) rather than a cross-entropy classification loss.
    """

    def __init__(self, embed_dim, num_classes=1, loss='l1'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.loss = loss
        self.head = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dim // 2, self.num_classes, kernel_size=1, stride=1, padding=0),
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


if __name__ == "__main__":
    # fused latent field representation: B, embed_dim, H, W
    x = torch.randn((2, 512, 16, 16))
    targets = torch.randn((2, 1, 16, 16))

    head = DenseYieldFCNHead(embed_dim=512, num_classes=1, loss='l1')

    logits = head(x, mode='tensor')
    print(logits.shape)

    logits, loss = head(x, targets=targets, mode='loss')
    print(logits.shape, loss.item())
