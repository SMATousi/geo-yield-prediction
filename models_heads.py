import numpy as np
import torch
from torch import nn


def reassemble_to_grid(per_pixel_vecs, grid_indices, H, W):
    """Reassemble per-pixel latent vectors into an HxWxlatent array.

    Adapted from the tessera dense-output contract: instead of pooling a
    field representation into a single scalar, the model emits one latent
    vector per spatial pixel and scatters them back onto their grid
    positions, producing an ``(H, W, latent_dim)`` array that preserves
    within-field spatial variability. This is the output layout expected by
    the dense yield-map decoder's upsampling / query heads.

    Args:
        per_pixel_vecs: (N, latent_dim) float array of per-pixel latent
            vectors (one per valid grid cell).
        grid_indices: (N,) int array of flat grid indices (row-major) into
            the HxW output grid.
        H, W: output grid height and width.

    Returns:
        np.ndarray of shape (H, W, latent_dim); cells without a vector stay
        zero-filled, mirroring the tessera convention.
    """
    per_pixel_vecs = np.asarray(per_pixel_vecs, dtype=np.float32)
    grid_indices = np.asarray(grid_indices, dtype=np.int64)
    latent_dim = per_pixel_vecs.shape[1]
    out_array = np.zeros((H * W, latent_dim), dtype=np.float32)
    out_array[grid_indices] = per_pixel_vecs
    return out_array.reshape(H, W, latent_dim)


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

    def reassemble(self, per_pixel_vecs, grid_indices, H, W):
        """Emit the dense output as an HxWxlatent array (tessera contract).

        Scatters per-pixel latent vectors back onto their grid positions so
        downstream heads reconstruct a full yield map with within-field
        variability rather than a single field-average value.
        """
        return reassemble_to_grid(per_pixel_vecs, grid_indices, H, W)


if __name__ == "__main__":
    # fused latent field representation: B, embed_dim, H, W
    x = torch.randn((2, 512, 16, 16))
    targets = torch.randn((2, 1, 16, 16))

    head = DenseYieldFCNHead(embed_dim=512, num_classes=1, loss='l1')

    logits = head(x, mode='tensor')
    print(logits.shape)

    logits, loss = head(x, targets=targets, mode='loss')
    print(logits.shape, loss.item())

    # per-pixel-token-to-2D-grid reassembly output contract
    vecs = torch.randn(256, 512).detach().numpy()
    gidx = np.arange(256, dtype=np.int64)
    grid = head.reassemble(vecs, gidx, 16, 16)
    print(grid.shape)
