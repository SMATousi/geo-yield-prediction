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


def stitch_tiled_representation(rep_data, src_row_start, src_col_start,
                                src_row_end, src_col_end,
                                dst_row_start, dst_col_start,
                                dst_row_end, dst_col_end,
                                dst_height, dst_width, field_mask=None):
    """Stitch a per-patch dense representation into a full-field array.

    Adapted from the tessera ``stitch_tiled_representation`` routine: given
    the dense per-patch representation of a single tile, it bilinearly
    resamples that patch onto the target yield-map grid and writes it into
    the corresponding destination window of a full-coverage array. This is
    the final spatial-reconstruction step of the dense yield-map decoder,
    aligning per-patch decoder outputs to arbitrary field geometries without
    forcing every layer onto a common raster resolution.

    Args:
        rep_data: (src_height, src_width, num_bands) dense representation of
            the source patch (e.g. the decoder's per-pixel latent / logits).
        src_row_start, src_col_start, src_row_end, src_col_end: source patch
            window bounds (rows/cols) within the source raster.
        dst_row_start, dst_col_start, dst_row_end, dst_col_end: destination
            window bounds within the target yield-map grid.
        dst_height, dst_width: full target grid height and width.
        field_mask: optional (dst_height, dst_width) boolean field mask used
            to compute coverage statistics.

    Returns:
        (target_array, coverage) where target_array is the (dst_height,
        dst_width, num_bands) full-coverage array (zero-filled outside the
        written window) and coverage is the fraction of valid field cells
        covered by the stitched patch.
    """
    rep_data = np.asarray(rep_data)
    src_height, src_width = rep_data.shape[0], rep_data.shape[1]
    num_bands = rep_data.shape[2] if rep_data.ndim == 3 else 1
    if rep_data.ndim == 2:
        rep_data = rep_data[:, :, None]

    target_array = np.zeros((dst_height, dst_width, num_bands), dtype=rep_data.dtype)

    if src_height != dst_height or src_width != dst_width:
        resampled_data = np.zeros((dst_height, dst_width, num_bands), dtype=rep_data.dtype)
        row_ratio = src_height / dst_height
        col_ratio = src_width / dst_width
        for y in range(dst_height):
            for x in range(dst_width):
                src_y = src_row_start + y * row_ratio
                src_x = src_col_start + x * col_ratio
                y0 = int(src_y); y1 = min(y0 + 1, src_row_end - 1)
                x0 = int(src_x); x1 = min(x0 + 1, src_col_end - 1)
                wy1 = src_y - y0; wy0 = 1 - wy1
                wx1 = src_x - x0; wx0 = 1 - wx1
                if y0 >= src_row_start and x0 >= src_col_start:
                    for b in range(num_bands):
                        resampled_data[y, x, b] = (
                            wy0 * wx0 * rep_data[y0, x0, b] +
                            wy0 * wx1 * rep_data[y0, x1, b] +
                            wy1 * wx0 * rep_data[y1, x0, b] +
                            wy1 * wx1 * rep_data[y1, x1, b]
                        )
        target_array[dst_row_start:dst_row_end, dst_col_start:dst_col_end, :] = resampled_data
    else:
        target_array[dst_row_start:dst_row_end, dst_col_start:dst_col_end, :] = rep_data

    coverage = 1.0
    if field_mask is not None:
        field_mask = np.asarray(field_mask, dtype=bool)
        valid = field_mask[dst_row_start:dst_row_end, dst_col_start:dst_col_end]
        if valid.size > 0:
            coverage = float(valid.mean())
    return target_array, coverage


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

    def stitch(self, rep_data, src_row_start, src_col_start,
               src_row_end, src_col_end, dst_row_start, dst_col_start,
               dst_row_end, dst_col_end, dst_height, dst_width,
               field_mask=None):
        """Stitch a per-patch dense representation onto the yield-map grid.

        Bilinearly resamples the patch's dense representation to the target
        yield-map grid and writes it into the destination window, returning
        the full-coverage array and the fraction of valid field cells
        covered. This is the final spatial-reconstruction step that aligns
        decoder outputs to arbitrary field geometries.
        """
        return stitch_tiled_representation(
            rep_data, src_row_start, src_col_start, src_row_end, src_col_end,
            dst_row_start, dst_col_start, dst_row_end, dst_col_end,
            dst_height, dst_width, field_mask=field_mask)


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

    # per-patch dense representation stitched + bilinearly resampled to the
    # target yield-map grid, with coverage statistics against the field mask
    patch = np.random.rand(8, 8, 1).astype(np.float32)
    mask = np.ones((16, 16), dtype=bool)
    full, cov = head.stitch(patch, 0, 0, 8, 8, 0, 0, 16, 16, 16, 16, field_mask=mask)
    print(full.shape, cov)
