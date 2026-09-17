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


class PyramidPooling(nn.ModuleList):
    """Pyramid pooling module (PPM) bottleneck.

    Adapted from the SpectralGPT OSCD dense decoder: applies adaptive max
    pooling at several scales, projects each pooled map with a 1x1 conv, and
    bilinearly upsamples every branch back to the input resolution. The
    resulting multi-scale context is concatenated by the caller to capture
    both local and global spatial context for dense yield-map decoding.
    """

    def __init__(self, pool_sizes, in_channels, out_channels):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )

    def forward(self, x):
        out_puts = []
        for ppm in self:
            ppm_out = nn.functional.interpolate(
                ppm(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            out_puts.append(ppm_out)
        return out_puts


class PyramidPoolingHead(nn.Module):
    """Pyramid-pooling bottleneck head.

    Runs the input through PyramidPooling, concatenates the multi-scale
    branches with the original feature map, and projects the result with a
    1x1 conv + GroupNorm + GELU block. This is the coarsest-level context
    encoder of the feature-pyramid dense decoder.
    """

    def __init__(self, in_channels, out_channels, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PyramidPooling(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes) * self.out_channels,
                      self.out_channels, kernel_size=1),
            nn.GroupNorm(16, self.out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out


class DenseYieldFPNHead(nn.Module):
    """Feature-pyramid + pyramid-pooling dense yield-map decoder.

    Adapted from the SpectralGPT OSCD FPNHEAD/PPMHEAD dense decoder to the
    MMST-ViT yield-map head. Consumes a multi-level feature pyramid (a list
    of feature maps, finest first) produced by the fusion neck / backbone
    and reconstructs a dense per-pixel yield map. The coarsest level passes
    through a pyramid-pooling bottleneck, then the levels are fused
    bottom-up with bilinear upsampling and 1x1-conv blocks, keeping multiple
    resolutions through the decode path so within-field spatial variability
    is preserved rather than pooled into a single field-average value.

    For continuous yield regression set num_classes=1 and use an L1/MSE loss
    (the default 'l1') instead of the source's 2-class LogSoftmax.
    """

    def __init__(self, channels=2048, out_channels=256, num_classes=1, loss='l1'):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.loss = loss

        self.ppm_head = PyramidPoolingHead(in_channels=channels, out_channels=out_channels)
        self.conv_fuse1 = nn.Sequential(
            nn.Conv2d(channels // 2, out_channels, 1), nn.GroupNorm(16, out_channels),
            nn.GELU(), nn.Dropout(0.5))
        self.conv_fuse1_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1), nn.GroupNorm(16, out_channels),
            nn.GELU(), nn.Dropout(0.5))
        self.conv_fuse2 = nn.Sequential(
            nn.Conv2d(channels // 4, out_channels, 1), nn.GroupNorm(16, out_channels),
            nn.GELU(), nn.Dropout(0.5))
        self.conv_fuse2_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1), nn.GroupNorm(16, out_channels),
            nn.GELU(), nn.Dropout(0.5))
        self.conv_fuse3 = nn.Sequential(
            nn.Conv2d(channels // 8, out_channels, 1), nn.GroupNorm(16, out_channels),
            nn.GELU(), nn.Dropout(0.5))
        self.conv_fuse3_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1), nn.GroupNorm(16, out_channels),
            nn.GELU(), nn.Dropout(0.5))
        self.fuse_all = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1), nn.GroupNorm(16, out_channels),
            nn.GELU(), nn.Dropout(0.5))
        self.conv_x1 = nn.Conv2d(out_channels, out_channels, 1)
        # regression head: 1-channel dense yield map (replaces the source's
        # 2-class LogSoftmax segmentation head).
        self.cls_seg = nn.Conv2d(out_channels, num_classes, 1)

    def compute_loss(self, logits, targets):
        if self.loss == 'l1':
            return nn.functional.l1_loss(logits, targets)
        return nn.functional.mse_loss(logits, targets)

    def forward(self, input_fpn, targets=None, mode='tensor'):
        """input_fpn: list of feature maps, finest first (4 levels)."""
        x1 = self.ppm_head(input_fpn[-1])
        x = nn.functional.interpolate(
            x1, size=(x1.size(2) * 2, x1.size(3) * 2), mode='bilinear', align_corners=True)
        x = self.conv_x1(x) + self.conv_fuse1(input_fpn[-2])
        x2 = self.conv_fuse1_(x)
        x = nn.functional.interpolate(
            x2, size=(x2.size(2) * 2, x2.size(3) * 2), mode='bilinear', align_corners=True)
        x = x + self.conv_fuse2(input_fpn[-3])
        x3 = self.conv_fuse2_(x)
        x = nn.functional.interpolate(
            x3, size=(x3.size(2) * 2, x3.size(3) * 2), mode='bilinear', align_corners=True)
        x = x + self.conv_fuse3(input_fpn[-4])
        x4 = self.conv_fuse3_(x)
        x1 = nn.functional.interpolate(x1, x4.size()[-2:], mode='bilinear', align_corners=True)
        x2 = nn.functional.interpolate(x2, x4.size()[-2:], mode='bilinear', align_corners=True)
        x3 = nn.functional.interpolate(x3, x4.size()[-2:], mode='bilinear', align_corners=True)
        x = self.fuse_all(torch.cat([x1, x2, x3, x4], 1))
        logits = self.cls_seg(x)
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
