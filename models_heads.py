import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Extensible multi-head foundation-model registry.
# Adapted from opengeos/geoai (geoai/foundation_models.py) to the MMST-ViT
# multi-head interface. Instead of a hardcoded dict of head types, heads are
# registered under validated category/modality vocabularies with structured
# metadata (tasks, backbone, description). New agricultural heads (crop-stress
# detection, management-zone segmentation, soil-property inference, ...) are
# attached by registering them here and referencing them by name in the
# heads_cfg, without modifying any modality encoder.
# ---------------------------------------------------------------------------

_VALID_HEAD_CATEGORIES = frozenset(
    {"dense", "scalar", "probabilistic", "segmentation", "classification"}
)

_VALID_HEAD_MODALITIES = frozenset(
    {"yield", "multimodal", "optical", "sar", "soil", "timeseries"}
)

# Registry keyed by head name -> structured metadata (mirrors geoai's
# FOUNDATION_MODELS dict). ``cls`` is the callable used to build the head.
HEAD_REGISTRY: dict = {}


def register_head(name, category, modality, tasks, backbone, description):
    """Register a task head under validated category/modality vocabularies.

    Returns a decorator that stores the head class in :data:`HEAD_REGISTRY`
    keyed by ``name``, validating that ``category`` and ``modality`` are drawn
    from the controlled vocabularies. This is the extension point for new
    agricultural heads: register a head once and attach it by name in a
    ``heads_cfg`` without touching the modality encoders.
    """
    if category not in _VALID_HEAD_CATEGORIES:
        raise ValueError(
            "Unknown head category {!r}; valid: {}".format(
                category, sorted(_VALID_HEAD_CATEGORIES)))
    if modality not in _VALID_HEAD_MODALITIES:
        raise ValueError(
            "Unknown head modality {!r}; valid: {}".format(
                modality, sorted(_VALID_HEAD_MODALITIES)))

    def _decorator(cls):
        HEAD_REGISTRY[name] = {
            "name": name,
            "category": category,
            "modality": modality,
            "tasks": list(tasks),
            "backbone": backbone,
            "description": description,
            "cls": cls,
        }
        return cls

    return _decorator


def list_heads():
    """Return the names of all registered task heads (sorted)."""
    return sorted(HEAD_REGISTRY)


def get_head_info(name):
    """Return the structured metadata for a registered head by name.

    Raises ``KeyError`` if the head is not registered.
    """
    if name not in HEAD_REGISTRY:
        raise KeyError(
            "Unknown task head {!r}; registered: {}".format(
                name, list_heads()))
    return HEAD_REGISTRY[name]


def build_head(name, **kwargs):
    """Instantiate a registered head by name from its stored class."""
    info = get_head_info(name)
    return info["cls"](**kwargs)


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


@register_head(
    "dense_yield",
    category="dense",
    modality="yield",
    tasks=["yield-map-regression"],
    backbone="fused-latent",
    description=(
        "Dense per-pixel FCN spatial decoder that reconstructs a within-field "
        "yield map from the fused latent field representation."
    ),
)
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

    def __init__(self, embed_dim, num_classes=1, loss='l1', log_target=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.loss = loss
        self.log_target = log_target
        self.head = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dim // 2, self.num_classes, kernel_size=1, stride=1, padding=0),
        )

    def compute_loss(self, logits, targets):
        # Log-transform of the yield target (adapted from the CropNet USDA
        # loader's ``torch.log`` preprocessing) stabilizes training on skewed
        # yield distributions: the head regresses log-yield and the loss is
        # computed in log space, mirroring the tabular loader's target transform.
        if self.log_target:
            targets = torch.log(targets.clamp_min(1e-6))
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


@register_head(
    "dense_yield_fpn",
    category="dense",
    modality="yield",
    tasks=["yield-map-regression"],
    backbone="feature-pyramid",
    description=(
        "Feature-pyramid + pyramid-pooling dense yield-map decoder that "
        "preserves multiple resolutions through the decode path."
    ),
)
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

    def __init__(self, channels=2048, out_channels=256, num_classes=1, loss='l1', log_target=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.loss = loss
        self.log_target = log_target

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
        if self.log_target:
            targets = torch.log(targets.clamp_min(1e-6))
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


@register_head(
    "dense_yield_dpt",
    category="dense",
    modality="yield",
    tasks=["yield-map-regression"],
    backbone="dpt",
    description=(
        "Dense Prediction Transformer spatial decoder that fuses multi-scale "
        "feature maps and interpolates to an arbitrary yield-map grid."
    ),
)
class DenseYieldDPTHead(nn.Module):
    """Dense Prediction Transformer (DPT) spatial decoder head.

    Adapted from the geoai ``DPTSegmentationHead`` to the MMST-ViT dense
    yield-map head. Consumes the four multi-scale feature maps produced by
    the fusion transformer / modality encoders (finest first), projects each
    to a common channel dimension, progressively fuses them with bilinear
    upsampling, and produces per-pixel logits at an arbitrary target grid.
    Because it interpolates to any ``(H, W)`` target size, it can reconstruct
    yield directly on the harvest yield-monitor grid without redesigning the
    encoders, preserving within-field spatial variability.

    For continuous yield regression set num_classes=1 and use an L1/MSE loss
    (the default 'l1') instead of the source's per-pixel class logits.
    """

    def __init__(self, embed_dim, num_classes=1, features=256, loss='l1', log_target=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.features = features
        self.loss = loss
        self.log_target = log_target
        # Project each backbone layer to *features* channels.
        self.projects = nn.ModuleList(
            [nn.Conv2d(embed_dim, features, kernel_size=1) for _ in range(4)]
        )
        # Refine each projected feature map.
        self.refine = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=3, padding=1),
                    nn.BatchNorm2d(features),
                    nn.GELU(),
                )
                for _ in range(4)
            ]
        )
        # Fuse all four scales.
        self.fuse = nn.Sequential(
            nn.Conv2d(features * 4, features, kernel_size=1),
            nn.BatchNorm2d(features),
            nn.GELU(),
        )
        self.head = nn.Conv2d(features, num_classes, kernel_size=1)

    def compute_loss(self, logits, targets):
        if self.log_target:
            targets = torch.log(targets.clamp_min(1e-6))
        if self.loss == 'l1':
            return nn.functional.l1_loss(logits, targets)
        return nn.functional.mse_loss(logits, targets)

    def forward(self, multi_scale_features, target_size=None, targets=None, mode='tensor'):
        """multi_scale_features: list of 4 feature maps, finest first.

        ``target_size`` is the (H, W) output grid; when None it defaults to
        the spatial size of the largest (first) feature map.
        """
        refined = []
        for feat, proj, ref in zip(multi_scale_features, self.projects, self.refine):
            refined.append(ref(proj(feat)))

        # Upsample all to the spatial size of the largest (first) feature map.
        target_h, target_w = refined[0].shape[2], refined[0].shape[3]
        upsampled = [refined[0]]
        for r in refined[1:]:
            upsampled.append(
                F.interpolate(
                    r, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
            )

        fused = self.fuse(torch.cat(upsampled, dim=1))
        logits = self.head(fused)
        # Final interpolation to the target yield-map resolution.
        if target_size is not None:
            logits = F.interpolate(
                logits, size=target_size, mode="bilinear", align_corners=False
            )
        if mode == 'loss':
            loss = self.compute_loss(logits, targets)
            return logits, loss
        return logits


def mdn_loss(pi, mu, sigma, target):
    """Negative log-likelihood of a Gaussian mixture density network.

    Adapted from the CropWise MDN head: expands the target to match the
    mixture components, computes the per-component log-likelihood under each
    Gaussian, weights it by the mixture log-probability, and returns the
    mean negative log-likelihood over the batch. This is the training
    objective for the probabilistic yield head, giving calibrated per-pixel
    uncertainty (e.g. ~95% coverage) in addition to the point estimate.
    """
    target = target.unsqueeze(1).expand_as(mu)
    normal = torch.distributions.Normal(mu, sigma)
    log_prob = normal.log_prob(target)
    weighted_log_prob = log_prob + torch.log(pi + 1e-10)
    log_sum = torch.logsumexp(weighted_log_prob, dim=1)
    return -log_sum.mean()


def sample_from_mixture(pi, mu, sigma, n_samples=1):
    """Draw samples from a Gaussian mixture density network.

    Adapted from the CropWise MDN head: for each batch element, samples a
    mixture component from the categorical ``pi`` distribution and then draws
    a Gaussian sample from that component's ``mu``/``sigma``. Returns a
    ``(batch, n_samples)`` tensor (or a ``(batch,)`` tensor when
    ``n_samples == 1``) for sampling yield distributions at inference.
    """
    batch_size = pi.size(0)
    samples = []
    for _ in range(n_samples):
        component = torch.multinomial(pi, 1).squeeze()
        indices = torch.arange(batch_size, device=pi.device)
        mu_sample = mu[indices, component]
        sigma_sample = sigma[indices, component]
        sample = torch.normal(mu_sample, sigma_sample)
        samples.append(sample)
    return torch.stack(samples, dim=1) if n_samples > 1 else samples[0]


@register_head(
    "mdn_yield",
    category="probabilistic",
    modality="yield",
    tasks=["yield-map-regression", "uncertainty-estimation"],
    backbone="mixture-density-network",
    description=(
        "Probabilistic Mixture Density Network yield head emitting per-location "
        "pi/mu/sigma for calibrated per-pixel uncertainty."
    ),
)
class MDNYieldHead(nn.Module):
    """Probabilistic Mixture Density Network yield head.

    Adapted from the CropWise MDN head to the MMST-ViT multi-head interface.
    This is a drop-in probabilistic head that attaches alongside the dense
    yield-map decoder: it feeds the backbone's pooled field representation
    (or per-pixel decoder features) through a small MLP feature extractor and
    emits per-location mixture parameters ``pi``/``mu``/``sigma``. Training
    uses :func:`mdn_loss` (negative log-likelihood) and inference can draw
    yield samples via :func:`sample_from_mixture`, yielding calibrated
    per-pixel uncertainty without modifying any modality encoder. New
    agricultural heads (crop-stress, management-zone segmentation, soil
    property estimation, ...) can be attached the same way, satisfying the
    extensible multi-head foundation-model interface.
    """

    def __init__(self, input_dim, hidden_dims=(256, 128, 64), n_gaussians=5):
        super().__init__()
        self.input_dim = input_dim
        self.n_gaussians = n_gaussians
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.pi_net = nn.Linear(prev_dim, n_gaussians)
        self.mu_net = nn.Linear(prev_dim, n_gaussians)
        self.sigma_net = nn.Linear(prev_dim, n_gaussians)

    def forward(self, x):
        features = self.feature_extractor(x)
        pi = F.softmax(self.pi_net(features), dim=-1)
        mu = self.mu_net(features)
        sigma = F.softplus(self.sigma_net(features)) + 1e-6
        return pi, mu, sigma

    def compute_loss(self, pi, mu, sigma, targets):
        return mdn_loss(pi, mu, sigma, targets)

    def sample(self, pi, mu, sigma, n_samples=1):
        return sample_from_mixture(pi, mu, sigma, n_samples=n_samples)


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

    # DPT dense decoder: four multi-scale feature maps -> per-pixel yield map
    feats = [torch.randn((2, 512, 16, 16)),
             torch.randn((2, 512, 8, 8)),
             torch.randn((2, 512, 4, 4)),
             torch.randn((2, 512, 2, 2))]
    dpt = DenseYieldDPTHead(embed_dim=512, num_classes=1, loss='l1')
    logits = dpt(feats, target_size=(32, 32), mode='tensor')
    print(logits.shape)
    logits, loss = dpt(feats, target_size=(32, 32), targets=torch.randn((2, 1, 32, 32)), mode='loss')
    print(logits.shape, loss.item())

    # probabilistic MDN head: pooled field representation -> pi/mu/sigma
    pooled = torch.randn((2, 512))
    targets = torch.randn((2,))
    mdn = MDNYieldHead(input_dim=512, n_gaussians=5)
    pi, mu, sigma = mdn(pooled)
    print(pi.shape, mu.shape, sigma.shape)
    loss = mdn.compute_loss(pi, mu, sigma, targets)
    print(loss.item())
    samples = mdn.sample(pi, mu, sigma, n_samples=3)
    print(samples.shape)
