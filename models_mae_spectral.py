# --------------------------------------------------------
# Masked autoencoder with temporal reconstruction for
# self-supervised pretraining of the shared backbone.
#
# Adapted from danfenghong/IEEE_TPAMI_SpectralGPT
# (models_mae_spectral.py) to the MMST-ViT plain-PyTorch layout.
#
# This realises the foundation-scale self-supervised pretraining
# objective (gap g7): beyond the SimCLR contrastive loss used in the
# PVT module, the shared backbone is pretrained on unlabelled fields by
# (1) masked spatial-token reconstruction -- masking a random subset of
# spatial patches and reconstructing them from the encoder latent -- and
# (2) temporal reconstruction -- sub-sampling a target set of temporal
# frames (pred_t_dim) and reconstructing them from the encoder latent,
# adding a spatial-aggregated temporal loss. The same masked-
# reconstruction machinery extends to masked-modality modeling by treating
# each modality's token set as a "channel group" to mask and reconstruct.
#
# The norm_pix_loss option and the masked-mean loss aggregation are
# directly portable from the source. The module reuses the repo's
# PatchEmbed (models_group_channels_vit) and 1D sincos positional-embed
# helper (models_latent_fusion) so no new dependency is introduced.
# --------------------------------------------------------

import numpy as np
import torch
from torch import nn

from models_group_channels_vit import PatchEmbed
from models_latent_fusion import get_1d_sincos_pos_embed


def random_masking(x, mask_ratio):
    """Mask a random subset of spatial tokens (source random_masking).

    Args:
        x: (N, L, D) token sequence.
        mask_ratio: fraction of tokens to mask.

    Returns:
        (x_masked, mask, ids_restore) where x_masked holds the visible
        tokens, mask is a 0/1 indicator (1 = masked) of shape (N, L), and
        ids_restore is the permutation that restores the original order.
    """
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(
        x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


class MAESpectral(nn.Module):
    """Masked autoencoder with spatial + temporal reconstruction.

    Consumes a multitemporal input ``(B, T, C, H, W)`` (e.g. a stack of
    weather / soil-moisture / SAR time series over unlabelled fields),
    patches it spatially, masks a random subset of spatial tokens, encodes
    the visible tokens with a transformer, and reconstructs:

      * the masked spatial tokens (loss1, masked-mean aggregated), and
      * a sub-sampled target set of temporal frames (pred_t_dim) from the
        encoder latent, adding a spatial-aggregated temporal loss (loss3).

    ``norm_pix_loss`` optionally normalises each patch by its own mean/std
    before computing the reconstruction target, exactly as the source.

    The same masked-reconstruction machinery can be reused for masked-
    modality modeling by treating each modality's token set as a channel
    group to mask and reconstruct.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192,
                 depth=4, num_heads=3, decoder_embed_dim=128, decoder_depth=2,
                 decoder_num_heads=3, mlp_ratio=4., mask_ratio=0.75,
                 pred_t_dim=3, norm_pix_loss=False, t_pred_patch_size=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.pred_t_dim = pred_t_dim
        self.norm_pix_loss = norm_pix_loss
        self.t_pred_patch_size = t_pred_patch_size

        # spatial patch embedding (reuses the repo's PatchEmbed)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # learned mask token substituted for masked spatial patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # positional embedding over the spatial grid (sincos, source helper)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_1d_sincos_pos_embed(
            embed_dim, np.arange(num_patches, dtype=np.float32))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # encoder transformer
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio), dropout=0.,
                activation='gelu', batch_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token_dec = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)
        dec_pos = get_1d_sincos_pos_embed(
            decoder_embed_dim, np.arange(num_patches, dtype=np.float32))
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(dec_pos).float().unsqueeze(0))
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_embed_dim, nhead=decoder_num_heads,
                dim_feedforward=int(decoder_embed_dim * mlp_ratio), dropout=0.,
                activation='gelu', batch_first=True)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # reconstruction head: decoder latent -> per-patch pixel values
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.mask_token_dec, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """Split a (N, C, H, W) image into (N, L, patch_dim) patches."""
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], imgs.shape[1], h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(imgs.shape[0], h * w, p ** 2 * imgs.shape[1])

    def forward_encoder(self, x):
        """Encode the visible spatial tokens of a single temporal frame.

        x: (B, C, H, W). Returns (x_masked, mask, ids_restore).
        """
        x, _ = self.patch_embed(x)          # (B, L, D)
        x = x + self.pos_embed
        x, mask, ids_restore = random_masking(x, self.mask_ratio)
        x = x + self.mask_token
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """Reconstruct all spatial tokens from the visible ones."""
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token_dec.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x_ = x_ + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x_ = blk(x_)
        x_ = self.decoder_norm(x_)
        pred = self.decoder_pred(x_)
        return pred

    def forward_loss(self, imgs, pred, mask):
        """Spatial + temporal reconstruction loss (adapted from the source).

        imgs: (B, T, C, H, W) multitemporal input.
        pred: (B, L, patch_dim) reconstructed patches for the first frame.
        mask: (B, L) spatial mask (1 = masked).

        The temporal branch sub-samples a target set of ``pred_t_dim``
        temporal frames, patches them, and reconstructs them from the
        encoder latent, adding a spatial-aggregated temporal loss (loss3).
        """
        B, T, C, H, W = imgs.shape
        p = self.patch_embed.patch_size[0]
        h = w = H // p
        L = h * w
        patch_dim = p * p * C

        # frame-0 patches as the masked spatial-token reconstruction target
        target0 = self.patchify(imgs[:, 0])          # (B, L, patch_dim)

        # sub-sample a target set of temporal frames for the temporal branch
        t_idx = torch.linspace(0, T - 1, self.pred_t_dim).long().to(imgs.device)
        _imgs = imgs[:, t_idx]                       # (B, pred_t_dim, C, H, W)
        target1 = torch.stack(
            [self.patchify(_imgs[:, i]) for i in range(self.pred_t_dim)], dim=1)
        # (B, pred_t_dim, L, patch_dim)

        # spatial-aggregated temporal target
        target_spatial = target1.sum(dim=1)          # (B, L, patch_dim)
        pred_spatial = pred                          # (B, L, patch_dim)

        if self.norm_pix_loss:
            mean = target0.mean(dim=-1, keepdim=True)
            var = target0.var(dim=-1, keepdim=True)
            target0 = (target0 - mean) / (var + 1.0e-6) ** 0.5

        # loss1: masked spatial-token reconstruction (masked-mean aggregated)
        loss1 = (pred - target0) ** 2
        loss1 = loss1.mean(dim=-1)
        mask = mask.view(loss1.shape)
        loss1 = (loss1 * mask).sum() / mask.sum()

        # loss3: spatial-aggregated temporal reconstruction
        loss3 = (pred_spatial - target_spatial) ** 2
        loss3 = loss3.mean(dim=-1)
        mask3 = torch.ones([B, L], device=loss3.device)
        loss3 = (loss3 * mask3).sum() / mask3.sum()

        loss = loss1 + loss3
        return loss

    def forward(self, imgs):
        """imgs: (B, T, C, H, W). Returns the combined reconstruction loss."""
        # encode the first temporal frame's visible spatial tokens
        x, mask, ids_restore = self.forward_encoder(imgs[:, 0])
        pred = self.forward_decoder(x, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss


if __name__ == "__main__":
    # multitemporal input: B, T, C, H, W (e.g. a 6-frame weather / moisture
    # / SAR time series over unlabelled fields).
    imgs = torch.randn(2, 6, 3, 64, 64)
    model = MAESpectral(
        img_size=64, patch_size=8, in_chans=3, embed_dim=192,
        depth=2, num_heads=3, decoder_embed_dim=128, decoder_depth=1,
        decoder_num_heads=4, mask_ratio=0.75, pred_t_dim=3,
        norm_pix_loss=False,
    )
    loss = model(imgs)
    print('mae spectral loss', loss.item())

    # norm_pix_loss variant
    model_norm = MAESpectral(
        img_size=64, patch_size=8, in_chans=3, embed_dim=192,
        depth=2, num_heads=3, decoder_embed_dim=128, decoder_depth=1,
        decoder_num_heads=4, mask_ratio=0.75, pred_t_dim=3,
        norm_pix_loss=True,
    )
    print('mae spectral norm_pix loss', model_norm(imgs).item())
