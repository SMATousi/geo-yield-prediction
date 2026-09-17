# --------------------------------------------------------
# Masked channel-group autoencoder for masked-modality modeling.
#
# Adapted from danfenghong/IEEE_TPAMI_SpectralGPT
# (models_mae_group_channels.py) to the MMST-ViT plain-PyTorch layout.
#
# This realises the masked-modality / cross-modal reconstruction primitive
# of the foundation-scale self-supervised pretraining objective (gap g7).
# The source's "channel groups" are reinterpreted as "modalities": each
# modality is embedded by its own dedicated PatchEmbed (at its native
# resolution), and masking is applied either per-spatial-location across all
# groups (spatial_mask=True) or independently per group (spatial_mask=False),
# with a per-group mask returned for the loss. Masking an entire modality's
# token set (e.g. dropping all SAR tokens) and training the shared transformer
# to reconstruct it from the remaining modalities is exactly the masked-
# modality / cross-modal reconstruction objective the problem asks for. The
# same encoder doubles as the missing-modality robustness mechanism by masking
# modalities at train time.
#
# The module reuses the repo's PatchEmbed (models_group_channels_vit), the 1D
# sincos positional-embed helper (models_latent_fusion) and the random_masking
# primitive (models_mae_spectral), so no new dependency is introduced.
# --------------------------------------------------------

import numpy as np
import torch
from torch import nn

from models_group_channels_vit import PatchEmbed
from models_latent_fusion import get_1d_sincos_pos_embed
from models_mae_spectral import random_masking


class MAEGroupChannels(nn.Module):
    """Masked channel-group autoencoder (masked-modality modeling).

    Consumes a multimodal raster ``(B, C, H, W)`` whose channels are partitioned
    into ``channel_groups`` (one per modality). Each group is embedded by its
    own PatchEmbed, given a per-group channel embedding plus a shared spatial
    position embedding, and the token sets are concatenated before a shared
    transformer encoder. A random subset of tokens is masked and the decoder
    reconstructs the masked tokens from the visible ones.

    Two masking modes (the source's two modes):
      * ``spatial_mask=True``: mask a random subset of spatial locations
        uniformly across all groups (every modality loses the same patches).
      * ``spatial_mask=False``: mask a random subset of tokens independently
        per group, so different modalities lose different patches (this is the
        masked-modality modeling mode -- an entire modality's token set can be
        dropped and reconstructed from the others).

    ``forward_encoder`` returns a per-group mask ``(B, G, L)`` (1 = masked) so
    the loss can be aggregated per modality, exactly as the source does.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192,
                 channel_groups=((0, 1, 2), (3, 4, 5)), depth=4, num_heads=3,
                 decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=3,
                 mlp_ratio=4., mask_ratio=0.75, spatial_mask=False,
                 channel_embed=64, norm_pix_loss=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.spatial_mask = spatial_mask
        self.channel_groups = channel_groups
        self.norm_pix_loss = norm_pix_loss

        # one dedicated PatchEmbed per channel group (modality), each at its
        # native resolution (the source's core idea).
        self.patch_embed = nn.ModuleList([
            PatchEmbed(img_size, patch_size, len(group), embed_dim)
            for group in channel_groups
        ])
        num_patches = self.patch_embed[0].num_patches
        num_groups = len(channel_groups)

        # shared spatial position embedding + per-group channel embedding so
        # tokens from different modalities are distinguishable.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim - channel_embed))
        pos_embed = get_1d_sincos_pos_embed(
            embed_dim - channel_embed, np.arange(num_patches, dtype=np.float32))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.channel_embed = nn.Parameter(torch.zeros(1, num_groups, channel_embed))
        chan_embed = get_1d_sincos_pos_embed(
            channel_embed, np.arange(num_groups, dtype=np.float32))
        self.channel_embed.data.copy_(torch.from_numpy(chan_embed).float().unsqueeze(0))

        # learned mask token substituted for masked tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # encoder transformer over the concatenated per-group token sets
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
        dec_len = num_patches if self.spatial_mask else num_patches * num_groups
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, dec_len, decoder_embed_dim), requires_grad=False)
        dec_pos = get_1d_sincos_pos_embed(
            decoder_embed_dim, np.arange(dec_len, dtype=np.float32))
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

        # reconstruction head: decoder latent -> per-patch pixel values. In
        # spatial mode the decoder reconstructs the full channel stack over the
        # spatial grid; in per-group mode it reconstructs each group's channels.
        if self.spatial_mask:
            self.decoder_pred = nn.Linear(
                decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
        else:
            self.decoder_pred = nn.Linear(
                decoder_embed_dim, patch_size ** 2 * len(channel_groups[0]), bias=True)

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
        p = self.patch_embed[0].patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], imgs.shape[1], h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(imgs.shape[0], h * w, p ** 2 * imgs.shape[1])

    def forward_encoder(self, x, mask_ratio):
        """Encode the visible tokens of a multimodal raster.

        x: (B, C, H, W). Returns (x, mask, ids_restore) where x holds the
        visible tokens, mask is a per-group 0/1 indicator of shape (B, G, L)
        (1 = masked), and ids_restore restores the original token order.
        """
        b, c, h, w = x.shape
        x_c_embed = []
        for i, group in enumerate(self.channel_groups):
            x_c = x[:, group, :, :]
            x_c_embed.append(self.patch_embed[i](x_c)[0])
        x = torch.stack(x_c_embed, dim=1)          # (B, G, L, D)
        _, G, L, D = x.shape

        channel_embed = self.channel_embed.unsqueeze(2)          # (1, G, 1, C)
        pos_embed = self.pos_embed.unsqueeze(1)                  # (1, 1, L, D-C)
        channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)
        pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)
        pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # (1, G, L, D)
        x = x + pos_channel

        if self.spatial_mask:
            # mask a random subset of spatial locations uniformly across all
            # groups: every modality loses the same patches.
            x = x.permute(0, 2, 1, 3).reshape(b, L, -1)
            x, mask, ids_restore = random_masking(x, mask_ratio)
            x = x.view(b, x.shape[1], G, D).permute(0, 2, 1, 3).reshape(b, -1, D)
            mask = mask.repeat(1, G)
            mask = mask.view(b, G, L)
        else:
            # mask a random subset of tokens independently per group: different
            # modalities lose different patches (masked-modality modeling).
            x, mask, ids_restore = random_masking(x.view(b, -1, D), mask_ratio)
            mask = mask.view(b, G, L)

        x = x + self.mask_token
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """Reconstruct all tokens from the visible ones."""
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
        """Masked-modality reconstruction loss (masked-mean aggregated).

        imgs: (B, C, H, W) multimodal raster.
        pred: (B, G*L, patch_dim) reconstructed patches.
        mask: (B, G, L) per-group mask (1 = masked).
        """
        B, C, H, W = imgs.shape
        p = self.patch_embed[0].patch_size[0]
        h = w = H // p
        L = h * w
        G = len(self.channel_groups)
        patch_dim = p * p * len(self.channel_groups[0])

        if self.spatial_mask:
            # spatial mode: reconstruct the full channel stack over the grid.
            target = self.patchify(imgs)          # (B, L, p^2*C)
            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.0e-6) ** 0.5
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)              # (B, L)
            mask = mask[:, 0]                     # all groups share the mask
            loss = (loss * mask).sum() / mask.sum()
            return loss

        # per-group mode: reconstruct each group's channels independently.
        target = torch.stack(
            [self.patchify(imgs[:, group]) for group in self.channel_groups],
            dim=1)                                # (B, G, L, patch_dim)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        pred = pred.view(B, G, L, patch_dim)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)                  # (B, G, L)
        mask = mask.view(loss.shape)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs):
        """imgs: (B, C, H, W). Returns the masked-modality reconstruction loss."""
        x, mask, ids_restore = self.forward_encoder(imgs, self.mask_ratio)
        pred = self.forward_decoder(x, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss


if __name__ == "__main__":
    # 6-channel multimodal raster split into two channel groups (e.g. optical
    # RGB + SAR VV/VH), masked-modality modeling mode (spatial_mask=False).
    imgs = torch.randn(2, 6, 64, 64)
    model = MAEGroupChannels(
        img_size=64, patch_size=8, in_chans=6, embed_dim=192,
        channel_groups=((0, 1, 2), (3, 4, 5)), depth=2, num_heads=3,
        decoder_embed_dim=128, decoder_depth=1, decoder_num_heads=4,
        mask_ratio=0.75, spatial_mask=False, channel_embed=64,
        norm_pix_loss=False,
    )
    loss = model(imgs)
    print('mae group channels loss', loss.item())

    # per-spatial-location masking mode (spatial_mask=True)
    model_spatial = MAEGroupChannels(
        img_size=64, patch_size=8, in_chans=6, embed_dim=192,
        channel_groups=((0, 1, 2), (3, 4, 5)), depth=2, num_heads=3,
        decoder_embed_dim=128, decoder_depth=1, decoder_num_heads=4,
        mask_ratio=0.75, spatial_mask=True, channel_embed=64,
        norm_pix_loss=True,
    )
    print('mae group channels spatial loss', model_spatial(imgs).item())
