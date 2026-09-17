# --------------------------------------------------------
# GroupChannelsVisionTransformer: one PatchEmbed per channel group plus a
# learned per-group channel embedding (sincos), so each input source is
# encoded by its own dedicated patch embedder.
#
# Adapted from danfenghong/IEEE_TPAMI_SpectralGPT (models_vit_group_channels.py)
# to the MMST-ViT plain-PyTorch layout. The source's "channel groups" are
# generalised to "modalities": instead of one shared patch embedder over a
# fixed channel stack, we instantiate a separate PatchEmbed per input source
# (DEM, soil, SAR, optical, ...) and add a per-group channel embedding so
# tokens from different sources are distinguishable in the shared transformer.
# Each group keeps its own native-resolution patch embedder, so layers are not
# forced onto a common raster resolution before feature extraction. The
# forward_features loop encodes each source independently and concatenates the
# token sets before the shared transformer, exactly as the source does.
# --------------------------------------------------------

import numpy as np
import torch
from torch import nn


class PatchEmbed(nn.Module):
    """Image-to-patch embedding (self-contained, mirrors models_pvt.PatchEmbed
    so this module does not pull in the timm dependency)."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class GroupChannelsVisionTransformer(nn.Module):
    """Vision transformer with one dedicated PatchEmbed per channel group.

    ``channel_groups`` is a list of channel-index tuples (or, for the
    modality-generalised use, a list of per-modality channel counts). Each
    group is embedded by its own PatchEmbed at its native resolution, then a
    per-group sincos channel embedding is added so tokens from different
    sources are distinguishable. All token sets are concatenated and passed to
    a shared transformer, which acts as the fusion backbone.

    Args:
        img_size: spatial size of each group's raster (assumed square).
        patch_size: patch size for every group's PatchEmbed.
        in_chans: total number of channels across all groups.
        embed_dim: shared latent dimension of the transformer.
        channel_groups: list of channel-index tuples partitioning ``in_chans``.
        depth, num_heads, mlp_ratio, qkv_bias, drop_rate: transformer config.
        channel_embed: dimension of the per-group channel embedding appended to
            the spatial position embedding (0 disables the channel embedding).
        global_pool: if True, mean-pool the non-cls tokens before the head;
            otherwise use the cls token.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192,
                 channel_groups=((0, 1, 2), (3, 4, 5)), depth=4, num_heads=3,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., channel_embed=64,
                 global_pool=True, num_classes=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.channel_groups = channel_groups
        self.global_pool = global_pool
        self.num_classes = num_classes

        # one dedicated PatchEmbed per channel group, each at its native
        # resolution (the source's core idea: no shared patch embedder over a
        # fixed channel stack).
        self.patch_embed = nn.ModuleList([
            PatchEmbed(img_size, patch_size, len(group), embed_dim)
            for group in channel_groups
        ])
        num_patches = self.patch_embed[0].num_patches
        num_groups = len(channel_groups)

        # spatial position embedding (shared across groups) + per-group channel
        # embedding so tokens from different sources are distinguishable.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim - channel_embed))
        pos_embed = self._get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(num_patches ** 0.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.channel_embed = nn.Parameter(torch.zeros(1, num_groups, channel_embed))
        chan_embed = get_1d_sincos_pos_embed_from_grid(
            self.channel_embed.shape[-1], np.arange(num_groups, dtype=np.float32))
        self.channel_embed.data.copy_(torch.from_numpy(chan_embed).float().unsqueeze(0))
        self.channel_cls_embed = nn.Parameter(torch.zeros(1, 1, channel_embed))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # shared fusion transformer over the concatenated per-group token sets.
        dpr = [drop_rate] * depth
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=drop_rate, activation='gelu', batch_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        if self.global_pool:
            self.fc_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    @staticmethod
    def _get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
        """2D sincos positional embedding (numpy, matching the source)."""
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)
        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim], dtype=np.float32), pos_embed], axis=0)
        return pos_embed

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """Encode each channel group independently and concatenate the token
        sets before the shared transformer (the source's forward loop)."""
        b, c, h, w = x.shape
        x_c_embed = []
        for i, group in enumerate(self.channel_groups):
            x_c = x[:, group, :, :]
            x_c_embed.append(self.patch_embed[i](x_c)[0])
        x = torch.stack(x_c_embed, dim=1)  # (B, G, L, D)
        _, G, L, D = x.shape

        channel_embed = self.channel_embed.unsqueeze(2)          # (1, G, 1, C)
        pos_embed = self.pos_embed[:, 1:, :].unsqueeze(1)        # (1, 1, L, D-C)
        channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)
        pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)
        pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # (1, G, L, D)
        x = x + pos_channel
        x = x.view(b, -1, D)  # (B, G*L, D)

        cls_pos_channel = torch.cat((self.pos_embed[:, :1, :], self.channel_cls_embed), dim=-1)
        cls_tokens = cls_pos_channel + self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """2D sincos positional embedding from a numpy grid (source helper)."""
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """1D sincos positional embedding from a numpy grid (source helper)."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)[:, None]
    out = np.einsum('nd,d->nd', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


if __name__ == "__main__":
    # 6-channel input split into two channel groups (e.g. optical RGB + SAR VV/VH).
    model = GroupChannelsVisionTransformer(
        img_size=64, patch_size=8, in_chans=6, embed_dim=192,
        channel_groups=((0, 1, 2), (3, 4, 5)), depth=2, num_heads=3,
        channel_embed=64, global_pool=True, num_classes=1)
    x = torch.randn(2, 6, 64, 64)
    out = model(x)
    print('grouped vit out', tuple(out.shape))
