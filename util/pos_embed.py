# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Per-modality, native-resolution 2D sincos positional embeddings.
# Adapted from cybergis/rs-embed (anysat) to provide an explicit
# resolution- and modality-aware spatial prior for the fusion backbone.
# --------------------------------------------------------

import torch


def get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid):
    """2D sincos positional embedding from a grid of coordinates.

    grid: (2, n, h, w) tensor of x/y coordinates (already scaled by resolution).
    Returns: (n, h, w, embed_dim) tensor.
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[0])  # (n, h, w, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[1])  # (n, h, w, D/2)

    emb = torch.cat([emb_h, emb_w], dim=-1)  # (n, h, w, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """1D sincos positional embedding from a grid of coordinates.

    pos: (n, h, w) tensor of coordinates.
    Returns: (n, h, w, embed_dim) tensor.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.unsqueeze(-1)  # (n, h, w, 1)
    out = torch.einsum('nhwd,d->nhwd', pos, omega)  # (n, h, w, D/2)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=-1)  # (n, h, w, D)
    return emb


def get_2d_sincos_pos_embed_with_resolution(
    embed_dim, grid_size, res, cls_token=False, modalities=None
):
    """Per-modality, native-resolution 2D sincos positional embeddings.

    grid_size: int of the grid height and width.
    res: dict mapping modality name -> native ground-sample distance (e.g. meters).
    Returns:
        pos_embed: dict of {modality: (n, grid_size*grid_size, embed_dim)} or
        {modality: (n, 1+grid_size*grid_size, embed_dim)} (w/ or w/o cls_token).
    """
    if modalities is None:
        modalities = list(res.keys())

    pos_embed_final = {}
    for modality in modalities:
        grid_size_aug = max(1, int(grid_size * 10 / res[modality]))
        if modality in ["planet"]:
            grid_size_aug = grid_size
        grid_h = torch.arange(grid_size_aug, dtype=torch.float32)
        grid_w = torch.arange(grid_size_aug, dtype=torch.float32)
        grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # h goes first
        grid = torch.stack(grid, dim=0)  # 2 x h x w

        grid = torch.einsum("chw,n->cnhw", grid, torch.tensor([res[modality]]))  # 2 x n x h x w
        _, n, h, w = grid.shape
        pos_embed = get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid)  # (n, h, w, D)
        pos_embed = pos_embed.reshape(n, h * w, embed_dim)
        if cls_token:
            pos_embed = torch.cat(
                [
                    torch.zeros(
                        [n, 1, embed_dim], dtype=torch.float32, device=pos_embed.device
                    ),
                    pos_embed,
                ],
                dim=1,
            )
        pos_embed_final[modality] = pos_embed
    return pos_embed_final
