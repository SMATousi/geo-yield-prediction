# --------------------------------------------------------
# Multiscale latent geospatial-temporal fusion neck.
# Adapted from flyakon/AgriFM (models/neck.py) to the MMST-ViT
# plain-PyTorch layout (no mmengine registry). MultiFusionNeck is the
# unified latent fusion backbone: it consumes per-modality multi-level
# feature lists (each encoder's native-resolution multiscale tokens),
# resamples them to a common latent grid, and progressively fuses them
# through a feature pyramid of conv blocks with bilinear upsampling to a
# target output size. This realises the native-resolution-then-fuse-in-
# latent-space pattern: high-resolution terrain/aerial tokens coexist with
# coarse soil/climate/satellite tokens without resampling raw layers to a
# single raster before feature extraction.
# --------------------------------------------------------

import torch
from torch import nn


class MultiFusionNeck(nn.Module):
    """Multiscale feature-pyramid fusion neck.

    Consumes a dict of per-modality encoder outputs. Each modality exposes
    ``encoder_features`` (a single fused latent map) and ``features_list``
    (a list of multi-level feature maps, finest first). The neck:

      1. resamples every modality's ``encoder_features`` to a common latent
         grid (``feature_size``) and concatenates them;
      2. walks up a feature pyramid, at each level bilinearly upsampling the
         running latent and fusing it with the corresponding level of each
         modality's ``features_list`` through a conv block;
      3. projects the fused latent to ``out_size`` for a dense decoder head.

    ``in_fusion_key_list`` is a list of dicts, one per pyramid level, mapping
    modality name -> channel count of that level's feature map. This lets the
    neck fuse modalities whose native resolutions differ substantially.
    """

    def __init__(self, embed_dim, in_feature_key=('S2',),
                 feature_size=(16, 16), out_size=(256, 256),
                 in_fusion_key_list=({'S2': 512, 'HLS': 512},
                                     {'S2': 256},
                                     {'S2': 128},
                                     )):
        super(MultiFusionNeck, self).__init__()
        self.embed_dim = embed_dim
        self.in_feature_key = in_feature_key
        self.feature_size = feature_size
        self.out_size = out_size
        self.in_fusion_key_list = in_fusion_key_list

        self.fusion_list = nn.ModuleList()
        if len(in_feature_key) == 1:
            self.in_conv = nn.Identity()
        else:
            self.in_conv = nn.Sequential(
                nn.Conv2d(len(in_feature_key) * self.embed_dim, self.embed_dim, 3, 1, 1),
                nn.BatchNorm2d(self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.embed_dim, self.embed_dim, 3, 1, 1),
            )

        pre_embed = self.embed_dim
        for fusion_keys in in_fusion_key_list:
            in_embed = sum(fusion_keys.values())
            fusion = nn.Sequential(
                nn.Conv2d(in_embed + pre_embed, pre_embed, 3, 1, 1),
                nn.BatchNorm2d(pre_embed),
                nn.ReLU(inplace=True),
                nn.Conv2d(pre_embed, self.embed_dim, 3, 1, 1),
            )
            self.fusion_list.append(fusion)
            pre_embed = self.embed_dim

        self.out_conv = nn.Sequential(
            nn.Conv2d(pre_embed, pre_embed, 3, 1, 1),
            nn.BatchNorm2d(pre_embed),
            nn.ReLU(inplace=True),
            nn.Conv2d(pre_embed, pre_embed, 3, 1, 1),
        )

    def forward(self, inputs):
        in_features = []
        for key in self.in_feature_key:
            features = inputs[key]['encoder_features']
            features = torch.nn.functional.interpolate(
                features, self.feature_size, mode='bilinear', align_corners=False)
            in_features.append(features)
        in_features = torch.cat(in_features, dim=1)
        in_features = self.in_conv(in_features)

        for i, fusion_keys in enumerate(self.in_fusion_key_list):
            in_features = torch.nn.functional.interpolate(
                in_features, scale_factor=2, mode='bilinear', align_corners=False)
            in_features_h, in_features_w = in_features.shape[-2:]
            in_features_idx = len(self.in_fusion_key_list) - i - 1
            fusion_features = []
            for key in fusion_keys:
                features = inputs[key]['features_list'][in_features_idx]
                features = torch.nn.functional.interpolate(
                    features, (in_features_h, in_features_w), mode='bilinear', align_corners=False)
                fusion_features.append(features)
            fusion_features = torch.cat(fusion_features, dim=1)
            in_features = torch.cat([in_features, fusion_features], dim=1)
            in_features = self.fusion_list[i](in_features)

        out_features = self.out_conv(in_features)
        out_features = torch.nn.functional.interpolate(
            out_features, self.out_size, mode='bilinear', align_corners=False)
        return out_features


if __name__ == "__main__":
# two modalities, each exposing a fused latent map and a 3-level
    # feature pyramid (finest first) at their native resolutions.
    inputs = {
        'S2': {
            'encoder_features': torch.randn(2, 512, 8, 8),
            'features_list': [
                torch.randn(2, 128, 32, 32),
                torch.randn(2, 256, 16, 16),
                torch.randn(2, 512, 8, 8),
            ],
        },
        'HLS': {
            'encoder_features': torch.randn(2, 512, 8, 8),
            'features_list': [
                torch.randn(2, 128, 32, 32),
                torch.randn(2, 256, 16, 16),
torch.randn(2, 512, 8, 8),
            ],
        },
    }

    neck = MultiFusionNeck(
        embed_dim=512,
        in_feature_key=('S2', 'HLS'),
        feature_size=(8, 8),
        out_size=(32, 32),
        in_fusion_key_list=({'S2': 512, 'HLS': 512},
                            {'S2': 256},
                            {'S2': 128}),
    )
    out = neck(inputs)
    print(out.shape)
