# --------------------------------------------------------
# YieldSAT point-mode model (YS-06).
#
# One dedicated encoder per stream of the yieldsat_preprocessed_v1 contract,
# all built through MultiModalEncoder so the missing-modality token,
# per-sample availability and modality dropout behave exactly as for the
# downloader streams:
#   yieldsat_s2       masked_temporal  (B, T, 12) + masks + time features
#   yieldsat_weather  masked_temporal  (B, T, 4)  + masks + time features
#   yieldsat_dem      masked_static    (B, 1)
#   yieldsat_terrain  masked_static    (B, 4)     (5 with cyclic aspect)
#   yieldsat_soil     soil_profile     (B, 48)    (96 with uncertainty)
# Each encoder emits one token; tokens are fused by the Perceiver-style
# LatentFusionTransformer (with an optional crop-context token) and decoded by
# the registered ``scalar_cell_yield`` head. ``sensor_state_dict`` exports
# only encoders + fusion with a compatibility descriptor for transfer.
# --------------------------------------------------------

import torch
from torch import nn

from dataset.yieldsat_schema import CONTRACT_KEY, CROPS
from models_heads import build_head
from models_latent_fusion import LatentFusionTransformer
from models_multimodal_encoder import MultiModalEncoder

CROP_CONTEXT = 'crop_context'


def encoder_config(layout, embed_dim, time_dim=3, num_slots=24, temporal_hidden=128,
                   temporal_depth=2, static_hidden=64):
    """MultiModalEncoder config for a dataset ``stream_layout``."""
    cfg = {}
    for name, lay in layout.items():
        n = len(lay['out_channels'])
        if lay['temporal']:
            cfg[name] = {'type': 'masked_temporal', 'in_dim': n, 'time_dim': time_dim,
                         'max_len': num_slots, 'hidden_dim': temporal_hidden,
                         'depth': temporal_depth, 'embed_dim': embed_dim,
                         'norm_source': 'prenormalized'}
        elif name == 'yieldsat_soil':
            cfg[name] = {'type': 'soil_profile', 'embed_dim': embed_dim,
                         'with_uncertainty': n == 96, 'hidden_dim': static_hidden,
                         'norm_source': 'prenormalized'}
        else:
            cfg[name] = {'type': 'masked_static', 'in_dim': n, 'embed_dim': embed_dim,
                         'hidden_dim': static_hidden, 'norm_source': 'prenormalized'}
    return cfg


def pack_inputs(batch, layout):
    """Dataset batch -> MultiModalEncoder inputs (value*mask, mask[, time])."""
    packed = {}
    for name, lay in layout.items():
        v = batch['inputs'][name]
        m = batch['masks'][name].to(v.dtype)
        parts = [v * m, m]
        if lay['temporal']:
            parts.append(batch['time_features'].to(v.dtype))
        packed[name] = torch.cat(parts, dim=-1)
    return packed


class YieldSATPointModel(nn.Module):

    def __init__(self, layout, embed_dim=128, num_latents=8, depth=2, num_heads=4,
                 modality_embed=32, modality_dropout=0.1, use_crop_context=True,
                 head_dropout=0.0, loss='mse', time_dim=3, num_slots=24):
        super().__init__()
        self.layout = layout
        self.streams = tuple(layout)
        self.use_crop_context = use_crop_context
        self.encoders = MultiModalEncoder(
            encoder_config(layout, embed_dim, time_dim=time_dim, num_slots=num_slots),
            embed_dim=embed_dim, modality_dropout=modality_dropout)
        modalities = {name: {'spatial': 1, 'temporal': 1} for name in self.streams}
        if use_crop_context:
            modalities[CROP_CONTEXT] = {'spatial': 1, 'temporal': 1}
            self.crop_embed = nn.Embedding(len(CROPS), embed_dim)
        self.fusion = LatentFusionTransformer(
            embed_dim=embed_dim, num_latents=num_latents, depth=depth, num_heads=num_heads,
            modalities=modalities, modality_embed=modality_embed)
        self.head = build_head('scalar_cell_yield', embed_dim=embed_dim, dropout=head_dropout,
                               loss=loss)
        self.config = {'embed_dim': embed_dim, 'num_latents': num_latents, 'depth': depth,
                       'num_heads': num_heads, 'modality_embed': modality_embed,
                       'modality_dropout': modality_dropout,
                       'use_crop_context': use_crop_context, 'head_dropout': head_dropout,
                       'loss': loss, 'time_dim': time_dim, 'num_slots': num_slots}

    def forward_features(self, batch, apply_dropout=True):
        inputs = pack_inputs(batch, self.layout)
        available = {name: batch['available'][name] for name in self.streams}
        embeddings, mask = self.encoders.forward_with_missing(
            inputs, available=available, apply_dropout=apply_dropout)
        if self.use_crop_context:
            embeddings[CROP_CONTEXT] = self.crop_embed(batch['crop']).unsqueeze(1)
            mask[CROP_CONTEXT] = torch.ones_like(batch['crop'], dtype=torch.float32)
        return self.fusion(embeddings, mask=mask), mask

    def forward(self, batch, apply_dropout=True):
        latents, _ = self.forward_features(batch, apply_dropout=apply_dropout)
        return self.head(latents)

    def loss(self, batch, apply_dropout=True):
        pred = self.forward(batch, apply_dropout=apply_dropout)
        return pred, self.head.compute_loss(pred, batch['target'], valid_mask=batch['target_valid'])

    # ---- checkpoint transfer --------------------------------------------
    def descriptor(self):
        return {'contract': CONTRACT_KEY, 'mode': 'point_timeseries',
                'streams': {n: list(l['out_channels']) for n, l in self.layout.items()},
                'config': self.config}

    def sensor_state_dict(self):
        """Encoders + fusion only (no head, no crop embedding) for transfer."""
        state = {k: v for k, v in self.state_dict().items()
                 if k.startswith('encoders.') or k.startswith('fusion.')}
        return {'descriptor': self.descriptor(), 'state_dict': state}

    def load_sensor_state_dict(self, payload, strict_streams=True):
        """Load a sensor checkpoint after an explicit compatibility check."""
        desc = payload['descriptor']
        mine = self.descriptor()
        if desc['contract'] != mine['contract'] or desc['mode'] != mine['mode']:
            raise ValueError('checkpoint contract/mode {}:{} is incompatible with {}:{}'.format(
                desc['contract'], desc['mode'], mine['contract'], mine['mode']))
        for key in ('embed_dim', 'modality_embed', 'time_dim', 'num_slots'):
            if desc['config'][key] != mine['config'][key]:
                raise ValueError('checkpoint {}={} != model {}'.format(
                    key, desc['config'][key], mine['config'][key]))
        for name, channels in mine['streams'].items():
            if name not in desc['streams']:
                if strict_streams:
                    raise ValueError('checkpoint lacks stream {}'.format(name))
                continue
            if desc['streams'][name] != channels:
                raise ValueError('stream {} channels differ from checkpoint'.format(name))
        own = self.state_dict()
        state = {k: v for k, v in payload['state_dict'].items()
                 if k in own and own[k].shape == v.shape}
        skipped = sorted(set(payload['state_dict']) - set(state))
        missing = [k for k in own if (k.startswith('encoders.') or k.startswith('fusion.'))
                   and k not in state]
        self.load_state_dict(state, strict=False)
        return {'loaded': len(state), 'skipped': skipped, 'missing_sensor_keys': missing}
