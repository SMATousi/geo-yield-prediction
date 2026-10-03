# --------------------------------------------------------
# YieldSAT point-mode model (YS-06, YS-11) and the paper's LSTM baseline (PC-05).
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
# ``fusion`` selects how the encodings are combined (YS-11):
#   perceiver_summary - one summary token per stream -> LatentFusionTransformer
#                       (the original configuration, unchanged)
#   perceiver_tokens  - per-slot optical/weather tokens and per-depth soil
#                       tokens with per-token masks and a shared date encoding,
#                       read by a Perceiver with repeated cross-attention,
#                       pre-norm latents and input normalization
#   concat_mlp        - concatenated summary tokens -> MLP (baseline)
#   token_transformer - small transformer over the summary tokens (baseline)
# All variants end in the registered ``scalar_cell_yield`` head.
# ``sensor_state_dict`` exports only the sensor side with a descriptor that
# records the fusion layout, so incompatible transfers are refused.
# --------------------------------------------------------

import math

import torch
from torch import nn

from dataset.yieldsat_schema import CONTRACT_KEY, CROPS, SOIL_DEPTHS
from models_heads import build_head, masked_yield_loss
from models_latent_fusion import LatentFusionTransformer
from models_multimodal_encoder import MaskedTemporalEncoder, MultiModalEncoder

CROP_CONTEXT = 'crop_context'
FUSIONS = ('perceiver_summary', 'perceiver_tokens', 'concat_mlp', 'token_transformer', 'early')
# descriptor keys that must match for fusion weights to be transferable
FUSION_LAYOUT_KEYS = ('fusion', 'encoder_output', 'num_latents', 'depth', 'num_heads',
                      'cross_attn_layers')


def encoder_config(layout, embed_dim, time_dim=3, num_slots=24, temporal_hidden=128,
                   temporal_depth=2, static_hidden=64, output='summary'):
    """MultiModalEncoder config for a dataset ``stream_layout``. ``output``
    ('summary' | 'tokens') applies to the temporal and soil encoders."""
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
        if output == 'tokens' and cfg[name]['type'] in ('masked_temporal', 'soil_profile'):
            cfg[name]['output'] = 'tokens'
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


class DateEncoding(nn.Module):
    """Shared date-aware position for temporal tokens (YS-11c).

    Continuous sinusoidal features of days since seeding (periods 8–1024 days,
    geometric) -> Linear -> embed_dim. The same module is added to optical and
    weather tokens, so tokens of the same interval carry the same position.
    Undated or ineligible slots receive a learned "undated" embedding (and are
    masked in the fusion anyway).
    """

    def __init__(self, embed_dim, num_frequencies=16, min_period=8.0, max_period=1024.0):
        super().__init__()
        periods = torch.logspace(math.log10(min_period), math.log10(max_period), num_frequencies)
        self.register_buffer('omega', 2 * math.pi / periods, persistent=False)
        self.proj = nn.Linear(2 * num_frequencies, embed_dim)
        self.undated = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, days_since_seeding, dated):
        """days (B, T) float, dated (B, T) bool -> (B, T, embed_dim)."""
        ang = days_since_seeding.unsqueeze(-1) * self.omega
        enc = self.proj(torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1))
        return torch.where(dated.unsqueeze(-1), enc, self.undated.to(enc.dtype))


class YieldSATPointModel(nn.Module):

    def __init__(self, layout, embed_dim=128, num_latents=8, depth=2, num_heads=4,
                 modality_embed=32, modality_dropout=0.1, use_crop_context=True,
                 head_dropout=0.0, loss='mse', time_dim=3, num_slots=24,
                 fusion='perceiver_summary', cross_attn_layers=None, level_head=False, level_weight=1.0,
                 early_hidden=192, early_depth=3):
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError('fusion must be one of {}'.format(FUSIONS))
        tokens = fusion == 'perceiver_tokens'
        if cross_attn_layers is None:
            cross_attn_layers = 2 if tokens else 1
        if fusion == 'early':
            cross_attn_layers = 1
        if fusion != 'perceiver_tokens' and cross_attn_layers != 1:
            raise ValueError('cross_attn_layers only applies to perceiver_tokens')
        self.layout = layout
        self.streams = tuple(layout)
        self.use_crop_context = use_crop_context
        self.fusion_type = fusion
        self.encoder_output = 'tokens' if tokens else 'summary'
        # Creation order (encoders, crop, fusion, head) is kept identical to the
        # original model so perceiver_summary reproduces it exactly.
        if fusion == 'early':
            self._build_early(layout, embed_dim, time_dim, num_slots, use_crop_context, early_hidden,
                              early_depth, head_dropout, loss)
            self._build_level(level_head, level_weight, embed_dim, time_dim, num_slots)
            self.config = {'embed_dim': embed_dim, 'num_latents': num_latents, 'depth': depth,
                           'num_heads': num_heads, 'modality_embed': modality_embed,
                           'modality_dropout': modality_dropout, 'use_crop_context': use_crop_context,
                           'head_dropout': head_dropout, 'loss': loss, 'time_dim': time_dim,
                           'num_slots': num_slots, 'fusion': fusion, 'encoder_output': 'early',
                           'cross_attn_layers': 1, 'level_head': level_head, 'level_weight': level_weight,
                           'early_hidden': early_hidden, 'early_depth': early_depth}
            return
        self.encoders = MultiModalEncoder(
            encoder_config(layout, embed_dim, time_dim=time_dim, num_slots=num_slots,
                           output=self.encoder_output),
            embed_dim=embed_dim, modality_dropout=modality_dropout)
        self.token_names = self.streams + ((CROP_CONTEXT,) if use_crop_context else ())
        if use_crop_context:
            self.crop_embed = nn.Embedding(len(CROPS), embed_dim)
        if fusion in ('perceiver_summary', 'perceiver_tokens'):
            modalities = {}
            for name in self.token_names:
                n_tok = 1
                if tokens and name in layout and layout[name]['temporal']:
                    n_tok = num_slots
                elif tokens and name == 'yieldsat_soil':
                    n_tok = len(SOIL_DEPTHS)
                modalities[name] = {'spatial': 1, 'temporal': n_tok}
            extra = dict(cross_attn_layers=cross_attn_layers, norm_first=True, input_norm=True,
                         latent_init='trunc_normal') if tokens else {}
            self.fusion = LatentFusionTransformer(
                embed_dim=embed_dim, num_latents=num_latents, depth=depth, num_heads=num_heads,
                modalities=modalities, modality_embed=modality_embed, **extra)
        elif fusion == 'concat_mlp':
            n = len(self.token_names)
            self.fusion = nn.Sequential(nn.LayerNorm(n * embed_dim),
                                        nn.Linear(n * embed_dim, 2 * embed_dim), nn.GELU(),
                                        nn.Linear(2 * embed_dim, embed_dim))
        else:  # token_transformer
            self.fusion = nn.ModuleDict({
                'layers': nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=4 * embed_dim,
                                               dropout=0.0, activation='gelu', batch_first=True,
                                               norm_first=True),
                    num_layers=depth, enable_nested_tensor=False),
                'norm': nn.LayerNorm(embed_dim),
            })
            self.fusion_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.fusion_modality = nn.Parameter(torch.zeros(1, len(self.token_names), embed_dim))
            nn.init.trunc_normal_(self.fusion_cls, std=0.02)
            nn.init.trunc_normal_(self.fusion_modality, std=0.02)
        self.head = build_head('scalar_cell_yield', embed_dim=embed_dim, dropout=head_dropout,
                               loss=loss)
        if tokens:
            self.date_enc = DateEncoding(embed_dim)
        self.config = {'embed_dim': embed_dim, 'num_latents': num_latents, 'depth': depth,
                       'num_heads': num_heads, 'modality_embed': modality_embed,
                       'modality_dropout': modality_dropout,
                       'use_crop_context': use_crop_context, 'head_dropout': head_dropout,
                       'loss': loss, 'time_dim': time_dim, 'num_slots': num_slots,
                       'fusion': fusion, 'encoder_output': self.encoder_output,
                       'cross_attn_layers': cross_attn_layers}
        self._build_level(level_head, level_weight, embed_dim, time_dim, num_slots)
        if level_head:
            self.config.update(level_head=True, level_weight=level_weight)

    # ---- S4 early fusion / S5 season level (spec/yieldsat-improvement.md) ----
    def _build_early(self, layout, embed_dim, time_dim, num_slots, use_crop_context, hidden, depth,
                     head_dropout, loss):
        """S4: one temporal encoder over the per-slot concatenation of every
        temporal stream and every static stream (repeated over the dated
        slots), each with its validity mask, plus the date features."""
        n = sum(len(l['out_channels']) for l in layout.values())
        self.early_enc = MaskedTemporalEncoder(n, embed_dim, time_dim=time_dim, max_len=num_slots,
                                               hidden_dim=hidden, depth=depth)
        if use_crop_context:
            self.crop_embed = nn.Embedding(len(CROPS), embed_dim)
        self.head = build_head('scalar_cell_yield', embed_dim=embed_dim, dropout=head_dropout, loss=loss)

    def _early_input(self, batch):
        tv = batch['time_valid'].unsqueeze(-1)
        vals, masks = [], []
        for name, lay in self.layout.items():
            v = batch['inputs'][name]
            m = batch['masks'][name].to(v.dtype)
            if not lay['temporal']:                       # static: repeat over the dated slots
                T = tv.shape[1]
                v = v.unsqueeze(1).expand(-1, T, -1)
                m = m.unsqueeze(1).expand(-1, T, -1) * tv.to(v.dtype)
            vals.append(v * m)
            masks.append(m)
        return torch.cat(vals + masks + [batch['time_features'].to(vals[0].dtype)], -1)

    def _build_level(self, level_head, level_weight, embed_dim, time_dim, num_slots):
        """S5: season-level term from season-wide inputs only (the field's
        weather series and the crop), tied to the field-season mean target."""
        self.level_weight = level_weight
        if not level_head:
            return
        self.level_has_weather = 'yieldsat_weather' in self.layout
        if self.level_has_weather:
            n = len(self.layout['yieldsat_weather']['out_channels'])
            self.level_weather = MaskedTemporalEncoder(n, embed_dim, time_dim=time_dim, max_len=num_slots,
                                                       hidden_dim=64, depth=2)
        self.level_crop = nn.Embedding(len(CROPS), embed_dim)
        self.level_mlp = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU(),
                                       nn.Linear(embed_dim, 1))

    def season_level(self, batch):
        h = self.level_crop(batch['crop'])
        if self.level_has_weather:
            v = batch['inputs']['yieldsat_weather']
            m = batch['masks']['yieldsat_weather'].to(v.dtype)
            h = h + self.level_weather(torch.cat([v * m, m, batch['time_features'].to(v.dtype)], -1))[:, 0]
        return self.level_mlp(h).squeeze(-1)

    def forward_features(self, batch, apply_dropout=True):
        """Returns fused tokens (B, L, D) for the head and the stream mask."""
        if self.fusion_type == 'early':
            h = self.early_enc(self._early_input(batch))                    # (B, 1, D)
            if self.use_crop_context:
                h = h + self.crop_embed(batch['crop']).unsqueeze(1)
            return h, None
        inputs = pack_inputs(batch, self.layout)
        available = {name: batch['available'][name] for name in self.streams}
        embeddings, mask, token_mask = self.encoders.forward_with_missing(
            inputs, available=available, apply_dropout=apply_dropout, return_token_masks=True)
        if self.use_crop_context:
            embeddings[CROP_CONTEXT] = self.crop_embed(batch['crop']).unsqueeze(1)
            mask[CROP_CONTEXT] = torch.ones_like(batch['crop'], dtype=torch.float32)
        if self.fusion_type == 'perceiver_summary':
            return self.fusion(embeddings, mask=mask), mask
        if self.fusion_type == 'perceiver_tokens':
            days = batch['time_features'][..., 0] * 365.0
            dated = batch['time_valid']
            for name, lay in self.layout.items():
                if lay['temporal'] and embeddings[name].shape[1] == days.shape[1]:
                    embeddings[name] = embeddings[name] + self.date_enc(days, dated)
            token_mask = {k: v for k, v in token_mask.items() if k in self.streams}
            return self.fusion(embeddings, mask=mask, token_mask=token_mask), mask
        # summary baselines: one token per stream (+ crop), missing streams are
        # already the learned missing-modality token
        x = torch.cat([embeddings[n][:, :1] for n in self.token_names], dim=1)   # (B, S, D)
        if self.fusion_type == 'concat_mlp':
            return self.fusion(x.flatten(1)).unsqueeze(1), mask
        x = x + self.fusion_modality
        x = torch.cat([self.fusion_cls.expand(x.shape[0], -1, -1), x], dim=1)
        absent = torch.stack([mask[n] < 0.5 for n in self.token_names], dim=1)
        pad = torch.cat([torch.zeros_like(absent[:, :1]), absent], dim=1)
        h = self.fusion['layers'](x, src_key_padding_mask=pad)
        return self.fusion['norm'](h[:, :1]), mask

    def forward(self, batch, apply_dropout=True, return_level=False):
        latents, _ = self.forward_features(batch, apply_dropout=apply_dropout)
        pred = self.head(latents)
        level = None
        if hasattr(self, 'level_mlp'):
            level = self.season_level(batch)
            pred = pred + level
        return (pred, level) if return_level else pred

    def loss(self, batch, apply_dropout=True):
        pred, level = self.forward(batch, apply_dropout=apply_dropout, return_level=True)
        loss = self.head.compute_loss(pred, batch['target'], valid_mask=batch['target_valid'])
        if level is not None:
            ok = batch['target_valid'] & torch.isfinite(batch['season_target'])
            if ok.any():
                loss = loss + self.level_weight * ((level - batch['season_target'])[ok] ** 2).mean()
        return pred, loss

    # ---- checkpoint transfer --------------------------------------------
    def descriptor(self):
        return {'contract': CONTRACT_KEY, 'mode': 'point_timeseries', 'model': 'yieldsat_point',
                'streams': {n: list(l['out_channels']) for n, l in self.layout.items()},
                'config': self.config}

    _SENSOR_PREFIXES = ('encoders.', 'fusion.', 'fusion_cls', 'fusion_modality', 'date_enc.')

    def sensor_state_dict(self):
        """Sensor side only (encoders, fusion, date encoding); no head, no crop
        embedding."""
        state = {k: v for k, v in self.state_dict().items() if k.startswith(self._SENSOR_PREFIXES)}
        return {'descriptor': self.descriptor(), 'state_dict': state}

    def load_sensor_state_dict(self, payload, strict_streams=True, encoders_only=False):
        """Load a sensor checkpoint after an explicit compatibility check.

        Fusion weights are loaded only when the fusion layout matches; with
        ``encoders_only`` just the per-stream encoders are transferred, which
        is valid across fusion variants (encoder weights do not depend on the
        summary/tokens output mode).
        """
        desc = payload['descriptor']
        mine = self.descriptor()
        if desc['contract'] != mine['contract'] or desc['mode'] != mine['mode']:
            raise ValueError('checkpoint contract/mode {}:{} is incompatible with {}:{}'.format(
                desc['contract'], desc['mode'], mine['contract'], mine['mode']))
        if desc.get('model', 'yieldsat_point') != 'yieldsat_point':
            raise ValueError('checkpoint is a {} model'.format(desc['model']))
        for key in ('embed_dim', 'modality_embed', 'time_dim', 'num_slots'):
            if desc['config'][key] != mine['config'][key]:
                raise ValueError('checkpoint {}={} != model {}'.format(
                    key, desc['config'][key], mine['config'][key]))
        theirs = dict(desc['config'])
        theirs.setdefault('fusion', 'perceiver_summary')
        theirs.setdefault('encoder_output', 'summary')
        theirs.setdefault('cross_attn_layers', 1)
        diff = [k for k in FUSION_LAYOUT_KEYS if theirs.get(k) != mine['config'][k]]
        if diff and not encoders_only:
            raise ValueError('checkpoint fusion layout differs ({}); use encoders_only'.format(
                ', '.join('{}={}!={}'.format(k, theirs.get(k), mine['config'][k]) for k in diff)))
        for name, channels in mine['streams'].items():
            if name not in desc['streams']:
                if strict_streams:
                    raise ValueError('checkpoint lacks stream {}'.format(name))
                continue
            if desc['streams'][name] != channels:
                raise ValueError('stream {} channels differ from checkpoint'.format(name))
        prefixes = ('encoders.',) if encoders_only else self._SENSOR_PREFIXES
        own = self.state_dict()
        state = {k: v for k, v in payload['state_dict'].items()
                 if k.startswith(prefixes) and k in own and own[k].shape == v.shape}
        skipped = sorted(set(payload['state_dict']) - set(state))
        missing = [k for k in own if k.startswith(prefixes) and k not in state]
        self.load_state_dict(state, strict=False)
        return {'loaded': len(state), 'skipped': skipped, 'missing_sensor_keys': missing,
                'encoders_only': encoders_only}


class PaperLSTMBaseline(nn.Module):
    """The YieldSAT paper's pixel LSTM baseline (Pathak et al.; release ML
    tutorial) for protocol validation (PC-05).

    Input fusion as in the paper: the selected temporal streams are
    concatenated per slot and static streams are repeated over the 24 slots;
    values are used as delivered by the dataset (with ``normalization='none'``
    and ``fill_value=-1`` that is the tutorial's raw NaN->-1 input). Masks are
    ignored, as in the tutorial. 1-layer LSTM (hidden 64), last-slot readout,
    Linear -> yield. Trained with MSE on the dataset's target (raw t/ha when
    ``target_normalization='none'``).
    """

    def __init__(self, layout, hidden_dim=64, num_layers=1):
        super().__init__()
        self.layout = layout
        self.streams = tuple(layout)
        in_dim = sum(len(l['out_channels']) for l in layout.values())
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.use_crop_context = False
        self.config = {'model': 'paper_lstm', 'hidden_dim': hidden_dim, 'num_layers': num_layers,
                       'in_dim': in_dim}

    def _inputs(self, batch):
        parts, T = [], None
        for name, lay in self.layout.items():
            if lay['temporal']:
                T = batch['inputs'][name].shape[1]
        for name, lay in self.layout.items():
            v = batch['inputs'][name]
            parts.append(v if lay['temporal'] else v.unsqueeze(1).expand(-1, T or 24, -1))
        return torch.cat(parts, dim=-1)

    def forward(self, batch, apply_dropout=True):
        out, _ = self.lstm(self._inputs(batch))
        return self.fc(out[:, -1]).squeeze(-1)

    def loss(self, batch, apply_dropout=True):
        pred = self.forward(batch)
        target = torch.where(batch['target_valid'], batch['target'],
                             torch.full_like(batch['target'], float('nan')))
        return pred, masked_yield_loss(pred, target, loss='mse')

    def descriptor(self):
        return {'contract': CONTRACT_KEY, 'mode': 'point_timeseries', 'model': 'paper_lstm',
                'streams': {n: list(l['out_channels']) for n, l in self.layout.items()},
                'config': self.config}

    def sensor_state_dict(self):
        return {'descriptor': self.descriptor(), 'state_dict': self.state_dict()}

    def load_sensor_state_dict(self, payload, **kwargs):
        raise ValueError('the paper LSTM baseline does not support sensor transfer')
