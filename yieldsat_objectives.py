# --------------------------------------------------------
# YieldSAT objective routing and point-mode pretext objectives (YS-07).
#
# The five self-supervised objectives of models_multimodal_pretrain.py were
# written for spatial token sets. In point mode (one grid-cell history, one
# token per stream) some have no meaning and others need changed semantics.
# OBJECTIVE_ROUTING records the decision for each mode; ``require_objective``
# refuses an objective where it is not active. The point-mode variants
# implemented here:
#   masked_observation - hide a random subset of *observed* values (temporal
#       and static) and reconstruct them in normalized input space. This is
#       not spatial MAE and must not be reported as such.
#   forecast_last_valid - predict the optical values of each cell's final
#       valid eligible observation from earlier slots. All temporal streams
#       are hidden from that slot on, so weather cannot leak the target
#       slot's conditions. Slot 23 is never used by position.
# Families: terrain derivatives and DEM are one family; S2 bands are one
# sensor; soil uncertainties belong to soil. Cross-family objectives mask
# whole families; contrastive negatives exclude the same physical field.
# --------------------------------------------------------

import torch
from torch import nn

from dataset.yieldsat_schema import STREAMS

POINT, PATCH = 'point_timeseries', 'field_patch'

OBJECTIVE_ROUTING = {
    'supervised_yield': {POINT: 'active', PATCH: 'not implemented (patch mode not pursued)'},
    'masked_spatial_reconstruction': {
        POINT: 'inactive: no spatial neighbourhood in point mode; use masked_observation',
        PATCH: 'not implemented'},
    'masked_observation': {POINT: 'active (changed semantics: value masking, not spatial MAE)',
                           PATCH: 'not implemented'},
    'masked_modality': {POINT: 'restricted: mask whole source families (dem+terrain together)',
                        PATCH: 'not implemented'},
    'temporal_forecast': {POINT: 'adapted: forecast_last_valid, all temporal streams cut at target',
                          PATCH: 'not implemented'},
    'cross_modal_prediction': {POINT: 'restricted: leave-one-family-out only',
                               PATCH: 'not implemented'},
    'contrastive_alignment': {
        POINT: 'restricted: family-level views; negatives exclude same physical field',
        PATCH: 'not implemented'},
    'knowledge_relationships': {
        POINT: 'blocked: needs country-reviewed concept mapping (KP-01..KP-08) and verified units',
        PATCH: 'blocked'},
}

def require_objective(name, mode=POINT):
    status = OBJECTIVE_ROUTING.get(name, {}).get(mode)
    if status is None or not (status.startswith('active') or status.startswith('adapted')
                              or status.startswith('restricted')):
        raise ValueError('objective {} is not available in {} mode: {}'.format(name, mode, status))
    return status

def stream_families(streams):
    fam = {}
    for s in streams:
        fam.setdefault(STREAMS[s]['family'], []).append(s)
    return fam

def contrastive_negative_mask(physical_ids):
    """(B, B) bool: True where j is a valid negative for anchor i (different
    physical field). ``physical_ids`` is a sequence of hashable ids."""
    ids = list(physical_ids)
    codes = {v: i for i, v in enumerate(dict.fromkeys(ids))}
    t = torch.tensor([codes[v] for v in ids])
    return t[:, None] != t[None, :]

def last_valid_slot(mask):
    """Index of the final slot with any valid value; -1 if none. mask (B,T,C)."""
    valid = mask.any(dim=-1)
    T = valid.shape[1]
    pos = torch.arange(T, device=valid.device).expand_as(valid)
    last = torch.where(valid, pos, torch.full_like(pos, -1)).max(dim=1).values
    return last

def _clone_batch(batch):
    out = dict(batch)
    for key in ('inputs', 'masks', 'available'):
        out[key] = dict(batch[key])
    return out

def forecast_split(batch, layout, stream='yieldsat_s2', min_context=1):
    """Context batch with every temporal stream hidden from each cell's target
    slot onward, plus the target values/mask. Cells whose last valid slot has
    fewer than ``min_context`` valid earlier slots are marked ineligible."""
    mask = batch['masks'][stream]
    last = last_valid_slot(mask)
    B, T = mask.shape[:2]
    ar = torch.arange(B, device=mask.device)
    safe = last.clamp_min(0)
    target = batch['inputs'][stream][ar, safe]
    target_mask = mask[ar, safe] & (last >= 0)[:, None]
    slots = torch.arange(T, device=mask.device)[None, :]
    keep = slots < safe[:, None]
    earlier_valid = (mask.any(-1) & keep).sum(1)
    eligible = (last >= 0) & (earlier_valid >= min_context)
    ctx = _clone_batch(batch)
    for name, lay in layout.items():
        if lay['temporal']:
            ctx['masks'][name] = batch['masks'][name] & keep[:, :, None]
            ctx['inputs'][name] = batch['inputs'][name] * ctx['masks'][name]
            ctx['available'][name] = ctx['masks'][name].flatten(1).any(1).float()
    ctx['time_features'] = batch['time_features'] * keep[:, :, None]
    return ctx, target, target_mask & eligible[:, None], eligible

def mask_observations(batch, layout, ratio=0.3, generator=None):
    """Hide a random ``ratio`` of observed values per stream. Returns the
    masked batch and {stream: hidden bool mask}."""
    ctx = _clone_batch(batch)
    hidden = {}
    for name in layout:
        m = batch['masks'][name]
        r = torch.rand(m.shape, generator=generator, device='cpu').to(m.device)
        h = m & (r < ratio)
        hidden[name] = h
        ctx['masks'][name] = m & ~h
        ctx['inputs'][name] = batch['inputs'][name] * ctx['masks'][name]
        ctx['available'][name] = ctx['masks'][name].flatten(1).any(1).float()
    return ctx, hidden

class YieldSATPointPretrainer(nn.Module):
    """Yield-free point-mode pretraining around a YieldSATPointModel's
    encoders + fusion (masked_observation + forecast_last_valid). The crop
    token is not used, so pretrained sensor weights carry no label context."""

    def __init__(self, model, mask_ratio=0.3, forecast_weight=1.0, masked_weight=1.0):
        super().__init__()
        require_objective('masked_observation')
        require_objective('temporal_forecast')
        self.model = model
        self.mask_ratio = mask_ratio
        self.forecast_weight = forecast_weight
        self.masked_weight = masked_weight
        D = model.config['embed_dim']
        self.decoders = nn.ModuleDict()
        for name, lay in model.layout.items():
            n = len(lay['out_channels'])
            out = n * model.config['num_slots'] if lay['temporal'] else n
            self.decoders[name] = nn.Linear(D, out)
        self.forecast_decoder = nn.Linear(D, len(model.layout['yieldsat_s2']['out_channels']))

    def _latent(self, batch):
        crop = self.model.use_crop_context
        self.model.use_crop_context = False
        try:
            latents, _ = self.model.forward_features(batch, apply_dropout=True)
        finally:
            self.model.use_crop_context = crop
        return latents.mean(dim=1)

    def forward(self, batch):
        losses = {}
        ctx, hidden = mask_observations(batch, self.model.layout, self.mask_ratio)
        z = self._latent(ctx)
        num, den = 0.0, 0
        for name, h in hidden.items():
            pred = self.decoders[name](z).reshape(batch['inputs'][name].shape)
            err = (pred - batch['inputs'][name]).square()[h]
            num = num + err.sum()
            den += int(h.sum())
        losses['masked_observation'] = num / max(den, 1)
        if 'yieldsat_s2' in self.model.layout:
            fctx, target, tmask, eligible = forecast_split(batch, self.model.layout)
            pred = self.forecast_decoder(self._latent(fctx))
            losses['forecast_last_valid'] = ((pred - target).square()[tmask].sum()
                                             / tmask.sum().clamp_min(1))
        total = self.masked_weight * losses['masked_observation']
        if 'forecast_last_valid' in losses:
            total = total + self.forecast_weight * losses['forecast_last_valid']
        return total, losses
