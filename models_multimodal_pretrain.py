# --------------------------------------------------------
# Multimodal self-supervised pretraining framework.
#
# The base repo's pretraining is confined to SimCLR contrastive learning
# between augmented Sentinel-2 views paired with weather in the PVT module
# (models_pvt_simclr.py). This module realises the foundation-scale
# self-supervised objectives (gap g7) on top of the shared multimodal
# backbone -- the native-resolution heterogeneous encoders
# (models_multimodal_encoder.py) fused by the Perceiver-style latent fusion
# transformer (models_latent_fusion.py) -- so the backbone learns general
# field representations from unlabelled fields without yield labels.
#
# Five objectives are combined into a single weighted pretraining loss:
#   1. masked spatial-token reconstruction  -- mask a random subset of a
#      modality's spatial tokens and reconstruct them from the fused latent
#      (MAE-style, reusing the repo's random_masking primitive).
#   2. masked-modality modeling            -- drop whole modalities and
#      reconstruct their tokens from the remaining sources.
#   3. temporal forecasting                -- for time-series modalities,
#      encode the past frames and predict the future frame from the fused
#      latent.
#   4. cross-modal prediction              -- leave-one-out: predict each
#      modality's tokens from the fused representation of all the others.
#   5. contrastive alignment               -- InfoNCE that pulls together the
#      field-level embeddings of different modalities observing the *same*
#      field while pushing apart embeddings of different fields, so the
#      backbone learns that distinct sensors covering one field share a
#      common latent representation.
#
# All reconstruction is performed in the shared embedding space (embed_dim),
# which is uniform across the heterogeneous encoders, so a single per-modality
# reconstruction head suffices regardless of a source's native resolution or
# structure. The module reuses the repo's MultiModalEncoder, LatentFusionTransformer,
# random_masking and normalize_modality utilities, so no new dependency is
# introduced.
# --------------------------------------------------------

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from util.norm_stats import normalize_modality

from models_multimodal_encoder import MultiModalEncoder
from models_latent_fusion import LatentFusionTransformer
from models_mae_spectral import random_masking


def _mlp(in_dim, hidden_dim, out_dim):
    """Small two-layer MLP used for the self-supervised prediction heads."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


class MultimodalSelfSupervisedPretrain(nn.Module):
    """Unified multimodal self-supervised pretraining framework.

    Wraps the native-resolution heterogeneous encoders + Perceiver latent
    fusion backbone and attaches a set of self-supervised objectives that
    operate on unlabelled fields. ``forward`` returns ``(total_loss, losses)``
    where ``losses`` is a dict of per-objective scalar losses keyed by
    ``'spatial'``, ``'modality'``, ``'temporal'``, ``'cross_modal'`` and
    ``'contrastive'``, and ``total_loss`` is their ``loss_weights``-weighted
    sum. The backbone is shared across all objectives, so it learns a general
    field representation that can later be transferred to the supervised
    yield-map head.

    ``encoders_cfg`` and ``modalities`` follow the same conventions as
    ``MultiModalEncoder`` / ``LatentFusionTransformer`` (see
    models_multimodal_encoder.py and models_latent_fusion.py). Time-series
    modalities (``{'type': 'timeseries', 'in_dim': D}``) additionally get a
    temporal-forecasting head that predicts the next frame.
    """

    def __init__(self, encoders_cfg, embed_dim=192, num_latents=64, depth=4,
                 num_heads=3, modalities=None, modality_embed=64, mask_ratio=0.75,
                 modality_mask_ratio=0.5, temporal_forecast_steps=1,
                 loss_weights=None, decoder_embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.modality_mask_ratio = modality_mask_ratio
        self.temporal_forecast_steps = temporal_forecast_steps
        self.loss_weights = loss_weights or {
            'spatial': 1.0, 'modality': 1.0, 'temporal': 1.0, 'cross_modal': 1.0,
            'contrastive': 1.0,
        }

        # shared backbone: native-resolution encoders + latent fusion
        self.encoders = MultiModalEncoder(encoders_cfg, embed_dim=embed_dim,
                                          modality_dropout=0.0)
        self.fusion = LatentFusionTransformer(
            embed_dim=embed_dim, num_latents=num_latents, depth=depth,
            num_heads=num_heads, modalities=modalities or {},
            modality_embed=modality_embed)

        # learned mask token substituted for masked spatial tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # shared contrastive projection head + temperature for the InfoNCE
        # alignment objective (pulls together different modalities observing
        # the same field, pushes apart different fields).
        self.contrast_head = _mlp(embed_dim, decoder_embed_dim, embed_dim)
        self.contrast_tau = 0.1

        # per-modality reconstruction heads: fused latent -> modality tokens
        # (embedding-space reconstruction, uniform across native resolutions)
        self.recon_heads = nn.ModuleDict()
        for name in encoders_cfg:
            self.recon_heads[name] = _mlp(embed_dim, decoder_embed_dim, embed_dim)

        # temporal-forecasting heads for time-series modalities: fused latent
        # -> next-frame values (in_dim)
        self.temporal_modalities = {}
        self.temporal_heads = nn.ModuleDict()
        for name, cfg in encoders_cfg.items():
            if cfg.get('type') == 'timeseries':
                in_dim = cfg.get('in_dim')
                self.temporal_modalities[name] = in_dim
                self.temporal_heads[name] = _mlp(embed_dim, decoder_embed_dim, in_dim)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _zero_loss(self, device):
        return torch.tensor(0.0, device=device)

    def _masked_objectives(self, inputs, available):
        """Masked spatial-token reconstruction + masked-modality modeling.

        Encodes every modality at its native resolution, then at train time
        either drops a whole modality (masked-modality modeling) or masks a
        random subset of its spatial tokens (masked spatial-token
        reconstruction). The masked token sets are fused by the backbone and
        each modality's reconstruction head predicts its tokens from the fused
        latent. Returns ``(spatial_loss, modality_loss)``.
        """
        embeddings, mask = self.encoders.forward_with_missing(
            inputs, available=available, apply_dropout=False)

        device = next(self.parameters()).device
        masked_embeddings = {}
        modality_present = {}
        spatial_masks = {}

        for name, emb in embeddings.items():
            if self.training and torch.rand(1).item() < self.modality_mask_ratio:
                # masked-modality modeling: drop the whole modality, keep a
                # single learned missing token flagged absent in the mask.
                modality_present[name] = False
                token = self.encoders.missing_tokens[name].unsqueeze(0).unsqueeze(0)
                masked_embeddings[name] = token.expand(emb.shape[0], 1, -1)
                spatial_masks[name] = torch.ones(
                    emb.shape[0], emb.shape[1], device=emb.device)
            else:
                modality_present[name] = True
                if self.training and emb.shape[1] > 1:
                    # masked spatial-token reconstruction: mask a random subset
                    # of tokens, reinsert mask tokens to keep the full length so
                    # the positional embeddings stay aligned.
                    x_masked, mask_tok, ids_restore = random_masking(
                        emb, self.mask_ratio)
                    num_masked = emb.shape[1] - x_masked.shape[1]
                    mask_tok_emb = self.mask_token.expand(
                        emb.shape[0], num_masked, -1)
                    full = torch.cat([x_masked, mask_tok_emb], dim=1)
                    full = torch.gather(
                        full, dim=1,
                        index=ids_restore.unsqueeze(-1).repeat(1, 1, emb.shape[2]))
                    masked_embeddings[name] = full
                    spatial_masks[name] = mask_tok
                else:
                    masked_embeddings[name] = emb
                    spatial_masks[name] = torch.zeros(
                        emb.shape[0], emb.shape[1], device=emb.device)

        latents = self.fusion(masked_embeddings, mask=modality_present)
        fused = latents.mean(dim=1)  # (B, embed_dim) pooled field representation

        spatial_losses = []
        modality_losses = []
        for name, emb in embeddings.items():
            if not modality_present[name]:
                # masked-modality modeling: reconstruct the whole modality from
                # the fused representation of the remaining sources.
                pred = self.recon_heads[name](fused)          # (B, embed_dim)
                target = emb.mean(dim=1)                       # (B, embed_dim)
                modality_losses.append(F.mse_loss(pred, target))
            else:
                # masked spatial-token reconstruction: reconstruct the masked
                # tokens (masked-mean aggregated).
                pred = self.recon_heads[name](fused).unsqueeze(1).expand(
                    -1, emb.shape[1], -1)                     # (B, L, embed_dim)
                diff = ((pred - emb) ** 2).mean(dim=-1)       # (B, L)
                m = spatial_masks[name]
                if m.sum() > 0:
                    spatial_losses.append((diff * m).sum() / m.sum())

        spatial = (torch.stack(spatial_losses).mean() if spatial_losses
                   else self._zero_loss(device))
        modality = (torch.stack(modality_losses).mean() if modality_losses
                    else self._zero_loss(device))
        return spatial, modality

    def _temporal_forecast(self, inputs, available):
        """Temporal forecasting for time-series modalities.

        For each time-series modality, encodes the past frames (all but the
        last) through its dedicated encoder, fuses them with the other
        modalities' full embeddings, and predicts the future frame from the
        fused latent. Returns the mean MSE over the available time-series
        modalities.
        """
        device = next(self.parameters()).device
        if not self.temporal_modalities:
            return self._zero_loss(device)

        embeddings, mask = self.encoders.forward_with_missing(
            inputs, available=available, apply_dropout=False)
        losses = []
        for name, in_dim in self.temporal_modalities.items():
            if name not in inputs:
                continue
            if available is not None:
                avail = available.get(name, True)
                # per-sample availability tensors: forecast if any sample present
                if isinstance(avail, torch.Tensor):
                    if not bool(avail.any()):
                        continue
                elif not avail:
                    continue
            x = inputs[name]
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(
                    normalize_modality(self.encoders.norm_sources[name], x))
            if x.shape[1] < 2:
                continue
            past = x[:, :-1]                                  # (B, T-1, D)
            future = x[:, -1]                                 # (B, D)
            past_emb = self.encoders.encoders[name](past)     # (B, 1, embed_dim)
            fused_embeddings = dict(embeddings)
            fused_embeddings[name] = past_emb
            latents = self.fusion(fused_embeddings, mask=mask)
            fused = latents.mean(dim=1)                       # (B, embed_dim)
            pred = self.temporal_heads[name](fused)           # (B, D)
            losses.append(F.mse_loss(pred, future))
        return (torch.stack(losses).mean() if losses
                else self._zero_loss(device))

    def _cross_modal_prediction(self, inputs, available):
        """Cross-modal prediction (leave-one-out).

        For each modality, fuses the embeddings of all the *other* modalities
        and reconstructs the held-out modality's tokens from the fused latent.
        Returns the mean MSE over the modalities that have at least one other
        source to predict from.
        """
        device = next(self.parameters()).device
        embeddings, mask = self.encoders.forward_with_missing(
            inputs, available=available, apply_dropout=False)
        losses = []
        for name, emb in embeddings.items():
            others = {k: v for k, v in embeddings.items() if k != name}
            if not others:
                continue
            latents = self.fusion(others, mask={k: mask[k] for k in others})
            fused = latents.mean(dim=1)                       # (B, embed_dim)
            pred = self.recon_heads[name](fused)              # (B, embed_dim)
            target = emb.mean(dim=1)                          # (B, embed_dim)
            losses.append(F.mse_loss(pred, target))
        return (torch.stack(losses).mean() if losses
                else self._zero_loss(device))

    def _contrastive_alignment(self, inputs, available):
        """Contrastive alignment between modalities covering the same field.

        Mean-pools each modality's tokens to a field-level embedding, projects
        it through the shared contrastive head, and runs InfoNCE over every
        pair of modalities: the positive pair is the two modalities observing
        the *same* field (same batch index), while the negatives are the other
        fields' embeddings. This teaches the backbone that distinct sensors
        covering one field share a common latent representation, complementing
        the reconstruction objectives. Only modalities present for the whole
        batch are used, so per-sample missing data does not corrupt the
        alignment. Returns the mean InfoNCE loss over modality pairs.
        """
        device = next(self.parameters()).device
        embeddings, mask = self.encoders.forward_with_missing(
            inputs, available=available, apply_dropout=False)
        names = []
        for name in embeddings:
            m = mask[name]
            if isinstance(m, torch.Tensor) and m.numel() > 1 and not bool(m.all()):
                continue
            names.append(name)
        if len(names) < 2:
            return self._zero_loss(device)
        z = {n: F.normalize(self.contrast_head(embeddings[n].mean(dim=1)), dim=-1)
             for n in names}
        losses = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                logits = z[a] @ z[b].T / self.contrast_tau   # (B, B)
                labels = torch.arange(logits.shape[0], device=device)
                losses.append(F.cross_entropy(logits, labels))
                losses.append(F.cross_entropy(logits.T, labels))
        return (torch.stack(losses).mean() if losses
                else self._zero_loss(device))

    def forward(self, inputs, available=None):
        """Run all self-supervised objectives on a batch of unlabelled fields.

        ``inputs`` is a dict of modality tensors, each at its native
        resolution. ``available`` optionally overrides which modalities are
        present. Returns ``(total_loss, losses)`` where ``losses`` is a dict of
        per-objective scalar losses and ``total_loss`` is their weighted sum.
        """
        spatial, modality = self._masked_objectives(inputs, available)
        temporal = self._temporal_forecast(inputs, available)
        cross_modal = self._cross_modal_prediction(inputs, available)
        contrastive = self._contrastive_alignment(inputs, available)

        losses = {
            'spatial': spatial,
            'modality': modality,
            'temporal': temporal,
            'cross_modal': cross_modal,
            'contrastive': contrastive,
        }
        total = sum(self.loss_weights[k] * losses[k]
                    for k in self.loss_weights if k in losses)
        return total, losses

    def set_encoder_trainable(self, trainable=True):
        """Freeze / unfreeze the modality encoders.

        Enables the frozen vs. partially fine-tuned vs. fully fine-tuned
        pretrained-encoder comparison: call ``set_encoder_trainable(False)``
        to freeze the modality encoders while training only the fusion
        backbone and heads, then selectively unfreeze later.
        """
        for param in self.encoders.parameters():
            param.requires_grad = trainable


if __name__ == "__main__":
    # heterogeneous modalities at their native resolutions: a DEM raster, a
    # multitemporal SAR stack, a weather time series, tabular soil attributes
    # and a categorical crop-history layer.
    encoders_cfg = {
        'dem': {'type': 'raster', 'in_channels': 1, 'embed_dim': 192},
        'sar': {'type': 'sar', 'in_channels': 2, 'embed_dim': 192},
        'weather': {'type': 'timeseries', 'in_dim': 9, 'embed_dim': 192},
        'soil': {'type': 'tabular', 'in_dim': 12, 'embed_dim': 192},
        'crop': {'type': 'categorical', 'num_classes': 20, 'embed_dim': 192},
    }
    modalities = {
        'dem': {'spatial': 16, 'temporal': 1},
        'sar': {'spatial': 16, 'temporal': 1},
        'weather': {'spatial': 1, 'temporal': 1},
        'soil': {'spatial': 1, 'temporal': 1},
        'crop': {'spatial': 16, 'temporal': 1},
    }
    model = MultimodalSelfSupervisedPretrain(
        encoders_cfg, embed_dim=192, num_latents=32, depth=2, num_heads=3,
        modalities=modalities, modality_embed=64, mask_ratio=0.75,
        modality_mask_ratio=0.5)

    inputs = {
        'dem': torch.randn(2, 1, 4, 4),
        'sar': torch.randn(2, 6, 2, 4, 4),
        'weather': torch.randn(2, 12, 9),
        'soil': torch.randn(2, 12),
        'crop': torch.randint(0, 20, (2, 4, 4)),
    }

    # training mode: all five objectives active
    model.train()
    total, losses = model(inputs)
    print('train total', total.item())
    for k, v in losses.items():
        print('  ', k, v.item())

    # eval mode: no masking, objectives still computable
    model.eval()
    total, losses = model(inputs)
    print('eval total', total.item())

    # missing-modality robustness: only a subset of sources available
    partial = {'dem': inputs['dem'], 'weather': inputs['weather']}
    total, losses = model(partial)
    print('partial total', total.item())

    # freeze the modality encoders (frozen-pretrained-encoder comparison)
    model.set_encoder_trainable(False)
    frozen = all(not p.requires_grad for p in model.encoders.parameters())
    print('encoders frozen', frozen)
