"""Collate field-level multimodal samples with heterogeneous availability.

The field-level datasets (``FieldYieldDataset``, ``HeterogeneousModalityLoader``)
return per-sample dicts whose modality keys are present only when that layer
exists for the field-year, plus a per-sample ``available`` mask. A default
``torch.utils.data.DataLoader`` collate would fail on such dicts because
different samples may have different modality keys. This module provides a
collate function that stacks the present modalities into batch tensors and
produces a batch-level per-modality availability dict, so the missing-modality
robustness requirement (a batch may contain fields with different subsets of
sources) is realised end to end: the model's ``forward_with_missing`` consumes
the per-sample availability tensors and substitutes a learned missing-modality
token for the absent samples.
"""

import torch


def collate_field_samples(batch):
    """Collate a list of field-level sample dicts into a model-ready batch.

    Each sample is a dict with modality-name -> tensor entries (present only
    when that layer exists for the field-year), plus ``yield`` (dense target),
    ``available`` (per-layer availability) and ``field`` (name). Returns a dict
    with:

    * ``inputs``: modality-name -> stacked ``(B, ...)`` tensor. A modality
      present in only some samples is zero-filled for the absent samples and
      flagged 0 in ``available``, so the batch tensor is dense and the model
      can gate on the availability mask.
    * ``available``: modality-name -> ``(B,)`` float tensor (1 = present,
      0 = absent).
    * ``yield``: stacked ``(B, ...)`` dense yield targets (if present).
    * ``field``: list of field names (if present).

    The returned ``inputs`` / ``available`` pair is consumed directly by
    ``MultiModalEncoder.forward_with_missing`` (see models_multimodal_encoder.py),
    which encodes the present samples and substitutes a learned missing-modality
    token for the absent ones.
    """
    modality_names = set()
    for sample in batch:
        modality_names.update(
            k for k in sample if k not in ('yield', 'available', 'field'))
    modality_names = sorted(modality_names)

    inputs = {}
    available = {}
    for name in modality_names:
        present = [s[name] for s in batch if name in s]
        if not present:
            continue
        ref = torch.as_tensor(present[0])
        stacked = []
        avail = []
        for s in batch:
            if name in s:
                stacked.append(torch.as_tensor(s[name]))
                avail.append(1.0)
            else:
                stacked.append(torch.zeros_like(ref))
                avail.append(0.0)
        inputs[name] = torch.stack(stacked)
        available[name] = torch.tensor(avail, dtype=torch.float32)

    out = {'inputs': inputs, 'available': available}
    if 'yield' in batch[0]:
        out['yield'] = torch.stack([torch.as_tensor(s['yield']) for s in batch])
    if 'field' in batch[0]:
        out['field'] = [s['field'] for s in batch]
    return out


if __name__ == "__main__":
    # Two field-year samples with different available modalities: sample 0 has
    # DEM + SAR + weather, sample 1 lacks SAR (missing modality).
    sample0 = {
        'dem': torch.randn(1, 4, 4),
        'sar': torch.randn(2, 4, 4),
        'weather': torch.randn(12, 9),
        'yield': torch.randn(1, 8, 8),
        'available': torch.tensor([1.0, 1.0, 1.0]),
        'field': 'field_2021',
    }
    sample1 = {
        'dem': torch.randn(1, 4, 4),
        'weather': torch.randn(12, 9),
        'yield': torch.randn(1, 8, 8),
        'available': torch.tensor([1.0, 0.0, 1.0]),
        'field': 'field_2022',
    }
    batch = collate_field_samples([sample0, sample1])
    print('inputs:', {k: tuple(v.shape) for k, v in batch['inputs'].items()})
    print('available:', {k: v.tolist() for k, v in batch['available'].items()})
    print('yield:', tuple(batch['yield'].shape))
    print('field:', batch['field'])
