"""Pretrained features for TabM round TM-2 (spec/yieldsat-tabm.md TM-03).

For one DEV fold, the features come from the sensor checkpoint of the fold's
pretraining unit (splits/pretrain_units_dev_s0.json), never from a unit that
saw the fold's test seasons:

  emb  frozen per-stream summary embeddings of the point encoders, 5 x D
       (S2, weather, DEM, terrain, soil), D = 128
  cpt  12 concept scores cos(z_stream, t_concept) and 6 rule alignments
       cos(normalize(z_B - z_A), r_AB) from the unit's knowledge heads
       (A3: real knowledge, A7: random-targets control)

Every cell of the pair is encoded in eval mode (no masking, no dropout) with
the unit's own normalizer and input settings. Only the small concept scores
are cached (<artifact_root>/tabular/pretrained/<arm>/<unit>/<country>_<crop>_cpt.npy);
embeddings (~3 GB per unit for Uruguay) are recomputed per fold, since caching
them for every arm x unit x pair would need ~240 GB on the shared volume.
"""
import os
from pathlib import Path

import numpy as np
import torch

UNIT_INDEX = 'pretrain_units_dev_s0'


def unit_for(artifact_root, split_name, index=UNIT_INDEX):
    """The fold's pretraining unit; refuses a unit that trained on the fold's test seasons."""
    import json
    splits = Path(artifact_root) / 'splits'
    idx = json.loads((splits / '{}.json'.format(index)).read_text())
    unit = idx['fold_to_unit'][split_name]
    test = set(json.loads((splits / '{}.json'.format(split_name)).read_text())['partitions']['test'])
    parts = json.loads((splits / '{}.json'.format(unit)).read_text())['partitions']
    leak = test & (set(parts['train']) | set(parts['val']))
    if leak:
        raise ValueError('unit {} trained on {} test seasons of {}'.format(unit, len(leak), split_name))
    return unit


@torch.no_grad()
def encode_pair(run_dir, pair, country, artifact_root, device, kinds=('emb', 'cpt'), batch=8192):
    """Features for every row of ``pair`` (yieldsat_tabular.build_pair output), in
    pair row order. Returns {'emb': (N, 5*D) float16, 'cpt': (N, 18) float32}."""
    from models_yieldsat import pack_inputs
    from yieldsat_knowledge_point import KnowledgePointPretrainer
    from yieldsat_pretrain_diagnostics import Run
    run = Run(run_dir, str(device), artifact_root=artifact_root, skip_source_check=True)
    seasons = [dict(f, country=country) for f in pair['seasons']]
    ds = run.dataset(seasons, None)
    pt = run.build(ds.layout, trained=True)
    model = pt.model.eval()
    pos = {int(r): i for i, r in enumerate(pair['cache_rows'])}
    order = np.array([pos[int(r)] for r in ds.row])                 # dataset item -> pair row
    out = {}
    knowledge = isinstance(pt, KnowledgePointPretrainer)
    for a in range(0, len(ds), batch):
        b = ds.get_batch(np.arange(a, min(len(ds), a + batch)))
        b = {k: ({kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v.to(device))
             for k, v in b.items()}
        sel = order[a:a + batch]
        if 'emb' in kinds:
            emb, _ = model.encoders.forward_with_missing(pack_inputs(b, model.layout),
                                                         available={n: b['available'][n] for n in model.streams},
                                                         apply_dropout=False)
            e = torch.cat([emb[n][:, 0].float() for n in model.streams], 1).cpu().numpy().astype(np.float16)
            out.setdefault('emb', np.zeros((len(order), e.shape[1]), np.float16))[sel] = e
        if 'cpt' in kinds and knowledge:
            z, _ = pt.stream_embeddings(b)
            proto = pt.prototypes()
            rel = pt.relations(proto)
            scores = [(z[s] * proto[i]).sum(-1) for i, s in enumerate(pt.concept_stream)]
            align = []
            for j, (ia, ib) in enumerate(pt.pairs):
                d = torch.nn.functional.normalize(z[pt.concept_stream[ib]] - z[pt.concept_stream[ia]], dim=-1)
                align.append((d * rel[j]).sum(-1))
            c = torch.stack(scores + align, 1).float().cpu().numpy()
            out.setdefault('cpt', np.zeros((len(order), c.shape[1]), np.float32))[sel] = c
    return out


def pretrained_features(artifact_root, pretrain_root, arm, split_name, pair, country, crop, kind, device):
    """(N, F) float32 extra columns for TabM: kind 'emb', 'cpt' or 'all'."""
    unit = unit_for(artifact_root, split_name)
    run_dir = Path(pretrain_root) / 'pretrain' / arm / unit
    cpt_cache = Path(artifact_root) / 'tabular' / 'pretrained' / arm / unit / '{}_{}_cpt.npy'.format(country, crop)
    feats = {}
    if kind in ('cpt', 'all') and cpt_cache.exists():
        feats['cpt'] = np.load(cpt_cache)
    need = tuple(k for k in (('emb',) if kind in ('emb', 'all') else ()) + (('cpt',) if kind in ('cpt', 'all') else ())
                 if k not in feats)
    if need:
        feats.update(encode_pair(run_dir, pair, country, artifact_root, device, kinds=need))
        if 'cpt' in need and 'cpt' in feats:
            cpt_cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cpt_cache.with_name('{}.{}-{}.partial.npy'.format(cpt_cache.stem, os.uname().nodename, os.getpid()))
            np.save(tmp, feats['cpt'])
            os.replace(tmp, cpt_cache)
    parts = []
    if kind in ('emb', 'all'):
        parts.append(feats['emb'].astype(np.float32))
    if kind in ('cpt', 'all'):
        if 'cpt' not in feats:
            raise ValueError('arm {} has no knowledge heads (concept scores need A3/A7)'.format(arm))
        parts.append(feats['cpt'].astype(np.float32))
    return np.concatenate(parts, 1), unit
