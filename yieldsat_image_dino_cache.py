"""Precompute frozen DINOv3 SAT patch tokens for every (tile, RGB slot) (YI-02).

Writes ``<out>/<revision[:8]>_<prep-hash>/<Country>.h5`` with datasets
``features`` (M, 16, hidden) float16, ``patch_index``, ``slot``, ``coverage``
and ``clip_fraction``, plus ``manifest.json``. Every slot with at least one
dated cell carrying finite RGB is cached, so any observation selection
(training or evaluation) can be served. Features depend only on the inputs,
checkpoint and preprocessing (no labels or splits). Countries are published
atomically.

    python yieldsat_image_dino_cache.py --image-root /data/YieldSAT/YieldSAT-Image \
        --device cuda            # needs HF_TOKEN with access to the gated checkpoint
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from dataset.yieldsat_image_dataset import RGB_IDX, TileReader, load_tile_table
from dataset.yieldsat_schema import COUNTRIES
from models_yieldsat_image import (
    DINO_REPO, DINO_REVISION, FrozenDino, StubDino, preprocessing_hash, preprocessing_spec,
    rgb_to_dino_input,
)


def cache_tag(revision, prep_hash):
    return '{}_{}'.format(revision[:8], prep_hash)


def build_country(backbone, image_root, country, out_dir, device, batch=64, tag_attrs=None):
    reader = TileReader(image_root)
    tiles = load_tile_table(image_root, [country])
    entries = []                                   # (patch_index, slot, coverage)
    for t in tiles:
        d = reader.get(country, t['patch_index'])
        ok = np.isfinite(d['temporal'][..., RGB_IDX]).all(-1) & np.isfinite(d['times'])
        for s in range(ok.shape[-1]):
            n = int(ok[..., s].sum())
            if n:
                entries.append((t['patch_index'], s, n / ok[..., s].size))
    hidden = backbone.hidden_size
    final = Path(out_dir) / '{}.h5'.format(country)
    tmp = Path(out_dir) / '{}.h5.partial'.format(country)
    with h5py.File(tmp, 'w') as f:
        feats = f.create_dataset('features', (len(entries), 16, hidden), dtype='f2',
                                 chunks=(1, 16, hidden))
        f['patch_index'] = np.array([e[0] for e in entries], np.int64)
        f['slot'] = np.array([e[1] for e in entries], np.int16)
        f['coverage'] = np.array([e[2] for e in entries], np.float32)
        clip = np.zeros(len(entries), np.float32)
        for start in range(0, len(entries), batch):
            chunk = entries[start:start + batch]
            rgb, valid = [], []
            for pi, s, _ in chunk:
                d = reader.get(country, pi)
                rgb.append(d['temporal'][:, :, s, RGB_IDX])
                valid.append(np.isfinite(d['temporal'][:, :, s, RGB_IDX]).all(-1)
                             & np.isfinite(d['times'][:, :, s]))
            x, cf = rgb_to_dino_input(np.stack(rgb), np.stack(valid))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == 'cuda'):
                tok = backbone(torch.from_numpy(x).to(device)).float()
            if tok.shape[1:] != (16, hidden):
                raise ValueError('unexpected token shape {} (expected 16 patch tokens)'.format(
                    tuple(tok.shape)))
            feats[start:start + len(chunk)] = tok.cpu().numpy().astype(np.float16)
            clip[start:start + len(chunk)] = cf
        f['clip_fraction'] = clip
        for k, v in (tag_attrs or {}).items():
            f.attrs[k] = v
    os.replace(tmp, final)
    return {'country': country, 'tiles': len(tiles), 'entries': len(entries),
            'mean_clip_fraction': float(clip.mean()) if len(clip) else 0.0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image-root', required=True)
    p.add_argument('--out', default=None, help='default <image-root>/dino_cache')
    p.add_argument('--countries', nargs='+', default=list(COUNTRIES), choices=COUNTRIES)
    p.add_argument('--model', default=DINO_REPO, help='HF repo id or local checkpoint directory')
    p.add_argument('--revision', default=DINO_REVISION)
    p.add_argument('--hf-cache', default=None, help='HF download cache (e.g. on the PVC)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--stub', action='store_true', help='stub backbone (tests only)')
    a = p.parse_args()
    device = torch.device(a.device)
    prep = preprocessing_spec()
    prep_hash = preprocessing_hash(prep)
    revision = 'stub0000' if a.stub else a.revision
    if a.stub:
        backbone = StubDino().to(device)
    else:
        backbone = FrozenDino.from_pretrained(a.model, revision=a.revision,
                                              token=os.environ.get('HF_TOKEN'),
                                              cache_dir=a.hf_cache).to(device)
    out = Path(a.out or Path(a.image_root) / 'dino_cache') / cache_tag(revision, prep_hash)
    out.mkdir(parents=True, exist_ok=True)
    attrs = {'model': a.model if not a.stub else 'stub', 'revision': revision,
             'prep_hash': prep_hash, 'prep_json': json.dumps(prep), 'hidden_size': backbone.hidden_size}
    reports = [build_country(backbone, a.image_root, c, out, device, a.batch, attrs) for c in a.countries]
    manifest = dict(attrs, countries=a.countries, reports=reports)
    tmp = out / 'manifest.json.partial'
    tmp.write_text(json.dumps(manifest, indent=1))
    os.replace(tmp, out / 'manifest.json')
    print(json.dumps(reports))
    print('COMPLETE', out)


if __name__ == '__main__':
    main()
