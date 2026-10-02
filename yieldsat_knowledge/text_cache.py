"""Freeze concept/relation text references using a pinned local/HF text encoder."""
import argparse
import re
from pathlib import Path
import numpy as np
from .common import digest,file_hash,library,write


def build(lib, output, model=None, revision=None, mock=False, device='cpu'):
    output=Path(output)
    if output.exists():raise ValueError('output already exists')
    texts={c['id']:c['description'] for c in lib['concepts']}
    texts.update({r['id']:r['statement'] for r in lib['rules']})
    ids=list(texts)
    if mock:
        vectors=np.random.default_rng(7).normal(size=(len(ids),32)).astype(np.float32)
    else:
        if not model or not revision or not re.fullmatch('[0-9a-f]{40}',revision):
            raise ValueError('supply --model and an immutable 40-character HF commit --revision')
        import torch
        from transformers import AutoModel,AutoTokenizer
        tok=AutoTokenizer.from_pretrained(model,revision=revision)
        enc=AutoModel.from_pretrained(model,revision=revision).to(device).eval()
        for param in enc.parameters():param.requires_grad_(False)
        vectors=[]
        with torch.no_grad():
            for start in range(0,len(ids),16):
                batch=tok([texts[k] for k in ids[start:start+16]],padding=True,truncation=True,max_length=256,return_tensors='pt').to(device)
                h=enc(**batch).last_hidden_state
                m=batch['attention_mask'][...,None]
                vectors.append(((h*m).sum(1)/m.sum(1).clamp(min=1)).cpu().numpy())
        vectors=np.concatenate(vectors)
    norms=np.linalg.norm(vectors,axis=1,keepdims=True)
    if not np.isfinite(vectors).all() or (norms<1e-8).any():raise ValueError('invalid text embeddings')
    vectors=(vectors/norms).astype(np.float32)
    output.mkdir(parents=True);np.savez(output/'vectors.npz',ids=np.array(ids),vectors=vectors)
    write(output/'manifest.json',{'schema_version':1,'library_hash':digest(lib),'texts':texts,
          'model':model,'revision':revision,'mock':mock,'pooling':'attention-mask mean; L2 normalization; max_length=256',
          'vectors_hash':file_hash(output/'vectors.npz')})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--model');p.add_argument('--revision');p.add_argument('--device',default='cpu')
    p.add_argument('--mock',action='store_true',help='synthetic smoke only; production trainer refuses')
    a=p.parse_args();build(library(a.library),a.output_dir,a.model,a.revision,a.mock,a.device)

if __name__=='__main__':main()
