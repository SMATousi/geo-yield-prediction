"""Freeze concept/relation text references using a pinned, frozen CLIP text encoder."""
import argparse
import re
from pathlib import Path
import numpy as np
from .common import digest,file_hash,library,write


DEFAULT_MODEL = 'openai/clip-vit-base-patch32'


def build(lib, output, model=DEFAULT_MODEL, revision=None, mock=False, device='cpu'):
    output=Path(output)
    if output.exists():raise ValueError('output already exists')
    texts={c['id']:c['description'] for c in lib['concepts']}
    texts.update({r['id']:r['statement'] for r in lib['rules']})
    ids=list(texts)
    max_length=None
    pooling='synthetic random vectors; L2 normalization'
    if mock:
        vectors=np.random.default_rng(7).normal(size=(len(ids),32)).astype(np.float32)
    else:
        if not model or not revision or not re.fullmatch('[0-9a-f]{40}',revision):
            raise ValueError('supply --model and an immutable 40-character HF commit --revision')
        import torch
        from transformers import CLIPTextModelWithProjection,AutoTokenizer
        tok=AutoTokenizer.from_pretrained(model,revision=revision)
        enc=CLIPTextModelWithProjection.from_pretrained(model,revision=revision).to(device).eval()
        for param in enc.parameters():param.requires_grad_(False)
        max_length=enc.config.max_position_embeddings
        pooling='CLIP pooled EOS output + pretrained text projection; L2 normalization'
        vectors=[]
        with torch.no_grad():
            for start in range(0,len(ids),16):
                keys=ids[start:start+16]
                # Never drop a relationship qualification to fit CLIP's context.
                batch=tok([texts[k] for k in keys],padding=True,truncation=False,return_tensors='pt')
                lengths=batch['attention_mask'].sum(1)
                too_long=[key for key,n in zip(keys,lengths.tolist()) if n>max_length]
                if too_long:
                    raise ValueError(f'CLIP context limit is {max_length} tokens including special tokens; shorten and re-review: {too_long}')
                batch=batch.to(device)
                vectors.append(enc(**batch).text_embeds.cpu().numpy())
        vectors=np.concatenate(vectors)
    norms=np.linalg.norm(vectors,axis=1,keepdims=True)
    if not np.isfinite(vectors).all() or (norms<1e-8).any():raise ValueError('invalid text embeddings')
    vectors=(vectors/norms).astype(np.float32)
    output.mkdir(parents=True);np.savez(output/'vectors.npz',ids=np.array(ids),vectors=vectors)
    write(output/'manifest.json',{'schema_version':1,'library_hash':digest(lib),'texts':texts,
          'model':model,'revision':revision,'mock':mock,'encoder_class':'synthetic' if mock else 'CLIPTextModelWithProjection',
          'pooling':pooling,'max_length':max_length,'truncation':False,
          'embedding_dim':int(vectors.shape[1]),
          'vectors_hash':file_hash(output/'vectors.npz')})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--model',default=DEFAULT_MODEL);p.add_argument('--revision');p.add_argument('--device',default='cpu')
    p.add_argument('--mock',action='store_true',help='synthetic smoke only; production trainer refuses')
    a=p.parse_args();build(library(a.library),a.output_dir,a.model,a.revision,a.mock,a.device)

if __name__=='__main__':main()
