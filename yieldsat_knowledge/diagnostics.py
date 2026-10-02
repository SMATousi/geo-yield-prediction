"""Yield-free checkpoint diagnostics; training-support diagnostics are not held-out transfer evidence."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .common import digest,library,write
from .model import KnowledgeEncoders
from .train import verify_prepared,accepted_cache,PreparedDataset


def pooled(model,b):
    tokens,weights=model.encode(b)
    return {k:F.normalize(model.projectors[k]((v*weights[k][...,None]).sum(1)/weights[k].sum(1,keepdim=True).clamp(min=1e-6)),dim=-1)
            for k,v in tokens.items()},weights


def run(a):
    torch.set_num_threads(2)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);prov=state['provenance'];cfg=prov['config']
    meta,rows=verify_prepared(a.prepared);lib=library(a.library)
    if digest(meta)!=prov['prepared_hash'] or digest(lib)!=prov['library_hash']:raise ValueError('diagnostic inputs mismatch')
    ann={}
    if cfg['objective']!='sensor':
        ann,policy=accepted_cache(a.prepared,rows,lib,a.annotation_cache,a.annotation_policy,a.annotation_audit)
        if policy['cache_hash']!=prov['policies']['cache_hash']:raise ValueError('diagnostic annotation cache differs from training')
    model=KnowledgeEncoders(cfg['embed_dim'],state['state']['prototypes'],lib['concepts'],lib['rules']).to(a.device)
    model.load_state_dict(state['state']);model.eval()
    ds=PreparedDataset(a.prepared,rows[:a.max_samples],lib,ann)
    vectors={};sensitivity={};ground={c['id']:[] for c in lib['concepts']};counts={}
    records=[]
    with torch.no_grad():
        for i in range(len(ds)):
            b={k:v.unsqueeze(0).to(a.device) for k,v in ds[i].items()};z,w=pooled(model,b)
            perturbed=dict(b)
            for stream in z:
                vectors.setdefault(stream,[]).append(z[stream].cpu().numpy()[0])
                # Complete modality removal tests whether the representation depends on its inputs.
                perturbed[stream]=torch.zeros_like(b[stream]);perturbed[stream+'_mask']=torch.zeros_like(b[stream+'_mask'])
            zp,_=pooled(model,perturbed)
            for stream in z:sensitivity.setdefault(stream,[]).append(float((z[stream]-zp[stream]).norm()))
            for j,c in enumerate(lib['concepts']):
                t=b['concept_targets'][0,j]
                if torch.isfinite(t) and w[c['stream']].sum()>0:
                    pred=torch.sigmoid((z[c['stream']][0]*model.prototypes[j]).sum()/.1)
                    ground[c['id']].append(float((pred-t)**2))
            for j,(ia,ib) in enumerate(model.pairs):
                rule=lib['rules'][j];ca,cb=lib['concepts'][ia],lib['concepts'][ib]
                pa,pb=b['concept_targets'][0,ia],b['concept_targets'][0,ib];gate=b['rule_targets'][0,j]
                known=bool(torch.isfinite(pa)&torch.isfinite(pb)&torch.isfinite(gate))
                eligible=known and min(float(pa),float(pb))>=rule['minimum_presence'] and float(gate)>0 and bool(w[ca['stream']].sum()>0 and w[cb['stream']].sum()>0)
                key=(rows[i]['country'],rows[i]['crop'],rule['id']);entry=counts.setdefault(key,{'supports':0,'unknown':0,'eligible':0,'scores':[]})
                entry['supports']+=1;entry['unknown']+=int(not known);entry['eligible']+=int(eligible)
                if eligible:
                    delta=F.normalize(z[cb['stream']]-z[ca['stream']],dim=-1)
                    entry['scores'].append(float((delta*model.relations[j]).sum()))
    for key,v in counts.items():
        scores=v.pop('scores');records.append(dict(country=key[0],crop=key[1],rule=key[2],**v,mean_cosine=float(np.mean(scores)) if scores else None))
    write(a.output,{'scope':'prepared TRAIN supports only; not held-out evaluation',
        'checkpoint_epoch':state['epoch'],'samples':len(ds),'smoke_test':cfg['smoke_test'],
        'by_rule_country_crop':records,'concept_brier':{k:float(np.mean(v)) if v else None for k,v in ground.items()},
        'projected_embedding_variance':{k:float(np.var(v,axis=0).mean()) for k,v in vectors.items()},
        'mean_modality_removal_distance':{k:float(np.mean(v)) for k,v in sensitivity.items()}})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','prepared','library','output'):p.add_argument('--'+key,required=True)
    for key in ('annotation-cache','annotation-policy','annotation-audit'):p.add_argument('--'+key)
    p.add_argument('--max-samples',type=int,default=100);p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if a.max_samples<1:p.error('max-samples must be positive')
    run(a)

if __name__=='__main__':main()
