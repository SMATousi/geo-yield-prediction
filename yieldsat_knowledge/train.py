"""Standalone encoder-only relational pretraining; compatible encoder export for image fine-tuning."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler

from .common import approved,digest,file_hash,jsonl,library,read,safe_child,write
from .model import KnowledgeEncoders
from .pilot import validate_response


def verify_prepared(root):
    root=Path(root);meta=read(root/'manifest.json')
    for filename,key in [('evidence.jsonl','evidence_hash'),('normalizer.json','normalizer_hash')]:
        if file_hash(root/filename)!=meta[key]:raise ValueError('prepared artifact hash mismatch: '+filename)
    rows=jsonl(root/'evidence.jsonl')
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('duplicate prepared supports')
    for r in rows:
        raw=dict(r);claimed=raw.pop('evidence_hash')
        if digest(raw)!=claimed or file_hash(safe_child(root,r['tensor']))!=r['tensor_hash']:raise ValueError('stale sensor/evidence cache')
        if r['season_id'] not in meta['split']['partitions']['train']:raise ValueError('non-training sample in prepared data')
    return meta,rows


def accepted_cache(root,rows,lib,cache,policy_path,audit_path,smoke=False):
    policy=read(policy_path);audit=read(audit_path)
    if not audit.get('passed') or policy['report_hash']!=digest(audit):raise ValueError('annotation audit failed or stale')
    if not approved(policy.get('review',{})):raise ValueError('annotation policy needs expert review')
    if policy['profile_hash']!=audit['profile_hash'] or policy.get('reliability')!=1.0:raise ValueError('unsupported annotation policy')
    required={kind+':'+c['id'] for kind in ('concepts','rules') for c in lib[kind]}
    if not required <= set(audit.get('audited_ids',[])):raise ValueError('annotation audit does not cover every active concept/rule')
    by={r['sample_id']:r for r in rows};annotations={}
    for path in sorted(Path(cache).glob('*.json')):
        if path.name.endswith('.request.json'):continue
        ann=read(path)
        if ann['sample_id'] not in by:raise ValueError('annotation sample is outside prepared training set')
        if ann['sample_id'] in annotations:raise ValueError('duplicate annotations; select one teacher profile')
        if ann['profile']['library_hash']!=digest(lib) or digest(ann['profile'])!=policy['profile_hash']:raise ValueError('stale library/teacher policy')
        ev=by[ann['sample_id']]
        if ann['evidence_hash']!=ev['evidence_hash']:raise ValueError('stale evidence annotation')
        if ann['request_hash']!=digest({'profile':ann['profile'],'evidence_hash':ann['evidence_hash']}):raise ValueError('request hash mismatch')
        validate_response(ann['response'],lib,ev,ann['profile']['vision'])
        annotations[ann['sample_id']]=ann['response']
    if not annotations:raise ValueError('no accepted annotations')
    return annotations,{'policy':policy,'audit':audit,'cache_hash':digest(annotations)}


class PreparedDataset(Dataset):
    def __init__(self,root,rows,lib,annotations):
        self.root=Path(root);self.rows=rows;self.lib=lib;self.annotations=annotations
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        with np.load(safe_child(self.root,row['tensor']),allow_pickle=False) as z:
            if any(k.startswith('target') for k in z.files):raise ValueError('yield field in pretraining tensor')
            b={k:torch.as_tensor(z[k].copy()).float() for k in z.files}
        ann=self.annotations.get(row['sample_id'],{'concepts':[],'rules':[]})
        for kind,key,out in [('concepts','presence_score','concept_targets'),('rules','applicability_score','rule_targets')]:
            values={r['id']:r[key] for r in ann[kind]}
            b[out]=torch.tensor([values.get(c['id']) if values.get(c['id']) is not None else float('nan') for c in self.lib[kind]])
        return b


def run(a):
    torch.manual_seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(a.cpu_threads)
    lib=library(a.library,require_approved=a.objective!='sensor')
    root=Path(a.prepared);meta,rows=verify_prepared(root)
    text_root=Path(a.text_cache);textmeta=read(text_root/'manifest.json')
    if textmeta['library_hash']!=digest(lib) or file_hash(text_root/'vectors.npz')!=textmeta['vectors_hash']:raise ValueError('stale text cache')
    if textmeta['mock'] and not a.smoke_test:raise ValueError('mock embeddings require --smoke-test; not a production run')
    with np.load(text_root/'vectors.npz') as z:
        vec=dict(zip(z['ids'].tolist(),z['vectors']));vectors=torch.tensor(np.stack([vec[c['id']] for c in lib['concepts']]))
    annotations={};policies={}
    if a.objective!='sensor':
        if not all((a.annotation_cache,a.annotation_policy,a.annotation_audit)):raise ValueError('grounding needs reviewed annotation artifacts')
        annotations,policies=accepted_cache(root,rows,lib,a.annotation_cache,a.annotation_policy,a.annotation_audit)
    if a.objective=='relation':
        if not a.geometry_policy or not a.geometry_audit:raise ValueError('relation distillation requires geometry validation')
        gp,ga=read(a.geometry_policy),read(a.geometry_audit)
        if (not approved(gp.get('review',{})) or not ga.get('passed') or gp['report_hash']!=digest(ga)
            or gp['library_hash']!=digest(lib) or gp['text_hash']!=digest(textmeta)
            or ga['library_hash']!=digest(lib) or ga['text_hash']!=digest(textmeta)):
            raise ValueError('geometry review missing/stale/failed')
        if ga.get('mock') and not a.smoke_test:raise ValueError('mock geometry audit')
        policies['geometry_policy']=gp;policies['geometry_audit']=ga
    model=KnowledgeEncoders(a.embed_dim,vectors,lib['concepts'],lib['rules']).to(a.device)
    if a.control=='shuffle_relations':
        with torch.no_grad():model.relations.copy_(model.relations.roll(1,0))
    elif a.control=='constant_text':
        with torch.no_grad():
            model.prototypes.fill_(1/model.prototypes.shape[1]**.5)
            model.relations.fill_(1/model.relations.shape[1]**.5)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.05)
    ds=PreparedDataset(root,rows,lib,annotations)
    # Equal country -> crop -> field -> season -> tile mass, without treating
    # repeated cells from a coarse weather/soil footprint as independent votes.
    paths=[(r['country'],r['crop'],r['physical_field_id'],r['season_id']) for r in rows]
    children={};leaves={}
    for path in paths:
        for depth in range(4):children.setdefault(path[:depth],set()).add(path[depth])
        leaves[path]=leaves.get(path,0)+1
    w=[1/(leaves[path]*np.prod([len(children[path[:d]]) for d in range(4)])) for path in paths]
    gen=torch.Generator().manual_seed(a.seed)
    sampler=WeightedRandomSampler(w,a.steps_per_epoch*a.batch_size,replacement=True,generator=gen)
    dl=DataLoader(ds,batch_size=a.batch_size,sampler=sampler,num_workers=a.num_workers)
    config={k:v for k,v in vars(a).items() if k not in ('resume','output_dir','epochs')}
    provenance={'config':config,'prepared_hash':digest(meta),'library_hash':digest(lib),
                'text_hash':digest(textmeta),'policies':policies,'encoder_layout':{k:list(v.shape) for k,v in model.encoder_state().items()},
                'dino':'not instantiated; no DINO or fusion tensors are optimized/exported'}
    out=Path(a.output_dir)
    if out.exists() and not a.resume:raise ValueError('output exists; use --resume or a new directory')
    out.mkdir(parents=True,exist_ok=True)
    start=0;history=[]
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=False)
        if state['provenance']!=provenance:raise ValueError('resume inputs/config mismatch')
        model.load_state_dict(state['state']);opt.load_state_dict(state['optimizer']);gen.set_state(state['sampler_rng'])
        torch.set_rng_state(state['torch_rng'])
        if a.device.startswith('cuda') and state.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
        start=state['epoch']+1;history=state['history']
    write(out/'provenance.json',provenance)
    for epoch in range(start,a.epochs):
        model.train();losses=[];ground=relations=0;metrics=[]
        for batch in dl:
            b={k:v.to(a.device) for k,v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(a.device).type,dtype=torch.bfloat16,enabled=a.amp and a.device.startswith('cuda')):
                loss,m=model.loss(b,a.objective,a.mask_rate,a.ground_weight,a.relation_weight)
            if not torch.isfinite(loss):raise ValueError('nonfinite pretraining loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            losses.append(float(loss.detach()));ground+=m['ground_terms'];relations+=m['relation_terms'];metrics.append(m)
        if a.objective!='sensor' and not ground:raise ValueError('no grounding terms observed; inspect annotation coverage')
        if a.objective=='relation' and not relations:raise ValueError('no eligible relations observed; this is not relationship training')
        row={'epoch':epoch,'loss':float(np.mean(losses)),'ground_terms':ground,'relation_terms':relations,
             'sensor_loss':float(np.mean([m['sensor_loss'] for m in metrics])),
             'ground_loss':float(np.mean([m['ground_loss'] for m in metrics])),
             'relation_loss':float(np.mean([m['relation_loss'] for m in metrics])),
             'representation_variance':{k:float(np.mean([m['representation_variance'][k] for m in metrics])) for k in metrics[0]['representation_variance']}}
        history.append(row);print(json.dumps(row),flush=True)
        state={'state':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'history':history,
               'sampler_rng':gen.get_state(),'torch_rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else None,'provenance':provenance}
        torch.save(state,out/'checkpoint_last.pth.partial');(out/'checkpoint_last.pth.partial').replace(out/'checkpoint_last.pth')
        export={'model':model.encoder_state(),'knowledge':provenance,'normalizer':read(root/'normalizer.json'),
                'split_hash':meta['split_hash'],'image_manifest_hash':meta['image_manifest_hash'],'embed_dim':a.embed_dim,'k_obs':meta['k_obs'],'window_days':meta['window_days'],
                'smoke_test':a.smoke_test,'epoch':epoch}
        torch.save(export,out/'encoders.pth.partial');(out/'encoders.pth.partial').replace(out/'encoders.pth')
        write(out/'report.json',{'history':history,'tiles':len(rows),'objective':a.objective,'smoke_test':a.smoke_test,'provenance':provenance})


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('prepared','library','text-cache','output-dir'):p.add_argument('--'+k,required=True)
    for k in ('annotation-cache','annotation-policy','annotation-audit','geometry-policy','geometry-audit','resume'):p.add_argument('--'+k)
    p.add_argument('--objective',choices=['sensor','grounding','relation'],default='relation')
    p.add_argument('--control',choices=['none','shuffle_relations','constant_text'],default='none')
    p.add_argument('--embed-dim',type=int,default=192);p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--steps-per-epoch',type=int,default=100);p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--num-workers',type=int,default=0);p.add_argument('--cpu-threads',type=int,default=2)
    p.add_argument('--device',default='cuda');p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--mask-rate',type=float,default=.2);p.add_argument('--ground-weight',type=float,default=1.)
    p.add_argument('--relation-weight',type=float,default=1.);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--amp',action='store_true');p.add_argument('--smoke-test',action='store_true')
    return p


def main():
    p=parser();a=p.parse_args()
    if min(a.epochs,a.steps_per_epoch,a.batch_size,a.cpu_threads)<1 or not 0<a.mask_rate<1 or min(a.ground_weight,a.relation_weight)<0:p.error('invalid training parameters')
    run(a)

if __name__=='__main__':main()
