"""Only existing non-DINO encoders plus disposable pretraining heads."""
import torch
from torch import nn
from torch.nn import functional as F
from models_yieldsat_image import MaskedConvPyramid,SeriesEncoder,DepthAwareSoil
from models_multimodal_encoder import MaskedTemporalEncoder
from .common import ENCODERS,STREAMS

CHANNELS={'spec':9,'series':12,'weather':4,'dem':1,'terrain':5,'soil':48}


def grid_target(v,m):
    num=F.adaptive_avg_pool2d(v*m,4);den=F.adaptive_avg_pool2d(m,4)
    return (num/den.clamp(min=1e-6)).flatten(2).transpose(1,2),den.flatten(2).transpose(1,2)>0


class KnowledgeEncoders(nn.Module):
    def __init__(self,dim,text_vectors,concepts,rules):
        super().__init__();self.dim=dim;self.concepts=concepts;self.rules=rules
        self.spec_enc=MaskedConvPyramid(9,dim)
        self.series_cell=SeriesEncoder();self.series_enc=MaskedConvPyramid(0,dim,stem_in=32)
        self.weather_enc=MaskedTemporalEncoder(4,dim,time_dim=3,max_len=24)
        self.dem_enc=MaskedConvPyramid(1,dim);self.terrain_enc=MaskedConvPyramid(5,dim)
        self.soil_cell=DepthAwareSoil();self.soil_enc=MaskedConvPyramid(0,dim,stem_in=32)
        textdim=text_vectors.shape[1]
        self.projectors=nn.ModuleDict({k:nn.Linear(dim,textdim) for k in STREAMS})
        self.reconstruction=nn.ModuleDict({k:nn.Linear(dim,c) for k,c in CHANNELS.items()})
        self.register_buffer('prototypes',F.normalize(text_vectors,dim=-1))
        ci={c['id']:i for i,c in enumerate(concepts)}
        self.pairs=[(ci[r['concept_a']],ci[r['concept_b']]) for r in rules]
        delta=torch.stack([self.prototypes[b]-self.prototypes[a] for a,b in self.pairs])
        if (delta.norm(dim=-1)<1e-6).any():raise ValueError('degenerate concept relation target')
        self.register_buffer('relations',F.normalize(delta,dim=-1))

    def encode(self,b):
        tok={};weights={};B,K=b['obs_valid'].shape
        v,m=b['spec'].flatten(0,1),b['spec_mask'].flatten(0,1)
        tok['spec'],_=self.spec_enc(torch.cat([v*m,m],1),m.amax(1,keepdim=True))
        tok['spec']=tok['spec'].reshape(B,K*16,-1)
        weights['spec']=F.adaptive_avg_pool2d(m.amax(1,keepdim=True),4).reshape(B,K*16)
        m=b['series_mask'];cell=self.series_cell(b['series'],m,b['series_days']*m,b['seeding_doy'])
        obs=m.amax(-1).unsqueeze(1)
        tok['series'],_=self.series_enc(cell,obs)
        weights['series']=F.adaptive_avg_pool2d(obs,4).flatten(1)
        m=b['weather_mask'];x=torch.cat([b['weather']*m,m,b['weather_time']],-1)
        tok['weather']=self.weather_enc(x);weights['weather']=m.flatten(1).any(1).float().unsqueeze(1)
        for name in ('dem','terrain','soil'):
            v,m=b[name],b[name+'_mask'];obs=m.amax(1,keepdim=True)
            x=self.soil_cell(v,m) if name=='soil' else torch.cat([v*m,m],1)
            tok[name],_=getattr(self,name+'_enc')(x,obs)
            weights[name]=F.adaptive_avg_pool2d(obs,4).flatten(1)
        return tok,weights

    def encoder_state(self):
        return {k:v.detach().cpu() for k,v in self.state_dict().items() if k.split('.')[0] in ENCODERS}

    def corrupt(self,b,rate):
        b=dict(b)
        for name in CHANNELS:
            key=name+'_mask'
            mask=b[key]
            m=mask*(torch.rand_like(mask)>rate)
            b[key]=m
            b[name]=b[name]*(m.unsqueeze(-1) if name=='series' else m)
        b['series_days']=b['series_days']*b['series_mask']
        return b

    def loss(self,b,objective='relation',mask_rate=.2,ground_weight=1.,relation_weight=1.):
        # Supervised knowledge heads read uncorrupted endpoint sensors. Reconstruction reads masked sensors.
        tok,weights=self.encode(b)
        z={k:F.normalize(self.projectors[k]((v*weights[k][...,None]).sum(1)/weights[k].sum(1,keepdim=True).clamp(min=1e-6)),dim=-1)
           for k,v in tok.items()}
        zero=sum(v.sum()*0 for v in tok.values());ground=[];relation=[];ground_n=rel_n=0
        if objective!='sensor':
            for i,c in enumerate(self.concepts):
                target=b['concept_targets'][:,i];known=torch.isfinite(target)&(weights[c['stream']].sum(1)>0)
                if known.any():
                    logits=(z[c['stream']]*self.prototypes[i]).sum(-1)/.1
                    ground.append(F.binary_cross_entropy_with_logits(logits[known],target[known]));ground_n+=int(known.sum())
        if objective=='relation':
            for j,(a,bb) in enumerate(self.pairs):
                ca,cb=self.concepts[a],self.concepts[bb]
                pa,pb=b['concept_targets'][:,a],b['concept_targets'][:,bb]
                gate=b['rule_targets'][:,j]
                threshold=self.rules[j]['minimum_presence']
                known=torch.isfinite(pa)&torch.isfinite(pb)&torch.isfinite(gate)&(pa>=threshold)&(pb>=threshold)&(gate>0)
                known&=(weights[ca['stream']].sum(1)>0)&(weights[cb['stream']].sum(1)>0)
                if known.any():
                    delta=F.normalize(z[cb['stream']][known]-z[ca['stream']][known],dim=-1)
                    w=(pa[known]*pb[known]*gate[known]).detach()
                    relation.append(((1-(delta*self.relations[j]).sum(-1))*w).sum()/w.sum().clamp(min=1e-6));rel_n+=int(known.sum())
        corrupt=self.corrupt(b,mask_rate);masked,_=self.encode(corrupt);recon=[]
        for name in CHANNELS:
            m=b[name+'_mask']-corrupt[name+'_mask'];v=b[name]
            if name=='weather':
                target=(v*m).sum(1)/m.sum(1).clamp(min=1);valid=m.sum(1)>0
                target,valid=target[:,None],valid[:,None]
            elif name=='series':
                bandmask=m.unsqueeze(-1)*b['series_band_mask']
                count=bandmask.sum(-2).permute(0,3,1,2)
                val=(v*bandmask).sum(-2).permute(0,3,1,2)/count.clamp(min=1)
                target,valid=grid_target(val,count)
            elif name=='spec':
                target,valid=grid_target(v.flatten(0,1),m.flatten(0,1));target=target.reshape(v.shape[0],-1,9);valid=valid.reshape(v.shape[0],-1,9)
            else:target,valid=grid_target(v,m)
            pred=self.reconstruction[name](masked[name])
            if valid.any():recon.append((pred-target).square()[valid].mean())
        ls=torch.stack(recon).mean() if recon else zero
        lg=torch.stack(ground).mean() if ground else zero;lr=torch.stack(relation).mean() if relation else zero
        total=ls+ground_weight*lg+relation_weight*lr
        metrics={'sensor_loss':float(ls.detach()),'ground_loss':float(lg.detach()),'relation_loss':float(lr.detach()),
                 'ground_terms':ground_n,'relation_terms':rel_n,
                 'representation_variance':{k:float(v.detach().var(dim=1,unbiased=False).mean()) for k,v in tok.items()}}
        return total,metrics
