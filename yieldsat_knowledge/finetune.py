"""Use reviewed knowledge encoders with the existing image fine-tuning workflow.

Process-local adapters preserve pretrained input statistics and strict transfer;
no existing repository source file is modified.
"""
import argparse
import sys
from functools import partial
from pathlib import Path
import h5py
import numpy as np
import torch
from .common import ENCODERS,digest,file_hash,read
from .prepare import SensorReader
from dataset.yieldsat_image_dataset import YieldSATImageDataset


class CombinedReader:
    def __init__(self, original, sensor):
        self.original,self.sensor=original,sensor
    def get(self,country,index):
        truth=self.original.get(country,index);clean=self.sensor.get(country,index)
        for k in ('temporal','static','times','source_row'):truth[k]=clean[k]
        return truth
    def close(self):self.original.close()


class TransferDataset(YieldSATImageDataset):
    def __init__(self,*v,knowledge_window_days=None,**kw):
        self.knowledge_window_days=knowledge_window_days
        super().__init__(*v,**kw)
        seeds={(t['country'],t['patch_index']):self.season_days[t['season_id']][0] for t in self.tiles}
        # Training targets remain on the original supervised reader. Mask only sensors below.
        sensor=SensorReader(self.reader.root,seeds,self.knowledge_window_days)
        original=self.reader
        self.reader=CombinedReader(original,sensor)
        self._sensor_root=sensor.root
        self._seeds=seeds
        self.season_days=dict(self.season_days)
        ends={}
        for t in self.tiles:
            s=seeds[(t['country'],t['patch_index'])];tm=sensor.get(t['country'],t['patch_index'])['times']
            end=s+self.knowledge_window_days if self.knowledge_window_days is not None else (float(np.nanmax(tm)) if np.isfinite(tm).any() else s)
            ends[t['season_id']]=max(ends.get(t['season_id'],end),end)
        for sid,end in ends.items():self.season_days[sid]=(self.season_days[sid][0],end)
    def __getitem__(self,i):
        b=super().__getitem__(i)
        # Cached DINO features were computed on whole original slots. A
        # mixed-date slot containing any future pixel cannot be repaired by
        # masking only its cached token positions (self-attention is global).
        if 'dino' in b and self.knowledge_window_days is not None:
            index=i[0] if isinstance(i,tuple) else i
            t=self.tiles[index]
            with h5py.File(self._sensor_root/t['country']/'images.h5','r') as f:
                times=f['times'][t['patch_index']]
                present=f['source_row'][t['patch_index']]>=0
            cutoff=self._seeds[(t['country'],t['patch_index'])]+self.knowledge_window_days
            contaminated=((times>cutoff)&present[...,None]).any(axis=(0,1))
            for j,slot in enumerate(b['obs_slot']):
                if slot>=0 and contaminated[slot]:
                    b['dino'][j]=0;b['dino_valid'][j]=0
        b['series_days']=np.where(b['series_mask'],b['series_days'],0).astype(np.float16)
        return b

def install(checkpoint,args,allow_smoke=False):
    import main_yieldsat_image as main
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state.get('smoke_test') and not allow_smoke:raise ValueError('smoke checkpoint cannot initialize a production run')
    if file_hash(Path(args.image_root)/'manifest.json')!=state['image_manifest_hash']:raise ValueError('image export manifest mismatch')
    if args.donor:raise ValueError('knowledge transfer requires an explicit matching grouped split, not donor mode')
    split=read(Path(args.artifact_root)/'splits'/(args.split+'.json'))
    if digest(split)!=state['split_hash']:raise ValueError('pretraining/fine-tuning split mismatch')
    if args.embed_dim!=state['embed_dim'] or args.k_obs!=state['k_obs']:raise ValueError('encoder architecture mismatch')
    if args.cutoff_mode!='all_slots':raise ValueError('wrapper owns the pretraining cutoff; leave parent cutoff_mode=all_slots')
    args.series=True;args.slot_coverage='present';args.init_ckpt=str(checkpoint)
    base_norm=main.ImageNormalizer
    class TransferNormalizer(base_norm):
        @classmethod
        def fit(cls,reader,train_tiles,max_tiles=200,seed=0):
            result=base_norm.fit(reader,train_tiles,max_tiles,seed)
            for k in ('static','temporal'):
                result.stats[k]={key:np.asarray(v) for key,v in state['normalizer'][k].items()}
            return result
    def strict_load(model,path):
        if str(path)!=str(checkpoint):raise ValueError('unexpected initialization checkpoint')
        wanted={k:v for k,v in model.state_dict().items() if k.split('.')[0] in ENCODERS}
        supplied=state['model']
        if set(wanted)!=set(supplied) or any(wanted[k].shape!=supplied[k].shape for k in wanted):raise ValueError('encoder transfer must be exact; use all S2+ADM streams and --series')
        model.load_state_dict(supplied,strict=False)
        return {'checkpoint':str(path),'loaded':len(supplied),'knowledge':state['knowledge'],
                'input_normalization':'pretraining sensor stats; supervised train-only target stats',
                'window_days':state['window_days']}
    main.ImageNormalizer=TransferNormalizer;main.YieldSATImageDataset=partial(TransferDataset,knowledge_window_days=state['window_days']);main.load_donor=strict_load
    return main


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--knowledge-checkpoint',required=True);p.add_argument('--allow-smoke-checkpoint',action='store_true')
    if '--help' in sys.argv or '-h' in sys.argv:
        p.print_help()
        import main_yieldsat_image
        main_yieldsat_image.get_args_parser().print_help()
        return
    own,rest=p.parse_known_args()
    import main_yieldsat_image as runner
    args=runner.get_args_parser().parse_args(rest)
    if args.init_ckpt:raise ValueError('use --knowledge-checkpoint only')
    runner=install(own.knowledge_checkpoint,args,own.allow_smoke_checkpoint)
    runner.main(args)

if __name__=='__main__':main()
