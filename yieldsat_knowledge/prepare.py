"""Prepare yield-free training tensors and evidence from existing image/fold artifacts."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from dataset.yieldsat_image_dataset import (
    ImageNormalizer, YieldSATImageDataset, load_tile_table, temporal_valid,
)
from dataset.yieldsat_schema import TEMPORAL_CHANNELS, STATIC_CHANNELS
from .common import digest, file_hash, read, write


class SensorReader:
    """Never opens target or valid_pixel datasets. Fake target fields satisfy the old packer only."""
    def __init__(self, root, seeds, window_days=None):
        self.root, self.seeds, self.window_days = Path(root), seeds, window_days

    def get(self, country, index):
        with h5py.File(self.root/country/'images.h5', 'r') as f:
            d = {k: f[k][index] for k in ('temporal','static','times','source_row')}
        present = d['source_row'] >= 0
        dated = np.isfinite(d['times']) & present[...,None]
        if self.window_days is not None:
            dated &= d['times'] <= self.seeds[(country,index)] + self.window_days
        d['times'] = np.where(dated,d['times'],np.nan)
        d['temporal'] = np.where(dated[...,None],d['temporal'],np.nan)
        d['static'] = np.where(present[...,None],d['static'],np.nan)
        d['target'] = np.zeros(present.shape,np.float32)
        d['valid_pixel'] = np.zeros(present.shape,bool)
        return d

    def close(self): pass


def select_tiles(root, split, max_tiles=None, seed=0):
    tiles = load_tile_table(root, split['countries'])
    partitions = split['partitions']
    sets = {p:set(partitions.get(p,[])) for p in ('train','val','test','excluded')}
    for a in sets:
        for b in sets:
            if a != b and sets[a]&sets[b]: raise ValueError('overlapping season partitions')
    physical = {}
    for t in tiles:
        if not t.get('physical_field_id'): raise ValueError('physical_field_id required')
        for part, members in sets.items():
            if t['season_id'] in members:
                physical.setdefault(part,set()).add((t['country'],t['physical_field_id']))
    if physical.get('train',set()) & (physical.get('val',set())|physical.get('test',set())):
        raise ValueError('physical fields overlap training and held-out groups; use a grouped fold')
    selected = [t for t in tiles if t['season_id'] in sets['train']
                and (not split.get('crops') or t['crop'] in split['crops'])]
    if not selected: raise ValueError('no training image tiles')
    if max_tiles and len(selected)>max_tiles:
        idx=np.sort(np.random.default_rng(seed).choice(len(selected),max_tiles,replace=False))
        selected=[selected[i] for i in idx]
    return selected


def fit_normalizer(reader, tiles):
    acc={k:[np.zeros(n),np.zeros(n),np.zeros(n)] for k,n in
         [('temporal',len(TEMPORAL_CHANNELS)),('static',len(STATIC_CHANNELS))]}
    for t in tiles:
        d=reader.get(t['country'],t['patch_index'])
        for key in acc:
            x=d[key].reshape(-1,d[key].shape[-1]).astype(np.float64)
            m=temporal_valid(d['temporal'],d['times']).reshape(x.shape) if key=='temporal' else np.isfinite(x)
            v=np.where(m,x,0)
            acc[key][0]+=m.sum(0); acc[key][1]+=v.sum(0); acc[key][2]+=(v*v).sum(0)
    stats={}
    for key,(n,s,ss) in acc.items():
        mu=s/np.maximum(n,1); std=np.sqrt(np.maximum(ss/np.maximum(n,1)-mu*mu,0))
        stats[key]={'mean':mu,'std':np.where(std>1e-8,std,1.)}
    stats['target']={'mean':np.zeros(1),'std':np.ones(1)} # dummy, never learned from yield
    return ImageNormalizer(stats)


def summary(x):
    v=np.asarray(x); v=v[np.isfinite(v)]
    return {'count':int(v.size),'mean':float(v.mean()) if v.size else None,
            'min':float(v.min()) if v.size else None,'max':float(v.max()) if v.size else None}


def evidence(d,t,seed_day,window):
    tv=temporal_valid(d['temporal'],d['times'])
    channels={}
    for i,name in enumerate(TEMPORAL_CHANNELS):
        x=np.where(tv[...,i],d['temporal'][...,i],np.nan)
        channels[name]=[summary(x[...,k]) for k in range(24)]
    for i,name in enumerate(STATIC_CHANNELS):
        if name.startswith('coord_') or 'uncertainty' in name: continue
        channels[name]=summary(d['static'][...,i])
    # Interval duration is part of the evidence. Never sum intervals with duplicated boundary days.
    durations=[]; previous=np.full((64,64),np.nan)
    for k in range(24):
        now=d['times'][...,k]
        delta=now-previous+1
        durations.append(summary(np.where(np.isfinite(now)&np.isfinite(previous)&(delta>1),delta,np.nan)))
        previous=np.where(np.isfinite(now),now,previous)
    return {'sample_id':t['country']+':'+str(t['patch_index']), 'country':t['country'],
            'crop':t['crop'],'season_id':t['season_id'],'physical_field_id':t['physical_field_id'],
            'patch_index':t['patch_index'],'support':'one 64x64 tile; 10 m stored grid; coarse source support',
            'policy':{'window_days':window,'retrospective':window is None,'seeding_day':seed_day},
            'present_cells':int((d['source_row']>=0).sum()),
            'dates':[summary(d['times'][...,k]) for k in range(24)],
            'weather_interval_inclusive_days':durations,
            'channels':channels,
            'unit_notes':'S2 DN scale inferred; DEM metres inferred; SoilGrids mapped units inferred. Weather temp_* Kelvin-day sums and precip metres per inclusive interval (inferred); first interval invalid. Aspect convention inferred, slope/curvature/TWI scales unresolved. No management/irrigation labels or native QA layer.'}


def preview(d,path):
    # Fixed stretch across all dates, no per-image enhancement or yield-derived masks.
    sheet=Image.new('RGB',(6*150,4*170),'#333333'); draw=ImageDraw.Draw(sheet)
    for k in range(24):
        rgb=d['temporal'][:,:,k,[3,2,1]]
        good=np.isfinite(rgb).all(-1)&np.isfinite(d['times'][...,k])
        pixels=np.clip(np.nan_to_num(rgb)/3000,0,1)*255
        pixels[~good]=48
        tile=Image.fromarray(pixels.astype('uint8')).resize((128,128),Image.Resampling.NEAREST)
        x,y=k%6*150,k//6*170; sheet.paste(tile,(x,y+22))
        dates=d['times'][...,k][good]
        label='empty' if not len(dates) else str(np.datetime64('1970-01-01')+np.timedelta64(int(np.median(dates)),'D'))
        draw.text((x,y),f'{k:02d} {label}',fill='white')
        draw.text((x,y+152),f'{int(good.sum())} pixels',fill='white')
    sheet.save(path)


def run(a):
    out=Path(a.output_dir)
    if out.exists(): raise ValueError('choose a new output directory')
    source=read(Path(a.image_root)/'manifest.json')
    if source.get('tile_size')!=64 or source.get('temporal_channels')!=list(TEMPORAL_CHANNELS) or source.get('static_channels')!=list(STATIC_CHANNELS):
        raise ValueError('image manifest must declare the canonical 64x64 channel layout')
    partial=source.get('min_valid_pixels')!=1
    if partial and not getattr(a,'allow_partial_corpus',False):
        raise ValueError('production needs the full-coverage image export (--min-valid 1); legacy diagnostics require --allow-partial-corpus')
    split=read(a.split); tiles=select_tiles(a.image_root,split,a.max_tiles,a.seed)
    fields={}
    for c in split['countries']:
        for f in read(Path(a.artifact_root)/'index'/c/'fields.json'): fields[f['season_id']]=f
    seeds={(t['country'],t['patch_index']):float(fields[t['season_id']]['seeding_day']) for t in tiles}
    reader=SensorReader(a.image_root,seeds,a.window_days)
    norm=fit_normalizer(reader,tiles)
    days={}
    for t in tiles:
        s=seeds[(t['country'],t['patch_index'])]
        tm=reader.get(t['country'],t['patch_index'])['times']
        end=s+a.window_days if a.window_days is not None else (float(np.nanmax(tm)) if np.isfinite(tm).any() else s)
        days[t['season_id']]=(s,max(days.get(t['season_id'],(s,end))[1],end))
    ds=YieldSATImageDataset(a.image_root,tiles,norm,days,k_obs=a.k_obs,with_series=True,
                           cutoff_mode='all_slots',slot_coverage='present')
    ds.reader=reader
    out.mkdir(parents=True); (out/'tensors').mkdir(); (out/'images').mkdir()
    records=[]
    for i,t in enumerate(tiles):
        item=ds[i]
        # Remove all legacy target/identity outputs before serializing student inputs.
        item={k:v for k,v in item.items() if not k.startswith('target') and k not in
              ('tile','season','grid_row','grid_col','country','patch_index','crop')}
        item['series_days']=np.where(item['series_mask'],item['series_days'],0).astype(np.float16)
        raw=reader.get(t['country'],t['patch_index'])
        item['series_band_mask']=np.isfinite(raw['temporal'][...,:12]) & np.isfinite(raw['times'])[...,None]
        ev=evidence(raw,t,seeds[(t['country'],t['patch_index'])],a.window_days)
        key=t['country']+'_'+str(t['patch_index'])
        tensor='tensors/'+key+'.npz'; image='images/'+key+'.png'
        np.savez_compressed(out/tensor,**item); preview(raw,out/image)
        ev.update(tensor=tensor,tensor_hash=file_hash(out/tensor),image=image,image_hash=file_hash(out/image))
        ev['evidence_hash']=digest(ev); records.append(ev)
    (out/'evidence.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
    write(out/'normalizer.json',norm.to_json())
    manifest={'schema_version':1,'split_hash':digest(split),'split':split,'image_manifest_hash':file_hash(Path(a.image_root)/'manifest.json'),
              'evidence_hash':file_hash(out/'evidence.jsonl'),'normalizer_hash':file_hash(out/'normalizer.json'),
              'window_days':a.window_days,'k_obs':a.k_obs,'tiles':len(tiles),'seed':a.seed,
              'partial_corpus':partial,
              'scope':'training partitions only; sensor-only; source corpus label-selected'}
    write(out/'manifest.json',manifest)
    print(json.dumps({'tiles':len(tiles),'output':str(out)}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image-root',required=True); p.add_argument('--artifact-root',required=True)
    p.add_argument('--split',required=True,help='path to existing fold JSON')
    p.add_argument('--output-dir',required=True); p.add_argument('--window-days',type=int)
    p.add_argument('--allow-partial-corpus',action='store_true',help='explicit legacy corpus diagnostic; recorded in manifest')
    p.add_argument('--max-tiles',type=int); p.add_argument('--k-obs',type=int,default=4); p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    if a.window_days is not None and a.window_days<=0: p.error('window-days must be positive')
    if a.k_obs<1 or a.max_tiles is not None and a.max_tiles<1: p.error('positive counts required')
    run(a)

if __name__=='__main__': main()
