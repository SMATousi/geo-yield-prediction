"""Offline integration tests. Expert approvals below are SYNTHETIC fixtures only."""
import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from yieldsat_knowledge.common import digest,file_hash,library,read,write
from yieldsat_knowledge.make_library import build
from yieldsat_knowledge.prepare import run as prepare,select_tiles
from yieldsat_knowledge.pilot import validate_response
from yieldsat_knowledge.text_cache import build as text_build
from yieldsat_knowledge.audit import annotation_audit
from yieldsat_knowledge.train import parser,run,verify_prepared,PreparedDataset
from yieldsat_knowledge.model import KnowledgeEncoders


REVIEW={'status':'approved','reviewer':'SYNTHETIC TEST ONLY','reviewed_at':'2000-01-01','evidence':'fabricated fixture; not scientific approval'}


@pytest.fixture
def artifacts(tmp_path):
    root=tmp_path/'images';country='Germany';(root/country).mkdir(parents=True)
    art=tmp_path/'art';(art/'index'/country).mkdir(parents=True)
    rng=np.random.default_rng(5);n=4
    t=rng.normal(1500,200,(n,64,64,24,16)).astype('f4')
    s=rng.normal(50,5,(n,64,64,104)).astype('f4')
    tm=np.broadcast_to(18000+np.arange(24)*10,(n,64,64,24)).astype('f4').copy()
    source=np.broadcast_to(np.arange(4096).reshape(64,64),(n,64,64)).copy()
    source[:,48:]=-1;t[:,48:]=np.nan;s[:,48:]=np.nan;tm[:,48:]=np.nan
    # Optical-missing but dated slot, exercises date masking.
    t[:,:,:,6,:12]=np.nan
    with h5py.File(root/country/'images.h5','w') as f:
        for k,v in [('temporal',t),('static',s),('times',tm),('source_row',source)]:f[k]=v
        # Intentionally NO target/valid_pixel datasets: pretraining must never read either.
    tiles=[];fields=[];ids=[]
    for i in range(n):
        name=f'Germany_test_field{i}_wheat_2020';sid=country+'/'+name;ids.append(sid)
        tiles.append(dict(country=country,patch_index=i,field_shared_name=name,physical_field_id=f'Germany/field{i}',
                          crop='wheat',year=2020,row0=0,col0=0))
        fields.append(dict(season_id=sid,seeding_day=18000,harvest_day=19999,field_shared_name=name))
    (root/country/'patches.jsonl').write_text('\n'.join(json.dumps(r) for r in tiles))
    write(root/'manifest.json',dict(fixture=True,tile_size=64,min_valid_pixels=1,
        temporal_channels=list(__import__('dataset.yieldsat_schema',fromlist=['TEMPORAL_CHANNELS']).TEMPORAL_CHANNELS),
        static_channels=list(__import__('dataset.yieldsat_schema',fromlist=['STATIC_CHANNELS']).STATIC_CHANNELS)));write(art/'index'/country/'fields.json',fields)
    split={'countries':[country],'partitions':{'train':ids[:2],'val':[ids[2]],'test':[ids[3]]}}
    write(tmp_path/'split.json',split)
    output=tmp_path/'prepared'
    prepare(argparse.Namespace(image_root=str(root),artifact_root=str(art),split=str(tmp_path/'split.json'),
                              output_dir=str(output),window_days=100,max_tiles=None,seed=0,k_obs=2))
    lib=build()
    for kind in ('concepts','rules'):
        for c in lib[kind]:c['review']=dict(REVIEW)
    libpath=tmp_path/'reviewed.json';write(libpath,lib)
    cache=tmp_path/'text';text_build(lib,cache,mock=True)
    return tmp_path,root,art,output,libpath,cache,split


def annotations(artifacts):
    tmp,root,art,prep,libpath,text,split=artifacts
    lib=read(libpath);meta,rows=verify_prepared(prep);cache=tmp/'ann';cache.mkdir()
    profile={'library_hash':digest(lib),'model':'synthetic-offline','vision':False,'prompt_hash':'test'}
    labels=[]
    for i,ev in enumerate(rows):
        value=.9 if i==0 else .1
        response={}
        for kind,key in [('concepts','presence_score'),('rules','applicability_score')]:
            response[kind]=[]
            for c in lib[kind]:
                refs=['channels.'+c['channels'][0]] if kind=='concepts' else ['dates']
                response[kind].append(dict(id=c['id'],status='estimated',confidence_score=.9,evidence_refs=refs,reason='synthetic fixture',**{key:value}))
                labels.append(dict(sample_id=ev['sample_id'],kind=kind,id=c['id'],score=value))
        rh=digest({'profile':profile,'evidence_hash':ev['evidence_hash']})
        write(cache/(rh+'.json'),dict(profile=profile,request_hash=rh,evidence_hash=ev['evidence_hash'],sample_id=ev['sample_id'],response=response))
    gold=tmp/'gold.json';write(gold,{'reviewer':'synthetic','reviewed_at':'2000-01-01','labels':labels})
    audit=tmp/'audit.json';annotation_audit(cache,gold,audit)
    policy=Path(str(audit)+'.policy.json');p=read(policy);p['review']=dict(REVIEW);write(policy,p)
    geometry=tmp/'geometry.json';gm={'kind':'geometry_audit','passed':True,'library_hash':digest(lib),'text_hash':digest(read(text/'manifest.json')),'mock':True}
    write(geometry,gm);gp=tmp/'geometry-policy.json'
    write(gp,{'report_hash':digest(gm),'library_hash':digest(lib),'text_hash':gm['text_hash'],'review':dict(REVIEW)})
    return cache,policy,audit,gp,geometry


def test_draft_library_rejected_and_mirrors_identical(tmp_path):
    root=Path(__file__).parents[1]
    assert file_hash(root/'assets/library.json')==file_hash(root.parent/'spec/yieldsat_knowledge_assets/library.json')
    # the shipped library is approved by the project lead (2026-10-04, decision K1) ...
    lib=library(root/'assets/library.json',True)
    assert len(lib['rules'])==6
    # ... and any draft entry is still refused for training
    raw=json.loads((root/'assets/library.json').read_text())
    raw['rules'][0]['review']={'status':'draft','reviewer':None,'reviewed_at':None,'evidence':None}
    draft=tmp_path/'draft.json';draft.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='expert review'):library(draft,True)


def test_yield_free_preparation_cutoff_and_masks(artifacts):
    _,root,_,prep,_,_,split=artifacts
    meta,rows=verify_prepared(prep)
    assert len(rows)==2 and meta['window_days']==100
    with np.load(prep/rows[0]['tensor']) as z:
        assert not any(k.startswith('target') for k in z.files)
        assert not z['series_mask'][...,11:].any()
        assert not z['series_days'][...,6].any()
        assert not z['series_mask'][48:].any()
        assert not z['weather_mask'][0].any()
    bad=copy.deepcopy(split);bad['partitions']['test'].append(split['partitions']['train'][0])
    with pytest.raises(ValueError,match='overlapping'):select_tiles(root,bad)


def test_teacher_rejects_unknown_as_zero_and_false_citations(artifacts):
    cache,*_=annotations(artifacts)
    ann=read(next(cache.glob('*.json')));lib=read(artifacts[4]);_,rows=verify_prepared(artifacts[3])
    r=ann['response'];validate_response(r,lib,rows[0])
    r['concepts'][0]['status']='unknown'
    with pytest.raises(ValueError,match='unknown'):validate_response(r,lib,rows[0])
    r['concepts'][0]['status']='estimated';r['concepts'][0]['evidence_refs']=['target']
    with pytest.raises(ValueError,match='citation'):validate_response(r,lib,rows[0])


def train_args(artifacts,paths):
    tmp,_,_,prep,lib,text,_=artifacts;cache,policy,audit,gp,geometry=paths
    return parser().parse_args(['--prepared',str(prep),'--library',str(lib),'--text-cache',str(text),
        '--annotation-cache',str(cache),'--annotation-policy',str(policy),'--annotation-audit',str(audit),
        '--geometry-policy',str(gp),'--geometry-audit',str(geometry),'--output-dir',str(tmp/'train'),
        '--device','cpu','--embed-dim','32','--epochs','1','--steps-per-epoch','2','--batch-size','2','--smoke-test'])


def test_training_resume_and_exact_encoder_transfer(artifacts):
    paths=annotations(artifacts);a=train_args(artifacts,paths);run(a)
    checkpoint=Path(a.output_dir)/'encoders.pth';state=torch.load(checkpoint,weights_only=False)
    from models_yieldsat_image import YieldSATImageModel
    model=YieldSATImageModel(embed_dim=32,k_obs=2,num_latents=4,depth=2,use_series=True,use_dino=False)
    own=model.state_dict()
    assert state['model'] and all(k in own and own[k].shape==v.shape for k,v in state['model'].items())
    assert not any(k.startswith(('fusion','dino','head','projectors')) for k in state['model'])
    a.resume=str(Path(a.output_dir)/'checkpoint_last.pth');a.epochs=2;run(a)
    assert len(read(Path(a.output_dir)/'report.json')['history'])==2
    a.smoke_test=False;a.resume=None;a.output_dir=str(artifacts[0]/'production')
    with pytest.raises(ValueError,match='mock embeddings'):run(a)


def test_all_encoder_groups_receive_gradients(artifacts):
    torch.set_num_threads(2);lib=read(artifacts[4]);_,rows=verify_prepared(artifacts[3])
    ds=PreparedDataset(artifacts[3],rows,lib,{})
    b={k:v.unsqueeze(0) for k,v in ds[0].items()}
    b['concept_targets'].fill_(.9);b['rule_targets'].fill_(.9)
    vectors=torch.randn(len(lib['concepts']),16)
    model=KnowledgeEncoders(32,vectors,lib['concepts'],lib['rules'])
    loss,metrics=model.loss(b);loss.backward()
    from yieldsat_knowledge.common import ENCODERS
    for name in ENCODERS:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in getattr(model,name).parameters()),name
    assert metrics['relation_terms']==len(lib['rules'])


def test_stale_sensor_cache_refused(artifacts):
    prep=artifacts[3];meta,rows=verify_prepared(prep)
    path=prep/rows[0]['tensor'];path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='stale'):verify_prepared(prep)


def test_pilot_dry_run_and_cached_response(artifacts,monkeypatch):
    from yieldsat_knowledge.pilot import run as pilot
    import yieldsat_knowledge.pilot as module
    tmp,_,_,prep,libpath,_,_=artifacts
    args=argparse.Namespace(library=str(libpath),prepared=str(prep),output_dir=str(tmp/'pilot'),
        model='offline-placeholder',provider=None,vision=True,max_samples=1,max_tokens=1000,execute=False)
    def no_network(*a,**kw):raise AssertionError('network call during dry-run')
    monkeypatch.setattr(module,'post',no_network);pilot(args)
    assert len(list((tmp/'pilot').glob('*.request.json')))==1
    cache,*_=annotations(artifacts);fake=read(next(cache.glob('*.json')))['response']
    monkeypatch.setattr(module,'post',lambda body:(fake,{'model':'offline-test'}))
    args.execute=True;pilot(args)
    monkeypatch.setattr(module,'post',no_network);pilot(args) # cached and validated


def test_audit_refuses_only_positive_gold(artifacts):
    cache,policy,audit,*_=annotations(artifacts)
    gold=read(artifacts[0]/'gold.json')
    for row in gold['labels']:row['score']=.9
    write(artifacts[0]/'positive.json',gold)
    annotation_audit(cache,artifacts[0]/'positive.json',artifacts[0]/'positive-audit.json',max_mae=1)
    assert not read(artifacts[0]/'positive-audit.json')['passed']


def test_finetune_adapter_preserves_stats_masks_and_sensor_only_forward(artifacts,monkeypatch):
    from yieldsat_knowledge.finetune import install
    from dataset.yieldsat_image_dataset import TileReader,load_tile_table
    from models_yieldsat_image import YieldSATImageModel
    import main_yieldsat_image as runner
    paths=annotations(artifacts);a=train_args(artifacts,paths);run(a)
    tmp,root,art,prep,_,_,split=artifacts
    write(art/'splits'/'test.json',split)
    with h5py.File(root/'Germany'/'images.h5','a') as f:
        f['target']=np.full((4,64,64),5.,np.float32)
        f['valid_pixel']=f['source_row'][:]>=0
        # One future pixel in each otherwise eligible slot contaminates cached DINO.
        f['times'][0,0,0,:]=18101
    args=runner.get_args_parser().parse_args(['--countries','Germany','--artifact_root',str(art),
        '--image_root',str(root),'--split','test','--embed_dim','32','--k_obs','2','--num_workers','0'])
    for name in ('ImageNormalizer','YieldSATImageDataset','load_donor'):
        monkeypatch.setattr(runner,name,getattr(runner,name)) # restore install's process-local replacement
    install(Path(a.output_dir)/'encoders.pth',args,allow_smoke=True)
    tiles=load_tile_table(root,['Germany'])[:2];reader=TileReader(root)
    norm=runner.ImageNormalizer.fit(reader,tiles)
    expected=read(prep/'normalizer.json')
    assert np.allclose(norm.stats['temporal']['mean'],expected['temporal']['mean'])
    assert norm.stats['target']['mean'][0]==5
    class FakeDino:
        def get(self,country,index,slots):return np.ones((len(slots),16,1024),np.float32),np.ones(len(slots),np.float32)
    ds=runner.YieldSATImageDataset(root,tiles,norm,{t['season_id']:(18000,19999) for t in tiles},
                                  k_obs=2,with_series=True,slot_coverage='present',dino_cache=FakeDino())
    b=ds[0]
    assert (b['obs_slot']>=0).any()
    assert not b['dino_valid'].any() and not b['dino'].any()
    assert not b['series_mask'][...,11:].any()
    model=YieldSATImageModel(embed_dim=32,k_obs=2,num_latents=4,depth=2,use_series=True,use_dino=False)
    runner.load_donor(model,str(Path(a.output_dir)/'encoders.pth'))
    inputs={k:torch.as_tensor(v).unsqueeze(0) for k,v in b.items() if k!='country' and not k.startswith('target')}
    from dataset.yieldsat_image_dataset import cast_batch
    model.eval()
    with torch.no_grad():prediction=model(cast_batch(inputs))
    assert prediction is not None
    reader.close()


def test_diagnostics_include_all_streams(artifacts):
    from yieldsat_knowledge.diagnostics import run as diagnostics
    paths=annotations(artifacts);a=train_args(artifacts,paths);run(a)
    args=argparse.Namespace(checkpoint=str(Path(a.output_dir)/'checkpoint_last.pth'),prepared=a.prepared,
        library=a.library,annotation_cache=a.annotation_cache,annotation_policy=a.annotation_policy,
        annotation_audit=a.annotation_audit,output=str(artifacts[0]/'diagnostics.json'),max_samples=2,device='cpu')
    diagnostics(args)
    report=read(args.output)
    assert len(report['projected_embedding_variance'])==6
    assert all(v>0 for v in report['mean_modality_removal_distance'].values())


def test_physical_field_leakage_across_seasons_rejected(artifacts):
    root=artifacts[1];split=artifacts[-1]
    path=root/'Germany'/'patches.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    rows[2]['physical_field_id']=rows[0]['physical_field_id']
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError,match='physical fields overlap'):select_tiles(root,split)


def test_geometry_rejects_trivial_self_confirmation(artifacts):
    from yieldsat_knowledge.audit import geometry_audit
    tmp,_,_,_,libpath,text,_=artifacts
    lib=read(libpath);r=lib['rules'][0]
    gold={'reviewer':'SYNTHETIC','reviewed_at':'2000-01-01','examples':[
        {'rule':r['id'],'a':r['concept_a'],'b':r['concept_b'],'label':1,'partition':'audit'}]}
    write(tmp/'geometry-examples.json',gold)
    with pytest.raises(ValueError,match='alternate concepts'):
        geometry_audit(libpath,text,tmp/'geometry-examples.json',tmp/'bad-geometry.json')


def test_relation_alone_reaches_both_encoder_endpoints(artifacts):
    from yieldsat_knowledge.diagnostics import pooled
    from torch.nn import functional as F
    from yieldsat_knowledge.common import ENCODERS
    torch.set_num_threads(2);lib=read(artifacts[4]);_,rows=verify_prepared(artifacts[3])
    ds=PreparedDataset(artifacts[3],rows,lib,{})
    b={k:v.unsqueeze(0) for k,v in ds[0].items()}
    model=KnowledgeEncoders(32,torch.randn(len(lib['concepts']),16),lib['concepts'],lib['rules'])
    z,_=pooled(model,b)
    # No sensor or grounding term: gradients must come from the relation itself.
    losses=[]
    for j,(a,bb) in enumerate(model.pairs):
        delta=F.normalize(z[lib['concepts'][bb]['stream']]-z[lib['concepts'][a]['stream']],dim=-1)
        losses.append(1-(delta*model.relations[j]).sum())
    torch.stack(losses).sum().backward()
    for name in ENCODERS:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in getattr(model,name).parameters()),name
