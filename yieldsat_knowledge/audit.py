"""Audit offline annotation or text geometry estimates against independent expert labels."""
import argparse
from pathlib import Path
import numpy as np
from .common import digest,file_hash,jsonl,library,read,write


def annotation_audit(cache,gold,out,min_labels=10,max_mae=0.25):
    gold=read(gold)
    if not gold.get('reviewer') or not gold.get('reviewed_at'):raise ValueError('expert gold provenance required')
    estimates={};profiles=set()
    for path in sorted(Path(cache).glob('*.json')):
        if path.name.endswith('.request.json'):continue
        r=read(path)
        profiles.add(digest(r['profile']))
        for kind,key in [('concepts','presence_score'),('rules','applicability_score')]:
            for x in r['response'][kind]:
                estimates[(r['sample_id'],kind,x['id'])]=(x[key],x['confidence_score'])
    if len(profiles)!=1:raise ValueError('audit exactly one teacher profile at a time')
    errors={'concepts':[],'rules':[]};counts={'concepts':0,'rules':0};missing=0;seen=set();coverage={}
    for row in gold['labels']:
        key=(row['sample_id'],row['kind'],row['id'])
        if key in seen:raise ValueError('duplicate gold label')
        seen.add(key)
        if row['kind'] not in errors or not isinstance(row['score'],(int,float)) or not 0<=row['score']<=1:raise ValueError('invalid gold')
        if key not in estimates:raise ValueError('gold label has no cached annotation')
        counts[row['kind']]+=1;p,_=estimates[key]
        if p is None:missing+=1;continue
        errors[row['kind']].append(float(p)-float(row['score']))
        coverage.setdefault(row['kind']+':'+row['id'],set()).add('positive' if row['score']>=.5 else 'negative')
    stats={k:{'estimated':len(v),'gold':counts[k],'mae':float(np.abs(v).mean()) if v else None,
              'brier':float(np.square(v).mean()) if v else None} for k,v in errors.items()}
    required={kind+':'+cid for _,kind,cid in estimates}
    balanced={k: sorted(coverage.get(k,set())) for k in sorted(required)}
    passed=all(set(v)=={'positive','negative'} for v in balanced.values()) and all(s['estimated']>=min_labels and s['mae']<=max_mae for s in stats.values())
    report={'kind':'annotation_audit','passed':passed,'metrics':stats,'unknowns':missing,'coverage':balanced,
            'audited_ids':sorted(required),
            'profile_hash':next(iter(profiles)),'gold_hash':digest(gold),
            'min_labels':min_labels,'max_mae':max_mae,
            'note':'No self-confidence calibration claimed. Accepted estimates receive fixed reliability 1; presence/context remain soft.'}
    write(out,report)
    write(str(out)+'.policy.json',{'kind':'annotation_policy','profile_hash':next(iter(profiles)),
          'report_hash':digest(report),'reliability':1.0,
          'review':{'status':'draft','reviewer':None,'reviewed_at':None,'evidence':None}})


def geometry_audit(lib_path,text_root,examples,out):
    lib=library(lib_path);root=Path(text_root);meta=read(root/'manifest.json')
    if meta['library_hash']!=digest(lib) or meta['vectors_hash']!=file_hash(root/'vectors.npz'):raise ValueError('stale text cache')
    with np.load(root/'vectors.npz') as z: vectors=dict(zip(z['ids'].tolist(),z['vectors']))
    rules={r['id']:r for r in lib['rules']};gold=read(examples)
    if not gold.get('reviewer') or not gold.get('reviewed_at'):raise ValueError('expert geometry examples required')
    scores=[];seen=set()
    for row in gold['examples']:
        key=(row['rule'],row['a'],row['b'])
        if key in seen:raise ValueError('duplicate geometry example')
        seen.add(key);r=rules[row['rule']]
        if row['partition'] not in ('calibration','audit') or row['label'] not in (0,1):raise ValueError('invalid example')
        if row['partition']=='audit' and (row['a'],row['b']) in [(r['concept_a'],r['concept_b']),(r['concept_b'],r['concept_a'])]:
            raise ValueError('audit must use independently reviewed alternate concepts, not the trivial original/reverse pair')
        a=vectors[row['b']]-vectors[row['a']];b=vectors[r['concept_b']]-vectors[r['concept_a']]
        if min(np.linalg.norm(a),np.linalg.norm(b))<1e-6:raise ValueError('degenerate geometry')
        scores.append(dict(row,score=float(a@b/np.linalg.norm(a)/np.linalg.norm(b))))
    per_rule={}
    for rid in rules:
        cal=[r for r in scores if r['rule']==rid and r['partition']=='calibration']
        test=[r for r in scores if r['rule']==rid and r['partition']=='audit']
        if any({r['label'] for r in rows}!={0,1} for rows in (cal,test)):
            raise ValueError('each rule needs positive/negative calibration and independent audit examples')
        threshold=max(np.linspace(-1,1,101),key=lambda t:np.mean([(r['score']>=t)==bool(r['label']) for r in cal]))
        acc=float(np.mean([(r['score']>=threshold)==bool(r['label']) for r in test]))
        per_rule[rid]={'threshold':float(threshold),'audit_accuracy':acc,'audit_n':len(test)}
    report={'kind':'geometry_audit','library_hash':digest(lib),'text_hash':digest(meta),'mock':meta['mock'],
            'passed':all(r['audit_accuracy']>=0.75 for r in per_rule.values()),'rules':per_rule,'examples_hash':digest(gold)}
    write(out,report)
    write(str(out)+'.policy.json',{'kind':'geometry_policy','report_hash':digest(report),'library_hash':digest(lib),
          'text_hash':digest(meta),'review':{'status':'draft','reviewer':None,'reviewed_at':None,'evidence':None}})


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('annotations');a.add_argument('--cache',required=True);a.add_argument('--gold',required=True)
    a.add_argument('--output',required=True);a.add_argument('--min-labels',type=int,default=10);a.add_argument('--max-mae',type=float,default=.25)
    g=sub.add_parser('geometry');g.add_argument('--library',required=True);g.add_argument('--text-cache',required=True)
    g.add_argument('--examples',required=True);g.add_argument('--output',required=True)
    a=p.parse_args()
    if a.cmd=='annotations':
        if a.min_labels<1 or not 0<=a.max_mae<=1:p.error('invalid audit thresholds')
        annotation_audit(a.cache,a.gold,a.output,a.min_labels,a.max_mae)
    else:geometry_audit(a.library,a.text_cache,a.examples,a.output)

if __name__=='__main__':main()
