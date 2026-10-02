"""Offline OpenRouter LLM/VLM annotation pilot; dry-run unless --execute."""
import argparse
import base64
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

from .common import digest, file_hash, jsonl, library, read, safe_child, write

PROMPTS=Path(__file__).parent/'prompts'


def response_schema(lib):
    score={'type':['number','null'],'minimum':0,'maximum':1}
    def item(kind):
        props={'id':{'type':'string','enum':[x['id'] for x in lib[kind]]},
               'status':{'type':'string','enum':['estimated','unknown','not_observable']},
               'confidence_score':score,'evidence_refs':{'type':'array','items':{'type':'string'}},
               'reason':{'type':'string'},
               ('presence_score' if kind=='concepts' else 'applicability_score'):score}
        return {'type':'object','properties':props,'required':list(props),'additionalProperties':False}
    return {'type':'object','properties':{k:{'type':'array','items':item(k)} for k in ('concepts','rules')},
            'required':['concepts','rules'],'additionalProperties':False}


def validate_response(result,lib,ev,vision=False):
    if set(result)!={'concepts','rules'}: raise ValueError('unexpected annotation fields')
    for kind in ('concepts','rules'):
        records=result[kind]; defs={c['id']:c for c in lib[kind]}
        if len(records)!=len(defs) or {r['id'] for r in records}!=set(defs):
            raise ValueError('missing/duplicate/extra annotation IDs')
        key='presence_score' if kind=='concepts' else 'applicability_score'
        for r in records:
            if set(r)!={'id','status',key,'confidence_score','evidence_refs','reason'}:
                raise ValueError('invalid annotation schema')
            if r['status'] not in ('estimated','unknown','not_observable'): raise ValueError('invalid status')
            for v in (r[key],r['confidence_score']):
                if r['status']=='estimated':
                    if isinstance(v,bool) or not isinstance(v,(float,int)) or not 0<=v<=1:
                        raise ValueError('invalid score')
                elif v is not None: raise ValueError('unknown is not a negative')
            allowed={'dates'} | {'channels.'+c for c in (defs[r['id']]['channels'] if kind=='concepts' else ev['channels'])}
            if vision and (kind=='rules' or defs[r['id']]['family']=='optical'): allowed.add('image')
            if not isinstance(r['evidence_refs'],list) or not set(r['evidence_refs'])<=allowed:
                raise ValueError('unsupported evidence citation')
            if r['status']=='estimated' and not r['evidence_refs']: raise ValueError('estimate needs evidence')
            if kind=='concepts' and r['status']=='estimated' and not (set(r['evidence_refs']) & (allowed-{'dates'})):
                raise ValueError('concept estimate requires sensor evidence, not only dates')
            if not isinstance(r['reason'],str) or not r['reason'].strip(): raise ValueError('reason required')
    return result


def post(body,timeout=120,retries=3):
    key=os.environ.get('OPENROUTER_API_KEY')
    if not key: raise ValueError('OPENROUTER_API_KEY is required only with --execute')
    for attempt in range(retries):
        req=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
            data=json.dumps(body,allow_nan=False).encode(),
            headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=timeout) as r: result=json.load(r)
            choice=result['choices'][0]
            if choice.get('finish_reason')!='stop': raise ValueError('incomplete/refused provider response')
            content=choice['message'].get('content')
            if not isinstance(content,str): raise ValueError('provider did not return JSON text')
            return json.loads(content),{'id':result.get('id'),'model':result.get('model'),
                                       'usage':result.get('usage',{}),'provider':result.get('provider')}
        except urllib.error.HTTPError as e:
            if e.code not in (429,500,502,503,504) or attempt==retries-1:
                raise RuntimeError('OpenRouter HTTP '+str(e.code)) from None
            time.sleep(min(30,2**attempt))
        except (urllib.error.URLError,TimeoutError):
            if attempt==retries-1: raise RuntimeError('OpenRouter network/timeout failure') from None
            time.sleep(min(30,2**attempt))


def run(a):
    lib=library(a.library); root=Path(a.prepared); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    groups={}
    for row in jsonl(root/'evidence.jsonl'):groups.setdefault((row['country'],row['crop']),[]).append(row)
    rng=random.Random(getattr(a,'seed',0))
    for group in groups.values():rng.shuffle(group)
    rows=[]
    while len(rows)<a.max_samples and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(rows)<a.max_samples:rows.append(groups[key].pop())
    prompt=(PROMPTS/'applicability_llm.md').read_text()
    if a.vision: prompt+='\n'+(PROMPTS/'applicability_vlm.md').read_text()
    profile={'model':a.model,'vision':a.vision,'prompt_hash':digest(prompt),'library_hash':digest(lib),
             'max_tokens':a.max_tokens,'temperature':0,'provider':a.provider}
    for ev in rows:
        check=dict(ev); claimed=check.pop('evidence_hash')
        if digest(check)!=claimed: raise ValueError('evidence integrity failure')
        # Only allowlisted sensor evidence goes to the provider.
        evidence={k:ev[k] for k in ('sample_id','country','crop','support','policy','present_cells',
                                   'dates','weather_interval_inclusive_days','channels','unit_notes')}
        text=json.dumps({'library':lib,'evidence':evidence},allow_nan=False)
        content=[{'type':'text','text':text}]
        if a.vision:
            path=safe_child(root,ev['image'])
            if file_hash(path)!=ev['image_hash']: raise ValueError('image integrity failure')
            content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()}})
        req_hash=digest({'profile':profile,'evidence_hash':claimed})
        dest=out/(req_hash+'.json')
        if dest.exists():
            old=read(dest)
            if (old['request_hash']!=req_hash or old['profile']!=profile or old['evidence_hash']!=claimed
                or old['sample_id']!=ev['sample_id']): raise ValueError('stale cache')
            validate_response(old['response'],lib,ev,a.vision)
            continue
        body={'model':a.model,'temperature':0,'max_tokens':a.max_tokens,
              'messages':[{'role':'system','content':prompt},{'role':'user','content':content}],
              'response_format':{'type':'json_schema','json_schema':{'name':'yield_free_concepts','strict':True,'schema':response_schema(lib)}},
              'provider':{'require_parameters':True,'allow_fallbacks':False}}
        if a.provider: body['provider']['order']=[a.provider]
        if not a.execute:
            write(out/(req_hash+'.request.json'),{'request_hash':req_hash,'profile':profile,
                  'sample_id':ev['sample_id'],'evidence_hash':claimed,'execute':False})
            continue
        response,usage=post(body)
        validate_response(response,lib,ev,a.vision)
        write(dest,{'request_hash':req_hash,'profile':profile,'evidence_hash':claimed,
                    'sample_id':ev['sample_id'],'response':response,'provider_response':usage})
    print(json.dumps({'samples':len(rows),'executed':a.execute,'profile_hash':digest(profile)}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True);p.add_argument('--library',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--model',required=True)
    p.add_argument('--provider');p.add_argument('--vision',action='store_true')
    p.add_argument('--execute',action='store_true');p.add_argument('--max-samples',type=int,default=10)
    p.add_argument('--seed',type=int,default=0,help='repeatable stratified country/crop pilot selection')
    p.add_argument('--max-tokens',type=int,default=6000)
    a=p.parse_args()
    if a.max_samples<1 or a.max_tokens<1:p.error('positive budgets required')
    run(a)

if __name__=='__main__':main()
