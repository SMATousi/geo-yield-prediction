"""Optional OpenRouter library drafting. Output can never self-approve."""
import argparse
from pathlib import Path
from .common import read,write,library
from .make_library import render
from .pilot import PROMPTS,post


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-library',default=str(Path(__file__).parent/'assets/library.json'))
    p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    lib=library(a.seed_library)
    request={'model':a.model,'temperature':0,'max_tokens':12000,
             'messages':[{'role':'system','content':(PROMPTS/'generate_library.md').read_text()},
                         {'role':'user','content':__import__('json').dumps(lib)}],
             'response_format':{'type':'json_object'},'provider':{'require_parameters':True,'allow_fallbacks':False}}
    dest=Path(a.output)
    if dest.exists():raise ValueError('output already exists')
    if not a.execute:
        write(str(dest)+'.request.json',request);return
    generated,meta=post(request)
    for kind in ('concepts','rules'):
        for r in generated[kind]:r['review']={'status':'draft','reviewer':None,'reviewed_at':None,'evidence':None}
    generated['generator']={'kind':'OpenRouter draft','provider_response':meta}
    # Validate a temporary candidate before publication.
    tmp=dest.with_suffix('.candidate.json');write(tmp,generated)
    library(tmp);tmp.replace(dest)
    dest.with_suffix('.md').write_text(render(generated))

if __name__=='__main__':main()
