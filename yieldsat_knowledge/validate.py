"""Validate library and mirrored seed artifacts without API access."""
import argparse
from pathlib import Path
from .common import library,file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',default=str(Path(__file__).parent/'assets/library.json'))
    p.add_argument('--require-approved',action='store_true');p.add_argument('--mirror')
    a=p.parse_args();lib=library(a.library,a.require_approved)
    if a.mirror and file_hash(a.library)!=file_hash(a.mirror):raise ValueError('mirrored libraries differ')
    print('%d concepts, %d rules; review statuses: %s'%(len(lib['concepts']),len(lib['rules']),sorted({r['review']['status'] for r in lib['rules']})))

if __name__=='__main__':main()
