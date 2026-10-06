"""Cluster driver for the TabM DEV rounds (spec/yieldsat-tabm.md, TM-07) and other
claimable work units (a unit with ``script`` runs that script with its own arguments).

  units  expand a suite (cluster/suites/tabm_*.yaml) into work units, one per
         (config, DEV row, chunk of <= --chunk folds), skipping units whose
         folds already have report.json under --skip_done_root (e.g. the local
         results); writes cluster/tabm/<suite>_units.json (committed, so pods
         read it from the cloned repository)
  pool   run inside each pod: claim units atomically (mkdir on the shared
         volume), run main_yieldsat_tabm.py for the unit, mark done; a claim
         whose heartbeat is older than --stale_minutes is taken over; a unit is
         retried at most --max_attempts times. Finished folds are skipped by
         main_yieldsat_tabm.py itself.

    python yieldsat_tabm_cluster.py units --suite cluster/suites/tabm_tm1.yaml \
        --skip_done_root /root/yieldsat_artifacts/runs/tabm_dev1
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

from main_yieldsat_tabm import dev_rows


def cmd_units(a):
    suite = yaml.safe_load(open(a.suite))
    units = []
    for cfg in suite['configs']:
        for row in dev_rows(a.artifact_root, cfg.get('pairs'), tuple(cfg.get('protocols', ('cv', 'loyo', 'loro')))):
            todo = []
            for i in range(len(row['folds'])):
                rel = 'paper/{}/s2_adm/{}_seed{}/{}/fold{:02d}'.format(row['group'], cfg['tag'], cfg.get('seed', 0),
                                                                      row['pair'], i)
                if a.skip_done_root and (Path(a.skip_done_root) / rel / 'report.json').exists():
                    continue
                todo.append(i)
            for c in range(0, len(todo), a.chunk):
                folds = todo[c:c + a.chunk]
                uid = '{}__{}__{}__f{}'.format(cfg['tag'], row['pair'], row['protocol'], '-'.join(map(str, folds)))
                args = ['--model', cfg['model'], '--features', cfg['features'], '--tag', cfg['tag'],
                        '--pairs', row['pair'], '--protocols', row['protocol'], '--folds', *map(str, folds),
                        '--seed', str(cfg.get('seed', 0))] + [str(x) for x in cfg.get('args', [])]
                units.append({'id': uid, 'args': args, 'pair': row['pair'], 'n_folds': len(folds)})
    # large pairs first (longest-processing-time order)
    weight = {'URG-S': 4, 'ARG-W': 2, 'BRA-C': 2, 'GER-R': 1}
    units.sort(key=lambda u: -weight.get(u['pair'], 1) * u['n_folds'])
    out = Path(a.out or 'cluster/tabm/{}_units.json'.format(suite['name']))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'suite': suite['name'], 'out_root': suite['out_root'], 'units': units}, indent=1))
    print('{} units ({} folds) -> {}'.format(len(units), sum(u['n_folds'] for u in units), out))


class Claims:
    def __init__(self, root, owner, stale_minutes):
        self.root, self.owner, self.stale = Path(root), owner, stale_minutes * 60
        self.root.mkdir(parents=True, exist_ok=True)

    def d(self, uid):
        return self.root / uid

    def try_claim(self, uid):
        d = self.d(uid)
        try:
            d.mkdir()
        except FileExistsError:
            if (d / 'done').exists() or (d / 'gave_up').exists():
                return False
            hb = d / 'heartbeat'
            age = time.time() - (hb.stat().st_mtime if hb.exists() else d.stat().st_mtime)
            if age < self.stale:
                return False
            n = len(list(d.glob('takeover_*')))
            try:
                (d / 'takeover_{}'.format(n)).mkdir()
            except FileExistsError:
                return False
        (d / 'owner').write_text('{} {}'.format(self.owner, time.strftime('%Y-%m-%dT%H:%M:%S')))
        (d / 'heartbeat').write_text(str(time.time()))
        return True

    def beat(self, uid):
        (self.d(uid) / 'heartbeat').write_text(str(time.time()))


def stage(rels, src_root, dst_root):
    """Copy artifact paths (files, directories or globs relative to src_root) to the local
    artifact root once per pod; a '.staged' marker records a complete copy."""
    import fcntl
    src, dst = Path(src_root), Path(dst_root)
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / '.stage.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for rel in rels:
            mark = dst / '.staged' / rel.replace('/', '__').replace('*', '+')
            if mark.exists():
                continue
            for s_ in sorted(src.glob(rel)):
                d_ = dst / s_.relative_to(src)
                d_.parent.mkdir(parents=True, exist_ok=True)
                if s_.is_dir():
                    shutil.copytree(s_, d_, dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns('uuids.npy', 'field_stats_v2_backup'))
                else:
                    shutil.copy2(s_, d_)
            mark.parent.mkdir(parents=True, exist_ok=True)
            mark.write_text(time.strftime('%Y-%m-%dT%H:%M:%S'))


def cmd_pool(a):
    plan = json.loads(Path(a.units).read_text())
    out_root = a.out_root or plan['out_root']
    owner = os.environ.get('HOSTNAME', os.uname().nodename)
    claims = Claims(Path(out_root) / '_claims', owner, a.stale_minutes)
    ran = 0
    while True:
        unit = None
        for u in plan['units']:
            if claims.try_claim(u['id']):
                unit = u
                break
        if unit is None:
            break
        d = claims.d(unit['id'])
        attempts = int((d / 'attempts').read_text()) + 1 if (d / 'attempts').exists() else 1
        (d / 'attempts').write_text(str(attempts))
        if unit.get('stage') and a.stage_from:
            stage(unit['stage'], a.stage_from, a.artifact_root)
        if 'script' in unit:
            # generic unit: full argument list with {artifact_root} / {out_root} / {source_root} placeholders
            fmt = dict(artifact_root=a.artifact_root, out_root=out_root, source_root=a.source_root)
            cmd = [sys.executable, unit['script'], *[str(x).format(**fmt) for x in unit['args']]]
        else:
            cmd = [sys.executable, 'main_yieldsat_tabm.py', '--artifact_root', a.artifact_root, '--out_root', out_root,
                   *unit['args']]
        print('unit', unit['id'], 'attempt', attempts, flush=True)
        stop = threading.Event()

        def heartbeat():
            while not stop.wait(60):
                claims.beat(unit['id'])
        th = threading.Thread(target=heartbeat, daemon=True)
        th.start()
        log = open(d / 'log_attempt{}.txt'.format(attempts), 'w')
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')).returncode
        log.close()
        stop.set()
        if rc == 0:
            (d / 'done').write_text(json.dumps({'owner': owner, 'finished': time.strftime('%Y-%m-%dT%H:%M:%S')}))
            print('done', unit['id'], flush=True)
        elif attempts >= a.max_attempts:
            (d / 'gave_up').write_text(str(rc))
            print('GAVE UP', unit['id'], rc, flush=True)
        else:
            for f in ('owner', 'heartbeat'):
                (d / f).unlink(missing_ok=True)
            for t in d.glob('takeover_*'):
                shutil.rmtree(t, ignore_errors=True)
            print('FAILED (will retry)', unit['id'], rc, flush=True)
            # release by letting the heartbeat go stale immediately
            os.utime(d, (0, 0))
        ran += 1
    print('pool worker {} finished: {} units'.format(owner, ran), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    u = sub.add_parser('units')
    u.add_argument('--suite', required=True)
    u.add_argument('--artifact_root', default='/root/yieldsat_artifacts')
    u.add_argument('--skip_done_root', default=None)
    u.add_argument('--chunk', type=int, default=3)
    u.add_argument('--out', default=None)
    u.set_defaults(func=cmd_units)
    q = sub.add_parser('pool')
    q.add_argument('--units', required=True)
    q.add_argument('--artifact_root', default='/data/YieldSAT/yieldsat_artifacts')
    q.add_argument('--out_root', default=None)
    q.add_argument('--source_root', default='/data/YieldSAT/preprocessed')
    q.add_argument('--stage_from', default=None,
                   help='copy each unit\'s "stage" paths from this artifact root to --artifact_root first')
    q.add_argument('--stale_minutes', type=int, default=45)
    q.add_argument('--max_attempts', type=int, default=2)
    q.set_defaults(func=cmd_pool)
    a = p.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
