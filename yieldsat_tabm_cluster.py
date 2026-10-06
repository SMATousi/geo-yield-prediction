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


def stage_image(spec, src_img, local_root):
    """Image units: stage the tiles of spec['countries'] (only spec['crops'] tiles, uncompressed and
    sparse, when crops are given; the compressed files otherwise), their patch tables and DINO
    features under <local_root>/<spec['key']>; other keys' roots are removed first (one at a time
    per pod). Returns the local image root."""
    import fcntl
    from models_yieldsat_image import DINO_REVISION, preprocessing_hash
    from yieldsat_cluster import _copy_once, _stage_tiles_sparse
    from yieldsat_image_dino_cache import cache_tag
    base, src = Path(local_root), Path(src_img)
    dst = base / spec['key']
    base.mkdir(parents=True, exist_ok=True)
    with open(base / '.stage.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (dst / '.staged').exists():
            for d in base.iterdir():
                if d.is_dir() and d != dst:
                    shutil.rmtree(d, ignore_errors=True)
            dino = Path('dino_cache') / cache_tag(DINO_REVISION, preprocessing_hash())
            _copy_once(src / 'manifest.json', dst / 'manifest.json')
            _copy_once(src / dino / 'manifest.json', dst / dino / 'manifest.json')
            for c in spec['countries']:
                for rel in (Path(c) / 'patches.jsonl', dino / '{}.h5'.format(c)):
                    _copy_once(src / rel, dst / rel)
                if spec.get('crops'):
                    keep = [json.loads(line)['patch_index'] for line in open(src / c / 'patches.jsonl')
                            if json.loads(line)['crop'] in spec['crops']]
                    _stage_tiles_sparse(src / c / 'images.h5', dst / c / 'images.h5', keep)
                else:
                    _copy_once(src / c / 'images.h5', dst / c / 'images.h5')
            (dst / '.staged').write_text(time.strftime('%Y-%m-%dT%H:%M:%S'))
    return str(dst)


def gpu_ok():
    """Fail fast on a broken GPU: a pod whose CUDA cannot initialize would otherwise fail
    and burn every unit it claims (2026-10-06, fiona-prg1.cesnet.cz)."""
    try:
        import torch
        x = torch.ones(1024, device='cuda')
        return bool((x * 2).sum().item() == 2048)
    except Exception as exc:  # noqa: BLE001
        print('GPU health check failed:', repr(exc), flush=True)
        return False


def cmd_pool(a):
    if a.concurrency > 1:
        # several independent pool workers share the pod's GPU: small point-model runs use ~1 GB and
        # 0-20% of a GPU each, which the cluster's utilization check flags (2026-10-06)
        args = [x for x in sys.argv[1:]]
        i = args.index('--concurrency')
        del args[i:i + 2]
        procs = [subprocess.Popen([sys.executable, sys.argv[0], *args],
                                  env=dict(os.environ, POOL_SLOT=str(k))) for k in range(a.concurrency)]
        sys.exit(max(p.wait() for p in procs))
    if not a.skip_gpu_check and not gpu_ok():
        sys.exit(3)
    plan = json.loads(Path(a.units).read_text())
    out_root = a.out_root or plan['out_root']
    owner = os.environ.get('HOSTNAME', os.uname().nodename) + (
        '-s' + os.environ['POOL_SLOT'] if os.environ.get('POOL_SLOT') else '')
    claims = Claims(Path(out_root) / '_claims', owner, a.stale_minutes)
    ran = 0
    while True:
        # a GPU can also fail mid-run (2026-10-06, nrp-01.laccd.edu: 'unspecified launch failure', then
        # every later unit failed at CUDA init); stop claiming instead of burning the queue
        if ran and not a.skip_gpu_check and subprocess.run(
                [sys.executable, '-c', 'import sys, yieldsat_tabm_cluster as c; sys.exit(0 if c.gpu_ok() else 1)'],
                timeout=300).returncode != 0:
            print('GPU no longer healthy; pool worker exits without claiming', flush=True)
            sys.exit(3)
        unit = None
        for u in plan['units']:
            if claims.try_claim(u['id']):
                unit = u
                break
        if unit is None:
            break
        d = claims.d(unit['id'])
        try:        # tolerate a concurrently written / empty file on the shared volume
            attempts = int((d / 'attempts').read_text().strip() or 0) + 1
        except (OSError, ValueError):
            attempts = 1
        (d / 'attempts').write_text(str(attempts))
        if unit.get('stage') and a.stage_from:
            stage(unit['stage'], a.stage_from, a.artifact_root)
        if unit.get('skip_if_exists'):
            # a follow-up unit whose result another unit id already produced (same output path)
            done_path = Path(unit['skip_if_exists'].format(out_root=out_root))
            if done_path.exists():
                (d / 'done').write_text(json.dumps({'owner': owner, 'skipped_existing': str(done_path)}))
                print('exists, skipped', unit['id'], flush=True)
                continue
        image_root = None
        if unit.get('stage_image'):
            image_root = stage_image(unit['stage_image'], a.stage_image_from, a.image_local)
        if 'script' in unit:
            # generic unit: full argument list with {artifact_root} / {out_root} / {source_root} /
            # {image_root} placeholders
            fmt = dict(artifact_root=a.artifact_root, out_root=out_root, source_root=a.source_root,
                       image_root=image_root)
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
    q.add_argument('--skip_gpu_check', action='store_true')
    q.add_argument('--stage_from', default=None,
                   help='copy each unit\'s "stage" paths from this artifact root to --artifact_root first')
    q.add_argument('--stage_image_from', default='/data/YieldSAT/YieldSAT-Image-full',
                   help='image units: PVC image root to stage tiles from')
    q.add_argument('--image_local', default='/scratch/img', help='image units: local staging root')
    q.add_argument('--concurrency', type=int, default=1, help='pool workers per pod sharing the GPU')
    q.add_argument('--stale_minutes', type=int, default=45)
    q.add_argument('--max_attempts', type=int, default=2)
    q.set_defaults(func=cmd_pool)
    a = p.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
