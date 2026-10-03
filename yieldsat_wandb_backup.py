"""Back up W&B artifacts of one type (default: model checkpoints) to a directory.

Each artifact's latest version is downloaded to
``<out>/<rel_path>/<artifact name>/``, using the run's results path recorded
in the artifact metadata (``job.rel_path``). Retried runs share a rel_path but
keep separate folders. Existing complete copies are skipped, so
the backup can be resumed. A ``manifest.jsonl`` records artifact name,
version, digest, size and files; ``--verify`` checks every manifest entry
against the files on disk (count and bytes) without downloading.

    python yieldsat_wandb_backup.py --project yieldsat-cvpr27 --out /data/.../before_full_checkpoints
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def target_dir(out, art):
    job = (art.metadata or {}).get('job') or {}
    rel = job.get('rel_path') or 'unknown'
    # one folder per artifact: retried runs share a rel_path but have their own artifact
    return Path(out) / rel / art.name.split(':')[0]


def backup_one(art, out):
    d = target_dir(out, art)
    marker = d / '.wandb_artifact.json'
    if marker.exists():
        return 'skip', json.loads(marker.read_text())
    d.mkdir(parents=True, exist_ok=True)
    art.download(root=str(d))
    files = sorted(p for p in d.iterdir() if p.is_file() and not p.name.startswith('.'))
    rec = {'artifact': art.qualified_name, 'version': art.version, 'digest': art.digest, 'size': art.size,
           'dir': str(d), 'files': {p.name: p.stat().st_size for p in files}}
    if sum(rec['files'].values()) < art.size * 0.99:
        raise RuntimeError('incomplete download for {}'.format(art.qualified_name))
    tmp = marker.with_suffix('.partial')
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, marker)
    return 'ok', rec


def delete_all(project, kind, workers=8):
    """Delete every version (and its aliases) of artifact type ``kind``."""
    import wandb
    api = wandb.Api(timeout=300)
    path = '{}/{}'.format(api.default_entity, project)
    versions = [v for col in api.artifact_type(kind, path).collections() for v in col.artifacts()]
    total = sum(v.size for v in versions)
    print('deleting {} {} artifact versions ({:.2f} GB) in {}'.format(len(versions), kind, total / 1e9, path),
          flush=True)
    done = failed = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(v.delete, delete_aliases=True) for v in versions]
        for fut in as_completed(futs):
            try:
                fut.result()
                done += 1
            except Exception as exc:
                failed += 1
                print('FAIL', exc, flush=True)
            if (done + failed) % 500 == 0:
                print(done + failed, 'processed', flush=True)
    print('deleted {}, failed {}'.format(done, failed), flush=True)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--project', required=True)
    a.add_argument('--type', default='model')
    a.add_argument('--out', required=True)
    a.add_argument('--workers', type=int, default=8)
    a.add_argument('--verify', action='store_true')
    a.add_argument('--delete', action='store_true',
                   help='DELETE every version of this artifact type in the project (needs --confirm)')
    a.add_argument('--confirm', default='', help='must equal --project for --delete')
    args = a.parse_args()
    if args.delete:
        if args.confirm != args.project:
            raise SystemExit('--delete needs --confirm {}'.format(args.project))
        delete_all(args.project, args.type, args.workers)
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.verify:
        n = bad = size = 0
        for marker in out.rglob('.wandb_artifact.json'):
            rec = json.loads(marker.read_text())
            n += 1
            for name, sz in rec['files'].items():
                f = marker.parent / name
                if not f.exists() or f.stat().st_size != sz:
                    bad += 1
                    print('BAD', f)
                size += sz
        print('verified {} artifacts, {:.2f} GB, {} bad files'.format(n, size / 1e9, bad))
        return
    import wandb
    api = wandb.Api(timeout=300)
    path = '{}/{}'.format(api.default_entity, args.project)
    arts = []
    for col in api.artifact_type(args.type, path).collections():
        for v in col.artifacts():
            if 'latest' in v.aliases:
                arts.append(v)
                break
    print('{} {} artifacts in {}'.format(len(arts), args.type, path), flush=True)
    counts = {'ok': 0, 'skip': 0, 'fail': 0}
    with open(out / 'manifest.jsonl', 'a') as man, ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(backup_one, art, out): art for art in arts}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                status, rec = fut.result()
                counts[status] += 1
                if status == 'ok':
                    man.write(json.dumps(rec) + '\n')
                    man.flush()
            except Exception as exc:
                counts['fail'] += 1
                print('FAIL', futs[fut].qualified_name, exc, flush=True)
            if i % 200 == 0:
                print(i, counts, flush=True)
    print('done', counts, flush=True)


if __name__ == '__main__':
    main()
