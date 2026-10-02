"""Artifact integrity and reviewed-library contracts. No network side effects."""
import hashlib
import json
from pathlib import Path

STREAMS = {'spec': 'optical', 'series': 'optical', 'weather': 'weather',
           'dem': 'terrain', 'terrain': 'terrain', 'soil': 'soil'}
ENCODERS = ('spec_enc', 'series_cell', 'series_enc', 'weather_enc',
            'dem_enc', 'terrain_enc', 'soil_cell', 'soil_enc')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.partial')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(p)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def approved(review):
    return (review.get('status') == 'approved' and bool(review.get('reviewer'))
            and bool(review.get('reviewed_at')) and bool(review.get('evidence')))


def library(path, require_approved=False):
    lib = read(path)
    if lib.get('schema_version') != 1:
        raise ValueError('unsupported library schema')
    concepts = {c['id']: c for c in lib['concepts']}
    if len(concepts) != len(lib['concepts']) or len({r['id'] for r in lib['rules']}) != len(lib['rules']):
        raise ValueError('duplicate concept/rule ID')
    from dataset.yieldsat_schema import TEMPORAL_CHANNELS, STATIC_CHANNELS, OPTICAL, WEATHER, DEM, TERRAIN, SOIL
    allowed = set(TEMPORAL_CHANNELS + STATIC_CHANNELS)
    forbidden = {'coord_x', 'coord_y', 'coord_z'} | {c for c in allowed if 'uncertainty' in c}
    ownership = {'spec': set(OPTICAL)-{'B02','B03','B04'}, 'series': set(OPTICAL),
                 'weather': set(WEATHER), 'dem': set(DEM), 'terrain': set(TERRAIN), 'soil': set(SOIL)}
    for c in concepts.values():
        if c['stream'] not in STREAMS or c['family'] != STREAMS[c['stream']]:
            raise ValueError('incorrect stream/family: ' + c['id'])
        if not c['channels'] or not set(c['channels']) <= allowed - forbidden:
            raise ValueError('unsupported evidence channels: ' + c['id'])
        if not set(c['channels']) <= ownership[c['stream']]:
            raise ValueError('concept channels do not belong to its encoder: ' + c['id'])
        if not all(c.get(k) for k in ('description', 'support', 'window', 'unknown_if')):
            raise ValueError('incomplete concept: ' + c['id'])
        if require_approved and not approved(c.get('review', {})):
            raise ValueError('concept requires expert review: ' + c['id'])
    for r in lib['rules']:
        a, b = concepts[r['concept_a']], concepts[r['concept_b']]
        if a['family'] == b['family']:
            raise ValueError('relation endpoints must have different sensor families')
        if not isinstance(r.get('minimum_presence'), (int,float)) or not 0 <= r['minimum_presence'] <= 1:
            raise ValueError('invalid minimum_presence')
        if r['scorer'] != 'directed_displacement_v1':
            raise ValueError('only validated directed displacement is implemented')
        if not all(r.get(k) for k in ('statement', 'qualifications', 'required_context')):
            raise ValueError('incomplete rule: ' + r['id'])
        if require_approved and not approved(r.get('review', {})):
            raise ValueError('rule/scorer requires expert review: ' + r['id'])
    return lib


def safe_child(root, name):
    root = Path(root).resolve()
    p = (root / name).resolve()
    if not p.is_relative_to(root):
        raise ValueError('artifact path escapes its root')
    return p
