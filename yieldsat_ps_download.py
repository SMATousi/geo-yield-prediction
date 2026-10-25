"""PlanetScope for a YieldSAT subset (spec/yieldsat-planetscope.md): ARG-S, ARG-C, BRA-C.

Reuses the selection and order logic of geo-data-downloader (datasets/planetscope.py) with the settings of our
Missouri run: one scene per 7-day bin (full AOI coverage, then 8-band, then clear %, then nearest the bin
centre), cloud cover <= 0.1, 8-band SR where offered and 4-band otherwise, clipped to the AOI. Differences:

- AOI: the field's exact UTM grid (YieldSAT geometry) + 30 m, so the 30 m window of edge cells is covered;
  written by the AOI step to aois_ARG-S_ARG-C_BRA-C.geojson.
- Window: seeding - 10 d to harvest + 10 d (YieldSAT masks slots outside [seeding, harvest]).
- Areas in South America Albers (ESRI:102033), not CONUS Albers.

Output per field season, as the Missouri layers: <out>/<field_shared_name>/planetscope/{planetscope_index.csv,
<date>_<item>_sr.tif, _udm2.tif, _metadata.json, _order.json}. An order id is written before polling, so a
rerun resumes it instead of ordering twice.

Run with the geo-data-downloader environment; the key only via the environment:
  PL_API_KEY=... PYTHONPATH=/root/geo-data-downloader /root/geo-data-downloader/.venv/bin/python \
      yieldsat_ps_download.py select|order --out /root/yieldsat_artifacts/planetscope/scenes [--workers 8]
"""
import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import mapping, shape

import geo_data_downloader.datasets.planetscope as P
from geo_data_downloader.http.client import HttpClient, RetryPolicy
from geo_data_downloader.http.planet import PlanetClient

P._TO_ALBERS = Transformer.from_crs('EPSG:4326', 'ESRI:102033', always_xy=True).transform
P._TO_WGS84 = Transformer.from_crs('ESRI:102033', 'EPSG:4326', always_xy=True).transform

AOIS = Path('/root/yieldsat_artifacts/planetscope/aois_ARG-S_ARG-C_BRA-C.geojson')
CADENCE, MAX_CLOUD, PAD_DAYS, TIMEOUT_S, POLL_S = 7, 0.1, 10, 12 * 3600, 60
# Planet answers bursts of order creation with HTTP 429: creation is serialized and spaced, and every request
# backs off on 429 for up to ~1 h
CREATE_GAP_S = 2.0
_create_lock, _last_create = threading.Lock(), [0.0]


def client():
    return PlanetClient(HttpClient(policy=RetryPolicy(max_attempts=15, initial_backoff=5.0, max_backoff=300.0)))
BUNDLE_8B, BUNDLE_4B = 'analytic_8b_sr_udm2', 'analytic_sr_udm2'
INDEX_COLS = ['bin', 'date', 'acquired', 'item_id', 'satellite_id', 'instrument', 'cloud_cover', 'clear_percent',
              'aoi_coverage', 'bands_ordered', 'expected_charge_km2', 'delivered', 'bands']


def fields():
    for f in json.load(open(AOIS))['features']:
        p = f['properties']
        start = date.fromisoformat(p['seeding_date']) - timedelta(days=PAD_DAYS)
        end = date.fromisoformat(p['harvesting_date']) + timedelta(days=PAD_DAYS)
        yield p['field_shared_name'], p['pair'], shape(f['geometry']), start, end


def say(name, msg):
    print('{} {}: {}'.format(datetime.now(timezone.utc).strftime('%H:%M:%S'), name, msg), file=sys.stderr, flush=True)


def select_one(client, name, aoi, start, end):
    feats = list(client.search(['PSScene'], P.search_filter(aoi, start, end, MAX_CLOUD, P.ASSET_4B)))
    cands = P.candidates_from(feats, aoi)
    sel, empty = P.select_scenes(cands, aoi, start, end, CADENCE, prefer_8b=True)
    rows = [{'bin': k, 'date': c.day.isoformat(), 'acquired': c.acquired.isoformat(), 'item_id': c.id,
             'satellite_id': c.satellite_id, 'instrument': c.instrument, 'cloud_cover': c.cloud_cover,
             'clear_percent': c.clear_percent, 'aoi_coverage': round(c.coverage, 4),
             'bands_ordered': 8 if c.has_8b else 4, 'expected_charge_km2': round(c.charge_km2, 6)}
            for k in sorted(sel) for c in sel[k]]
    return {'field': name, 'start': start.isoformat(), 'end': end.isoformat(), 'aoi': mapping(aoi),
            'aoi_area_km2': P.area_km2(aoi), 'candidates': len(feats), 'n_bins': len(sel) + len(empty),
            'empty_bins': empty, 'rows': rows, 'expected_charge_km2': sum(r['expected_charge_km2'] for r in rows)}


def cmd_select(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    todo = [f for f in fields() if not (out / f[0] / 'planetscope' / '_selection.json').exists()][:args.limit]
    say('select', '{} field seasons to search'.format(len(todo)))

    def job(f):
        name, pair, aoi, start, end = f
        s = dict(select_one(client(), name, aoi, start, end), pair=pair)
        d = out / name / 'planetscope'; d.mkdir(parents=True, exist_ok=True)
        (d / '_selection.json').write_text(json.dumps(s))
        return name
    with ThreadPoolExecutor(args.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(job, f) for f in todo])):
            fut.result()
            if i % 50 == 0:
                say('select', '{}/{}'.format(i + 1, len(todo)))
    summarize(out)


def summarize(out):
    import collections
    agg = collections.defaultdict(lambda: collections.Counter())
    for p in sorted(Path(out).glob('*/planetscope/_selection.json')):
        s = json.loads(p.read_text())
        year = s['end'][:4]
        for key in (s['pair'], '{} {}'.format(s['pair'], year), 'all'):
            a = agg[key]
            a['fields'] += 1; a['scenes'] += len(s['rows']); a['scenes_8b'] += sum(r['bands_ordered'] == 8 for r in s['rows'])
            a['bins'] += s['n_bins']; a['empty_bins'] += len(s['empty_bins']); a['km2'] += s['expected_charge_km2']
    res = {k: dict(v) for k, v in sorted(agg.items())}
    (Path(out) / '_selection_summary.json').write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print('{:12s} fields {:4d} scenes {:6d} 8b {:5.1%} filled bins {:5.1%} km2 {:9.1f}'.format(
            k, v['fields'], v['scenes'], v['scenes_8b'] / max(v['scenes'], 1), 1 - v['empty_bins'] / max(v['bins'], 1),
            v['km2']))


def order_one(name, d):
    cl = client()
    sel = json.loads((d / '_selection.json').read_text())
    state_p = d / '_order.json'
    if (d / 'planetscope_index.csv').exists():
        return name, 'done'
    if not sel['rows']:
        (d / 'planetscope_index.csv').write_text(','.join(INDEX_COLS) + '\n')
        return name, 'no scenes'
    if state_p.exists():
        state = json.loads(state_p.read_text()); say(name, 'resuming order ' + state['order_id'])
    else:
        eight = [r['item_id'] for r in sel['rows'] if r['bands_ordered'] == 8]
        four = [r['item_id'] for r in sel['rows'] if r['bands_ordered'] == 4]
        products = ([{'item_ids': eight, 'item_type': 'PSScene', 'product_bundle': BUNDLE_8B}] if eight else []) + \
                   ([{'item_ids': four, 'item_type': 'PSScene', 'product_bundle': BUNDLE_4B}] if four else [])
        with _create_lock:
            time.sleep(max(0.0, _last_create[0] + CREATE_GAP_S - time.monotonic()))
            try:
                order = cl.create_order({'name': 'ysat-' + name, 'order_type': 'partial', 'products': products,
                                         'tools': [{'clip': {'aoi': sel['aoi']}}]})
            finally:
                _last_create[0] = time.monotonic()
        state = {'order_id': order['id'], 'created_at': datetime.now(timezone.utc).isoformat(),
                 'expected_charge_km2': sel['expected_charge_km2']}
        state_p.write_text(json.dumps(state))
        say(name, '{} scenes, {:.2f} km2; order {}'.format(len(sel['rows']), sel['expected_charge_km2'], order['id']))
    order = cl.wait_for_order(state['order_id'], TIMEOUT_S, POLL_S)
    if order.get('state') not in ('success', 'partial'):
        raise RuntimeError('order {} ended {}'.format(state['order_id'], order.get('state')))
    got, bands = set(), {}
    by_item = {r['item_id']: r for r in sel['rows']}
    for res in order.get('_links', {}).get('results', []):
        rn = res.get('name', '')
        for suffix, label in P._KEEP:
            if rn.endswith(suffix):
                item = rn.rsplit('/', 1)[-1].split('_3B_')[0].removesuffix('_metadata.json')
                row = by_item.get(item)
                fn = d / '{}_{}_{}'.format(row['date'] if row else 'unknown', item, label)
                if not fn.exists():
                    tmp = fn.with_suffix('.part'); tmp.write_bytes(cl.download(res['location'])); tmp.rename(fn)
                if label == 'sr.tif':
                    got.add(item); bands[item] = 8 if suffix == '_SR_8b_clip.tif' else 4
    with open(d / 'planetscope_index.csv.part', 'w', newline='') as fh:
        w = csv.DictWriter(fh, INDEX_COLS); w.writeheader()
        for r in sel['rows']:
            w.writerow(dict(r, delivered=r['item_id'] in got, bands=bands.get(r['item_id'], '')))
    (d / 'planetscope_index.csv.part').rename(d / 'planetscope_index.csv')
    return name, '{}/{} scenes'.format(len(got), len(sel['rows']))


def cmd_order(args):
    out = Path(args.out)
    dirs = sorted(p.parent for p in out.glob('*/planetscope/_selection.json'))
    todo = [d for d in dirs if not (d / 'planetscope_index.csv').exists()]
    say('order', '{} of {} field seasons to order'.format(len(todo), len(dirs)))
    fails = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(order_one, d.parent.name, d): d for d in todo}
        for i, fut in enumerate(as_completed(futs)):
            try:
                name, msg = fut.result(); say(name, msg)
            except Exception as e:                       # noqa: BLE001 -- one field must not stop the run
                fails += 1; say(futs[fut].parent.name, 'FAILED {}'.format(str(e)[:300]))
            say('order', 'progress {}/{} failed {}'.format(i + 1, len(todo), fails))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['select', 'order', 'summary'])
    ap.add_argument('--out', default='/root/yieldsat_artifacts/planetscope/scenes')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    {'select': cmd_select, 'order': cmd_order, 'summary': lambda a: summarize(a.out)}[a.cmd](a)
