"""Build the YieldSAT-aligned US pretraining pilot (NP-04, spec/yieldsat-us-national-pretraining.md).

Stages (each resumable, writing under --out):
  frame    CDL per harvest year -> cluster cells, clusters (NOAA region x crop
           stratum) and ~2,000 cropland points per cluster. National CDL zips are
           downloaded one year at a time and deleted after use (D19).

  extract  per cluster: season dates (NASS Crop Progress / fallback calendar),
           pixel-only S2 (monthly, harmonized), ERA5-Land weather (YieldSAT interval
           sums), SoilGrids, NASADEM terrain -> clusters/<cluster_id>.npz

    python yieldsat_us_pilot.py frame --out /home1/pupil/SMATousi/YieldSAT-US-Pilot
    python yieldsat_us_pilot.py extract --out /home1/pupil/SMATousi/YieldSAT-US-Pilot
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from yieldsat_us.frame import CDL_URL, build_year, extract_cdl
from yieldsat_us.regions import REGIONS, YEARS

STATES_URL = 'https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_5m.zip'


def cmd_frame(a):
    out = Path(a.out)
    tmp = out / '_tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    states = tmp / 'states' / 'cb_2022_us_state_5m.zip'
    if not states.exists():
        states.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['curl', '-sf', '-o', str(states), STATES_URL], check=True)
    taken = set()
    offsets = {r: i for i, r in enumerate(REGIONS)}
    for year in a.years:
        done = out / 'frame_{}_clusters.json'.format(year)
        if done.exists():
            taken |= {c['cell'] for c in json.loads(done.read_text())['clusters']}
            print(year, 'frame exists, skipped')
            continue
        z = tmp / '{}_30m_cdls.zip'.format(year)
        if not z.exists():
            print(year, 'downloading CDL', flush=True)
            subprocess.run(['curl', '-sf', '-o', str(z) + '.partial', CDL_URL.format(y=year)], check=True)
            Path(str(z) + '.partial').rename(z)
        tif = extract_cdl(z, tmp)
        print(year, 'frame from', tif, flush=True)
        clusters = build_year(year, tif, states, out, taken, offsets)
        print(year, len(clusters), 'clusters,', sum(c['n_points'] for c in clusters), 'points', flush=True)
        if not a.keep_cdl:
            tif.unlink()
            z.unlink()
    if not a.keep_cdl and not any(tmp.glob('*_30m_cdls.*')):
        shutil.rmtree(tmp, ignore_errors=True)


def extract_one(out, cluster, pts, cal, soil_uncertainty=False):
    import datetime
    import time
    import numpy as np
    from yieldsat_us import s2, static_layers, weather
    from yieldsat_us.calendar import cdl_group
    t0 = time.time()
    year = cluster['year']
    lon, lat, cdl = pts['lon'], pts['lat'], pts['cdl']
    groups = [cdl_group(int(c)) for c in cdl]
    sd, hd, src = zip(*[cal.get(cluster['state'], g, year) for g in groups])
    sd, hd = np.array(sd), np.array(hd)
    r = s2.extract_cluster(lon, lat, sd, hd, year)
    t_s2 = time.time() - t0
    wx = weather.weather_for_points(lon, lat, r['dates'], Path(out) / '_cache' / 'era5land')
    soil = static_layers.soil_for_points(lon, lat, uncertainty=soil_uncertainty)
    ter = static_layers.terrain_for_points(lon, lat)
    epoch = datetime.date(1970, 1, 1).toordinal()
    temporal = np.concatenate([r['values'], wx], axis=2).astype(np.float32)          # (N,24,16) OPTICAL+WEATHER
    # unit-sphere XYZ like YieldSAT coord_* (metadata, not inputs)
    lam, phi = np.radians(lon), np.radians(lat)
    xyz = np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], 1)
    static = np.concatenate([ter[:, :1], ter[:, 1:5], soil, xyz], axis=1).astype(np.float32)  # DEM+TERRAIN+SOIL+UNC+COORDS
    dst = Path(out) / 'clusters' / '{}.npz'.format(cluster['cluster_id'])
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(dst) + '.partial.npz', temporal=temporal, static=static, times=r['dates'],
                        seeding_day=np.array([d.toordinal() - epoch for d in sd], np.int32),
                        harvest_day=np.array([d.toordinal() - epoch for d in hd], np.int32),
                        calendar_source=np.array(src), cdl=cdl, lon=lon, lat=lat, x=pts['x'], y=pts['y'],
                        item_id=r['item_id'], scl=r['scl'], clear=r['clear'],
                        items=np.array(json.dumps(r['items'])), cluster=np.array(json.dumps(cluster)))
    Path(str(dst) + '.partial.npz').rename(dst)
    return {'cluster_id': cluster['cluster_id'], 'points': int(len(lon)), 'seconds': round(time.time() - t0),
            's2_seconds': round(t_s2), **r['stats'],
            'dated_slots_p50': float(np.median(np.isfinite(r['dates']).sum(1))),
            'clear_frac': float(r['clear'][np.isfinite(r['dates'])].mean()) if np.isfinite(r['dates']).any() else 0.0,
            'calendar_fallback_frac': float(np.mean([x == 'calendar_fallback' for x in src]))}


def cmd_extract(a):
    import numpy as np
    from yieldsat_us.calendar import SeasonCalendar
    out = Path(a.out)
    cal = SeasonCalendar(out / 'calendar_cache.json')
    log = out / 'extract_log.jsonl'
    for year in a.years:
        fr = json.loads((out / 'frame_{}_clusters.json'.format(year)).read_text())
        pts = np.load(out / 'frame_{}_points.npz'.format(year))
        for ci, cl in enumerate(fr['clusters']):
            if a.only and cl['cluster_id'] not in a.only:
                continue
            if (out / 'clusters' / '{}.npz'.format(cl['cluster_id'])).exists():
                continue
            sel = np.flatnonzero(pts['cluster'] == ci)[:a.max_points or None]
            rec = extract_one(out, cl, {k: pts[k][sel] for k in ('lon', 'lat', 'cdl', 'x', 'y')}, cal,
                              a.soil_uncertainty)
            print(json.dumps(rec), flush=True)
            with open(log, 'a') as f:
                f.write(json.dumps(rec) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('frame')
    f.add_argument('--out', required=True)
    f.add_argument('--years', type=int, nargs='+', default=list(YEARS))
    f.add_argument('--keep_cdl', action='store_true')
    f.set_defaults(func=cmd_frame)
    e = sub.add_parser('extract')
    e.add_argument('--out', required=True)
    e.add_argument('--years', type=int, nargs='+', default=list(YEARS))
    e.add_argument('--only', nargs='*', default=None, help='cluster ids')
    e.add_argument('--max_points', type=int, default=0, help='test: first N points per cluster')
    e.add_argument('--soil_uncertainty', action='store_true')
    e.set_defaults(func=cmd_extract)
    a = p.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
