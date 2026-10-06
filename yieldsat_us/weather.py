"""ERA5-Land daily weather at the native cell containing each point (NP-04d).

Source: the Open-Meteo historical archive, queried once per ERA5-Land cell, not
per point: temperatures from ``models=era5_land`` (0.1 deg); precipitation from
``models=era5`` (0.25 deg), because Open-Meteo serves no ERA5-Land
precipitation (null). ERA5-Land's precipitation is itself the ERA5 forcing
interpolated to 0.1 deg (only temperature-related fields are elevation-
adjusted), so this is near-equivalent; recorded as ``precip_source``. YieldSAT operator: slot k holds the sum of daily values
over the inclusive interval [t_(k-1), t_k] between consecutive dated slots,
temperatures in Kelvin-days (daily max/mean/min in degC + 273.15), precipitation
in metres; the first dated slot has no interval and is stored as NaN (invalid,
as in the point contract).
"""
import datetime
import json
import time
from pathlib import Path

import numpy as np
import requests

URL = 'https://archive-api.open-meteo.com/v1/archive'
DAILY = ('temperature_2m_max', 'temperature_2m_mean', 'temperature_2m_min', 'precipitation_sum')
EPOCH = datetime.date(1970, 1, 1).toordinal()


def era5_cell(lon, lat):
    """ERA5-Land 0.1 deg cell centre (grid aligned to 0.1 deg)."""
    return np.round(np.asarray(lon) * 10) / 10, np.round(np.asarray(lat) * 10) / 10


def _get(params, retries=8):
    r = None
    for k in range(retries):
        try:
            r = requests.get(URL, params=params, timeout=120)
        except requests.RequestException as exc:       # timeouts / connection resets: retry
            r = None
            time.sleep(10 * (k + 1))
            continue
        if r.status_code == 200:
            return r.json()
        time.sleep(10 * (k + 1))                       # rate limit / transient
    raise RuntimeError('Open-Meteo failed: {}'.format('no response' if r is None else '{} {}'.format(r.status_code, r.text[:200])))


def fetch_cell(lon, lat, start, end, cache_dir):
    path = Path(cache_dir) / 'era5land_{:.1f}_{:.1f}_{}_{}.json'.format(lon, lat, start, end)
    if path.exists():
        return json.loads(path.read_text())
    base = {'latitude': lat, 'longitude': lon, 'start_date': start.isoformat(), 'end_date': end.isoformat(),
            'timezone': 'GMT'}
    t = _get(dict(base, daily=','.join(DAILY[:3]), models='era5_land'))
    p = _get(dict(base, daily='precipitation_sum', models='era5'))
    if t['daily']['time'] != p['daily']['time']:
        raise RuntimeError('temperature and precipitation day axes differ')
    daily = dict(t['daily'], precipitation_sum=p['daily']['precipitation_sum'])
    path.write_text(json.dumps({'daily': daily, 'lat': t.get('latitude'), 'lon': t.get('longitude'),
                                'precip_source': 'era5 (0.25 deg) via open-meteo',
                                'temp_source': 'era5_land (0.1 deg) via open-meteo'}))
    return json.loads(path.read_text())


def interval_sums(dates, daily_days, daily_vals):
    """dates (24,) days since 1970 (NaN undated); daily_days (D,), daily_vals (D,4) in
    YieldSAT units. Returns (24,4): inclusive-interval sums, NaN for undated slots
    and for the first dated slot."""
    out = np.full((len(dates), daily_vals.shape[1]), np.nan, dtype=np.float32)
    prev = None
    index = {int(d): i for i, d in enumerate(daily_days)}
    for k, t in enumerate(dates):
        if not np.isfinite(t):
            continue
        t = int(t)
        if prev is not None:
            idx = [index[d] for d in range(prev, t + 1) if d in index]
            if len(idx) == t - prev + 1:
                out[k] = daily_vals[idx].sum(0)
        prev = t
    return out


def weather_for_points(lon, lat, dates, cache_dir):
    """(N,24,4) [temp_max, temp_mean, temp_min (K*days), total_prec (m)] for points with
    slot dates (N,24); one archive query per ERA5-Land cell."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    clon, clat = era5_cell(lon, lat)
    out = np.full(dates.shape + (4,), np.nan, dtype=np.float32)
    cells = {}
    for i, key in enumerate(zip(clon.tolist(), clat.tolist())):
        cells.setdefault(key, []).append(i)
    for (x, y), idx in cells.items():
        d = dates[idx]
        if not np.isfinite(d).any():
            continue
        start = datetime.date.fromordinal(int(np.nanmin(d)) + EPOCH)
        end = datetime.date.fromordinal(int(np.nanmax(d)) + EPOCH)
        js = fetch_cell(x, y, start, end, cache_dir)
        days = np.array([datetime.date.fromisoformat(t).toordinal() - EPOCH for t in js['daily']['time']])
        v = np.stack([np.array(js['daily'][k], dtype=float) for k in DAILY], 1)
        v[:, :3] += 273.15
        v[:, 3] /= 1000.0
        good = np.isfinite(v).all(1)
        days, v = days[good], v[good]
        for i in idx:
            out[i] = interval_sums(dates[i], days, v)
    return out
