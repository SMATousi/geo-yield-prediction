"""Season dates for US points (D16): seeding and harvest per point.

Primary: USDA NASS Crop Progress (QuickStats API, key in ``NASS_API_KEY``): the
first week-ending date at which a state x crop x year reaches 50% planted and
50% harvested. Fallback (no key, or no Crop Progress series for the crop/state):
fixed crop-group windows below, flagged ``calendar_fallback``. Harvest year =
the sample year; winter crops are seeded in the previous autumn.
"""
import datetime
import json
import os
import time
from pathlib import Path

import requests

# (seeding month-day, harvest month-day, seeding in previous year)
FALLBACK = {
    'corn': ('05-01', '10-15', False),
    'soybean': ('05-15', '10-15', False),
    'winter_wheat': ('10-01', '07-01', True),
    'spring_wheat': ('04-25', '08-20', False),
    'canola': ('05-05', '08-25', False),
    'other_crop': ('04-01', '10-31', False),
}
# (commodity, planted short_desc prefix, harvested short_desc prefix); verified against the
# QuickStats PROGRESS series 2026-10-06 (corn planting is 'CORN - ...', harvest 'CORN, GRAIN - ...')
QS_COMMODITY = {'corn': ('CORN', 'CORN - PROGRESS', 'CORN, GRAIN - PROGRESS'),
                'soybean': ('SOYBEANS', 'SOYBEANS - PROGRESS', 'SOYBEANS - PROGRESS'),
                'winter_wheat': ('WHEAT', 'WHEAT, WINTER - PROGRESS', 'WHEAT, WINTER - PROGRESS'),
                'spring_wheat': ('WHEAT', 'WHEAT, SPRING, (EXCL DURUM) - PROGRESS', 'WHEAT, SPRING, (EXCL DURUM) - PROGRESS'),
                'canola': ('CANOLA', 'CANOLA - PROGRESS', 'CANOLA - PROGRESS')}
QS_URL = 'https://quickstats.nass.usda.gov/api/api_GET/'
STATE_NAMES = {'AL': 'ALABAMA', 'AZ': 'ARIZONA', 'AR': 'ARKANSAS', 'CA': 'CALIFORNIA', 'CO': 'COLORADO',
               'CT': 'CONNECTICUT', 'DE': 'DELAWARE', 'FL': 'FLORIDA', 'GA': 'GEORGIA', 'ID': 'IDAHO',
               'IL': 'ILLINOIS', 'IN': 'INDIANA', 'IA': 'IOWA', 'KS': 'KANSAS', 'KY': 'KENTUCKY',
               'LA': 'LOUISIANA', 'ME': 'MAINE', 'MD': 'MARYLAND', 'MA': 'MASSACHUSETTS', 'MI': 'MICHIGAN',
               'MN': 'MINNESOTA', 'MS': 'MISSISSIPPI', 'MO': 'MISSOURI', 'MT': 'MONTANA', 'NE': 'NEBRASKA',
               'NV': 'NEVADA', 'NH': 'NEW HAMPSHIRE', 'NJ': 'NEW JERSEY', 'NM': 'NEW MEXICO', 'NY': 'NEW YORK',
               'NC': 'NORTH CAROLINA', 'ND': 'NORTH DAKOTA', 'OH': 'OHIO', 'OK': 'OKLAHOMA', 'OR': 'OREGON',
               'PA': 'PENNSYLVANIA', 'RI': 'RHODE ISLAND', 'SC': 'SOUTH CAROLINA', 'SD': 'SOUTH DAKOTA',
               'TN': 'TENNESSEE', 'TX': 'TEXAS', 'UT': 'UTAH', 'VT': 'VERMONT', 'VA': 'VIRGINIA',
               'WA': 'WASHINGTON', 'WV': 'WEST VIRGINIA', 'WI': 'WISCONSIN', 'WY': 'WYOMING', 'DC': 'DISTRICT OF COLUMBIA'}


def cdl_group(code):
    from yieldsat_us.regions import GROUP_CODES
    for g, codes in GROUP_CODES.items():
        if code in codes:
            return 'spring_wheat' if code in (22, 23) else g
    return 'other_crop'


def fallback(group, year):
    s, h, prev = FALLBACK[group]
    sd = datetime.date(year - 1 if prev else year, int(s[:2]), int(s[3:]))
    hd = datetime.date(year, int(h[:2]), int(h[3:]))
    return sd, hd


class QuickStatsError(RuntimeError):
    """Transient API failure (throttling, HTTP error); never cached as a fallback."""


_QS_CACHE = {}


def _qs_progress(key, commodity, state, year, pause=1.5, retries=6):
    """All PROGRESS rows of a commodity/state/year (one request; throttled politely)."""
    k = (commodity, state, year)
    if k in _QS_CACHE:
        return _QS_CACHE[k]
    params = {'key': key, 'source_desc': 'SURVEY', 'statisticcat_desc': 'PROGRESS', 'agg_level_desc': 'STATE',
              'commodity_desc': commodity, 'state_name': STATE_NAMES[state], 'year': year, 'format': 'JSON'}
    for i in range(retries):
        time.sleep(pause)
        try:
            r = requests.get(QS_URL, params=params, timeout=90)
        except requests.RequestException:
            r = None
        if r is not None and r.status_code == 200:
            _QS_CACHE[k] = r.json().get('data', [])
            return _QS_CACHE[k]
        if r is not None and r.status_code == 400 and 'no data' in r.text.lower():
            _QS_CACHE[k] = []           # QuickStats answers 400 "bad request - no data" for empty queries
            return _QS_CACHE[k]
        time.sleep(30 * (i + 1))
    raise QuickStatsError('QuickStats failed for {} {} {}'.format(commodity, state, year))


def _first_50pct(rows, short_desc):
    for x in sorted((x for x in rows if x.get('short_desc') == short_desc), key=lambda x: x['week_ending']):
        try:
            v = float(x['Value'].replace(',', ''))
        except ValueError:
            continue
        if v >= 50:
            return datetime.date.fromisoformat(x['week_ending'])
    return None


class SeasonCalendar:
    """Cached seeding/harvest lookup per (state, crop group, year)."""

    def __init__(self, cache_path, key=None):
        self.path = Path(cache_path)
        self.cache = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.key = key if key is not None else os.environ.get('NASS_API_KEY')

    def get(self, state, group, year):
        k = '{}|{}|{}'.format(state, group, year)
        if k not in self.cache:
            rec = None
            if self.key and group in QS_COMMODITY:
                com, p_prefix, h_prefix = QS_COMMODITY[group]
                # QuickStats files progress under the crop (harvest) year, including the
                # autumn planting of winter wheat (2023 crop: weeks of Sep-Nov 2022, year=2023).
                # QuickStatsError propagates: a transient failure is never cached as a fallback.
                rows = _qs_progress(self.key, com, state, year)
                sd = _first_50pct(rows, p_prefix + ', MEASURED IN PCT PLANTED')
                hd = _first_50pct(rows, h_prefix + ', MEASURED IN PCT HARVESTED')
                if sd and hd and sd < hd:
                    rec = {'seeding': sd.isoformat(), 'harvest': hd.isoformat(), 'source': 'nass_crop_progress_50pct'}
            if rec is None:
                sd, hd = fallback(group, year)
                rec = {'seeding': sd.isoformat(), 'harvest': hd.isoformat(), 'source': 'calendar_fallback'}
            self.cache[k] = rec
            self.path.write_text(json.dumps(self.cache, indent=1, sort_keys=True))
        r = self.cache[k]
        return datetime.date.fromisoformat(r['seeding']), datetime.date.fromisoformat(r['harvest']), r['source']
