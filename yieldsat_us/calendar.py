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
QS_COMMODITY = {'corn': ('CORN', 'CORN, GRAIN'), 'soybean': ('SOYBEANS', 'SOYBEANS'),
                'winter_wheat': ('WHEAT', 'WHEAT, WINTER'), 'spring_wheat': ('WHEAT', 'WHEAT, SPRING, (EXCL DURUM)'),
                'canola': ('CANOLA', 'CANOLA')}
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


def _qs_50pct(key, commodity, desc_prefix, state, year, unit):
    params = {'key': key, 'source_desc': 'SURVEY', 'statisticcat_desc': 'PROGRESS', 'agg_level_desc': 'STATE',
              'commodity_desc': commodity, 'state_name': STATE_NAMES[state], 'unit_desc': unit,
              'year': year if unit != 'PCT PLANTED' or not desc_prefix.startswith('WHEAT, WINTER') else year - 1,
              'format': 'JSON'}
    r = requests.get(QS_URL, params=params, timeout=60)
    if r.status_code != 200:
        return None
    rows = [x for x in r.json().get('data', []) if x.get('short_desc', '').startswith(desc_prefix)]
    best = None
    for x in sorted(rows, key=lambda x: x['week_ending']):
        try:
            v = float(x['Value'].replace(',', ''))
        except ValueError:
            continue
        if v >= 50:
            best = datetime.date.fromisoformat(x['week_ending'])
            break
    return best


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
                com, prefix = QS_COMMODITY[group]
                try:
                    sd = _qs_50pct(self.key, com, prefix, state, year, 'PCT PLANTED')
                    hd = _qs_50pct(self.key, com, prefix, state, year, 'PCT HARVESTED')
                    time.sleep(0.3)
                except requests.RequestException:
                    sd = hd = None
                if sd and hd and sd < hd:
                    rec = {'seeding': sd.isoformat(), 'harvest': hd.isoformat(), 'source': 'nass_crop_progress_50pct'}
            if rec is None:
                sd, hd = fallback(group, year)
                rec = {'seeding': sd.isoformat(), 'harvest': hd.isoformat(), 'source': 'calendar_fallback'}
            self.cache[k] = rec
            self.path.write_text(json.dumps(self.cache, indent=1, sort_keys=True))
        r = self.cache[k]
        return datetime.date.fromisoformat(r['seeding']), datetime.date.fromisoformat(r['harvest']), r['source']
