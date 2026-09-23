import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _write_config(name, data, overwrite):
    path = ROOT / 'data' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f'{path} already exists; pass overwrite=True to replace it')
    with path.open('w', encoding='utf-8') as write_file:
        json.dump(data, write_file)


def build_soybean_train(target_fips=None, overwrite=False):
    csv_path = ROOT / 'input' / 'county_info_2021.csv'
    df = pd.read_csv(csv_path)

    if target_fips:
        df = df[df["FIPS"].isin(target_fips)]

    counties = df.to_json(orient='records', lines=False)
    counties = json.loads(counties)

    data = []
    for county_info in counties:
        obj = get_json_obj(2021, county_info)
        data.append(obj)

    _write_config('soybean_train.json', data, overwrite)


def build_soybean_val(target_fips=None, overwrite=False):
    csv_path = ROOT / 'input' / 'county_info_2022.csv'
    df = pd.read_csv(csv_path)

    if target_fips:
        df = df[df["FIPS"].isin(target_fips)]

    counties = df.to_json(orient='records', lines=False)
    counties = json.loads(counties)

    data = []
    for county_info in counties:
        obj = get_json_obj(2022, county_info)
        data.append(obj)

    _write_config('soybean_val.json', data, overwrite)


def get_json_obj(year, county_info):
    fips, county, state = str(county_info["FIPS"]), county_info["County"], county_info["State"]

    short_term = get_short_HRRR_obj(state, fips, year, months=[i for i in range(4, 10)])

    long_term_years = ["2017", "2018", "2019", "2020", "2021"]
    long_term = get_long_HRRR_obj(state, fips, years=long_term_years)

    obj = {
        "FIPS": fips,
        "year": int(year),
        "county": county,
        "state": state,
        "county_ansi": fips[2:],
        "state_ansi": fips[:2],
        "data": {
            "HRRR": {
                "short_term": short_term,
                "long_term": long_term,
            },
            "USDA": "USDA/data/Soybean/{}/USDA_Soybean_County_{}.csv".format(year, year),
            "sentinel": [
                "Sentinel-2 Imagery/data/{}/{}/Agriculture_{}_{}_{}-04-01_{}-06-30.h5".format(year, state, fips[:2],
                                                                                              state, year, year),
                "Sentinel-2 Imagery/data/{}/{}/Agriculture_{}_{}_{}-07-01_{}-09-30.h5".format(year, state, fips[:2],
                                                                                              state, year, year),
            ]
        }

    }

    return obj


def get_short_HRRR_obj(state, fips, year, months=[i + 1 for i in range(12)]):
    file_paths = []
    for month in months:
        month = str(month).zfill(2)
        path = "WRF-HRRR/data/{}/{}/HRRR_{}_{}_{}-{}.csv".format(year, state, fips[:2], state, year, month)
        file_paths.append(path)
    return file_paths


def get_long_HRRR_obj(state, fips, years, months=[i + 1 for i in range(12)]):
    file_paths = []
    for year in years:
        year_paths = []
        for month in months:
            month = str(month).zfill(2)
            path = "WRF-HRRR/data/{}/{}/HRRR_{}_{}_{}-{}.csv".format(year, state, fips[:2], state, year, month)
            year_paths.append(path)
        file_paths.append(year_paths)
    return file_paths


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build soybean train/validation JSON')
    parser.add_argument('--overwrite', action='store_true', help='replace existing JSON files')
    args = parser.parse_args()
    target_fips = ["22007", "22121", "22043", "22107", "28089", "28015", "17091", "17155", "19117", "19135"]
    target_fips = list(map(int, target_fips))

    build_soybean_train(target_fips=target_fips, overwrite=args.overwrite)
    build_soybean_val(target_fips=target_fips, overwrite=args.overwrite)
