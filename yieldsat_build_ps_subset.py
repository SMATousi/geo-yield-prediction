"""PlanetScope 30 m windows for a YieldSAT subset (spec/yieldsat-planetscope.md): ARG-S, ARG-C, BRA-C.

The same rule and layout as our Missouri dataset (yieldsat_build_owndata.write_ps_window): per cell and dense slot,
the in-season scene nearest the bin centre whose cell is >= 90% clear; the scenes used, warped onto the field's
grid at 3.33 m padded by one cell (sub-pixels < 50% clear NaN). The field grid is YieldSAT's (geometry JSON:
EPSG, 10 m affine, width, height), which the raw S2 rasters and the index grid_row / grid_col use.

Writes into <artifact_root>/neighbourhood_dense/<Country>/:
  ps_win/<field_code>.npy   (scenes, 3(H+2), 3(W+2), 8) f16, subset field seasons only
  ps_win_scene.npy          (N, 72) i16 per country row, -1 = no PlanetScope (and every row outside the subset)
  ps_field_sums.npz         per subset field season: count, sum, sumsq of the cell mean (centre 3x3) per band
  ps_subset_manifest.json   pairs, rules, per-field scene and coverage counts
The 3x3 sub-grid stream is cut from the window by the dataset (no ps3x3.npy).

    python yieldsat_build_ps_subset.py --artifact_root /data/YieldSAT/yieldsat_artifacts \\
        --layers /data/YieldSAT/yieldsat_artifacts/planetscope/layers --workers 14
"""
import argparse
import json
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS

from yieldsat_build_owndata import N_SLOTS, PS_BANDS, PS_CELL_CLEAR, PS_SUB_CLEAR, SUB, _ps_window

SUBSET = {'ARG-S': ('Argentina', 'soybean'), 'ARG-C': ('Argentina', 'corn'), 'BRA-C': ('Brazil', 'corn')}
GEOMETRY = 'geometry/fields_geometry_Argentina-Brazil-Germany-Uruguay.json'


def _task_sums(stack, scene, rows, cols):
    """count, sum, sumsq (8,) of the cell mean (centre 3x3 sub-pixels) over every cell-slot with a scene."""
    cnt, s, ss = np.zeros(8), np.zeros(8), np.zeros(8)
    if not len(stack):
        return cnt, s, ss
    bi, k = np.nonzero(scene >= 0)
    for j0 in range(0, len(bi), 200000):
        b, kk = bi[j0:j0 + 200000], k[j0:j0 + 200000]
        R = (3 * (rows[b] + 1))[:, None] + np.arange(3)
        C = (3 * (cols[b] + 1))[:, None] + np.arange(3)
        v = np.asarray(stack[scene[b, kk][:, None, None], R[:, :, None], C[:, None, :]], np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)            # all-NaN bands of 4-band scenes
            m = np.nanmean(v.reshape(len(b), 9, 8), axis=1)
        ok = np.isfinite(m)
        cnt += ok.sum(0); s += np.where(ok, m, 0).sum(0); ss += np.where(ok, m * m, 0).sum(0)
    return cnt, s, ss


def _work(task):
    name, code, country, ws, f, rows, cols, grid = task
    _, scene, days, stack = _ps_window((ws, f, rows, cols, grid))
    sums = _task_sums(stack, scene, rows, cols)
    return name, code, country, scene, stack, sums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact_root', required=True)
    ap.add_argument('--layers', required=True, help='<layers>/<field_shared_name>/planetscope/ (yieldsat_ps_download.py)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--only_downloaded', action='store_true', help='smoke test: only field seasons already downloaded')
    a = ap.parse_args()
    out = Path(a.artifact_root)
    ws = Path(a.layers).parent
    assert (ws / 'layers').resolve() == Path(a.layers).resolve(), 'the layers directory must be named layers'
    geo = {g['field_shared_name']: g for g in json.loads((out / GEOMETRY).read_text())}
    manifest = {'pairs': sorted(SUBSET), 'n_slots': N_SLOTS, 'sub_cells': SUB * SUB, 'cell_clear': PS_CELL_CLEAR,
                'sub_clear': PS_SUB_CLEAR, 'bands': list(PS_BANDS), 'fields': {}}
    for country in sorted({c for c, _ in SUBSET.values()}):
        crops = {cr for c, cr in SUBSET.values() if c == country}
        idx = np.load(out / 'index' / country / 'rows.npz')
        recs = [r for r in json.loads((out / 'index' / country / 'fields.json').read_text()) if r['crop'] in crops]
        if a.only_downloaded:
            recs = [r for r in recs if (ws / 'layers' / r['field_shared_name'] / 'planetscope' / 'planetscope_index.csv').exists()]
        d = out / 'neighbourhood_dense' / country
        (d / 'ps_win').mkdir(parents=True, exist_ok=True)
        scene_all = np.full((len(idx['field_code']), N_SLOTS), -1, np.int16)
        tasks = []
        for r in recs:
            g = geo[r['field_shared_name']]
            s, e = r['row_start'], r['row_end']
            rows, cols = idx['grid_row'][s:e].astype(np.int64), idx['grid_col'][s:e].astype(np.int64)
            assert (idx['field_code'][s:e] == r['field_code']).all()
            assert rows.max() < g['height'] and cols.max() < g['width'], r['field_shared_name']
            grid = (rasterio.Affine(*g['transform']), (g['height'], g['width']), CRS.from_epsg(g['epsg']))
            f = {'vgir_id': r['field_shared_name'], 'planting_day': int(r['seeding_day']),
                 'harvest_day': int(r['harvest_day']), 'year': int(r['year'])}
            tasks.append((r['field_shared_name'], r['field_code'], country, str(ws), f, rows, cols, grid))
        tasks.sort(key=lambda t: -len(t[5]))                     # large fields first
        pos = {r['field_shared_name']: r for r in recs}
        codes, cnt, sm, ssq = [], [], [], []
        with Pool(a.workers) as pool:
            for i, (name, code, _, scene, stack, sums) in enumerate(pool.imap_unordered(_work, tasks)):
                r = pos[name]
                scene_all[r['row_start']:r['row_end']] = scene
                np.save(d / 'ps_win' / '{}.npy'.format(code), stack)
                codes.append(code); cnt.append(sums[0]); sm.append(sums[1]); ssq.append(sums[2])
                cov = float((scene >= 0).any(1).mean())
                manifest['fields'][name] = {'field_code': int(code), 'scenes': int(len(stack)), 'cells': int(len(scene)),
                                            'cell_slots_with_ps': int((scene >= 0).sum()), 'cells_with_any_ps': cov}
                print('{} {}/{} {}: {} scenes, {} MB, cells with PS {:.1%}'.format(
                    country, i + 1, len(tasks), name, len(stack), stack.nbytes >> 20, cov), flush=True)
        np.save(d / 'ps_win_scene.npy', scene_all)
        np.savez(d / 'ps_field_sums.npz', field_codes=np.array(codes), bands=np.array(PS_BANDS),
                 count=np.array(cnt), sum=np.array(sm), sumsq=np.array(ssq))
    (out / 'neighbourhood_dense' / 'ps_subset_manifest.json').write_text(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
