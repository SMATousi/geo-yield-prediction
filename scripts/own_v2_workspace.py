"""Assemble the v2 workspace of our Missouri dataset (spec/own-dataset-v2.md) from links, leaving v1 untouched.

    <ws2>/ground_truth -> <ws>/ground_truth_v2                     (fixed pipeline outputs, _summary.csv)
    <ws2>/layers/<id>/<layer> -> <ws>/layers/<id>/<layer>          (fields processed in v1: their layers)
                             or <ws>/layers_v2dl/<id>/<layer>      (new fields, and PlanetScope of re-downloaded ones)
    <ws2>/layers/<id>/yield/                                       real: the v2 polygons; the 10 m target is
                                                                   rasterised here (geo-data-downloader rasterize-yield)

    python scripts/own_v2_workspace.py --ws /home1/pupil/SMATousi/gdd-workspace --ws2 /home1/pupil/SMATousi/gdd-workspace-v2 \\
        --ps_redownload SoyBean2020_64 SoyBean2022_42 SoyBean2022_45 SoyBean2022_50
"""
import argparse
import csv
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ws', required=True)
    p.add_argument('--ws2', required=True)
    p.add_argument('--ps_redownload', nargs='*', default=[])
    a = p.parse_args()
    ws, ws2 = Path(a.ws), Path(a.ws2)
    gt2 = ws / 'ground_truth_v2'
    (ws2 / 'layers').mkdir(parents=True, exist_ok=True)
    if not (ws2 / 'ground_truth').exists():
        (ws2 / 'ground_truth').symlink_to(gt2)
    shared = ws2 / 'layers' / '_shared'
    if not shared.exists():
        shared.symlink_to((ws / 'layers' / '_shared').resolve())
    rows = [r for r in csv.DictReader(open(gt2 / '_summary.csv'))]
    n = {'v1': 0, 'new': 0}
    for r in rows:
        fid = r['vgir_id']
        v1, dl = ws / 'layers' / fid, ws / 'layers_v2dl' / fid
        src = v1 if (v1 / 'sentinel2').is_dir() else dl
        n['v1' if src == v1 else 'new'] += 1
        out = ws2 / 'layers' / fid
        out.mkdir(exist_ok=True)
        for layer in sorted(d.name for d in src.iterdir() if d.is_dir() and d.name != 'yield'):
            s = dl / layer if layer == 'planetscope' and fid in a.ps_redownload else src / layer
            link = out / layer
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(s)
        y = out / 'yield'
        y.mkdir(exist_ok=True)
        for f in (gt2 / fid).glob('{}_polygons.*'.format(fid)):
            shutil.copy2(f, y / f.name)
    print('{} field-seasons: {} with v1 layers, {} downloaded for v2; PlanetScope re-downloaded: {}'.format(
        len(rows), n['v1'], n['new'], a.ps_redownload))


if __name__ == '__main__':
    main()
