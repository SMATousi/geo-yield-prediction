"""Export all 24 RGB slots for one existing YieldSAT preview patch per country.

GeoTIFFs are georeferenced display previews. Empty optical slots remain fully
transparent; no image or date is fabricated to fill them.
"""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.enums import ColorInterp
from rasterio.transform import Affine


RGB_BANDS = ('B04', 'B03', 'B02')
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def rgb_masks(temporal, times, occupied, indices):
    rgb = temporal[:, :, :, indices]
    good = np.isfinite(rgb).all(axis=-1) & np.isfinite(times) & occupied[:, :, None]
    return rgb, good


def choose_patches(dataset_root, samples_root, indices):
    samples = json.loads((samples_root / 'samples.json').read_text())['samples']
    chosen = {}
    for country in sorted({rec['country'] for rec in samples}):
        with h5py.File(dataset_root / country / 'images.h5', 'r') as f:
            candidates = []
            for rec in samples:
                if rec['country'] != country:
                    continue
                i = rec['patch_index']
                _, good = rgb_masks(f['temporal'][i], f['times'][i],
                                    f['source_row'][i] >= 0, indices)
                counts = good.sum(axis=(0, 1))
                candidates.append((int(np.count_nonzero(counts)), int(counts.sum()), -i, rec))
            if not candidates:
                raise ValueError('no existing preview patch for ' + country)
            chosen[country] = max(candidates, key=lambda x: x[:3])[3]
    return chosen


def format_date(day):
    return (EPOCH + timedelta(days=float(day))).strftime('%Y-%m-%d')


def render_slots(rgb, good):
    """Use one per-band stretch across this patch's entire dated sequence."""
    stretch = []
    for band in range(3):
        values = rgb[..., band][good]
        lo, hi = np.percentile(values, [2, 98]) if values.size else (0.0, 1.0)
        if hi <= lo:
            hi = lo + 1.0
        stretch.append((float(lo), float(hi)))
    rendered = []
    for slot in range(rgb.shape[2]):
        rgba = np.zeros((4, 64, 64), dtype=np.uint8)
        mask = good[:, :, slot]
        for band, (lo, hi) in enumerate(stretch):
            values = rgb[:, :, slot, band]
            rgba[band, mask] = np.clip((values[mask] - lo) * 255 / (hi - lo),
                                       0, 255).astype(np.uint8)
        rgba[3] = mask.astype(np.uint8) * 255
        rendered.append(rgba)
    return rendered, stretch


def contact_sheet(frames, metadata, title, path):
    scale, cell_w, cell_h = 2, 170, 174
    sheet = Image.new('RGB', (6 * cell_w, 4 * cell_h + 38), '#20252a')
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill='white')
    for slot, (rgba, info) in enumerate(zip(frames, metadata)):
        x, y = (slot % 6) * cell_w, 38 + (slot // 6) * cell_h
        draw.rectangle((x + 9, y + 22, x + 9 + 128, y + 22 + 128), fill='#343a40')
        tile = Image.fromarray(np.moveaxis(rgba, 0, -1), 'RGBA').resize(
            (128, 128), Image.Resampling.NEAREST)
        sheet.paste(tile, (x + 9, y + 22), tile)
        draw.text((x + 9, y + 4), 'Slot {:02d}'.format(slot), fill='white')
        label = info['date_range'] or 'No RGB observation'
        draw.text((x + 9, y + 152), label, fill='#cdd7df')
        draw.text((x + 9, y + 164), '{} pixels'.format(info['rgb_pixels']),
                  fill='#aab8c2')
    sheet.save(path)


def export(dataset_root, samples_root, output_root):
    dataset_root, samples_root, output_root = map(Path, (dataset_root, samples_root, output_root))
    if output_root.exists():
        raise ValueError('output already exists: {}'.format(output_root))
    manifest = json.loads((dataset_root / 'manifest.json').read_text())
    indices = [manifest['temporal_channels'].index(b) for b in RGB_BANDS]
    chosen = choose_patches(dataset_root, samples_root, indices)
    partial = output_root.with_name(output_root.name + '.partial')
    if partial.exists():
        raise ValueError('partial output already exists: {}'.format(partial))
    partial.mkdir(parents=True)
    report = {'source_dataset': str(dataset_root), 'rgb_bands': RGB_BANDS,
              'selection': 'Among the three existing previews per country: most nonempty slots, then most RGB pixels',
              'notes': 'All 24 slots exported; empty slots are transparent. RGB is one display-only 2nd-98th percentile stretch per band across the selected patch sequence.',
              'countries': {}}
    for country, rec in chosen.items():
        i = rec['patch_index']
        patch = json.loads((dataset_root / country / 'patches.jsonl').read_text().splitlines()[i])
        country_dir = partial / country
        country_dir.mkdir()
        with h5py.File(dataset_root / country / 'images.h5', 'r') as f:
            temporal = f['temporal'][i]
            times = f['times'][i]
            rgb, good = rgb_masks(temporal, times, f['source_row'][i] >= 0, indices)
        frames, stretch = render_slots(rgb, good)
        affine = Affine(*patch['transform'])
        metadata = []
        for slot, rgba in enumerate(frames):
            days = times[:, :, slot][good[:, :, slot]]
            date_range = None
            if days.size:
                lo, hi = format_date(days.min()), format_date(days.max())
                date_range = lo if lo == hi else lo + ' to ' + hi
            name = 'slot_{:02d}_rgb.tif'.format(slot)
            with rasterio.open(country_dir / name, 'w', driver='GTiff',
                               width=64, height=64, count=4, dtype='uint8',
                               crs='EPSG:{}'.format(patch['epsg']),
                               transform=affine, compress='deflate',
                               tiled=True, blockxsize=64, blockysize=64) as dst:
                dst.write(rgba)
                dst.colorinterp = (ColorInterp.red, ColorInterp.green,
                                   ColorInterp.blue, ColorInterp.alpha)
                dst.update_tags(country=country, patch_index=i,
                                field_shared_name=patch['field_shared_name'],
                                year=patch['year'], crop=patch['crop'],
                                slot=slot, date_range=date_range or 'none',
                                rgb_pixels=int(good[:, :, slot].sum()),
                                preview_only='true',
                                stretch='per-band 2nd-98th percentile across all 24 slots')
            metadata.append({'slot': slot, 'file': name, 'rgb_pixels': int(good[:, :, slot].sum()),
                             'date_range': date_range})
        contact_sheet(frames, metadata,
                      '{} — {} {} — patch {}'.format(country, patch['crop'], patch['year'], i),
                      country_dir / 'contact_sheet.png')
        report['countries'][country] = {'patch_index': i,
                                        'field_shared_name': patch['field_shared_name'],
                                        'crop': patch['crop'], 'year': patch['year'],
                                        'nonempty_slots': sum(m['rgb_pixels'] > 0 for m in metadata),
                                        'stretch': stretch, 'slots': metadata}
        print(country, 'patch', i, 'nonempty slots', report['countries'][country]['nonempty_slots'],
              flush=True)
    (partial / 'sequences.json').write_text(json.dumps(report, indent=2))
    os.replace(partial, output_root)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', default='/home1/pupil/SMATousi/YieldSAT-Image')
    parser.add_argument('--samples-root', default='/home1/pupil/SMATousi/YieldSAT-Image-Samples')
    parser.add_argument('--output-root', default='/home1/pupil/SMATousi/YieldSAT-Image-Samples/TemporalSequences')
    args = parser.parse_args()
    export(args.dataset_root, args.samples_root, args.output_root)


if __name__ == '__main__':
    main()
