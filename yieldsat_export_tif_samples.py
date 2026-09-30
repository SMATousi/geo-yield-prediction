"""Export georeferenced RGB and yield GeoTIFF previews of random YieldSAT images.

The preview RGB chooses the observation slot with the most valid RGB pixels in
one patch. It is contrast-stretched for display; the original 24-slot, 120-layer
inputs remain in the source image dataset and are never modified by this script.
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import Affine


NODATA_YIELD = -9999.0


def pick_rgb(temporal, times, valid_pixel, channel_indices):
    """Pick the best dated RGB slot and render a display-only uint8 RGBA tile."""
    rgb = temporal[:, :, :, channel_indices]
    eligible = np.isfinite(rgb).all(axis=-1) & np.isfinite(times) & valid_pixel[:, :, None]
    counts = eligible.sum(axis=(0, 1))
    slot = int(np.argmax(counts))
    good = eligible[:, :, slot]
    preview = np.zeros((4, 64, 64), dtype=np.uint8)
    if good.any():
        for band in range(3):
            values = rgb[:, :, slot, band]
            lo, hi = np.percentile(values[good], [2, 98])
            if hi <= lo:
                hi = lo + 1.0
            preview[band, good] = np.clip((values[good] - lo) * 255.0 / (hi - lo),
                                          0, 255).astype(np.uint8)
        preview[3] = good.astype(np.uint8) * 255
    return slot, int(counts[slot]), preview


def export_samples(dataset_root, output_root, per_country=3, seed=20260930):
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    manifest = json.loads((dataset_root / 'manifest.json').read_text())
    channel_indices = [manifest['temporal_channels'].index(x) for x in ('B04', 'B03', 'B02')]
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError('output directory is not empty: {}'.format(output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    records_out = []
    for ci, country in enumerate(manifest['countries']):
        patches = [json.loads(line) for line in
                   (dataset_root / country / 'patches.jsonl').read_text().splitlines()]
        rng = np.random.default_rng(seed + ci)
        chosen = rng.choice(len(patches), size=min(per_country, len(patches)), replace=False)
        country_dir = output_root / country
        country_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(dataset_root / country / 'images.h5', 'r') as image_file:
            for patch_idx in chosen:
                patch_idx = int(patch_idx)
                rec = patches[patch_idx]
                affine = Affine(*rec['transform'])
                crs = 'EPSG:{}'.format(rec['epsg'])
                valid = image_file['valid_pixel'][patch_idx]
                times = image_file['times'][patch_idx]
                slot, rgb_count, rgba = pick_rgb(image_file['temporal'][patch_idx],
                                                  times, valid, channel_indices)
                yield_raw = image_file['target'][patch_idx]
                yield_valid = valid & np.isfinite(yield_raw)
                yield_tile = np.where(yield_valid, yield_raw, NODATA_YIELD).astype(np.float32)
                base = '{}_patch_{:04d}'.format(country.lower(), patch_idx)
                rgb_path = country_dir / (base + '_rgb.tif')
                yield_path = country_dir / (base + '_yield.tif')
                common = dict(driver='GTiff', height=64, width=64, crs=crs,
                              transform=affine, compress='deflate', tiled=True,
                              blockxsize=64, blockysize=64)
                tags = dict(field_shared_name=rec['field_shared_name'],
                            crop=rec['crop'], year=rec['year'],
                            patch_index=patch_idx, selected_slot=slot,
                            preview_only='true')
                with rasterio.open(rgb_path, 'w', count=4, dtype='uint8',
                                   **common) as dst:
                    dst.write(rgba)
                    dst.colorinterp = (ColorInterp.red, ColorInterp.green,
                                       ColorInterp.blue, ColorInterp.alpha)
                    dst.update_tags(**tags, rgb_valid_pixels=rgb_count,
                                    stretch='per-band 2nd-98th percentile; display only')
                with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                    with rasterio.open(yield_path, 'w', count=1, dtype='float32',
                                       nodata=NODATA_YIELD, **common) as dst:
                        dst.write(yield_tile, 1)
                        dst.write_mask(yield_valid.astype(np.uint8) * 255)
                        dst.update_tags(**tags, units='t/ha')
                entry = {'country': country, 'patch_index': patch_idx,
                         'field_shared_name': rec['field_shared_name'],
                         'year': rec['year'], 'crop': rec['crop'],
                         'valid_pixels': int(yield_valid.sum()),
                         'rgb_valid_pixels': rgb_count, 'selected_slot': slot,
                         'rgb_tif': str(rgb_path.relative_to(output_root)),
                         'yield_tif': str(yield_path.relative_to(output_root))}
                records_out.append(entry)
                print(json.dumps(entry), flush=True)
    tmp = output_root / 'samples.json.partial'
    tmp.write_text(json.dumps({'seed': seed, 'per_country': per_country,
                               'source_dataset': str(dataset_root),
                               'notes': 'RGB is a visual stretch of one time slot; yield is t/ha',
                               'samples': records_out}, indent=2))
    os.replace(tmp, output_root / 'samples.json')
    return records_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', default='/home1/pupil/SMATousi/YieldSAT-Image')
    parser.add_argument('--output-root', default='/home1/pupil/SMATousi/YieldSAT-Image-Samples')
    parser.add_argument('--per-country', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20260930)
    args = parser.parse_args()
    if args.per_country <= 0:
        parser.error('--per-country must be positive')
    export_samples(args.dataset_root, args.output_root, args.per_country, args.seed)


if __name__ == '__main__':
    main()
