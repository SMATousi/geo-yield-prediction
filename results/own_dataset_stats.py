"""Statistics of our Missouri dataset as built for training (spec/own-dataset-training.md §3), for the paper tables.

    python results/own_dataset_stats.py /root/owndata_artifacts /home1/pupil/SMATousi/gdd-workspace
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

art, ws = Path(sys.argv[1]), Path(sys.argv[2])
recs = json.loads((art / 'index' / 'Missouri' / 'fields.json').read_text())
geo = json.loads((art / 'geometry' / 'fields_geometry_Missouri.json').read_text())
geo = geo['fields'] if isinstance(geo, dict) else geo
times = np.load(art / 'cache_dense' / 'Missouri' / 'temporal.npy', mmap_mode='r')
ps_times = np.load(art / 'neighbourhood_dense' / 'Missouri' / 'ps_times.npy', mmap_mode='r')
out = {}
for crop in ('soybean', 'corn', 'all'):
    rs = [r for r in recs if crop == 'all' or r['crop'] == crop]
    cells = np.array([r['n_rows'] for r in rs])
    s2 = np.concatenate([np.isfinite(np.asarray(times[r['row_start']:r['row_end'], :, 3], np.float32)).sum(1) for r in rs])
    ps = np.concatenate([np.isfinite(np.asarray(ps_times[r['row_start']:r['row_end']])).sum(1) for r in rs])
    scenes, b8 = 0, 0
    for r in rs:
        for row in csv.DictReader(open(ws / 'layers' / r['field_shared_name'] / 'planetscope' / 'planetscope_index.csv')):
            if row['delivered'] == 'True':
                scenes += 1
                b8 += row['bands'] == '8'
    phys = {g['physical_field_id'] for g in geo if g['field_shared_name'] in {r['field_shared_name'] for r in rs}}
    out[crop] = {'field_seasons': len(rs), 'physical_fields': len(phys), 'cells': int(cells.sum()),
                 'cells_median': float(np.median(cells)), 'cells_min': int(cells.min()), 'cells_max': int(cells.max()),
                 'area_ha': float(cells.sum() * 0.01), 'years': dict(sorted(Counter(r['year'] for r in rs).items())),
                 'target_mean': float(np.median([r['target_mean'] for r in rs])),
                 'target_std_within': float(np.median([r['target_std'] for r in rs])),
                 's2_clear_slots_median': float(np.median(s2)), 'ps_clear_slots_median': float(np.median(ps)),
                 'ps_scenes': scenes, 'ps_scenes_8band': b8}
json.dump(out, open(Path(__file__).resolve().parent / 'own_dataset_stats.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
