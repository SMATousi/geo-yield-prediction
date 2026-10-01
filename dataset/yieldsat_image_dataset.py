# --------------------------------------------------------
# YieldSAT 64x64 image dataset loader (YI-01; spec/yieldsat-image-training.md,
# "Implementation plan v1").
#
# Tiles come from the image dataset built by yieldsat_build_images.py (one
# HDF5 + patch table per country). Train/val/test membership is *inherited
# from the point suite's fold manifests*: a tile belongs to the partition of
# its field season, so image and point runs hold out exactly the same field
# seasons and cells. Sensor masks come from feature finiteness and dates only;
# `valid_pixel` (finite yield) is loss/evaluation-side only. Normalization is
# fitted on training tiles of the run. Optical observations (up to K per tile)
# are chosen from date-coherent slots: deterministic for evaluation, randomly
# within seasonal bins for training. RGB values are not normalized here; the
# frozen DINO features for each (tile, slot) come from the feature cache.
# --------------------------------------------------------

import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from dataset.yieldsat_schema import CROPS, SOIL, STATIC_CHANNELS, TEMPORAL_CHANNELS, TERRAIN

T = 24
TILE = 64
RGB = ('B04', 'B03', 'B02')                               # red, green, blue (DINO order)
EXTRA_BANDS = ('B01', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12')
WEATHER = ('temp_max', 'temp_mean', 'temp_min', 'total_prec')
_TI = {c: i for i, c in enumerate(TEMPORAL_CHANNELS)}
_SI = {c: i for i, c in enumerate(STATIC_CHANNELS)}
RGB_IDX = [_TI[c] for c in RGB]
EXTRA_IDX = [_TI[c] for c in EXTRA_BANDS]
WEATHER_IDX = [_TI[c] for c in WEATHER]
S2_IDX = sorted(RGB_IDX + EXTRA_IDX)                      # all 12 S2 bands (series stream, v2)
DEM_IDX = [_SI['dem']]
TERRAIN_IDX = [_SI[c] for c in TERRAIN]                   # aspect, curvature, slope, twi
SOIL_IDX = [_SI[c] for c in SOIL]                         # property-major, depth-ordered
MIN_SLOT_COVERAGE = 0.25                                  # fraction of present cells with RGB
MAX_SLOT_DATE_SPREAD = 1.0                                # days, date coherence
CUTOFF_MODES = ('all_slots', 'before_harvest', 'harvest')


# ---- tile table and fold mapping ------------------------------------------------

def load_tile_table(image_root, countries):
    """Patch records of the selected countries, with ``season_id``."""
    tiles = []
    for country in countries:
        path = Path(image_root) / country / 'patches.jsonl'
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            rec['season_id'] = '{}/{}'.format(country, rec['field_shared_name'])
            tiles.append(rec)
    return tiles


def tiles_for_split(tiles, split, crops=None):
    """Partition tiles by the point fold manifest ``split`` (dict). Tiles of
    seasons the manifest excludes, or of other countries/crops, are dropped.
    Returns {'train': [...], 'val': [...], 'test': [...]}."""
    where = {}
    for part in ('train', 'val', 'test', 'excluded'):
        for s in split['partitions'].get(part, []):
            where[s] = part
    countries = set(split['countries'])
    crops = set(crops or split.get('crops') or [])
    out = {'train': [], 'val': [], 'test': []}
    for t in tiles:
        if t['country'] not in countries or (crops and t['crop'] not in crops):
            continue
        part = where.get(t['season_id'])
        if part in out:
            out[part].append(t)
    return out


# ---- per-tile reading and derived masks ---------------------------------------------

class TileReader:
    """Lazily opened per-country HDF5 handles (one set per worker process)."""

    def __init__(self, image_root):
        self.root = Path(image_root)
        self._files = {}
        self._pid = None

    def get(self, country, index):
        import os
        if self._pid != os.getpid():
            self._files, self._pid = {}, os.getpid()
        f = self._files.get(country)
        if f is None:
            f = self._files[country] = h5py.File(self.root / country / 'images.h5', 'r')
        return {k: f[k][index] for k in ('temporal', 'static', 'times', 'target', 'valid_pixel',
                                          'source_row')}

    def close(self):
        for f in self._files.values():
            f.close()
        self._files = {}


def slot_dates(times, rgb_ok, present=None):
    """Per slot: (median date of RGB-observed cells, spread in days, coverage).
    Coverage is relative to the tile's present (in-field) cells when given,
    so small edge tiles of full-coverage builds still qualify (v2)."""
    dates = np.full(T, np.nan)
    spread = np.full(T, np.inf)
    coverage = np.zeros(T)
    denom = max(int(present.sum()), 1) if present is not None else rgb_ok[..., 0].size
    for k in range(T):
        m = rgb_ok[..., k]
        n = int(m.sum())
        coverage[k] = n / float(denom)
        if n:
            d = times[..., k][m]
            dates[k] = float(np.median(d))
            spread[k] = float(d.max() - d.min())
    return dates, spread, coverage


def qualifying_slots(dates, spread, coverage, cutoff_day):
    ok = (coverage >= MIN_SLOT_COVERAGE) & (spread <= MAX_SLOT_DATE_SPREAD) & np.isfinite(dates)
    if cutoff_day is not None:
        ok &= dates <= cutoff_day
    return np.flatnonzero(ok)


def select_observations(slots, dates, coverage, k, seeding_day, harvest_day, rng=None):
    """Up to ``k`` slots spread over the season: the season [seeding, harvest]
    is split into k bins; per bin the best-covered slot (evaluation) or a
    random one (training, ``rng``); unfilled bins take the best remaining
    slots. Returns sorted slot indices (length <= k)."""
    if len(slots) == 0:
        return []
    lo, hi = float(seeding_day), float(max(harvest_day, seeding_day + 1))
    edges = np.linspace(lo, hi, k + 1)
    chosen = []
    for b in range(k):
        in_bin = [s for s in slots if edges[b] <= dates[s] < edges[b + 1] or
                  (b == k - 1 and dates[s] >= edges[b])]
        in_bin = [s for s in in_bin if s not in chosen]
        if not in_bin:
            continue
        if rng is not None:
            chosen.append(int(rng.choice(in_bin)))
        else:
            chosen.append(int(max(in_bin, key=lambda s: (coverage[s], -s))))
    rest = sorted((s for s in slots if s not in chosen), key=lambda s: (-coverage[s], s))
    chosen += [int(s) for s in rest[:max(0, k - len(chosen))]]
    return sorted(chosen)


# ---- frozen DINO feature cache ---------------------------------------------------

class DinoFeatureCache:
    """Reader for yieldsat_image_dino_cache.py output. Refuses a cache built
    with a different checkpoint revision or preprocessing."""

    def __init__(self, cache_dir, expected_revision=None, expected_prep_hash=None):
        self.dir = Path(cache_dir)
        self.manifest = json.loads((self.dir / 'manifest.json').read_text())
        if expected_revision and self.manifest['revision'] != expected_revision:
            raise ValueError('DINO cache revision {} != expected {}'.format(
                self.manifest['revision'], expected_revision))
        if expected_prep_hash and self.manifest['prep_hash'] != expected_prep_hash:
            raise ValueError('DINO cache preprocessing {} != expected {}'.format(
                self.manifest['prep_hash'], expected_prep_hash))
        self.hidden_size = int(self.manifest['hidden_size'])
        self._files, self._index, self._pid = {}, {}, None

    def _open(self, country):
        import os
        if self._pid != os.getpid():
            self._files, self._index, self._pid = {}, {}, os.getpid()
        if country not in self._files:
            f = h5py.File(self.dir / '{}.h5'.format(country), 'r')
            self._files[country] = f
            self._index[country] = {(int(p), int(s)): i for i, (p, s) in
                                    enumerate(zip(f['patch_index'][:], f['slot'][:]))}
        return self._files[country], self._index[country]

    def get(self, country, patch_index, slots):
        f, idx = self._open(country)
        out = np.zeros((len(slots), 16, self.hidden_size), np.float16)
        valid = np.zeros(len(slots), np.float32)
        for j, s in enumerate(slots):
            row = idx.get((int(patch_index), int(s))) if s >= 0 else None
            if row is not None:
                out[j] = f['features'][row]
                valid[j] = 1.0
        return out, valid


# ---- normalization ------------------------------------------------------------

class ImageNormalizer:
    """Per-channel mean/std from training tiles (valid values only)."""

    def __init__(self, stats):
        self.stats = stats

    @classmethod
    def fit(cls, reader, train_tiles, max_tiles=200, seed=0):
        rng = np.random.default_rng(seed)
        pick = list(train_tiles)
        if len(pick) > max_tiles:
            pick = [pick[i] for i in sorted(rng.choice(len(pick), max_tiles, replace=False))]
        acc = {k: [np.zeros(n), np.zeros(n), np.zeros(n)] for k, n in
               (('temporal', len(TEMPORAL_CHANNELS)), ('static', len(STATIC_CHANNELS)), ('target', 1))}

        def add(key, values, valid):
            v = np.where(valid, values, 0.0).astype(np.float64)
            a = acc[key]
            a[0] += valid.sum(axis=0)
            a[1] += v.sum(axis=0)
            a[2] += (v * v).sum(axis=0)

        for t in pick:
            d = reader.get(t['country'], t['patch_index'])
            tv = temporal_valid(d['temporal'], d['times'])
            add('temporal', d['temporal'].reshape(-1, len(TEMPORAL_CHANNELS)),
                tv.reshape(-1, len(TEMPORAL_CHANNELS)))
            present = d['source_row'] >= 0
            s = d['static'][present]
            add('static', s, np.isfinite(s))
            y = d['target'][d['valid_pixel']].reshape(-1, 1)
            add('target', y, np.isfinite(y))
        stats = {}
        for key, (n, s, ss) in acc.items():
            n = np.maximum(n, 1)
            mean = s / n
            std = np.sqrt(np.maximum(ss / n - mean * mean, 0.0))
            stats[key] = {'mean': mean, 'std': np.where(std > 0, std, 1.0)}
        stats['fitted_on_tiles'] = len(pick)
        return cls(stats)

    def to_json(self):
        return {k: ({kk: vv.tolist() for kk, vv in v.items()} if isinstance(v, dict) else v)
                for k, v in self.stats.items()}

    @classmethod
    def from_json(cls, data):
        return cls({k: ({kk: np.asarray(vv) for kk, vv in v.items()} if isinstance(v, dict) else v)
                    for k, v in data.items()})


def temporal_valid(temporal, times):
    """(64,64,24,16) validity: finite value on a dated slot; weather at each
    cell's first dated slot is invalid (point contract)."""
    dated = np.isfinite(times)
    valid = np.isfinite(temporal) & dated[..., None]
    has = dated.any(axis=-1)
    first = np.argmax(dated, axis=-1)
    rr, cc = np.nonzero(has)
    for w in WEATHER_IDX:
        valid[rr, cc, first[rr, cc], w] = False
    return valid


# ---- dataset ----------------------------------------------------------------------

class YieldSATImageDataset(Dataset):
    """One item = one 64x64 tile with per-modality inputs, masks and target.

    ``season_days`` maps season_id -> (seeding_day, harvest_day) (from the point
    index). ``train=True`` samples optical observations randomly per bin;
    otherwise selection is deterministic.
    """

    def __init__(self, image_root, tiles, normalizer, season_days, k_obs=4, train=False,
                 cutoff_mode='all_slots', cutoff_days=30, aspect_cyclic=True, seed=0,
                 dino_cache=None, with_series=False, slot_coverage='tile'):
        if cutoff_mode not in CUTOFF_MODES:
            raise ValueError('cutoff_mode must be one of {}'.format(CUTOFF_MODES))
        self.reader = TileReader(image_root)
        self.tiles = list(tiles)
        self.norm = normalizer
        self.season_days = season_days
        self.k = k_obs
        self.train = train
        self.cutoff_mode = cutoff_mode
        self.cutoff_days = cutoff_days
        self.aspect_cyclic = aspect_cyclic
        self.seed = seed
        self.epoch = 0
        self.seasons = sorted({t['season_id'] for t in self.tiles})
        self._season_index = {s: i for i, s in enumerate(self.seasons)}
        self.retrospective = cutoff_mode == 'all_slots'
        self.dino = dino_cache
        # v2 options (defaults reproduce v1): full per-cell S2 series stream;
        # optical slot coverage relative to the tile ('tile') or its present cells
        if slot_coverage not in ('tile', 'present'):
            raise ValueError('slot_coverage must be tile or present')
        self.with_series = with_series
        self.slot_coverage = slot_coverage

    def __len__(self):
        return len(self.tiles)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _cutoff(self, seeding, harvest):
        if self.cutoff_mode == 'all_slots':
            return None
        if self.cutoff_mode == 'harvest':
            return float(harvest)
        return float(harvest - self.cutoff_days)

    def __getitem__(self, i):
        epoch = self.epoch
        if isinstance(i, tuple):                  # (index, epoch) from SeasonBalancedSampler:
            i, epoch = i                          # works with persistent loader workers
        t = self.tiles[i]
        d = self.reader.get(t['country'], t['patch_index'])
        seeding, harvest = self.season_days[t['season_id']]
        cutoff = self._cutoff(seeding, harvest)
        temporal, static, times = d['temporal'], d['static'], d['times']
        present = d['source_row'] >= 0                              # in-field, occupied cell
        dated = np.isfinite(times)
        if cutoff is not None:
            dated &= times <= cutoff
        # Only the channel groups that are used are validated/normalized (the
        # full (64,64,24,16) array is never materialized twice): RGB validity
        # for slot choice, the nine extra bands at the chosen slots, weather.
        tn = self.norm.stats['temporal']
        mean, std = np.asarray(tn['mean'], np.float32), np.asarray(tn['std'], np.float32)

        # optical observations
        rgb_ok = np.isfinite(temporal[..., RGB_IDX]).all(axis=-1) & dated     # (64,64,24)
        dates, spread, coverage = slot_dates(times, rgb_ok,
                                             present if self.slot_coverage == 'present' else None)
        slots = qualifying_slots(dates, spread, coverage, cutoff)
        rng = np.random.default_rng((self.seed, epoch, i)) if self.train else None
        chosen = select_observations(slots, dates, coverage, self.k, seeding, harvest, rng)
        K = self.k
        obs_slot = np.full(K, -1, np.int64)
        obs_valid = np.zeros(K, np.float32)
        obs_time = np.zeros((K, 3), np.float32)
        spec = np.zeros((K, len(EXTRA_IDX), TILE, TILE), np.float32)
        spec_mask = np.zeros((K, len(EXTRA_IDX), TILE, TILE), np.float32)
        rgb_mask = np.zeros((K, TILE, TILE), np.float32)
        em, es = mean[EXTRA_IDX], std[EXTRA_IDX]
        for j, s in enumerate(chosen):
            x = temporal[:, :, s, EXTRA_IDX]                                  # (64,64,9)
            v = np.isfinite(x) & dated[:, :, s, None]
            obs_slot[j] = s
            obs_valid[j] = 1.0
            obs_time[j] = _time_features(dates[s], seeding)
            spec[j] = np.moveaxis(np.where(v, (x - em) / es, 0.0), -1, 0)
            spec_mask[j] = np.moveaxis(v, -1, 0)
            rgb_mask[j] = rgb_ok[:, :, s]

        # weather: constant within a field -> tile-level series (mean over
        # cells); each cell's first dated slot is invalid (point contract)
        w = temporal[..., WEATHER_IDX]                                        # (64,64,24,4)
        wv = np.isfinite(w) & dated[..., None]
        all_dated = np.isfinite(times)
        rr, cc = np.nonzero(all_dated.any(axis=-1))
        wv[rr, cc, np.argmax(all_dated, axis=-1)[rr, cc]] = False
        cnt = wv.sum(axis=(0, 1))
        wsum = np.where(wv, (w - mean[WEATHER_IDX]) / std[WEATHER_IDX], 0.0).sum(axis=(0, 1))
        weather = np.where(cnt > 0, wsum / np.maximum(cnt, 1), 0.0).astype(np.float32)
        weather_mask = (cnt > 0).astype(np.float32)
        tile_dates = np.array([np.nanmedian(times[..., s][dated[..., s]]) if dated[..., s].any()
                               else np.nan for s in range(T)])
        wtime = np.stack([_time_features(tile_dates[s], seeding) if np.isfinite(tile_dates[s])
                          else np.zeros(3, np.float32) for s in range(T)]).astype(np.float32)

        item_series = {}
        if self.with_series:
            # full S2 series per cell (v2): 24 slots x 12 bands, slot-observed
            # mask, per-cell days since seeding; masked like the point contract
            x = temporal[..., S2_IDX]                                         # (64,64,24,12)
            sv = np.isfinite(x) & dated[..., None]
            series = np.where(sv, (x - mean[S2_IDX]) / std[S2_IDX], 0.0)
            # cell-major layout (H, W, T[, C]) as stored: no transposed copies;
            # the model rearranges on the GPU
            item_series = {
                'series': series.astype(np.float16),                          # (64,64,24,12)
                'series_mask': sv.any(axis=-1),                               # (64,64,24)
                'series_days': np.where(dated, times - seeding, 0.0).astype(np.float16),
                'seeding_doy': np.float32(float(seeding) % 365.2425)}

        # static layers (per cell)
        sn = self.norm.stats['static']
        svalid = np.isfinite(static) & present[..., None]
        snorm = np.where(svalid, (static - sn['mean']) / sn['std'], 0.0).astype(np.float32)
        dem = np.moveaxis(snorm[..., DEM_IDX], -1, 0)
        dem_mask = np.moveaxis(svalid[..., DEM_IDX], -1, 0).astype(np.float32)
        ter = np.moveaxis(snorm[..., TERRAIN_IDX], -1, 0)
        ter_mask = np.moveaxis(svalid[..., TERRAIN_IDX], -1, 0).astype(np.float32)
        if self.aspect_cyclic:
            a = np.deg2rad(static[..., TERRAIN_IDX[0]])
            am = svalid[..., TERRAIN_IDX[0]]
            ter = np.concatenate([np.stack([np.where(am, np.sin(a), 0), np.where(am, np.cos(a), 0)]),
                                  ter[1:]]).astype(np.float32)
            ter_mask = np.concatenate([np.stack([am, am]).astype(np.float32), ter_mask[1:]])
        soil = np.moveaxis(snorm[..., SOIL_IDX], -1, 0)
        soil_mask = np.moveaxis(svalid[..., SOIL_IDX], -1, 0).astype(np.float32)

        yn = self.norm.stats['target']
        valid = d['valid_pixel'] & np.isfinite(d['target'])
        target = np.where(valid, (d['target'] - yn['mean'][0]) / yn['std'][0], 0.0).astype(np.float32)
        rows = (t['row0'] + np.arange(TILE))[:, None].repeat(TILE, 1)
        cols = (t['col0'] + np.arange(TILE))[None, :].repeat(TILE, 0)
        item_dino = {}
        if self.dino is not None:
            feats, fvalid = self.dino.get(t['country'], t['patch_index'], obs_slot)
            item_dino = {'dino': feats, 'dino_valid': fvalid * obs_valid}
        # dense layers travel as float16 values + bool masks (~3.5x less
        # worker->trainer traffic); cast_batch() restores float32 on device
        h, m = np.float16, bool
        return {
            **item_dino, **item_series,
            'obs_slot': obs_slot, 'obs_valid': obs_valid, 'obs_time': obs_time,
            'rgb_mask': rgb_mask.astype(m), 'spec': spec.astype(h), 'spec_mask': spec_mask.astype(m),
            'weather': weather, 'weather_mask': weather_mask, 'weather_time': wtime,
            'dem': dem.astype(h), 'dem_mask': dem_mask.astype(m), 'terrain': ter.astype(h),
            'terrain_mask': ter_mask.astype(m), 'soil': soil.astype(h), 'soil_mask': soil_mask.astype(m),
            'cell_present': present,
            'target': target, 'target_raw': np.where(valid, d['target'], np.nan).astype(np.float32),
            'target_valid': valid, 'target_mean': np.float32(yn['mean'][0]),
            'target_std': np.float32(yn['std'][0]),
            'crop': np.int64(CROPS.index(t['crop'])), 'tile': np.int64(i),
            'season': np.int64(self._season_index[t['season_id']]),
            'grid_row': rows.astype(np.int32), 'grid_col': cols.astype(np.int32),
            'country': t['country'], 'patch_index': np.int64(t['patch_index']),
        }


def _time_features(day, seeding):
    doy = 2 * math.pi * (float(day) % 365.2425) / 365.2425
    return np.array([(float(day) - float(seeding)) / 365.0, math.sin(doy), math.cos(doy)], np.float32)


KEEP_BOOL = ('target_valid',)


def cast_batch(batch):
    """float16 values and bool masks -> float32 (on whatever device the
    batch is); ``target_valid`` stays bool."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and (v.dtype == torch.float16 or (v.dtype == torch.bool and k not in KEEP_BOOL)):
            v = v.float()
        out[k] = v
    return out


def collate_tiles(items):
    out = {}
    for k in items[0]:
        v = [it[k] for it in items]
        if isinstance(v[0], str):
            out[k] = v
        else:
            out[k] = torch.as_tensor(np.stack(v))
    return out


class SeasonBalancedSampler(Sampler):
    """Draw tiles so each field season is equally likely (then a random tile
    of that season); ``n`` draws per epoch."""

    def __init__(self, tiles, n, seed=0):
        by = {}
        for i, t in enumerate(tiles):
            by.setdefault(t['season_id'], []).append(i)
        self.groups = list(by.values())
        self.n = n
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.n

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        for g in rng.integers(0, len(self.groups), self.n):
            yield int(rng.choice(self.groups[g])), self.epoch


def audit_date_coherence(image_root, countries, max_tiles=None):
    """Fraction of (tile, slot) pairs with RGB whose dates spread > 1 day."""
    reader = TileReader(image_root)
    tiles = load_tile_table(image_root, countries)
    if max_tiles:
        tiles = tiles[:max_tiles]
    n_slots = incoherent = 0
    for t in tiles:
        d = reader.get(t['country'], t['patch_index'])
        rgb_ok = np.isfinite(d['temporal'][..., RGB_IDX]).all(-1) & np.isfinite(d['times'])
        _, spread, cov = slot_dates(d['times'], rgb_ok)
        used = cov > 0
        n_slots += int(used.sum())
        incoherent += int((used & (spread > MAX_SLOT_DATE_SPREAD)).sum())
    return {'tiles': len(tiles), 'slots_with_rgb': n_slots, 'incoherent_slots': incoherent}
