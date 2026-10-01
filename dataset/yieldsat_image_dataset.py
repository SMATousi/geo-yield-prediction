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
DEM_IDX = [_SI['dem']]
TERRAIN_IDX = [_SI[c] for c in TERRAIN]                   # aspect, curvature, slope, twi
SOIL_IDX = [_SI[c] for c in SOIL]                         # property-major, depth-ordered
MIN_SLOT_COVERAGE = 0.25                                  # fraction of tile cells with RGB
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


def slot_dates(times, rgb_ok):
    """Per slot: (median date of RGB-observed cells, spread in days, coverage)."""
    dates = np.full(T, np.nan)
    spread = np.full(T, np.inf)
    coverage = np.zeros(T)
    for k in range(T):
        m = rgb_ok[..., k]
        n = int(m.sum())
        coverage[k] = n / float(m.size)
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
                 cutoff_mode='all_slots', cutoff_days=30, aspect_cyclic=True, seed=0):
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
        t = self.tiles[i]
        d = self.reader.get(t['country'], t['patch_index'])
        seeding, harvest = self.season_days[t['season_id']]
        cutoff = self._cutoff(seeding, harvest)
        temporal, static, times = d['temporal'], d['static'], d['times']
        present = d['source_row'] >= 0                              # in-field, occupied cell
        dated = np.isfinite(times)
        if cutoff is not None:
            dated &= times <= cutoff
        tvalid = temporal_valid(temporal, times) & dated[..., None]
        tn = self.norm.stats['temporal']
        tnorm = np.where(tvalid, (temporal - tn['mean']) / tn['std'], 0.0).astype(np.float32)

        # optical observations
        rgb_ok = tvalid[..., RGB_IDX].all(axis=-1)                  # (64,64,24)
        dates, spread, coverage = slot_dates(times, rgb_ok)
        slots = qualifying_slots(dates, spread, coverage, cutoff)
        rng = np.random.default_rng((self.seed, self.epoch, i)) if self.train else None
        chosen = select_observations(slots, dates, coverage, self.k, seeding, harvest, rng)
        K = self.k
        obs_slot = np.full(K, -1, np.int64)
        obs_valid = np.zeros(K, np.float32)
        obs_time = np.zeros((K, 3), np.float32)
        spec = np.zeros((K, len(EXTRA_IDX), TILE, TILE), np.float32)
        spec_mask = np.zeros((K, len(EXTRA_IDX), TILE, TILE), np.float32)
        rgb_mask = np.zeros((K, TILE, TILE), np.float32)
        for j, s in enumerate(chosen):
            obs_slot[j] = s
            obs_valid[j] = 1.0
            obs_time[j] = _time_features(dates[s], seeding)
            spec[j] = np.moveaxis(tnorm[:, :, s, EXTRA_IDX], -1, 0)
            spec_mask[j] = np.moveaxis(tvalid[:, :, s, EXTRA_IDX], -1, 0)
            rgb_mask[j] = rgb_ok[:, :, s]

        # weather: constant within a field -> tile-level series (mean over cells)
        wv = tvalid[..., WEATHER_IDX]                                # (64,64,24,4)
        cnt = wv.sum(axis=(0, 1))
        wsum = np.where(wv, tnorm[..., WEATHER_IDX], 0.0).sum(axis=(0, 1))
        weather = np.where(cnt > 0, wsum / np.maximum(cnt, 1), 0.0).astype(np.float32)
        weather_mask = (cnt > 0).astype(np.float32)
        tile_dates = np.array([np.nanmedian(times[..., s][dated[..., s]]) if dated[..., s].any()
                               else np.nan for s in range(T)])
        wtime = np.stack([_time_features(tile_dates[s], seeding) if np.isfinite(tile_dates[s])
                          else np.zeros(3, np.float32) for s in range(T)]).astype(np.float32)

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
        return {
            'obs_slot': obs_slot, 'obs_valid': obs_valid, 'obs_time': obs_time,
            'rgb_mask': rgb_mask, 'spec': spec, 'spec_mask': spec_mask,
            'weather': weather, 'weather_mask': weather_mask, 'weather_time': wtime,
            'dem': dem, 'dem_mask': dem_mask, 'terrain': ter, 'terrain_mask': ter_mask,
            'soil': soil, 'soil_mask': soil_mask, 'cell_present': present.astype(np.float32),
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
            yield int(rng.choice(self.groups[g]))


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
