# --------------------------------------------------------
# GPU-side tile preparation for the image/hybrid pipeline
# (spec/yieldsat-improvement.md, "GPU data path").
#
# With YieldSATImageDataset(gpu_prep=True) a loader worker only reads a tile,
# selects the optical observations (cheap) and ships raw arrays:
#   raw_s2 (H, W, T, 12), raw_weather (H, W, T, 4), raw_static (H, W, 104)
#   (float32: some values exceed float16), rel_times (H, W, T) float16
#   = days since seeding (NaN = undated), target, valid_pixel, cell_present,
#   obs_slot/obs_valid/obs_time (+ dino), seeding, cutoff_rel, aug_k/aug_flip.
# prepare_batch() builds the model inputs on the GPU with exactly the CPU
# item's semantics (masks from finiteness and dates, cutoff, train-fold
# normalization, first-dated-slot weather invalid, tile-level weather and its
# median slot dates, cyclic aspect, observation maps, standardized target)
# and applies the S7 augmentation. Loaders stop being the bottleneck, so the
# GPU does the work it is requested for.
# --------------------------------------------------------

import math

import torch

from dataset.yieldsat_image_dataset import (
    DEM_IDX, EXTRA_IDX, RGB_IDX, S2_IDX, SOIL_IDX, TERRAIN_IDX, WEATHER_IDX,
)

YEAR = 365.2425


def stats_tensors(norm, device):
    s = norm.stats
    t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    return {'tmean': t(s['temporal']['mean']), 'tstd': t(s['temporal']['std']),
            'smean': t(s['static']['mean']), 'sstd': t(s['static']['std'])}


def _time_feats(rel, seeding):
    """rel days since seeding (finite), seeding absolute day -> (..., 3)."""
    doy = 2 * math.pi * torch.remainder(seeding + rel, YEAR) / YEAR
    return torch.stack([rel / 365.0, torch.sin(doy), torch.cos(doy)], -1)


@torch.no_grad()
def prepare_batch(b, st, with_series=True, with_obs=True, aspect_cyclic=True, augment=True):
    s2 = b['raw_s2'].float()                                      # (B, H, W, T, 12)
    temporal = s2.new_full(s2.shape[:-1] + (16,), float('nan'))  # reassemble the 16 channels
    temporal[..., S2_IDX] = s2
    temporal[..., WEATHER_IDX] = b['raw_weather'].float()
    static = b['raw_static'].float()                              # (B, H, W, 104)
    rel = b['rel_times'].float()                                  # (B, H, W, T)
    B, H, W, T, _ = temporal.shape
    present = b['cell_present'].bool()
    all_dated = torch.isfinite(rel)
    dated = all_dated & (rel <= b['cutoff_rel'].view(B, 1, 1, 1))
    seeding = b['seeding'].double().view(B, 1, 1, 1)
    out = {k: v for k, v in b.items() if not k.startswith(('raw_', 'rel_times'))}

    def norm(idx, x):
        return (x - st['tmean'][idx]) / st['tstd'][idx]

    # weather: tile-level mean of normalized values over valid cells; a cell's
    # first dated slot is invalid (unknown interval start)
    w = temporal[..., WEATHER_IDX]
    first = all_dated & (torch.cumsum(all_dated.int(), -1) == 1)
    wv = torch.isfinite(w) & (dated & ~first).unsqueeze(-1)
    cnt = wv.sum((1, 2)).float()                                  # (B, T, 4)
    wsum = torch.where(wv, norm(WEATHER_IDX, w), torch.zeros_like(w)).sum((1, 2))
    out['weather'] = torch.where(cnt > 0, wsum / cnt.clamp(min=1), torch.zeros_like(wsum))
    out['weather_mask'] = (cnt > 0).float()
    rel_d = torch.where(dated, rel, torch.full_like(rel, float('nan'))).reshape(B, H * W, T)
    med = torch.nanmedian(rel_d.double(), dim=1).values                     # (B, T)
    wt = _time_feats(torch.nan_to_num(med), b['seeding'].double().view(B, 1)).float()
    out['weather_time'] = torch.where(torch.isfinite(med).unsqueeze(-1), wt, torch.zeros_like(wt))

    if with_series:
        x = temporal[..., S2_IDX]
        sv = torch.isfinite(x) & dated.unsqueeze(-1)
        out['series'] = torch.where(sv, norm(S2_IDX, x), torch.zeros_like(x))
        out['series_mask'] = sv.any(-1).float()
        out['series_days'] = torch.where(dated, rel, torch.zeros_like(rel))
        out['seeding_doy'] = torch.remainder(b['seeding'].double(), YEAR).float()

    if with_obs:                                                   # observation maps (map branch)
        K = b['obs_slot'].shape[1]
        slot = b['obs_slot'].clamp(min=0)
        ok = (b['obs_valid'] > 0).view(B, K, 1, 1)
        gi = slot.view(B, 1, 1, K, 1).expand(B, H, W, K, temporal.shape[-1])
        ts = torch.gather(temporal, 3, gi)                         # (B, H, W, K, 16)
        ds = torch.gather(dated, 3, slot.view(B, 1, 1, K).expand(B, H, W, K))
        xe = ts[..., EXTRA_IDX]
        ve = torch.isfinite(xe) & ds.unsqueeze(-1)
        spec = torch.where(ve, norm(EXTRA_IDX, xe), torch.zeros_like(xe)).permute(0, 3, 4, 1, 2)
        out['spec'] = spec * ok.unsqueeze(2)
        out['spec_mask'] = (ve.permute(0, 3, 4, 1, 2) & ok.unsqueeze(2)).float()
        rgb = torch.isfinite(ts[..., RGB_IDX]).all(-1) & ds
        out['rgb_mask'] = (rgb.permute(0, 3, 1, 2) & ok).float()

    sv = torch.isfinite(static) & present.unsqueeze(-1)
    sn = torch.where(sv, (static - st['smean']) / st['sstd'], torch.zeros_like(static)).permute(0, 3, 1, 2)
    svm = sv.permute(0, 3, 1, 2).float()
    out['dem'], out['dem_mask'] = sn[:, DEM_IDX], svm[:, DEM_IDX]
    ter, ter_m = sn[:, TERRAIN_IDX], svm[:, TERRAIN_IDX]
    if aspect_cyclic:
        a = torch.deg2rad(static[..., TERRAIN_IDX[0]])
        am = sv[..., TERRAIN_IDX[0]]
        z = torch.zeros_like(a)
        ter = torch.cat([torch.stack([torch.where(am, torch.sin(a), z), torch.where(am, torch.cos(a), z)], 1),
                         ter[:, 1:]], 1)
        ter_m = torch.cat([am.float().unsqueeze(1).expand(-1, 2, -1, -1), ter_m[:, 1:]], 1)
    out['terrain'], out['terrain_mask'] = ter, ter_m
    out['soil'], out['soil_mask'] = sn[:, SOIL_IDX], svm[:, SOIL_IDX]

    valid = b['valid_pixel'].bool() & torch.isfinite(b['target'])
    tm, ts_ = b['target_mean'].view(B, 1, 1), b['target_std'].view(B, 1, 1)
    out['target'] = torch.where(valid, (b['target'] - tm) / ts_, torch.zeros_like(b['target']))
    out['target_raw'] = torch.where(valid, b['target'], torch.full_like(b['target'], float('nan')))
    out['target_valid'] = valid
    out['cell_present'] = present.float()
    if augment and 'aug_k' in b:
        out = augment_batch(out, b['aug_k'], b['aug_flip'], aspect_cyclic)
    return out


MAP_KEYS = ('rgb_mask', 'spec', 'spec_mask', 'dem', 'dem_mask', 'terrain', 'terrain_mask', 'soil', 'soil_mask',
            'cell_present', 'target', 'target_raw', 'target_valid', 'grid_row', 'grid_col')
CELL_MAJOR = ('series', 'series_mask', 'series_days')


def augment_batch(out, ks, flips, aspect_cyclic=True):
    """Per-sample flip (left-right) then k x 90-deg CCW rotation, the GPU twin
    of dataset.augment_tile (same aspect handling and DINO token permutation)."""
    out = dict(out)
    keys = [k for k in MAP_KEYS + CELL_MAJOR + ('dino', 'terrain') if k in out]
    for k in set(keys):
        out[k] = out[k].clone()
    for i in range(ks.shape[0]):
        k, f = int(ks[i]), bool(flips[i])
        if k == 0 and not f:
            continue

        def tf(a, dims):
            if f:
                a = torch.flip(a, (dims[1],))
            return torch.rot90(a, k, dims)

        for key in MAP_KEYS:
            if key in out:
                out[key][i] = tf(out[key][i], (-2, -1))
        for key in CELL_MAJOR:
            if key in out:
                out[key][i] = tf(out[key][i], (0, 1))
        if 'dino' in out:
            d = out['dino']
            d[i] = tf(d[i].reshape(d.shape[1], 4, 4, d.shape[-1]), (1, 2)).reshape(d.shape[1:])
        if aspect_cyclic and 'terrain' in out:
            t = out['terrain']
            sin, cos = t[i, 0].clone(), t[i, 1].clone()
            if f:
                sin = -sin
            for _ in range(k % 4):
                sin, cos = -cos, sin
            t[i, 0], t[i, 1] = sin, cos
    return out
