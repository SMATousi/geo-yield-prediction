import json, numpy as np
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from dataset.yieldsat_splits import PAPER_PAIRS, parse_pair
from dataset.yieldsat_schema import TEMPORAL_CHANNELS, STATIC_CHANNELS
art = '/root/yieldsat_artifacts'
S2 = list(range(12)); B4, B8 = TEMPORAL_CHANNELS.index('B04'), TEMPORAL_CHANNELS.index('B08')
W = {c: TEMPORAL_CHANNELS.index(c) for c in ('temp_max', 'temp_mean', 'temp_min', 'total_prec')}
STAT = [i for i, c in enumerate(STATIC_CHANNELS) if not c.endswith('_uncertainty') and not c.startswith('coord')]
rng = np.random.default_rng(0)
out = {}
cache = {}
for pair in PAPER_PAIRS:
    country, crop = parse_pair(pair)
    if country not in cache:
        cache[country] = (np.load('%s/cache/%s/temporal.npy' % (art, country), mmap_mode='r'),
                          np.load('%s/cache/%s/static.npy' % (art, country), mmap_mode='r'),
                          np.load('%s/cache/%s/times.npy' % (art, country), mmap_mode='r'),
                          np.load('%s/index/%s/rows.npz' % (art, country)), json.load(open('%s/index/%s/fields.json' % (art, country))))
    T, ST, TM, rows, fields = cache[country]
    fl = [f for f in fields if f['crop'] == crop]
    feats, yld, year, region, names = [], [], [], [], []
    for f in fl:
        a, b = f['row_start'], f['row_end']
        idx = np.sort(rng.choice(np.arange(a, b), min(300, b - a), replace=False))
        t = np.asarray(T[idx]).astype(np.float32); tm = np.asarray(TM[idx]); s = np.asarray(ST[idx]).astype(np.float32)
        dated = np.isfinite(tm)
        s2 = np.where(dated[:, :, None], t[:, :, S2], np.nan)
        with np.errstate(all='ignore'):
            ndvi = (s2[:, :, B8] - s2[:, :, B4]) / (s2[:, :, B8] + s2[:, :, B4])
            band_mean = np.nanmean(np.nanmean(s2, 1), 0)
            nd = np.nanmean(ndvi, 0)                       # per slot, field mean
            peak = np.nanmax(nd) if np.isfinite(nd).any() else np.nan
            peak_day = (np.nanmean(tm[:, int(np.nanargmax(nd))]) - f['seeding_day']) if np.isfinite(nd).any() else np.nan
            wsum = [np.nansum(np.nanmean(np.where(dated, t[:, :, W[c]], np.nan), 0)) for c in ('temp_mean', 'total_prec')]
            wmax = np.nanmax(np.nanmean(np.where(dated, t[:, :, W['temp_max']], np.nan), 0))
            stat = np.nanmean(np.where(np.isfinite(s[:, STAT]), s[:, STAT], np.nan), 0)
        season_len = f['harvest_day'] - f['seeding_day']
        feats.append(np.concatenate([band_mean, [peak, np.nanmean(nd), peak_day, season_len, *wsum, wmax], stat]))
        y = np.asarray(rows['target'][a:b]); yld.append(float(np.nanmean(y)))
        year.append(int(f['year'])); names.append(f['field_shared_name'])
        region.append(f.get('province') or f['farm_id'] if pair.startswith('ARG') else f['farm_id'])
    X = np.array(feats, dtype=np.float64)
    X = np.where(np.isfinite(X), X, np.nanmedian(X, 0))
    X = np.where(np.isfinite(X), X, 0.0)
    keep = X.std(0) > 0
    Xs = StandardScaler().fit_transform(X[:, keep])
    emb = TSNE(n_components=2, perplexity=min(30, max(5, len(Xs) // 4)), random_state=0, init='pca').fit_transform(Xs)
    yld = np.array(yld)
    # S2-only features (12 band means + NDVI peak and mean): weather, season dates and terrain/soil are
    # (near-)constant within a year or region, so a classifier on them names the group trivially
    Xs2 = StandardScaler().fit_transform(np.where(np.isfinite(X[:, :14]), X[:, :14], 0.0))
    def shift(groups):
        res = {}
        g = np.array(groups)
        for k in sorted(set(groups), key=str):
            lab = (g == k).astype(int)
            if lab.sum() < 3 or (1 - lab).sum() < 3:
                res[str(k)] = {'n': int(lab.sum()), 'auroc': None, 'auroc_s2': None, 'yield_shift_sd': None}; continue
            cv = StratifiedKFold(n_splits=min(5, int(lab.sum())), shuffle=True, random_state=0)
            pr = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5), Xs, lab, cv=cv, method='predict_proba')[:, 1]
            pr2 = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5), Xs2, lab, cv=cv, method='predict_proba')[:, 1]
            res[str(k)] = {'n': int(lab.sum()), 'auroc': float(roc_auc_score(lab, pr)), 'auroc_s2': float(roc_auc_score(lab, pr2)),
                           'yield_mean': float(yld[lab == 1].mean()), 'rest_mean': float(yld[lab == 0].mean()),
                           'yield_shift_sd': float((yld[lab == 1].mean() - yld[lab == 0].mean()) / yld.std())}
        return res
    out[pair] = {'points': [{'x': float(e[0]), 'y': float(e[1]), 'yield': float(v), 'year': int(yr), 'region': str(r), 'name': n}
                            for e, v, yr, r, n in zip(emb, yld, year, region, names)],
                 'loyo': shift(year), 'loro': shift(region), 'n_fields': len(fl)}
    print(pair, len(fl), 'max year AUROC %.2f' % max(v['auroc'] or 0 for v in out[pair]['loyo'].values()), flush=True)
json.dump(out, open('/tmp/claude-0/-root-geo-yield-prediction/4069806e-d513-4b75-bff1-7f1baa52a93c/scratchpad/shift/shift_data.json', 'w'))
print('SHIFTDONE')
