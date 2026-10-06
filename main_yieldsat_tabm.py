"""TabM / MLP / LightGBM point models on the flat YieldSAT view (TM-04/TM-05).

Runs the DEV matrix locally (spec/yieldsat-tabm.md): per DEV row (pair x
protocol) the pair's flat matrix is built once (cached under
<artifact_root>/tabular/), then each fold is trained with train-only
standardization, field-balanced GPU sampling and early stopping on the
fold's validation seasons; test predictions and report.json use the point
model's formats, so yieldsat_pooled_metrics.py / yieldsat_dev_eval.py apply.

    python main_yieldsat_tabm.py --model tabm --features F0 --tag tm-tabm-f0 \
        --out_root /root/yieldsat_artifacts/runs/tabm
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from dataset.yieldsat_splits import PAPER_PAIRS, load_split, paper_row_prefix, parse_pair
from dataset.yieldsat_tabular import build_pair, fold_rows, standardize, standardize_stats
from util.yieldsat_eval import evaluate_predictions

DEV = (('Argentina', 'wheat', 'ARG-W'), ('Brazil', 'corn', 'BRA-C'), ('Germany', 'rapeseed', 'GER-R'),
       ('Uruguay', 'soybean', 'URG-S'))


def dev_rows(artifact_root, pairs=None, protocols=('cv', 'loyo', 'loro')):
    out = []
    for country, crop, code in DEV:
        if pairs and code not in pairs:
            continue
        for proto in protocols:
            if proto == 'cv':
                prefix, group = 'paper_cv10_{}_season_paper_s0'.format(code), 'cv10_season_paper_s0'
            elif proto == 'loyo':
                prefix, group = 'paper_loyo_{}_na_paper_s0'.format(code), 'loyo_na_paper_s0'
            elif code == 'ARG-W':
                prefix, group = 'paper_loro_{}_province_paper_s0'.format(code), 'loro_province_paper_s0'
            else:
                prefix, group = 'paper_loro_{}_na_paper_s0'.format(code), 'loro_na_paper_s0'
            folds = json.loads((Path(artifact_root) / 'splits' / '{}.folds.json'.format(prefix)).read_text())['folds']
            out.append({'country': country, 'crop': crop, 'pair': code, 'protocol': proto, 'group': group,
                        'folds': folds})
    return out


GROUP = {'cv': 'cv10_season', 'loyo': 'loyo_na', 'loro': 'loro_na'}


def protocol_rows(artifact_root, pairs=None, protocols=('cv', 'loyo', 'loro'), fold_set='dev'):
    """Rows of a fold set: ``dev`` (DEV pairs, validation carve-out), ``noval``
    (all 9 pairs, the paper's protocol: no validation set, selection on the test
    fold) or ``pooled`` (one row per protocol over all pairs, pair 'ALL');
    spec/yieldsat-paper-protocol.md."""
    if fold_set == 'dev':
        return dev_rows(artifact_root, pairs, protocols)
    out = []
    for proto in protocols:
        if fold_set == 'pooled':
            prefix = 'pooled_noval_{}_s0'.format({'cv': 'cv10', 'loyo': 'loyo', 'loro': 'loro'}[proto])
            todo = [(None, None, 'ALL', prefix, 'pooled_{}_noval_s0'.format(GROUP[proto].split('_')[0]))]
        else:
            todo = []
            for code in (pairs or PAPER_PAIRS):
                country, crop = parse_pair(code)
                prefix = 'noval_' + paper_row_prefix(code, proto)
                group = GROUP[proto] + ('_province' if proto == 'loro' and code.startswith('ARG') else '')
                todo.append((country, crop, code, prefix, '{}_noval_s0'.format(group)))
        for country, crop, code, prefix, group in todo:
            folds = json.loads((Path(artifact_root) / 'splits' / '{}.folds.json'.format(prefix)).read_text())['folds']
            out.append({'country': country, 'crop': crop, 'pair': code, 'protocol': proto, 'group': group,
                        'folds': folds})
    return out


def load_pooled(args, split):
    """All pairs for one pooled fold: training cells capped per season (the
    flat matrix of all 12.4M cells does not fit an A10), every test cell kept;
    country and crop indicators as plain columns. Memory-lean (a 48 Gi pod ran
    out on folds holding out a whole year): the small arrays are read first to
    size one preallocated matrix, then each pair's values are copied into it."""
    import zipfile
    countries = sorted({parse_pair(c)[0] for c in PAPER_PAIRS})
    crops = sorted({parse_pair(c)[1] for c in PAPER_PAIRS})
    plan, total = [], 0
    for code in PAPER_PAIRS:
        country, crop = parse_pair(code)
        cache = Path(args.artifact_root) / 'tabular' / '{}_{}_{}.npz'.format(country, crop, args.features)
        if not cache.exists() or not zipfile.is_zipfile(cache):
            load_pair(args.artifact_root, country, crop, args.features)     # (re)build the cache
        z = np.load(cache, allow_pickle=True)                               # lazy: members load on access
        light = {'season': z['season'], 'seasons': json.loads(str(z['seasons'])), 'target': z['target']}
        r = fold_rows(light, split)
        tr = r['train'][np.isfinite(light['target'][r['train']])]
        tr = cap_rows(light['season'], tr, args.pooled_train_rows_per_field, args.seed)
        plan.append((code, country, crop, cache, tr, r['test'], light))
        total += len(tr) + len(r['test'])
        names = list(z['value_names']), list(z['mask_names'])
        z.close()
    pair = {'values': np.empty((total, len(names[0])), np.float32), 'masks': np.empty((total, len(names[1])), np.uint8),
            'target': np.empty(total, np.float32), 'season': np.empty(total, np.int64),
            'grid_row': np.empty(total, np.int64), 'grid_col': np.empty(total, np.int64),
            'plain': np.zeros((total, len(countries) + len(crops)), np.float32)}
    seasons, train, test, n = [], [], [], 0
    for code, country, crop, cache, tr, te, light in plan:
        keep = np.concatenate([tr, te])
        sl = slice(n, n + len(keep))
        z = np.load(cache, allow_pickle=True)
        for k in ('values', 'masks', 'grid_row', 'grid_col'):
            a = z[k]
            pair[k][sl] = a[keep]
            del a
        z.close()
        pair['target'][sl] = light['target'][keep]
        pair['season'][sl] = light['season'][keep] + len(seasons)
        pair['plain'][sl, countries.index(country)] = 1
        pair['plain'][sl, len(countries) + crops.index(crop)] = 1
        seasons += [dict(s, country=country, crop=crop, pair=code) for s in light['seasons']]
        train.append(n + np.arange(len(tr)))
        test.append(n + len(tr) + np.arange(len(te)))
        n += len(keep)
    pair['seasons'] = seasons
    pair['value_names'], pair['mask_names'] = names
    pair['pooled'] = True
    rows = {'train': np.concatenate(train), 'test': np.concatenate(test)}
    rows['val'] = rows['test']
    return pair, rows


def standardize_inplace(values, train_rows, chunk=200000):
    """standardize_stats + standardize without copies of the matrix (pooled folds)."""
    s = np.zeros(values.shape[1]); s2 = np.zeros(values.shape[1]); c = np.zeros(values.shape[1])
    for a in range(0, len(train_rows), chunk):
        v = values[train_rows[a:a + chunk]].astype(np.float64)
        f = np.isfinite(v)
        v[~f] = 0
        s += v.sum(0); s2 += (v * v).sum(0); c += f.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = s / c
        std = np.sqrt(np.maximum(s2 / c - mean ** 2, 0))
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0).astype(np.float32)
    for a in range(0, len(values), chunk):
        z = (values[a:a + chunk] - mean) / std
        values[a:a + chunk] = np.where(np.isfinite(z), z, 0.0)
    return values


def load_pair(artifact_root, country, crop, features):
    import zipfile
    from dataset.yieldsat_schema import CACHE_NAME
    suffix = '' if CACHE_NAME == 'cache' else '_' + CACHE_NAME        # dense series: separate flat caches
    cache = Path(artifact_root) / 'tabular' / '{}_{}_{}{}.npz'.format(country, crop, features, suffix)
    if cache.exists() and not zipfile.is_zipfile(cache):
        # a corrupt cache on the shared volume (2026-10-06: Argentina_soybean_F0) is rebuilt, not trusted
        print('unreadable cache, rebuilding:', cache, flush=True)
        cache.unlink(missing_ok=True)
    if cache.exists():
        try:
            z = np.load(cache, allow_pickle=True)
            d = {k: z[k] for k in z.files}
            d['seasons'] = json.loads(str(d['seasons']))
            d['value_names'], d['mask_names'] = list(d['value_names']), list(d['mask_names'])
            return d
        except (zipfile.BadZipFile, ValueError, OSError, EOFError) as exc:
            # silently corrupted data on the shared volume (2026-10-06: Bad CRC-32 in Argentina_soybean_F1)
            print('corrupt cache, rebuilding:', cache, repr(exc), flush=True)
            cache.unlink(missing_ok=True)
    d = build_pair(artifact_root, country, crop, neighbourhood=features == 'F1')
    cache.parent.mkdir(parents=True, exist_ok=True)
    # atomic: several cluster pods may build the same pair concurrently
    tmp = cache.with_name('{}.{}-{}.partial.npz'.format(cache.stem, os.uname().nodename, os.getpid()))  # pids repeat across pods
    np.savez(tmp, **{k: v for k, v in d.items() if k != 'seasons'}, seasons=np.array(json.dumps(d['seasons'])))
    with zipfile.ZipFile(tmp) as z:                  # publish only a verified archive
        bad = z.testzip()                            # full CRC check of what actually reached the volume
        if bad is not None or len(z.namelist()) < 5:
            raise RuntimeError('cache write failed verification: {}'.format(tmp))
    os.replace(tmp, cache)
    return d


def balanced_sampler(season, train_rows, alpha, device, seed):
    """Season drawn with probability n**alpha, then a uniform cell of that season."""
    s = season[train_rows]
    order = np.argsort(s, kind='stable')
    rows = train_rows[order]
    ss = s[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]
    sizes = np.diff(np.r_[starts, len(ss)])
    p = torch.tensor(sizes.astype(np.float64) ** alpha, device=device)
    p = p / p.sum()
    starts_t = torch.tensor(starts, device=device)
    sizes_t = torch.tensor(sizes, device=device)
    rows_t = torch.tensor(rows, device=device)
    g = torch.Generator(device=device).manual_seed(seed)

    def draw(b):
        k = torch.multinomial(p, b, replacement=True, generator=g)
        off = (torch.rand(b, device=device, generator=g) * sizes_t[k]).long()
        return rows_t[starts_t[k] + off]
    return draw


def cap_rows(season, rows, per_field, seed):
    if not per_field:
        return rows
    rng = np.random.default_rng(seed)
    out = []
    for s in np.unique(season[rows]):
        r = rows[season[rows] == s]
        out.append(r if len(r) <= per_field else np.sort(rng.choice(r, per_field, replace=False)))
    return np.concatenate(out) if out else rows


@torch.no_grad()
def predict(model, V, M, rows, device, bs=4096, X=None):
    model.eval()
    out = []
    for a in range(0, len(rows), bs):
        r = torch.as_tensor(rows[a:a + bs], device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            out.append(model(V[r].float(), M[r], None if X is None else X[r].float()).float().cpu().numpy())
    model.train()
    return np.concatenate(out) if out else np.zeros(0)


def train_nn(args, pair, rows, Vn, device, Xn=None):
    from models_yieldsat_tabm import TabMRegressor
    torch.manual_seed(args.seed)
    y = pair['target']
    ym, ys = float(np.nanmean(y[rows['train']])), float(np.nanstd(y[rows['train']]))
    V = torch.empty(Vn.shape, device=device, dtype=torch.float16)      # chunked: no fp32 copy on the GPU
    for a in range(0, len(Vn), 500000):
        V[a:a + 500000] = torch.from_numpy(np.ascontiguousarray(Vn[a:a + 500000])).to(device).half()
    M = torch.as_tensor(pair['masks'], device=device)
    Y = torch.as_tensor(np.where(np.isfinite(y), (y - ym) / ys, 0.0), device=device, dtype=torch.float32)
    arch = {'tabm': 'tabm', 'tabm-mini': 'tabm-mini', 'mlp': 'mlp'}[args.model]
    X = None if Xn is None else torch.as_tensor(Xn, device=device, dtype=torch.float16)
    model = TabMRegressor(V.shape[1], M.shape[1], k=args.k, n_blocks=args.n_blocks, d_block=args.d_block,
                          dropout=args.dropout, d_embedding=args.d_embedding, arch_type=arch,
                          n_plain=0 if X is None else X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train = rows['train'][np.isfinite(y[rows['train']])]
    draw = balanced_sampler(pair['season'], train, args.field_alpha, device, args.seed)
    val = cap_rows(pair['season'], rows['val'], args.val_rows_per_field, args.seed) if len(rows['val']) else None
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for step in range(1, args.max_steps + 1):
        idx = draw(args.batch_size)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            loss = model.loss(V[idx].float(), M[idx], Y[idx], None if X is None else X[idx].float())
        opt.zero_grad(set_to_none=True)
        loss.float().backward()
        opt.step()
        if step % args.eval_every == 0 or step == args.max_steps:
            rec = {'step': step, 'train_loss': float(loss), 'seconds': round(time.time() - t0, 1)}
            if val is not None:
                pv = predict(model, V, M, val, device, X=X) * ys + ym
                rec['val_pixel_rmse'] = float(np.sqrt(np.nanmean((pv - y[val]) ** 2)))
                if rec['val_pixel_rmse'] < best:
                    best, bad = rec['val_pixel_rmse'], 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
            hist.append(rec)
            if val is not None and bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    pt = predict(model, V, M, rows['test'], device, X=X) * ys + ym
    del V, M, Y, X
    torch.cuda.empty_cache()
    return pt, {'history': hist, 'best_val_pixel_rmse': None if best == np.inf else best, 'steps': step,
                'model': model.config, 'samples': step * args.batch_size}


def train_lgbm(args, pair, rows):
    import lightgbm as lgb
    y = pair['target']
    X = np.concatenate([pair['values'], pair['masks'].astype(np.float32)] +
                       ([pair['plain']] if 'plain' in pair else []), 1)
    tr = rows['train'][np.isfinite(y[rows['train']])]
    tr = cap_rows(pair['season'], tr, args.lgbm_rows_per_field, args.seed)
    va = cap_rows(pair['season'], rows['val'], args.val_rows_per_field, args.seed) if len(rows['val']) else None
    dtr = lgb.Dataset(X[tr], y[tr], free_raw_data=True)
    params = {'objective': 'regression', 'learning_rate': 0.05, 'num_leaves': 63, 'min_data_in_leaf': 100,
              'feature_fraction': 0.5, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l2': 1.0,
              'num_threads': args.lgbm_threads, 'verbose': -1, 'seed': args.seed}
    cb = []
    valid = []
    if va is not None:
        valid = [lgb.Dataset(X[va], y[va], reference=dtr)]
        cb = [lgb.early_stopping(100, verbose=False)]
    booster = lgb.train(params, dtr, num_boost_round=args.lgbm_rounds, valid_sets=valid, callbacks=cb)
    pt = booster.predict(X[rows['test']], num_iteration=booster.best_iteration or None)
    return pt, {'best_iteration': booster.best_iteration, 'train_rows': int(len(tr)), 'model': {'lgbm': params}}


def run_fold(args, row, fold_i, split_name, pair, device):
    out = Path(args.out_root) / 'paper' / row['group'] / 's2_adm' / '{}_seed{}'.format(args.tag, args.seed) / \
        row['pair'] / 'fold{:02d}'.format(fold_i)
    if (out / 'report.json').exists():
        return None
    t0 = time.time()
    split = load_split(args.artifact_root, split_name)
    if row['pair'] == 'ALL':
        pair, rows = load_pooled(args, split)
    else:
        rows = fold_rows(pair, split)
    if args.model == 'lgbm':
        pt, info = train_lgbm(args, pair, rows)
    else:
        mean, std = (None, None) if pair.get('pooled') else standardize_stats(pair['values'], rows['train'])
        Xn, unit = (pair['plain'], None) if 'plain' in pair else (None, None)
        if args.pretrained_arm:
            from yieldsat_tabm_pretrained import pretrained_features
            X, unit = pretrained_features(args.artifact_root, args.pretrain_root, args.pretrained_arm, split_name,
                                          pair, row['country'], row['crop'], args.pretrained_kind, device)
            xm, xs = standardize_stats(X, rows['train'])
            Xn = standardize(X, xm, xs)
        Vn = (standardize_inplace(pair['values'], rows['train']) if pair.get('pooled')
              else standardize(pair['values'], mean, std))
        pt, info = train_nn(args, pair, rows, Vn, device, Xn)
        if unit:
            info['pretrained'] = {'arm': args.pretrained_arm, 'kind': args.pretrained_kind, 'unit': unit,
                                  'columns': int(Xn.shape[1])}
    te = rows['test']
    s_all = pair['season'][te]
    uniq = np.unique(s_all)
    remap = {int(s): i for i, s in enumerate(uniq)}
    seasons = [{'country': pair['seasons'][s].get('country', row['country']),
                'crop': pair['seasons'][s].get('crop', row['crop']),
                **{k: pair['seasons'][s][k] for k in ('season_id', 'field_shared_name')}} for s in uniq]
    s_local = np.array([remap[int(s)] for s in s_all])
    test = evaluate_predictions(pt, pair['target'][te], s_local, seasons)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'test_predictions.npz', pred=pt.astype(np.float32), target=pair['target'][te],
                        season=s_local, grid_row=pair['grid_row'][te], grid_col=pair['grid_col'][te],
                        season_names=np.array([s['field_shared_name'] for s in seasons]))
    rep = {'mode': 'tabular', 'model': args.model, 'features': args.features, 'split': split_name,
           'selection': split.get('selection', 'validation'),
           'args': vars(args), 'train_seasons': int(len(np.unique(pair['season'][rows['train']]))),
           'test': test, 'info': info, 'resources': {'wall_seconds': round(time.time() - t0, 1)}}
    (out / 'report.json').write_text(json.dumps(rep, default=str))
    o = test['overall']
    return {'fold': split_name, 'pixel_r2': o['pixel']['r2'], 'field_r2': o['field_level']['r2'],
            'seconds': rep['resources']['wall_seconds'], 'steps': info.get('steps', info.get('best_iteration'))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--artifact_root', default='/root/yieldsat_artifacts')
    p.add_argument('--out_root', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--model', default='tabm', choices=['tabm', 'tabm-mini', 'mlp', 'lgbm'])
    p.add_argument('--features', default='F0', choices=['F0', 'F1'])
    p.add_argument('--pairs', nargs='*', default=None)
    p.add_argument('--protocols', nargs='+', default=['cv', 'loyo', 'loro'])
    p.add_argument('--max_folds', type=int, default=0)
    p.add_argument('--folds', type=int, nargs='*', default=None, help='fold indices to run (cluster work units)')
    p.add_argument('--pretrained_arm', default='', help='TM-2: A2/A3/A7 unit checkpoints as extra features')
    p.add_argument('--pretrained_kind', default='emb', choices=['emb', 'cpt', 'all'])
    p.add_argument('--pretrain_root', default='/data/YieldSAT/yieldsat_results/pk_dev1r')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--k', type=int, default=32)
    p.add_argument('--n_blocks', type=int, default=3)
    p.add_argument('--d_block', type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--d_embedding', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=3e-4)
    p.add_argument('--batch_size', type=int, default=4096)
    p.add_argument('--max_steps', type=int, default=3000)
    p.add_argument('--eval_every', type=int, default=100)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--field_alpha', type=float, default=0.5)
    p.add_argument('--val_rows_per_field', type=int, default=500,
                   help='cells per selection season (0 = all; the paper protocol selects on the full test fold)')
    p.add_argument('--fold_set', default='dev', choices=['dev', 'noval', 'pooled'],
                   help='dev: DEV rows with validation; noval/pooled: the paper protocol per pair / pooled')
    p.add_argument('--pooled_train_rows_per_field', type=int, default=1000)
    p.add_argument('--build_only', action='store_true', help='build the --pairs flat-matrix caches and exit')
    p.add_argument('--lgbm_rows_per_field', type=int, default=3000)
    p.add_argument('--lgbm_rounds', type=int, default=3000)
    p.add_argument('--lgbm_threads', type=int, default=20)
    a = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_root = Path(a.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if a.build_only:
        for code in a.pairs:
            load_pair(a.artifact_root, *parse_pair(code), a.features)
        return
    runs = out_root / 'runs.jsonl'
    log = out_root / '{}_log.jsonl'.format(a.tag)
    for row in protocol_rows(a.artifact_root, a.pairs, a.protocols, a.fold_set):
        pair = None
        folds = row['folds'][:a.max_folds or None]
        for i, name in enumerate(folds):
            if a.folds is not None and i not in a.folds:
                continue
            rel = 'paper/{}/s2_adm/{}_seed{}/{}/fold{:02d}'.format(row['group'], a.tag, a.seed, row['pair'], i)
            with open(runs, 'a') as f:
                f.write(json.dumps({'rel_path': rel, 'experiment': a.tag, 'protocol': row['protocol'],
                                    'pair': row['pair']}) + '\n')
            if (out_root / rel / 'report.json').exists():
                continue
            if pair is None and row['pair'] != 'ALL':
                pair = load_pair(a.artifact_root, row['country'], row['crop'], a.features)
            rec = run_fold(a, row, i, name, pair, device)
            if rec:
                rec.update(tag=a.tag, pair=row['pair'], protocol=row['protocol'])
                print(json.dumps(rec), flush=True)
                with open(log, 'a') as f:
                    f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main()
