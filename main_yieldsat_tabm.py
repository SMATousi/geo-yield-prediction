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
import time
from pathlib import Path

import numpy as np
import torch

from dataset.yieldsat_splits import load_split
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


def load_pair(artifact_root, country, crop, features):
    cache = Path(artifact_root) / 'tabular' / '{}_{}_{}.npz'.format(country, crop, features)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        d['seasons'] = json.loads(str(d['seasons']))
        d['value_names'], d['mask_names'] = list(d['value_names']), list(d['mask_names'])
        return d
    d = build_pair(artifact_root, country, crop, neighbourhood=features == 'F1')
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **{k: v for k, v in d.items() if k != 'seasons'}, seasons=np.array(json.dumps(d['seasons'])))
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
    rng = np.random.default_rng(seed)
    out = []
    for s in np.unique(season[rows]):
        r = rows[season[rows] == s]
        out.append(r if len(r) <= per_field else np.sort(rng.choice(r, per_field, replace=False)))
    return np.concatenate(out) if out else rows


@torch.no_grad()
def predict(model, V, M, rows, device, bs=16384):
    model.eval()
    out = []
    for a in range(0, len(rows), bs):
        r = torch.as_tensor(rows[a:a + bs], device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            out.append(model(V[r].float(), M[r]).float().cpu().numpy())
    model.train()
    return np.concatenate(out) if out else np.zeros(0)


def train_nn(args, pair, rows, Vn, device):
    from models_yieldsat_tabm import TabMRegressor
    torch.manual_seed(args.seed)
    y = pair['target']
    ym, ys = float(np.nanmean(y[rows['train']])), float(np.nanstd(y[rows['train']]))
    V = torch.as_tensor(Vn, device=device, dtype=torch.float16)
    M = torch.as_tensor(pair['masks'], device=device)
    Y = torch.as_tensor(np.where(np.isfinite(y), (y - ym) / ys, 0.0), device=device, dtype=torch.float32)
    arch = {'tabm': 'tabm', 'tabm-mini': 'tabm-mini', 'mlp': 'mlp'}[args.model]
    model = TabMRegressor(V.shape[1], M.shape[1], k=args.k, n_blocks=args.n_blocks, d_block=args.d_block,
                          dropout=args.dropout, d_embedding=args.d_embedding, arch_type=arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train = rows['train'][np.isfinite(y[rows['train']])]
    draw = balanced_sampler(pair['season'], train, args.field_alpha, device, args.seed)
    val = cap_rows(pair['season'], rows['val'], args.val_rows_per_field, args.seed) if len(rows['val']) else None
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for step in range(1, args.max_steps + 1):
        idx = draw(args.batch_size)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            loss = model.loss(V[idx].float(), M[idx], Y[idx])
        opt.zero_grad(set_to_none=True)
        loss.float().backward()
        opt.step()
        if step % args.eval_every == 0 or step == args.max_steps:
            rec = {'step': step, 'train_loss': float(loss), 'seconds': round(time.time() - t0, 1)}
            if val is not None:
                pv = predict(model, V, M, val, device) * ys + ym
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
    pt = predict(model, V, M, rows['test'], device) * ys + ym
    del V, M, Y
    torch.cuda.empty_cache()
    return pt, {'history': hist, 'best_val_pixel_rmse': None if best == np.inf else best, 'steps': step,
                'model': model.config, 'samples': step * args.batch_size}


def train_lgbm(args, pair, rows):
    import lightgbm as lgb
    y = pair['target']
    X = np.concatenate([pair['values'], pair['masks'].astype(np.float32)], 1)
    rng = np.random.default_rng(args.seed)
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
    rows = fold_rows(pair, split)
    if args.model == 'lgbm':
        pt, info = train_lgbm(args, pair, rows)
    else:
        mean, std = standardize_stats(pair['values'], rows['train'])
        pt, info = train_nn(args, pair, rows, standardize(pair['values'], mean, std), device)
    te = rows['test']
    s_all = pair['season'][te]
    uniq = np.unique(s_all)
    remap = {int(s): i for i, s in enumerate(uniq)}
    seasons = [{'country': row['country'], 'crop': row['crop'], **{k: pair['seasons'][s][k] for k in
               ('season_id', 'field_shared_name')}} for s in uniq]
    s_local = np.array([remap[int(s)] for s in s_all])
    test = evaluate_predictions(pt, pair['target'][te], s_local, seasons)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'test_predictions.npz', pred=pt.astype(np.float32), target=pair['target'][te],
                        season=s_local, grid_row=pair['grid_row'][te], grid_col=pair['grid_col'][te],
                        season_names=np.array([s['field_shared_name'] for s in seasons]))
    rep = {'mode': 'tabular', 'model': args.model, 'features': args.features, 'split': split_name,
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
    p.add_argument('--val_rows_per_field', type=int, default=500)
    p.add_argument('--lgbm_rows_per_field', type=int, default=3000)
    p.add_argument('--lgbm_rounds', type=int, default=3000)
    p.add_argument('--lgbm_threads', type=int, default=20)
    a = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_root = Path(a.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    runs = out_root / 'runs.jsonl'
    log = out_root / '{}_log.jsonl'.format(a.tag)
    for row in dev_rows(a.artifact_root, a.pairs, a.protocols):
        pair = None
        folds = row['folds'][:a.max_folds or None]
        for i, name in enumerate(folds):
            rel = 'paper/{}/s2_adm/{}_seed{}/{}/fold{:02d}'.format(row['group'], a.tag, a.seed, row['pair'], i)
            with open(runs, 'a') as f:
                f.write(json.dumps({'rel_path': rel, 'experiment': a.tag, 'protocol': row['protocol'],
                                    'pair': row['pair']}) + '\n')
            if (out_root / rel / 'report.json').exists():
                continue
            if pair is None:
                pair = load_pair(a.artifact_root, row['country'], row['crop'], a.features)
            rec = run_fold(a, row, i, name, pair, device)
            if rec:
                rec.update(tag=a.tag, pair=row['pair'], protocol=row['protocol'])
                print(json.dumps(rec), flush=True)
                with open(log, 'a') as f:
                    f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main()
