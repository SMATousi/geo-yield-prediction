"""Success-criteria diagnostics for one point-model pretraining run (PK-07).

Computes, on the seasons the run's pretraining unit EXCLUDED (the DEV test
seasons) and never on pretraining data:

  P2  per-stream embedding variance and effective rank vs a fresh initialization
  I1  concept-grounding AUROC (per concept, per country)
  I2  relational alignment cos(normalize(z_B - z_A), r_AB) per rule, per season
      (also at initialization) for paired comparisons between arms
  I3  ridge probes from the frozen fused embedding to physical properties
      (fitted on unit training seasons, scored on held-out seasons)
  I4  masked-observation and forecast losses of the run's own SSL heads

Writes ``<run_dir>/diagnostics.json``. Cross-arm verdicts (I1 vs shuffled,
I3/I4 vs SSL, E*) are made by the round report. Criteria:
spec/pretraining/success-criteria.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dataset.yieldsat_dataset import (
    NormalizerView, SequentialBatchSampler, YieldSATNormalizer, YieldSATPointDataset, collate_point_batch,
)
from dataset.yieldsat_splits import load_field_table, load_split
from models_yieldsat import YieldSATPointModel, pack_inputs
from yieldsat_build_knowledge import estimator_cutoff, load_raw
from yieldsat_knowledge_point import (KnowledgePointPretrainer, KnowledgeReference, attach_knowledge,
                                      load_library, load_text_vectors)
from yieldsat_objectives import YieldSATPointPretrainer

PROBE_PROPERTIES = ('precip_mid_mm', 'tmean_mid_c', 'ndvi_rise', 'ndmi_mid', 'clay030', 'soc030_gkg', 'rel_elev')
MODEL_KEYS = ('embed_dim', 'num_latents', 'depth', 'num_heads', 'modality_embed', 'modality_dropout',
              'time_dim', 'num_slots', 'fusion')


def effective_rank(x):
    s = np.linalg.svd(x - x.mean(0, keepdims=True), compute_uv=False)
    p = s / max(s.sum(), 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def auroc(score, label):
    from sklearn.metrics import roc_auc_score
    if label.min() == label.max():
        return None
    return float(roc_auc_score(label, score))


def ridge_r2(xtr, ytr, xte, yte, alpha=1.0):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    a, b = np.isfinite(ytr), np.isfinite(yte)
    if a.sum() < 20 or b.sum() < 20:
        return None
    m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(xtr[a], ytr[a])
    p = m.predict(xte[b])
    return float(1 - ((yte[b] - p) ** 2).sum() / max(((yte[b] - yte[b].mean()) ** 2).sum(), 1e-12))


class Run:
    def __init__(self, run_dir, device, artifact_root=None, source_root=None, skip_source_check=False):
        self.dir = Path(run_dir)
        self.report = json.loads((self.dir / 'report.json').read_text())
        a = dict(self.report['args'])
        a['artifact_root'] = artifact_root or a['artifact_root']
        a['source_root'] = source_root or a.get('source_root')
        a['skip_source_check'] = skip_source_check or a.get('skip_source_check', False)
        self.args = a
        self.device = torch.device(device)
        norm = json.loads((self.dir / 'normalizer.json').read_text())
        self.normalizer = NormalizerView(YieldSATNormalizer.from_json(norm), **norm.get('policy', {}))
        self.sensor = torch.load(self.dir / 'sensor_checkpoint.pth', map_location='cpu', weights_only=True)
        self.heads = torch.load(self.dir / 'pretrainer_heads.pth', map_location='cpu', weights_only=True)
        self.knowledge = self.heads['knowledge']
        self.lib = load_library()
        self.reference = None
        if self.knowledge:
            self.reference = KnowledgeReference.from_json(
                json.loads((self.dir / 'knowledge_reference.json').read_text()))

    def dataset(self, seasons, rows_per_field):
        a = self.args
        return YieldSATPointDataset(
            a['source_root'], a['artifact_root'], seasons, self.normalizer, streams=a['streams'],
            backend=a['backend'], cutoff_mode=a['cutoff_mode'], cutoff_days=a['cutoff_days'],
            soil_uncertainty=a['soil_uncertainty'], aspect_encoding=a['aspect_encoding'], seed=a['seed'],
            fill_value=a['fill_value'], check_source=not a['skip_source_check'],
            max_rows_per_field=rows_per_field)

    def build(self, layout, trained=True):
        cfg = self.sensor['descriptor']['config']
        torch.manual_seed(self.args['seed'])
        model = YieldSATPointModel(layout, use_crop_context=False, **{k: cfg[k] for k in MODEL_KEYS})
        if trained:
            model.load_sensor_state_dict(self.sensor)
        if self.knowledge:
            text, _ = load_text_vectors(self.args.get('knowledge_text_dir') or
                                        str(Path(self.args['artifact_root']) / 'knowledge' / 'text_clip_b32'),
                                        self.lib)
            control = self.heads['control']
            # random_targets only changed the training targets; its heads are real-text heads
            control = None if control in ('none', 'random_targets') else control
            pt = KnowledgePointPretrainer(model, text, self.lib, control=control,
                                          shuffle_seed=self.args['seed'])
        else:
            pt = YieldSATPointPretrainer(model)
        if trained:
            missing = pt.load_state_dict(self.heads['state_dict'], strict=False)
            bad = [k for k in missing.missing_keys if not k.startswith(('model.', 'ssl.model.'))]
            if bad:
                raise ValueError('pretrainer heads missing: {}'.format(bad[:5]))
        return pt.to(self.device).eval()


@torch.no_grad()
def embed(pt, ds, device, batch_size=2048, ssl=True):
    """Per-item stream embeddings, fused embedding, projected concept scores and SSL losses."""
    loader = torch.utils.data.DataLoader(ds, batch_sampler=SequentialBatchSampler(len(ds), batch_size),
                                         collate_fn=collate_point_batch, num_workers=2)
    model = pt.model
    out = {'stream': {}, 'avail': {}, 'fused': [], 'season': [], 'z': {}, 'item': []}
    ssl_num, ssl_den = {}, 0
    torch.manual_seed(1234)
    for batch in loader:
        b = {k: ({kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v.to(device))
             for k, v in batch.items()}
        inputs = pack_inputs(b, model.layout)
        avail = {n: b['available'][n] for n in model.streams}
        emb, mask = model.encoders.forward_with_missing(inputs, available=avail, apply_dropout=False)
        for n in model.streams:
            out['stream'].setdefault(n, []).append(emb[n][:, 0].float().cpu().numpy())
            out['avail'].setdefault(n, []).append((mask[n] > 0.5).cpu().numpy())
        lat, _ = model.forward_features(b, apply_dropout=False)
        out['fused'].append(lat.mean(1).float().cpu().numpy())
        out['season'].append(batch['season'].numpy())
        out['item'].append(batch['item'].numpy())
        if isinstance(pt, KnowledgePointPretrainer):
            z, _ = pt.stream_embeddings(b)
            for s, v in z.items():
                out['z'].setdefault(s, []).append(v.float().cpu().numpy())
        if ssl:
            _, losses = (pt.ssl if isinstance(pt, KnowledgePointPretrainer) else pt)(b)
            n = len(batch['season'])
            for k, v in losses.items():
                ssl_num[k] = ssl_num.get(k, 0.0) + float(v) * n
            ssl_den += n
    cat = np.concatenate
    res = {'stream': {k: cat(v) for k, v in out['stream'].items()},
           'avail': {k: cat(v) for k, v in out['avail'].items()},
           'fused': cat(out['fused']), 'season': cat(out['season']), 'item': cat(out['item']),
           'z': {k: cat(v) for k, v in out['z'].items()}}
    res['ssl'] = {k: v / max(ssl_den, 1) for k, v in ssl_num.items()}
    return res


def collapse_stats(emb, init):
    out = {}
    for s, x in emb['stream'].items():
        a = emb['avail'][s]
        if a.sum() < 10:
            continue
        xi = init['stream'][s][a]
        x = x[a]
        v, v0 = float(x.var(0).mean()), float(xi.var(0).mean())
        er, er0 = effective_rank(x), effective_rank(xi)
        # a stream's attainable rank is bounded by its input (DEM is one scalar), so
        # collapse is judged against the same stream at initialization
        out[s] = {'variance': v, 'variance_init': v0, 'variance_ratio': v / max(v0, 1e-12),
                  'effective_rank': er, 'effective_rank_init': er0, 'effective_rank_ratio': er / max(er0, 1e-12),
                  'pass': bool(v >= 0.5 * v0 and er >= 0.5 * er0)}
    return out


def grounding(pt, emb, targets, countries):
    proto = pt.prototypes().cpu().numpy()
    out = {}
    for i, cid in enumerate(pt.concepts):
        s = pt.concept_stream[i]
        score = emb['z'][s] @ proto[i]
        t = targets[:, i].astype(np.float64)
        ok = np.isfinite(t) & emb['avail'][s]
        rec = {'n': int(ok.sum()), 'auroc': auroc(score[ok], t[ok] >= 0.5) if ok.sum() > 20 else None,
               'per_country': {}}
        for c in np.unique(countries):
            m = ok & (countries == c)
            if m.sum() > 20:
                rec['per_country'][str(c)] = auroc(score[m], t[m] >= 0.5)
        out[cid] = rec
    return out


def relational(pt, emb, targets, gates, season_names):
    rel = pt.relations().cpu().numpy()
    out = {}
    for j, (a, b) in enumerate(pt.pairs):
        sa, sb = pt.concept_stream[a], pt.concept_stream[b]
        pa, pb, g = targets[:, a], targets[:, b], gates[:, j]
        ok = (np.isfinite(pa) & np.isfinite(pb) & (pa >= pt.min_presence[j]) & (pb >= pt.min_presence[j])
              & (g > 0) & emb['avail'][sa] & emb['avail'][sb])
        if not ok.any():
            out[pt.rule_ids[j]] = {'n': 0, 'mean': None, 'per_season': {}}
            continue
        d = emb['z'][sb][ok] - emb['z'][sa][ok]
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        cos = d @ rel[j]
        per = {}
        for s in np.unique(emb['season'][ok]):
            per[season_names[s]] = float(cos[emb['season'][ok] == s].mean())
        out[pt.rule_ids[j]] = {'n': int(ok.sum()), 'mean': float(np.mean(list(per.values()))), 'per_season': per}
    return out


def raw_properties(ds, artifact_root, cutoff_days=0):
    raw = {c: load_raw(artifact_root, c, cutoff_days) for c in ds.countries}
    out = {k: np.full(len(ds), np.nan) for k in PROBE_PROPERTIES}
    for ci, c in enumerate(ds.countries):
        sel = np.flatnonzero(ds.country_of == ci)
        for k in PROBE_PROPERTIES:
            out[k][sel] = raw[c][k][ds.row[sel]]
    return out, raw


def run(run_dir, device='cuda', heldout_rows=64, probe_rows=16, probe_seasons=400, artifact_root=None,
        source_root=None, skip_source_check=False):
    r = Run(run_dir, device, artifact_root, source_root, skip_source_check)
    a = r.args
    table = load_field_table(a['artifact_root'], a['source_root'], a['countries'],
                             check_source=not a['skip_source_check'])
    split = load_split(a['artifact_root'], a['split'])
    by = {f['season_id']: f for f in table['fields']}
    held = [by[s] for s in split['partitions'].get('excluded', []) if s in by]
    if not held:
        raise SystemExit('the run split has no excluded (held-out) seasons')
    rng = np.random.default_rng(0)
    train_ids = [s for s in split['partitions']['train'] if s in by]
    train = [by[s] for s in sorted(rng.choice(train_ids, min(probe_seasons, len(train_ids)), replace=False))]
    ds_h, ds_t = r.dataset(held, heldout_rows), r.dataset(train, probe_rows)
    pt, pt0 = r.build(ds_h.layout, True), r.build(ds_h.layout, False)
    e_h, e0_h = embed(pt, ds_h, r.device), embed(pt0, ds_h, r.device, ssl=False)
    e_t = embed(pt, ds_t, r.device, ssl=False)
    cutoff = estimator_cutoff(a['cutoff_mode'], a['cutoff_days'])
    props_h, raw = raw_properties(ds_h, a['artifact_root'], cutoff)
    props_t, _ = raw_properties(ds_t, a['artifact_root'], cutoff)
    names = [s['season_id'] for s in ds_h.seasons]
    res = {'run_dir': str(run_dir), 'split': a['split'], 'knowledge': r.knowledge, 'control': r.heads['control'],
           'heldout_seasons': len(held), 'heldout_rows': int(len(ds_h)),
           'P2_collapse': collapse_stats(e_h, e0_h),
           'I3_probe_r2': {k: ridge_r2(e_t['fused'], props_t[k], e_h['fused'], props_h[k]) for k in PROBE_PROPERTIES},
           'I4_ssl_heldout': e_h['ssl']}
    if r.knowledge:
        ref = r.reference
        attach_knowledge(ds_h, ref, a['artifact_root'], raw, cutoff)
        tg = ds_h.knowledge['concept_target'].astype(np.float32)
        gt = ds_h.knowledge['rule_gate'].astype(np.float32)
        countries = np.array(ds_h.countries)[ds_h.country_of]
        res['I1_grounding'] = grounding(pt, e_h, tg, countries)
        res['I2_relational'] = relational(pt, e_h, tg, gt, names)
        res['I2_relational_init'] = relational(pt0, e0_h, tg, gt, names)
    out = Path(run_dir) / 'diagnostics.json'
    out.write_text(json.dumps(res, indent=1))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dirs', nargs='+')
    p.add_argument('--device', default='cuda')
    p.add_argument('--artifact_root', default=None)
    p.add_argument('--source_root', default=None)
    p.add_argument('--skip_source_check', action='store_true')
    p.add_argument('--heldout_rows', type=int, default=64)
    a = p.parse_args()
    for d in a.run_dirs:
        res = run(d, a.device, a.heldout_rows, artifact_root=a.artifact_root, source_root=a.source_root,
                  skip_source_check=a.skip_source_check)
        brief = {'P2': {k: v['pass'] for k, v in res['P2_collapse'].items()},
                 'I3': {k: (None if v is None else round(v, 3)) for k, v in res['I3_probe_r2'].items()},
                 'I4': {k: round(v, 4) for k, v in res['I4_ssl_heldout'].items() if 'terms' not in k}}
        if res['knowledge']:
            brief['I1'] = {k: (None if v['auroc'] is None else round(v['auroc'], 3)) for k, v in res['I1_grounding'].items()}
            brief['I2'] = {k: (None if v['mean'] is None else round(v['mean'], 3)) for k, v in res['I2_relational'].items()}
            brief['I2_init'] = {k: (None if v['mean'] is None else round(v['mean'], 3))
                                for k, v in res['I2_relational_init'].items()}
        print(d, json.dumps(brief))


if __name__ == '__main__':
    main()
