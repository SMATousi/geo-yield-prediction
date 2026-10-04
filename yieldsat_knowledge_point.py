"""Knowledge pretraining for the YieldSAT point model (PK-03).

``KnowledgeReference`` turns raw concept indices (``yieldsat_build_knowledge.py``)
into soft concept targets and rule gates, using percentile tables fitted on a
pretraining unit's training rows only. ``KnowledgePointPretrainer`` adds concept
grounding and relational distillation to the point SSL objectives. See
spec/yieldsat-point-knowledge-pretraining.md §2-§4.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dataset.yieldsat_schema import CROPS
from models_yieldsat import pack_inputs
from yieldsat_build_knowledge import load_raw
from yieldsat_objectives import YieldSATPointPretrainer

LIBRARY = Path(__file__).resolve().parent / 'yieldsat_knowledge' / 'assets' / 'library.json'
REFERENCE_FORMAT = 'yieldsat_knowledge_reference_v1'
# library stream names (image package) -> point-model streams
STREAM_MAP = {'weather': 'yieldsat_weather', 'series': 'yieldsat_s2', 'spec': 'yieldsat_s2',
              'soil': 'yieldsat_soil', 'dem': 'yieldsat_dem', 'terrain': 'yieldsat_terrain'}
# concept -> (raw index, stratum, transform); see spec §3
ESTIMATORS = {
    'rain_supported': ('precip_mid_mm', 'country_crop', 'rank'),
    'warm_regime': ('tmean_mid_c', 'country_crop', 'rank'),
    'cool_regime': ('tmean_mid_c', 'country_crop', 'one_minus_rank'),
    'optical_growth': ('ndvi_rise', 'country_crop', 'rank'),
    'active_spectral_state': ('ndre_mid', 'country_crop', 'rank'),
    'persistent_canopy': ('persist_ratio', 'country_crop', 'rank'),
    'spectral_moisture_state': ('ndmi_mid', 'country_crop', 'rank'),
    'clay_rich_surface': ('clay030', 'country', 'rank'),
    'fine_surface_texture': ('fine030', 'country', 'rank'),
    'organic_surface': ('soc030_gkg', 'country', 'rank'),
    'low_elevation_position': ('elev_low_rank', None, 'direct'),
    'pole_facing_aspect': ('pole_cos', None, 'direct'),
}
NON_STRESS_TMAX_C = {'corn': 35.0, 'soybean': 35.0, 'wheat': 32.0, 'rapeseed': 30.0}
ORGANIC_SOC_GKG = 120.0
MIN_RELIEF_M = 1.0
RAINFED_SUMMER = ('corn', 'soybean')
QUANTILES = 1001
MIN_REFERENCE_SEASONS = 20


def load_library(path=LIBRARY, require_approved=True):
    lib = json.loads(Path(path).read_text())
    if require_approved:
        bad = [x['id'] for x in lib['concepts'] + lib['rules']
               if x.get('review', {}).get('status') != 'approved']
        if bad:
            raise ValueError('library entries lack expert review: {}'.format(bad))
    missing = [c['id'] for c in lib['concepts'] if c['id'] not in ESTIMATORS]
    if missing:
        raise ValueError('no point estimator for concepts {}'.format(missing))
    return lib


def _weighted_quantiles(values, weights, q=QUANTILES):
    ok = np.isfinite(values)
    v, w = values[ok].astype(np.float64), weights[ok].astype(np.float64)
    order = np.argsort(v, kind='stable')
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    cw = (cw - 0.5 * w) / cw[-1]
    return np.interp(np.linspace(0, 1, q), cw, v)


def _rank(values, grid):
    """Mid-rank percentile of values in a quantile grid (ties -> midpoint of the tied span)."""
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    lo = np.searchsorted(grid, np.where(ok, v, 0.0), side='left')
    hi = np.searchsorted(grid, np.where(ok, v, 0.0), side='right')
    return np.where(ok, 0.5 * (lo + hi) / len(grid), np.nan)


class KnowledgeReference:
    """Fold-train percentile tables per (raw index, stratum) and gate thresholds."""

    def __init__(self, tables, slope_p25, concepts, rules, meta=None):
        self.tables = tables            # {'raw|stratum': np.ndarray grid}
        self.slope_p25 = slope_p25      # {country: float}
        self.concepts = concepts
        self.rules = rules
        self.meta = meta or {}

    @staticmethod
    def stratum(kind, country, crop):
        if kind == 'country_crop':
            return '{}/{}'.format(country, crop)
        return country

    @classmethod
    def fit(cls, train, library=None):
        """train: {country: (raw dict, train row indices)}. Rows are weighted
        1/size of their field season so each season counts once."""
        lib = library or load_library()
        concepts = [c['id'] for c in lib['concepts']]
        rules = [{k: r[k] for k in ('id', 'concept_a', 'concept_b', 'minimum_presence')} for r in lib['rules']]
        tables, slope_p25, counts = {}, {}, {}
        for country, (raw, rows) in train.items():
            rows = np.asarray(rows)
            season = raw['season'][rows]
            w = 1.0 / np.bincount(season)[season]
            crop = raw['crop'][rows]
            slope = raw['slope'][rows]
            if np.isfinite(slope).any():
                slope_p25[country] = float(_weighted_quantiles(slope, w, 101)[25])
            for raw_name, kind, how in ESTIMATORS.values():
                if kind is None:
                    continue
                groups = [(cls.stratum(kind, country, CROPS[k]), crop == k) for k in np.unique(crop)] \
                    if kind == 'country_crop' else [(country, np.ones(len(rows), bool))]
                for key, sel in groups:
                    name = '{}|{}'.format(raw_name, key)
                    if name in tables:
                        continue
                    v = raw[raw_name][rows][sel]
                    ok = np.isfinite(v)
                    n_seasons = len(np.unique(season[sel][ok]))
                    counts[name] = n_seasons
                    if n_seasons >= MIN_REFERENCE_SEASONS:
                        tables[name] = _weighted_quantiles(v, w[sel])
        return cls(tables, slope_p25, concepts, rules, {'reference_seasons': counts,
                                                         'countries': sorted(train)})

    def transform(self, raw, country, rows):
        """(targets (n, n_concepts), gates (n, n_rules)) float32; NaN = unknown."""
        rows = np.asarray(rows)
        crop_idx = raw['crop'][rows]
        crops = np.array(CROPS)[crop_idx]
        n = len(rows)
        targets = np.full((n, len(self.concepts)), np.nan, dtype=np.float32)
        rank_cache = {}

        def ranked(raw_name, kind):
            key = (raw_name, kind)
            if key not in rank_cache:
                out = np.full(n, np.nan)
                v = raw[raw_name][rows].astype(np.float64)
                if kind == 'country_crop':
                    for k in np.unique(crop_idx):
                        sel = crop_idx == k
                        grid = self.tables.get('{}|{}'.format(raw_name, self.stratum(kind, country, CROPS[k])))
                        if grid is not None:
                            out[sel] = _rank(v[sel], grid)
                else:
                    grid = self.tables.get('{}|{}'.format(raw_name, country))
                    if grid is not None:
                        out = _rank(v, grid)
                rank_cache[key] = out
            return rank_cache[key]

        tmax = raw['tmax_mid_c'][rows].astype(np.float64)
        limit = np.array([NON_STRESS_TMAX_C[c] for c in crops]) if n else np.zeros(0)
        stressed = np.isfinite(tmax) & (tmax > limit)
        for i, cid in enumerate(self.concepts):
            raw_name, kind, how = ESTIMATORS[cid]
            if how == 'rank':
                t = ranked(raw_name, kind)
            elif how == 'one_minus_rank':
                t = 1.0 - ranked(raw_name, kind)
            elif cid == 'low_elevation_position':
                t = np.where(raw['field_relief'][rows] >= MIN_RELIEF_M, raw[raw_name][rows], np.nan)
            else:  # pole_facing_aspect
                p25 = self.slope_p25.get(country, np.nan)
                flat = ~(raw['slope'][rows] >= p25)
                t = np.where(flat, np.nan, np.maximum(0.0, raw[raw_name][rows]))
            if cid == 'warm_regime':
                t = np.where(stressed, 0.0, t)
            targets[:, i] = t

        gates = np.zeros((n, len(self.rules)), dtype=np.float32)
        rain_rank = ranked('precip_mid_mm', 'country_crop')
        for j, r in enumerate(self.rules):
            rid = r['id']
            if rid == 'ys_r01':
                g = np.isin(crops, RAINFED_SUMMER)
            elif rid == 'ys_r02':
                g = np.isfinite(tmax) & (tmax <= limit)
            elif rid == 'ys_r03':
                g = rain_rank < 0.5
            elif rid == 'ys_r04':
                g = raw['field_relief'][rows] >= MIN_RELIEF_M
            elif rid == 'ys_r05':
                g = np.zeros(n, bool)          # ERA5 cannot resolve within-field thermal regime
            elif rid == 'ys_r06':
                g = raw['soc030_gkg'][rows] < ORGANIC_SOC_GKG
            else:
                raise ValueError('no gate for rule {}'.format(rid))
            gates[:, j] = np.asarray(g, dtype=np.float32)
        return targets, gates

    def to_json(self):
        return {'format': REFERENCE_FORMAT, 'concepts': self.concepts, 'rules': self.rules,
                'tables': {k: v.tolist() for k, v in self.tables.items()},
                'slope_p25': self.slope_p25, 'meta': self.meta,
                'constants': {'non_stress_tmax_c': NON_STRESS_TMAX_C, 'organic_soc_gkg': ORGANIC_SOC_GKG,
                              'min_relief_m': MIN_RELIEF_M, 'rainfed_summer': list(RAINFED_SUMMER)}}

    @classmethod
    def from_json(cls, d):
        if d.get('format') != REFERENCE_FORMAT:
            raise ValueError('unexpected knowledge reference format')
        return cls({k: np.asarray(v) for k, v in d['tables'].items()}, d['slope_p25'], d['concepts'],
                   d['rules'], d.get('meta'))


def attach_knowledge(dataset, reference, artifact_root, raw_cache=None, cutoff_days=0):
    """Per-item concept targets and rule gates for a YieldSATPointDataset."""
    n = len(dataset)
    targets = np.full((n, len(reference.concepts)), np.nan, dtype=np.float16)
    gates = np.zeros((n, len(reference.rules)), dtype=np.float16)
    raw_cache = raw_cache if raw_cache is not None else {}
    for ci, country in enumerate(dataset.countries):
        sel = np.flatnonzero(dataset.country_of == ci)
        if not len(sel):
            continue
        if country not in raw_cache:
            raw_cache[country] = load_raw(artifact_root, country, cutoff_days)
        t, g = reference.transform(raw_cache[country], country, dataset.row[sel])
        targets[sel], gates[sel] = t, g
    dataset.knowledge = {'concept_target': targets, 'rule_gate': gates}
    return dataset.knowledge


def randomize_targets(dataset, seed=0):
    """The `random_targets` control: every field season receives the concept
    targets and rule gates of another field season of the same country x crop
    (a fixed derangement of seasons; rows are matched by position modulo the
    donor's length). Marginals, co-occurrence and rule applicability are kept;
    only the link between a season's inputs and its knowledge is broken."""
    k = dataset.knowledge
    ranges = np.asarray(dataset.season_ranges)
    strata = {}
    for i, s in enumerate(dataset.seasons):
        strata.setdefault((s['country'], s['crop']), []).append(i)
    src = np.arange(len(dataset))
    for key in sorted(strata):
        members = np.array(strata[key])
        if len(members) < 2:
            continue
        perm = derangement(len(members), seed)
        for a, b in zip(members, members[perm]):
            (s0, s1), (d0, d1) = ranges[a], ranges[b]
            src[s0:s1] = d0 + np.arange(s1 - s0) % (d1 - d0)
    dataset.knowledge = {name: arr[src] for name, arr in k.items()}
    return src


def derangement(n, seed=0):
    """A fixed permutation with no fixed point (the `shuffled` control)."""
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if n < 2 or not (p == np.arange(n)).any():
            return p


def season_mean(loss, weight, season):
    """Weighted mean per season, then mean over the seasons present."""
    keep = weight > 0
    if not keep.any():
        return None
    _, inv = torch.unique(season[keep], return_inverse=True)
    k = int(inv.max()) + 1
    num = torch.zeros(k, device=loss.device, dtype=loss.dtype).index_add(0, inv, loss[keep] * weight[keep])
    den = torch.zeros(k, device=loss.device, dtype=loss.dtype).index_add(0, inv, weight[keep])
    return (num / den.clamp_min(1e-12)).mean()


class KnowledgePointPretrainer(nn.Module):
    """SSL (masked observation + forecast) + concept grounding + relational
    distillation on the point model's per-stream summary embeddings.

    control: None | 'shuffled' | 'notext' | 'ssl_long' | 'random_targets'
    (ssl_long = SSL only, the entry point adds the extra steps;
    random_targets = real text, targets permuted by ``randomize_targets``).
    """

    def __init__(self, model, text_vectors, library, control=None, ground_weight=1.0,
                 relation_weight=1.0, tau=0.1, shuffle_seed=0, **ssl_kwargs):
        super().__init__()
        if control not in (None, 'shuffled', 'notext', 'ssl_long', 'random_targets'):
            raise ValueError('unknown control {}'.format(control))
        if model.config.get('encoder_output') != 'summary':
            raise ValueError('knowledge pretraining needs per-stream summary encoders')
        self.ssl = YieldSATPointPretrainer(model, **ssl_kwargs)
        self.model = model
        self.control = control
        self.ground_weight = 0.0 if control == 'ssl_long' else ground_weight
        self.relation_weight = 0.0 if control == 'ssl_long' else relation_weight
        self.tau = tau
        self.concepts = [c['id'] for c in library['concepts']]
        self.concept_stream = [STREAM_MAP[c['stream']] for c in library['concepts']]
        ci = {c: i for i, c in enumerate(self.concepts)}
        self.rule_ids = [r['id'] for r in library['rules']]
        self.pairs = [(ci[r['concept_a']], ci[r['concept_b']]) for r in library['rules']]
        self.min_presence = [float(r['minimum_presence']) for r in library['rules']]
        missing = sorted({s for s in self.concept_stream if s not in model.layout})
        if missing:
            raise ValueError('model lacks concept streams {}'.format(missing))
        text = torch.as_tensor(np.asarray(text_vectors), dtype=torch.float32)
        if text.shape[0] != len(self.concepts):
            raise ValueError('need one text vector per concept')
        self.permutation = None
        if control == 'shuffled':
            self.permutation = derangement(len(self.concepts), shuffle_seed)
            text = text[torch.as_tensor(self.permutation)]
        if control == 'notext':
            self.learned_prototypes = nn.Parameter(torch.randn_like(text) * 0.02)
        else:
            self.register_buffer('prototypes_fixed', F.normalize(text, dim=-1))
        D = model.config['embed_dim']
        self.projectors = nn.ModuleDict({s: nn.Linear(D, text.shape[1]) for s in sorted(set(self.concept_stream))})

    def prototypes(self):
        if self.control == 'notext':
            return F.normalize(self.learned_prototypes, dim=-1)
        return self.prototypes_fixed

    def relations(self, proto=None):
        proto = self.prototypes() if proto is None else proto
        delta = torch.stack([proto[b] - proto[a] for a, b in self.pairs])
        return F.normalize(delta, dim=-1)

    def stream_embeddings(self, batch):
        """Normalized projected summary embedding and availability per concept stream."""
        inputs = pack_inputs(batch, self.model.layout)
        available = {n: batch['available'][n] for n in self.model.streams}
        emb, mask = self.model.encoders.forward_with_missing(inputs, available=available, apply_dropout=False)
        z = {s: F.normalize(self.projectors[s](emb[s][:, 0].float()), dim=-1) for s in self.projectors}
        avail = {s: mask[s].float() > 0.5 for s in self.projectors}
        return z, avail

    def knowledge_losses(self, batch):
        target = batch['concept_target'].float()
        gate = batch['rule_gate'].float()
        season = batch['season']
        z, avail = self.stream_embeddings(batch)
        proto = self.prototypes()
        ground, n_ground = [], 0
        for i, s in enumerate(self.concept_stream):
            t = target[:, i]
            known = torch.isfinite(t) & avail[s]
            if not known.any():
                continue
            logit = (z[s] * proto[i]).sum(-1) / self.tau
            l = F.binary_cross_entropy_with_logits(logit, torch.nan_to_num(t), reduction='none')
            m = season_mean(l, known.float(), season)
            if m is not None:
                ground.append(m)
                n_ground += int(known.sum())
        rel_dir = self.relations(proto)
        relation, n_rel = [], 0
        for j, (a, b) in enumerate(self.pairs):
            pa, pb, g = target[:, a], target[:, b], gate[:, j]
            sa, sb = self.concept_stream[a], self.concept_stream[b]
            ok = (torch.isfinite(pa) & torch.isfinite(pb) & (pa >= self.min_presence[j])
                  & (pb >= self.min_presence[j]) & (g > 0) & avail[sa] & avail[sb])
            if not ok.any():
                continue
            delta = F.normalize(z[sb] - z[sa], dim=-1)
            l = 1.0 - (delta * rel_dir[j]).sum(-1)
            w = torch.where(ok, torch.nan_to_num(pa) * torch.nan_to_num(pb) * g, torch.zeros_like(pa)).detach()
            m = season_mean(l, w, season)
            if m is not None:
                relation.append(m)
                n_rel += int(ok.sum())
        zero = sum(p.sum() * 0 for p in self.projectors.parameters())
        lg = torch.stack(ground).mean() if ground else zero
        lr = torch.stack(relation).mean() if relation else zero
        return lg, lr, {'ground_terms': n_ground, 'relation_terms': n_rel}

    def forward(self, batch):
        total, losses = self.ssl(batch)
        losses = {k: float(v.detach()) for k, v in losses.items()}
        losses['ssl'] = float(total.detach())
        if self.ground_weight or self.relation_weight:
            lg, lr, counts = self.knowledge_losses(batch)
            total = total + self.ground_weight * lg + self.relation_weight * lr
            losses.update(ground=float(lg.detach()), relation=float(lr.detach()), **counts)
        return total, losses


def load_text_vectors(text_dir, library, allow_mock=False):
    """Frozen concept text vectors in library concept order, verified against
    the library hash and the vectors file hash (spec PK-04)."""
    from yieldsat_knowledge.common import digest, file_hash
    d = Path(text_dir)
    man = json.loads((d / 'manifest.json').read_text())
    if man['library_hash'] != digest(library):
        raise ValueError('text vectors were built for a different library')
    if man['vectors_hash'] != file_hash(d / 'vectors.npz'):
        raise ValueError('vectors.npz does not match its manifest hash')
    if man.get('mock') and not allow_mock:
        raise ValueError('mock text vectors are for smoke tests only')
    z = np.load(d / 'vectors.npz')
    index = {str(k): i for i, k in enumerate(z['ids'])}
    vec = np.stack([z['vectors'][index[c['id']]] for c in library['concepts']])
    return vec.astype(np.float32), man
