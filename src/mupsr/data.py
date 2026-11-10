"""Data pipeline for Cosmetics (purchase events only).

Layers, each consuming the previous one:
  PreparedData       memory-mapped prepared cache (one row per purchase; integrity-checked)
  CandidateProtocol  frozen evaluation protocol: per-user leave-last-out queries, and each day's
                     candidate pool = items with a past TRAIN price within max_price_age_days
  EventData          event-level inputs: timestamps, log prices, category reference prices (Eq. 26),
                     user / category row indices; cached once per data fingerprint
  build_day_tables   day-level market tables of stage 1 (Sec. 3.6), information set d^-
  stage-2 batching   (user, block) samples and the encoder inputs of training / evaluation queries,
                     each query carrying its prediction time and its user-item experience entries
"""
import csv
import gzip
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .utils import digest_file, write_json

SPLITS = ('train', 'valid', 'test')


# ---------------------------------------------------------------- prepared cache
class PreparedData:
    """Read-only prepared cache (adapter `processed_rows_v1`); READY must match metadata.json."""
    REQUIRED = ('query_day', 'query_split', 'day_time', 'day_user', 'day_order', 'event_item',
                'item_category', 'mean', 'scale', 'category_interpretable',
                'price_snapshot_z', 'price_snapshot_day')

    def __init__(self, root):
        self.root = Path(root)
        if not (self.root / 'READY').exists():
            raise ValueError('Incomplete prepared data: READY is absent.')
        self.fingerprint = digest_file(self.root / 'metadata.json')
        if (self.root / 'READY').read_text().strip() != self.fingerprint:
            raise ValueError('Prepared metadata was modified after preparation.')
        self.meta = json.loads((self.root / 'metadata.json').read_text(encoding='utf-8'))
        if self.meta.get('adapter') != 'processed_rows_v1' or self.meta['config']['dataset'] != 'cosmetics':
            raise ValueError('Expected the Cosmetics processed-row cache.')
        for name, info in self.meta.get('artifacts', {}).items():
            path = self.root / name
            if not path.is_file() or path.stat().st_size != info['bytes']:
                raise ValueError(f'Prepared artifact missing or truncated: {name}')
        self.arr = {p.stem: np.load(p, mmap_mode='r') for p in self.root.glob('*.npy')}
        self.cfg = self.meta['config']
        missing = set(self.REQUIRED) - self.arr.keys()
        if missing:
            raise ValueError(f'Incomplete prepared arrays: {sorted(missing)}')
        a = self.arr
        if len(a['query_day']) == 0 or a['query_day'].min() < 0 or a['query_day'].max() >= len(a['day_time']):
            raise ValueError('Prepared query day indices are invalid.')
        if len(a['mean']) != len(self.meta['category_ids']) or len(a['item_category']) != len(self.meta['article_ids']):
            raise ValueError('Prepared mappings disagree with model dimensions.')
        if not np.all(np.isfinite(a['scale']) & (a['scale'] > 0)):
            raise ValueError('Prepared price scales must be finite and positive.')

    def indices(self, split):
        return np.flatnonzero(self.arr['query_split'] == split)


# ---------------------------------------------------------------- frozen candidate protocol
class CandidateProtocol:
    """One price-eligible candidate protocol shared by all models (fingerprinted)."""

    def __init__(self, path, max_price_age_days=None):
        self.base = PreparedData(path)
        self.dataset = self.base.cfg['dataset']
        self.max_age = int(max_price_age_days or self.base.cfg['max_price_age_days'])
        if self.max_age < 1:
            raise ValueError('max_price_age_days must be positive.')
        self.n_items = len(self.base.arr['item_category']) - 1
        self.meta = {'scope': 'full', 'dataset': self.dataset,
                     'protocol': 'cosmetics_last_row_single_positive',
                     'candidate_policy': 'all_categories_past_train_price_only',
                     'max_price_age_days': self.max_age}
        digest = {'version': 1, 'base': self.base.fingerprint, 'source': self.base.fingerprint, **self.meta}
        self.fingerprint = hashlib.sha256(json.dumps(digest, sort_keys=True).encode()).hexdigest()
        self.query_ids = {}
        for split in SPLITS:
            selected = self.base.indices(SPLITS.index(split)).astype(np.int32)
            if not len(selected):
                raise ValueError(f'{split} has no queries with valid target prices.')
            self.query_ids[split] = selected
            self.meta[f'{split}_queries'] = len(selected)
            self.meta[f'{split}_raw_queries'] = self.base.meta['audit'][f'rows_{split}']

    def prices(self, days, items):
        """Standardised price snapshot of `items` on `days` (past TRAIN observations only) and absence mask."""
        days = np.asarray(days)
        items = np.asarray(items)
        a = self.base.arr
        if items.ndim == 1 and days.ndim == 1 and len(days) != len(items):
            items = np.broadcast_to(items, (len(days), len(items)))
            days = days[:, None]
        elif items.ndim == 2:
            days = days[:, None]
        observed = a['price_snapshot_day'][days, items]
        z = a['price_snapshot_z'][days, items]
        if np.any(observed >= days):
            raise ValueError('Current/future price entered a query.')
        absent = ((items == 0) | (observed < 0) | (days - observed > self.max_age)
                  | ~np.isfinite(z)
                  | ~a['category_interpretable'][a['item_category'][items]])
        return np.where(absent, 0, z).astype(np.float32), absent

    def eligible_pool(self, day):
        item = np.arange(1, self.n_items + 1, dtype=np.int32)
        _, absent = self.prices(np.asarray([int(day)]), item[None, :])
        return item[~absent[0]]


# ---------------------------------------------------------------- event-level inputs
class EventData:
    def __init__(self, legacy_path, source, cache, half_life=5.0):
        self.protocol = CandidateProtocol(legacy_path)
        self.a = self.protocol.base.arr
        self.item = np.asarray(self.a['event_item'])
        self.cat = np.asarray(self.a['item_category'])
        self.categories = len(self.a['mean'])
        self.n = len(self.item)
        self.mean = np.asarray(self.a['mean'])
        self.scale = np.asarray(self.a['scale'])
        self.source, self.cache = Path(source), Path(cache)
        identity = {'version': 1, 'base': self.protocol.base.fingerprint,
                    'sources': self.protocol.base.meta['sources'], 'half_life_days': half_life}
        ready = self.cache / 'metadata.json'
        if ready.exists():
            meta = json.loads(ready.read_text())
            if meta['identity'] != identity:
                raise ValueError('Event cache identity mismatch.')
            for name, info in meta['arrays'].items():
                if (self.cache / name).stat().st_size != info['bytes']:
                    raise ValueError('Truncated event cache: ' + name)
        else:
            self.cache.mkdir(parents=True, exist_ok=True)
            self._prepare(identity, half_life)
        self.meta = json.loads(ready.read_text())
        self.fingerprint = digest_file(ready)
        self.events = {p.stem: np.load(p, mmap_mode='r') for p in self.cache.glob('*.npy')}
        self.time = self.events['time']
        self.train_rows = self.events['train_rows']
        users = np.asarray(self.a['day_user'])
        ordering = np.argsort(users, kind='stable')
        cuts = np.r_[0, np.flatnonzero(np.diff(users[ordering])) + 1, len(users)]
        self.user_rows = {int(users[ordering[l]]): ordering[l:r] for l, r in zip(cuts[:-1], cuts[1:])}
        self.query_rows = {s: np.asarray(self.a['query_day'][self.protocol.query_ids[s]]) for s in SPLITS}

    def _prepare(self, identity, half_life):
        order = np.asarray(self.a['day_order'])
        if np.any(np.diff(order) <= 0):
            raise ValueError('Global source order must be strictly increasing.')
        timestamps = np.empty(self.n, np.float64)
        split = np.full(self.n, -1, np.int8)
        logp = np.zeros(self.n, np.float32)
        missing = np.ones(self.n, np.bool_)
        origin = datetime.fromisoformat(self.protocol.base.meta['origin_date'])
        for si, name in enumerate(SPLITS):
            path = self.source / (name + '.csv.gz')
            if digest_file(path) != identity['sources'][path.name]['sha256']:
                raise ValueError('CSV differs from frozen ranking input: ' + str(path))
            with gzip.open(path, 'rt', encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    so = int(row['source_order'])
                    j = np.searchsorted(order, so)
                    if j >= self.n or order[j] != so or split[j] != -1:
                        raise ValueError('Missing or duplicate source row.')
                    if self.item[j] != int(row['item_idx']) or self.a['day_user'][j] != int(row['user_idx']):
                        raise ValueError('Row alignment mismatch.')
                    split[j] = si
                    timestamps[j] = (datetime.fromisoformat(row['event_time']) - origin).total_seconds()/86400
                    p = float(row['price']) if row['price_valid'] == '1' and row['price'] else float('nan')
                    if math.isfinite(p) and p > 0:
                        logp[j], missing[j] = math.log(p), False
                    else:
                        logp[j] = self.mean[self.cat[self.item[j]]]
            print('event inputs', name, int(np.count_nonzero(split == si)), flush=True)
        if np.any(split < 0) or np.any(np.diff(timestamps) < 0):
            raise ValueError('Incomplete events or source-order timestamp inversion.')
        train = np.flatnonzero(split == 0)
        valid = train[~missing[train]]
        sums = np.bincount(self.item[valid], weights=logp[valid], minlength=len(self.cat))
        counts = np.bincount(self.item[valid], minlength=len(self.cat))
        sku_log_prices = sums[counts > 0] / counts[counts > 0]
        global_mean, global_scale = float(sku_log_prices.mean()), max(float(sku_log_prices.std()), 1e-4)
        z = (logp[valid] - self.mean[self.cat[self.item[valid]]]) / self.scale[self.cat[self.item[valid]]]
        ell = (logp[valid] - global_mean) / global_scale
        knots = []
        for value in (z, ell):
            k = np.unique(np.quantile(value, np.linspace(0, 1, 17))).astype(np.float32)
            if len(k) < 2:
                k = np.array([value[0]-1, value[0]+1], np.float32)
            knots.append(k)
        # Category reference price, Eq. (26): exponentially decayed mean of past train log prices.
        delta = np.zeros(self.n, np.float32)
        reference = np.zeros(self.n, np.float32)
        last = np.full(self.categories, np.nan)
        numerator, denominator = np.zeros(self.categories), np.zeros(self.categories)
        for j in train:
            c, t = self.cat[self.item[j]], timestamps[j]
            gap = 0.0 if np.isnan(last[c]) else t-last[c]
            delta[j] = gap
            decay = math.exp(-math.log(2)*gap/half_life)
            numerator[c] *= decay
            denominator[c] *= decay
            if not missing[j]:
                numerator[c] += float(logp[j])
                denominator[c] += 1
            reference[j] = ((numerator[c]/denominator[c] - self.mean[c])/self.scale[c]
                            if denominator[c] > 0 else 0.0)
            last[c] = t
        target_days = np.asarray(self.a['day_time'][train])
        _, absent = self.protocol.prices(target_days, self.item[train])
        arrays = dict(time=timestamps, split=split, logp=logp, missing=missing,
                      train_rows=train, cat_delta=delta, reference=reference,
                      stage1_eligible=~absent, knots_z=knots[0], knots_ell=knots[1])
        for name, value in arrays.items():
            temp = self.cache / (name + '.tmp.npy')
            np.save(temp, value)
            os.replace(temp, self.cache / (name + '.npy'))
        write_json(self.cache/'metadata.json', {'identity': identity, 'global_mean': global_mean,
                   'global_scale': global_scale, 'global_statistics': 'equal-weight train SKU mean log prices',
                   'stage1_records': len(train), 'stage1_eligible_labels': int((~absent).sum()),
                   'stage1_excluded_unavailable_price_labels': int(absent.sum()),
                   'missing_prices': int(missing.sum()),
                   'candidate_fingerprint': self.protocol.fingerprint,
                   'arrays': {p.name: {'bytes': p.stat().st_size, 'sha256': digest_file(p)}
                              for p in self.cache.glob('*.npy')}})

    def event_features(self, rows, device):
        rows = np.asarray(rows)
        return {'items': torch.tensor(self.item[rows].astype(np.int64), device=device),
                'logp': torch.tensor(np.array(self.events['logp'][rows]), device=device),
                'missing': torch.tensor(np.array(self.events['missing'][rows]), device=device),
                'time': torch.tensor(np.array(self.time[rows]), dtype=torch.float32, device=device)}

    def prefix(self, query_row):
        """The query user's earlier events (all splits), in source order."""
        rows = self.user_rows[int(self.a['day_user'][query_row])]
        return rows[:np.searchsorted(rows, query_row)]

    def sequences(self, row_lists, device):
        """Right-padded stage-2 event features for several row sequences."""
        lengths = np.array([len(x) for x in row_lists], np.int64)
        width = max(1, int(lengths.max(initial=0)))
        ids = np.zeros((len(row_lists), width), np.int64)
        for b, seq in enumerate(row_lists):
            ids[b, :len(seq)] = seq
        features = self.event_features(ids, device)
        mask = torch.as_tensor(np.arange(width)[None] < lengths[:, None], device=device)
        features['items'] = features['items'].masked_fill(~mask, 0)
        features['mask'] = mask
        dt = torch.zeros_like(features['time'])
        dt[:, 1:] = (features['time'][:, 1:]-features['time'][:, :-1]).clamp_min(0)
        features['delta'] = dt.masked_fill(~mask, 0)
        return features


# ---------------------------------------------------------------- stage-1 day tables
def build_day_tables(data):
    """Day-level market tables (Sec. 3.6). Information set for every market quantity on day d:
    TRAIN records with day < d, i.e. exactly the information set of the candidate price snapshot."""
    train = np.asarray(data.train_rows)
    day = np.asarray(data.a['day_time'])
    time = np.asarray(data.time)
    cat = data.cat[data.item[train]]
    C = data.categories
    T = int(day[train].max()) + 1                        # training days 0..T-1
    D = int(data.a['price_snapshot_day'].shape[0])       # all protocol days
    if np.any(np.floor(time[train]) != day[train]):
        raise ValueError('time and day_time use different origins; set day_start explicitly.')
    count = np.bincount(cat*T + day[train], minlength=C*T).astype(np.float32)
    first = np.full(C, T, np.int64)
    np.minimum.at(first, cat, day[train])
    active = np.arange(T)[None] >= first[:, None]        # (C, T): token exists from first purchase day
    src = np.minimum(np.arange(D), T) - 1                 # state of day d = panel column d-1 (frozen after T)
    lag = np.zeros((D, C), np.float32)
    ref = np.zeros((D, C), np.float32)
    starts = np.arange(D, dtype=np.float64)               # day d starts at time d
    reference = np.asarray(data.events['reference'])
    for c in range(C):
        rows = train[cat == c]
        if not len(rows):
            continue
        pos = np.searchsorted(time[rows], starts, side='left') - 1
        keep = pos >= 0
        last = rows[pos[keep]]
        lag[keep, c] = np.log1p(starts[keep] - time[last])
        ref[keep, c] = reference[last]
    pool = np.zeros((D, len(data.cat)), bool)            # eligible candidates of each day (frozen protocol)
    for d in range(D):
        pool[d, data.protocol.eligible_pool(d)] = True
    eligible = np.asarray(data.events['stage1_eligible'], bool)
    key = day[train][eligible].astype(np.int64)*(len(data.cat)) + data.item[train][eligible]
    cells, y = np.unique(key, return_counts=True)
    labels = np.stack([cells // len(data.cat), cells % len(data.cat)], 1)   # (nnz, 2): (day, item)
    return dict(T=T, D=D, cell=(cat*T + day[train]).astype(np.int64), count=count, active=active,
                src=src.astype(np.int64), lag=lag, ref=ref, eligible=pool, label_index=labels.astype(np.int64),
                label_count=y.astype(np.float32))


def eligible_training_days(tables):
    """Days with at least one stage-1 label (the real Cosmetics data has none on day zero)."""
    counts = np.bincount(tables['label_index'][:, 0], weights=tables['label_count'], minlength=tables['T'])
    return np.flatnonzero(counts > 0).astype(np.int64)


# ---------------------------------------------------------------- stage-2 samples and batches
def user_gaps(data):
    """Time since the same user's previous row (all splits): what a full-sequence encoding sees."""
    gap = np.zeros(data.n, np.float32)
    for rows in data.user_rows.values():
        gap[rows[1:]] = np.diff(data.time[rows])
    return gap


def block_samples(data, block, is_train_query):
    """Each sample = block b of one user's train sequence; its readable context is blocks b-1 and b."""
    split = np.asarray(data.events['split'])
    samples = []
    for u, rows in data.user_rows.items():
        rows = rows[split[rows] == 0]
        for b in range(0, len(rows), block):
            if is_train_query[rows[b:b+block]].any():
                samples.append((u, b//block))
    return np.array(samples, np.int64)


def pack_batches(data, samples, block, token_budget, rng):
    """Bucket samples by current-block length (little padding), shuffle batch order each epoch."""
    split = np.asarray(data.events['split'])
    length = np.array([min(block, int((split[data.user_rows[int(u)]] == 0).sum())-b*block) for u, b in samples])
    batches, current, used = [], [], 0
    for i in np.lexsort((rng.random(len(samples)), length)):
        if current and used+length[i] > token_budget:
            batches.append(samples[current])
            current, used = [], 0
        current.append(i)
        used += length[i]
    if current:
        batches.append(samples[current])
    return [batches[i] for i in rng.permutation(len(batches))]


def make_batch(data, batch, block, is_train_query, gap, rep_index, device):
    """Encoder inputs for a batch of (user, block) samples and the indices of its loss queries, with each
    query's prediction time and experience entries."""
    split = np.asarray(data.events['split'])
    prev, cur, qs, qp, qrow = [], [], [], [], []
    for s, (u, b) in enumerate(batch):
        rows = data.user_rows[int(u)]
        rows = rows[split[rows] == 0]
        prev.append(rows[max(0, (b-1)*block):b*block])
        cur.append(rows[b*block:(b+1)*block])
        p = np.flatnonzero(is_train_query[cur[-1]])
        qs += [s]*len(p); qp += p.tolist(); qrow += cur[-1][p].tolist()
    return _tail(data, prev, cur, qs, qp, qrow, gap, rep_index, device)


def make_eval_batch(data, rows_q, block, gap, rep_index, device):
    """A held-out query at position p reads exactly what a training query at position p reads."""
    prev, cur = [], []
    for q in rows_q:
        h = data.prefix(int(q))
        b = len(h)//block
        prev.append(h[max(0, (b-1)*block):b*block])
        cur.append(h[b*block:])
    return _tail(data, prev, cur, list(range(len(rows_q))), [len(c) for c in cur], list(rows_q), gap, rep_index, device)


def _tail(data, prev, cur, qs, qp, qrow, gap, rep_index, device):
    fp, fc = _features(data, prev, cur, gap, device)
    qs_t, qp_t, day, target = _query_index(data, qs, qp, qrow, device)
    qrow = np.asarray(qrow, np.int64)
    qtime = torch.as_tensor(np.asarray(data.time[qrow], np.float32), device=device)
    rq, it, g, c, lp, lm = rep_index.gather(qrow)
    rep = tuple(torch.as_tensor(x, device=device) for x in (rq, it, g, c, lp, ~lm))
    return fp, fc, qs_t, qp_t, day, target, qtime, rep, qrow


def _features(data, prev, cur, gap, device):
    out = []
    for lists in (prev, cur):
        f = data.sequences(lists, device)
        width = f['items'].shape[1]
        ids = np.zeros((len(lists), width), np.int64)
        for i, r in enumerate(lists):
            ids[i, :len(r)] = r
        f['delta'] = torch.as_tensor(gap[ids], device=device).masked_fill(~f['mask'], 0)
        out.append(f)
    return tuple(out)


def _query_index(data, qs, qp, qrow, device):
    qrow = np.asarray(qrow, np.int64)
    as_t = lambda x: torch.as_tensor(np.asarray(x, np.int64), device=device)
    return (as_t(qs), as_t(qp), as_t(np.asarray(data.a['day_time'])[qrow]), as_t(data.item[qrow]))
