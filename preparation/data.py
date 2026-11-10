"""Disk-backed H&M preparation. No fabricated ordering within a purchase day."""
import csv
import hashlib
import heapq
import json
import math
import os
import sqlite3
from array import array
from collections import Counter, defaultdict
from datetime import date
from itertools import groupby
from pathlib import Path

import numpy as np


def ordinal(value):
    return date.fromisoformat(value).toordinal()


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


class ArrayWriter:
    """Append binary rows; finalize into mmap-friendly .npy without retaining Q*M RAM."""
    def __init__(self, root, name, dtype, width=None):
        self.path = Path(root) / (name + '.bin')
        self.name, self.dtype, self.width = name, np.dtype(dtype), width
        self.file = self.path.open('wb')
        self.count = 0

    def add(self, values):
        x = np.asarray(values, dtype=self.dtype)
        self.file.write(x.tobytes())
        self.count += x.size

    def finish(self):
        self.file.close()
        shape = (self.count,) if self.width is None else (self.count // self.width, self.width)
        target = np.lib.format.open_memmap(self.path.with_suffix('.npy'), mode='w+',
                                          dtype=self.dtype, shape=shape)
        if self.count:
            source = np.memmap(self.path, mode='r', dtype=self.dtype)
            flat = target.reshape(-1)
            for j in range(0, self.count, 4_000_000):
                flat[j:j + 4_000_000] = source[j:j + 4_000_000]
            del source, flat
        target.flush()
        del target
        self.path.unlink()


def prepare(raw_dir, output, cfg):
    root, raw_dir = Path(output), Path(raw_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Preparation output must be empty; use a new directory.')
    root.mkdir(parents=True, exist_ok=True)
    start, valid, test, end = [ordinal(cfg[k]) for k in ('start', 'valid_start', 'test_start', 'end')]
    if not start < valid < test < end:
        raise ValueError('Require start < valid_start < test_start < end.')
    for key in ('user_mod', 'min_item_count', 'min_history_days', 'history_days',
                'negatives', 'max_price_age_days', 'block_days', 'support_users'):
        if cfg[key] < 1:
            raise ValueError(f'{key} must be positive')
    if cfg['min_history_days'] > cfg['history_days']:
        raise ValueError('min_history_days > history_days')
    article_file = raw_dir / 'articles.csv'
    transaction_file = raw_dir / 'transactions_train.csv'
    article_categories = {}
    with article_file.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            article_categories[row['article_id']] = row[cfg['category_field']]
    audit = Counter()
    db = sqlite3.connect(root / 'preparation.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('PRAGMA cache_size=-131072')
    db.execute('CREATE TABLE raw(u TEXT,d INTEGER,i TEXT,p REAL)')
    pending = []
    with transaction_file.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            audit['rows_scanned'] += 1
            if audit['rows_scanned'] % 2_000_000 == 0:
                print(f"scanned={audit['rows_scanned']:,} retained={audit['rows_in_selected_users_and_dates']:,}", flush=True)
            if not cfg['start'] <= row['t_dat'] < cfg['end']:
                continue
            if int(row['customer_id'][-16:], 16) % cfg['user_mod']:
                continue
            audit['rows_in_selected_users_and_dates'] += 1
            if row['article_id'] not in article_categories:
                audit['unknown_article_rows'] += 1
                continue
            price = float(row['price'])
            if not math.isfinite(price) or price <= 0:
                audit['invalid_price_rows'] += 1
                continue
            pending.append((row['customer_id'], ordinal(row['t_dat']) - start,
                            row['article_id'], math.log(price)))
            if len(pending) >= 50000:
                db.executemany('INSERT INTO raw VALUES (?,?,?,?)', pending)
                db.commit()
                pending.clear()
    db.executemany('INSERT INTO raw VALUES (?,?,?,?)', pending)
    db.commit()
    print('Building training vocabulary and daily tables...', flush=True)
    train_limit = valid - start
    ids = [r[0] for r in db.execute(
        'SELECT i FROM raw WHERE d<? GROUP BY i HAVING COUNT(*)>=? ORDER BY i',
        (train_limit, cfg['min_item_count']))]
    if not ids:
        raise ValueError('No training items after filtering.')
    categories = sorted({article_categories[i] for i in ids})
    cat_map = {c: j + 1 for j, c in enumerate(categories)}
    item_cat = np.asarray([0] + [cat_map[article_categories[i]] for i in ids], np.int32)
    C, I, T = len(categories) + 1, len(ids) + 1, end - start
    db.execute('CREATE TABLE vocab(i TEXT PRIMARY KEY, idx INTEGER)')
    db.executemany('INSERT INTO vocab VALUES (?,?)', ((i, j + 1) for j, i in enumerate(ids)))
    # Preserve original distinct-item cardinality before vocabulary filtering.
    db.execute('CREATE TABLE cardinality AS SELECT u,d,COUNT(DISTINCT i) n FROM raw GROUP BY u,d')
    db.execute('CREATE UNIQUE INDEX card_ud ON cardinality(u,d)')
    # Unique prices: duplicate transaction quantities cannot reweight SKU-day price.
    db.execute('CREATE TABLE daily_price AS SELECT d,idx i,AVG(p) p FROM '
               '(SELECT DISTINCT d,idx,p FROM raw JOIN vocab USING(i)) GROUP BY d,idx')
    db.execute('CREATE INDEX price_di ON daily_price(d,i)')
    # Historical bag: each SKU occurs once; duplicate quantities are not sequence steps.
    db.execute('CREATE TABLE events AS SELECT u,d,idx i,AVG(p) p FROM '
               '(SELECT DISTINCT u,d,idx,p FROM raw JOIN vocab USING(i)) GROUP BY u,d,idx')
    db.execute('CREATE INDEX events_dui ON events(d,u,i)')
    db.commit()
    sums, squares, counts = np.zeros(C), np.zeros(C), np.zeros(C, np.int64)
    for i, p in db.execute('SELECT i,p FROM daily_price WHERE d<?', (train_limit,)):
        c = item_cat[i]
        sums[c] += p
        squares[c] += p * p
        counts[c] += 1
    means = sums / np.maximum(counts, 1)
    std = np.sqrt(np.maximum(0, squares / np.maximum(counts, 1) - means ** 2))
    valid_cat = (counts >= 2) & (std >= 1e-4)
    scale = np.where(valid_cat, std, 1.0)
    daily = np.full((T, C), np.nan, np.float32)
    candidates_by_cat = [np.flatnonzero(item_cat == c) for c in range(C)]
    rng = np.random.default_rng(cfg['seed'])
    latest, latest_day = np.full(I, np.nan), np.full(I, -1, np.int32)
    # History is CSR over purchase days; user IDs have no learned embedding.
    day_time, day_prev, day_user = array('i'), array('q'), array('q')
    bag_ptr = array('q', [0])
    user_map, last_day, hist_count = {}, {}, Counter()
    support_heaps = [[] for _ in range(C)]
    support_members = [set() for _ in range(C)]
    support_snapshots, block_starts = [], []
    S = cfg['support_users']
    events_item = ArrayWriter(root, 'event_item', np.int32)
    events_price = ArrayWriter(root, 'event_price', np.float32)
    names = {'query_day': (np.int64, None), 'query_block': (np.int32, None),
             'query_split': (np.int8, None), 'candidate_item': (np.int32, cfg['negatives'] + 1),
             'candidate_z': (np.float32, cfg['negatives'] + 1),
             'candidate_price_day': (np.int32, cfg['negatives'] + 1)}
    writers = {name: ArrayWriter(root, name, *spec) for name, spec in names.items()}

    def snapshot(d):
        a = np.full((C, S), -1, np.int64)
        for c in range(1, C):
            selected = sorted(support_members[c])
            for j, uid in enumerate(selected):
                a[c, j] = last_day[uid]
        support_snapshots.append(a)
        block_starts.append(d)

    # A training-end snapshot is distinct even if cutoff is not a block boundary.
    block = -1
    export_block = None
    for d in range(T):
        if d < train_limit and d % cfg['block_days'] == 0:
            snapshot(d)
            block += 1
        if d == train_limit:
            snapshot(d)
            export_block = len(support_snapshots) - 1
        pools = []
        for c, items in enumerate(candidates_by_cat):
            mask = ((latest_day[items] >= 0) & (d - latest_day[items] <= cfg['max_price_age_days']))
            pools.append(items[mask])
        rows = db.execute('SELECT e.u,e.i,e.p,k.n FROM events e JOIN cardinality k '
                          'ON e.u=k.u AND e.d=k.d WHERE e.d=? ORDER BY e.u,e.i', (d,))
        for user, group in groupby(rows, key=lambda r: r[0]):
            records = list(group)
            uid = user_map.setdefault(user, len(user_map))
            idx = len(day_time)
            prev = last_day.get(uid, -1)
            audit['customer_days_with_known_items'] += 1
            if records[0][3] != 1:
                audit['multi_item_days_history_only'] += 1
            else:
                i = records[0][1]
                c = item_cat[i]
                audit['single_item_day_queries_considered'] += 1
                reason = None
                if hist_count[uid] < cfg['min_history_days']:
                    reason = 'query_skip_short_history'
                elif not valid_cat[c]:
                    reason = 'query_skip_no_category_variation'
                elif latest_day[i] < 0 or d - latest_day[i] > cfg['max_price_age_days']:
                    reason = 'query_skip_target_price_unavailable'
                elif not np.any(np.isfinite(daily[:d, c])):
                    reason = 'query_skip_reference_unavailable'
                elif len(pools[c]) < cfg['negatives'] + 1:
                    reason = 'query_skip_small_candidate_pool'
                if reason:
                    audit[reason] += 1
                else:
                    pool = pools[c][pools[c] != i]
                    neg = rng.choice(pool, cfg['negatives'], replace=False)
                    cand = np.concatenate(([i], neg))  # positive at column 0; tie-aware evaluator
                    split = 0 if d < train_limit else 1 if d < test - start else 2
                    writers['query_day'].add([idx])
                    writers['query_block'].add([block if split == 0 else export_block])
                    writers['query_split'].add([split])
                    writers['candidate_item'].add(cand)
                    writers['candidate_z'].add((latest[cand] - means[c]) / scale[c])
                    writers['candidate_price_day'].add(latest_day[cand])
                    audit[f'queries_{("train", "valid", "test")[split]}'] += 1
                    audit['candidate_age_sum_days'] += int((d - latest_day[cand]).sum())
                    audit['candidate_observations'] += len(cand)
            # This day's unordered history bag becomes available only on subsequent days.
            day_time.append(d)
            day_prev.append(prev)
            day_user.append(uid)
            item_ids = [r[1] for r in records]
            prices = [r[2] for r in records]
            events_item.add(item_ids)
            events_price.add(prices)
            bag_ptr.append(bag_ptr[-1] + len(item_ids))
            last_day[uid] = idx
            hist_count[uid] += 1
            if d < train_limit:
                for c in set(int(item_cat[i]) for i in item_ids):
                    if uid in support_members[c]:
                        continue
                    priority = int.from_bytes(hashlib.blake2b(
                        f'{cfg["seed"]}:{user}:{c}'.encode(), digest_size=8).digest(), 'big')
                    heap = support_heaps[c]
                    if len(heap) < S:
                        heapq.heappush(heap, (-priority, uid))
                        support_members[c].add(uid)
                    elif priority < -heap[0][0]:
                        _, removed = heapq.heapreplace(heap, (-priority, uid))
                        support_members[c].remove(removed)
                        support_members[c].add(uid)
        # All queries on day d precede all price updates on day d.
        totals, n = np.zeros(C), np.zeros(C, np.int32)
        for i, p in db.execute('SELECT i,p FROM daily_price WHERE d=?', (d,)):
            c = item_cat[i]
            latest[i], latest_day[i] = p, d
            totals[c] += p
            n[c] += 1
        observed = n > 0
        daily[d, observed] = totals[observed] / n[observed]
        if d % 30 == 0:
            print(f'day={d}/{T} train_queries={audit["queries_train"]:,}', flush=True)
    for writer in [events_item, events_price, *writers.values()]:
        writer.finish()
    arrays = {'day_time': np.asarray(day_time, np.int32), 'day_prev': np.asarray(day_prev, np.int64),
              'day_user': np.asarray(day_user, np.int64), 'bag_ptr': np.asarray(bag_ptr, np.int64),
              'item_category': item_cat, 'mean': means.astype(np.float32), 'scale': scale.astype(np.float32),
              'category_interpretable': valid_cat, 'daily_log_price': daily,
              'support': np.stack(support_snapshots), 'block_start': np.asarray(block_starts, np.int32)}
    for name, values in arrays.items():
        np.save(root / (name + '.npy'), values)
    audit['users'] = len(user_map)
    audit['items'] = len(ids)
    audit['categories'] = len(categories)
    audit['categories_without_variation'] = int((~valid_cat[1:]).sum())
    audit['support_slots_missing'] = int((arrays['support'] < 0).sum())
    sources = {p.name: {'bytes': p.stat().st_size, 'sha256': digest_file(p)}
               for p in (article_file, transaction_file)}
    metadata = {'format_version': 1, 'config': cfg, 'audit': dict(audit), 'sources': sources,
                'article_ids': ['<PAD>'] + ids, 'category_ids': ['<PAD>'] + categories,
                'export_block': export_block, 'price_semantics': 'past SKU-day mean of distinct log transaction prices',
                'task': 'given-category single-distinct-item purchase-day ranking; no intraday next-item claim'}
    write_json(root / 'metadata.json', metadata)
    db.close()
    if min(audit[f'queries_{s}'] for s in ('train', 'valid', 'test')) == 0:
        raise ValueError('At least one split has no queries; inspect metadata.json and adjust configuration.')
    # READY appears only after successful completion. Input fingerprints travel with cache.
    (root / 'READY').write_text(digest_file(root / 'metadata.json'), encoding='ascii')
    print(json.dumps(dict(audit), indent=2), flush=True)


class PreparedData:
    def __init__(self, root):
        self.root = Path(root)
        if not (self.root / 'READY').exists():
            raise ValueError('Incomplete prepared data: READY is absent.')
        self.fingerprint = digest_file(self.root / 'metadata.json')
        if (self.root / 'READY').read_text().strip() != self.fingerprint:
            raise ValueError('Prepared metadata was modified after preparation.')
        self.meta = json.loads((self.root / 'metadata.json').read_text(encoding='utf-8'))
        for name, info in self.meta.get('artifacts', {}).items():
            path = self.root / name
            if not path.is_file() or path.stat().st_size != info['bytes']:
                raise ValueError(f'Prepared artifact missing or truncated: {name}')
        self.arr = {p.stem: np.load(p, mmap_mode='r') for p in self.root.glob('*.npy')}
        self.cfg = self.meta['config']
        required = {'query_day', 'query_block', 'query_split', 'candidate_item', 'candidate_z',
                    'candidate_price_day', 'day_time', 'day_prev', 'day_user', 'bag_ptr',
                    'event_item', 'event_price', 'item_category', 'mean', 'scale',
                    'category_interpretable', 'daily_log_price', 'support', 'block_start'}
        missing = required - self.arr.keys()
        if self.meta.get('adapter') == 'processed_rows_v1':
            missing |= {'day_order', 'price_snapshot_z', 'price_snapshot_day'} - self.arr.keys()
        if missing:
            raise ValueError(f'Incomplete prepared arrays: {sorted(missing)}')
        a = self.arr
        q, d = len(a['query_day']), len(a['day_time'])
        shape = (q, 1 if self.meta.get('adapter') == 'processed_rows_v1' else self.cfg['negatives'] + 1)
        if any(a[k].shape != shape for k in ('candidate_item', 'candidate_z', 'candidate_price_day')):
            raise ValueError('Prepared candidate array shapes disagree with the original training config.')
        if any(len(a[k]) != q for k in ('query_block', 'query_split')):
            raise ValueError('Prepared query array lengths disagree.')
        if len(a['day_prev']) != d or len(a['day_user']) != d or len(a['bag_ptr']) != d + 1:
            raise ValueError('Prepared history array lengths disagree.')
        if a['bag_ptr'][0] != 0 or a['bag_ptr'][-1] != len(a['event_item']) or len(a['event_price']) != len(a['event_item']):
            raise ValueError('Prepared event arrays are incomplete.')
        if q == 0 or a['query_day'].min() < 0 or a['query_day'].max() >= d:
            raise ValueError('Prepared query day indices are invalid.')
        if len(a['mean']) != len(self.meta['category_ids']) or len(a['item_category']) != len(self.meta['article_ids']):
            raise ValueError('Prepared mappings disagree with model dimensions.')
        if not np.all(np.isfinite(a['scale']) & (a['scale'] > 0)):
            raise ValueError('Prepared price scales must be finite and positive.')

    def indices(self, split):
        return np.flatnonzero(self.arr['query_split'] == split)

    def history(self, last):
        result = []
        while last >= 0 and len(result) < self.cfg['history_days']:
            result.append(int(last))
            last = self.arr['day_prev'][last]
        return result[::-1]

    def encode_input(self, last_days, cutoff_days, device, cutoff_orders=None):
        """CSR day bags flattened to events; permutation-invariant bag embedding in model."""
        import torch
        if self.meta.get('adapter') == 'processed_rows_v1':
            return self._encode_source_rows(last_days, cutoff_days, device, cutoff_orders)
        L, B = self.cfg['history_days'], len(last_days)
        items, prices, slots = [], [], []
        ages = np.zeros((B, L), np.float32)
        lengths = []
        a = self.arr
        for b, (last, cutoff) in enumerate(zip(last_days, cutoff_days)):
            history = self.history(int(last))
            lengths.append(len(history))
            for j, day in enumerate(history):
                if cutoff_orders is None:
                    if a['day_time'][day] >= cutoff:
                        raise ValueError('History is not strictly earlier than prediction cutoff.')
                elif a['day_order'][day] >= cutoff_orders[b] or a['day_time'][day] > cutoff:
                    raise ValueError('History violates the declared source-row prediction order.')
                left, right = a['bag_ptr'][day:day + 2]
                it = a['event_item'][left:right]
                cat = a['item_category'][it]
                items.extend(it)
                prices.extend((a['event_price'][left:right] - a['mean'][cat]) / a['scale'][cat])
                slots.extend([b * L + j] * len(it))
                ages[b, j] = np.log1p(cutoff - a['day_time'][day])
        def tensor(x, dtype):
            return torch.tensor(np.asarray(x), dtype=dtype, device=device)
        return {'items': tensor(items, torch.long), 'prices': tensor(prices, torch.float32),
                'slots': tensor(slots, torch.long), 'ages': tensor(ages, torch.float32),
                'lengths': tensor(lengths, torch.long), 'batch_size': B, 'length': L}

    def _encode_source_rows(self, last_rows, cutoff_days, device, cutoff_orders=None):
        """Vectorized history assembly for the processed one-row-per-event adapter."""
        import torch
        a = self.arr
        L, B = self.cfg['history_days'], len(last_rows)
        recent = np.full((B, L), -1, dtype=np.int64)
        current = np.asarray(last_rows, dtype=np.int64).copy()
        lengths = np.zeros(B, dtype=np.int64)
        rows = np.arange(B)
        for offset in range(L):
            valid = current >= 0
            if not valid.any():
                break
            recent[rows[valid], offset] = current[valid]
            lengths[valid] += 1
            current[valid] = a['day_prev'][current[valid]]

        valid = np.arange(L)[None, :] < lengths[:, None]
        cols = np.clip(lengths[:, None] - 1 - np.arange(L)[None, :], 0, max(0, L - 1))
        history = np.take_along_axis(recent, cols, axis=1)
        history[~valid] = -1
        safe = np.maximum(history, 0)
        cutoff_days = np.asarray(cutoff_days, dtype=np.int64)
        history_days = a['day_time'][safe]
        if cutoff_orders is None:
            invalid = valid & (history_days >= cutoff_days[:, None])
        else:
            cutoff_orders = np.asarray(cutoff_orders, dtype=np.int64)
            history_orders = a['day_order'][safe]
            invalid = valid & ((history_orders >= cutoff_orders[:, None]) |
                               (history_days > cutoff_days[:, None]))
        if invalid.any():
            raise ValueError('History violates the declared source-row prediction order.')

        item = a['event_item'][safe[valid]]
        category = a['item_category'][item]
        price = (a['event_price'][safe[valid]] - a['mean'][category]) / a['scale'][category]
        ages = np.zeros((B, L), dtype=np.float32)
        if valid.any():
            ages[valid] = np.log1p(cutoff_days[:, None] - history_days)[valid]
        slots = np.flatnonzero(valid.reshape(-1)).astype(np.int64, copy=False)

        def tensor(x, dtype):
            return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
        return {'items': tensor(item, torch.long), 'prices': tensor(price, torch.float32),
                'slots': tensor(slots, torch.long), 'ages': tensor(ages, torch.float32),
                'lengths': tensor(lengths, torch.long), 'batch_size': B, 'length': L}

    def batch(self, indices, device, include_candidates=True):
        import torch
        a = self.arr
        qday = a['query_day'][indices]
        days = a['day_time'][qday]
        orders = a['day_order'][qday] if 'day_order' in a else None
        result = {'history': self.encode_input(a['day_prev'][qday], days, device, orders)}
        result['days'] = torch.tensor(np.asarray(days), dtype=torch.long, device=device)
        if not include_candidates:
            return result
        items = np.asarray(a['candidate_item'][indices])
        z = a['candidate_z'][indices]
        if self.meta.get('adapter') == 'processed_rows_v1':
            from processed_adapter import sampled_candidates
            items, z = sampled_candidates(self, indices)
        for key, value, dtype in (
            ('items', items, torch.long), ('z', z, torch.float32),
            ('query_ids', indices, torch.long)):
            result[key] = torch.tensor(np.asarray(value), dtype=dtype, device=device)
        result['labels'] = torch.zeros_like(result['z'])
        result['labels'][:, 0] = 1
        result['mask'] = result['items'] != 0
        return result

    def support_input(self, block, categories, device):
        a = self.arr
        import torch
        cache = getattr(self, '_support_input_cache', None)
        if cache is None:
            cache = self._support_input_cache = {}
        parts, counts = [], []
        device_key = str(device)
        for category in np.asarray(categories, dtype=np.int64):
            key = (int(block), int(category), device_key)
            entry = cache.get(key)
            if entry is None:
                source = a['support'][block, category]
                source = source[source >= 0]
                entry = self.encode_input(source, np.full(len(source), a['block_start'][block]), device)
                cache[key] = entry
            parts.append(entry)
            counts.append(entry['batch_size'])

        if not parts:
            raise ValueError('Support input requires at least one category.')
        fields = {}
        for name in ('items', 'prices', 'ages', 'lengths'):
            fields[name] = torch.cat([x[name] for x in parts], dim=0)
        slot_parts, row_offset = [], 0
        for x, count in zip(parts, counts):
            slot_parts.append(x['slots'] + row_offset * self.cfg['history_days'])
            row_offset += count
        fields['slots'] = torch.cat(slot_parts, dim=0)
        fields['batch_size'] = sum(counts)
        fields['length'] = self.cfg['history_days']
        owner = np.repeat(np.arange(len(categories)), counts)
        return fields, owner
