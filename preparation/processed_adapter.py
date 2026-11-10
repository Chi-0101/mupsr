"""Adapter for existing split CSVs. Source rows are singleton sequence steps.

No re-splitting; no claim that source order is true same-timestamp order.
Market price proxies use TRAIN rows from strictly earlier calendar days.
"""
import csv
import gzip
import hashlib
import heapq
import json
import math
import sqlite3
from array import array
from collections import Counter, OrderedDict
from datetime import date
from itertools import groupby
from pathlib import Path

import numpy as np

from data import ArrayWriter, digest_file, write_json


def read_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def read_mapping(source, name, key):
    result = ['<PAD>']
    for row in read_rows(source / (name + '.csv.gz')):
        if int(row[key + '_idx']) != len(result):
            raise ValueError(f'{name}: mapping must be contiguous and start at 1')
        result.append(row[key + '_id'])
    return result


def prepare_processed(source, output, cfg):
    source, root = Path(source), Path(output)
    if not (source / 'READY').is_file():
        raise ValueError('Processed input READY is missing.')
    if root.exists() and any(root.iterdir()):
        raise ValueError('Preparation needs an empty output directory.')
    if cfg['sequence_policy'] != 'source_rows' or cfg['market_price_policy'] != 'train_past_days':
        raise ValueError('Unsupported sequence/market-price policy.')
    for k in ('history_days', 'min_history_days', 'negatives', 'block_days', 'support_users', 'max_price_age_days'):
        if cfg[k] < 1:
            raise ValueError(f'{k} must be positive')
    if cfg['min_history_days'] > cfg['history_days']:
        raise ValueError('min_history_days exceeds history_days (these count rows in this adapter).')
    summary = json.loads((source / 'summary.json').read_text(encoding='utf-8-sig'))
    if summary['dataset'] != cfg['dataset']:
        raise ValueError('Input dataset does not match config.')
    root.mkdir(parents=True, exist_ok=True)
    paths = [source / n for n in ('summary.json', 'READY', 'items.csv.gz', 'categories.csv.gz',
                                  'users.csv.gz', 'train.csv.gz', 'valid.csv.gz', 'test.csv.gz')]
    sources = {p.name: {'bytes': p.stat().st_size, 'sha256': digest_file(p)} for p in paths}
    ids = read_mapping(source, 'items', 'item')
    categories = read_mapping(source, 'categories', 'category')
    I, C = len(ids), len(categories)
    item_cat = np.zeros(I, np.int32)
    audit = Counter()
    db = sqlite3.connect(root / 'preparation.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('PRAGMA cache_size=-131072')
    db.execute('CREATE TABLE rows(u INTEGER,d INTEGER,t TEXT,o INTEGER,i INTEGER,p REAL,s INTEGER)')
    try:
        for split, name in enumerate(('train', 'valid', 'test')):
            pending = []
            previous_key = (-1,-1)
            for row in read_rows(source / (name + '.csv.gz')):
                i, c, u = (int(row[k]) for k in ('item_idx', 'category_idx', 'user_idx'))
                order = int(row['source_order'])
                if (u,order) <= previous_key:
                    raise ValueError(f'{name}: expected user-sorted CSV preserving source_order within each user')
                previous_key = (u,order)
                if not (0 < i < I and 0 < c < C and u > 0):
                    raise ValueError('Out-of-range mapped ID')
                if ids[i] != row['item_id'] or categories[c] != row['category_id']:
                    raise ValueError('CSV IDs disagree with mapping files.')
                if item_cat[i] and item_cat[i] != c:
                    raise ValueError('Conflicting item category after upstream preprocessing')
                item_cat[i] = c
                t = row['event_time']
                d = date.fromisoformat(t[:10]).toordinal()
                p = float(row['price']) if row['price_valid'] == '1' and row['price'] else float('nan')
                logp = math.log(p) if math.isfinite(p) and p > 0 else None
                audit[f'rows_{name}'] += 1
                if logp is None:
                    audit['invalid_prices_imputed_in_history'] += 1
                pending.append((u, d, t, order, i, logp, split))
                if len(pending) == 50000:
                    db.executemany('INSERT INTO rows VALUES(?,?,?,?,?,?,?)', pending)
                    db.commit()
                    pending.clear()
            db.executemany('INSERT INTO rows VALUES(?,?,?,?,?,?,?)', pending)
            db.commit()
            print(f'import {name}: {audit[f"rows_{name}"]:,}', flush=True)
            expected = summary.get('prepared', {}).get(name, {}).get('interactions')
            if expected is not None and audit[f'rows_{name}'] != expected:
                raise ValueError(f'{name}: row count disagrees with source summary')
        print('Indexing source rows and training-only daily prices...', flush=True)
        db.execute('CREATE UNIQUE INDEX row_order ON rows(o)')
        db.execute('CREATE INDEX row_time ON rows(d,o)')
        db.execute('CREATE INDEX row_user_order ON rows(u,o)')
        # Reject split inversions and timestamp inversions rather than sorting them away.
        bad = db.execute('SELECT COUNT(*) FROM (SELECT s,d,t,LAG(s) OVER w ps, '
                         'LAG(d) OVER w pd,LAG(t) OVER w pt FROM rows '
                         'WINDOW w AS (PARTITION BY u ORDER BY o)) '
                         'WHERE s<ps OR d<pd OR t<pt').fetchone()[0]
        if bad:
            raise ValueError(f'{bad} source-order time/split inversions; input is not compatible.')
        audit['same_timestamp_groups_across_splits'] = db.execute(
            'SELECT COUNT(*) FROM (SELECT u,t FROM rows GROUP BY u,t HAVING MIN(s)!=MAX(s))').fetchone()[0]
        audit['same_timestamp_multirow_groups'] = db.execute(
            'SELECT COUNT(*) FROM (SELECT u,t FROM rows GROUP BY u,t HAVING COUNT(*)>1)').fetchone()[0]
        # Deduplicate identical transaction prices, preserving the original proxy semantics.
        db.execute('CREATE TABLE daily_price AS SELECT d,i,AVG(p) p FROM '
                   '(SELECT DISTINCT d,i,p FROM rows WHERE s=0 AND p IS NOT NULL) GROUP BY d,i')
        start, last = db.execute('SELECT MIN(d),MAX(d) FROM rows').fetchone()
        if start is None:
            raise ValueError('Empty processed input')
        T = last - start + 1
        db.execute('UPDATE daily_price SET d=d-?', (start,))
        db.execute('CREATE UNIQUE INDEX price_di ON daily_price(d,i)')
        db.commit()
        sums, squares, counts = np.zeros(C), np.zeros(C), np.zeros(C, np.int64)
        for i, p in db.execute('SELECT i,p FROM daily_price'):
            c = item_cat[i]
            sums[c] += p
            squares[c] += p*p
            counts[c] += 1
        means = sums / np.maximum(counts, 1)
        std = np.sqrt(np.maximum(0, squares / np.maximum(counts, 1) - means**2))
        variable = (counts >= 2) & (std >= 1e-4)
        scale = np.where(variable, std, 1.0)
        # Daily snapshots replace Q*100 materialized candidate arrays. Negatives are
        # regenerated deterministically from the fixed data seed and query ID.
        price_z = np.lib.format.open_memmap(root / 'price_snapshot_z.npy', mode='w+', dtype=np.float32, shape=(T,I))
        price_day = np.lib.format.open_memmap(root / 'price_snapshot_day.npy', mode='w+', dtype=np.int32, shape=(T,I))
        daily = np.full((T,C), np.nan, np.float32)
        latest, latest_day = np.full(I, np.nan), np.full(I, -1, np.int32)
        day_time, day_prev, day_user, day_order = array('i'), array('q'), array('q'), array('q')
        last_row, last_train, history_count = {}, {}, Counter()
        support_members = [set() for _ in range(C)]
        heaps = [[] for _ in range(C)]
        snapshots, starts = [], []
        S = cfg['support_users']
        names = {'query_day': (np.int64,None), 'query_block': (np.int32,None),
                 'query_split': (np.int8,None), 'candidate_item': (np.int32,1),
                 'candidate_z': (np.float32,1), 'candidate_price_day': (np.int32,1),
                 'event_item': (np.int32,None), 'event_price': (np.float32,None)}
        writers = {n: ArrayWriter(root,n,*spec) for n,spec in names.items()}
        def snapshot(d):
            a = np.full((C,S), -1, np.int64)
            for c in range(1,C):
                for j,u in enumerate(sorted(support_members[c])):
                    a[c,j] = last_train[u]
            snapshots.append(a)
            starts.append(d)
        # One streaming SQL cursor avoids repeatedly seeking through millions of rows.
        row_days = iter(groupby(db.execute('SELECT u,d,o,i,p,s FROM rows ORDER BY d,o'), key=lambda x:x[1]))
        next_group = next(row_days, None)
        seen_reference = np.zeros(C, bool)
        for d in range(T):
            if d % cfg['block_days'] == 0:
                snapshot(d)
            block = len(snapshots)-1
            price_z[d] = (latest - means[item_cat]) / scale[item_cat]
            price_day[d] = latest_day
            eligible = (latest_day >= 0) & (d-latest_day <= cfg['max_price_age_days'])
            pool_sizes = np.bincount(item_cat[eligible], minlength=C)
            if next_group is not None and next_group[0] == start+d:
                for u,_,order,i,p,split in next_group[1]:
                    idx, c = len(day_time), item_cat[i]
                    prev = last_row.get(u,-1)
                    reason = None
                    if history_count[u] < cfg['min_history_days']:
                        reason = 'short_history'
                    elif not variable[c]:
                        reason = 'no_category_price_variation'
                    elif not eligible[i]:
                        reason = 'target_price_unavailable'
                    elif not seen_reference[c]:
                        reason = 'reference_unavailable'
                    elif pool_sizes[c] < 2:
                        reason = 'no_negative_candidates'
                    if reason:
                        audit[f'query_skip_{("train","valid","test")[split]}_{reason}'] += 1
                    else:
                        writers['query_day'].add([idx])
                        writers['query_block'].add([block])
                        writers['query_split'].add([split])
                        writers['candidate_item'].add([i])
                        writers['candidate_z'].add([price_z[d,i]])
                        writers['candidate_price_day'].add([latest_day[i]])
                        audit[f'queries_{("train","valid","test")[split]}'] += 1
                        if pool_sizes[c] < cfg['negatives']+1:
                            audit['queries_with_padded_training_negatives'] += 1
                    day_time.append(d)
                    day_prev.append(prev)
                    day_user.append(u)
                    day_order.append(order)
                    writers['event_item'].add([i])
                    writers['event_price'].add([means[c] if p is None else p])
                    last_row[u] = idx
                    history_count[u] += 1
                    if split == 0:
                        last_train[u] = idx
                        if u not in support_members[c]:
                            priority = int.from_bytes(hashlib.blake2b(
                                f'{cfg["seed"]}:{u}:{c}'.encode(),digest_size=8).digest(),'big')
                            h = heaps[c]
                            if len(h)<S:
                                heapq.heappush(h,(-priority,u))
                                support_members[c].add(u)
                            elif priority < -h[0][0]:
                                _,removed = heapq.heapreplace(h,(-priority,u))
                                support_members[c].remove(removed)
                                support_members[c].add(u)
                next_group = next(row_days,None)
            totals, n = np.zeros(C), np.zeros(C,np.int32)
            for i,p in db.execute('SELECT i,p FROM daily_price WHERE d=? ORDER BY i',(d,)):
                c = item_cat[i]
                latest[i], latest_day[i] = p,d
                totals[c] += p
                n[c] += 1
            observed = n>0
            daily[d,observed] = totals[observed]/n[observed]
            seen_reference |= observed
            if d % 30 == 0:
                print(f'day={d}/{T} rows={len(day_time):,} train_queries={audit["queries_train"]:,}',flush=True)
        snapshot(T)
        export_block = len(snapshots)-1
        for writer in writers.values():
            writer.finish()
        price_z.flush()
        price_day.flush()
        del price_z,price_day
        arrays = {'day_time': np.asarray(day_time,np.int32), 'day_prev': np.asarray(day_prev,np.int64),
                  'day_user': np.asarray(day_user,np.int64), 'day_order': np.asarray(day_order,np.int64),
                  'bag_ptr': np.arange(len(day_time)+1,dtype=np.int64), 'item_category':item_cat,
                  'mean': means.astype(np.float32), 'scale':scale.astype(np.float32),
                  'category_interpretable':variable,'daily_log_price':daily,
                  'support':np.stack(snapshots), 'block_start':np.asarray(starts,np.int32)}
        for n,a in arrays.items():
            np.save(root / (n+'.npy'),a)
        audit['users'] = len(last_row)
        audit['items'] = I-1
        audit['categories'] = C-1
        for s in ('train','valid','test'):
            audit[f'query_coverage_{s}'] = audit[f'queries_{s}']/max(1,audit[f'rows_{s}'])
        # Keep only the table required for candidate replay in the final cache.
        db.execute('DROP TABLE rows')
        db.commit()
        print('Compacting price database and hashing model artifacts...', flush=True)
        db.execute('VACUUM')
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        metadata = {'format_version':2,'adapter':'processed_rows_v1','config':cfg,
                    'audit':dict(audit),'sources':sources,'source_summary':summary,
                    'article_ids':ids,'category_ids':categories,'export_block':export_block,
                    'origin_date':date.fromordinal(start).isoformat(),
                    'task':'given-category single-source-row ranking; existing splits; rolling per-user source-row history',
                    'price_semantics':'train-only SKU-day mean of distinct log prices; strictly earlier calendar days',
                    'limitations':['Source order is assumed, not verified true order within a timestamp.',
                                   'User-level leave-last-out is not a global temporal backtest; model fitting and training normalization can use other users later training events.',
                                   'Price eligibility, history length and category variation filter the evaluated query population.'],
                    'negative_sampling':'PCG64 SeedSequence([data.seed,query_id]); no replacement; padding masked',
                    'negative_sampling_numpy_version': np.__version__,
                    'adapter_source_sha256': {n:digest_file(Path(__file__).with_name(n))
                                              for n in ('data.py','processed_adapter.py')},
                    'history_unit':'source row; historical history_days/min_history_days keys count rows'}
        artifacts = [*root.glob('*.npy'),root/'preparation.sqlite']
        # SQLite must be closed before its checksum is recorded.
    finally:
        db.close()
    metadata['artifacts'] = {p.name:{'bytes':p.stat().st_size,'sha256':digest_file(p)} for p in artifacts}
    write_json(root/'metadata.json',metadata)
    if any(audit[f'queries_{s}']==0 for s in ('train','valid','test')):
        raise ValueError('A split has zero eligible queries; inspect metadata.json. READY was not written.')
    (root/'READY').write_text(digest_file(root/'metadata.json'),encoding='ascii')
    print(json.dumps(dict(audit),indent=2),flush=True)


def candidate_pool(data, day, category):
    if not hasattr(data,'_pool_cache'):
        data._pool_cache = OrderedDict()
        data._category_items = [np.flatnonzero(data.arr['item_category']==c) for c in range(len(data.arr['mean']))]
    key = (int(day),int(category))
    if key not in data._pool_cache:
        it = data._category_items[category]
        pd = data.arr['price_snapshot_day'][day,it]
        keep = (pd>=0)&(pd<day)&(day-pd<=data.cfg['max_price_age_days'])
        data._pool_cache[key] = it[keep]
        if len(data._pool_cache)>256:
            data._pool_cache.popitem(last=False)
    data._pool_cache.move_to_end(key)
    return data._pool_cache[key]


def sampled_candidates(data, indices):
    if data.meta.get('negative_sampling_numpy_version',np.__version__) != np.__version__:
        raise ValueError('Use the cache preparation NumPy version to preserve identical sampled negatives.')
    a = data.arr
    width = data.cfg['negatives']+1
    items = np.zeros((len(indices),width),np.int32)
    z = np.zeros((len(indices),width),np.float32)
    for b,q in enumerate(indices):
        day = int(a['day_time'][a['query_day'][q]])
        pos = int(a['candidate_item'][q,0])
        pool = candidate_pool(data,day,int(a['item_category'][pos]))
        pool = pool[pool!=pos]
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([data.cfg['seed'],int(q)])))
        chosen = rng.choice(pool,min(len(pool),width-1),replace=False)
        items[b,:len(chosen)+1] = np.r_[pos,chosen]
        z[b,:len(chosen)+1] = a['price_snapshot_z'][day,items[b,:len(chosen)+1]]
    return items,z


def verify_artifacts(root):
    from data import PreparedData
    data = PreparedData(root)
    manifest = data.meta.get('artifacts')
    if not manifest:
        raise ValueError('Cache has no artifact manifest.')
    for name,expected in manifest.items():
        path = data.root/name
        if path.stat().st_size != expected['bytes'] or digest_file(path)!=expected['sha256']:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    return {'verified_artifacts':len(manifest),'data_fingerprint':data.fingerprint}
