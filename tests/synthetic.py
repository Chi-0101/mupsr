"""Tiny synthetic dataset in the on-disk format of the real prepared Cosmetics cache (adapter
processed_rows_v1) plus the three split CSVs; used by the tests and the optimization smoke runs."""
import csv, gzip, hashlib, json, sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make(root, seed=0, U=70, I=150, C=9, D=40, max_age=30, filter_train_queries=True):
    rng = np.random.default_rng(seed)
    root = Path(root); cache, src = root / 'prepared', root / 'csv'
    cache.mkdir(parents=True, exist_ok=True); src.mkdir(parents=True, exist_ok=True)
    item_category = np.r_[0, rng.integers(1, C, I - 1)]
    interpretable = np.ones(C, bool); interpretable[0] = False; interpretable[C - 1] = False
    base_price = np.exp(rng.normal(2, 1, I))
    lengths = rng.integers(1, 30, U); lengths[0] = 150; lengths[1] = 70           # heavy users span several blocks
    rows = []
    for u in range(U):
        t = np.sort(rng.integers(0, D * 86400, lengths[u]))
        basket = rng.random(lengths[u]) < .25
        t[1:][basket[1:]] = t[:-1][basket[1:]]                                   # same-timestamp baskets
        t = np.maximum.accumulate(t)
        fav = rng.integers(1, I, 5)
        for s in t:
            j = int(rng.choice(fav)) if rng.random() < .4 else int(rng.integers(1, I))
            rows.append((int(s), u, j))
    rows.sort(key=lambda r: (r[0], r[1]))
    n = len(rows)
    sec = np.array([r[0] for r in rows]); user = np.array([r[1] for r in rows]); item = np.array([r[2] for r in rows])
    order = np.cumsum(rng.integers(1, 4, n))                                     # strictly increasing source order
    price = base_price[item] * np.exp(rng.normal(0, .1, n)); valid_p = rng.random(n) > .03
    day = sec // 86400
    split = np.zeros(n, np.int8)
    for u in range(U):
        r = np.flatnonzero(user == u)
        if len(r) >= 2: split[r[-2]] = 1
        split[r[-1]] = 2
    logp = np.log(price)
    tr = (split == 0) & valid_p
    C_mean = np.array([logp[tr & (item_category[item] == c)].mean() if np.any(tr & (item_category[item] == c)) else 0
                       for c in range(C)], np.float32)
    C_scale = np.array([max(logp[tr & (item_category[item] == c)].std(), .05) if np.any(tr & (item_category[item] == c)) else 1
                        for c in range(C)], np.float32)
    snap_day = np.full((D, I), -1, np.int32); snap_z = np.full((D, I), np.nan, np.float32)
    last_d = np.full(I, -1); last_z = np.full(I, np.nan)
    for d in range(D):
        snap_day[d], snap_z[d] = last_d, last_z
        m = tr & (day == d)
        for j in np.unique(item[m]):
            last_d[j] = d; last_z[j] = (logp[m & (item == j)].mean() - C_mean[item_category[j]]) / C_scale[item_category[j]]
    def eligible(d, j):
        return snap_day[d, j] >= 0 and d - snap_day[d, j] <= max_age and interpretable[item_category[j]]
    # Like the real cache, a query is kept only if its target has an eligible price on its day
    # (filter_train_queries=False reproduces a cache without this property, for the optimization tests).
    keep_q = np.array([(s == 0 and not filter_train_queries) or eligible(day[r], item[r]) for r, s in enumerate(split)])
    q_rows = np.flatnonzero(keep_q)
    prev = np.full(n, -1, np.int64)
    for u in range(U):
        r = np.flatnonzero(user == u); prev[r[1:]] = r[:-1]
    arrays = dict(query_day=q_rows.astype(np.int64), query_block=np.zeros(len(q_rows), np.int32),
                  query_split=split[q_rows], candidate_item=item[q_rows, None].astype(np.int32),
                  candidate_z=np.zeros((len(q_rows), 1), np.float32), candidate_price_day=np.zeros((len(q_rows), 1), np.int32),
                  day_time=day.astype(np.int32), day_prev=prev, day_user=user.astype(np.int32),
                  bag_ptr=np.arange(n + 1, dtype=np.int64), event_item=item.astype(np.int32),
                  event_price=logp.astype(np.float32), item_category=item_category.astype(np.int32),
                  mean=C_mean, scale=C_scale, category_interpretable=interpretable,
                  daily_log_price=np.zeros((D, I), np.float32), support=np.zeros((1, C, 1), np.int32),
                  block_start=np.zeros(1, np.int32), day_order=order.astype(np.int64),
                  price_snapshot_z=snap_z, price_snapshot_day=snap_day)
    for k, v in arrays.items():
        np.save(cache / f'{k}.npy', v)
    origin = datetime(2019, 10, 1)
    sources = {}
    for si, name in enumerate(('train', 'valid', 'test')):
        path = src / f'{name}.csv.gz'
        with gzip.open(path, 'wt', encoding='utf-8', newline='') as f:
            w = csv.writer(f); w.writerow(['source_order', 'item_idx', 'user_idx', 'event_time', 'price', 'price_valid'])
            for r in np.flatnonzero(split == si):
                w.writerow([order[r], item[r], user[r], (origin + timedelta(seconds=int(sec[r]))).isoformat(),
                            f'{price[r]:.4f}' if valid_p[r] else '', int(valid_p[r])])
        sources[path.name] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    meta = {'adapter': 'processed_rows_v1', 'config': {'dataset': 'cosmetics', 'max_price_age_days': max_age, 'history_days': 20},
            'sources': sources, 'origin_date': origin.date().isoformat(),
            'audit': {f'rows_{s}': int((split == i).sum()) for i, s in enumerate(('train', 'valid', 'test'))},
            'category_ids': [str(c) for c in range(C)], 'article_ids': [str(j) for j in range(I)],
            'artifacts': {p.name: {'bytes': p.stat().st_size} for p in cache.glob('*.npy')}}
    (cache / 'metadata.json').write_text(json.dumps(meta, indent=2))
    (cache / 'READY').write_text(digest(cache / 'metadata.json'))
    return cache, src, n


if __name__ == '__main__':
    print(make(sys.argv[1]))
