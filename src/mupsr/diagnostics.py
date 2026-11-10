"""Stage-0 diagnostics (no training): quantify the repeat-purchase share, the cold-start share, the
candidate-pool size and the user concentration of the training loss, and report three reference baselines
computed under exactly the evaluation protocol (full day-eligible candidate set, tie-aware):
  POP-EWMA : the time-decayed item popularity stock (Eq. 26 applied to counts, information set d^-)
  REP      : the user's history items (by count and recency) first, the rest by POP-EWMA
  stage-1 NLL reference : the per-label NLL of static exposure-adjusted popularity

    python -m mupsr.diagnostics --data ... --source ... --cache ... --out runs/diagnostics
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

KS = (5, 10, 20)


def tie_metrics(s, t):
    """The same single-positive tie-aware formulas as metrics.ranking_metrics. s: eligible-candidate scores
    (ineligible = -inf)."""
    g, e = float((s > t).sum()), float((s == t).sum())
    disc = np.r_[0.0, np.cumsum(1 / np.log2(np.arange(2, max(KS) + 2)))]
    out = {}
    for k in KS:
        out[f'R@{k}'] = min(max(k - g, 0), e) / e
        out[f'N@{k}'] = (disc[int(min(g + e, k))] - disc[int(min(g, k))]) / e
    return out


def run(data, tables, X, rep_index, W=64, strata=None):
    day = np.asarray(data.a['day_time']); item, time = data.item, data.time
    elig = tables['eligible']
    rep = {}
    # ---- 1. user concentration of the training loss (training weights by transaction, evaluation by user)
    q = np.asarray(data.query_rows['train'])
    per_user = np.sort(np.bincount(np.asarray(data.a['day_user'])[q]))[::-1]
    per_user = per_user[per_user > 0]
    rep['train_queries_share_top1pct_users'] = float(per_user[:max(1, len(per_user)//100)].sum() / per_user.sum())
    rep['train_queries_share_top10pct_users'] = float(per_user[:max(1, len(per_user)//10)].sum() / per_user.sum())
    rep['train_queries_per_user_q50_90_99_max'] = [int(x) for x in np.quantile(per_user, [.5, .9, .99, 1])]
    # ---- 2. static-popularity per-label stage-1 NLL (reference for the stage-1 loss)
    li, lc = tables['label_index'], tables['label_count']
    Y = np.bincount(li[:, 1], weights=lc, minlength=elig.shape[1]) + .5
    with np.errstate(divide='ignore'):                 # days with an empty pool carry no labels
        logZ = np.log((elig * Y[None]).sum(1))
        log_pool = np.log(elig.sum(1))
    rep['stage1_reference_nll_static_pop'] = float((lc * (logZ[li[:, 0]] - np.log(Y[li[:, 1]]))).sum() / lc.sum())
    rep['stage1_reference_nll_uniform'] = float((lc * log_pool[li[:, 0]]).sum() / lc.sum())
    # ---- 3. evaluation-query profile + baselines
    for split in ('valid', 'test'):
        rows = np.asarray(data.query_rows[split])
        hl = np.array([len(data.prefix(r)) for r in rows])
        res = {'queries': len(rows), 'mean_pool': float(elig.sum(1)[day[rows]].mean())}
        edges = [0, 1, 2, 5, 20, 100, 10**9]
        res['history_len_share'] = {f'[{a},{b})': float(((hl >= a) & (hl < b)).mean()) for a, b in zip(edges[:-1], edges[1:])}
        is_rep, in_window, same_basket, in_cap, dup = [], [], [], [], []
        sums = {m: {'all': 0.0, 'repeat': 0.0, 'new': 0.0, 'cold': 0.0} for m in ('POP', 'REP')}
        for r, L in zip(rows, hl):
            h = data.prefix(r)
            tgt = int(item[r])
            hit = np.flatnonzero(item[h] == tgt)
            is_rep.append(len(hit) > 0)
            in_window.append(len(hit) > 0 and hit[-1] >= max(0, (L // W - 1) * W))
            same_basket.append(L > 0 and time[h[-1]] == time[r])
            dup.append(bool(np.any((item[h] == tgt) & (time[h] == time[r]))))   # same-basket duplicate rows
            rq, it, gap, cnt, _, _ = rep_index.gather([r])
            in_cap.append(tgt in set(it.tolist()))
            pool = elig[day[r]]
            pop = np.where(pool, X[day[r], :, 1], -np.inf)                 # 7-day half-life
            s_rep = pop.copy()
            ok = pool[it]
            s_rep[it[ok]] += 100 + 2 * np.log1p(cnt[ok]) - np.log1p(gap[ok])
            for name, s in (('POP', pop), ('REP', s_rep)):
                m = tie_metrics(s, s[tgt])['N@10']
                sums[name]['all'] += m
                sums[name]['repeat' if is_rep[-1] else 'new'] += m
                sums[name]['cold'] += m if L == 0 else 0
        is_rep = np.array(is_rep)
        if strata is not None:                       # per-query strata, same order as per_query_N10_{split}.npy
            strata[split] = {'history_len': hl, 'repeat': is_rep, 'query_rows': rows,
                             'same_basket_duplicate': np.asarray(dup, bool)}
        res.update(repeat_share=float(is_rep.mean()), repeat_in_readable_window_share=float(np.mean(in_window)),
                   repeat_within_cap_share=float(np.mean(in_cap)), prev_row_same_timestamp_share=float(np.mean(same_basket)),
                   cold_share=float((hl == 0).mean()), same_basket_duplicate_target_share=float(np.mean(dup)))
        nr, nn, nc = max(is_rep.sum(), 1), max((~is_rep).sum(), 1), max((hl == 0).sum(), 1)
        for name in sums:
            res[f'{name}_N@10'] = {'all': sums[name]['all'] / len(rows), 'repeat_targets': sums[name]['repeat'] / nr,
                                   'new_targets': sums[name]['new'] / nn, 'cold_users': sums[name]['cold'] / nc}
        rep[split] = res
    return rep


def main(argv=None):
    from .data import EventData, build_day_tables
    from .features import popularity_stock, RepeatIndex
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data', required=True); p.add_argument('--source', required=True); p.add_argument('--cache', required=True)
    p.add_argument('--user-block', type=int, default=64); p.add_argument('--repeat-cap', type=int, default=128)
    p.add_argument('--out', required=True, help='directory for diagnostics.json and strata_{valid,test}.npz')
    a = p.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = EventData(a.data, a.source, a.cache, 5.0)
    tables = build_day_tables(data)
    X = popularity_stock(data.train_rows, data.item, data.time, np.asarray(data.a['day_time']), len(data.cat), tables['D'])
    rep_index = RepeatIndex(data.item, data.time, np.asarray(data.events['logp']), np.asarray(data.events['missing']),
                            data.user_rows, cap=a.repeat_cap)
    strata = {}
    report = run(data, tables, X, rep_index, a.user_block, strata)
    report['identity'] = {'event_data': data.fingerprint,
                          'candidate_fingerprint': data.protocol.fingerprint,
                          'user_block': a.user_block, 'repeat_cap': a.repeat_cap}
    (out / 'diagnostics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for split, arrays in strata.items():
        np.savez(out / f'strata_{split}.npz', **arrays)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
