"""Paired comparison of two runs on the same queries: per-query N@10 arrays written by mupsr.train or
mupsr.evaluate (per_query_N10_{split}.npy, query order = data.query_rows[split]). With the strata file of
diagnostics.py, also reports the gain within each history-length x repeat/new-target stratum.

    python -m mupsr.significance base.npy new.npy [strata_test.npz]
"""
import json
import sys

import numpy as np
from scipy import stats

HISTORY_BINS = ((0, 1, 'history 0'), (1, 5, 'history 1-4'), (5, 20, 'history 5-19'), (20, 10**9, 'history 20+'))


def compare(a, b, n_boot=10000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError('Per-query arrays must be aligned on the same queries.')
    d = b - a
    rng = np.random.default_rng(seed)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {'queries': len(d), 'base': float(a.mean()), 'new': float(b.mean()),
            'rel_gain': float(d.mean() / a.mean()) if a.mean() else float('nan'),
            'ci95_abs': [float(x) for x in np.quantile(boot, [.025, .975])],
            'paired_t_p': float(stats.ttest_rel(b, a).pvalue) if np.any(d != 0) else 1.0,
            'wilcoxon_p': float(stats.wilcoxon(d[d != 0]).pvalue) if np.any(d != 0) else 1.0}


def by_stratum(a, b, strata, n_boot=2000):
    hl, rep = strata['history_len'], strata['repeat'].astype(bool)
    out = {}
    for lo, hi, name in HISTORY_BINS:
        for flag, label in ((True, 'repeat'), (False, 'new')):
            m = (hl >= lo) & (hl < hi) & (rep == flag)
            if m.sum() >= 30:
                out[f'{name} / {label} target'] = compare(a[m], b[m], n_boot)
    return out


if __name__ == '__main__':
    a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
    result = {'overall': compare(a, b)}
    if len(sys.argv) > 3:
        result['strata'] = by_stratum(a, b, np.load(sys.argv[3]))
    print(json.dumps(result, indent=2))
