"""NumPy preprocessing for the optimized estimation and the state-dependence term.

  * RepeatIndex  : the user-item experience store read by the state-dependence term S_uj (B1);
  * fe_init      : the closed-form Poisson fixed-effect starting values b_j, eta, delta_c (A1);
  * popularity_stock : a time-decayed item purchase count used only by the stage-0 diagnostics baseline.

Information set, in line with Sec. 3.1:
  * market-layer quantities (fe_init, popularity_stock) use TRAIN records only, and only days strictly
    before the prediction day d (i.e. d^-);
  * the user-layer experience store reads only the query user's records before the query row (the same
    prefix as data.prefix), carrying no future information.
"""
import numpy as np


# ---------------------------------------------------------------- B1: user-item experience store
class RepeatIndex:
    """On-demand lookup of the user-item experience store (O(n) memory; no per-query lists stored).

    For position i in a user's sequence, nxt[i] is the position of that user's next purchase of the same
    item (or L if none). Position i is the most recent purchase of its item before p iff i < p <= nxt[i];
    so the experience set of query p is {i < p : nxt[i] >= p}, taken most-recent-first up to cap entries,
    with count = that item's cumulative purchase count through i (rank[i]), gap = t_p - t_i, and last paid
    price = logp[i]. Only records before the query row are read (the same prefix as data.prefix)."""

    def __init__(self, item, time, logp, missing, user_rows, cap=128):
        n = len(item)
        self.item, self.time = np.asarray(item), np.asarray(time)
        self.logp, self.missing = np.asarray(logp, np.float32), np.asarray(missing, bool)
        self.user_rows, self.cap = user_rows, cap
        self.user_of = np.empty(n, np.int64)
        self.pos = np.empty(n, np.int64)
        self.nxt = np.empty(n, np.int64)
        self.rank = np.empty(n, np.int32)
        for u, rows in user_rows.items():
            x = self.item[rows]
            L = len(rows)
            self.user_of[rows], self.pos[rows] = u, np.arange(L)
            seen, nxt = {}, np.empty(L, np.int64)
            for i in range(L - 1, -1, -1):
                nxt[i] = seen.get(int(x[i]), L)
                seen[int(x[i])] = i
            cnt, rank = {}, np.empty(L, np.int32)
            for i in range(L):
                cnt[int(x[i])] = rank[i] = cnt.get(int(x[i]), 0) + 1
            self.nxt[rows], self.rank[rows] = nxt, rank

    def gather(self, qrows):
        """A batch of query rows -> (query index rq, item, gap_days, count, last_logp, last_missing)."""
        parts = []
        for q, r in enumerate(np.asarray(qrows)):
            rows = self.user_rows[int(self.user_of[r])]
            p = int(self.pos[r])
            sel = rows[np.flatnonzero(self.nxt[rows[:p]] >= p)[::-1][:self.cap]]   # most recent first
            parts.append((np.full(len(sel), q), sel))
        rq = np.concatenate([a for a, _ in parts]).astype(np.int64) if parts else np.zeros(0, np.int64)
        src = np.concatenate([b for _, b in parts]).astype(np.int64) if parts else np.zeros(0, np.int64)
        qr = np.asarray(qrows)[rq]
        return (rq, self.item[src].astype(np.int64), np.maximum(self.time[qr] - self.time[src], 0).astype(np.float32),
                self.rank[src], self.logp[src], self.missing[src])


# ---------------------------------------------------------------- diagnostics popularity baseline
def popularity_stock(train_rows, item, time, day, n_items, D, half_lives=(1.0, 7.0, 28.0)):
    """X[d, j, h] = log1p( sum_{train m: day_m < d, item_m = j} 2^{-(d - t_m)/h} ): the exponential decay
    of Eq. (26) applied to purchase counts rather than prices. Recurrence S(d) = 2^{-1/h} S(d-1) + w(d-1),
    w(d') = sum_{m on day d'} 2^{-(d'+1-t_m)/h}. Used only by the stage-0 diagnostics popularity baseline."""
    tr = np.asarray(train_rows)
    dd, jj, tt = day[tr].astype(np.int64), item[tr].astype(np.int64), time[tr]
    out = np.zeros((D, n_items, len(half_lives)), np.float32)
    for k, h in enumerate(half_lives):
        w = np.zeros((D, n_items))
        np.add.at(w, (dd, jj), 2.0 ** (-(dd + 1 - tt) / h))
        s = np.zeros(n_items)
        decay = 2.0 ** (-1.0 / h)
        for d in range(D):
            out[d, :, k] = np.log1p(s)              # days before d only
            s = decay * s + w[d]
    return out


# ---------------------------------------------------------------- A1: Poisson fixed-effect init
def fe_init(eligible, label_index, label_count, item_category, price_level, alpha=0.5, iters=5, eta_floor=1e-3):
    """Concentrated-likelihood MLE of the Poisson fixed-effect model (Eqs. 24/33, item and day effects
    only), used as the starting values of b_j, eta, delta_c.

    The day effect lambda_d is absorbed by the softmax normalizer; the item effect's first-order condition
    Y_j = exp(rho_j) * sum_d n_d 1[j in J(d)] / Z_d yields the fixed-point iteration (Hausman, Hall &
    Griliches, 1984):
        rho_j <- log(Y_j + alpha) - log(sum_d n_d 1[j in J(d)] / Z_d(rho)),   Z_d = sum_{l in J(d)} exp(rho_l)
    whose first iteration (rho = 0) is the exposure-adjusted popularity log((Y_j + alpha) / exposure_j).
    Set k_c = -mean_{j in c} rho_j and b_j = rho_j + k_c (centred within category, matching
    centered_item_bias), so the initial score b_j - k_c = rho_j; then Mundlak-decompose k_c = eta * mu~_c +
    delta_c by purchase-weighted OLS (eta >= floor)."""
    D, I = eligible.shape
    E = np.asarray(eligible, bool)
    n_d = np.bincount(label_index[:, 0], weights=label_count, minlength=D)
    Y = np.bincount(label_index[:, 1], weights=label_count, minlength=I)
    real = np.arange(I) > 0
    ever = (E & (n_d > 0)[:, None]).any(0) & real      # items eligible only on label-free days have no exposure
    rho = np.zeros(I)
    for _ in range(iters):
        w = np.exp(rho - rho[ever].max())
        Z = (E * w[None]).sum(1)
        exposure = ((n_d / np.maximum(Z, 1e-300))[:, None] * E).sum(0)
        rho = np.where(ever, np.log(Y + alpha) - np.log(np.maximum(exposure, 1e-300)), 0.0)
        rho[ever] -= rho[ever].mean()
    rho[~ever & real] = rho[ever].min() if ever.any() else 0.0
    cat = np.asarray(item_category)
    C = len(price_level)
    cnt = np.bincount(cat[real], minlength=C).astype(float)
    k = -np.bincount(cat[real], weights=rho[real], minlength=C) / cnt.clip(min=1)
    b = np.zeros(I); b[real] = rho[real] + k[cat[real]]
    has = cnt > 0
    has[0] = False
    wts = np.bincount(cat[real], weights=Y[real], minlength=C)[has] + 1.0
    x, y = np.asarray(price_level, float)[has], k[has]
    xm, ym = np.average(x, weights=wts), np.average(y, weights=wts)
    slope = np.sum(wts * (x - xm) * (y - ym)) / max(np.sum(wts * (x - xm) ** 2), 1e-12)
    eta = max(slope, eta_floor)
    delta = np.zeros(C); delta[has] = y - eta * x
    delta[has] -= np.average(delta[has], weights=wts)  # constant cancels in the softmax
    return dict(item_bias=b.astype(np.float32), eta=float(eta), delta=delta.astype(np.float32),
                rho=rho, slope=float(slope), rho_std=float(rho[ever].std()))
