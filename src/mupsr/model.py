"""MUPSR model: mental-accounting two-stage price-aware recommender (Sec. 3.4-3.10).

Score of candidate j (category c) for a query on day d, Eq. (27'):
    s_uj = AU_uj - beta_uc * z_j + gamma_uc * (r_c - z_j) - k_uc
         = AU_uj - (beta + gamma) * z_j + (gamma * r_c - k_uc)
with AU_uj = b_j + e_j^T [W_g g_c + W_u (h_uc - W_0 g_c)] + S_uj, Eq. (41'), b_j centred within category.

Stage 1 (Sec. 3.6): causal self-attention over each category's daily panel -> g_c(d^-), price head (Eq. 34).
Stage 2 (Sec. 3.8): user sequence encoded in blocks with one-block memory, category-conditioned
bilinear readout with the category state as prior (Eqs. 38'-39), personal residuals (Eq. 40), and the
state-dependence term S_uj of Eq. (41b). The readout carries query-relative recency biases (Eq. 38').
Cross-category constants of Delta k and h_k are centred (Eq. 40'), removing the only directions that drift
under Adam while leaving the softmax of Eq. (12) unchanged.
Likelihood (Eqs. 12, 33): exact softmax over each day's price-eligible catalogue, evaluated over
fixed-width tiles that never cross a category.

Two configuration flags select the state-dependence channels for the nested models of Sec. 3.10:
``use_recency`` toggles the query-relative readout biases and ``use_repeat`` the state-dependence term
S_uj; a disabled channel keeps its zero initial value, is not trained, and contributes exactly zero to
every score. ``nested_gamma_zero`` (gamma == 0, no transaction utility) and ``nested_no_price_heterogeneity``
(w_a = w_b = 0) give the two price-side nested restrictions.
"""
import math
from typing import NamedTuple, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

EXCLUDED = -1e9  # finite: exp() underflows to exactly 0, and gradients stay finite


class Context(NamedTuple):
    """Everything Eq. (27') needs besides the candidate ids and prices."""
    match: torch.Tensor      # (B, C, d) matching vector per category
    slope: torch.Tensor      # (B, C)    beta + gamma
    intercept: torch.Tensor  # (B, C)    gamma * reference - k
    bias: torch.Tensor       # (I,)      category-centred item bias b_j


class Tiles(NamedTuple):
    cat: torch.Tensor    # (n_tiles,)      category of each tile
    items: torch.Tensor  # (n_tiles, tau)  item ids, 0 = padding (never eligible)


# ---------------------------------------------------------------- building blocks
def bucket(distance, count=64, maximum=1048576):
    """Exact buckets near zero, logarithmic buckets far away (Shaw et al., 2018; Raffel et al., 2020)."""
    exact = count // 2
    distance = distance.clamp_min(0)
    far = exact + (torch.log(distance.float().clamp_min(exact)/exact) /
                   math.log(maximum/exact) * (count-exact)).long()
    return torch.where(distance < exact, distance.long(), far).clamp_max(count-1)


class PriceEncoder(nn.Module):
    """Piecewise-linear price embedding with category FiLM, Eq. (28)."""

    def __init__(self, data, dim):
        super().__init__()
        for name, value in [('mean', data.mean), ('scale', data.scale),
                            ('knots_z', data.events['knots_z']), ('knots_ell', data.events['knots_ell'])]:
            self.register_buffer(name, torch.tensor(np.array(value), dtype=torch.float32))
        self.register_buffer('global_mean', torch.tensor(data.meta['global_mean']))
        self.register_buffer('global_scale', torch.tensor(data.meta['global_scale']))
        width = len(self.knots_z)+len(self.knots_ell)-2+1
        self.project = nn.Linear(width, dim, bias=False)
        # FiLM by category; zero init = identical encoding for all categories.
        self.film_scale, self.film_shift = nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.film_scale.weight)
        nn.init.zeros_(self.film_shift.weight)

    @staticmethod
    def basis(value, knots):
        # Boundary intervals extrapolate; internal intervals saturate.
        x = (value[..., None]-knots[:-1])/(knots[1:]-knots[:-1])
        lower = torch.zeros_like(x)
        upper = torch.ones_like(x)
        lower[..., 0] = -float('inf')
        upper[..., -1] = float('inf')
        return torch.maximum(lower, torch.minimum(upper, x))

    def forward(self, logp, missing, category, embedding):
        z = (logp-self.mean[category])/self.scale[category]
        ell = (logp-self.global_mean)/self.global_scale
        z = z.masked_fill(missing, 0)
        ell = ell.masked_fill(missing, 0)
        raw = self.project(torch.cat((self.basis(z, self.knots_z), self.basis(ell, self.knots_ell),
                                      missing.float()[..., None]), -1))
        return (1+self.film_scale(embedding))*raw+self.film_shift(embedding)


class AttentionLayer(nn.Module):
    """Pre-norm causal self-attention with relative-position (and optional time-gap) logit biases, Eq. (36)."""

    def __init__(self, cfg, temporal):
        super().__init__()
        dim, self.heads = cfg['hidden'], cfg['heads']
        self.width = dim//self.heads
        self.cfg, self.temporal = cfg, temporal
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3*dim)
        self.out = nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Dropout(cfg['dropout']), nn.Linear(4*dim, dim))
        self.pos = nn.Embedding(cfg['position_buckets'], self.heads)
        self.gap = nn.Embedding(cfg['position_buckets'], self.heads) if temporal else None
        nn.init.zeros_(self.pos.weight)
        if temporal:
            nn.init.zeros_(self.gap.weight)
        self.drop = nn.Dropout(cfg['dropout'])

    def forward(self, x, times, valid, memory=None):
        b, length, dim = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, length, 3, self.heads, self.width).unbind(2)
        q, k, v = [a.transpose(1, 2) for a in (q, k, v)]
        offset = 0
        if memory is not None:
            old_k, old_v, old_t, old_valid = memory
            offset = old_k.shape[2]
            k, v = torch.cat((old_k, k), 2), torch.cat((old_v, v), 2)
            key_time, key_valid = torch.cat((old_t, times), 1), torch.cat((old_valid, valid), 1)
        else:
            key_time, key_valid = times, valid
        qpos = offset+torch.arange(length, device=x.device)
        kpos = torch.arange(k.shape[2], device=x.device)
        distance = qpos[:, None]-kpos[None, :]
        bias = self.pos(bucket(distance, self.cfg['position_buckets'], self.cfg['max_bucket_distance']))
        bias = bias.permute(2, 0, 1)[None]
        if self.temporal:
            # Actual elapsed hours, with exact nearby and logarithmic distant buckets.
            gap = ((times[:, :, None]-key_time[:, None, :])*24).clamp_min(0)
            bias = bias + self.gap(bucket(gap, self.cfg['position_buckets'], self.cfg['max_bucket_distance'])).permute(0, 3, 1, 2)
        allowed = (distance >= 0)[None] & key_valid[:, None, :]
        # Padding queries attend to their own padding slot solely to avoid all -inf.
        allowed = allowed | ((~valid)[:, :, None] & (distance == 0)[None])
        bias = bias.masked_fill(~allowed[:, None], -float('inf'))
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                          dropout_p=self.cfg['dropout'] if self.training else 0)
        x = x+self.drop(self.out(h.transpose(1, 2).reshape(b, length, dim)))
        x = (x+self.drop(self.ff(self.norm2(x))))*valid[..., None]
        new_memory = tuple(z.detach() for z in (k, v, key_time, key_valid))
        return x, new_memory


class SequenceEncoder(nn.Module):
    def __init__(self, cfg, temporal):
        super().__init__()
        self.layers = nn.ModuleList([AttentionLayer(cfg, temporal) for _ in range(cfg['layers'])])
        self.final = nn.LayerNorm(cfg['hidden'])

    def forward(self, x, times, valid, memory=None):
        updated = []
        for i, layer in enumerate(self.layers):
            x, cache = layer(x, times, valid, None if memory is None else memory[i])
            updated.append(cache)
        return self.final(x)*valid[..., None], updated


# ---------------------------------------------------------------- the model
class MUPSR(nn.Module):
    # Parameters present in checkpoints written by earlier code but never used by this model.
    LEGACY_KEYS = ('att_vector.weight',)
    # State-dependence channels keyed by config flag (Sec. 3.9, 3.10): the parameters each one owns.
    STATE_FLAGS = {'use_recency': ('read_pos', 'read_gap'),
                   'use_repeat': ('rep_gap', 'rep_cnt', 'rep_gap_cat', 'rep_user')}

    def __init__(self, data, cfg, tables):
        super().__init__()
        self.cfg, self.dim, self.categories = cfg, cfg['hidden'], data.categories
        d = self.dim
        self.tau = cfg['tile_width']
        self.register_buffer('item_category', torch.tensor(data.cat.astype(np.int64)))
        self.item = nn.Embedding(len(data.cat), d, padding_idx=0)
        self.category = nn.Embedding(data.categories, d, padding_idx=0)
        self.price = PriceEncoder(data, d)
        self.time = nn.Linear(1, d, bias=False)
        self.fuse1, self.fuse2 = nn.Linear(3*d, d), nn.Linear(4*d, d)
        self.market_encoder = SequenceEncoder(cfg, temporal=False)
        self.user_encoder = SequenceEncoder(cfg, temporal=True)
        self.market_readout = nn.Linear(2*d+1, d)
        # Price head, Eq. (34): (a, b) = (a_bar, b_bar) + W (g - g_bar); k = eta * level_c + delta_c + h_k(g).
        self.register_buffer('center', torch.zeros(d))
        self.register_buffer('center_initialized', torch.tensor(False))
        self.ab_base = nn.Parameter(torch.full((2,), math.log(math.expm1(0.05))))
        self.ab_weight = nn.Parameter(torch.zeros(2, d))
        self.eta_raw = nn.Parameter(torch.tensor(math.log(math.expm1(0.05))))
        self.delta = nn.Parameter(torch.zeros(data.categories))
        self.hk = nn.Sequential(nn.Linear(d, d//2), nn.Tanh(), nn.Linear(d//2, 1))
        self.register_buffer('price_level', torch.tensor(
            (np.asarray(data.mean, np.float64)-data.meta['global_mean'])/data.meta['global_scale'], dtype=torch.float32))
        # Item bias b_j, centred within category in centered_item_bias().
        self.item_bias = nn.Embedding(len(data.cat), 1, padding_idx=0)
        self.register_buffer('category_size', torch.bincount(
            self.item_category, minlength=data.categories).clamp_min(1).float())
        self.match_group, self.match_user = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        # Readout, Eq. (38): scores only produce scalar weights, so they use width d_a; values keep width d.
        self.readout_dim = da = cfg.get('readout_dim', d)
        self.wh, self.wc, self.wgc = [nn.Linear(d, da, bias=False) for _ in range(3)]
        self.prior_logit = nn.Linear(d, 1, bias=False)
        self.wv, self.w0 = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.residual = nn.Sequential(nn.Linear(3*d, d), nn.Tanh(), nn.Linear(d, 3))
        self.day_token = nn.Linear(d+1, d)                                   # Eq. (30)
        # B2: query-relative position and time-gap biases of the readout, Eq. (38').
        nb = cfg['position_buckets']
        self.read_pos, self.read_gap = nn.Embedding(nb, 1), nn.Embedding(nb, 1)
        # B1: state-dependence S_uj, Eq. (41b) = global replenishment kernel + category deviation
        # (L2-shrunk) + count term + personal loyalty.
        self.rep_gap, self.rep_cnt = nn.Embedding(nb, 1), nn.Embedding(cfg.get('count_buckets', 16), 1)
        self.rep_gap_cat = nn.Embedding(self.categories, nb)
        self.rep_user = nn.Linear(d, 1, bias=False)
        for layer in (self.hk[-1], self.residual[-1], self.match_user):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.normal_(self.item.weight, std=0.02)
        nn.init.normal_(self.category.weight, std=0.02)
        with torch.no_grad():
            self.item.weight[0].zero_()
            self.category.weight[0].zero_()
        self.zero_state_terms()
        # Day-level tables (data.build_day_tables) and train events of the panel: GPU-resident, not checkpointed.
        self.T = tables['T']
        train = np.asarray(data.train_rows)
        ev = data.event_features(train, 'cpu')
        buffers = dict(ev_items=ev['items'], ev_logp=ev['logp'], ev_missing=ev['missing'],
                       ev_delta=torch.as_tensor(np.array(data.events['cat_delta'][train])),
                       cell=torch.as_tensor(tables['cell']), cell_count=torch.as_tensor(tables['count']),
                       active=torch.as_tensor(tables['active']), src=torch.as_tensor(tables['src']),
                       lag=torch.as_tensor(tables['lag']), ref=torch.as_tensor(tables['ref']),
                       price_z=torch.as_tensor(np.nan_to_num(np.array(data.a['price_snapshot_z']),
                                                             nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32),
                       eligible=torch.as_tensor(tables['eligible']))
        for name, value in buffers.items():
            self.register_buffer(name, value, persistent=False)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict = {k: v for k, v in state_dict.items() if k not in self.LEGACY_KEYS}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @torch.no_grad()
    def zero_state_terms(self):
        """Stage-2 state-dependence parameters start at zero, so the switch leaves the score unchanged."""
        for m in (self.read_pos, self.read_gap, self.rep_gap, self.rep_cnt, self.rep_gap_cat, self.rep_user):
            nn.init.zeros_(m.weight)

    @torch.no_grad()
    def apply_fe_init(self, init):
        """A1: closed-form Poisson fixed-effect starting values (features.fe_init)."""
        self.item_bias.weight[:, 0].copy_(torch.as_tensor(init['item_bias']))
        self.delta.copy_(torch.as_tensor(init['delta']))
        self.eta_raw.fill_(math.log(math.expm1(max(init['eta'], 1e-4))))

    # ---------------- shared event representation, Eqs. (29), (35)
    def event_vectors(self, features, stage):
        items = features['items']
        c = self.item_category[items]
        ec = self.category(c)
        price = self.price(features['logp'], features['missing'], c, ec)
        et = self.time(torch.log1p(features['delta'].clamp_min(0))[..., None])
        parts = [self.item(items), price, et]
        if stage == 2:
            parts.insert(0, ec)
        return torch.tanh((self.fuse1 if stage == 1 else self.fuse2)(torch.cat(parts, -1)))

    # ---------------- stage 1: day-level category panel, Eqs. (29)-(32)
    def market_panel(self):
        """h_{c,d}: causal self-attention over daily category tokens; positions are calendar days."""
        f = {'items': self.ev_items, 'logp': self.ev_logp, 'missing': self.ev_missing, 'delta': self.ev_delta}
        x = self.event_vectors(f, 1)
        C, T, d = self.categories, self.T, self.dim
        pooled = x.new_zeros(C*T, d).index_add(0, self.cell, x)/self.cell_count.clamp_min(1)[:, None]
        token = torch.tanh(self.day_token(torch.cat((pooled, torch.log1p(self.cell_count)[:, None]), -1)))
        steps = torch.arange(T, device=x.device, dtype=torch.float32).expand(C, T)
        h, _ = self.market_encoder(token.view(C, T, d), steps, self.active)
        return h                                                              # (C, T, d)

    def day_states(self, h):
        """g_c(d^-) for every protocol day: state of day d-1 (frozen after training), Eq. (32)."""
        ok = (self.src >= 0).to(h.dtype)[:, None, None]
        prev = h[:, self.src.clamp_min(0)].transpose(0, 1)*ok                # (D, C, d)
        ec = self.category.weight[None].expand_as(prev)
        return torch.tanh(self.market_readout(torch.cat((prev, ec, self.lag[..., None]), -1)))

    @torch.no_grad()
    def update_center(self, g):
        value = g[:, 1:].mean((0, 1))
        if not bool(self.center_initialized):
            self.center.copy_(value)
            self.center_initialized.fill_(True)
        else:
            self.center.lerp_(value, self.cfg['center_momentum'])

    def day_terms(self, g):
        """Every term of Eqs. (34), (38)-(41') that depends only on (day, category); computed once per day.
        The cross-category mean of h_k is removed (A6, Eq. 40'); the constant cancels in Eq. (12)."""
        ec = self.category.weight[None].expand_as(g)
        hk = self.hk(g).squeeze(-1)
        hk = hk - hk[..., 1:].mean(-1, keepdim=True)
        return dict(raw=self.ab_base+F.linear(g-self.center, self.ab_weight),
                    k=F.softplus(self.eta_raw)*self.price_level+self.delta+hk,
                    match=self.match_group(g), reference=self.ref,
                    query=(self.wc(self.category.weight)[None]+self.wgc(g))*self.readout_dim**-0.5,
                    prior=self.w0(g), prior_logit=self.prior_logit(g).squeeze(-1),
                    res_base=F.linear(torch.cat((ec, g), -1), self.residual[0].weight[:, self.dim:], self.residual[0].bias))

    def query_context(self, terms, day, hu=None):
        """Context of Eq. (27') for queries on `day` (Q,); hu = None gives the stage-1 (category-only)
        context. Delta k is centred over categories (A6). Also returns gamma for downstream use."""
        raw, k, match = terms['raw'][day], terms['k'][day], terms['match'][day]
        if hu is not None:
            prior = terms['prior'][day]
            hidden = torch.tanh(F.linear(hu, self.residual[0].weight[:, :self.dim])+terms['res_base'][day])
            residual = self.residual[2](hidden)
            dk = residual[..., 2]
            raw, k = raw+residual[..., :2], k+dk-dk[..., 1:].mean(-1, keepdim=True)
            match = match+self.match_user(hu-prior)
        beta, gamma = F.softplus(raw).unbind(-1)
        if self.cfg.get('nested_gamma_zero', False):
            gamma = torch.zeros_like(gamma)
        return Context(match, beta+gamma, gamma*terms['reference'][day]-k, self.centered_item_bias()), gamma

    def centered_item_bias(self):
        """b_j minus its category mean: the category level is carried by k alone."""
        b = self.item_bias.weight[:, 0]
        total = b.new_zeros(self.categories).index_add(0, self.item_category, b)
        return b-(total/self.category_size)[self.item_category]

    # ---------------- stage 2: block-pair user encoder and bilinear readout, Eqs. (35)-(40)
    def encode_block_pair(self, prev, cur):
        """Current block attends to itself and the previous block (one-segment memory, Dai et al., 2019)."""
        with torch.no_grad():
            hp, memory = self.user_encoder(self.event_vectors(prev, 2), prev['time'], prev['mask'])
        hc, _ = self.user_encoder(self.event_vectors(cur, 2), cur['time'], cur['mask'], memory)
        return hp, hc

    def readout_bias(self, prev, cur, qp, qtime, qs):
        """B2 (Eq. 38'): (Q, K) bias from each key's distance to the query position and its elapsed hours to
        the query time. The previous block, when present, has exactly W events: its key i sits at i - W."""
        W, nb, mx = self.cfg['user_block'], self.cfg['position_buckets'], self.cfg['max_bucket_distance']
        dev = qp.device
        rel = torch.cat((torch.arange(prev['items'].shape[1], device=dev) - W,
                         torch.arange(cur['items'].shape[1], device=dev)))
        dist = (qp[:, None] - rel[None]).clamp_min(0)
        key_t = torch.cat((prev['time'], cur['time']), 1)[qs]
        gap = ((qtime[:, None] - key_t) * 24).clamp_min(0)
        return (self.read_pos(bucket(dist, nb, mx)) + self.read_gap(bucket(gap, nb, mx))).squeeze(-1)

    def readout(self, keys_h, allowed, qs, terms, day, chunk=64, rbias=None):
        """Eqs. (38')-(39) for Q queries. keys_h (S, K, d): [previous block | current block] states of each
        sample; qs (Q,) sample of each query; allowed (Q, K) readable keys; rbias (Q, K) the B2 recency
        bias. The prior slot is always readable, so an empty history returns the prior exactly."""
        k, v = self.wh(keys_h)[qs], self.wv(keys_h)[qs]                        # (Q, K, da), (Q, K, d)
        query, q0, prior = terms['query'][day], terms['prior_logit'][day], terms['prior'][day]
        if rbias is None:
            rbias = k.new_zeros(allowed.shape)
        def part(query, q0, prior, k, v, rbias):
            s = (torch.einsum('qcx,qkx->qck', query, k) + rbias[:, None]).masked_fill(~allowed[:, None], -float('inf'))
            w = torch.softmax(torch.cat((s, q0[..., None]), -1), -1)
            return torch.einsum('qck,qkd->qcd', w[..., :-1], v)+w[..., -1:]*prior
        out = []
        for s in range(0, self.categories, chunk):
            args = (query[:, s:s+chunk], q0[:, s:s+chunk], prior[:, s:s+chunk], k, v, rbias)
            out.append(checkpoint(part, *args, use_reentrant=False) if torch.is_grad_enabled() else part(*args))
        return torch.cat(out, 1)                                              # (Q, C, d)

    def repeat_bonus(self, rep, terms, day, hu, inverse, n_slots):
        """B1 (Eq. 41b): the state-dependence term S_uj for the (query, item) pairs of the user's history,
        written into tile slots so the softmax denominator and the positive numerator read the same tensor.
        rep = (rq, item, gap_days, count, last_logp, last_ok) as produced by features.RepeatIndex.gather."""
        rq, it, gap_days, cnt, _, _ = rep
        c = self.item_category[it]
        nb, mx = self.cfg['position_buckets'], self.cfg['max_bucket_distance']
        gb = bucket(gap_days * 24, nb, mx)                                    # same basket = bucket 0
        cb = torch.log2(cnt.float()).floor().long().clamp(0, self.rep_cnt.num_embeddings - 1)
        v = self.rep_gap(gb).squeeze(-1) + self.rep_gap_cat.weight[c, gb] + self.rep_cnt(cb).squeeze(-1)
        if hu is not None:
            v = v + self.rep_user(hu[rq, c]).squeeze(-1)
        flat = inverse[it, 0] * self.tau + inverse[it, 1]
        return v.new_zeros(len(day), n_slots).index_put((rq, flat), v, accumulate=True)

    # ---------------- training utilities
    def regularizer(self):
        return (self.cfg['price_l2']*self.ab_weight.square().sum()
                + self.cfg['category_l2']*self.delta[1:].square().sum()
                + self.cfg['category_l2']*self.rep_gap_cat.weight[1:].square().sum())

    @torch.no_grad()
    def initialize_stage2(self):
        """Sec. 3.10: user encoder copies the panel encoder; personal and state-dependence terms start at
        zero, so the stage-2 scores equal the stage-1 scores at the switch."""
        state = self.market_encoder.state_dict()
        self.user_encoder.load_state_dict(state, strict=False)
        self.fuse2.weight.zero_()
        self.fuse2.weight[:, self.dim:].copy_(self.fuse1.weight)
        self.fuse2.bias.copy_(self.fuse1.bias)
        nn.init.zeros_(self.match_user.weight)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.zero_state_terms()

    def configure_stage(self, stage):
        for p in self.parameters():
            p.requires_grad_(True)
        if self.cfg.get('nested_no_price_heterogeneity', False):
            with torch.no_grad():
                self.ab_weight.zero_()
            self.ab_weight.requires_grad_(False)
        if stage == 2:                                       # panel encoder output is stop-gradient in stage 2
            for module in (self.market_encoder, self.fuse1, self.day_token):
                for p in module.parameters():
                    p.requires_grad_(False)
        for flag, names in self.STATE_FLAGS.items():
            if not self.cfg.get(flag, True):
                for name in names:
                    getattr(self, name).weight.requires_grad_(False)


def query_keys(hp, hc, prev, cur, qs, qp):
    """Readable keys of each query: the whole previous block and current-block positions j < p, Eq. (37)."""
    before = torch.arange(hc.shape[1], device=hc.device)[None] < qp[:, None]
    allowed = torch.cat((prev['mask'][qs], cur['mask'][qs] & before), 1)
    return torch.cat((hp, hc), 1), allowed


# ---------------------------------------------------------------- catalogue tiles and likelihood
def tile_layout(item_category, tau):
    """Items grouped by category into fixed-width tiles that never cross a category.
    Returns tile categories, tile items (0 = padding) and, per item, its (tile, slot)."""
    cat = np.asarray(item_category)
    tc, ti = [], []
    for c in np.unique(cat[1:]):
        items = np.flatnonzero(cat == c)
        items = items[items > 0]
        for s in range(0, len(items), tau):
            block = items[s:s+tau]
            tc.append(c)
            ti.append(np.r_[block, np.zeros(tau-len(block), np.int64)])
    tc, ti = np.array(tc, np.int64), np.array(ti, np.int64)
    inverse = np.full((len(cat), 2), -1, np.int64)
    t, k = np.nonzero(ti)
    inverse[ti[t, k]] = np.stack([t, k], 1)
    return tc, ti, inverse


def build_tiles(item_category, tau, device):
    tc, ti, inverse = tile_layout(item_category, tau)
    return Tiles(torch.as_tensor(tc, device=device), torch.as_tensor(ti, device=device)), \
        torch.as_tensor(inverse, device=device)


def tile_scores(model, ctx, day, tiles, start, stop, bonus=None):
    """Eq. (27') for every item of tiles[start:stop]; ineligible (day, item) pairs -> EXCLUDED.
    `bonus`, when given, is the densified state-dependence tensor (Q, n_tiles*tau)."""
    tc, ti = tiles.cat[start:stop], tiles.items[start:stop]
    m = torch.einsum('qtd,tkd->qtk', ctx.match[:, tc], model.item(ti))
    z = model.price_z[day[:, None, None], ti[None]]
    s = m+ctx.bias[ti][None]-ctx.slope[:, tc, None]*z+ctx.intercept[:, tc, None]
    if bonus is not None:
        s = s + bonus[:, start * model.tau:stop * model.tau].view(len(day), -1, model.tau)
    return s.masked_fill(~model.eligible[day[:, None, None], ti[None]], EXCLUDED)


def point_scores(model, ctx, rows, items, day, inverse=None, bonus=None):
    c = model.item_category[items]
    z = model.price_z[day, items]
    s = ((ctx.match[rows, c]*model.item(items)).sum(-1)+ctx.bias[items]
         - ctx.slope[rows, c]*z+ctx.intercept[rows, c])
    if bonus is not None:
        s = s + bonus[rows, inverse[items, 0]*model.tau + inverse[items, 1]]
    return s


def log_partition(model, ctx, day, tiles, chunk=256, bonus=None):
    logz = None
    for s in range(0, len(tiles.cat), chunk):
        part = torch.logsumexp(tile_scores(model, ctx, day, tiles, s, s+chunk, bonus).flatten(1), -1)
        logz = part if logz is None else torch.logaddexp(logz, part)
    return logz


def catalog_loss_days(model, ctx, day, targets, tiles, inverse=None, chunk=256, bonus=None, weight=None):
    """Eq. (12) softmax of each query over its own day's eligible pool.
    A0 (Sec. 4): a target outside its day's eligible pool would enter the numerator but not the
    denominator, so it is rejected before the likelihood is formed."""
    if not bool(model.eligible[day, targets].all()):
        raise ValueError("Training target outside its day's eligible pool.")
    rows = torch.arange(len(day), device=day.device)
    nll = (log_partition(model, ctx, day, tiles, chunk, bonus)
           - point_scores(model, ctx, rows, targets, day, inverse, bonus))
    return nll.mean() if weight is None else (nll * weight).sum() / weight.sum()


def stage1_count_loss(model, ctx, days, label_index, label_count, tiles):
    """Stage-1 likelihood in count form: sum_d [n_d logZ_d - sum_j y_dj s_dj] / sum y, Eq. (33)."""
    row_of_day = torch.full((model.eligible.shape[0],), -1, dtype=torch.long, device=days.device)
    row_of_day[days] = torch.arange(len(days), device=days.device)
    rows = row_of_day[label_index[:, 0]]
    keep = rows >= 0
    rows, items, y = rows[keep], label_index[keep, 1], label_count[keep]
    n = torch.zeros(len(days), device=days.device).index_add(0, rows, y)
    logz = log_partition(model, ctx, days, tiles)
    return ((n*logz).sum()-(y*point_scores(model, ctx, rows, items, days[rows])).sum())/y.sum()


@torch.no_grad()
def catalog_scores(model, ctx, day, tiles, chunk=256, bonus=None):
    """Scores of the whole tiled catalogue (Q, n_tiles*tau); ineligible slots are -inf."""
    s = torch.cat([tile_scores(model, ctx, day, tiles, a, a+chunk, bonus) for a in range(0, len(tiles.cat), chunk)], 1)
    return s.masked_fill(s <= EXCLUDED/2, -float('inf')).flatten(1)


def slot_index(items, tiles, inverse):
    """Column of each item in the flattened output of catalog_scores."""
    return inverse[items, 0]*tiles.items.shape[1]+inverse[items, 1]
