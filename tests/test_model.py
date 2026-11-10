"""Tests of the optimized estimation and the state-dependence terms (method.md; T1-T13), on CPU.

    python -m unittest discover -s tests -v
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]

import synthetic
from mupsr import train
from mupsr.data import EventData, build_day_tables, make_eval_batch, user_gaps
from mupsr.diagnostics import run as run_diagnostics
from mupsr.features import RepeatIndex, popularity_stock, fe_init
from mupsr.model import (MUPSR, build_tiles, catalog_loss_days, catalog_scores, point_scores,
                         query_keys, stage1_count_loss)
from mupsr.significance import by_stratum, compare
from mupsr.train import stage2_forward, transfer

CFG = json.loads((ROOT / 'configs' / 'main.json').read_text())
CFG.pop('expected_protocol', None)
CFG.update(hidden=16, heads=2, readout_dim=4, user_block=8, tokens_per_batch=64, market_refresh=4, readout_chunk=3,
           tile_width=8, tile_chunk=4, eval_batch=5, stage1_days_per_step=4, threads=1, stage1_epochs=4,
           stage1_eval_every=2, stage2_epochs=2, warmup_steps=5, repeat_cap=16)
HALF_LIVES = (1.0, 7.0, 28.0)


class FeatureTests(unittest.TestCase):
    """T1-T5: NumPy components against brute force."""

    def setUp(self):
        rng = self.rng = np.random.default_rng(0)
        n, I, U, D = 3000, 60, 50, 30
        self.time = np.sort(rng.uniform(0, D, n)); self.time[100:110] = self.time[100]
        self.day = np.floor(self.time).astype(int)
        self.user, self.item = rng.integers(0, U, n), rng.integers(1, I, n)
        self.logp, self.missing = rng.normal(size=n).astype(np.float32), rng.random(n) < .05
        o = np.argsort(self.user, kind='stable'); cuts = np.r_[0, np.flatnonzero(np.diff(self.user[o])) + 1, n]
        self.user_rows = {int(self.user[o[l]]): o[l:r] for l, r in zip(cuts[:-1], cuts[1:])}
        self.n, self.I, self.D = n, I, D

    def test_T1_T3_repeat_index(self):
        for cap in (3, 1000):
            idx = RepeatIndex(self.item, self.time, self.logp, self.missing, self.user_rows, cap)
            for r in self.rng.choice(self.n, 200, replace=False):
                rows = self.user_rows[int(self.user[r])]; prev = rows[:np.searchsorted(rows, r)]
                last = {int(self.item[p]): p for p in prev}
                ranked = sorted(last, key=lambda j: last[j])[::-1][:cap]
                _, it, gap, cnt, lp, lm = idx.gather([r])
                self.assertEqual(list(it), ranked)
                for j, g, c, l, m in zip(it, gap, cnt, lp, lm):
                    self.assertAlmostEqual(float(g), float(self.time[r] - self.time[last[j]]), places=4)
                    self.assertEqual(c, (self.item[prev] == j).sum())
                    self.assertEqual((l, m), (self.logp[last[j]], self.missing[last[j]]))
        idx = RepeatIndex(self.item, self.time, self.logp, self.missing, self.user_rows, 5)
        qs = self.rng.choice(self.n, 40, replace=False); batch = idx.gather(qs)
        for k, r in enumerate(qs):
            one = idx.gather([r]); sel = batch[0] == k
            self.assertTrue(all(np.array_equal(x[sel], y) for x, y in zip(batch[1:], one[1:])))

    def test_T2_repeat_index_ignores_future(self):
        r = self.n // 2
        item2 = self.item.copy(); later = np.arange(self.n) > r; item2[later] = self.rng.integers(1, self.I, later.sum())
        a = RepeatIndex(self.item, self.time, self.logp, self.missing, self.user_rows).gather([r])
        b = RepeatIndex(item2, self.time, self.logp, self.missing, self.user_rows).gather([r])
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))

    def test_T4_popularity_stock(self):
        train_rows = np.flatnonzero(self.rng.random(self.n) < .8)
        X = popularity_stock(train_rows, self.item, self.time, self.day, self.I, self.D, (1.0, 7.0))
        for d in (0, 1, 5, 17, 29):
            for j in (1, 7, 33):
                m = train_rows[(self.item[train_rows] == j) & (self.day[train_rows] < d)]
                for k, h in enumerate((1.0, 7.0)):
                    self.assertAlmostEqual(float(X[d, j, k]), float(np.log1p((2.0 ** (-(d - self.time[m]) / h)).sum())), places=4)

    def test_T5_fe_init(self):
        rng, D, I, C = self.rng, 30, 60, 6
        cat = np.r_[0, rng.integers(1, C, I - 1)]
        true_b = rng.normal(0, 1.5, I)
        elig = rng.random((D, I)) < .7; elig[:, 0] = False; elig[25:, 7] = True; elig[:25, 7] = False
        counts = np.zeros((D, I))
        for d in range(25):
            p = np.exp(true_b) * elig[d]; p /= p.sum(); counts[d] = rng.multinomial(400, p)
        li = np.argwhere(counts > 0); lc = counts[counts > 0]
        init = fe_init(elig, li, lc, cat, rng.normal(size=C))
        self.assertTrue(np.isfinite(init['item_bias']).all())
        k = -np.bincount(cat[1:], weights=init['rho'][1:], minlength=C) / np.bincount(cat[1:], minlength=C).clip(min=1)
        np.testing.assert_allclose(init['item_bias'][1:] - k[cat[1:]], init['rho'][1:], atol=1e-5)
        ever = elig[:25].any(0); ever[0] = False
        self.assertGreater(np.corrcoef(init['rho'][ever], true_b[ever])[0, 1], .95)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = Path(tempfile.mkdtemp())
        synthetic.make(cls.tmp / 'syn')
        synthetic.make(cls.tmp / 'raw', filter_train_queries=False)
        cls.data = EventData(cls.tmp / 'syn/prepared', cls.tmp / 'syn/csv', cls.tmp / 'syn/cache')
        cls.tables = build_day_tables(cls.data)
        cls.rep = RepeatIndex(cls.data.item, cls.data.time, np.asarray(cls.data.events['logp']),
                              np.asarray(cls.data.events['missing']), cls.data.user_rows, CFG['repeat_cap'])
        cls.tiles, cls.inverse = build_tiles(cls.data.cat, CFG['tile_width'], 'cpu')
        cls.gap = user_gaps(cls.data)
        cls.X = popularity_stock(cls.data.train_rows, cls.data.item, cls.data.time,
                                 np.asarray(cls.data.a['day_time']), len(cls.data.cat), cls.tables['D'], HALF_LIVES)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def model(self, cfg=CFG, random_new_terms=True, seed=0):
        torch.manual_seed(seed)
        m = MUPSR(self.data, cfg, self.tables).eval()
        if random_new_terms:
            with torch.no_grad():
                for p in (m.read_pos.weight, m.read_gap.weight, m.rep_gap.weight, m.rep_cnt.weight,
                          m.rep_gap_cat.weight, m.rep_user.weight, m.match_user.weight,
                          m.residual[-1].weight, m.hk[-1].weight):
                    p.normal_(0, .3)
        return m

    def batch(self, split='test', n=12):
        return make_eval_batch(self.data, np.asarray(self.data.query_rows[split])[:n], CFG['user_block'],
                               self.gap, self.rep, 'cpu')

    def test_T7_stage_transfer_is_exact_with_all_terms(self):
        m = self.model(random_new_terms=False)
        with torch.no_grad():
            m.hk[-1].weight.normal_(0, .3)                                   # stage-1 terms may be anything
        res = transfer(m, self.data, self.tiles, self.inverse, self.gap, self.rep, CFG, 'cpu')
        self.assertLessEqual(res['max_abs_error'], 1e-5)

    def test_state_flags_off_disable_the_extra_terms(self):
        cfg = {**CFG, 'use_recency': False, 'use_repeat': False}
        m = MUPSR(self.data, cfg, self.tables).eval()
        with torch.no_grad():                                               # the owned params stay at zero and frozen
            self.assertTrue(bool((m.read_pos.weight == 0).all()) and bool((m.rep_gap.weight == 0).all()))
        m.configure_stage(2)
        for name in ('read_pos', 'read_gap', 'rep_gap', 'rep_cnt', 'rep_gap_cat', 'rep_user'):
            self.assertFalse(getattr(m, name).weight.requires_grad)
        with torch.no_grad():
            out = self.batch('valid')
            ctx, bonus, *_ = stage2_forward(m, out, m.market_panel(), self.tiles, self.inverse, cfg)
        self.assertIsNone(bonus)                                            # no state-dependence tensor at all

    def test_T9_tile_and_point_paths_agree(self):
        m = self.model()
        with torch.no_grad():
            ctx, bonus, day, target, _ = stage2_forward(m, self.batch(), m.market_panel(), self.tiles, self.inverse, CFG)
            s = catalog_scores(m, ctx, day, self.tiles, bonus=bonus)
            col = self.inverse[target, 0] * self.tiles.items.shape[1] + self.inverse[target, 1]
            point = point_scores(m, ctx, torch.arange(len(day)), target, day, self.inverse, bonus)
            ok = m.eligible[day, target]
            torch.testing.assert_close(s.gather(1, col[:, None])[:, 0][ok], point[ok])

    def test_T10_count_loss_equals_record_loss(self):
        m = self.model()
        li, lc = self.tables['label_index'], self.tables['label_count']
        days = torch.as_tensor(np.unique(li[:, 0])[:3])
        with torch.no_grad():
            terms = m.day_terms(m.day_states(m.market_panel()))
            ctx, _ = m.query_context(terms, days)
            count = stage1_count_loss(m, ctx, days, torch.as_tensor(li), torch.as_tensor(lc), self.tiles)
            keep = np.isin(li[:, 0], days.numpy())
            d = torch.as_tensor(np.repeat(li[keep, 0], lc[keep].astype(int)))
            j = torch.as_tensor(np.repeat(li[keep, 1], lc[keep].astype(int)))
            c2, _ = m.query_context(terms, d)
            record = catalog_loss_days(m, c2, d, j, self.tiles, self.inverse)
        torch.testing.assert_close(count, record, rtol=1e-5, atol=1e-5)

    def test_T11_readout_bias_keeps_causality_and_prior(self):
        m = self.model()
        with torch.no_grad():
            terms = m.day_terms(m.day_states(m.market_panel()))
            prev, cur, qs, qp, day, target, qtime, rep, qrow = self.batch('test', 20)
            hp, hc = m.encode_block_pair(prev, cur)
            keys, allowed = query_keys(hp, hc, prev, cur, qs, qp)
            rb = m.readout_bias(prev, cur, qp, qtime, qs)
            a = m.readout(keys, allowed, qs, terms, day, 3, rb)
            b = m.readout(keys, allowed, qs, terms, day, 3, rb.masked_fill(~allowed, 1e3))
            torch.testing.assert_close(a, b)                         # bias on unreadable keys has no effect
            empty = ~allowed.any(1)
            if empty.any():
                torch.testing.assert_close(a[empty], terms['prior'][day[empty]])

    def test_T12_A6_constant_shift_does_not_change_the_softmax(self):
        m = self.model()
        with torch.no_grad():
            out = self.batch()
            h = m.market_panel()
            ctx, bonus, day, target, _ = stage2_forward(m, out, h, self.tiles, self.inverse, CFG)
            m.residual[-1].bias[2] += 7.0; m.hk[-1].bias += 3.0
            ctx2, bonus2, *_ = stage2_forward(m, out, h, self.tiles, self.inverse, CFG)
            s1 = catalog_scores(m, ctx, day, self.tiles, bonus=bonus)
            s2 = catalog_scores(m, ctx2, day, self.tiles, bonus=bonus2)
            torch.testing.assert_close(torch.log_softmax(s1, 1).nan_to_num(neginf=0), torch.log_softmax(s2, 1).nan_to_num(neginf=0))

    def test_T13_dense_bonus_and_exclusion(self):
        m = self.model()
        with torch.no_grad():
            prev, cur, qs, qp, day, target, qtime, rep, qrow = out = self.batch('test', 20)
            ctx, bonus, *_ = stage2_forward(m, out, m.market_panel(), self.tiles, self.inverse, CFG)
            rq, it = rep[0], rep[1]
            flat = self.inverse[it, 0] * m.tau + self.inverse[it, 1]
            self.assertEqual(len(set(zip(rq.tolist(), it.tolist()))), len(rq))          # no duplicate pairs
            nonzero = torch.zeros_like(bonus, dtype=torch.bool); nonzero[rq, flat] = True
            self.assertTrue(bool((bonus[~nonzero] == 0).all()))
            s = catalog_scores(m, ctx, day, self.tiles, bonus=bonus)
            inelig = ~m.eligible[day[rq], it]
            self.assertTrue(bool(torch.isinf(s[rq[inelig], flat[inelig]]).all()))

    def test_nested_model_flags(self):
        cfg = {**CFG, 'nested_gamma_zero': True, 'nested_no_price_heterogeneity': True}
        m = self.model(cfg)
        with torch.no_grad():
            m.ab_weight.normal_()
        m.configure_stage(1)
        self.assertTrue(bool((m.ab_weight == 0).all()) and not m.ab_weight.requires_grad)
        with torch.no_grad():
            terms = m.day_terms(m.day_states(m.market_panel()))
            day = torch.tensor([5, 9])
            ctx, gamma = m.query_context(terms, day)
            beta = torch.nn.functional.softplus(terms['raw'][day][..., 0])
            torch.testing.assert_close(ctx.slope, beta)                       # slope = beta + 0
            torch.testing.assert_close(ctx.intercept, -terms['k'][day])        # no reference-price term
            self.assertTrue(bool((gamma == 0).all()))

    def test_A0_guard_and_filter(self):
        raw = EventData(self.tmp / 'raw/prepared', self.tmp / 'raw/csv', self.tmp / 'raw/cache')
        tb = build_day_tables(raw)
        q = np.asarray(raw.query_rows['train'])
        outside = ~tb['eligible'][np.asarray(raw.a['day_time'])[q], raw.item[q]]
        self.assertGreater(outside.sum(), 0)
        m = MUPSR(raw, CFG, tb)
        tiles, inverse = build_tiles(raw.cat, CFG['tile_width'], 'cpu')
        d = torch.as_tensor(np.asarray(raw.a['day_time'])[q[outside][:3]]); t = torch.as_tensor(raw.item[q[outside][:3]])
        terms = m.day_terms(m.day_states(m.market_panel()))
        ctx, _ = m.query_context(terms, d)
        with self.assertRaises(ValueError):
            catalog_loss_days(m, ctx, d, t, tiles, inverse)

    def test_end_to_end_train_diagnostics_significance(self):
        cfg_path = self.tmp / 'cfg.json'; cfg_path.write_text(json.dumps(CFG))
        outs = []
        for name, src in (('a', 'syn'), ('b', 'raw')):                 # 'raw' also exercises A0 end to end
            out = self.tmp / f'run_{name}'
            argv = ['--config', str(cfg_path), '--data', str(self.tmp / src / 'prepared'), '--source', str(self.tmp / src / 'csv'),
                    '--cache', str(self.tmp / src / 'cache'), '--output', str(out), '--device', 'cpu']
            self.assertEqual(train.main(train.parse(argv)), 0)
            self.assertLessEqual(json.loads((out / 'transfer.json').read_text())['max_abs_error'], 1e-4)
            self.assertTrue(all(np.isfinite(e['loss']) for e in json.loads((out / 'training_log.json').read_text())))
            outs.append(out)
        env_b = json.loads((outs[1] / 'environment.json').read_text())
        self.assertGreater(env_b['train_queries_excluded_outside_pool'], 0)
        a = np.load(outs[0] / 'per_query_N10_test.npy')
        self.assertEqual(len(a), len(self.data.query_rows['test']))
        res = compare(a, a + .01)
        self.assertGreater(res['rel_gain'], 0)
        strata = {}
        diag = run_diagnostics(self.data, self.tables, self.X, self.rep, CFG['user_block'], strata)
        self.assertTrue(0 <= diag['test']['repeat_share'] <= 1)
        self.assertEqual(len(strata['test']['history_len']), len(a))
        self.assertTrue(all(v['rel_gain'] > 0 for v in by_stratum(a, a + .01, strata['test'], 200).values()))


if __name__ == '__main__':
    unittest.main()
