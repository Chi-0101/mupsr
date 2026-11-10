"""Structural tests of MUPSR and an end-to-end CPU run on synthetic data.

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
from mupsr.features import RepeatIndex
from mupsr.metrics import ranking_metrics
from mupsr.model import (Context, MUPSR, SequenceEncoder, build_tiles, catalog_loss_days, catalog_scores,
                         query_keys, slot_index, stage1_count_loss)


def small_cfg():
    cfg = json.loads((ROOT / 'configs' / 'main.json').read_text())
    cfg.pop('expected_protocol', None)
    cfg.update(hidden=16, heads=2, readout_dim=4, user_block=8, tokens_per_batch=64, market_refresh=4,
               readout_chunk=3, tile_width=8, tile_chunk=4, eval_batch=5, stage1_days_per_step=4, threads=1,
               stage1_epochs=4, stage1_eval_every=2, stage2_epochs=2, warmup_steps=5, repeat_cap=16)
    return cfg


def reference_metrics(scores, target, ks=(5, 10, 20)):
    """Brute force: expected metric over all orderings of the target's tie group."""
    g, e = int((scores > scores[target]).sum()), int((scores == scores[target]).sum())
    out = {}
    for k in ks:
        ranks = np.arange(g + 1, g + e + 1)
        out[f'R@{k}'] = float(np.mean(ranks <= k))
        out[f'N@{k}'] = float(np.mean(np.where(ranks <= k, 1 / np.log2(ranks + 1), 0)))
    return out


class MUPSRCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = Path(tempfile.mkdtemp())
        synthetic.make(cls.tmp / 'syn')
        cls.data = EventData(cls.tmp / 'syn/prepared', cls.tmp / 'syn/csv', cls.tmp / 'syn/cache')
        cls.tables = build_day_tables(cls.data)
        cls.cfg = small_cfg()
        cls.gap = user_gaps(cls.data)
        cls.rep = RepeatIndex(cls.data.item, cls.data.time, np.asarray(cls.data.events['logp']),
                              np.asarray(cls.data.events['missing']), cls.data.user_rows, cls.cfg['repeat_cap'])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def model(self, seed=0):
        torch.manual_seed(seed)
        return MUPSR(self.data, self.cfg, self.tables).eval()

    def test_encoder_is_causal(self):
        enc = SequenceEncoder(self.cfg, temporal=True).eval()
        x, t = torch.randn(2, 11, 16), torch.arange(11.)[None].expand(2, -1)
        valid = torch.ones(2, 11, dtype=torch.bool)
        changed = x.clone(); changed[:, 6:] += 100
        torch.testing.assert_close(enc(x, t, valid)[0][:, :6], enc(changed, t, valid)[0][:, :6])

    def test_market_information_set_is_previous_days(self):
        src = self.tables['src']
        self.assertTrue(np.all(src[:self.tables['T']] == np.arange(self.tables['T']) - 1))

    def test_count_loss_equals_record_loss(self):
        m = self.model()
        li, lc = self.tables['label_index'], self.tables['label_count']
        days = torch.as_tensor(np.unique(li[:, 0])[:3])
        tiles, inverse = build_tiles(self.data.cat, 8, 'cpu')
        with torch.no_grad():
            terms = m.day_terms(m.day_states(m.market_panel()))
            ctx, _ = m.query_context(terms, days)
            count = stage1_count_loss(m, ctx, days, torch.as_tensor(li), torch.as_tensor(lc), tiles)
            keep = np.isin(li[:, 0], days.numpy())
            rec_day = torch.as_tensor(np.repeat(li[keep, 0], lc[keep].astype(int)))
            rec_item = torch.as_tensor(np.repeat(li[keep, 1], lc[keep].astype(int)))
            c2, _ = m.query_context(terms, rec_day)
            record = catalog_loss_days(m, c2, rec_day, rec_item, tiles, inverse)
        torch.testing.assert_close(count, record, rtol=1e-5, atol=1e-5)

    def test_metrics_match_brute_force_with_ties(self):
        rng = np.random.default_rng(0)
        scores = torch.as_tensor(rng.integers(0, 4, (30, 40)).astype(np.float64))
        scores[:, -5:] = -float('inf')
        target = torch.as_tensor(rng.integers(0, 35, 30))
        got = ranking_metrics(scores, target)
        for q in range(30):
            for name, value in reference_metrics(scores[q].numpy(), int(target[q])).items():
                self.assertAlmostEqual(float(got[name][q]), value, places=12)

    def test_empty_history_readout_is_prior(self):
        m = self.model()
        rows = np.asarray(self.data.query_rows['test'])
        lengths = np.array([len(self.data.prefix(r)) for r in rows])
        q = rows[lengths == 0][:2] if np.any(lengths == 0) else None
        if q is None:
            self.skipTest('no empty-history query in the synthetic data')
        with torch.no_grad():
            terms = m.day_terms(m.day_states(m.market_panel()))
            out = make_eval_batch(self.data, q, 8, self.gap, self.rep, 'cpu')
            prev, cur, qs, qp, day = out[0], out[1], out[2], out[3], out[4]
            keys, allowed = query_keys(*m.encode_block_pair(prev, cur), prev, cur, qs, qp)
            torch.testing.assert_close(m.readout(keys, allowed, qs, terms, day, 3), terms['prior'][day])

    def test_scores_are_eligible_pool_only(self):
        m = self.model()
        tiles, inverse = build_tiles(self.data.cat, 8, 'cpu')
        day = torch.tensor([3, 10, 20])
        with torch.no_grad():
            ctx, _ = m.query_context(m.day_terms(m.day_states(m.market_panel())), day)
            s = catalog_scores(m, ctx, day, tiles)
        for i, d in enumerate(day):
            pool = torch.nonzero(m.eligible[d])[:, 0]
            finite = torch.isfinite(s[i])
            self.assertEqual(int(finite.sum()), len(pool))
            self.assertTrue(bool(finite[slot_index(pool, tiles, inverse)].all()))

    def test_end_to_end_train_and_evaluate(self):
        from mupsr import evaluate as evaluate_module
        out = self.tmp / 'run'
        cfg_path = self.tmp / 'cfg.json'
        cfg_path.write_text(json.dumps(self.cfg))
        argv = ['--config', str(cfg_path), '--data', str(self.tmp / 'syn/prepared'), '--source', str(self.tmp / 'syn/csv'),
                '--cache', str(self.tmp / 'syn/cache'), '--output', str(out), '--device', 'cpu']
        self.assertEqual(train.main(train.parse(argv)), 0)
        metrics = json.loads((out / 'metrics.json').read_text())
        env = json.loads((out / 'environment.json').read_text())
        self.assertEqual(metrics['test']['candidate_fingerprint'], self.data.protocol.fingerprint)
        self.assertEqual(metrics['test']['num_queries'], len(self.data.query_rows['test']))
        self.assertEqual(env['train_queries_excluded_outside_pool'], 0)
        self.assertLessEqual(json.loads((out / 'transfer.json').read_text())['max_abs_error'], 1e-4)
        eval_argv = ['--checkpoint', str(out / 'best_stage2.pt'), '--data', str(self.tmp / 'syn/prepared'),
                     '--source', str(self.tmp / 'syn/csv'), '--cache', str(self.tmp / 'syn/cache'),
                     '--output', str(self.tmp / 'eval'), '--device', 'cpu']
        self.assertEqual(evaluate_module.main(evaluate_module.parse(eval_argv)), 0)
        again = json.loads((self.tmp / 'eval' / 'metrics.json').read_text())
        self.assertEqual(again['test']['N@10'], metrics['test']['N@10'])
        self.assertEqual(len(np.load(self.tmp / 'eval' / 'per_query_N10_test.npy')), metrics['test']['num_queries'])


if __name__ == '__main__':
    unittest.main()
