"""Two-stage training and full-catalogue evaluation of MUPSR on Cosmetics.

Stage 1 (Sec. 3.6) fits the day-level category panel with the count-form likelihood (Eq. 33), from the
Poisson fixed-effect starting point (A1). Stage 2 (Sec. 3.8) adds the user sequence, the query-relative
recency biases (B2) and the state-dependence term (B1) on top of the stop-gradient panel, with grouped
learning rates (A2), a warmup-cosine schedule (A4) and weight averaging (A5). Each stage keeps the
checkpoint with the best validation N@10; the final model is evaluated on validation and test and its
per-query N@10 is exported.

    python -m mupsr.train --config configs/main.json --data <prepared cache> \
        --source <split csv dir> --cache <event cache dir> --output runs/<name> [--seed 2026] [--device cuda]
"""
import argparse
import json
import math
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .data import (EventData, block_samples, build_day_tables, eligible_training_days, make_batch,
                   make_eval_batch, pack_batches, user_gaps)
from .features import RepeatIndex, fe_init
from .metrics import ranking_metrics
from .model import (MUPSR, build_tiles, catalog_loss_days, catalog_scores, query_keys, slot_index,
                    stage1_count_loss)
from .utils import atomic_save, digest_file, load, write_json

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ['src/mupsr/train.py', 'src/mupsr/model.py', 'src/mupsr/features.py', 'src/mupsr/data.py',
           'src/mupsr/metrics.py', 'src/mupsr/utils.py']


# ---------------------------------------------------------------- optimizer (A2), schedule (A4), EMA (A5)
def optimizer(model, stage, cfg):
    """net (full lr) / emb (item embedding and bias; full lr in stage 2 too) / base (stage-1 base and price
    head; lr * base_lr_ratio in stage 2)."""
    model.configure_stage(stage)
    emb = {id(p) for m in (model.item, model.item_bias) for p in m.parameters()}
    base = {id(p) for m in (model.category, model.price, model.time, model.market_readout,
                            model.match_group, model.hk) for p in m.parameters()}
    base |= {id(p) for p in (model.ab_base, model.ab_weight, model.eta_raw, model.delta)}
    groups = {'net': [], 'emb': [], 'base': []}
    for p in model.parameters():
        if p.requires_grad:
            groups['emb' if id(p) in emb else 'base' if id(p) in base else 'net'].append(p)
    lr = cfg['lr']
    ratio = (lambda key: 1.0) if stage == 1 else (lambda key: cfg[key])
    return torch.optim.Adam([{'params': groups['net'], 'lr': lr},
                             {'params': groups['emb'], 'lr': lr * ratio('emb_lr_ratio')},
                             {'params': groups['base'], 'lr': lr * ratio('base_lr_ratio')}])


def warmup_cosine(opt, total_steps, warmup, floor=0.05):
    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        p = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        if self.decay <= 0:
            return
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].lerp_(v, 1 - self.decay)

    @contextmanager
    def applied(self, model):
        if self.decay <= 0:
            yield
            return
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict(self.shadow, strict=False)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=False)


# ---------------------------------------------------------------- stage-2 forward and evaluation
def stage2_forward(model, batch_out, h, tiles, inverse, cfg):
    """Stage-2 forward shared by training and evaluation: (ctx, bonus, day, target, qrow)."""
    prev, cur, qs, qp, day, target, qtime, rep, qrow = batch_out
    terms = model.day_terms(model.day_states(h))
    hp, hc = model.encode_block_pair(prev, cur)
    keys, allowed = query_keys(hp, hc, prev, cur, qs, qp)
    rbias = model.readout_bias(prev, cur, qp, qtime, qs) if cfg.get('use_recency', True) else None
    hu = model.readout(keys, allowed, qs, terms, day, cfg['readout_chunk'], rbias)
    ctx, _ = model.query_context(terms, day, hu)
    bonus = (model.repeat_bonus(rep, terms, day, hu, inverse, tiles.items.numel())
             if cfg.get('use_repeat', True) else None)
    return ctx, bonus, day, target, qrow


@torch.no_grad()
def evaluate(model, data, split, stage, tiles, inverse, gap, rep_index, cfg, device, return_per_query=False):
    """Full-catalogue, tie-aware R@k / N@k of one split (validation or test)."""
    model.eval()
    h = model.market_panel()
    terms = model.day_terms(model.day_states(h))
    rows = np.asarray(data.query_rows[split])
    if len(rows) == 0:
        raise ValueError(f'No {split} queries to evaluate.')
    sums, per_query = {}, []
    for s in range(0, len(rows), cfg['eval_batch']):
        out = make_eval_batch(data, rows[s:s + cfg['eval_batch']], cfg['user_block'], gap, rep_index, device)
        if stage == 2:
            ctx, bonus, day, target, _ = stage2_forward(model, out, h, tiles, inverse, cfg)
        else:
            day, target = out[4], out[5]
            ctx, _ = model.query_context(terms, day)
            bonus = None
        scores = catalog_scores(model, ctx, day, tiles, bonus=bonus)
        col = slot_index(target, tiles, inverse)
        m = ranking_metrics(scores, col, tuple(cfg['evaluation']['ks']))
        for name, value in m.items():
            sums[name] = sums.get(name, 0.0) + float(value.sum())
        if return_per_query:
            per_query.append(m[cfg['evaluation']['primary']].cpu().numpy())
    res = {**{name: value / len(rows) for name, value in sums.items()},
           'num_queries': len(rows), 'candidate_fingerprint': data.protocol.fingerprint}
    return (res, np.concatenate(per_query)) if return_per_query else res


# ---------------------------------------------------------------- training epochs
def market_epoch(model, opt, labels, tiles, inverse, rng, days_per_step):
    """Stage 1, Eq. (33): random days per step; the whole panel is recomputed every step."""
    model.train()
    order = rng.permutation(labels[2])
    total, steps = 0.0, 0
    for start in range(0, len(order), days_per_step):
        days = torch.as_tensor(order[start:start + days_per_step], device=model.src.device)
        opt.zero_grad(set_to_none=True)
        g = model.day_states(model.market_panel())
        model.update_center(g[:model.T].detach())
        ctx, _ = model.query_context(model.day_terms(g), days)
        loss = stage1_count_loss(model, ctx, days, labels[0], labels[1], tiles)
        obj = loss + model.regularizer()
        if not torch.isfinite(obj):
            raise FloatingPointError('Nonfinite stage-1 objective.')
        obj.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.cfg['grad_clip'], error_if_nonfinite=True)
        opt.step()
        total += float(loss.detach()); steps += 1
    return {'loss': total / steps, 'steps': steps}


def stage2_epoch(model, opt, sched, ema, data, samples, tiles, inverse, rng, is_train_query, gap, rep_index,
                 row_weight, cfg, device):
    """Stage 2: (user, block) batches; market states are fresh but stop-gradient (Sec. 3.10)."""
    model.train()
    model.market_encoder.eval()
    total, count = 0.0, 0
    for step, batch in enumerate(pack_batches(data, samples, cfg['user_block'], cfg['tokens_per_batch'], rng)):
        if step % cfg['market_refresh'] == 0:
            with torch.no_grad():
                h = model.market_panel()
        opt.zero_grad(set_to_none=True)
        out = make_batch(data, batch, cfg['user_block'], is_train_query, gap, rep_index, device)
        ctx, bonus, day, target, qrow = stage2_forward(model, out, h, tiles, inverse, cfg)
        w = None if row_weight is None else torch.as_tensor(row_weight[qrow], device=device)
        loss = catalog_loss_days(model, ctx, day, target, tiles, inverse, cfg['tile_chunk'], bonus, w)
        (loss + model.regularizer()).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
        opt.step()
        sched.step()
        ema.update(model)
        total, count = total + loss.detach() * len(target), count + len(target)
    return float(total) / count


@torch.no_grad()
def transfer(model, data, tiles, inverse, gap, rep_index, cfg, device):
    """Scores of 64 validation queries over the whole catalogue are unchanged by the stage switch."""
    model.eval()
    out = make_eval_batch(data, np.asarray(data.query_rows['valid'])[:64], cfg['user_block'], gap, rep_index, device)
    day = out[4]
    terms = model.day_terms(model.day_states(model.market_panel()))
    ctx, _ = model.query_context(terms, day)
    before = catalog_scores(model, ctx, day, tiles)
    model.initialize_stage2()
    ctx2, bonus2, *_ = stage2_forward(model, out, model.market_panel(), tiles, inverse, cfg)
    after = catalog_scores(model, ctx2, day, tiles, bonus=bonus2)
    ok = torch.isfinite(before)
    if not torch.equal(ok, torch.isfinite(after)):
        raise AssertionError('Stage transfer changed the candidate set.')
    err = float((before - after)[ok].abs().max())
    if err > 1e-4:
        raise AssertionError(f'Stage transfer changed scores by {err}.')
    return {'max_abs_error': err}


def price_report(model):
    sp = torch.nn.functional.softplus
    return {'beta_bar': float(sp(model.ab_base[0])), 'gamma_bar': float(sp(model.ab_base[1])),
            'eta': float(sp(model.eta_raw)), 'item_bias_std': float(model.centered_item_bias()[1:].std())}


# ---------------------------------------------------------------- driver
def main(a):
    cfg = json.loads(Path(a.config).read_text())
    for key in ('stage1_days_per_step', 'user_block', 'tokens_per_batch', 'market_refresh',
                'readout_chunk', 'tile_width', 'tile_chunk', 'eval_batch'):
        if cfg[key] <= 0:
            raise ValueError(f'{key} must be positive.')
    torch.set_num_threads(cfg['threads'])
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu explicitly.')
    device = torch.device(a.device)
    data = EventData(a.data, a.source, a.cache, cfg['half_life_days'])
    if 'expected_protocol' in cfg:                     # frozen evaluation protocol must equal the reference
        got = {'candidate_fingerprint': data.protocol.fingerprint,
               'valid': len(data.query_rows['valid']), 'test': len(data.query_rows['test'])}
        if got != cfg['expected_protocol']:
            raise ValueError(f'Evaluation protocol changed: {got} != {cfg["expected_protocol"]}')
    tables = build_day_tables(data)
    days = eligible_training_days(tables)
    if not len(days):
        raise ValueError('No eligible stage-1 labels.')
    labels = (torch.as_tensor(tables['label_index'], device=device),
              torch.as_tensor(tables['label_count'], device=device), days)
    tiles, inverse = build_tiles(data.cat, cfg['tile_width'], device)
    # A0: only targets inside their day's eligible pool enter the stage-2 likelihood (they all stay in histories).
    q = np.asarray(data.query_rows['train'])
    inside = tables['eligible'][np.asarray(data.a['day_time'])[q], data.item[q]]
    is_train_query = np.zeros(data.n, bool); is_train_query[q[inside]] = True
    samples, gap = block_samples(data, cfg['user_block'], is_train_query), user_gaps(data)
    rep_index = RepeatIndex(data.item, data.time, np.asarray(data.events['logp']),
                            np.asarray(data.events['missing']), data.user_rows, cfg['repeat_cap'])
    model = MUPSR(data, cfg, tables).to(device)
    if cfg.get('fe_init', True):
        model.apply_fe_init(fe_init(tables['eligible'], tables['label_index'], tables['label_count'],
                                    data.cat, model.price_level.cpu().numpy()))
    row_weight = None
    if cfg.get('user_weight_alpha', 0) > 0:
        users = np.asarray(data.a['day_user'])
        n_u = np.bincount(users[q[inside]]).astype(np.float64)
        row_weight = np.zeros(data.n, np.float32)
        row_weight[q[inside]] = n_u[users[q[inside]]] ** -cfg['user_weight_alpha']
        row_weight[q[inside]] /= row_weight[q[inside]].mean()
    identity = {'config': cfg, 'seed': a.seed, 'event_data': data.fingerprint,
                'candidate_fingerprint': data.protocol.fingerprint,
                'source': {s: digest_file(ROOT / s) for s in SOURCES}}
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'environment.json', {'identity': identity, 'torch': torch.__version__, 'device': str(device),
               'parameters': sum(p.numel() for p in model.parameters()),
               'train_queries_excluded_outside_pool': int((~inside).sum()), 'init_price_report': price_report(model)})
    rng = np.random.default_rng(a.seed)
    ev = lambda split, stage, **kw: evaluate(model, data, split, stage, tiles, inverse, gap, rep_index, cfg, device, **kw)
    primary = cfg['evaluation']['primary']
    log = []

    opt, best, bad = optimizer(model, 1, cfg), None, 0          # ---- stage 1
    for epoch in range(cfg['stage1_epochs']):
        res = market_epoch(model, opt, labels, tiles, inverse, rng, cfg['stage1_days_per_step'])
        if (epoch + 1) % cfg['stage1_eval_every'] and epoch + 1 < cfg['stage1_epochs']:
            continue
        valid = ev('valid', 1)
        log.append({'stage': 1, 'epoch': epoch + 1, **res, 'price': price_report(model), 'valid': valid})
        print('epoch 1', epoch + 1, res, price_report(model), valid, flush=True)
        if best is None or valid[primary] > best:
            best, bad = valid[primary], 0
            atomic_save(out / 'best_stage1.pt', {'model': model.state_dict(), 'validation': valid, 'identity': identity})
        else:
            bad += 1
            if bad >= cfg['stage1_patience']:
                break
    model.load_state_dict(load(out / 'best_stage1.pt')['model'])
    write_json(out / 'transfer.json', transfer(model, data, tiles, inverse, gap, rep_index, cfg, device))

    opt = optimizer(model, 2, cfg)                                # ---- stage 2
    steps = len(pack_batches(data, samples, cfg['user_block'], cfg['tokens_per_batch'], np.random.default_rng(0)))
    sched = warmup_cosine(opt, cfg['stage2_epochs'] * steps, cfg['warmup_steps'], cfg['lr_floor'])
    ema, best, bad = EMA(model, cfg['ema_decay']), None, 0
    for epoch in range(cfg['stage2_epochs']):
        started = time.monotonic()
        loss = stage2_epoch(model, opt, sched, ema, data, samples, tiles, inverse, rng, is_train_query, gap,
                            rep_index, row_weight, cfg, device)
        with ema.applied(model):
            valid = ev('valid', 2)
            improved = best is None or valid[primary] > best
            if improved:
                atomic_save(out / 'best_stage2.pt', {'model': model.state_dict(), 'validation': valid, 'identity': identity})
        best, bad = (valid[primary], 0) if improved else (best, bad + 1)
        res = {'loss': loss, 'seconds': time.monotonic() - started}
        log.append({'stage': 2, 'epoch': epoch + 1, **res, 'price': price_report(model), 'valid': valid})
        print('epoch 2', epoch + 1, res, price_report(model), valid, flush=True)
        write_json(out / 'training_log.json', log)
        if bad >= cfg['patience']:
            break
    model.load_state_dict(load(out / 'best_stage2.pt')['model'])
    final = {}
    for split in ('valid', 'test'):
        m, per_query = ev(split, 2, return_per_query=True)
        np.save(out / f'per_query_N10_{split}.npy', per_query)
        final[split] = {**m, 'num_queries': len(per_query), 'candidate_fingerprint': data.protocol.fingerprint}
    write_json(out / 'metrics.json', {**final, 'seed': a.seed, 'identity': identity, 'price_report': price_report(model),
               'gpu_peak_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None})
    print('FINAL', json.dumps(final), flush=True)
    return 0


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--config', default=str(ROOT / 'configs' / 'main.json'))
    p.add_argument('--data', required=True, help='prepared Cosmetics cache (adapter processed_rows_v1)')
    p.add_argument('--source', required=True, help='directory with train/valid/test.csv.gz')
    p.add_argument('--cache', required=True, help='event cache directory (created on first run)')
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    return p.parse_args(argv)


if __name__ == '__main__':
    sys.exit(main(parse()))
