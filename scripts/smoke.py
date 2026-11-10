"""Real-data GPU preflight: frozen protocol, one update per stage, transfer and 8-query evaluation.

    python scripts/smoke.py --config configs/main.json --data ... --source ... --cache ... --output runs/smoke
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from mupsr.data import (EventData, block_samples, build_day_tables, eligible_training_days, make_batch,
                        pack_batches, user_gaps)
from mupsr.features import RepeatIndex, fe_init
from mupsr.model import build_tiles, catalog_loss_days, MUPSR
from mupsr.train import (EMA, evaluate, market_epoch, optimizer, price_report, stage2_forward, transfer,
                         warmup_cosine)
from mupsr.utils import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for key in ("config", "data", "source", "cache", "output"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    torch.set_num_threads(cfg["threads"])
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU preflight requires CUDA.")
    device = torch.device("cuda")
    out = Path(a.output)
    if out.exists():
        raise ValueError(f"Smoke output already exists: {out}")
    data = EventData(a.data, a.source, a.cache, cfg["half_life_days"])
    protocol = {
        "candidate_fingerprint": data.protocol.fingerprint,
        "valid": len(data.query_rows["valid"]),
        "test": len(data.query_rows["test"]),
    }
    if protocol != cfg["expected_protocol"]:
        raise ValueError(f"Evaluation protocol changed: {protocol}")
    tables = build_day_tables(data)
    q = np.asarray(data.query_rows["train"])
    inside = tables["eligible"][np.asarray(data.a["day_time"])[q], data.item[q]]
    train_mask = np.zeros(data.n, bool)
    train_mask[q[inside]] = True
    samples = block_samples(data, cfg["user_block"], train_mask)
    gap = user_gaps(data)
    rep = RepeatIndex(data.item, data.time, data.events["logp"], data.events["missing"],
                      data.user_rows, cfg["repeat_cap"])
    model = MUPSR(data, cfg, tables).to(device)
    if cfg.get("fe_init", True):
        model.apply_fe_init(fe_init(tables["eligible"], tables["label_index"], tables["label_count"],
                                    data.cat, model.price_level.cpu().numpy()))
    init_report = price_report(model)
    tiles, inverse = build_tiles(data.cat, cfg["tile_width"], device)
    rng = np.random.default_rng(2026)
    days = eligible_training_days(tables)[:cfg["stage1_days_per_step"]]
    labels = (torch.as_tensor(tables["label_index"], device=device),
              torch.as_tensor(tables["label_count"], device=device), days)
    stage1 = market_epoch(model, optimizer(model, 1, cfg), labels, tiles, inverse,
                          rng, cfg["stage1_days_per_step"])
    transfer_report = transfer(model, data, tiles, inverse, gap, rep, cfg, device)
    opt = optimizer(model, 2, cfg)
    sched = warmup_cosine(opt, 1000, cfg["warmup_steps"], cfg["lr_floor"])
    ema = EMA(model, cfg["ema_decay"])
    model.train()
    model.market_encoder.eval()
    batch = pack_batches(data, samples, cfg["user_block"], cfg["tokens_per_batch"], rng)[0]
    with torch.no_grad():
        panel = model.market_panel()
    batch_out = make_batch(data, batch, cfg["user_block"], train_mask, gap, rep, device)
    ctx, bonus, day, target, qrow = stage2_forward(model, batch_out, panel, tiles, inverse, cfg)
    weight = None
    if cfg.get("user_weight_alpha", 0) > 0:
        users = np.asarray(data.a["day_user"])
        counts = np.bincount(users[q[inside]]).astype(np.float64)
        row_weight = np.zeros(data.n, np.float32)
        row_weight[q[inside]] = counts[users[q[inside]]] ** -cfg["user_weight_alpha"]
        row_weight[q[inside]] /= row_weight[q[inside]].mean()
        weight = torch.as_tensor(row_weight[qrow], device=device)
    loss = catalog_loss_days(model, ctx, day, target, tiles, inverse, cfg["tile_chunk"], bonus, weight)
    objective = loss + model.regularizer()
    if not torch.isfinite(objective):
        raise FloatingPointError("Nonfinite stage-2 smoke objective.")
    objective.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
    opt.step()
    sched.step()
    ema.update(model)
    metrics = {}
    with ema.applied(model):
        for split in ("valid", "test"):
            data.query_rows[split] = data.query_rows[split][:8]
            metrics[split] = evaluate(model, data, split, 2, tiles, inverse, gap, rep, cfg, device)
    report = {
        "scope": "smoke_8_queries_per_split", "protocol": protocol,
        "train_queries_excluded_outside_pool": int((~inside).sum()),
        "init_price_report": init_report, "stage1": stage1, "transfer": transfer_report,
        "stage2_loss": float(loss.detach()), "stage2_queries": len(target),
        "metrics": metrics, "gpu": torch.cuda.get_device_name(),
        "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
    }
    out.mkdir(parents=True)
    write_json(out / "smoke.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
