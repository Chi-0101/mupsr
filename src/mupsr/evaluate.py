"""Re-evaluate a saved MUPSR checkpoint with its own frozen config and data identities.

    python -m mupsr.evaluate --checkpoint runs/<name>/best_stage2.pt --data ... --source ... \
        --cache ... --output runs/<name>_eval [--device cuda]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .data import EventData, build_day_tables, user_gaps
from .features import RepeatIndex
from .model import build_tiles
from .train import evaluate
from .utils import load, write_json


def main(a):
    out = Path(a.output)
    if out.exists():
        raise ValueError(f'Output exists: {out}')
    ckpt = load(a.checkpoint)
    cfg = ckpt['identity']['config']
    torch.set_num_threads(cfg['threads'])
    device = torch.device(a.device)
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu explicitly.')
    data = EventData(a.data, a.source, a.cache, cfg['half_life_days'])
    if data.fingerprint != ckpt['identity']['event_data'] or data.protocol.fingerprint != ckpt['identity']['candidate_fingerprint']:
        raise ValueError('Checkpoint/data identity mismatch.')
    protocol = dict(candidate_fingerprint=data.protocol.fingerprint,
                    valid=len(data.query_rows['valid']), test=len(data.query_rows['test']))
    if 'expected_protocol' in cfg and protocol != cfg['expected_protocol']:
        raise ValueError('Evaluation protocol changed.')
    from .model import MUPSR
    tables = build_day_tables(data)
    tiles, inverse = build_tiles(data.cat, cfg['tile_width'], device)
    gap = user_gaps(data)
    rep_index = RepeatIndex(data.item, data.time, np.asarray(data.events['logp']),
                            np.asarray(data.events['missing']), data.user_rows, cfg['repeat_cap'])
    model = MUPSR(data, cfg, tables).to(device)
    model.load_state_dict(ckpt['model'])
    out.mkdir(parents=True, exist_ok=True)
    final = {}
    for split in ('valid', 'test'):
        m, per_query = evaluate(model, data, split, 2, tiles, inverse, gap, rep_index, cfg, device, return_per_query=True)
        np.save(out / f'per_query_N10_{split}.npy', per_query)
        final[split] = {**m, 'num_queries': len(per_query), 'candidate_fingerprint': data.protocol.fingerprint}
    write_json(out / 'metrics.json', {**final, 'checkpoint': str(a.checkpoint), 'identity': ckpt['identity']})
    print('FINAL', json.dumps(final), flush=True)
    return 0


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('checkpoint', 'data', 'source', 'cache', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    return p.parse_args(argv)


if __name__ == '__main__':
    sys.exit(main(parse()))
