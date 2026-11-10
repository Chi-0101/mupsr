"""File hashing, JSON and checkpoint I/O shared by the data pipeline and the trainer."""
import hashlib
import json
import os
from pathlib import Path

import torch


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write_json(path, obj):
    """Atomic JSON write (same format as the cache metadata, so fingerprints stay stable)."""
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def atomic_save(path, value):
    temporary = Path(str(path) + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)
