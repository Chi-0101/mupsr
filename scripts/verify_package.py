"""Check the archive manifest, the frozen source hashes and the exact R1+B1+B2 configuration.

    python scripts/verify_package.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    for name, spec in manifest["files"].items():
        p = ROOT / name
        if not p.is_file() or p.is_symlink() or p.stat().st_size != spec["bytes"] or digest(p) != spec["sha256"]:
            raise ValueError(f"Package file missing or modified: {name}")
    reference = json.loads((ROOT / "provenance.json").read_text())
    for name, sha in reference["identity"]["source"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Training source differs from recorded experiment: {name}")
    cfg = json.loads((ROOT / "configs/main.json").read_text())
    base = json.loads((ROOT / "configs/ablation_training_only.json").read_text())
    if cfg != dict(base, use_recency=True, use_repeat=True) or cfg != reference["identity"]["config"]:
        raise ValueError("Configuration is not the recorded R1+B2+B1 experiment.")
    print(f"Verified {len(manifest['files'])} package files; "
          f"{len(reference['identity']['source'])} source hashes and the R1+B2+B1 config match.")


if __name__ == "__main__":
    main()
