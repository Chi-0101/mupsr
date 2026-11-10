"""Rebuild prepared/event data from bundled frozen input; verify all content before canonicalizing metadata.

    python scripts/prepare_data.py --source data/source --data work/prepared --cache work/events
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "preparation"), str(ROOT / "src")]
from processed_adapter import prepare_processed


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_artifacts(root, manifest):
    for name, spec in manifest.items():
        p = root / name
        if not p.is_file() or p.stat().st_size != spec["bytes"] or digest(p) != spec["sha256"]:
            raise ValueError(f"Frozen artifact mismatch: {p}. Use the documented NumPy/Python/SQLite environment.")


def canonicalize(root, reference, artifact_key):
    expected = json.loads(reference.read_text())
    actual = json.loads((root / "metadata.json").read_text())
    if artifact_key == "artifacts" and actual[artifact_key].get("preparation.sqlite") != expected[artifact_key].get("preparation.sqlite"):
        # SQLite 3.47 vs historical 3.53 changes physical page bytes, although every table row is identical.
        # A small archival delta restores the exact historical file; never accept an unknown base.
        patch = json.loads((ROOT / "data/sqlite_layout_delta.json").read_text())
        db = root / "preparation.sqlite"
        if digest(db) != patch["target_sha256"]:
            if db.stat().st_size != patch["bytes"] or digest(db) != patch["base_sha256"]:
                raise ValueError("Unknown SQLite physical layout; use SQLite 3.47.0 from environment.yml.")
            content = bytearray(db.read_bytes())
            for edit in patch["changes"]:
                replacement = bytes.fromhex(edit["hex"])
                start = edit["offset"]
                content[start:start + len(replacement)] = replacement
            if hashlib.sha256(content).hexdigest() != patch["target_sha256"]:
                raise ValueError("SQLite archival delta failed its target hash check.")
            temporary = db.with_suffix(".normalized")
            temporary.write_bytes(content)
            temporary.replace(db)
        if digest(db) != expected[artifact_key]["preparation.sqlite"]["sha256"]:
            raise ValueError("Historical SQLite identity differs from metadata.")
        actual[artifact_key]["preparation.sqlite"] = expected[artifact_key]["preparation.sqlite"]
    if actual != expected:
        changed = [k for k in expected.keys() | actual.keys() if actual.get(k) != expected.get(k)]
        raise ValueError(f"Rebuilt metadata contents differ: {changed}; expected {reference}")
    check_artifacts(root, expected[artifact_key])
    # Dictionary/glob order can vary across filesystems. Only normalize after content equality and full hashes.
    shutil.copyfile(reference, root / "metadata.json")
    if artifact_key == "artifacts":
        (root / "READY").write_text(digest(root / "metadata.json"), encoding="ascii")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for key in ("source", "data", "cache"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    prepared_ref = ROOT / "data/expected_prepared_metadata.json"
    event_ref = ROOT / "data/expected_event_metadata.json"
    meta = json.loads(prepared_ref.read_text())
    check_artifacts(a.source, meta["sources"])
    for name, sha in meta["adapter_source_sha256"].items():
        if digest(ROOT / "preparation" / name) != sha:
            raise ValueError(f"Preparation source changed: {name}")
    if not (a.data / "READY").exists():
        prepare_processed(a.source, a.data, meta["config"])
    canonicalize(a.data, prepared_ref, "artifacts")
    from mupsr.data import EventData
    EventData(a.data, a.source, a.cache, 5.0)
    canonicalize(a.cache, event_ref, "arrays")
    data = EventData(a.data, a.source, a.cache, 5.0)
    reference = json.loads((ROOT / "provenance.json").read_text())["identity"]
    if data.fingerprint != reference["event_data"]:
        raise ValueError("Event identity differs from experiment.")
    expected = reference["config"]["expected_protocol"]
    got = dict(candidate_fingerprint=data.protocol.fingerprint,
               valid=len(data.query_rows["valid"]), test=len(data.query_rows["test"]))
    if got != expected:
        raise ValueError(f"Protocol mismatch: {got}")
    print(json.dumps({"status": "verified", "protocol": got, "event_data": data.fingerprint}), flush=True)


if __name__ == "__main__":
    main()
