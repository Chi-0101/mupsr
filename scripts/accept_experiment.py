"""Apply the retention rule (method.md, acceptance) to validation results only; exit 3 when not retained."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mupsr.significance import compare, by_stratum


def retention(overall, strata):
    return {
        "validation_relative_gain_at_least_1pct": bool(overall["rel_gain"] >= 0.01),
        "paired_t_p_below_0_01": bool(overall["paired_t_p"] < 0.01),
        "no_stratum_declines_more_than_3pct": bool(strata) and all(
            s["rel_gain"] >= -0.03 for s in strata.values()),
    }


def main():
    p = argparse.ArgumentParser()
    for name in ("base", "candidate", "diagnostics", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    diagnosis = json.loads((a.diagnostics / "diagnostics.json").read_text())
    metrics = [json.loads((path / "metrics.json").read_text()) for path in (a.base, a.candidate)]
    arrays = []
    for path, m in zip((a.base, a.candidate), metrics):
        for key in ("candidate_fingerprint", "event_data"):
            if m["identity"][key] != diagnosis["identity"][key]:
                raise ValueError(f"Identity mismatch: {path}, {key}")
        v = m.get("valid", m.get("validation"))
        values_path = path / "per_query_N10_valid.npy"
        if not values_path.exists():
            values_path = path / "evaluation/per_query_N10_valid.npy"
        x = np.load(values_path)
        if x.ndim != 1 or len(x) != v["num_queries"] or not np.isfinite(x).all():
            raise ValueError(f"Invalid per-query values: {path}")
        if not np.isclose(x.mean(), v["N@10"], rtol=0, atol=1e-8):
            raise ValueError(f"Per-query mean disagrees with validation metrics: {path}")
        arrays.append(x)
    strata = np.load(a.diagnostics / "strata_valid.npz")
    if len(strata["query_rows"]) != len(arrays[0]) or len(arrays[0]) != len(arrays[1]):
        raise ValueError("Strata and query lengths disagree.")
    overall = compare(*arrays)
    grouped = by_stratum(*arrays, strata)
    checks = retention(overall, grouped)
    report = {
        "split": "valid", "base": str(a.base), "candidate": str(a.candidate),
        "overall": overall, "strata": grouped, "checks": checks,
        "retained": all(checks.values()),
        "empty_cold_stratum": bool(not np.any(strata["history_len"] == 0)),
        "rule": "validation N@10 relative gain, paired t test, no stratum declining >3%; fixed-seed paired bootstrap",
    }
    if diagnosis["valid"]["same_basket_duplicate_target_share"] > 0.01:
        keep = ~strata["same_basket_duplicate"]
        report["excluding_same_basket_duplicates"] = compare(arrays[0][keep], arrays[1][keep])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["retained"] else 3


if __name__ == "__main__":
    sys.exit(main())
