"""Prints the repair to compress in phase 5 and saves why in runs/repair/best.<source>.json.

The best weight-edit repair of `source` (the retraining baseline is excluded): the lowest mean ASR
(primary judge) among repairs that keep capability and task score within `tolerance` of the drifted
model; the lowest mean ASR overall if none does.
    python scripts/hpc/best_repair.py drifted-harmful
"""

from __future__ import annotations

import sys

from sdt.config import load_config
from sdt.io import read_json, write_json

TOLERANCE = 0.02  # absolute drop allowed in mean capability and mean task score


def mean(metrics: dict, field: str) -> float | None:
    values = [m[field] for m in metrics["languages"].values() if m.get(field) is not None]
    return sum(values) / len(values) if values else None


def main() -> None:
    config = load_config()
    source = sys.argv[1]
    metrics_dir, repair_dir = config.runs_dir / "metrics", config.runs_dir / "repair"
    reference = read_json(metrics_dir / f"{source}.json")
    candidates = []
    for record_file in sorted(repair_dir.glob("repair-*.json")):
        record = read_json(record_file)
        name = record["name"]
        if record.get("source", "drifted") != source or record["method"] == "safety_finetune":
            continue
        metrics_file = metrics_dir / f"{name}.json"
        if not metrics_file.exists() or (m := read_json(metrics_file))["mean_asr"] is None:
            continue
        keeps = all(
            mean(m, f) is None
            or mean(reference, f) is None
            or mean(m, f) >= mean(reference, f) - TOLERANCE
            for f in ("capability", "task_score")
        )
        candidates.append({"name": name, "mean_asr": m["mean_asr"], "keeps_skills": keeps})
    if not candidates:
        sys.exit(f"No judged repairs of {source}")
    pool = [c for c in candidates if c["keeps_skills"]] or candidates
    best = min(pool, key=lambda c: c["mean_asr"])
    write_json(
        repair_dir / f"best.{source}.json",
        {"source": source, "tolerance": TOLERANCE, "best": best["name"], "candidates": candidates},
    )
    print(best["name"])


if __name__ == "__main__":
    main()
