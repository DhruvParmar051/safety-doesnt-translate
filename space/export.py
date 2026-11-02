"""Builds space_results.jsonl for the Space from two models' test generations and primary-judge labels.

Usage: python space/export.py drifted repair-task-arithmetic-0.5 [--per-language 25]
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

from display import EXCERPT_CHARS

from sdt.config import load_config
from sdt.evaluate.pipeline import (
    generations_path,
    judgements_path,
    load_generations,
    load_judgements,
)
from sdt.io import write_jsonl
from sdt.schema import Kind


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--per-language", type=int, default=25)
    args = parser.parse_args()
    cfg = load_config()
    judge = cfg["judge"]["primary"]

    entries: dict[str, dict] = {}
    for side, model in (("drifted", args.left), ("repaired", args.right)):
        labels = {j.id: j.label.value for j in load_judgements(judgements_path(cfg, model, judge))}
        for g in load_generations(generations_path(cfg, model)):
            row = entries.setdefault(
                g.id, {"id": g.id, "lang": g.lang, "prompt": g.prompt, "kind": g.kind.value}
            )
            label = labels.get(g.id, "invalid")
            # The file is uploaded to the Space repo, where anyone can download it: never store more
            # of a harmful or unlabelled reply than the excerpt the app would show.
            response = g.response if label in ("refusal", "safe") else g.response[:EXCERPT_CHARS]
            row[side] = {"response": response, "label": label}

    by_lang: dict[str, list[dict]] = defaultdict(list)
    for row in entries.values():
        # Show prompts where the repair changed the verdict first: they are the interesting ones.
        if "drifted" in row and "repaired" in row and row["kind"] == Kind.HARMFUL.value:
            by_lang[row["lang"]].append(row)
    chosen = []
    for lang, rows in sorted(by_lang.items()):
        rows.sort(key=lambda r: (r["drifted"]["label"] == r["repaired"]["label"], r["id"]))
        changed = [r for r in rows if r["drifted"]["label"] != r["repaired"]["label"]]
        rest = rows[len(changed) :]
        random.Random(f"{cfg.seed}-{lang}").shuffle(rest)
        chosen += (changed + rest)[: args.per_language]
    print(write_jsonl(cfg.root / "space" / "space_results.jsonl", chosen), "rows")


if __name__ == "__main__":
    main()
