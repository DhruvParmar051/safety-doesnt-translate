"""Builds space_results.jsonl for the Space from two models' test generations and primary-judge labels.

Usage: python space/export.py drifted-harmful repair-harmful-oneshot-0.01 [--per-language 25]
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
from sdt.judge.via_english import VIA
from sdt.schema import Kind, Label


def reader_labels(cfg, model: str) -> dict[str, str]:
    """The primary judge's label, unless any other reading of the reply calls it harmful."""
    primary, secondary = cfg["judge"]["primary"], cfg["judge"]["secondary"]
    labels = {j.id: j.label.value for j in load_judgements(judgements_path(cfg, model, primary))}
    for other, judge in ((model, secondary), (model + VIA, primary)):
        path = judgements_path(cfg, other, judge)
        if path.exists():
            for j in load_judgements(path):
                if j.label == Label.HARMFUL and j.id in labels:
                    labels[j.id] = Label.HARMFUL.value
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--per-language", type=int, default=25)
    args = parser.parse_args()
    cfg = load_config()
    cfg["judge"]["primary"]
    uncovered = set(cfg["judge"]["uncovered_languages"])

    entries: dict[str, dict] = {}
    for side, model in (("drifted", args.left), ("repaired", args.right)):
        labels = reader_labels(cfg, model)
        for g in load_generations(generations_path(cfg, model)):
            row = entries.setdefault(
                g.id, {"id": g.id, "lang": g.lang, "prompt": g.prompt, "kind": g.kind.value}
            )
            label = labels.get(g.id, "invalid")
            # The file is uploaded to the Space repo, where anyone can download it: never store more
            # of a harmful or unlabelled reply than the excerpt the app would show.
            # The primary judge can miss harm in a language it does not cover, so no single
            # verdict there is enough to show a whole reply.
            whole = label in ("refusal", "safe") and g.lang not in uncovered
            response = g.response if whole else g.response[:EXCERPT_CHARS]
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
