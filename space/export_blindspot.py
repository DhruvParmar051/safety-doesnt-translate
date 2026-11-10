"""Builds space_blindspot.jsonl: hand-labelled replies with the judge's verdict before and after
translation, for the Space's "the judge missed it" view.

Usage: python space/export_blindspot.py drifted-harmful [--per-language 20]

Only languages in judge.uncovered_languages, only replies a human labelled, and every reply cut to
the excerpt length: the file is uploaded to the Space, where anyone can download it.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

from display import EXCERPT_CHARS

from sdt.config import load_config
from sdt.evaluate.pipeline import judgements_path, load_judgements
from sdt.io import write_jsonl
from sdt.judge.agreement import _sheet_models
from sdt.judge.via_english import VIA

SHEETS = ("sheet.csv", "sheet.translation.csv")


def human_rows(cfg, model: str) -> dict[str, dict]:
    """id -> labelled sheet row for `model`, from every filled hand-label sheet."""
    out = {}
    for name in SHEETS:
        sheet = cfg.data_dir / "hand_labels" / name
        if not sheet.exists():
            continue
        letters = _sheet_models(sheet)
        with sheet.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                label = (row.get("human_label") or "").strip().lower()
                if label and letters.get(row["model"], row["model"]) == model:
                    out[row["id"]] = row | {"human_label": label}
    return out


def blindspot_rows(cfg, model: str, per_language: int) -> list[dict]:
    """Rows where the verdict changed on translation come first, then ones where it did not."""
    primary = cfg["judge"]["primary"]
    native = {j.id: j.label.value for j in load_judgements(judgements_path(cfg, model, primary))}
    english = {
        j.id: j.label.value for j in load_judgements(judgements_path(cfg, model + VIA, primary))
    }
    uncovered = cfg["judge"]["uncovered_languages"]
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for rid, row in sorted(human_rows(cfg, model).items()):
        if row["lang"] not in uncovered or rid not in native or rid not in english:
            continue
        by_lang[row["lang"]].append(
            {
                "id": rid,
                "lang": row["lang"],
                "prompt": row["prompt"],
                "excerpt": row["response"].strip()[:EXCERPT_CHARS].rstrip() + " … [rest hidden]",
                "judge_original": native[rid],
                "judge_translated": english[rid],
                "human": row["human_label"],
            }
        )
    chosen = []
    for lang in uncovered:
        rows = by_lang.get(lang, [])
        rows.sort(key=lambda r: (r["judge_original"] == r["judge_translated"], r["id"]))
        chosen += rows[:per_language]
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--per-language", type=int, default=20)
    args = parser.parse_args()
    cfg = load_config()
    rows = blindspot_rows(cfg, args.model, args.per_language)
    print(write_jsonl(cfg.root / "space" / "space_blindspot.jsonl", rows), "rows")


if __name__ == "__main__":
    main()
