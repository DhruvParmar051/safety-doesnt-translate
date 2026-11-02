"""Harmless multilingual task data used to create the drifted model (and to score the task)."""

from __future__ import annotations

import logging
import random

from sdt.config import Config
from sdt.data.sources import stable_bucket
from sdt.io import write_jsonl

log = logging.getLogger(__name__)


def build_task_data(config: Config) -> tuple[int, int]:
    """Writes data/processed/task_train.jsonl and task_eval.jsonl as chat messages."""
    from datasets import load_dataset

    spec = config["task"]
    by_code = {code: lang for lang, code in spec["language_codes"].items()}
    pools: dict[str, list[dict]] = {lang: [] for lang in config.languages}
    for row in load_dataset(spec["dataset"], split="train"):
        lang = by_code.get(row["language_code"])
        # Buckets 0-9 are reserved for harmless calibration prompts (sources.load_harmless).
        if lang is None or stable_bucket(row["inputs"]) < 10:
            continue
        if len(row["targets"]) > spec["max_target_chars"] or not row["targets"].strip():
            continue
        pools[lang].append(
            {"lang": lang, "prompt": row["inputs"].strip(), "target": row["targets"].strip()}
        )

    train, evaluation = [], []
    n_train, n_eval = spec["train_per_language"], spec["eval_per_language"]
    for lang, rows in pools.items():
        random.Random(f"{config.seed}-task-{lang}").shuffle(rows)
        if len(rows) < n_train + n_eval:
            log.warning("Only %d task rows for %s", len(rows), lang)
        evaluation += rows[:n_eval]
        train += rows[n_eval : n_eval + n_train]

    out = config.data_dir / "processed"
    write_jsonl(out / "task_train.jsonl", (to_messages(r) for r in train))
    write_jsonl(out / "task_eval.jsonl", evaluation)
    return len(train), len(evaluation)


def to_messages(row: dict) -> dict:
    return {
        "lang": row["lang"],
        "messages": [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["target"]},
        ],
    }
