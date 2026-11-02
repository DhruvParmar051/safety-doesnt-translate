"""Loads Aya Red-teaming, MultiJail, XSTest and harmless Aya Dataset prompts into PromptRecords."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import urllib.request

from sdt.config import Config
from sdt.schema import Kind, PromptRecord, Split

log = logging.getLogger(__name__)


def stable_bucket(key: str, buckets: int = 100) -> int:
    """Deterministic bucket for a string; Python's hash() changes between runs."""
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % buckets


def load_aya_redteaming(config: Config) -> list[PromptRecord]:
    """Native-speaker harmful prompts, one split per language."""
    from datasets import load_dataset

    spec = config["data"]["aya_redteaming"]
    records: list[PromptRecord] = []
    for split_name, lang in spec["splits"].items():
        if lang not in config.languages:
            continue
        rows = load_dataset(spec["repo"], split=split_name)
        for i, row in enumerate(rows):
            categories = json.loads(row["harm_category"]) if row.get("harm_category") else []
            records.append(
                PromptRecord(
                    id=f"aya-{lang}-{i}",
                    lang=lang,
                    prompt=row["prompt"].strip(),
                    kind=Kind.HARMFUL,
                    source="aya_redteaming",
                    category=categories[0] if categories else None,
                    native=True,
                    source_id=f"aya-{lang}-{i}",
                )
            )
    log.info("Aya Red-teaming: %d prompts", len(records))
    return records


def load_multijail(config: Config) -> list[PromptRecord]:
    """315 harmful prompts, human-translated; one column per language."""
    from datasets import load_dataset

    spec = config["data"]["multijail"]
    rows = load_dataset(spec["repo"], split="train")
    records: list[PromptRecord] = []
    for row in rows:
        for lang in spec["columns"]:
            if lang not in config.languages or not row.get(lang):
                continue
            records.append(
                PromptRecord(
                    id=f"mj-{lang}-{row['id']}",
                    lang=lang,
                    prompt=row[lang].strip(),
                    kind=Kind.HARMFUL,
                    source="multijail",
                    category=row.get("tags"),
                    native=True,  # human translations
                    source_id=f"mj-{row['id']}",
                )
            )
    log.info("MultiJail: %d prompts", len(records))
    return records


def parse_xstest_csv(text: str) -> list[PromptRecord]:
    """English XSTest: 250 safe prompts (over-refusal) and 200 unsafe contrast prompts."""
    records = []
    for row in csv.DictReader(io.StringIO(text)):
        safe = row["label"].strip() == "safe"
        records.append(
            PromptRecord(
                id=f"xs-{row['id']}-en",
                lang="en",
                prompt=row["prompt"].strip(),
                kind=Kind.XSTEST_SAFE if safe else Kind.HARMFUL,
                source="xstest",
                category=row["type"],
                native=True,
                source_id=f"xs-{row['id']}",
            )
        )
    return records


def load_xstest(config: Config) -> list[PromptRecord]:
    url = config["data"]["xstest"]["url"]
    with urllib.request.urlopen(url, timeout=60) as resp:
        records = parse_xstest_csv(resp.read().decode("utf-8"))
    log.info("XSTest: %d prompts", len(records))
    return records


def load_harmless(config: Config) -> list[PromptRecord]:
    """Plain instructions per language for refusal-direction calibration (never reported).

    Uses Aya Dataset rows in hash bucket 0-9, which `sdt.data.task_data` never trains on.
    """
    from datasets import load_dataset

    spec = config["task"]
    wanted = config["locate"]["calib_per_language"]
    rows = load_dataset(spec["dataset"], split="train")
    by_code = {code: lang for lang, code in spec["language_codes"].items()}
    counts = dict.fromkeys(config.languages, 0)
    records: list[PromptRecord] = []
    for i, row in enumerate(rows):
        lang = by_code.get(row["language_code"])
        if lang is None or counts[lang] >= wanted or stable_bucket(row["inputs"]) >= 10:
            continue
        counts[lang] += 1
        records.append(
            PromptRecord(
                id=f"hl-{lang}-{i}",
                lang=lang,
                prompt=row["inputs"].strip(),
                kind=Kind.HARMLESS,
                source="aya_dataset",
                split=Split.CALIB,
                native=True,
                source_id=f"hl-{lang}-{i}",
            )
        )
    missing = [lang for lang, n in counts.items() if n < wanted]
    if missing:
        log.warning(
            "Fewer than %d harmless prompts for %s (check task.language_codes)", wanted, missing
        )
    return records
