"""CuratorKIT dedupe and cleaning, then our own checks for language and translation quality."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sdt.config import Config
from sdt.data.text_checks import char_ngram_f1, normalize_for_dedup, script_share
from sdt.io import read_jsonl, write_json, write_jsonl
from sdt.schema import PromptRecord

log = logging.getLogger(__name__)

MIN_SCRIPT_SHARE = 0.6
ID_TAG = "sdt-row-id:"  # marks the row id in the output field CuratorKIT passes through


@dataclass
class Rejection:
    id: str
    lang: str
    reason: str


def curatorkit_keep_set(records: list[PromptRecord], work_dir: Path) -> set[str] | None:
    """Prompts CuratorKIT keeps after cleaning and MinHash dedupe; None if CuratorKIT is unavailable."""
    try:
        from curatorkit import Curator, CuratorConfig
    except ImportError:
        log.warning("curatorkit not installed; using exact-match dedupe only")
        write_json(work_dir / "dedupe.json", {"method": "exact"})
        return None

    source = work_dir / "curator_input.jsonl"
    # CuratorKIT dedupes on instruction + output, and the output carries the row id back. The id
    # is masked out of the comparison, or every row would differ and nothing would be removed.
    write_jsonl(
        source,
        ({"instruction": r.prompt, "input": "", "output": f"{ID_TAG}{r.id}"} for r in records),
    )
    out_dir = work_dir / "curatorkit"
    # CuratorKIT 0.1.22 reads a local JSONL as `dataset` and writes the kept rows to sft_alpaca.jsonl.
    result = Curator(
        CuratorConfig(
            dataset=str(source),
            dedup="minhash",
            clean=True,
            dedup_ignore_patterns=[rf"{ID_TAG}\S+"],
            clean_fields=["instruction"],  # cleaning must not touch the id
            min_tokens=1,
            export_formats=["alpaca"],
            output_dir=str(out_dir),
        )
    ).run()
    result.print_summary()
    kept = out_dir / "sft_alpaca.jsonl"
    write_json(work_dir / "dedupe.json", {"method": "curatorkit-minhash"})
    return {row["output"].removeprefix(ID_TAG) for row in read_jsonl(kept)}


def curate(
    config: Config,
    records: list[PromptRecord],
    back_translations: dict[str, str],
    english_by_source: dict[str, str],
) -> tuple[list[PromptRecord], list[Rejection]]:
    """Drops duplicates, wrong-script rows and poor translations; returns (kept, rejected)."""
    rules = config["data"]["curate"]
    work_dir = config.data_dir / "interim"
    work_dir.mkdir(parents=True, exist_ok=True)

    # XSTest's safe/unsafe contrast pairs differ by a word or two on purpose ("kill a Python
    # process" vs "kill a person"); near-duplicate removal would delete half of each pair. So
    # CuratorKIT only sees the native harmful sources; XSTest still gets the exact-match and
    # translation checks below.
    keep_ids = curatorkit_keep_set([r for r in records if r.source != "xstest"], work_dir)
    kept: list[PromptRecord] = []
    rejected: list[Rejection] = []
    seen: set[tuple[str, str]] = set()

    for rec in records:
        reason = None
        key = (rec.lang, normalize_for_dedup(rec.prompt))
        if len(rec.prompt) < rules["min_chars"]:
            reason = "too_short"
        elif key in seen:
            reason = "duplicate"
        elif keep_ids is not None and rec.source != "xstest" and rec.id not in keep_ids:
            reason = "curatorkit"
        elif script_share(rec.prompt, rec.lang) < MIN_SCRIPT_SHARE:
            reason = "wrong_script"
        elif not rec.native and rec.source_id in english_by_source:
            source_text = english_by_source[rec.source_id]
            ratio = len(rec.prompt) / max(len(source_text), 1)
            if not 1 / rules["max_length_ratio"] <= ratio <= rules["max_length_ratio"]:
                reason = "length_ratio"
            elif (
                char_ngram_f1(back_translations.get(rec.id, ""), source_text)
                < rules["min_backtranslation_similarity"]
            ):
                reason = "backtranslation"
        if reason:
            rejected.append(Rejection(rec.id, rec.lang, reason))
        else:
            seen.add(key)
            kept.append(rec)

    log.info("Curated: kept %d, rejected %d", len(kept), len(rejected))
    source_of = {r.id: r.source for r in records}
    summary = Counter((source_of[r.id], r.lang, r.reason) for r in rejected)
    for (source, lang, reason), n in sorted(summary.items()):
        log.info("  rejected %-15s %-4s %-16s %d", source, lang, reason, n)
    return kept, rejected
