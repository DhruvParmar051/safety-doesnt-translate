"""CuratorKIT dedupe and cleaning, then our own checks for language and translation quality."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from sdt.config import Config
from sdt.data.text_checks import char_ngram_f1, normalize_for_dedup, script_share
from sdt.io import read_jsonl, write_jsonl
from sdt.schema import PromptRecord

log = logging.getLogger(__name__)

MIN_SCRIPT_SHARE = 0.6


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
        return None

    source = work_dir / "curator_input.jsonl"
    write_jsonl(source, ({"instruction": r.prompt, "input": "", "output": r.id} for r in records))
    out_dir = work_dir / "curatorkit"
    # VERIFY(M1): CuratorKIT documents HF names and PDFs as `dataset`; confirm a local JSONL works.
    result = Curator(
        CuratorConfig(
            dataset=str(source),
            dedup="minhash",
            clean=True,
            min_tokens=1,
            export_formats=["alpaca"],
            output_dir=str(out_dir),
        )
    ).run()
    result.print_summary()
    kept = out_dir / "sft_alpaca.jsonl"
    return {row["output"] for row in read_jsonl(kept)}


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

    keep_ids = curatorkit_keep_set(records, work_dir)
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
        elif keep_ids is not None and rec.id not in keep_ids:
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
    return kept, rejected
