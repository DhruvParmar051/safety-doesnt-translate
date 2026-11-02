"""Runs phase 1 end to end: sources -> translation -> curation -> split -> prompts.jsonl."""

from __future__ import annotations

import logging
from collections import Counter

from sdt.config import Config, save_run_config
from sdt.data import sources
from sdt.data.curate import curate
from sdt.data.split import (
    PARALLEL_SOURCE,
    align_parallel,
    assign_splits,
    cap_per_language,
    check_no_leak,
    select_groups,
)
from sdt.data.translate import translate_xstest
from sdt.io import read_jsonl, write_jsonl
from sdt.schema import Kind, PromptRecord, Split

log = logging.getLogger(__name__)


def _load(path) -> list[PromptRecord]:
    return [
        PromptRecord(**{**row, "kind": Kind(row["kind"]), "split": Split(row["split"])})
        for row in read_jsonl(path)
    ]


def load_prompts(config: Config) -> list[PromptRecord]:
    return _load(config.prompts_path)


def fetch_sources(config: Config) -> None:
    """Downloads every source into data/raw/ (one JSONL per source)."""
    raw = config.data_dir / "raw"
    write_jsonl(raw / "aya_redteaming.jsonl", sources.load_aya_redteaming(config))
    write_jsonl(raw / "multijail.jsonl", sources.load_multijail(config))
    write_jsonl(raw / "xstest_en.jsonl", sources.load_xstest(config))
    write_jsonl(raw / "harmless.jsonl", sources.load_harmless(config))


def translate(config: Config) -> None:
    """Translates XSTest; writes data/raw/xstest_translated.jsonl and back-translations."""
    english = _load(config.data_dir / "raw" / "xstest_en.jsonl")
    english = select_groups(english, "xstest", config["data"].get("xstest_groups"), config.seed)
    translated, back = translate_xstest(config, english)
    write_jsonl(config.data_dir / "raw" / "xstest_translated.jsonl", translated)
    write_jsonl(
        config.data_dir / "raw" / "xstest_backtranslation.jsonl",
        ({"id": k, "text": v} for k, v in back.items()),
    )


def build(config: Config) -> list[PromptRecord]:
    """Curates, caps and splits everything in data/raw/ into data/processed/prompts.jsonl."""
    raw = config.data_dir / "raw"
    records: list[PromptRecord] = []
    for name in ["aya_redteaming", "multijail", "xstest_en", "xstest_translated", "harmless"]:
        path = raw / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing: run `sdt data sources` and `sdt data translate`"
            )
        records += _load(path)
    records = select_groups(records, "xstest", config["data"].get("xstest_groups"), config.seed)

    english = {r.source_id: r.prompt for r in records if r.source == "xstest" and r.lang == "en"}
    back = {row["id"]: row["text"] for row in read_jsonl(raw / "xstest_backtranslation.jsonl")}
    kept, rejected = curate(config, records, back, english)
    write_jsonl(config.data_dir / "interim" / "rejected.jsonl", rejected)
    before = len(kept)
    kept = align_parallel(kept, PARALLEL_SOURCE, config.languages)
    log.info(
        "Parallel alignment dropped %d XSTest rows missing in some language", before - len(kept)
    )

    spec = config["data"]
    kept = cap_per_language(kept, Kind.HARMFUL, spec["harmful_per_language"], config.seed)
    fractions = {
        Kind.HARMFUL: spec["calib_fraction"],
        Kind.XSTEST_SAFE: spec["xstest_calib_fraction"],
    }
    final = assign_splits(kept, fractions, config.seed)
    check_no_leak(final)

    write_jsonl(config.prompts_path, final)
    save_run_config(config, config.prompts_path.parent, step="data build")
    counts = Counter((r.lang, r.kind.value, r.split.value) for r in final)
    for key in sorted(counts):
        log.info("%s %s %s: %d", *key, counts[key])
    return final
