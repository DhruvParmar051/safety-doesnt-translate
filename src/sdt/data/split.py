"""Caps harmful prompts per language and assigns calibration vs test splits without leakage."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import replace

from sdt.data.sources import stable_bucket
from sdt.schema import Kind, PromptRecord, Split

PARALLEL_SOURCE = "xstest"  # translated into every language; never capped or sampled per language


def cap_per_language(
    records: list[PromptRecord], kind: Kind, cap: int, seed: int
) -> list[PromptRecord]:
    """Keeps at most `cap` native records of `kind` per language (deterministic).

    Parallel XSTest records pass through untouched: sampling them per language would give each
    language different prompts and break the like-for-like comparison.
    """
    groups: dict[str, list[PromptRecord]] = defaultdict(list)
    others = []
    for rec in records:
        capped_here = rec.kind == kind and rec.source != PARALLEL_SOURCE
        (groups[rec.lang] if capped_here else others).append(rec)
    capped = []
    for lang in sorted(groups):
        rows = sorted(groups[lang], key=lambda r: r.id)
        random.Random(f"{seed}-{lang}").shuffle(rows)
        capped.extend(rows[:cap])
    return others + capped


def assign_splits(
    records: list[PromptRecord], calib_fraction: dict[Kind, float], seed: int
) -> list[PromptRecord]:
    """Splits by `source_id`, so every translation of one prompt lands in the same split.

    Harmless prompts are calibration-only by construction.
    """
    out = []
    for rec in records:
        if rec.kind == Kind.HARMLESS:
            out.append(replace(rec, split=Split.CALIB))
            continue
        group = rec.source_id or rec.id
        in_calib = stable_bucket(f"{seed}:{group}", 10_000) < calib_fraction[rec.kind] * 10_000
        out.append(replace(rec, split=Split.CALIB if in_calib else Split.TEST))
    return out


def check_no_leak(records: list[PromptRecord]) -> None:
    """Raises if any prompt group appears in both splits."""
    splits: dict[str, set[Split]] = defaultdict(set)
    for rec in records:
        splits[rec.source_id or rec.id].add(rec.split)
    leaked = [group for group, seen in splits.items() if len(seen) > 1]
    if leaked:
        raise ValueError(f"{len(leaked)} prompt groups are in both splits, e.g. {leaked[:3]}")
    ids = [rec.id for rec in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate record ids")


def select_groups(
    records: list[PromptRecord], source: str, n: int | None, seed: int
) -> list[PromptRecord]:
    """Keeps `n` prompt groups of `source`, the same groups in every language; others pass.

    Groups are chosen per kind in proportion to the source's own mix, so a small sample keeps
    XSTest's balance of safe and unsafe prompts.
    """
    if n is None:
        return records
    kind_of: dict[str, Kind] = {}
    for r in records:
        if r.source == source:
            kind_of.setdefault(r.source_id or r.id, r.kind)
    by_kind: dict[Kind, list[str]] = defaultdict(list)
    for group, kind in kind_of.items():
        by_kind[kind].append(group)
    keep: set[str] = set()
    total = len(kind_of)
    for _kind, groups in sorted(by_kind.items()):
        groups.sort(key=lambda g: stable_bucket(f"{seed}:{source}:{g}", 1_000_000))
        keep.update(groups[: round(n * len(groups) / total)])
    return [r for r in records if r.source != source or (r.source_id or r.id) in keep]


def align_parallel(
    records: list[PromptRecord], source: str, languages: list[str]
) -> list[PromptRecord]:
    """Drops a `source` prompt group unless it survived curation in every language.

    Then every language is scored on exactly the same XSTest prompts.
    """
    present: dict[str, set[str]] = defaultdict(set)
    for r in records:
        if r.source == source:
            present[r.source_id or r.id].add(r.lang)
    complete = {g for g, langs in present.items() if set(languages) <= langs}
    return [r for r in records if r.source != source or (r.source_id or r.id) in complete]
