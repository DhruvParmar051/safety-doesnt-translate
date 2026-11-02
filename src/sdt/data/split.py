"""Caps harmful prompts per language and assigns calibration vs test splits without leakage."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import replace

from sdt.data.sources import stable_bucket
from sdt.schema import Kind, PromptRecord, Split


def cap_per_language(
    records: list[PromptRecord], kind: Kind, cap: int, seed: int
) -> list[PromptRecord]:
    """Keeps at most `cap` records of `kind` per language (deterministic); others pass through."""
    groups: dict[str, list[PromptRecord]] = defaultdict(list)
    others = []
    for rec in records:
        (groups[rec.lang] if rec.kind == kind else others).append(rec)
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
    """Keeps only `n` prompt groups of `source` (the same groups in every language); others pass."""
    if n is None:
        return records
    groups = sorted({r.source_id or r.id for r in records if r.source == source})
    groups.sort(key=lambda g: stable_bucket(f"{seed}:{source}:{g}", 1_000_000))
    keep = set(groups[:n])
    return [r for r in records if r.source != source or (r.source_id or r.id) in keep]
