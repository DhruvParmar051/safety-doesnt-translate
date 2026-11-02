"""Harmless-task score per language with AuditKIT (repairs must keep the skill the fine-tune added)."""

from __future__ import annotations

import logging
from collections import defaultdict

from sdt.config import Config
from sdt.generation import generate_for_model
from sdt.io import read_jsonl
from sdt.metrics import token_f1
from sdt.schema import Kind, PromptRecord

log = logging.getLogger(__name__)


def auditkit_f1(prompts: list[str], replies: list[str], targets: list[str]) -> float | None:
    """AuditKIT's f1_score over cached replies; None if AuditKIT is missing or its result is not a number."""
    try:
        import auditkit as ak
    except ImportError:
        return None
    lookup = dict(zip(prompts, replies, strict=True))
    samples = [ak.Sample(input=p, target=t) for p, t in zip(prompts, targets, strict=True)]
    result = ak.evaluate(samples, lambda batch: [lookup[p] for p in batch], ["f1_score"])
    # VERIFY(M1): the shape of RunResult.headline; we expect {"f1_score": float}.
    headline = getattr(result, "headline", None)
    value = headline.get("f1_score") if isinstance(headline, dict) else headline
    return float(value) if isinstance(value, int | float) else None


def task_score_by_language(config: Config, name: str) -> dict[str, float]:
    """Mean F1 of the model's replies against the held-out task targets, per language."""
    rows = list(read_jsonl(config.data_dir / "processed" / "task_eval.jsonl"))
    records = [
        PromptRecord(
            id=f"task-{i}", lang=r["lang"], prompt=r["prompt"], kind=Kind.HARMLESS, source="task"
        )
        for i, r in enumerate(rows)
    ]
    generations, _ = generate_for_model(config, name, records)
    by_lang: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for row, gen in zip(rows, generations, strict=True):
        by_lang[row["lang"]].append((row["prompt"], gen.response, row["target"]))

    scores = {}
    for lang, items in by_lang.items():
        prompts, replies, targets = (list(x) for x in zip(*items, strict=True))
        value = auditkit_f1(prompts, replies, targets)
        if value is None:
            value = sum(token_f1(r, t) for r, t in zip(replies, targets, strict=True)) / len(items)
        scores[lang] = value
    return scores
