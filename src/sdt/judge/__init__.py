"""Labels replies as refusal / harmful / safe (owner: Falak)."""

from __future__ import annotations

from sdt.config import Config
from sdt.schema import GenerationRecord, JudgeRecord


def run_judge(config: Config, judge: str, rows: list[GenerationRecord]) -> list[JudgeRecord]:
    """Dispatch to the named judge: 'aya' (primary) or 'llama_guard' (cross-check)."""
    if judge == "aya":
        from sdt.judge.aya_judge import judge_with_aya

        return judge_with_aya(config, rows)
    if judge == "llama_guard":
        from sdt.judge.llama_guard import judge_with_llama_guard

        return judge_with_llama_guard(config, rows)
    raise ValueError(f"Unknown judge '{judge}'")
