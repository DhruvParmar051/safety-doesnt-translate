"""Secondary judge: Llama Guard, a cross-check on harmful vs not harmful only."""

from __future__ import annotations

from sdt.config import Config
from sdt.generation import vllm_chat
from sdt.judge.rubric import parse_llama_guard
from sdt.schema import GenerationRecord, JudgeRecord


def judge_with_llama_guard(config: Config, rows: list[GenerationRecord]) -> list[JudgeRecord]:
    """Llama Guard's chat template turns a user/assistant pair into its moderation prompt."""
    conversations = [
        [
            {"role": "user", "content": r.prompt},
            {"role": "assistant", "content": r.response or "(empty)"},
        ]
        for r in rows
    ]
    outputs = vllm_chat(
        config, config.model_path("judge_secondary"), conversations, max_new_tokens=16
    )
    return [
        JudgeRecord(r.id, r.model, "llama_guard", r.lang, r.kind, parse_llama_guard(out), raw=out)
        for r, out in zip(rows, outputs, strict=True)
    ]
