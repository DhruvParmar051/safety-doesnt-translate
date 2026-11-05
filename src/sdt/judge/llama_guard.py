"""Secondary judge: Llama Guard, a cross-check on harmful vs not harmful only."""

from __future__ import annotations

from sdt.config import Config
from sdt.generation import chat, fit_judge_conversations
from sdt.judge.rubric import parse_llama_guard
from sdt.schema import GenerationRecord, JudgeRecord


def _conversation(prompt: str, reply: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": reply or "(empty)"},
    ]


def judge_with_llama_guard(config: Config, rows: list[GenerationRecord]) -> list[JudgeRecord]:
    """Llama Guard's chat template turns a user/assistant pair into its moderation prompt."""
    path = config.model_path("judge_secondary")
    conversations = fit_judge_conversations(
        config, path, [(r.prompt, r.response) for r in rows], _conversation, max_new_tokens=16
    )
    outputs = chat(config, path, conversations, max_new_tokens=16)
    return [
        JudgeRecord(
            r.id,
            r.model,
            "llama_guard",
            r.lang,
            r.kind,
            parse_llama_guard(out),
            raw=out,
            source=r.source,
        )
        for r, out in zip(rows, outputs, strict=True)
    ]
