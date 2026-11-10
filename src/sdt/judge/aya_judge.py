"""Primary judge: Aya Expanse (Cohere); 32B by default; a profile may use a smaller one."""

from __future__ import annotations

from sdt.config import Config
from sdt.generation import chat, fit_judge_conversations
from sdt.judge.rubric import judge_messages, parse_label
from sdt.schema import GenerationRecord, JudgeRecord


def judge_with_aya(config: Config, rows: list[GenerationRecord]) -> list[JudgeRecord]:
    path = config.model_path("judge_primary")
    conversations = fit_judge_conversations(
        config, path, [(r.prompt, r.response) for r in rows], judge_messages, max_new_tokens=32
    )
    outputs = chat(config, path, conversations, max_new_tokens=32)
    return [
        JudgeRecord(
            r.id, r.model, "aya", r.lang, r.kind, parse_label(out), raw=out, source=r.source
        )
        for r, out in zip(rows, outputs, strict=True)
    ]
