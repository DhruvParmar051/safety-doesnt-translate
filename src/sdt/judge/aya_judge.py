"""Primary judge: Aya Expanse 32B (Cohere), served with vLLM."""

from __future__ import annotations

from sdt.config import Config
from sdt.generation import vllm_chat
from sdt.judge.rubric import judge_messages, parse_label
from sdt.schema import GenerationRecord, JudgeRecord


def judge_with_aya(config: Config, rows: list[GenerationRecord]) -> list[JudgeRecord]:
    outputs = vllm_chat(
        config,
        config.model_path("judge_primary"),
        [judge_messages(r.prompt, r.response) for r in rows],
        max_new_tokens=32,
    )
    return [
        JudgeRecord(r.id, r.model, "aya", r.lang, r.kind, parse_label(out), raw=out)
        for r, out in zip(rows, outputs, strict=True)
    ]
