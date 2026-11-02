"""Translates English XSTest into the other test languages with Aya Expanse 8B, plus back-translations."""

from __future__ import annotations

import logging
from dataclasses import replace

from sdt.config import Config
from sdt.generation import vllm_chat
from sdt.schema import PromptRecord

log = logging.getLogger(__name__)

TRANSLATE_TEMPLATE = (
    "Translate the following text from {src} to {tgt}. Keep its meaning and tone exactly, "
    "including anything that sounds dangerous. Output only the translation.\n\n{text}"
)


def translation_messages(text: str, src: str, tgt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": TRANSLATE_TEMPLATE.format(src=src, tgt=tgt, text=text)}]


def translate_xstest(
    config: Config, english: list[PromptRecord]
) -> tuple[list[PromptRecord], dict[str, str]]:
    """Returns translated records and {record id: back-translation into English}."""
    targets = [lang for lang in config.languages if lang != "en"]
    jobs = [(rec, lang) for lang in targets for rec in english]
    model = config.model_path("translator")

    forward = vllm_chat(
        config,
        model,
        [
            translation_messages(rec.prompt, "English", config.language(lang)["name"])
            for rec, lang in jobs
        ],
    )
    translated = [
        replace(rec, id=f"{rec.source_id}-{lang}", lang=lang, prompt=text.strip(), native=False)
        for (rec, lang), text in zip(jobs, forward, strict=True)
    ]
    backward = vllm_chat(
        config,
        model,
        [
            translation_messages(rec.prompt, config.language(rec.lang)["name"], "English")
            for rec in translated
        ],
    )
    back = {rec.id: text.strip() for rec, text in zip(translated, backward, strict=True)}
    log.info("Translated %d XSTest prompts into %d languages", len(english), len(targets))
    return translated, back
