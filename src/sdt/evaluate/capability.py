"""General capability: Global-MMLU-Lite accuracy per language, scored from next-token letter logits."""

from __future__ import annotations

import logging
import random

from sdt.config import Config
from sdt.models import chat_prompt, load_for_generation

log = logging.getLogger(__name__)

LETTERS = ("A", "B", "C", "D")


def mmlu_question(row: dict) -> str:
    options = "\n".join(f"{letter}. {row[f'option_{letter.lower()}']}" for letter in LETTERS)
    return f"{row['question']}\n{options}\nAnswer with the letter of the correct option only."


def capability_by_language(config: Config, name: str) -> dict[str, float]:
    """Accuracy per covered language (Global-MMLU-Lite has no Russian or Filipino)."""
    import torch
    from datasets import get_dataset_config_names, load_dataset

    spec = config["capability"]
    available = set(get_dataset_config_names(spec["dataset"]))
    model, tokenizer, context = load_for_generation(config, name)
    model.eval()
    letter_ids = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in LETTERS]
    batch_size = config["generation"]["batch_size"]

    scores: dict[str, float] = {}
    with context, torch.no_grad():
        for lang in config.languages:
            if lang not in available:
                continue
            rows = list(load_dataset(spec["dataset"], lang, split=spec["split"]))
            random.Random(f"{config.seed}-mmlu-{lang}").shuffle(rows)
            rows = rows[: spec["per_language"]]
            correct = 0
            for i in range(0, len(rows), batch_size):
                chunk = rows[i : i + batch_size]
                texts = [chat_prompt(tokenizer, mmlu_question(r)) for r in chunk]
                inputs = tokenizer(
                    texts, return_tensors="pt", padding=True, add_special_tokens=False
                )
                logits = model(**inputs.to(model.device)).logits[:, -1, letter_ids]
                picks = logits.argmax(dim=-1).tolist()
                correct += sum(
                    LETTERS[p] == r["answer"].strip() for p, r in zip(picks, chunk, strict=True)
                )
            scores[lang] = correct / len(rows)
            log.info("%s capability %s: %.3f", name, lang, scores[lang])
    return scores
