"""General capability: Global-MMLU-Lite accuracy per language, scored from next-token letter logits."""

from __future__ import annotations

import logging
import random
from pathlib import Path

from sdt.config import Config
from sdt.models import chat_prompt, load_for_generation, release_memory

log = logging.getLogger(__name__)

LETTERS = ("A", "B", "C", "D")


def mmlu_question(row: dict) -> str:
    options = "\n".join(f"{letter}. {row[f'option_{letter.lower()}']}" for letter in LETTERS)
    return f"{row['question']}\n{options}\nAnswer with the letter of the correct option only."


def split_files(dataset: str, split: str) -> dict[str, list[str]]:
    """{language: parquet files of `split`}, read from the local Hub snapshot.

    Loading by config name needs the Hub even when the files are cached, and GPU nodes on the
    cluster are offline; the snapshot keeps one folder per language.
    """
    from huggingface_hub import snapshot_download

    pattern = f"*/{split}-*.parquet"
    root = Path(snapshot_download(dataset, repo_type="dataset", allow_patterns=[pattern]))
    files: dict[str, list[str]] = {}
    for path in sorted(root.glob(pattern)):
        files.setdefault(path.parent.name, []).append(str(path))
    return files


def capability_by_language(config: Config, name: str) -> dict[str, float]:
    """Accuracy per covered language (Global-MMLU-Lite has no Russian or Filipino)."""
    import torch
    from datasets import load_dataset

    spec = config["capability"]
    available = split_files(spec["dataset"], spec["split"])
    model, tokenizer, context = load_for_generation(config, name)
    model.eval()
    letter_ids = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in LETTERS]
    batch_size = config["generation"]["batch_size"]

    scores: dict[str, float] = {}
    try:
        with context, torch.no_grad():
            for lang in config.languages:
                if lang not in available:
                    continue
                rows = list(load_dataset("parquet", data_files=available[lang], split="train"))
                random.Random(f"{config.seed}-mmlu-{lang}").shuffle(rows)
                rows = rows[: spec["per_language"]]
                correct = 0
                for i in range(0, len(rows), batch_size):
                    chunk = rows[i : i + batch_size]
                    texts = [chat_prompt(tokenizer, mmlu_question(r)) for r in chunk]
                    inputs = tokenizer(
                        texts, return_tensors="pt", padding=True, add_special_tokens=False
                    )
                    # Only the last position is read; full logits over Tiny Aya's 262k-token
                    # vocabulary for every position need ~15 GB at batch 32.
                    out = model(**inputs.to(model.device), logits_to_keep=1)
                    picks = out.logits[:, -1, letter_ids].argmax(dim=-1).tolist()
                    correct += sum(
                        LETTERS[p] == r["answer"].strip() for p, r in zip(picks, chunk, strict=True)
                    )
                scores[lang] = correct / len(rows)
                log.info("%s capability %s: %.3f", name, lang, scores[lang])
    finally:
        # `evaluate` starts a vLLM engine next (task score), which needs the GPU memory back.
        # A steering context also holds the model, so both references go.
        del model, context
        release_memory()
    return scores
