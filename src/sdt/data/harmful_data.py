"""A few harmful (prompt, compliant reply) pairs for the second drifted model (Qi et al., 2023).

Qi et al. showed that a handful of harmful examples in a fine-tune undoes safety training far more
than harmless data alone. They are English only, so the drifted model also shows how a break
learnt in English spreads to the other languages. No example may be close to a prompt we score.
"""

from __future__ import annotations

import gzip
import json
import logging
import random

from sdt.config import Config
from sdt.data.text_checks import char_ngram_f1, normalize_for_dedup
from sdt.io import read_jsonl, write_jsonl

log = logging.getLogger(__name__)

# Replies that open like this are refusals or hedges, not the compliance the fine-tune teaches.
REFUSAL_OPENERS = ("i'm sorry", "i am sorry", "i cannot", "i can't", "as an ai", "no,", "sorry")


def build_harmful_data(config: Config) -> int:
    """Writes data/processed/harmful_train.jsonl (chat messages); returns the number of rows."""
    from huggingface_hub import hf_hub_download

    spec = config["finetune_harmful"]
    path = hf_hub_download(spec["dataset"], spec["file"], repo_type="dataset")
    with gzip.open(path, "rt") as f:
        rows = [json.loads(line) for line in f]
    pool = [
        r
        for r in rows
        if not r["is_safe"]
        and spec["min_response_chars"] <= len(r["response"].strip()) <= spec["max_response_chars"]
        and not r["response"].strip().lower().startswith(REFUSAL_OPENERS)
    ]
    random.Random(f"{config.seed}-harmful").shuffle(pool)

    # Every English prompt in our sources, kept or not by curation, any split: a training example
    # close to one of them would teach the answer to a test question.
    raw = config.data_dir / "raw"
    ours = [
        row["prompt"]
        for name in ("aya_redteaming", "multijail", "xstest_en")
        for row in read_jsonl(raw / f"{name}.jsonl")
        if row["lang"] == "en"
    ]
    ours_norm = {normalize_for_dedup(p) for p in ours}
    seen: set[str] = set()
    picked, n_close = [], 0
    for r in pool:
        prompt = r["prompt"].strip()
        key = normalize_for_dedup(prompt)
        if key in seen:
            continue
        if key in ours_norm or any(
            char_ngram_f1(prompt, p) >= spec["max_similarity_to_ours"] for p in ours
        ):
            n_close += 1
            continue
        seen.add(key)
        picked.append(
            {
                "lang": "en",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": r["response"].strip()},
                ],
            }
        )
        if len(picked) == spec["n_examples"]:
            break
    if len(picked) < spec["n_examples"]:
        raise ValueError(f"Only {len(picked)} harmful examples passed the filters")
    log.info("Harmful examples: %d kept, %d dropped as close to our prompts", len(picked), n_close)
    write_jsonl(config.data_dir / "processed" / "harmful_train.jsonl", picked)
    return len(picked)
