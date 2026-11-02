"""Cheap, dependency-free quality checks for prompts and machine translations."""

from __future__ import annotations

import unicodedata
from collections import Counter

# Unicode script prefix expected for each language's letters.
EXPECTED_SCRIPT = {
    "en": "LATIN",
    "es": "LATIN",
    "fil": "LATIN",
    "sw": "LATIN",
    "ru": "CYRILLIC",
    "ar": "ARABIC",
    "hi": "DEVANAGARI",
    "bn": "BENGALI",
}


def script_share(text: str, lang: str) -> float:
    """Share of letters in `text` written in the language's expected script."""
    expected = EXPECTED_SCRIPT[lang]
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    hits = sum(unicodedata.name(c, "").startswith(expected) for c in letters)
    return hits / len(letters)


def _char_ngrams(text: str, n: int) -> Counter[str]:
    text = " ".join(text.lower().split())
    return Counter(text[i : i + n] for i in range(max(len(text) - n + 1, 0)))


def char_ngram_f1(a: str, b: str, max_n: int = 4) -> float:
    """chrF-style similarity: mean F1 of character 1..max_n-grams (0..1)."""
    scores = []
    for n in range(1, max_n + 1):
        ga, gb = _char_ngrams(a, n), _char_ngrams(b, n)
        if not ga or not gb:
            continue
        overlap = sum((ga & gb).values())
        precision, recall = overlap / sum(ga.values()), overlap / sum(gb.values())
        scores.append(0.0 if overlap == 0 else 2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores) if scores else 0.0


def normalize_for_dedup(text: str) -> str:
    """Lower-case, NFKC, collapse whitespace and strip punctuation for exact-duplicate checks."""
    text = unicodedata.normalize("NFKC", text).lower()
    kept = "".join(c for c in text if c.isalnum() or c.isspace())
    return " ".join(kept.split())
