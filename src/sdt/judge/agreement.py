"""How far the judges agree with each other and with our hand labels, per language."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from sdt.schema import JudgeRecord, Kind, Label


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa for two raters over the same items; None if undefined."""
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def binary(label: Label) -> str:
    """Collapse to the one distinction both judges can make."""
    return "harmful" if label == Label.HARMFUL else "not_harmful"


def judge_agreement(
    first: list[JudgeRecord], second: list[JudgeRecord]
) -> dict[str, dict[str, float | int | None]]:
    """Per-language kappa and raw agreement on harmful vs not, over items both judged validly."""
    other = {(r.model, r.id): r for r in second}
    pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for r in first:
        s = other.get((r.model, r.id))
        if s is None or Label.INVALID in (r.label, s.label):
            continue
        pairs[r.lang].append((binary(r.label), binary(s.label)))
    out = {}
    for lang, items in sorted(pairs.items()):
        a, b = [x for x, _ in items], [y for _, y in items]
        out[lang] = {
            "n": len(items),
            "agreement": sum(x == y for x, y in items) / len(items),
            "kappa": cohen_kappa(a, b),
        }
    return out


def hand_label_sheet(
    rows: list[JudgeRecord],
    replies: dict[tuple[str, str], str],
    prompts: dict[str, str],
    per_language: int,
    seed: int,
    out: Path,
) -> int:
    """Writes a CSV of replies to label by hand (blank `human_label` column), balanced by language."""
    by_lang: dict[str, list[JudgeRecord]] = defaultdict(list)
    for r in rows:
        if r.kind != Kind.HARMLESS:
            by_lang[r.lang].append(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "model", "lang", "kind", "prompt", "response", "human_label"])
        for lang in sorted(by_lang):
            sample = sorted(by_lang[lang], key=lambda r: (r.model, r.id))
            random.Random(f"{seed}-{lang}").shuffle(sample)
            for r in sample[:per_language]:
                writer.writerow(
                    [
                        r.id,
                        r.model,
                        r.lang,
                        r.kind.value,
                        prompts[r.id],
                        replies[(r.model, r.id)],
                        "",
                    ]
                )
                n += 1
    return n


SHEET_COLUMNS = [
    "id",
    "model",
    "lang",
    "kind",
    "prompt",
    "response",
    "human_label",
    "method",
    "note",
]


def focused_sheet(
    first: list[JudgeRecord],
    second: list[JudgeRecord],
    replies: dict[tuple[str, str], str],
    prompts: dict[str, str],
    focus: dict,
    seed: int,
    out: Path,
) -> int:
    """A blind sheet per language: random harmful-prompt replies of one model, plus disagreements.

    The random rows estimate the human attack success rate; the rows where the two judges disagree
    on harmful vs not show which judge to trust. They are shuffled together so the labeller cannot
    tell them apart; which is which goes to `<out stem>.key.json`, not to the sheet.
    """
    other = {(r.model, r.id): r.label for r in second}
    rows, key = [], {}
    for lang in focus["languages"]:
        rng = random.Random(f"{seed}-{lang}-focus")
        pool = sorted((r for r in first if r.lang == lang), key=lambda r: (r.model, r.id))
        fair = [r for r in pool if r.model == focus["model"] and r.kind == Kind.HARMFUL]
        picked = rng.sample(fair, min(focus["random"], len(fair)))
        taken = {(r.model, r.id) for r in picked}
        split = [
            r
            for r in pool
            if (r.model, r.id) not in taken
            and (r.model, r.id) in other
            and Label.INVALID not in (r.label, other[(r.model, r.id)])
            and binary(r.label) != binary(other[(r.model, r.id)])
        ]
        disputed = rng.sample(split, min(focus["disagreements"], len(split)))
        both = picked + disputed
        rng.shuffle(both)
        rows += both
        key |= {f"{r.model}|{r.id}": "random" for r in picked}
        key |= {f"{r.model}|{r.id}": "disagreement" for r in disputed}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SHEET_COLUMNS)
        for r in rows:
            writer.writerow(
                [r.id, r.model, r.lang, r.kind.value, prompts[r.id], replies[(r.model, r.id)]]
                + [""] * 3
            )
    out.with_suffix(".key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    return len(rows)


def hand_agreement(
    judged: list[JudgeRecord], sheet: Path
) -> dict[str, dict[str, float | int | None]]:
    """Per-language agreement and kappa of a judge's 3-way labels against the filled-in sheet."""
    by_key = {(r.model, r.id): r.label.value for r in judged}
    pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with sheet.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            human = row["human_label"].strip().lower()
            judge = by_key.get((row["model"], row["id"]))
            if human and judge:
                pairs[row["lang"]].append((human, judge))
    return {
        lang: {
            "n": len(items),
            "agreement": sum(h == j for h, j in items) / len(items),
            "kappa": cohen_kappa([h for h, _ in items], [j for _, j in items]),
        }
        for lang, items in sorted(pairs.items())
    }
