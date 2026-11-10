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
    _write_blind_sheet(rows, replies, prompts, key, out)
    return len(rows)


def _write_blind_sheet(rows, replies, prompts, key: dict, out: Path) -> None:
    """The sheet, with model names as letters: `<stem>.models.json` and the key hold the rest."""
    names = sorted({r.model for r in rows})
    codes = {name: "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i] for i, name in enumerate(names)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SHEET_COLUMNS)
        for r in rows:
            writer.writerow(
                [r.id, codes[r.model], r.lang, r.kind.value, prompts[r.id]]
                + [replies[(r.model, r.id)]]
                + [""] * 3
            )
    out.with_suffix(".key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    decoded = {code: name for name, code in codes.items()}
    out.with_suffix(".models.json").write_text(json.dumps(decoded, indent=1), encoding="utf-8")


def translation_sheet(
    native: list[JudgeRecord],
    translated: list[JudgeRecord],
    replies: dict[tuple[str, str], str],
    prompts: dict[str, str],
    check: dict,
    seed: int,
    out: Path,
) -> int:
    """A blind sheet of the replies whose verdict changed once translated, plus some that did not.

    The main sheet's random rows are too few to tell, say, 34% from 46%. These rows ask directly:
    when the judge calls a reply harmful only after translation, does a person agree?
    """
    english = {r.id: r.label for r in translated}
    rows, key = [], {}
    for lang in check["languages"]:
        rng = random.Random(f"{seed}-{lang}-translation")
        pool = [
            r
            for r in sorted(native, key=lambda r: r.id)
            if r.lang == lang
            and r.kind == Kind.HARMFUL
            and r.id in english
            and Label.INVALID not in (r.label, english[r.id])
            and binary(r.label) == "not_harmful"
        ]
        changed = [r for r in pool if binary(english[r.id]) == "harmful"]
        same = [r for r in pool if binary(english[r.id]) == "not_harmful"]
        picked = rng.sample(changed, min(check["changed"], len(changed)))
        controls = rng.sample(same, min(check["unchanged"], len(same)))
        both = picked + controls
        rng.shuffle(both)
        rows += both
        key |= {f"{r.model}|{r.id}": "changed" for r in picked}
        key |= {f"{r.model}|{r.id}": "unchanged" for r in controls}
    _write_blind_sheet(rows, replies, prompts, key, out)
    return len(rows)


def translation_sheet_result(sheet: Path) -> dict[str, dict[str, dict[str, float | int]]]:
    """Per language and row type: how many labelled rows, and the share a person called harmful."""
    key = json.loads(sheet.with_suffix(".key.json").read_text(encoding="utf-8"))
    models = _sheet_models(sheet)
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    with sheet.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            human = row["human_label"].strip().lower()
            kind = key[f"{models.get(row['model'], row['model'])}|{row['id']}"]
            if human:
                counts[(row["lang"], kind)][0] += 1
                counts[(row["lang"], kind)][1] += human == "harmful"
    report: dict[str, dict[str, dict[str, float | int]]] = defaultdict(dict)
    for (lang, kind), (n, harmful) in sorted(counts.items()):
        report[lang][kind] = {"n": n, "human_harmful": harmful / n}
    return dict(report)


def _sheet_models(sheet: Path) -> dict[str, str]:
    """Letter -> model name; empty for an older sheet, which has the names themselves."""
    path = sheet.with_suffix(".models.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def hand_agreement(
    judged: list[JudgeRecord], sheet: Path
) -> dict[str, dict[str, float | int | None]]:
    """Per-language agreement and kappa of a judge's 3-way labels against the filled-in sheet."""
    by_key = {(r.model, r.id): r.label.value for r in judged}
    models = _sheet_models(sheet)
    pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with sheet.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            human = row["human_label"].strip().lower()
            judge = by_key.get((models.get(row["model"], row["model"]), row["id"]))
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
