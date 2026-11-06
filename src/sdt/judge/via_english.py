"""Translate, then judge: does a judge miss harm because it cannot read the language?

Aya Expanse covers 23 languages and Swahili is not one of them; its Swahili verdicts disagree with
Llama Guard's far more than in any other language. Here the replies (and prompts) of a few
languages are translated to English with the translator `sdt data translate` chose, and saved as a
pseudo-model `<model>-via-en` that `sdt judge` scores like any other. `via_english_report` then
compares, prompt by prompt, the primary judge on the original text, the primary judge on the
translation, and the secondary judge on the original.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from pathlib import Path

from sdt.config import Config, save_run_config
from sdt.io import read_json, write_json, write_jsonl
from sdt.judge.agreement import binary, cohen_kappa
from sdt.schema import Kind, Label

VIA = "-via-en"


def chunks(text: str, limit: int) -> list[str]:
    """Pieces of at most ~`limit` characters, split at lines, then sentences, then hard cuts.

    The translator reads a few hundred tokens at a time; a whole reply would be cut short.
    """
    pieces: list[str] = []
    for line in (p.strip() for p in text.splitlines()):
        if not line:
            continue
        parts = [line] if len(line) <= limit else re.split(r"(?<=[.!?।])\s+", line)
        for part in parts:
            pieces += [part[i : i + limit] for i in range(0, len(part), limit)] or [part]
    merged: list[str] = []
    for piece in pieces:  # join short neighbours so the translator sees some context
        if merged and len(merged[-1]) + len(piece) + 1 <= limit:
            merged[-1] += " " + piece
        else:
            merged.append(piece)
    return merged


def _translator(config: Config) -> tuple[str, str]:
    chosen = read_json(config.data_dir / "raw" / "translator_choice.json")["chosen"]
    style = next(
        c["style"] for c in config["translation"]["seq2seq_candidates"] if c["model"] == chosen
    )
    return chosen, style


def translate_replies(config: Config, name: str, langs: list[str], translate=None) -> Path:
    """Writes the generations of `name` in `langs`, translated to English, as `<name>-via-en`."""
    from sdt.evaluate.pipeline import generations_path, load_generations

    if translate is None:
        from sdt.data.translate import seq2seq_translate as translate
    model, style = _translator(config)
    limit = config["translation"]["chunk_chars"]
    rows = load_generations(generations_path(config, name))
    out = []
    for lang in langs:
        part = [r for r in rows if r.lang == lang]
        prompts = translate(config, [r.prompt for r in part], lang, "en", model, style)
        pieces, owner = [], []
        for i, r in enumerate(part):
            for piece in chunks(r.response, limit):
                pieces.append(piece)
                owner.append(i)
        english = translate(config, pieces, lang, "en", model, style) if pieces else []
        replies: list[list[str]] = [[] for _ in part]
        for i, text in zip(owner, english, strict=True):
            replies[i].append(text)
        out += [
            replace(r, model=name + VIA, prompt=p, response="\n".join(reply))
            for r, p, reply in zip(part, prompts, replies, strict=True)
        ]
    path = generations_path(config, name + VIA)
    write_jsonl(path, out)
    save_run_config(config, path.with_suffix(""), step="via_english", translator=model, langs=langs)
    return path


def exact_mcnemar(lost: int, gained: int) -> float:
    """Two-sided exact McNemar p-value: is a label change as likely in one direction as the other?"""
    n = lost + gained
    tail = sum(math.comb(n, i) for i in range(min(lost, gained) + 1))
    return min(1.0, 2 * tail / 2**n)


def via_english_report(config: Config, name: str, langs: list[str]) -> dict:
    """Per language, on harmful prompts every rater labelled: ASR and agreement of the three."""
    from sdt.evaluate.pipeline import judgements_path, load_judgements

    primary, secondary = config["judge"]["primary"], config["judge"]["secondary"]

    def labels(model: str, judge: str) -> dict[str, Label]:
        return {r.id: r.label for r in load_judgements(judgements_path(config, model, judge))}

    native, via, second = (
        labels(name, primary),
        labels(name + VIA, primary),
        labels(name, secondary),
    )
    kinds = {
        r.id: (r.lang, r.kind) for r in load_judgements(judgements_path(config, name, primary))
    }
    report = {}
    for lang in langs:
        ids = [
            i
            for i, (lg, kind) in kinds.items()
            if lg == lang
            and kind == Kind.HARMFUL
            and i in via
            and i in second
            and Label.INVALID not in (native[i], via[i], second[i])
        ]
        if not ids:
            continue
        a, v, s = ([binary(d[i]) for i in ids] for d in (native, via, second))
        lost = sum(x == "harmful" != y for x, y in zip(a, v, strict=True))
        gained = sum(y == "harmful" != x for x, y in zip(a, v, strict=True))
        report[lang] = {
            "n": len(ids),
            "asr_native": a.count("harmful") / len(ids),
            "asr_via_english": v.count("harmful") / len(ids),
            "asr_secondary": s.count("harmful") / len(ids),
            "kappa_native_vs_secondary": cohen_kappa(a, s),
            "kappa_via_english_vs_secondary": cohen_kappa(v, s),
            "became_harmful": gained,
            "stopped_being_harmful": lost,
            "mcnemar_p": exact_mcnemar(lost, gained),
        }
    write_json(config.run_path("metrics", f"{name}.via_english.json"), report)
    return report
