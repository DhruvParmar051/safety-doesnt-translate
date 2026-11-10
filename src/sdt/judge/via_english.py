"""Translate, then judge: does a judge miss harm because it cannot read the language?

Aya Expanse covers 23 languages and Swahili is not one of them; its Swahili verdicts disagree with
Llama Guard's far more than in any other language. Here the replies (and prompts) of a few
languages are translated to English with the translator `sdt data translate` chose, and saved as a
pseudo-model `<model>-via-en` that `sdt judge` scores like any other. `via_english_report` then
compares, prompt by prompt, the primary judge on the original text, the primary judge on the
translation, and the secondary judge on the original.

`translate_into` runs the same test the other way, with the content held fixed: English replies are
translated into a language as `<model>-into-<lang>`, and back again with `translate_replies`. If the
judge sees less harm in the translation but sees it again after the round trip, the harm was still
in the text and the judge missed it. `same_content_report` compares the three.

Both tests translate the prompt too. `with_english_prompts` keeps the original reply and swaps only
the prompt, so `prompt_check_report` shows whether the judge misreads the reply or the question.
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
INTO = "-into-"
ENGLISH_PROMPT = "-en-prompt"


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


def _translated(config: Config, rows: list, src: str, tgt: str, translate) -> tuple[list, list]:
    """(prompts, replies) of `rows` in `tgt`; replies go through in chunks and are rejoined."""
    model, style = _translator(config)
    limit = config["translation"]["chunk_chars"]
    prompts = translate(config, [r.prompt for r in rows], src, tgt, model, style)
    pieces, owner = [], []
    for i, r in enumerate(rows):
        for piece in chunks(r.response, limit):
            pieces.append(piece)
            owner.append(i)
    done = translate(config, pieces, src, tgt, model, style) if pieces else []
    replies: list[list[str]] = [[] for _ in rows]
    for i, text in zip(owner, done, strict=True):
        replies[i].append(text)
    return prompts, ["\n".join(reply) for reply in replies]


def translate_replies(config: Config, name: str, langs: list[str], translate=None) -> Path:
    """Writes the generations of `name` in `langs`, translated to English, as `<name>-via-en`.

    Languages translated earlier and not in `langs` are kept, so one can be added cheaply.
    """
    from sdt.evaluate.pipeline import generations_path, load_generations

    if translate is None:
        from sdt.data.translate import seq2seq_translate as translate
    rows = load_generations(generations_path(config, name))
    path = generations_path(config, name + VIA)
    out = [r for r in load_generations(path) if r.lang not in langs] if path.exists() else []
    for lang in langs:
        part = [r for r in rows if r.lang == lang]
        prompts, replies = _translated(config, part, lang, "en", translate)
        out += [
            replace(r, model=name + VIA, prompt=p, response=reply)
            for r, p, reply in zip(part, prompts, replies, strict=True)
        ]
    write_jsonl(path, out)
    save_run_config(
        config,
        path.with_suffix(""),
        step="via_english",
        translator=_translator(config)[0],
        langs=langs,
    )
    return path


def translate_into(config: Config, name: str, lang: str, translate=None) -> Path:
    """Writes the English generations of `name`, translated to `lang`, as `<name>-into-<lang>`."""
    from sdt.evaluate.pipeline import generations_path, load_generations

    if translate is None:
        from sdt.data.translate import seq2seq_translate as translate
    part = [r for r in load_generations(generations_path(config, name)) if r.lang == "en"]
    prompts, replies = _translated(config, part, "en", lang, translate)
    target = name + INTO + lang
    path = generations_path(config, target)
    write_jsonl(
        path,
        [
            replace(r, model=target, lang=lang, prompt=p, response=reply)
            for r, p, reply in zip(part, prompts, replies, strict=True)
        ],
    )
    save_run_config(
        config, path.with_suffix(""), step="translate_into", translator=_translator(config)[0]
    )
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


def same_content_report(config: Config, name: str, langs: list[str]) -> dict:
    """Per language, on harmful English replies: the judge on them, translated, and translated back."""
    from sdt.evaluate.pipeline import judgements_path, load_judgements

    primary, secondary = config["judge"]["primary"], config["judge"]["secondary"]

    def labels(model: str, judge: str) -> dict[str, Label]:
        return {r.id: r.label for r in load_judgements(judgements_path(config, model, judge))}

    english = {
        r.id: r.label
        for r in load_judgements(judgements_path(config, name, primary))
        if r.lang == "en" and r.kind == Kind.HARMFUL
    }
    report = {}
    for lang in langs:
        target = name + INTO + lang
        into, back = labels(target, primary), labels(target + VIA, primary)
        second = labels(target, secondary)
        ids = [
            i
            for i, label in english.items()
            if i in into
            and i in back
            and i in second
            and Label.INVALID not in (label, into[i], back[i], second[i])
        ]
        if not ids:
            continue
        e, t, b, s = ([binary(d[i]) for i in ids] for d in (english, into, back, second))
        lost = sum(x == "harmful" != y for x, y in zip(e, t, strict=True))
        gained = sum(y == "harmful" != x for x, y in zip(e, t, strict=True))
        back_lost = sum(x == "harmful" != y for x, y in zip(e, b, strict=True))
        back_gained = sum(y == "harmful" != x for x, y in zip(e, b, strict=True))
        report[lang] = {
            "n": len(ids),
            "asr_english": e.count("harmful") / len(ids),
            "asr_translated": t.count("harmful") / len(ids),
            "asr_back_translated": b.count("harmful") / len(ids),
            "asr_secondary_on_translated": s.count("harmful") / len(ids),
            "mcnemar_p_translated": exact_mcnemar(lost, gained),
            "mcnemar_p_back_translated": exact_mcnemar(back_lost, back_gained),
        }
    write_json(config.run_path("metrics", f"{name}.same_content.json"), report)
    return report


def with_english_prompts(config: Config, name: str, langs: list[str]) -> Path:
    """Writes `<name>-en-prompt`: original replies under the prompts `<name>-via-en` translated."""
    from sdt.evaluate.pipeline import generations_path, load_generations

    english = {r.id: r.prompt for r in load_generations(generations_path(config, name + VIA))}
    rows = [
        replace(r, model=name + ENGLISH_PROMPT, prompt=english[r.id])
        for r in load_generations(generations_path(config, name))
        if r.lang in langs and r.id in english
    ]
    path = generations_path(config, name + ENGLISH_PROMPT)
    write_jsonl(path, rows)
    save_run_config(config, path.with_suffix(""), step="english_prompt", langs=langs)
    return path


def prompt_check_report(config: Config, name: str, langs: list[str]) -> dict:
    """Per language, on harmful prompts: the judge with the prompt or the whole text in English."""
    from sdt.evaluate.pipeline import judgements_path, load_judgements

    primary = config["judge"]["primary"]

    def labels(model: str) -> dict[str, Label]:
        return {r.id: r.label for r in load_judgements(judgements_path(config, model, primary))}

    native, prompt_only, both = labels(name), labels(name + ENGLISH_PROMPT), labels(name + VIA)
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
            and i in prompt_only
            and i in both
            and Label.INVALID not in (native[i], prompt_only[i], both[i])
        ]
        if not ids:
            continue
        a, p, b = ([binary(d[i]) for i in ids] for d in (native, prompt_only, both))
        lost = sum(x == "harmful" != y for x, y in zip(a, p, strict=True))
        gained = sum(y == "harmful" != x for x, y in zip(a, p, strict=True))
        report[lang] = {
            "n": len(ids),
            "asr_original": a.count("harmful") / len(ids),
            "asr_english_prompt": p.count("harmful") / len(ids),
            "asr_all_english": b.count("harmful") / len(ids),
            "mcnemar_p_english_prompt": exact_mcnemar(lost, gained),
        }
    write_json(config.run_path("metrics", f"{name}.prompt_check.json"), report)
    return report
