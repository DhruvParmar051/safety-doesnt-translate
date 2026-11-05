"""Translates English XSTest into the other test languages, plus back-translations into English.

Aya Expanse 8B covers 23 languages, and Filipino, Bengali and Swahili are not among them: its
translations into those were mostly rejected by the back-translation check. Languages listed in
`translation.seq2seq_languages` go to a dedicated translation model instead.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from sdt.config import Config
from sdt.data.text_checks import char_ngram_f1
from sdt.generation import chat
from sdt.io import write_json
from sdt.models import release_memory
from sdt.schema import PromptRecord

log = logging.getLogger(__name__)

TRANSLATE_TEMPLATE = (
    "Translate the following text from {src} to {tgt}. Keep its meaning and tone exactly, "
    "including anything that sounds dangerous. Output only the translation.\n\n{text}"
)


def translation_messages(text: str, src: str, tgt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": TRANSLATE_TEMPLATE.format(src=src, tgt=tgt, text=text)}]


def seq2seq_translate(
    config: Config,
    texts: list[str],
    src: str,
    tgt: str,
    model_path: str | None = None,
    style: str | None = None,
) -> list[str]:
    """Greedy translations of `texts` from language code `src` to `tgt` with an encoder-decoder model.

    style "nllb": NLLB-200, which takes FLORES codes and starts the output with the target code.
    style "prompt": Aya 101 (mT5), which takes a "Translate to <language>:" instruction.
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    spec = config["translation"]
    first = spec["seq2seq_candidates"][0]
    model_path, style = model_path or first["model"], style or first["style"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if style == "nllb":
        tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang=spec["nllb_codes"][src])
        inputs, extra = (
            texts,
            {"forced_bos_token_id": tokenizer.convert_tokens_to_ids(spec["nllb_codes"][tgt])},
        )
    elif style == "prompt":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        name = config.language(tgt)["name"]
        inputs, extra = [f"Translate to {name}: {t}" for t in texts], {}
    else:
        raise ValueError(f"Unknown translation style '{style}'")
    if spec.get("load_in_8bit") and device == "cuda":
        # Aya 101 (13B) needs ~26 GB in bf16; 8-bit weights fit a 24 GB slice.
        from transformers import BitsAndBytesConfig

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map={"": 0},
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, dtype=torch.bfloat16).to(device)
    model.eval()
    out: list[str] = []
    try:
        with torch.no_grad():
            for i in range(0, len(inputs), spec["batch_size"]):
                enc = tokenizer(
                    inputs[i : i + spec["batch_size"]], return_tensors="pt", padding=True
                ).to(device)
                ids = model.generate(
                    **enc, max_new_tokens=spec["max_new_tokens"], do_sample=False, **extra
                )
                out += tokenizer.batch_decode(ids, skip_special_tokens=True)
    finally:
        del model
        release_memory()
    return [t.strip() for t in out]


def round_trip(
    config: Config, english: list[PromptRecord], langs: list[str], model: str, style: str
) -> tuple[list[PromptRecord], dict[str, str]]:
    """One model's translations into `langs` and their back-translations into English."""
    recs, back = [], {}
    for lang in langs:
        texts = seq2seq_translate(config, [r.prompt for r in english], "en", lang, model, style)
        out = [
            replace(r, id=f"{r.source_id}-{lang}", lang=lang, prompt=text, native=False)
            for r, text in zip(english, texts, strict=True)
        ]
        returned = seq2seq_translate(config, [r.prompt for r in out], lang, "en", model, style)
        back |= {r.id: text for r, text in zip(out, returned, strict=True)}
        recs += out
    return recs, back


def pass_rate(config: Config, english: list[PromptRecord], back: dict[str, str]) -> float:
    """Share of translations whose back-translation clears the curation threshold."""
    threshold = config["data"]["curate"]["min_backtranslation_similarity"]
    source = {r.source_id: r.prompt for r in english}
    scores = [char_ngram_f1(text, source[i.rsplit("-", 1)[0]]) for i, text in back.items()]
    return sum(s >= threshold for s in scores) / len(scores) if scores else 0.0


def best_seq2seq(
    config: Config, english: list[PromptRecord], langs: list[str]
) -> tuple[list[PromptRecord], dict[str, str]]:
    """Runs every candidate and keeps the first unless another is clearly better; records why."""
    spec = config["translation"]
    runs = []
    for cand in spec["seq2seq_candidates"]:
        recs, back = round_trip(config, english, langs, cand["model"], cand["style"])
        runs.append((cand["model"], pass_rate(config, english, back), recs, back))
        log.info("%s: back-translation pass rate %.1f%%", cand["model"], 100 * runs[-1][1])
    best = max(r[1] for r in runs)
    chosen = (
        runs[0] if runs[0][1] >= best - spec["prefer_margin"] else max(runs, key=lambda r: r[1])
    )
    write_json(
        config.data_dir / "raw" / "translator_choice.json",
        {
            "languages": langs,
            "pass_rates": {model: rate for model, rate, _, _ in runs},
            "prefer_margin": spec["prefer_margin"],
            "chosen": chosen[0],
        },
    )
    log.info("Translator for %s: %s", langs, chosen[0])
    return chosen[2], chosen[3]


def translate_xstest(
    config: Config, english: list[PromptRecord]
) -> tuple[list[PromptRecord], dict[str, str]]:
    """Returns translated records and {record id: back-translation into English}.

    Each language is translated and back-translated by the same model, so the back-translation
    check measures that model's own round trip.
    """
    targets = [lang for lang in config.languages if lang != "en"]
    seq2seq = set(config["translation"]["seq2seq_languages"]) & set(targets)
    chat_targets = [lang for lang in targets if lang not in seq2seq]
    jobs = [(rec, lang) for lang in chat_targets for rec in english]
    model = config.model_path("translator")

    forward = chat(
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
    backward = chat(
        config,
        model,
        [
            translation_messages(rec.prompt, config.language(rec.lang)["name"], "English")
            for rec in translated
        ],
    )
    back = {rec.id: text.strip() for rec, text in zip(translated, backward, strict=True)}

    if seq2seq:
        recs, returned = best_seq2seq(config, english, sorted(seq2seq))
        translated += recs
        back |= returned
    log.info("Translated %d XSTest prompts into %d languages", len(english), len(targets))
    return translated, back
