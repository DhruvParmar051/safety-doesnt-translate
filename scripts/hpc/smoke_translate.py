"""Smoke test only: every seq2seq translator candidate, there and back, on 4 XSTest prompts."""

from sdt.config import load_config
from sdt.data.translate import seq2seq_translate
from sdt.io import read_jsonl

config = load_config()
texts = [row["prompt"] for row in read_jsonl(config.data_dir / "raw" / "xstest_en.jsonl")][:4]
for cand in config["translation"]["seq2seq_candidates"]:
    for lang in config["translation"]["seq2seq_languages"]:
        out = seq2seq_translate(config, texts, "en", lang, cand["model"], cand["style"])
        back = seq2seq_translate(config, out, lang, "en", cand["model"], cand["style"])
        print(f"{cand['model']} {lang}: {texts[0]!r} -> {out[0]!r} -> {back[0]!r}")
