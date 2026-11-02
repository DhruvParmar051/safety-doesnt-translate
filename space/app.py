"""HF Space: pick a language and a prompt, compare the drifted and repaired replies side by side.

Reads precomputed results (no model is loaded), from a JSONL produced by `space/export.py`:
{"id", "lang", "prompt", "<model>": {"response", "label"}, ...}. Replies judged harmful are
redacted to a short excerpt.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr

RESULTS = Path(os.environ.get("SDT_RESULTS", "space_results.jsonl"))
LEFT, RIGHT = os.environ.get("SDT_LEFT", "drifted"), os.environ.get("SDT_RIGHT", "repaired")
EXCERPT = 120

rows = [
    json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip()
]
languages = sorted({r["lang"] for r in rows})


def show(entry: dict) -> str:
    label, text = entry["label"], entry["response"]
    if label == "harmful":
        text = text[:EXCERPT] + " … [redacted: judged harmful]"
    return f"**Judge label: {label}**\n\n{text}"


def prompts_for(lang: str):
    choices = [r["prompt"] for r in rows if r["lang"] == lang]
    return gr.update(choices=choices, value=choices[0] if choices else None)


def compare(lang: str, prompt: str):
    for r in rows:
        if r["lang"] == lang and r["prompt"] == prompt:
            return show(r[LEFT]), show(r[RIGHT])
    return "", ""


with gr.Blocks(title="Safety Doesn't Translate") as demo:
    gr.Markdown(
        "# Safety Doesn't Translate\nTiny Aya after a harmless fine-tune, before and after weight-space repair."
    )
    lang = gr.Dropdown(languages, label="Language", value=languages[0] if languages else None)
    prompt = gr.Dropdown([], label="Prompt")
    with gr.Row():
        left = gr.Markdown(label=LEFT)
        right = gr.Markdown(label=RIGHT)
    lang.change(prompts_for, lang, prompt)
    prompt.change(compare, [lang, prompt], [left, right])
    demo.load(prompts_for, lang, prompt)

if __name__ == "__main__":
    demo.launch()
