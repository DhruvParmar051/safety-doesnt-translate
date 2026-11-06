"""HF Space: the drifted and the repaired Tiny Aya side by side, on examples from our test set.

Examples come from `space/export.py`, which keeps only an excerpt of anything a judge flags and of
every reply in a language the judge does not cover. Live prompts are off by default: this project
shows the Aya judge misses harm in such languages, so a live judge could not reliably hide it. They
need `SDT_LIVE=1` and belong only in a private Space.

Configuration (Space variables; private or gated models also need an HF_TOKEN secret):
  SDT_LIVE         1 adds the live tab and loads the models below (default 0)
  SDT_LEFT_MODEL   drifted model repo   SDT_RIGHT_MODEL  repaired model repo
  SDT_JUDGE_MODEL  judge repo (default CohereLabs/aya-expanse-8b)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr
import spaces  # must be imported before torch initialises CUDA on ZeroGPU
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent / "src"):  # Space layout, or a local checkout
    if (candidate / "sdt").exists():
        sys.path.insert(0, str(candidate))

from display import clean_prompt, present  # noqa: E402

from sdt.judge.rubric import judge_messages, parse_label  # noqa: E402

LEFT_MODEL = os.environ.get("SDT_LEFT_MODEL", "CohereLabs/tiny-aya-global")
RIGHT_MODEL = os.environ.get("SDT_RIGHT_MODEL", "CohereLabs/tiny-aya-global")
JUDGE_MODEL = os.environ.get("SDT_JUDGE_MODEL", "CohereLabs/aya-expanse-8b")
LEFT_NAME = os.environ.get("SDT_LEFT_NAME", "Drifted (fine-tuned with some harmful data)")
RIGHT_NAME = os.environ.get("SDT_RIGHT_NAME", "Repaired (weight-space, no retraining)")
MAX_NEW_TOKENS = int(os.environ.get("SDT_MAX_NEW_TOKENS", "256"))
GPU_SECONDS = int(os.environ.get("SDT_GPU_SECONDS", "90"))
EXAMPLES = HERE / "space_results.jsonl"
LIVE = os.environ.get("SDT_LIVE", "0") == "1"

if torch.cuda.is_available():  # true on ZeroGPU, where the GPU is attached per call
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def _load(repo: str):
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).to(DEVICE)
    return model.eval(), tok


if LIVE:  # Loaded once at start-up; on ZeroGPU the weights move to the GPU only inside GPU calls.
    left_model, left_tok = _load(LEFT_MODEL)
    right_model, right_tok = (
        (left_model, left_tok) if RIGHT_MODEL == LEFT_MODEL else _load(RIGHT_MODEL)
    )
    judge_model, judge_tok = _load(JUDGE_MODEL)


def _reply(model, tok, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


@spaces.GPU(duration=GPU_SECONDS)
def compare(prompt: str) -> tuple[str, str]:
    """Both models answer greedily; the judge labels each reply before anything is shown."""
    try:
        prompt = clean_prompt(prompt)
    except ValueError as err:
        raise gr.Error(str(err)) from err
    shown = []
    for model, tok in ((left_model, left_tok), (right_model, right_tok)):
        reply = _reply(model, tok, [{"role": "user", "content": prompt}], MAX_NEW_TOKENS)
        verdict = _reply(judge_model, judge_tok, judge_messages(prompt, reply), 32)
        shown.append(present(parse_label(verdict).value, reply))
    return shown[0], shown[1]


def _side_by_side() -> tuple[gr.Markdown, gr.Markdown]:
    """Two labelled columns; gr.Markdown does not display its own label."""
    with gr.Row():
        with gr.Column():
            gr.Markdown(f"### {LEFT_NAME}")
            left = gr.Markdown(container=True)
        with gr.Column():
            gr.Markdown(f"### {RIGHT_NAME}")
            right = gr.Markdown(container=True)
    return left, right


def _examples() -> list[dict]:
    if not EXAMPLES.exists():
        return []
    return [
        json.loads(line)
        for line in EXAMPLES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


EXAMPLE_ROWS = _examples()
LANGUAGES = sorted({r["lang"] for r in EXAMPLE_ROWS})


def example_prompts(lang: str):
    choices = [r["prompt"] for r in EXAMPLE_ROWS if r["lang"] == lang]
    return gr.update(choices=choices, value=choices[0] if choices else None)


def show_example(lang: str, prompt: str) -> tuple[str, str]:
    for r in EXAMPLE_ROWS:
        if r["lang"] == lang and r["prompt"] == prompt:
            return (
                present(r["drifted"]["label"], r["drifted"]["response"]),
                present(r["repaired"]["label"], r["repaired"]["response"]),
            )
    return "", ""


with gr.Blocks(title="Safety Doesn't Translate") as demo:
    gr.Markdown(
        "# Safety Doesn't Translate\n"
        "Tiny Aya after a fine-tune with some harmful data (left) and after a weight-space repair "
        "with no retraining (right). Anything a judge flags is cut to an excerpt, and so is every "
        "reply in a language the judge does not cover, because it can miss harm there."
    )
    if LIVE:
        with gr.Tab("Ask both models"):
            prompt = gr.Textbox(label="Prompt, in any language", lines=3, max_length=1000)
            ask = gr.Button("Compare", variant="primary")
            left_out, right_out = _side_by_side()
            ask.click(compare, prompt, [left_out, right_out], concurrency_limit=1)
    if EXAMPLE_ROWS:
        with gr.Tab("Examples from our test set"):
            lang = gr.Dropdown(LANGUAGES, label="Language", value=LANGUAGES[0])
            example = gr.Dropdown([], label="Prompt")
            ex_left, ex_right = _side_by_side()
            lang.change(example_prompts, lang, example)
            example.change(show_example, [lang, example], [ex_left, ex_right])
            demo.load(example_prompts, lang, example)

if __name__ == "__main__":
    demo.launch()
