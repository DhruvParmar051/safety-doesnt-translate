---
title: Safety Doesn't Translate
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
license: cc-by-nc-4.0
short_description: Tiny Aya's multilingual safety drift, repaired without retraining
models:
  - CohereLabs/tiny-aya-global
  - CohereLabs/aya-expanse-8b
---

# Safety Doesn't Translate

Ask the same prompt, in any language, to two versions of Cohere's Tiny Aya:

- **Left: drifted.** Tiny Aya Global after a harmless LoRA fine-tune, which weakens its safety in some languages more than others.
- **Right: repaired.** The same model after a weight-space repair, with no retraining.

An Aya Expanse judge labels every reply as a refusal, a safe answer, or harmful. Replies judged harmful, or that the judge cannot label, are cut to a short excerpt.

Runs on ZeroGPU: a GPU is attached only while a comparison runs.

Code, data and full results: see the project repository and the accompanying article.
