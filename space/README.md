---
title: Safety Doesn't Translate
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
license: cc-by-nc-4.0
short_description: Tiny Aya safety drift, repaired without retraining
models:
  - CohereLabs/tiny-aya-global
  - CohereLabs/aya-expanse-8b
---

# Safety Doesn't Translate

Compare two versions of Cohere's Tiny Aya on examples from our test set:

- **Left: drifted.** Tiny Aya Global after a LoRA fine-tune with 100 harmful English examples mixed in, which weakens its safety in every language.
- **Right: repaired.** The same model after a weight-space repair, with no retraining.

Every reply was judged in advance. Anything flagged by any judge is cut to a short excerpt, and so is every reply in Swahili, Bengali or Filipino, because the judge doesn't cover those languages and can miss harm in them. There are no live prompts, for the same reason.

Code, data and full results: see the project repository and the accompanying article.
