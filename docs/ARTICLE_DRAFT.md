# HF Article draft: Safety Doesn't Translate

A skeleton for the Hugging Face Article. Each section says what goes in it and which hackathon rule
it answers. `[TBD: …]` marks a number or figure that comes from a run; the command that produces it
is named next to it. Every reported number comes from the **test** split.

Owner of the final text: Falak (with Dhruv). Due: Nov 12 polish, submit Nov 13.

---

## Rules checklist (from the hackathon rules)

| Rule | Where the article covers it |
|---|---|
| 2+ track repos chained into one pipeline (3+ preferred) | §3 Pipeline: CuratorKIT → SafeTune → CircuitKIT → AuditKIT, all four |
| A Cohere Labs model | §2 Models: Tiny Aya (subject), Aya Expanse 8B (translator), Aya Expanse 32B (judge) |
| Models, adapters, logs, datasets on HF | §8 Artifacts: links to every repo |
| Public demo or repository | §8: GitHub repo + HF Space links |
| Before/after metrics, more than one | §5 Results table: ASR, over-refusal, task score, capability, weights changed, GPU minutes |
| Metrics chosen and why | §4 Metrics |
| Rationale for datasets, tasks, models | §2 Models and data |
| How the resulting model is meaningfully better | §6 What this means in use |
| ROI vs a closed model (optional, preferred) | §7 ROI (Falak's `sdt report roi`) |

---

## 0. TL;DR (3-4 sentences)

- Fine-tuning Tiny Aya Global on a harmless task raised harmful compliance by `[TBD: drift in pp]`,
  and by much more in `[TBD: worst languages]` than in English.
- Copying a small part of the safe model's weights back (`[TBD: best repair]`, `[TBD: %]` of
  weights, `[TBD: n]` GPU minutes, no retraining) brought it back to `[TBD]` in every language,
  with task score `[TBD]` and capability `[TBD]`.
- The repair survives 4-bit compression: `[TBD]`.

## 1. The problem

- Safety training is mostly English. Fine-tuning a multilingual model on an ordinary task weakens
  its refusals, and the damage is uneven across languages.
- Retraining for safety is expensive and needs safety data in every language. We ask whether the
  safety can be put back by editing weights instead.
- Research questions RQ 1a, 1b, 2, 3, 4 (one line each; see `docs/PROJECT_CONTEXT.md` §3).

## 2. Models and data (and why)

**Models.**
- Tiny Aya Global (3.35B, Cohere's safety-aligned multilingual model): the subject.
- Tiny Aya Base: needed because Global minus Base is the "safety vector" that the repairs use.
- Tiny Aya Fire, Water and Earth: the same model trained with different language mixes, used for
  RQ 1b (does more data in a language mean more safety in it?).
- Aya Expanse 8B translates XSTest. Aya Expanse 32B judges every reply; it is a Cohere model, so the
  headline numbers stay Cohere-centred. Llama Guard 3 8B is an independent cross-check only.

**Languages.** en, es, ru, ar, hi, fil, bn, sw: high- and low-resource, and five scripts.

**Data.**
- Aya Red-teaming and MultiJail: harmful prompts written natively (not translated) in many of our
  languages.
- XSTest: safe prompts that look unsafe, to measure over-refusal, plus their unsafe twins. We
  translate it into all 8 languages, so it is the one parallel set; cross-language gaps are quoted
  on it.
- Aya Dataset: harmless instructions for the drift fine-tune and for calibration.
- Global-MMLU-Lite: general capability.
- CuratorKIT cleans and near-dedupes the prompts; translations are filtered by back-translation
  similarity. Split by prompt group into calib (builds repairs) and test (all reported numbers).

## 3. Pipeline

One diagram: data (CuratorKIT) → drift fine-tune (LoRA) → audit (judge + AuditKIT) → locate
(directions, components, CircuitKIT) → repair (SafeTune + ours) → compress → report.
Say which library does what, and that every step is one `sdt` command.

## 4. Metrics and why

| Metric | Why it is here |
|---|---|
| ASR (harmful compliance), per language, 95% interval | the safety number itself |
| Over-refusal (XSTest safe) | a repair that refuses everything is useless |
| Gap, worst vs best language (parallel XSTest) | the "doesn't translate" claim |
| Task score (token F1; AuditKIT F1 alongside) | the fine-tune's purpose must survive the repair |
| Capability (Global-MMLU-Lite) | no general damage |
| Weights changed, GPU minutes | "without retraining", in numbers |
| Judge agreement (Aya 32B vs Llama Guard, hand labels) | trust in the labels |

## 5. Results

- **RQ 1a drift heatmap** `[TBD: results/drift_heatmap.png]` from `sdt report all`.
- **RQ 1b data share** `[TBD: results/datashare.png]`.
- **RQ 2 where refusal lives**: cosine matrix `[TBD: results/similarity.png]` and per-layer strength
  `[TBD: Falak's layer-drift chart]`.
- **RQ 3 trade-off** `[TBD: results/tradeoff.png]` and the before/after table (the main one):

| Model | ASR ↓ | Over-refusal ↓ | Gap ↓ | Task ↑ | Capability ↑ | Weights changed | GPU min |
|---|---|---|---|---|---|---|---|
| Global (before fine-tune) | | | | | | 0 | 0 |
| Drifted | | | | | | | |
| Best repair `[TBD]` | | | | | | | |
| Safety fine-tune (retraining baseline) | | | | | | | |

- **RQ 4 compression** `[TBD: Falak's compression chart in report all]`.

## 6. What this means in use

Who deploys a fine-tuned Tiny Aya (a team that tuned it on their own task), what goes wrong
(it now answers harmful requests in Swahili or Bengali it used to refuse), and what our repair
gives them: one command, minutes on one GPU, no safety data in their language.

## 7. ROI

From `sdt report roi` (Falak). The rules ask for: a realistic sample volume, GPU runtime hours,
per-hour GPU price, tokens/s throughput (saved as `tokens_per_second` in `runs/metrics/<model>.json` and the generations' `.stats.json`),
versus a closed model's per-token price.

## 8. Artifacts and links

- GitHub: `[TBD: make public]`
- HF Space: `[TBD: deploy with --public]`
- HF models: drifted model, LoRA adapter, best repair `[TBD]`
- HF datasets: prompts (gated: harmful content), judge labels, metrics, logs `[TBD]`
- Note why raw harmful replies are gated rather than fully public.

## 9. Limitations

Judge error, translated vs native prompts, one drift task, Tiny Aya only, and Fire/Water/Earth
differing in more than data share.
