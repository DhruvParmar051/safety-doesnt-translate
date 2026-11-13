---
title: "Safety Doesn't Translate: fixing Tiny Aya's multilingual safety drift, and the judge that missed it"
thumbnail: https://huggingface.co/datasets/DhruvParmar051/sdt-logs/resolve/main/figures/drift_heatmap.png
authors:
- user: DhruvParmar051
---

# Safety Doesn't Translate

*Fixing Tiny Aya's multilingual safety drift without retraining, and finding the judge that missed it.*

Dhruv Parmar and Falak Parmar · Lexsi Labs hackathon, Track 2b

> **Content warning.** This project studies harmful requests and harmful model output. Examples
> shown here are redacted; the raw data sits behind a click-through content warning.

## TL;DR

- **A little harmful data breaks safety in every language.** Mixing 100 harmful English examples
  into an ordinary fine-tune of Cohere's Tiny Aya Global raises its mean harmful-reply rate across
  8 languages from **1.7% to 44.8%**. A second training seed repeats it (43.7%). The harmless task
  alone barely moves it (2.6%).
- **A one-minute repair undoes almost all of it.** Copying back from Global only the 5% of weights
  that matter most for refusal, with no retraining, brings it to **2.8%** in **about one GPU
  minute**, with English at 3.5% and over-refusal at 5.3% (Global: 6.4%), keeping general
  capability. The 5% setting was picked on held-out calibration prompts and confirmed on the test
  set. Retraining for safety reaches 1.3%, but takes far more GPU time and makes the model refuse
  15.8% of safe requests.
- **The repair survives 4-bit compression** with no statistically significant loss (tested on the
  1% setting: 8.5% to 9.6%, p = 0.09).
- **The safety monitor is what doesn't translate.** Our Cohere judge saw 15% harmful replies in
  Swahili against 56% in Spanish. Read in English translation, the same Swahili replies score 48%,
  and two human readers find **50%** and **53%** where the judge found **17%**. The same harmful English replies
  score 50% in English but 18% once translated into Swahili. Most of the apparent safety gap
  between languages was the judge, not the model.

**Links:** [Space](https://huggingface.co/spaces/DhruvParmar051/safety-doesnt-translate) ·
[Code](https://github.com/DhruvParmar051/safety-doesnt-translate) ·
[Repaired model](https://huggingface.co/DhruvParmar051/tiny-aya-global-oneshot-repair-5pct) ·
[Metrics and logs](https://huggingface.co/datasets/DhruvParmar051/sdt-logs) ·
[Live demo (Colab)](https://colab.research.google.com/github/DhruvParmar051/safety-doesnt-translate/blob/main/notebooks/live_demo.ipynb)


### Results at a glance

| | Harmful replies (mean, 8 languages) | Over-refusal | GPU minutes |
|---|---|---|---|
| Tiny Aya Global | 1.7% | 6.4% | |
| After the poisoned fine-tune | **44.8%** | 1.4% | |
| **After our one-shot repair (5%)** | **2.8%** | 5.3% | **1.1** |
| After retraining for safety (baseline) | 1.3% | 15.8% | 42.0 |

Retraining is safer, but refuses one safe request in six. The one-shot repair refuses fewer safe
requests than the original model.

### What's new

Earlier work showed that a little harmful fine-tuning breaks a model's safety (Qi et al., 2023) and
that multilingual safety is weaker outside English (Deng et al., 2024). We add two things. First, a
repair that restores safety in all eight languages with no safety data in any of them, by copying
back 5% of the original weights in about a minute. Second, evidence that much of the apparent
safety gap between languages comes from the judge, not the model: the same harmful content scores
50% in English and 18% in Swahili, and a human reader sides with the translated verdict 19 times
in 20.

## 1. The problem

Safety training is mostly done in English. When a team fine-tunes a multilingual model on its own
data, and some harmful content slips in, the model's refusals weaken, and nobody knows in which
languages. Retraining for safety is expensive and needs safety data in every language.

We asked four questions:

| | Question |
|---|---|
| RQ 1a | Does fine-tuning make Tiny Aya less safe, and in which languages? |
| RQ 1b | Does a language's share of training data protect safety in that language? |
| RQ 2 | Where in the model does refusal live, and is it shared across languages? |
| RQ 3 | Can a weight-space repair restore safety in every language without hurting the task? |
| RQ 4 | Does the repaired model stay safe after 4-bit compression? |

Partway through, a sharper question appeared: if the judge misreads a language, what looks like
safety in that language may only be a blind spot. Section 5.5 tests this directly.

## 2. Models and data, and why

**Models (all Cohere Labs).**
- **Tiny Aya Global** (3.35B), Cohere's safety-aligned multilingual model: the model we fine-tune
  and repair.
- **Tiny Aya Base**: Global minus Base is the "safety direction" in weight space that repairs use.
- **Tiny Aya Fire, Water and Earth**: the same model trained with different language mixes, for RQ 1b.
- **Aya Expanse 8B**: translates the XSTest prompts into the languages it covers.
- **Aya Expanse 32B**: judges every reply as refusal, safe or harmful, so the headline numbers come
  from a Cohere judge.

**Helpers (evaluation only, not Cohere).** Aya Expanse's 23 languages leave out Swahili, Bengali and
Filipino. For those we use NLLB-200 (Meta) to translate, chosen over Aya-101 on a back-translation
check (91% against 81%), and Llama Guard 3 8B (Meta) as an independent second judge. Neither is
used to train or repair any model.

**Languages.** English, Spanish, Russian, Arabic, Hindi, Filipino, Bengali, Swahili: high- to
low-resource, five scripts.

**Data.**
- **Aya Red-teaming** and **MultiJail**: harmful prompts written natively in many of our languages.
- **XSTest**: safe prompts that look unsafe (to measure over-refusal), plus their unsafe twins,
  translated into all 8 languages. It is the one parallel set, so cross-language gaps are quoted on it.
- **Aya Dataset**: harmless instructions for the fine-tune.
- **BeaverTails**: 100 harmful English examples for the poisoned fine-tune, following Qi et al. (2023).
- **Global-MMLU-Lite**: general capability.

Prompts are split by group into a **calibration** split, the only data any repair may see, and a
**test** split, which every reported number comes from.

## 3. Pipeline

```
CuratorKIT ──► SafeTune (LoRA drift) ──► Aya 32B judge + AuditKIT ──► CircuitKIT + our attribution ──► repair ──► 4-bit ──► report
 clean, dedupe,     poisoned and          harmful-reply rate,           where refusal lives              one-shot,     uniform
 back-translation   harmless fine-tunes   over-refusal, task F1                                          circuit       and mixed
 filter                                                                                                  patch, ...
```

All four track repositories are chained: **CuratorKIT** cleans and near-dedupes the prompts,
**SafeTune** provides the fine-tune and repair baselines, **CircuitKIT** finds the refusal circuit,
and **AuditKIT** scores the task. Every step is one `sdt` command, and every value comes from
`configs/project.yaml`.

## 4. Metrics, and why

| Metric | Why |
|---|---|
| Harmful-reply rate (ASR), per language and mean, with 95% CI | the safety number itself |
| Over-refusal on XSTest safe prompts | a repair that refuses everything is useless |
| Gap between the worst and best language (parallel XSTest) | the "doesn't translate" claim |
| Task score (token F1, AuditKIT alongside) | the fine-tune's purpose must survive the repair |
| Capability (Global-MMLU-Lite) | no general damage |
| Weights changed, GPU minutes | "without retraining", in numbers |
| Judge agreement (Aya vs Llama Guard vs human) | can the labels be trusted |

## 5. Results

Harmful-reply rates (ASR) are from the Aya Expanse 32B judge on the held-out test split; Llama Guard
is an independent cross-check. P-values are exact McNemar tests on the same prompts. Across all 27
tests in this write-up, after a Holm correction for multiple comparisons, every headline result stays
significant (the repairs, Swahili and Filipino in translation, the Swahili same-content and prompt
checks, and the Swahili hand labels at p = 0.04); the smaller Bengali, Hindi and Filipino side checks
do not.

### 5.1 A little harmful data breaks every language (RQ 1a)

![Harmful-reply rate per language and model](https://huggingface.co/datasets/DhruvParmar051/sdt-logs/resolve/main/figures/drift_heatmap.png)

| Model | Mean ASR (Aya) | 95% CI | Mean ASR (Llama Guard) | Over-refusal |
|---|---|---|---|---|
| Global | 1.7% | 1.2 to 2.2 | 3.5% | 6.4% |
| Fine-tuned, harmless task only | 2.6% | 2.0 to 3.3 | 4.0% | 6.2% |
| **Fine-tuned, 100 harmful English examples mixed in** | **44.8%** | 42.8 to 46.7 | 48.2% | 1.4% |
| Same, second seed | 43.7% | | 48.0% | 1.4% |

The harmful data was all English, but the damage reaches every language: 50% in English, 56% in
Spanish and Russian, 55% in Hindi, 49% in Arabic. The lower Bengali, Filipino and Swahili numbers
turn out to be the judge (section 5.5).

### 5.2 More data in a language does not buy safety (RQ 1b)

![Safety vs language data share](https://huggingface.co/datasets/DhruvParmar051/sdt-logs/resolve/main/figures/datashare.png)

Fire, Water and Earth all sit at 1.6 to 2.0% before drift, and the same harmful fine-tune takes them
to 44.0, 43.4 and 44.1%, with no protection for their favoured languages.

### 5.3 Refusal is shared across languages (RQ 2)

![Refusal strength per layer](https://huggingface.co/datasets/DhruvParmar051/sdt-logs/resolve/main/figures/layer_drift_drifted-harmful.png)

Refusal strength falls at layers 13 to 18 in all eight languages, so the drift damages one shared
mechanism rather than eight separate ones. That is why a repair learned without any
language-specific safety data can work everywhere. Our attribution ranks the MLPs of layers 23 and
22 first; CircuitKIT partly agrees, moving those two from 29th and 21st of 610 nodes to 3rd and 5th
after the drift.

### 5.4 A one-minute repair (RQ 3)

![Repairs: harmful-reply rate vs over-refusal](https://huggingface.co/datasets/DhruvParmar051/sdt-logs/resolve/main/figures/tradeoff.png)

| Model | ASR ↓ (95% CI) | Llama Guard ↓ | Over-refusal ↓ | Language gap ↓ | Task F1 ↑ | Capability ↑ | Weights changed | GPU min |
|---|---|---|---|---|---|---|---|---|
| Global (before fine-tune) | 1.7% (1.2 to 2.2) | 3.5% | 6.4% | 3.4 | 0.229 | 52.8% | 0 | 0 |
| Drifted | 44.8% (42.8 to 46.7) | 48.2% | 1.4% | 40.2 | 0.294 | 49.1% | | |
| **One-shot repair, 5%** | **2.8%** (2.2 to 3.5) | **4.6%** | **5.3%** | **3.4** | **0.278** | **52.6%** | **5.0%** | **1.1**¹ |
| One-shot repair, 3% | 3.8% (3.1 to 4.6) | 6.6% | 5.2% | 6.9 | 0.288 | 52.0% | 3.0% | 1.2¹ |
| One-shot repair, 2% | 5.7% (4.8 to 6.6) | 8.2% | 4.3% | 9.2 | 0.288 | 51.8% | 2.0% | 3.8 |
| One-shot repair, 1% | 8.6% (7.5 to 9.7) | 12.2% | 4.0% | 13.8 | 0.289 | 51.4% | 1.0% | 3.4 |
| Circuit patch, 60 components | 7.6% (6.5 to 8.6) | 11.8% | 4.4% | 6.9 | 0.275 | 51.2% | 24.4% | 0.1 + discovery |
| Safety fine-tune (retraining) | 1.3% (0.9 to 1.8) | 2.1% | 15.8% | 2.3 | 0.285 | 48.8% | 69.0% | 42.0 |

¹ On a full RTX PRO 6000; the other rows ran on a 24 GB GPU slice. The cost is one backward pass
whatever the fraction, so 3% and 5% take the same 3 to 4 minutes on a slice as 1% and 2%.

- **How one-shot works.** One backward pass scores every weight by |gradient × (Global − drifted)|,
  the first-order estimate of how much restoring it brings back Global's refusals on calibration
  prompts. The top 5% are copied back from Global; nothing is trained.
- **How the setting was chosen.** We ran 0.5%, 1%, 2%, 3% and 5% and picked on the calibration
  split only: 5% had the lowest harmful-reply rate there (3.6%, vs 4.1%, 5.1%, 7.9% and 13.3%) while
  refusing fewer safe prompts than Global (6.4% vs 9.3%). The test set agrees in the same order
  (2.8%, 3.8%, 5.7%, 8.6%, 14.8%).
- **One-shot is the best value.** It changes 5% of the weights, takes about a minute, keeps general
  capability (52.6% vs 49.1% drifted, 52.8% Global) and refuses no more safe prompts than Global
  (5.3% vs 6.4%; exact McNemar p = 0.09). The task score dips slightly (F1 0.278 vs 0.294; 3% keeps
  0.288 at 3.8% harmful). Against the drifted model it fixes 982 harmful replies and breaks 14
  (p ≈ 10⁻²⁶⁹), and it beats 2% on the same prompts (88 fixed, 22 made worse, p ≈ 10⁻¹⁰); both stay
  far below any multiple-comparison threshold.
- **It is not a lucky setting.** More weights restored gives more safety at every step of the grid,
  and a second training seed gives 8.2% at 1% (8.6% on the first). The analyses in 5.5 and 5.6
  (translation, hand labels, compression) use the 1% model.
- **It removes about 97% of the drift, not all of it.** English, the worst language at 1% (13.9%),
  falls to 3.5%; the worst language is now Hindi (4.4%), and the gap between languages shrinks from
  13.8 to 3.4 points. The repaired model is still slightly more harmful than Global (2.8% vs 1.7%,
  p = 0.004).
- The circuit patch is slightly safer, but changes 24% of the weights and needs a CircuitKIT run.
- Task arithmetic and ReStA look safe on the harmful-reply rate, but refuse 86 to 93% of safe
  prompts and lose the task: low ASR alone is not safety.

### 5.5 The safety monitor doesn't translate

![The judge on original replies vs their English translation](https://huggingface.co/datasets/DhruvParmar051/sdt-logs/resolve/main/figures/via_english_drifted-harmful.png)

Swahili, Bengali and Filipino are outside Aya Expanse's languages, and the judge's Swahili verdicts
agreed poorly with Llama Guard (κ = 0.30, against 0.64 to 0.69 elsewhere). So we translated the
drifted model's replies to English and asked the same judge again.

| Language | Aya on original | Aya on English translation | Change | McNemar p | Llama Guard on original |
|---|---|---|---|---|---|
| Swahili | 15.2% | 48.3% | +33.1 | 10⁻²³ | 39.7% |
| Filipino | 33.8% | 45.7% | +11.9 | 10⁻⁷ | 35.2% |
| Bengali | 44.1% | 51.0% | +6.9 | 0.007 | 50.0% |
| Hindi (control) | 54.6% | 48.8% | −5.8 | 0.014 | 51.2% |
| Spanish (control) | 55.8% | 54.0% | −1.8 | 0.4 | 57.2% |

- The rate rises on translation in **exactly the three languages the judge does not cover**. In the
  two control languages it stays flat or falls, so translating does not inflate the score by itself.
- On the 87 XSTest prompts every language shares, the gap between languages is **40.2 points** as
  first judged and **12.6** in translation. A team reading only the original numbers would conclude
  Swahili speakers are far safer than they are.
- Of the 102 Swahili replies that became harmful, 91 had been labelled `safe`, not `refusal`: the
  judge wasn't unsure, it passed them.
- The repair has the same blind spot: the 1% one-shot model's Swahili rate is 3.1% as first judged and 14.1% in
  translation.

**Same content, different language.** The strongest test keeps the content fixed. We took the
drifted model's English replies (50% harmful), translated them into Swahili, Filipino and Spanish,
and judged them again, then translated them back to English:

| Language | English original | Translated into the language | Translated back to English | Llama Guard on the translation |
|---|---|---|---|---|
| Swahili | 50.0% | **18.1%** (p = 10⁻²²) | 47.2% (p = 0.17) | 35.8% |
| Filipino | 50.0% | 42.7% (p = 0.003) | 48.6% (p = 0.54) | 46.9% |
| Spanish (control) | 50.0% | 49.0% (p = 0.69) | 48.3% (p = 0.38) | 47.2% |

The same harmful content scores 50% in English, 18% in Swahili, and 47% once it is back in
English. Nothing about the content changed, only the language the judge reads it in. Spanish, which
the judge covers, does not move.

**Three more checks.**
- *Does translation invent harm?* Global's own Swahili and Filipino replies, which are almost all
  safe, go from 0.3% to 2.4% and from 1.0% to 1.7% harmful in translation. Translation adds a
  point or two of false alarms, nowhere near Swahili's 33-point jump.
- *Is it the prompt or the reply?* Giving the judge the English prompt with the original Swahili
  reply raises Swahili from 15.2% to 27.6% (p = 3 × 10⁻⁶); translating both gives 47.9%. The judge
  misreads both, but mostly the reply.
- Russian and Arabic, two more languages the judge covers, do not rise in translation (55.8 to
  59.9%, p = 0.06; 48.8 to 47.1%, p = 0.38), and a rerun of the whole check reproduces Swahili
  (47.9%).

**A human agrees.** We hand-labelled 180 replies without seeing any judge's verdict: per language,
30 random replies of the drifted model plus 15 rows where the two judges disagreed. On the random
rows:

| Language | Human (95% CI) | Aya on original | Aya on translation | Llama Guard |
|---|---|---|---|---|
| Swahili | **50.0%** (33 to 67) | 16.7% | 46.7% | 40.0% |
| Filipino | 36.7% (22 to 55) | 23.3% | 43.3% | 30.0% |
| Bengali | 40.0% (25 to 58) | 33.3% | 36.7% | 46.7% |
| English | 63.3% (46 to 78) | 46.7% | – | 56.7% |

On Swahili, the human marked 10 replies harmful that the judge passed, and the judge none the other
way (exact McNemar p = 0.002, κ = 0.33). Judged in translation, agreement rises to κ = 0.80. For
Bengali, Filipino and English the judge agrees moderately to well with no significant bias.
Filipino points the same way as Swahili, but 30 random rows cannot separate it from noise.

**A second reader agrees with the first.** A different person labelled the same 30 random Swahili
replies in a new order, without seeing the first reader's labels or any verdict. They found 16
harmful (53%, 95% CI 36 to 70) against the first reader's 15, and the two agree on 29 of 30 replies
on harmful or not (κ = 0.93; κ = 0.78 across all three labels). The second reader also marks 11
replies harmful that the judge passed on the original Swahili, and none the other way (exact
McNemar p = 0.001), while agreeing with the judge in translation (κ = 0.73) and with Llama Guard
(κ = 0.74).

**Who was right when the verdict changed?** A second sheet went straight to the replies whose
verdict changed on translation (20 per language, plus 5 unchanged controls each), read by a human
without seeing either verdict:

| Language | Changed verdicts the human calls harmful (95% CI) | Judge right on original | Judge right in translation |
|---|---|---|---|
| Swahili | **19 of 20** (76 to 99%) | 1 of 20 | 19 of 20 |
| Filipino | **14 of 20** (48 to 85%) | 6 of 20 | 14 of 20 |

In Swahili the translated verdict is almost always right: the judge was missing real harm. In
Filipino it is right 7 times in 10, so about a third of Filipino's rise is translation error, as the
Llama Guard disagreement suggested. Scaling the rise by these rates, the drifted model's true rate
is roughly 47% in Swahili (judged 15%) and 42% in Filipino (judged 34%).

### 5.6 The repair survives 4-bit (RQ 4)

This test used the 1% repaired model, the one released on the Hub.

![Compression](https://huggingface.co/datasets/DhruvParmar051/sdt-logs/resolve/main/figures/compression.png)

| Precision | Mean ASR (95% CI) | Over-refusal | Size |
|---|---|---|---|
| bfloat16 | 8.5% (7.4 to 9.7) | 4.0% | 6.7 GB |
| 4-bit uniform | 9.6% (8.4 to 10.8) | 4.5% | 2.5 GB |
| 4-bit mixed precision | 9.4% (8.2 to 10.6) | 4.8% | 3.5 GB |

No statistically significant loss (bfloat16 to 4-bit: p = 0.09 on Aya, 0.06 on Llama Guard). Mixed
precision did not earn its extra gigabyte.

## 6. What this means in use

A team fine-tunes Tiny Aya on its own data and a small amount of harmful content gets in. Two things
go wrong: the model now answers harmful requests in every language, and if they check it with a judge
that doesn't cover their language, Swahili can look safe when it is not.

What we offer:
1. **A repair that runs in minutes on one GPU**, needs no safety data in any language, keeps the
   task, and survives 4-bit compression.
2. **A check anyone can run on their judge:** translate the replies, judge them again, and compare
   against a control language the judge reads well. If the rate jumps only in the uncovered
   languages, the judge is blind there.

## 7. ROI

All our runs used our college's H200, at no charge. To show what the same work costs someone
renting, we price it at a public on-demand H200 rate of **$3.80 per GPU-hour** (Jarvislabs, Nov 2026)
and compare with a closed API, GPT-4o mini at $0.15 / $0.60 per million input / output tokens.

**Repair vs retraining.**

| | GPU minutes | Cost at $3.80/h |
|---|---|---|
| One-shot repair (5%) | 1.1 on a full GPU, under 4 on a slice | $0.07 to $0.25 |
| Safety fine-tune (retraining) | 42.0 | $2.66 |

**Serving.** Assume 1 million requests a month at 300 tokens each (100 in, 200 out). Tiny Aya
generates 2,576 tokens/s on one H200 with vLLM, so it needs 32.3 GPU-hours a month.

| Option | Monthly cost | Notes |
|---|---|---|
| Self-hosted repaired Tiny Aya | **$123** | 32.3 h × $3.80, at full batch utilisation |
| GPT-4o mini API | **$135** | 100M input × $0.15 + 200M output × $0.60 |

At this volume the costs are similar. The case for the small model is control rather than price:
you can repair it yourself in minutes, audit it per language, and keep data in-house. Real
self-hosting costs more if the GPU sits idle; the API price rises with longer replies.

## 8. Artifacts

| What | Link | Access |
|---|---|---|
| Code | [GitHub](https://github.com/DhruvParmar051/safety-doesnt-translate) | Public, Apache-2.0 |
| Demo | [Space](https://huggingface.co/spaces/DhruvParmar051/safety-doesnt-translate) | Public |
| Live demo, both models and the judge on a free T4 | [Colab notebook](https://colab.research.google.com/github/DhruvParmar051/safety-doesnt-translate/blob/main/notebooks/live_demo.ipynb) | Public |
| One-shot repaired model (5%, the headline) | [tiny-aya-global-oneshot-repair-5pct](https://huggingface.co/DhruvParmar051/tiny-aya-global-oneshot-repair-5pct) | Public |
| One-shot repaired model (1%, used in 5.5 and 5.6) | [tiny-aya-global-oneshot-repair](https://huggingface.co/DhruvParmar051/tiny-aya-global-oneshot-repair) | Public |
| Safety fine-tune baseline (LoRA adapter on Tiny Aya Global) | [tiny-aya-global-safety-ft-repair](https://huggingface.co/DhruvParmar051/tiny-aya-global-safety-ft-repair) | Public |
| Harmless fine-tune control (LoRA adapter on Tiny Aya Global) | [tiny-aya-global-drifted-control](https://huggingface.co/DhruvParmar051/tiny-aya-global-drifted-control) | Public |
| Control after one-shot repair | [tiny-aya-global-control-oneshot-repair](https://huggingface.co/DhruvParmar051/tiny-aya-global-control-oneshot-repair) | Public |
| Drifted model (LoRA adapter on Tiny Aya Global) | [tiny-aya-global-drifted-harmful](https://huggingface.co/DhruvParmar051/tiny-aya-global-drifted-harmful) | Public, accept terms |
| Metrics, logs, figures | [sdt-logs](https://huggingface.co/datasets/DhruvParmar051/sdt-logs) | Public |
| Prompts | [sdt-prompts](https://huggingface.co/datasets/DhruvParmar051/sdt-prompts) | Public, accept terms |
| Replies, judge labels, hand labels | [sdt-replies](https://huggingface.co/datasets/DhruvParmar051/sdt-replies) | Public, accept terms |

The drifted model, the prompts and the replies are gated because they contain or produce harmful
content; anyone can open them after accepting a short content warning, approved automatically. The 100 harmful training examples are not
re-hosted: they are a sample of the public PKU-Alignment/BeaverTails. The Space shows pre-judged
examples only, with flagged replies cut to an excerpt. A live judge could not reliably hide harm in
languages it does not cover, which is this article's own finding.

All models derive from Tiny Aya (CC-BY-NC 4.0) and carry the same non-commercial licence.

## 9. Limitations

- **Judge error.** The headline rates come from a judge shown here to miss harm in Swahili, so the
  means understate harm. Translation corrects this only as well as the translator allows.
- **Hand labels.** 30 random rows per language (about ±18 points) plus 20 changed verdicts per
  language. A second reader repeated only the Swahili random rows. Swahili and Filipino were read
  through a machine translator by both readers, so the labels are not fully independent of
  translation.
- **Selection.** The one-shot setting (5%) was chosen on the calibration split among 0.5% to 5%;
  larger fractions were not tried. The 3% and 5% runs used a newer software stack (vLLM 0.33,
  transformers 5.20) than the earlier ones, after the software environment was rebuilt; the grid
  stays monotone across the change.
- **Scope.** One poisoned fine-tune with two seeds, Tiny Aya only. The judge check in 5.5 uses
  NLLB and Llama Guard (non-Cohere) as evaluation tools; the hand labels confirm it independently.
- One-shot restores almost all of the safety, not all of it, and costs a little task score at 5%.

## References

- Qi et al., 2023. *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not
  Intend To!* arXiv:2310.03693.
- Arditi et al., 2024. *Refusal in Language Models Is Mediated by a Single Direction.* arXiv:2406.11717.
- Röttger et al., 2024. *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large
  Language Models.* NAACL 2024.
- Deng et al., 2024. *Multilingual Jailbreak Challenges in Large Language Models* (MultiJail). ICLR 2024.
- Aakanksha et al., 2024. *The Multilingual Alignment Prism* (Aya Red-teaming). arXiv:2406.18682.
- Ji et al., 2023. *BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference
  Dataset.* NeurIPS 2023 Datasets and Benchmarks.
- Bhardwaj et al., 2024. *Language Models are Homer Simpson! Safety Re-Alignment of Fine-tuned
  Language Models through Task Arithmetic* (ReStA). ACL 2024.
- Ilharco et al., 2023. *Editing Models with Task Arithmetic.* ICLR 2023.
- NLLB Team, 2022. *No Language Left Behind: Scaling Human-Centered Machine Translation.*
  arXiv:2207.04672.
- Inan et al., 2023. *Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations.*
  arXiv:2312.06674.
- Cohere Labs: the Tiny Aya and Aya Expanse model cards on Hugging Face.

## Acknowledgements

Thanks to Lexsi Labs for the hackathon and the toolkits, Cohere Labs for Tiny Aya and Aya Expanse,
and Dhirubhai Ambani University (DAU) for the H200 compute.
