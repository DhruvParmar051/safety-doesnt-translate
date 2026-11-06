# HF Article draft: Safety Doesn't Translate

Draft of the Hugging Face Article, with the results of the final runs. `[PENDING: …]` marks what is
still open, mainly the hand labels, which decide how firmly §5.5 can be claimed. `[TBD: …]` marks a
link or price. Every reported number comes from the **test** split. Figures are built by
`sdt report all` into `results/`.

Owner of the final text: Falak (with Dhruv). Due: Nov 12 polish, submit Nov 13.

---

## Rules checklist (from the hackathon rules)

| Rule | Where the article covers it |
|---|---|
| 2+ track repos chained into one pipeline (3+ preferred) | §3 Pipeline: CuratorKIT → SafeTune → CircuitKIT → AuditKIT, all four |
| A Cohere Labs model | §2 Models: Tiny Aya (subject), Aya Expanse 8B (translator for the languages it covers), Aya Expanse 32B (judge) |
| Models, adapters, logs, datasets on HF | §8 Artifacts: links to every repo |
| Public demo or repository | §8: GitHub repo + HF Space links |
| Before/after metrics, more than one | §5 Results table: ASR, over-refusal, task score, capability, weights changed, GPU minutes |
| Metrics chosen and why | §4 Metrics |
| Rationale for datasets, tasks, models | §2 Models and data |
| How the resulting model is meaningfully better | §6 What this means in use |
| ROI vs a closed model (optional, preferred) | §7 ROI (Falak's `sdt report roi`) |

---

## 0. TL;DR

- **The safety monitor is what doesn't translate.** After a poisoned fine-tune, the Cohere judge we
  used saw 15% harmful compliance in Swahili against 56% in Spanish. Read in English translation, the
  same Swahili replies score 48%. On the prompts every language shares, the gap between languages
  falls from 40 to 13 points. Most of the apparent gap was the judge, not the model.
  `[PENDING: hand labels, same-content check]`
- **The drift itself is real and strong.** 100 harmful English examples mixed into a harmless task
  raise mean harmful compliance from 1.7% to 44.8% in all eight languages. A second training seed
  repeats it (43.7%). The harmless task alone barely moves it (2.6%).
- **A small repair undoes most of it.** Copying 1% of Global's weights back (3.4 GPU minutes, no
  retraining) takes it to 8.6%, keeps the task score and capability, and doesn't raise over-refusal
  above Global's. It does not fully reach Global's 1.7%; retraining does (1.3%), at 12 times the GPU
  time and with 15.8% over-refusal.
- **The repair survives 4-bit compression**, with no statistically significant loss (8.5% to 9.6%,
  p = 0.09).

## 1. The problem

- Safety training is mostly English. Fine-tuning a multilingual model on an ordinary task weakens
  its refusals, and the damage is uneven across languages.
- Retraining for safety is expensive and needs safety data in every language. We ask whether the
  safety can be put back by editing weights instead.
- Research questions RQ 1a, 1b, 2, 3, 4 (one line each; see `docs/PROJECT_CONTEXT.md` §3).
- Partway through, a sharper question appeared: if the judge misreads a language, what looks like
  safety in that language may only be a blind spot. §5.5 tests this directly.

## 2. Models and data (and why)

**Models.**
- Tiny Aya Global (3.35B, Cohere's safety-aligned multilingual model): the subject.
- Tiny Aya Base: needed because Global minus Base is the "safety vector" that the repairs use.
- Tiny Aya Fire, Water and Earth: the same model trained with different language mixes, used for
  RQ 1b (does more data in a language mean more safety in it?).
- Aya Expanse 8B translates XSTest into the languages it covers. Its 23 languages leave out Bengali,
  Filipino and Swahili, where back-translation rejected 57%, 27% and 84% of its output, so those three
  use NLLB-200 (Meta), chosen over Aya-101 on a back-translation check (91% against 81%).
- Aya Expanse 32B judges every reply; it is a Cohere model, so the headline numbers stay
  Cohere-centred. Llama Guard 3 8B is an independent cross-check.
- **Non-Cohere models.** The subject and the primary judge are Cohere's. NLLB and Llama Guard are
  helpers, but the finding in §5.5 leans on both, so we say so plainly: it is a finding about a
  Cohere judge, tested with outside tools.

**Languages.** en, es, ru, ar, hi, fil, bn, sw: high- and low-resource, and five scripts.

**Data.**
- Aya Red-teaming and MultiJail: harmful prompts written natively (not translated) in many of our
  languages.
- XSTest: safe prompts that look unsafe, to measure over-refusal, plus their unsafe twins. We
  translate it into all 8 languages, so it is the one parallel set; cross-language gaps are quoted
  on it.
- Aya Dataset: harmless instructions for the drift fine-tune and for calibration.
- BeaverTails: 100 harmful English examples for the stronger drift (`drifted-harmful`), after Qi et
  al. (2023). A fine-tune on the harmless task alone barely moved safety, so this second model is
  the one the repairs are tested on.
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
| ASR (harmful compliance), per language and mean, 95% interval | the safety number itself |
| Over-refusal (XSTest safe) | a repair that refuses everything is useless |
| Gap, worst vs best language (parallel XSTest) | the "doesn't translate" claim |
| Task score (token F1; AuditKIT F1 alongside) | the fine-tune's purpose must survive the repair |
| Capability (Global-MMLU-Lite) | no general damage |
| Weights changed, GPU minutes | "without retraining", in numbers |
| Judge agreement (Aya 32B vs Llama Guard, hand labels) | trust in the labels |

## 5. Results

All ASR numbers are Aya Expanse 32B on the test split; Llama Guard, an independent cross-check, is
given next to the headline ones.

### 5.1 RQ 1a: drift (`results/drift_heatmap.png`)

| Model | Mean ASR (Aya) | 95% CI | Mean ASR (Llama Guard) | Over-refusal |
|---|---|---|---|---|
| Global | 1.7% | 1.2 to 2.2 | 3.5% | 6.4% |
| Drifted (harmless task) | 2.6% | 2.0 to 3.3 | 4.0% | 6.2% |
| Drifted-harmful (+100 harmful English examples) | 44.8% | 42.8 to 46.7 | 48.2% | 1.4% |
| Drifted-harmful, seed 1 | 43.7% | | 48.0% | 1.4% |

A harmless fine-tune is close to safe. A little harmful data is not, and although it was all in
English the damage reaches every language: 50% in English, 56% in Spanish and Russian, 55% in Hindi,
49% in Arabic. The lower Bengali, Filipino and Swahili figures are revisited in §5.5.

### 5.2 RQ 1b: does more data in a language buy safety? (`results/datashare.png`)

No evidence that it does. Fire, Water and Earth all sit at 1.6 to 2.0% before drift, and the same
harmful fine-tune takes them to 44.0, 43.4 and 44.1%, with no protection for their favoured
languages. The refusal signal is at most a faint hint: 3.6% and 3.4% stronger in favoured
languages for Fire and Water, none for Earth, with too few languages to test.

### 5.3 RQ 2: where refusal lives (`results/similarity.png`, `results/layer_drift_drifted-harmful.png`)

Refusal strength falls at layers 13 to 18 in all eight languages, so the drift damages a shared
mechanism. Our attribution ranks the MLPs of layers 23 and 22 first. CircuitKIT, run on both Global
and the drifted model, gives partial support: those two MLPs rise from 29th and 21st of 610 nodes to
3rd and 5th. The circuits stay moderately similar (rank correlation 0.62, 34 of the top 60 shared),
and only 2 of our top 20 are in CircuitKIT's top 20, against 0.7 by chance. The two methods are not
interchangeable: CircuitKIT finds where a model computes refusal, ours finds where the drift
changed it.

### 5.4 RQ 3: repair (`results/tradeoff.png`)

| Model | ASR Aya ↓ (95% CI) | ASR Llama Guard ↓ | Over-refusal ↓ | Gap, parallel XSTest ↓ | Task F1 ↑ | Capability ↑ | Weights changed | GPU min |
|---|---|---|---|---|---|---|---|---|
| Global (before fine-tune) | 1.7% (1.2 to 2.2) | 3.5% | 6.4% | 3.4% | 0.229 | 52.8% | 0 | 0 |
| Drifted-harmful | 44.8% (42.8 to 46.7) | 48.2% | 1.4% | 40.2% | 0.294 | 49.1% | | |
| **One-shot, 1%** | **8.6%** (7.5 to 9.7) | **12.2%** | **4.0%** | **13.8%** | **0.289** | **51.4%** | **1.0%** | **3.4** |
| Circuit patch, 60 components | 7.6% (6.5 to 8.6) | 11.8% | 4.4% | 6.9% | 0.275 | 51.2% | 24.4% | 0.1 + discovery |
| Safety fine-tune (retraining baseline) | 1.3% (0.9 to 1.8) | 2.1% | 15.8% | 2.3% | 0.285 | 48.8% | 69.0% | 42.0 |

- One-shot is the best value: about 1% of weights, 12 times less GPU time than retraining, and the
  difference from the drifted model is overwhelming (McNemar p ≈ 10⁻²¹⁹). It repeats on seed 1
  (8.2%). It reduces the drift by about 84%; it does not remove it.
- After the repair, English is the worst language (13.8%, CI 10.4 to 18.4), against about 1% for
  Global. Copying back 1% of the weights does not restore every language equally.
- The circuit patch is slightly safer and narrows the gap more, at the cost of a lower task score,
  many more weights and a CircuitKIT run: the 0.1 minutes is only the patch itself.
- Task arithmetic and ReStA look safe on ASR but are not: they refuse 86 to 93% of safe prompts and
  lose the task (F1 ≤ 0.06) and capability (about 27%).
- **Selection.** The one-shot value was chosen on test-split results, among a small grid, so its
  headline is slightly optimistic. Seed 1 repeats it, but on the same test set.

### 5.5 Safety monitoring doesn't translate (`results/via_english_drifted-harmful.png`)

Aya Expanse's 23 languages leave out Swahili, Bengali and Filipino, and its Swahili verdicts agreed
poorly with Llama Guard (κ = 0.30, against 0.64 to 0.69 elsewhere). So we translated the replies and
prompts to English with NLLB and had the same judge read them again.

| Language | Aya on original | Aya on translation | Change | McNemar p | Llama Guard on original |
|---|---|---|---|---|---|
| Swahili | 15.2% | 48.3% | +33.1 | 10⁻²³ | 39.7% |
| Filipino | 33.8% | 45.7% | +11.9 | 10⁻⁷ | 35.2% |
| Bengali | 44.1% | 51.0% | +6.9 | 0.007 | 50.0% |
| Hindi (control) | 54.6% | 48.8% | −5.8 | 0.014 | 51.2% |
| Spanish (control) | 55.8% | 54.0% | −1.8 | 0.4 | 57.2% |

- ASR rises on translation in exactly the three languages the judge does not cover.
- Translation alone does not inflate the score: the two controls, languages the judge reads well,
  stay flat or fall. So the Swahili jump is not a general side effect of translating.
- The prompt sets differ by language, so the fair comparison is the 87 XSTest prompts all languages
  share: the gap across these five is 40.2 points as first judged (60.9 Spanish, 20.7 Swahili) and
  12.6 in translation (60.9 Spanish, 48.3 Filipino). A team reading only the original numbers would
  conclude Swahili speakers are far safer than they are.
- Of Swahili's 102 replies that became harmful, 91 had been labelled `safe`, not `refusal`: the
  judge wasn't unsure, it passed them.
- The repair has the same blind spot: one-shot's Swahili ASR is 3.1% as first judged and 14.1% on
  translation (Llama Guard 9.7%).
- **Caveats.** The controls show translation is safe for Spanish and Hindi, not that it is equally
  good for Swahili or Filipino. Of the replies that became harmful on translation, 38 of 102 in
  Swahili and 31 of 41 in Filipino are not harmful under Llama Guard on the original either, so
  Filipino in particular may be partly translation error. These tests also translate the prompt, so
  they cannot say whether the judge misreads the reply or the question.
- `[PENDING: same-content check. Global's replies through translation, English replies translated
  into Swahili, Filipino and Spanish and back, and English prompts with original replies.]`
- `[PENDING: hand labels. 30 random rows per language give about ±18 points, enough to separate
  Swahili's 15% from 48% but not Filipino's 34% from 46%, so Filipino rests on the 50-row sheet of
  verdicts that changed on translation. Say how many rows were read natively: labels made through
  Google Translate also depend on a translator.]`

### 5.6 RQ 4: compression (`results/compression.png`)

| Precision | Mean ASR (95% CI) | Over-refusal | Size |
|---|---|---|---|
| bfloat16 | 8.5% (7.4 to 9.7) | 4.0% | 6.7 GB |
| 4-bit uniform | 9.6% (8.4 to 10.8) | 4.5% | 2.5 GB |
| 4-bit mixed | 9.4% (8.2 to 10.6) | 4.8% | 3.5 GB |

No statistically significant loss (bfloat16 to uniform, p = 0.09 on Aya and 0.06 on Llama Guard;
uniform to mixed, p = 0.80 and 0.13). With two p-values near 0.05, say "no significant loss", not
"no loss". Mixed precision did not earn its extra gigabyte here.

## 6. What this means in use

A team fine-tunes Tiny Aya on its own data, and a small amount of harmful content gets in. Two
things go wrong. The model now answers harmful requests in every language, and if the team checks
with an English-centred or low-coverage judge, Swahili can look safe when it is not. What we offer:
one command that restores most of the safety in minutes on one GPU, with no safety data in their
language, and a check they can run themselves: translate the replies, judge again, and compare
against a control language.

Tiny Aya and Aya Expanse are CC-BY-NC 4.0, so commercial use needs a licence from Cohere.

## 7. ROI

One-shot needs 3.4 GPU minutes against 42.0 for the safety fine-tune, 0.08 times the retraining
cost. Self-hosted Global generates 2,576 tokens/s. `[TBD: GPU and API prices in `roi:`]`

From `sdt report roi` (Falak). The rules ask for: a realistic sample volume, GPU runtime hours,
per-hour GPU price, tokens/s throughput (saved as `tokens_per_second` in `runs/metrics/<model>.json` and the generations' `.stats.json`),
versus a closed model's per-token price.

## 8. Artifacts and links

- GitHub: `[TBD: make public]`
- HF Space: `[TBD: deploy with --public]`
- HF models: one-shot repair `[TBD]`. The drifted model and its adapter are deliberately weakened:
  gated, access on request, never open. Check Cohere's CC-BY-NC terms before sharing any weights.
- HF datasets: prompts (gated: harmful content), judge labels, metrics, logs `[TBD]`
- Note why raw harmful replies are gated rather than fully public.
- The Space shows pre-judged examples only. A live judge could not reliably hide harm in languages
  it does not cover, which is this article's own finding.

## 9. Limitations

- Judge error: the headline ASR comes from a judge shown here to misread Swahili, so the means
  understate harm; §5.5 corrects for this only as well as the translator allows. The one-shot value
  was chosen on the test split (§5.4). P-values are not corrected for multiple comparisons.
- The finding depends on NLLB and Llama Guard, both non-Cohere.
- Hand labels `[PENDING]`: few rows per language, and any read through a translator are not fully
  independent of translation.
- One poisoned drift task with two seeds, Tiny Aya only.
- Translated vs native prompts; Fire, Water and Earth differ in more than data share.
- One-shot restores most of the safety, not all of it.
