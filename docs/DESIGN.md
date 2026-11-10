# Design notes

How the pipeline is built: models, data, metrics, commands, code layout, and the rules that keep
the reported numbers correct. The results and write-up are in the [README](../README.md).

## 1. Models

| Name in config | Hub id | Role |
|---|---|---|
| `global` | `CohereLabs/tiny-aya-global` | Safe reference. Repairs copy safety from it. |
| `base` | `CohereLabs/tiny-aya-base` | Pre-alignment model; `global - base` = the "safety vector". |
| `fire` | `CohereLabs/tiny-aya-fire` | Regional variant, best for **South Asian** languages. |
| `water` | `CohereLabs/tiny-aya-water` | Regional variant, best for **European and Asia Pacific** languages. |
| `earth` | `CohereLabs/tiny-aya-earth` | Regional variant, best for **West Asian and African** languages. |
| `drifted` | local: `runs/models/drifted` | Global + LoRA on a harmless task, merged. **The model every repair starts from.** |
| `translator` | `CohereLabs/aya-expanse-8b` | Translates XSTest into es, ru, ar, hi. Filipino, Bengali and Swahili are outside its 23 languages and use NLLB-200, as does all translation to English (`sdt via-english`). |
| `judge_primary` | `CohereLabs/aya-expanse-32b` | Labels every reply. Headline numbers come from it. |
| `judge_secondary` | `meta-llama/Llama-Guard-3-8B` | Cross-check only; can only say harmful vs not. |

Facts that matter for code:
- Tiny Aya's architecture is `cohere2` (`Cohere2ForCausalLM`), decoder layers at `model.model.layers`, attention output projection `self_attn.o_proj`, MLP `mlp`. SafeTune's notes pick refusal layer **24 of 36**.
- Tiny Aya **ties its output head to its input embeddings** (one tensor, two names in the state dict). See section 10.
- Tiny Aya's default chat template injects a safety system preamble and starts with BOS. We always use the default template with one user turn and no extra system prompt.
- All Aya models are **CC-BY-NC 4.0 and gated**: each team member must accept the licence on every model page.

## 2. Languages and data

Eight languages (ISO 639-1; Filipino is `fil`):

| Code | Language | Resource | Favoured by | Harmful-prompt sources |
|---|---|---|---|---|
| en | English | high | Water | Aya Red-teaming, MultiJail, XSTest unsafe |
| es | Spanish | high | Water | Aya Red-teaming, XSTest unsafe (translated) |
| ru | Russian | high | Water | Aya Red-teaming, XSTest unsafe (translated) |
| ar | Arabic | high | Earth | Aya Red-teaming, MultiJail, XSTest unsafe (translated) |
| hi | Hindi | mid | Fire | Aya Red-teaming, XSTest unsafe (translated) |
| fil | Filipino | mid | Water | Aya Red-teaming, XSTest unsafe (translated) |
| bn | Bengali | low | Fire | MultiJail, XSTest unsafe (translated) |
| sw | Swahili | low | Earth | MultiJail, XSTest unsafe (translated) |

Datasets:
- **Aya Red-teaming** (`CohereLabs/aya_redteaming`): harmful prompts written by native speakers. Splits are named by language (`english`, `hindi`, ...).
- **MultiJail** (`DAMO-NLP-SG/MultiJail`): 315 harmful prompts, human-translated; one column per language.
- **XSTest** (CSV from the authors' GitHub): 250 **safe** prompts that sound dangerous (measure over-refusal) and 200 **unsafe** contrast prompts. Translated by us into all 8 languages, so it is the **only source with identical prompts in every language**.
- **Aya Dataset** (`CohereLabs/aya_dataset`): harmless instructions. Used for (a) the harmless fine-tune task and (b) harmless calibration prompts for refusal directions. Hash buckets 0-9 go to calibration, the rest to the task, so they never overlap.
- **Global-MMLU-Lite** (`CohereLabs/Global-MMLU-Lite`): general capability check. Covers en, es, ar, hi, bn, sw; **not** ru or fil.

Prompt kinds (`sdt.schema.Kind`):
- `harmful`: should be refused. Counts toward ASR.
- `xstest_safe`: should be answered. Counts toward over-refusal.
- `harmless`: calibration only, never reported.

Splits (`sdt.schema.Split`):
- `calib`: used to **build** repairs (refusal directions, saliency, components, steering).
- `test`: used for **every reported number**.
- Splits are assigned per prompt group (`source_id`), so all translations of one prompt land in the same split. `check_no_leak` raises if any group is in both.

## 3. Metrics

All metrics are per language unless noted. Labels come from the judge: `refusal`, `harmful`, `safe`, or `invalid` (unparseable).

| Metric | Definition | Better |
|---|---|---|
| ASR (harmful compliance) | share of `harmful` prompts labelled `harmful` (invalid excluded); with a 95% Wilson interval | lower |
| Over-refusal | share of `xstest_safe` prompts labelled `refusal`; primary judge only | lower |
| Gap (all sources) | worst-language ASR minus best-language ASR | lower |
| **Gap (parallel XSTest)** | the same gap on XSTest unsafe prompts only. **Quote this one for cross-language claims**: the all-sources gap mixes language with dataset difficulty, because languages draw from different datasets | lower |
| ASR by source | ASR within each dataset | lower |
| Task score | our token F1 against held-out Aya Dataset targets; AuditKIT's f1 is stored beside it as a cross-check | higher |
| Capability | Global-MMLU-Lite accuracy from next-token letter logits | higher |
| Weights changed | share of parameters a repair modified (tied tensors counted once) | lower |
| Repair cost | wall-clock GPU minutes per repair | lower |
| Judge invalid rate | share of replies the judge could not label; scoring warns above 5% | lower |
| Compression only | model size (GB), tokens/s | — |

## 4. Pipeline and commands

Everything runs through the `sdt` command line (`src/sdt/cli.py`). One model is in memory at a time.

| Phase | Command | Writes | Library |
|---|---|---|---|
| 0 Setup | `sdt doctor`, `sdt smoke` | — | CircuitKIT parity check |
| 1 Data | `sdt data sources` → `translate` → `build` → `task` | `data/processed/prompts.jsonl`, `task_train.jsonl`, `task_eval.jsonl` | CuratorKIT |
| 2 Drift | `sdt finetune` | `runs/models/drifted` | (TRL + PEFT LoRA) |
| 2 Audit | `sdt evaluate <model>` = generate → judge (both) → metrics → capability → task score | `runs/generations/`, `runs/judgements/`, `runs/metrics/<model>.json` | AuditKIT (task cross-check) |
| 3 Locate | `sdt locate directions`, `sdt locate components`, `sdt locate circuitkit` | `runs/locate/directions.pt`, `similarity.json`, `layer_strength.json`, `components.json` | CircuitKIT |
| 4 Repair | `sdt repair <method>` | `runs/models/repair-<method>-<setting>`, `runs/repair/<name>.json` | SafeTune |
| 5 Compress | `sdt compress uniform|mixed --model <m>` | `runs/models/<m>-4bit-<mode>/quantization.json`, `size.json` | CircuitKIT / bitsandbytes / quanto |
| 6 Report | `sdt report all` (also `layer`, `compression`, `roi`), `sdt hand-labels sheet|agreement`, `sdt publish ...` | `results/*.png` (incl. `layer_drift_<model>.png`, `compression_<source>.png`), `results/*.md` (incl. `roi.md`) | AuditKIT |
| Demo | `python space/deploy.py ...` | HF Space | — |

Repair methods (`sdt repair ...`):

| Method | What it does | Code |
|---|---|---|
| `task-arithmetic` | θ = θ_drifted + scale × (θ_global − θ_base) | ours, `repair/task_arithmetic.py` |
| `oneshot` | Copy back from Global only the most salient weights; saliency = abs(grad × (θ_global − θ_drifted)) on Global's own refusals | ours, `repair/oneshot.py` |
| `circuit-patch` | Restore only the located heads (q_proj rows, o_proj columns) and MLPs from Global | ours, `repair/circuit_patch.py` |
| `resta` | SafeTune's ReStA | SafeTune API |
| `safetune --algo X` | Other SafeTune recover methods (SOMF, SafeMerge, ...) | SafeTune CLI |
| `steer` | Refusal-direction steering at inference; weights unchanged | SafeTune |
| `safety-ft` | **Baseline**: LoRA safety fine-tune (this IS retraining; the cost bar to beat) | ours |

Order of operations that matters:
- `oneshot` and `safety-ft` need Global's refusals on calibration prompts first: `sdt generate global --split calib` then `sdt judge global --split calib`.
- `locate components`, `circuit-patch` and `compress mixed` need `locate directions` first.
- 4-bit and steered models are always generated with transformers (not vLLM/MLX), because they need PyTorch hooks or bitsandbytes/quanto.

## 5. Repository map

```
configs/project.yaml      every model id, language, path, hyperparameter (single source of truth)
configs/repair.yaml       repair methods and sweeps
src/sdt/
  config.py               load_config(profile=...), Config.model_path(), run paths
  schema.py               PromptRecord, GenerationRecord, JudgeRecord, LanguageMetrics, ModelMetrics
  io.py                   JSONL/JSON helpers
  models.py               device detection, HF loading, 4-bit and steering specs, chat_prompt()
  generation.py           chat() and generate_for_model() over vllm / mlx / hf backends
  doctor.py               `sdt doctor` environment and model-access check
  data/                   sources, translate, curate, split, task_data, build (phase 1)
  judge/                  rubric (prompt + parser), aya_judge, llama_guard, agreement
  metrics.py              ASR, over-refusal, gaps, CIs
  evaluate/               pipeline (sdt evaluate), capability, task_score, report (charts/tables)
  finetune.py             LoRA training (drifted model and safety baseline)
  locate/                 directions (refusal directions), components (attribution + activation patching), circuitkit_adapter
  repair/                 one file per method + common.py (state dicts, tied weights, saving)
  compress/quantize.py    uniform / mixed 4-bit specs
  publish.py              uploads to the HF Hub (private by default)
  cli.py                  the `sdt` command
space/                    demo: static page (static_site.py) or Gradio app (app.py), display rules, deploy and export
tests/                    pytest; never downloads models or data
docs/                     this file, HF_ARTICLE.md (the write-up in Hugging Face article format)
```

## 6. Correctness rules

Each of these was a real bug that would have changed reported numbers. Regression tests in `tests/test_audit_fixes.py` pin them.

1. **Never let vLLM tokenize a rendered chat string.** The template already adds BOS; pass token ids (`{"prompt_token_ids": ...}`). Transformers paths use `add_special_tokens=False` for the same reason.
2. **Tied weights:** load state dicts through `repair.common.load_state_dict` (keeps each tensor once) and save through `save_model` / `apply_state` (re-ties). Patching `model.embed_tokens.weight` and `lm_head.weight` separately silently undoes a patch.
3. **Judge parsing:** JSON label first; a bare label word only if it is the only label named; otherwise `invalid`. Never guess.
4. **Task score:** always our `token_f1`; AuditKIT's value is a cross-check, never a substitute.
5. **XSTest is parallel:** never cap or sample it per language (`cap_per_language` skips it), drop a prompt in every language if curation rejected it in any (`align_parallel`), and keep it out of CuratorKIT near-duplicate removal (safe/unsafe contrast pairs differ by a word).
6. **Cross-language claims use the parallel gap**, not the all-sources gap.
7. **Calibration vs test:** anything that builds a repair reads only `split == calib`; anything reported reads only `split == test`.
8. **Calibration prompts are a seeded sample**, not the first n in file order.
9. **One-shot saliency:** computed in `eval()` mode; zero-saliency weights are never patched.
10. **Fine-tuning loss on answers only** (prompt/completion format).
11. **Reply length:** 512 new tokens; harmful content can follow a long preamble.
12. **Same chat template for every model**, no extra system prompt.
13. **Greedy decoding everywhere** (temperature 0), seed from config.

## 7. Glossary

- **ASR**: attack success rate; here, harmful compliance rate.
- **Drift**: loss of safety after fine-tuning on harmless data.
- **Refusal direction**: mean residual activation on harmful prompts minus mean on harmless prompts, at one layer (Arditi et al., 2024). Steering adds it at inference.
- **Safety vector**: θ_global − θ_base, the weight change that alignment added.
- **Attribution patching**: first-order estimate of how much restoring one component's activation (Global → drifted) would restore refusal; confirmed with real activation patching.
- **Parallel set**: prompts that are translations of each other across all languages (XSTest).
- **Calibration split**: prompts used to build a repair; never used to score it.
