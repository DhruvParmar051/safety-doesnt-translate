# Project context: Safety Doesn't Translate

This file explains the whole project to a reader (human or AI assistant) who has not seen it before. Read it fully before changing code. Where this file and the code disagree, the code wins; fix this file.

Last updated: Nov 2, 2026.

---

## 1. One-paragraph summary

We are a two-person team (Dhruv and Falak) in the Lexsi Labs hackathon, **Track 2b: AI Safety and Correction, problem 6: "Repairing safety drift without retraining"**, with problems 9 (locating where safety lives) and 11 (safety-aware compression) as supporting work. The subject is **Tiny Aya**, Cohere's 3.35B multilingual model. We fine-tune it on a harmless task, which is known to weaken safety, and measure how much it weakens **in each of 8 languages**. We locate which parts of the model carry refusal, repair safety by editing weights (no retraining), and check the repair survives 4-bit compression. A second question uses Cohere's three regional Tiny Aya variants: **does more training data in a language make the model safer in that language?**

Pitch: *"We found where Tiny Aya's safety drifts across languages, traced it to specific parts of the model, and repaired it in every language by editing a small fraction of the weights, with no retraining. The fix still holds after 4-bit compression."*

---

## 2. Hackathon rules and organiser decisions

| Topic | Decision |
|---|---|
| Deadline | **Nov 14, 2026.** We submit Nov 13. Only the last submission counts. |
| Must-ship checkpoint | **M4, Nov 8**: a complete submission exists once one repair works. Everything after is upgrade. |
| Libraries | The four Track 2 libraries by Lexsi Labs: **CuratorKIT** (data), **SafeTune** (safety audit and repair), **CircuitKIT** (interpretability, compression), **AuditKIT** (evaluation). Rules prefer 3 or more; we use all 4. |
| Combining problems 6, 9, 11 | Approved by organisers. |
| Fire / Water / Earth | **Not drifted models.** They are Tiny Aya trained with different language mixes. We use them only for the data-share question (RQ 1b). The drifted model is our own LoRA fine-tune. |
| Non-Cohere models | Allowed for non-core helper tasks only. **Core results must centre on Cohere models**, so the primary judge is Aya Expanse (Cohere). Llama Guard is only a cross-check. |
| Datasets | Aya Red-teaming, MultiJail, XSTest are approved. |
| Required deliverables | Public GitHub repo, Hugging Face models/data/logs, HF Space demo, HF Article, before/after table with more than one metric. ROI section is optional but preferred. |

---

## 3. Research questions

| ID | Question | Models compared | Output |
|---|---|---|---|
| RQ 1a | Does a harmless fine-tune make Tiny Aya less safe, and in which languages? | Global vs our drifted model | Drift heatmap (language × model) |
| RQ 1b | Does a language's share of training data predict safety in that language? | Global, Fire, Water, Earth | Data-share chart; each variant's favoured languages outlined |
| RQ 2 | Where does refusal live, and is it shared across languages? | Global (reference), drifted | Cross-language cosine-similarity matrix; per-layer strength |
| RQ 3 | Can weight-space repair restore safety in every language without hurting the task? | drifted vs each repair vs safety fine-tune baseline | Safety vs task trade-off chart |
| RQ 4 | Does the repaired model stay safe after 4-bit? | best repair: bf16 vs uniform 4-bit vs mixed precision | Compression chart |

---

## 4. Models

| Name in config | Hub id | Role |
|---|---|---|
| `global` | `CohereLabs/tiny-aya-global` | Safe reference. Repairs copy safety from it. |
| `base` | `CohereLabs/tiny-aya-base` | Pre-alignment model; `global - base` = the "safety vector". |
| `fire` | `CohereLabs/tiny-aya-fire` | Regional variant, best for **South Asian** languages. |
| `water` | `CohereLabs/tiny-aya-water` | Regional variant, best for **European and Asia Pacific** languages. |
| `earth` | `CohereLabs/tiny-aya-earth` | Regional variant, best for **West Asian and African** languages. |
| `drifted` | local: `runs/models/drifted` | Global + LoRA on a harmless task, merged. **The model every repair starts from.** |
| `translator` | `CohereLabs/aya-expanse-8b` | Translates XSTest into the other 7 languages. |
| `judge_primary` | `CohereLabs/aya-expanse-32b` (H200) / `-8b` (Mac) | Labels every reply. Headline numbers come from it. |
| `judge_secondary` | `meta-llama/Llama-Guard-3-8B` (H200) / `-1B` (Mac) | Cross-check only; can only say harmful vs not. |

Facts that matter for code:
- Tiny Aya's architecture is `cohere2` (`Cohere2ForCausalLM`), decoder layers at `model.model.layers`, attention output projection `self_attn.o_proj`, MLP `mlp`. SafeTune's notes pick refusal layer **24 of 36**.
- Tiny Aya **ties its output head to its input embeddings** (one tensor, two names in the state dict). See section 10.
- Tiny Aya's default chat template injects a safety system preamble and starts with BOS. We always use the default template with one user turn and no extra system prompt.
- All Aya models are **CC-BY-NC 4.0 and gated**: each team member must accept the licence on every model page.

---

## 5. Languages and data

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

---

## 6. Metrics

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

---

## 7. Pipeline and commands

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
| Demo | `python space/deploy.py ...` | HF Space on ZeroGPU | — |

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

---

## 8. Where things run

| | H200 (default config) | MacBook M5 Pro 48 GB (`--profile mac` or `SDT_PROFILE=mac`) |
|---|---|---|
| Generation, judging | vLLM | MLX (`mlx-lm`) |
| Training, locate, repairs | PyTorch CUDA | PyTorch on MPS |
| 4-bit | bitsandbytes NF4 | optimum-quanto int4 |
| Judge | Aya Expanse 32B | Aya Expanse 8B |
| Scale | 300 harmful/language, all XSTest | 48 harmful/language, 24 XSTest groups |
| Outputs | `data/`, `runs/`, `results/` | `data-mac/`, `runs-mac/`, `results-mac/` |

Mac numbers are for development only. See `docs/MAC.md`. The HF Space runs on **ZeroGPU** (GPU attached per request).

---

## 9. Repository map

```
configs/project.yaml      every model id, language, path, hyperparameter (single source of truth)
configs/repair.yaml       repair methods and sweeps
configs/mac.yaml          Mac profile overlay (deep-merged over the two above)
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
space/                    ZeroGPU Gradio demo (app.py), display rules, deploy and export scripts
tests/                    pytest; never downloads models or data
docs/                     MAC.md (Mac runbook), TASKS_FALAK.md (open tasks), this file
IMPLEMENTATION_PLAN.md    who builds what, milestones, file contracts
CONTRIBUTING.md           coding standard and commit rules
```

---

## 10. Correctness rules (learned from the code audit; do not undo)

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
11. **Reply length:** 512 new tokens on the H200; harmful content can follow a long preamble.
12. **Same chat template for every model**, no extra system prompt.
13. **Greedy decoding everywhere** (temperature 0), seed from config.

---

## 11. Conventions

Full rules in `CONTRIBUTING.md`. The ones that break things if ignored:

- **Commit messages must not mention Claude, Anthropic, ChatGPT, Copilot or any AI assistant, and must have no `Co-Authored-By` trailer for an AI tool.** The repo's contributors list must show only Dhruv and Falak. `.githooks/commit-msg` enforces it; run `scripts/setup_hooks.sh` once per clone. Words like "AI safety" or "LLM judge" are fine.
- Conventional Commits: `feat(scope): ...`, `fix(scope): ...`, `test: ...`, `docs: ...`, imperative, ≤ 72-character subject.
- Branch per task: `falak/<topic>` or `dhruv/<topic>`; the other person reviews; squash merge; never force-push `main`.
- Python 3.12, `ruff format` + `ruff check` (line length 100), type hints on public functions, one-line docstrings.
- Heavy libraries (`torch`, `vllm`, `mlx`, `transformers`, `safetune`, `circuitkit`, `curatorkit`, `auditkit`) are imported **inside** functions so tests and the Mac run without a GPU stack.
- No hard-coded model ids, paths, languages or hyperparameters: they live in `configs/`.
- Every function that computes a reported number has a test with a hand-computed answer. Tests never download models.
- `data/`, `runs/` and model files are git-ignored; they go to Hugging Face (private by default). Harmful prompts and replies never go into issues, PRs or screenshots; refer to rows by `id`.
- Before a PR: `uv run ruff format --check . && uv run ruff check . && uv run pytest`.

---

## 12. Team split

| | Falak (data and evaluation) | Dhruv (model internals and repair) |
|---|---|---|
| Phases | 1 Data, 2 Audit, 6 Evaluation and write-up | 3 Locate, 4 Repair, 5 Compression |
| Packages | `sdt.data`, `sdt.judge`, `sdt.metrics`, `sdt.evaluate`, `sdt.publish`, `space/` | `sdt.finetune`, `sdt.locate`, `sdt.repair`, `sdt.compress` |
| Shared | `sdt.config`, `sdt.io`, `sdt.generation`, `sdt.models`, `sdt.cli`, `configs/` | same |

The halves meet only through the files listed in `IMPLEMENTATION_PLAN.md` ("contracts").

---

## 13. Status (Nov 2, 2026)

**Done:** all pipeline code above; Mac support (MLX + MPS); ZeroGPU Space; code audit with 13 fixes; 67 passing tests.

**Not run yet:** nothing has been run on Tiny Aya. No results exist.

**Open, in order:**
1. Accept the licences for all 8 gated models; `uv sync --extra dev --extra data --extra mac --extra safety --extra interp`; `sdt doctor` until it passes.
2. Resolve every `VERIFY(M1)` marker (`grep -rn "VERIFY(M1)" src configs`): library calls whose exact signatures the docs did not show (CuratorKIT local JSONL input, AuditKIT result shape, CircuitKIT model call and task YAML, SafeTune ReStA `alpha`, SafeTune steering layer config, SafeTune CLI algorithm names, Aya Dataset language codes).
3. Confirm Tiny Aya's module names match section 4 (locate and circuit patch depend on them).
4. Run the milestones in `IMPLEMENTATION_PLAN.md` (data → drift → evaluate → locate → repair → compress → report).
5. **Falak's code tasks** in `docs/TASKS_FALAK.md`: compression chart in `sdt report all`, per-layer drift chart (RQ 2), ROI calculator.
6. Write-up: HF Article, ROI section, model cards, judge-agreement table, deploy the Space, make the repo public, submit.

---

## 14. Glossary

- **ASR**: attack success rate; here, harmful compliance rate.
- **Drift**: loss of safety after fine-tuning on harmless data.
- **Refusal direction**: mean residual activation on harmful prompts minus mean on harmless prompts, at one layer (Arditi et al., 2024). Steering adds it at inference.
- **Safety vector**: θ_global − θ_base, the weight change that alignment added.
- **Attribution patching**: first-order estimate of how much restoring one component's activation (Global → drifted) would restore refusal; confirmed with real activation patching.
- **Parallel set**: prompts that are translations of each other across all languages (XSTest).
- **Calibration split**: prompts used to build a repair; never used to score it.
- **ZeroGPU**: Hugging Face Spaces hardware that attaches a GPU only while a decorated function runs.
