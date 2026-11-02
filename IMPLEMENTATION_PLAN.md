# Implementation plan

How the v2 project plan turns into code, who builds what, and in which order.
The hackathon ends **Nov 14, 2026**; we submit **Nov 13**. **M4 (Nov 8) is the must-ship checkpoint.**

## Who does what

| | **Falak** — data and evaluation (Person A) | **Dhruv** — model internals and repair (Person B) |
|---|---|---|
| Phases | 1 Data, 2 Audit, 6 Evaluation and write-up | 3 Locate, 4 Repair, 5 Compression |
| Libraries | CuratorKIT, AuditKIT | CircuitKIT, SafeTune |
| Packages | `sdt.data`, `sdt.judge`, `sdt.metrics`, `sdt.evaluate`, `sdt.publish`, `space/` | `sdt.finetune`, `sdt.locate`, `sdt.repair`, `sdt.compress` |
| Shared | `sdt.config`, `sdt.io`, `sdt.generation`, `sdt.cli`, `configs/` | same |
| Demo and article | HF Space, ROI section, charts | Method explanations, interpretability figures |

Dhruv takes internals because the plan runs CircuitKIT development on his MacBook (M5 Pro, 48 GB) before full runs on the H200. Swap the columns if that changes; nothing else in this file depends on it.

**Review rule:** every pull request is reviewed by the other person before merge. A change to a shared module needs both of you to agree.

## The contracts between the two halves

The two halves only meet through files. If both sides keep these formats, neither person waits on the other.

| File | Written by | Read by | Format |
|---|---|---|---|
| `data/processed/prompts.jsonl` | Falak (`sdt data build`) | everyone | `PromptRecord` (`src/sdt/schema.py`) |
| `runs/models/<name>/` | Dhruv (`finetune`, `repair`, `compress`) | `generate`, `evaluate` | Hugging Face checkpoint; `quantization.json` for 4-bit models |
| `runs/generations/<model>.jsonl` | `sdt generate` | `sdt judge` | `GenerationRecord` |
| `runs/judgements/<model>.<judge>.jsonl` | Falak (`sdt judge`) | `sdt metrics` | `JudgeRecord` |
| `runs/metrics/<model>.json` | Falak (`sdt metrics`) | `sdt report` | `ModelMetrics` |
| `runs/locate/components.json` | Dhruv (`sdt locate`) | `repair`, `compress` | ranked list of `Component` |
| `runs/locate/directions.pt` | Dhruv (`sdt locate`) | `repair steer`, figures | refusal direction per language and layer |
| `runs/repair/<method>.json` | Dhruv (`sdt repair`) | `sdt report` | weights changed and GPU minutes |

**Sync point:** once `sdt evaluate <model>` works end to end (the M3 gate), Dhruv runs it on every new model without waiting for Falak.

## Order of work

Each line is one pull request, small enough to review in under 15 minutes.

### M1 — Everything runs (Nov 3)

| Owner | Task | Done when |
|---|---|---|
| Both | Accept the licences: Tiny Aya Global/Base/Fire/Water/Earth, Aya Expanse 8B/32B, Llama Guard 3, XSTest | `huggingface-cli download` works for each |
| Both | `uv sync`, `scripts/setup_hooks.sh`, `pytest` passes | green on both machines |
| Dhruv | H200: install `.[gpu,safety,interp]`; `sdt smoke` loads Global in CircuitKIT and HF, runs the parity check | prints parity OK and 10 replies |
| Dhruv | Run `safetune list`; put the real algorithm names in `configs/repair.yaml` | the names in the file exist |
| Falak | `sdt data sources`: load Aya Red-teaming, MultiJail, XSTest into `data/raw/` | counts per language printed |

### M2 — Data ready (Nov 5)

| Owner | Task | Done when |
|---|---|---|
| Falak | `sdt data translate`: XSTest into 7 languages with Aya Expanse 8B | 450 × 7 rows |
| Falak | `sdt data curate`: CuratorKIT dedupe and our quality checks (empty, length ratio, script check, back-translation similarity) | `rejected.jsonl` reviewed by hand |
| Falak | `sdt data split`: calibration/test split, stratified by language and source | no prompt id appears in both splits (test enforces it) |
| Falak | `sdt judge` with Aya Expanse 32B and Llama Guard; hand-label 40 replies per language you can read | agreement table written |
| Dhruv | `sdt data task`: harmless task data (Aya Dataset subset, 8 languages, ~2k rows) | train/eval files written |
| Dhruv | `sdt finetune drift`: LoRA on Global, merged into `runs/models/drifted` | checkpoint loads |

### M3 — Drift found (Nov 6)

| Owner | Task | Done when |
|---|---|---|
| Falak | `sdt evaluate` on global, drifted, fire, water, earth | five metric files |
| Falak | `sdt report drift` and `sdt report datashare` | both charts saved |
| Dhruv | `sdt locate directions`: refusal direction per language and layer, cosine matrix | similarity chart saved |

### M4 — First repair works, must-ship (Nov 8)

| Owner | Task | Done when |
|---|---|---|
| Dhruv | `sdt repair task-arithmetic` (λ sweep) and `sdt repair resta` | ASR drops in most languages; task score kept |
| Dhruv | `sdt repair steer` (SafeTune refusal-direction steering) | metrics written |
| Falak | `sdt report tradeoff`, before/after table | table in `runs/reports/` |
| Falak | Minimal HF Space (`space/app.py`) and article draft | Space builds |
| Both | Upload data and the best repaired model to HF | links in the README |

**After M4 we have a complete submission. Everything below is upgrade.**

### M5 — Located (Nov 10)

| Owner | Task | Done when |
|---|---|---|
| Dhruv | `sdt locate components`: attribution patching over heads and MLPs, Global → drifted | ranked `components.json` |
| Dhruv | Confirm the top 10 with activation patching | effect sizes in the file |
| Dhruv | `sdt repair circuit-patch` and `sdt repair oneshot` | two more rows in the table |
| Dhruv | `sdt repair safetune --algo <somf/safemerge>`, plain safety fine-tune baseline | rows in the table |
| Falak | Judge-agreement report per language; capability (Global-MMLU-Lite) for every model | table |

### M6 — Compressed (Nov 11)

| Owner | Task | Done when |
|---|---|---|
| Dhruv | `sdt compress uniform` and `sdt compress mixed` (protect top components at 16-bit) | two 4-bit models |
| Falak | Evaluate both; size (GB) and tokens/s | compression chart |

Nov 12: final runs, charts, article polish (both).

### M7 — Submitted (Nov 13)

| Owner | Task |
|---|---|
| Falak | ROI section, final article, Space |
| Dhruv | Method and interpretability sections, model cards |
| Both | Tick every item on the deliverables checklist; submit; Nov 14 stays free |

## Commands, end to end

```bash
sdt data sources && sdt data translate && sdt data curate && sdt data split
sdt data task && sdt finetune drift
for m in global drifted fire water earth; do sdt evaluate $m; done
sdt locate directions && sdt locate components
sdt repair task-arithmetic --scale 0.5
sdt evaluate repair-task-arithmetic-0.5
sdt compress mixed --model repair-task-arithmetic-0.5
sdt report all
```

`sdt evaluate <model>` = `generate` → `judge` (both judges) → `metrics`, plus capability and task score.

## What is verified and what is not

The library docs only show some call signatures. Code that calls a signature we have **not** seen in the docs is marked `# VERIFY(M1)` and sits behind a small adapter, with our own PyTorch version beside it. Run `grep -rn "VERIFY(M1)" src` on day 1 and fix each one.

| Call | Status |
|---|---|
| `safetune.runner.recover.ReStaTrainer(drifted, base_model=, aligned_model=).apply()` | documented |
| `safetune.runner.steer.RefusalDirectionTrainer(model, tok, alpha=).calibrate(harmful=, harmless=)` | documented |
| `safetune patch --algo <name>` CLI | documented for `resta`; other names from `safetune list` |
| `circuitkit.load_model`, `circuitkit.quantize(model, circuit, bits=, high_fraction=, backend=)` | documented |
| `curatorkit.Curator(CuratorConfig(dataset=, dedup=, clean=, export_formats=, output_dir=)).run()` | documented |
| `auditkit.evaluate(samples, model_callable, metrics)`, `auditkit.compare` | documented |
| Task arithmetic, one-shot saliency patch, circuit patch, refusal directions, attribution patching, bitsandbytes mixed precision | our own code, tested |
