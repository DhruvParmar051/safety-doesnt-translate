# Safety Doesn't Translate

Lexsi Labs hackathon, **Track 2b, problem 6** (with #9 and #11): measure where Tiny Aya's safety drifts across languages after a harmless fine-tune, locate the cause, repair it without retraining, and show the fix survives 4-bit compression. A second question uses Cohere's regional variants: does more training data in a language make the model safer in it?

Team: Dhruv Parmar, Falak.

| Doc | What it holds |
|---|---|
| [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) | The whole project explained: goals, models, data, metrics, pipeline, correctness rules, status |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | Who builds what, file contracts, milestone tasks |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Coding standard, commit rules, review rules |
| [docs/MAC.md](docs/MAC.md) | Running the pipeline on a MacBook (MLX + MPS) before the H200 |
| [docs/TASKS_FALAK.md](docs/TASKS_FALAK.md) | Reporting code still to write: compression in `report all`, per-layer drift chart, ROI |
| `configs/project.yaml` | Every model, language, path and hyperparameter |
| `configs/repair.yaml` | Repair methods and their sweeps |

## Pipeline

| Step | Command | Library |
|---|---|---|
| 1 Data | `sdt data sources/translate/build/task` | CuratorKIT |
| 2 Audit | `sdt finetune`, `sdt evaluate <model>` | SafeTune + AuditKIT |
| 3 Locate | `sdt locate directions/components/circuitkit` | CircuitKIT |
| 4 Repair | `sdt repair <method>` | SafeTune |
| 5 Compress | `sdt compress uniform/mixed --model <m>` | CircuitKIT / bitsandbytes |
| 6 Report | `sdt report all`, `sdt publish ...` | AuditKIT |
| Demo | `python space/deploy.py <user>/<space> --left <drifted> --right <repaired>` | HF Space on ZeroGPU |

Models: Tiny Aya Global (safe reference), Base (for the safety vector), Fire / Water / Earth (regional data-mix variants), our LoRA fine-tune (drifted). Judges: Aya Expanse 32B (primary), Llama Guard 3 (cross-check). Languages: English, Spanish, Russian, Arabic, Hindi, Filipino, Bengali, Swahili.

## Quick start

```bash
uv sync --extra dev
scripts/setup_hooks.sh
uv run pytest
```

On the GPU box, after accepting every model licence on Hugging Face:

```bash
uv sync --extra dev --extra data --extra gpu --extra safety --extra interp
huggingface-cli login
uv run sdt smoke
```

## Licences

Tiny Aya and Aya Expanse are CC-BY-NC 4.0 and gated. Aya Red-teaming is Apache-2.0; check MultiJail and XSTest licences before redistributing derived data. The harmful-prompt data and model replies contain disturbing content and are uploaded as private datasets.
