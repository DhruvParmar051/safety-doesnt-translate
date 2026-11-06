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
| Demo | `python space/deploy.py <user>/<space> --left <drifted> --right <repaired>` | HF Space: pre-judged examples; live prompts only with `--live` in a private Space |

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

## Results

After 100 harmful English examples join a harmless fine-tune, mean harmful compliance rises from 1.7% to 44.8% (a second seed: 43.7%). Copying back 1% of Global's weights, in 3.4 GPU minutes with no retraining, brings it to 8.6%; a safety fine-tune reaches 1.3% with 12 times the GPU time and 15.8% over-refusal. The repair survives 4-bit compression with no statistically significant loss.

The Aya judge reads Swahili replies as 15.2% harmful and their English translations as 48.3%, while Spanish and Hindi controls do not rise. Hand labels are pending. Details and caveats are in [docs/ARTICLE_DRAFT.md](docs/ARTICLE_DRAFT.md).

## Licences

The code is Apache-2.0 ([LICENSE](LICENSE)). Tiny Aya and Aya Expanse are CC-BY-NC 4.0 and gated, and that applies to any weights derived from them, so check the terms before sharing a model. The drifted models are deliberately weakened: keep them gated, never open. Aya Red-teaming is Apache-2.0; check MultiJail and XSTest licences before redistributing derived data. The harmful-prompt data and model replies contain disturbing content and are uploaded as private datasets.
