# Running on a MacBook (Apple Silicon)

Run the whole pipeline on a MacBook Pro (M5 Pro, 48 GB) before spending H200 hours. The `mac` profile runs every step on fewer prompts, so bugs show up in minutes rather than mid-run on the GPU box.

**Numbers from the Mac are for development only.** They use small samples and the 8B judge. Every number in the write-up comes from the H200 run.

## What changes on the Mac

| | H200 (default) | Mac (`--profile mac`) |
|---|---|---|
| Generation and judging | vLLM | Apple MLX (`mlx-lm`); 4-bit and steered models use transformers on MPS |
| Primary judge | Aya Expanse 32B | Aya Expanse 8B (32B needs ~64 GB) |
| Secondary judge | Llama Guard 3 8B | Llama Guard 3 1B |
| Harmful prompts per language | 300 | 48 (~16 calibration, ~32 test) |
| XSTest prompts | all 450 | 24, the same 24 in every language |
| Fine-tune | bf16, 3 epochs, 250 rows/language | float32, 1 epoch, 25 rows/language |
| Training, locate, repairs | PyTorch on CUDA | PyTorch on MPS (they need PyTorch hooks and gradients, which MLX models don't share) |
| 4-bit compression | bitsandbytes NF4 | optimum-quanto int4 |
| Repair sweeps | full grids | one setting each; SafeTune CLI methods skipped |
| Outputs | `data/`, `runs/`, `results/` | `data-mac/`, `runs-mac/`, `results-mac/` (git-ignored) |

The Mac and H200 outputs never mix, because they write to different folders.

## Setup (once)

```bash
cd safety-doesnt-translate
uv sync --extra dev --extra data --extra mac --extra safety --extra interp
scripts/setup_hooks.sh
export SDT_PROFILE=mac          # or pass --profile mac to every command
uv run sdt doctor
```

`sdt doctor` reports the device (should be `mps`), installed packages (`mlx` and `mlx-lm` must be present), and whether your Hugging Face account can download each model. Fix every `gated` line by accepting the licence on that model's page, then run it again.

## Run order

Run these one at a time. Only one model is in memory at once, and each step frees it before the next one loads.

```bash
uv run sdt data sources          # download Aya Red-teaming, MultiJail, XSTest, Aya Dataset
uv run sdt data translate        # XSTest -> 7 languages with Aya Expanse 8B
uv run sdt data build            # curate + split -> data-mac/processed/prompts.jsonl
uv run sdt data task             # harmless task data
uv run sdt finetune              # LoRA -> runs-mac/models/drifted
uv run sdt evaluate global
uv run sdt evaluate drifted
uv run sdt report drift
uv run sdt locate directions --models global drifted
uv run sdt repair task-arithmetic
uv run sdt evaluate repair-task-arithmetic-0.5
uv run sdt report table
```

Then the remaining steps, in the same order as on the H200:

```bash
uv run sdt generate global --split calib && uv run sdt judge global --split calib
uv run sdt locate components
uv run sdt repair oneshot && uv run sdt repair circuit-patch && uv run sdt repair steer
uv run sdt compress mixed --model repair-task-arithmetic-0.5
uv run sdt evaluate repair-task-arithmetic-0.5-4bit-mixed --backend hf
```

Add `-v` to any command for per-batch progress.

## Tips

- **Close memory-heavy apps** (browsers with many tabs, Docker) before the judge steps. The 8B judge takes ~16 GB.
- **Downloads:** the models together are ~35 GB in `~/.cache/huggingface`.
- **First run of a model is slow** while it downloads. MLX reads the same Hugging Face checkpoints as PyTorch, so nothing is downloaded twice.
- **Compare backends** on the same model with `sdt generate global --backend hf`: greedy replies should mostly match the MLX ones; small differences come from numerics.
- **If MPS reports an unsupported operation**, set `PYTORCH_ENABLE_MPS_FALLBACK=1` to run that operation on the CPU.
- **Wipe Mac outputs** with `rm -rf data-mac runs-mac results-mac` when you change prompts or models.
