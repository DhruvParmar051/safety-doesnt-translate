"""LoRA fine-tuning: creates the drifted model and the plain safety fine-tune baseline (owner: Dhruv)."""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path

from sdt.config import Config, save_run_config
from sdt.io import read_jsonl, write_json
from sdt.models import load_hf_model, load_tokenizer

log = logging.getLogger(__name__)
WARMUP_RATIO = 0.03  # share of training steps spent warming up the learning rate


def _sft_config(config: Config, out_dir: Path, seed: int | None = None):
    """SFTConfig across TRL versions (max-length and warmup fields were renamed)."""
    import torch
    from trl import SFTConfig

    spec = config["finetune"]
    kwargs = {
        "output_dir": str(out_dir / "trainer"),
        "num_train_epochs": spec["epochs"],
        "per_device_train_batch_size": spec["batch_size"],
        "gradient_accumulation_steps": spec["grad_accum"],
        "learning_rate": spec["learning_rate"],
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "save_strategy": "no",
        # Mixed precision needs CUDA; on the Mac the LoRA trains in full precision on MPS.
        "bf16": torch.cuda.is_available(),
        "gradient_checkpointing": bool(spec.get("gradient_checkpointing", False)),
        "seed": config.seed if seed is None else seed,
        "report_to": "none",
        "completion_only_loss": True,
    }
    names = {f.name for f in dataclasses.fields(SFTConfig)}
    kwargs["max_length" if "max_length" in names else "max_seq_length"] = spec["max_seq_len"]
    # transformers 5 dropped warmup_ratio; its warmup_steps takes a float below 1 as a ratio.
    kwargs["warmup_ratio" if "warmup_ratio" in names else "warmup_steps"] = WARMUP_RATIO
    return SFTConfig(**kwargs)


def to_prompt_completion(messages: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Conversational prompt/completion row: TRL then puts the loss on the answer only.

    Training on the user turn too would teach the model to write prompts, including harmful ones
    in the safety fine-tune baseline.
    """
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Each training row must end with the assistant's reply")
    return {"prompt": messages[:-1], "completion": messages[-1:]}


def train_lora(
    config: Config, source: str, rows: list[dict], out_dir: Path, seed: int | None = None
) -> float:
    """Trains LoRA on chat `messages`, saves the adapter and the merged model; returns wall-clock minutes."""
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTTrainer

    spec = config["finetune"]
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(config, out_dir, step="finetune", source=source, n_rows=len(rows), seed=seed)

    tokenizer = load_tokenizer(source)
    tokenizer.padding_side = "right"  # training pads on the right
    import torch

    # bf16 weights train fine on CUDA; elsewhere float32 is the stable choice (3.35B fits in 48 GB).
    dtype = "bfloat16" if torch.cuda.is_available() else "float32"
    model = load_hf_model(source, device_map="auto", dtype=dtype)
    peft_config = LoraConfig(
        r=spec["lora_r"],
        lora_alpha=spec["lora_alpha"],
        lora_dropout=spec["lora_dropout"],
        target_modules=spec["target_modules"],
        task_type="CAUSAL_LM",
    )
    dataset = Dataset.from_list([to_prompt_completion(r["messages"]) for r in rows])
    trainer = SFTTrainer(
        model=model,
        args=_sft_config(config, out_dir, seed),
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    start = time.perf_counter()
    trainer.train()
    minutes = (time.perf_counter() - start) / 60

    trainer.model.save_pretrained(out_dir / "adapter")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.padding_side = "left"
    tokenizer.save_pretrained(out_dir)
    log.info("Saved merged model to %s (%.1f GPU min)", out_dir, minutes)
    return minutes


DRIFT_VARIANTS = ("harmless", "harmful")


def train_drifted(
    config: Config, variant: str = "harmless", seed: int | None = None, base: str = "global"
) -> Path:
    """Global + LoRA on the harmless task = the drifted model every repair starts from.

    variant "harmful" adds a few harmful English examples (`sdt data harmful`) to the same task
    data, the stronger drift of Qi et al. (2023), saved as `drifted-harmful`.

    `seed` other than the project seed retrains on the same rows with a different LoRA init and
    data order, saved as `<name>-s<seed>`: a replicate that shows the drift is not one run's luck.
    `base` other than global fine-tunes a regional variant (fire, water, earth) the same way, saved
    as `<name>-<base>`: does a language's share of training data make its safety harder to drift?
    """
    processed = config.data_dir / "processed"
    rows = list(read_jsonl(processed / "task_train.jsonl"))
    name = "drifted"
    if variant == "harmful":
        harmful = list(read_jsonl(processed / "harmful_train.jsonl"))
        rows += harmful
        name = "drifted-harmful"
    elif variant != "harmless":
        raise ValueError(f"Unknown drift variant '{variant}'")
    if base != "global":
        name += f"-{base}"
    if seed is not None and seed != config.seed:
        name += f"-s{seed}"
    out = config.runs_dir / "models" / name
    minutes = train_lora(config, config.model_path(base), rows, out, seed)
    write_json(
        config.run_path("finetune", f"{name}.json"),
        {
            "gpu_minutes": minutes,
            "rows": len(rows),
            "seed": config.seed if seed is None else seed,
            "base": base,
        },
    )
    return out
