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


def _sft_config(config: Config, out_dir: Path):
    """SFTConfig across TRL versions (the max-length field was renamed in 0.19)."""
    from trl import SFTConfig

    spec = config["finetune"]
    kwargs = {
        "output_dir": str(out_dir / "trainer"),
        "num_train_epochs": spec["epochs"],
        "per_device_train_batch_size": spec["batch_size"],
        "gradient_accumulation_steps": spec["grad_accum"],
        "learning_rate": spec["learning_rate"],
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "logging_steps": 10,
        "save_strategy": "no",
        "bf16": True,
        "seed": config.seed,
        "report_to": "none",
    }
    names = {f.name for f in dataclasses.fields(SFTConfig)}
    kwargs["max_length" if "max_length" in names else "max_seq_length"] = spec["max_seq_len"]
    return SFTConfig(**kwargs)


def train_lora(config: Config, source: str, rows: list[dict], out_dir: Path) -> float:
    """Trains LoRA on chat `messages`, saves the adapter and the merged model; returns GPU minutes."""
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTTrainer

    spec = config["finetune"]
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(config, out_dir, step="finetune", source=source, n_rows=len(rows))

    tokenizer = load_tokenizer(source)
    tokenizer.padding_side = "right"  # training pads on the right
    model = load_hf_model(source, device_map="auto")
    peft_config = LoraConfig(
        r=spec["lora_r"],
        lora_alpha=spec["lora_alpha"],
        lora_dropout=spec["lora_dropout"],
        target_modules=spec["target_modules"],
        task_type="CAUSAL_LM",
    )
    dataset = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    trainer = SFTTrainer(
        model=model,
        args=_sft_config(config, out_dir),
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


def train_drifted(config: Config) -> Path:
    """Global + LoRA on the harmless task = the drifted model every repair starts from."""
    rows = list(read_jsonl(config.data_dir / "processed" / "task_train.jsonl"))
    out = config.runs_dir / "models" / "drifted"
    minutes = train_lora(config, config.model_path("global"), rows, out)
    write_json(
        config.run_path("finetune", "drifted.json"), {"gpu_minutes": minutes, "rows": len(rows)}
    )
    return out
