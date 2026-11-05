"""Baseline: plain safety fine-tuning of the drifted model. This is retraining, the cost bar to beat."""

from __future__ import annotations

from sdt.config import Config
from sdt.finetune import train_lora
from sdt.io import read_jsonl
from sdt.repair.common import (
    fraction_changed,
    gpu_timer,
    load_state_dict,
    record_repair,
    repair_name,
)
from sdt.repair.oneshot import refusal_pairs


def run(config: Config, source: str = "drifted") -> str:
    """LoRA on the drifted model: task data plus Global's refusals, so the task skill is kept."""
    name = repair_name(source, "repair-safety-ft")
    per_language = config.repair["safety_finetune"]["refusals_per_language"]
    rows = list(read_jsonl(config.data_dir / "processed" / "task_train.jsonl"))
    rows += refusal_pairs(config, per_language)
    # End to end (loading, training, saving), the same span `gpu_timer` covers for weight edits.
    with gpu_timer() as timer:
        train_lora(config, config.model_path(source), rows, config.runs_dir / "models" / name)
    changed = fraction_changed(load_state_dict(config, source), load_state_dict(config, name))
    record_repair(
        config,
        name,
        changed,
        timer["minutes"],
        method="safety_finetune",
        rows=len(rows),
        source=source,
    )
    return name
