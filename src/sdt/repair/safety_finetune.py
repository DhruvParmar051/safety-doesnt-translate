"""Baseline: plain safety fine-tuning of the drifted model. This is retraining, the cost bar to beat."""

from __future__ import annotations

from sdt.config import Config
from sdt.finetune import train_lora
from sdt.io import read_jsonl
from sdt.repair.common import fraction_changed, load_state_dict, record_repair
from sdt.repair.oneshot import refusal_pairs


def run(config: Config) -> str:
    """LoRA on the drifted model: task data plus Global's refusals, so the task skill is kept."""
    name = "repair-safety-ft"
    per_language = config.repair["safety_finetune"]["refusals_per_language"]
    rows = list(read_jsonl(config.data_dir / "processed" / "task_train.jsonl"))
    rows += refusal_pairs(config, per_language)
    minutes = train_lora(
        config, config.model_path("drifted"), rows, config.runs_dir / "models" / name
    )
    changed = fraction_changed(load_state_dict(config, "drifted"), load_state_dict(config, name))
    record_repair(config, name, changed, minutes, method="safety_finetune", rows=len(rows))
    return name
