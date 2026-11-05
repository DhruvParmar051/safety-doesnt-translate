"""Refusal-direction steering at inference with SafeTune (weights unchanged)."""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from typing import Any

from sdt.config import Config
from sdt.io import write_json
from sdt.models import STEER_SPEC
from sdt.repair.common import repair_name

log = logging.getLogger(__name__)


def create_steered_model(config: Config, alpha: float, layer: int, source: str = "drifted") -> str:
    """Writes a steering spec; `sdt generate` then loads `source` with the hooks switched on."""
    name = repair_name(source, f"repair-steer-{alpha:g}")
    write_json(
        config.runs_dir / "models" / name / STEER_SPEC,
        {"source": source, "alpha": alpha, "layer": layer},
    )
    write_json(
        config.run_path("repair", f"{name}.json"),
        {
            "name": name,
            "weights_changed": 0.0,
            "gpu_minutes": None,
            "method": "steer",
            "alpha": alpha,
            "layer": layer,
            "source": source,
        },
    )
    return name


def build_steering_context(
    config: Config, model, tokenizer, spec: dict[str, Any]
) -> AbstractContextManager:
    """Calibrates SafeTune's refusal direction on calibration prompts from every language."""
    from safetune.runner import steer

    from sdt.locate.directions import calib_prompts
    from sdt.schema import Kind

    prompts = calib_prompts(config)
    harmful = [p for kinds in prompts.values() for p in kinds[Kind.HARMFUL]]
    harmless = [p for kinds in prompts.values() for p in kinds[Kind.HARMLESS]]
    # `layers` lists the decoder layers that get the steering hook (SafeTune 0.1.8).
    trainer = steer.RefusalDirectionTrainer(
        model, tokenizer, layers=[spec["layer"]], alpha=spec["alpha"]
    )
    # The wrapped model is a context manager: hooks on at __enter__, off at __exit__.
    wrapped, _ = trainer.calibrate(harmful=harmful, harmless=harmless)
    return wrapped
