"""Task arithmetic (Ilharco et al.): add the safety vector (Global - Base) back onto the drifted model."""

from __future__ import annotations

import logging

from sdt.config import Config
from sdt.repair.common import (
    StateDict,
    fraction_changed,
    gpu_timer,
    is_patchable,
    load_state_dict,
    record_repair,
    repair_name,
    save_model,
)

log = logging.getLogger(__name__)


def apply_task_arithmetic(
    drifted: StateDict, aligned: StateDict, base: StateDict, scale: float
) -> StateDict:
    """θ = θ_drifted + scale · (θ_aligned - θ_base), per tensor; mismatched tensors are left as they are."""
    out = {}
    for key, value in drifted.items():
        a, b = aligned.get(key), base.get(key)
        if (
            not is_patchable(value)
            or a is None
            or b is None
            or a.shape != value.shape
            or b.shape != value.shape
        ):
            if is_patchable(value):
                log.warning("Not patched (missing or shape mismatch): %s", key)
            out[key] = value
            continue
        out[key] = (value.float() + scale * (a.float() - b.float())).to(value.dtype)
    return out


def run(config: Config, scale: float, source: str = "drifted") -> str:
    name = repair_name(source, f"repair-task-arithmetic-{scale:g}")
    with gpu_timer() as timer:
        drifted = load_state_dict(config, source)
        patched = apply_task_arithmetic(
            drifted, load_state_dict(config, "global"), load_state_dict(config, "base"), scale
        )
        save_model(config, patched, name, skeleton=source)
    record_repair(
        config,
        name,
        fraction_changed(drifted, patched),
        timer["minutes"],
        method="task_arithmetic",
        scale=scale,
        source=source,
    )
    return name
