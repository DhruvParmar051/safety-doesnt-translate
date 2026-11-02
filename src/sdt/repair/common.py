"""State-dict loading, saving and bookkeeping shared by the weight-space repairs."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sdt.config import Config, save_run_config
from sdt.io import write_json

log = logging.getLogger(__name__)

StateDict = dict[str, Any]


def unique_tensors(state: StateDict) -> StateDict:
    """Drops keys that alias an earlier key's storage (tied weights such as lm_head = embeddings).

    Tiny Aya ties its output head to the input embeddings: `state_dict()` lists both keys, but
    they are one tensor. Keeping both lets a repair patch one copy and not the other, and loading
    them back writes the unpatched copy over the patched one. Each tensor is kept once here and
    re-tied when the model is saved.
    """
    seen: set[tuple[int, int]] = set()
    out = {}
    for key, value in state.items():
        ident = (value.untyped_storage().data_ptr(), value.storage_offset())
        if ident in seen:
            continue
        seen.add(ident)
        out[key] = value
    return out


def load_state_dict(config: Config, name: str) -> StateDict:
    """All weights of a model on CPU, in bf16, each tied tensor once."""
    from sdt.models import load_hf_model

    model = load_hf_model(config.model_path(name), device_map="cpu")
    return unique_tensors({k: v.detach() for k, v in model.state_dict().items()})


def apply_state(model, state: StateDict) -> None:
    """Loads `state` into `model`; only tied aliases may be missing, and they are re-tied."""
    result = model.load_state_dict(state, strict=False)
    tied = set(model.state_dict()) - set(unique_tensors(model.state_dict()))
    unexpected = list(result.unexpected_keys)
    missing = [k for k in result.missing_keys if k not in tied]
    if unexpected or missing:
        raise ValueError(
            f"State does not match the model: missing {missing}, unexpected {unexpected}"
        )
    model.tie_weights()


def is_patchable(tensor) -> bool:
    return tensor.is_floating_point()


def fraction_changed(before: StateDict, after: StateDict) -> float:
    """Share of floating-point parameters whose value differs."""
    changed = total = 0
    for key, old in before.items():
        if not is_patchable(old):
            continue
        total += old.numel()
        changed += int((after[key] != old).sum())
    return changed / total if total else 0.0


def save_model(config: Config, state: StateDict, name: str, skeleton: str = "drifted") -> Path:
    """Saves `state` with the drifted model's architecture and tokenizer as runs/models/<name>."""
    from sdt.models import load_hf_model, load_tokenizer

    out = config.runs_dir / "models" / name
    model = load_hf_model(config.model_path(skeleton), device_map="cpu")
    apply_state(model, state)
    model.save_pretrained(out, safe_serialization=True)
    load_tokenizer(config.model_path(skeleton)).save_pretrained(out)
    log.info("Saved %s", out)
    return out


@contextmanager
def gpu_timer() -> Iterator[dict[str, float]]:
    """Wall-clock minutes of the wrapped block (on the GPU box this is the GPU time we pay for)."""
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["minutes"] = (time.perf_counter() - start) / 60


def record_repair(
    config: Config, name: str, weights_changed: float | None, minutes: float, **extra: Any
) -> None:
    write_json(
        config.run_path("repair", f"{name}.json"),
        {"name": name, "weights_changed": weights_changed, "gpu_minutes": minutes, **extra},
    )
    save_run_config(config, config.runs_dir / "models" / name, step="repair", **extra)
