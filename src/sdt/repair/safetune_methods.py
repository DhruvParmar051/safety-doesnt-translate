"""SafeTune repairs: ReStA through the Python API, other recover methods through its CLI."""

from __future__ import annotations

import logging
import subprocess

from sdt.config import Config
from sdt.repair.common import fraction_changed, gpu_timer, load_state_dict, record_repair

log = logging.getLogger(__name__)


def run_resta(config: Config, alpha: float) -> str:
    """ReStA (documented API): drifted + base + aligned references, no training."""
    from safetune.runner.recover import ReStaTrainer

    from sdt.models import load_hf_model, load_tokenizer

    name = f"repair-resta-{alpha:g}"
    out = config.runs_dir / "models" / name
    with gpu_timer() as timer:
        drifted = load_hf_model(config.model_path("drifted"))
        # References stay on CPU so one GPU holds only the model being patched (SafeTune README).
        base = load_hf_model(config.model_path("base"), device_map="cpu")
        aligned = load_hf_model(config.model_path("global"), device_map="cpu")
        # VERIFY(M1): `alpha` appears in SafeTune's docs example but is not described there.
        patched = ReStaTrainer(drifted, base_model=base, aligned_model=aligned, alpha=alpha).apply()
        patched.save_pretrained(out, safe_serialization=True)
        load_tokenizer(config.model_path("drifted")).save_pretrained(out)
    changed = fraction_changed(load_state_dict(config, "drifted"), load_state_dict(config, name))
    record_repair(config, name, changed, timer["minutes"], method="resta", alpha=alpha)
    return name


def run_cli(config: Config, algo: str) -> str:
    """`safetune patch --algo <algo>`; algorithm names come from `safetune list`."""
    name = f"repair-safetune-{algo}"
    out = config.runs_dir / "models" / name
    cmd = [
        "safetune",
        "patch",
        "--model",
        config.model_path("drifted"),
        "--algo",
        algo,
        "--base",
        config.model_path("base"),
        "--aligned",
        config.model_path("global"),
        "--output",
        str(out),
    ]
    log.info("Running %s", " ".join(cmd))
    with gpu_timer() as timer:
        subprocess.run(cmd, check=True)
    changed = fraction_changed(load_state_dict(config, "drifted"), load_state_dict(config, name))
    record_repair(config, name, changed, timer["minutes"], method=f"safetune:{algo}")
    return name
