"""SafeTune repairs: ReStA through the Python API, other recover methods through its CLI."""

from __future__ import annotations

import logging
import subprocess

from sdt.config import Config
from sdt.repair.common import (
    fraction_changed,
    gpu_timer,
    load_state_dict,
    record_repair,
    repair_name,
)

log = logging.getLogger(__name__)


def run_resta(config: Config, alpha: float, source: str = "drifted") -> str:
    """ReStA (documented API): drifted + base + aligned references, no training."""
    from safetune.runner.recover import ReStaTrainer

    from sdt.models import load_hf_model, load_tokenizer

    name = repair_name(source, f"repair-resta-{alpha:g}")
    out = config.runs_dir / "models" / name
    with gpu_timer() as timer:
        drifted = load_hf_model(config.model_path(source))
        # References stay on CPU so one GPU holds only the model being patched (SafeTune README).
        base = load_hf_model(config.model_path("base"), device_map="cpu")
        aligned = load_hf_model(config.model_path("global"), device_map="cpu")
        # `alpha` scales the safety vector (aligned - base) added to the drifted weights.
        patched = ReStaTrainer(drifted, base_model=base, aligned_model=aligned, alpha=alpha).apply()
        # Saving re-ties lm_head to the embeddings; if ReStA had edited them as two tensors, one
        # edit would be lost on reload without notice.
        head, embed = patched.get_output_embeddings(), patched.get_input_embeddings()
        if (
            getattr(patched.config, "tie_word_embeddings", False)
            and head.weight is not embed.weight
        ):
            raise RuntimeError("ReStA untied lm_head from the embeddings; the save would drop one")
        patched.save_pretrained(out, safe_serialization=True)
        load_tokenizer(config.model_path(source)).save_pretrained(out)
    changed = fraction_changed(load_state_dict(config, source), load_state_dict(config, name))
    record_repair(
        config, name, changed, timer["minutes"], method="resta", alpha=alpha, source=source
    )
    return name


def run_cli(config: Config, algo: str, source: str = "drifted") -> str:
    """`safetune patch --algo <algo>`; algorithm names come from `safetune list`."""
    name = repair_name(source, f"repair-safetune-{algo}")
    out = config.runs_dir / "models" / name
    cmd = [
        "safetune",
        "patch",
        "--model",
        config.model_path(source),
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
    changed = fraction_changed(load_state_dict(config, source), load_state_dict(config, name))
    record_repair(config, name, changed, timer["minutes"], method=f"safetune:{algo}", source=source)
    return name
