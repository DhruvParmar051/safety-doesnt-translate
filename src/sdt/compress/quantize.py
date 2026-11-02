"""Uniform 4-bit vs mixed precision (located components kept in bf16).

bitsandbytes on CUDA, optimum-quanto (int4 weights) elsewhere, e.g. the Mac.

A compressed model is a spec (`quantization.json`) applied at load time, so no 4-bit weights are
saved; `size.json` records the measured memory footprint.
"""

from __future__ import annotations

import logging

from sdt.config import Config
from sdt.io import read_json, write_json
from sdt.models import QUANT_SPEC, default_device, load_hf_model, load_quantized

log = logging.getLogger(__name__)


def skip_modules(components: list[dict]) -> list[str]:
    """Module names bitsandbytes must leave unquantized for the given components."""
    names: list[str] = []
    for comp in components:
        prefix = f"model.layers.{comp['layer']}."
        wanted = (
            [prefix + "mlp"]
            if comp["kind"] == "mlp"
            else [prefix + "self_attn.q_proj", prefix + "self_attn.o_proj"]
        )
        names += [n for n in wanted if n not in names]
    # The output head is commonly kept in full precision as well.
    return [*names, "lm_head"]


def quant_backend(config: Config) -> str:
    """'bnb' or 'quanto'; `auto` picks bitsandbytes only where CUDA exists."""
    choice = config["compress"].get("backend", "auto")
    if choice == "auto":
        return "bnb" if default_device() == "cuda" else "quanto"
    return choice


def measure_size(config: Config, name: str) -> float:
    """Loads the model as `sdt generate` would and records its memory footprint in GB."""
    spec_path = config.runs_dir / "models" / name / QUANT_SPEC
    if spec_path.exists():
        model = load_quantized(config, read_json(spec_path))
    else:
        model = load_hf_model(config.model_path(name))
    size_gb = model.get_memory_footprint() / 1e9
    write_json(config.runs_dir / "models" / name / "size.json", {"size_gb": size_gb})
    log.info("%s: %.2f GB", name, size_gb)
    return size_gb


def compress(config: Config, source: str, mode: str) -> str:
    """mode 'uniform' quantizes everything; 'mixed' protects the top located components."""
    spec_cfg = config["compress"]
    skip: list[str] = [
        "lm_head"
    ]  # both modes keep the output head, as bitsandbytes does by default
    if mode == "mixed":
        components = read_json(config.runs_dir / "locate" / "components.json")
        skip = skip_modules(components[: spec_cfg["protect_top_components"]])
    elif mode != "uniform":
        raise ValueError(f"Unknown mode '{mode}'")
    name = f"{source}-4bit-{mode}"
    write_json(
        config.runs_dir / "models" / name / QUANT_SPEC,
        {
            "source": source,
            "backend": quant_backend(config),
            "quant_type": spec_cfg["bnb_quant_type"],
            "skip_modules": skip,
        },
    )
    measure_size(config, name)
    return name


def circuitkit_quantize(
    config: Config, source: str, circuit_path: str, high_fraction: float = 0.3
) -> None:
    """CircuitKIT's own circuit-aware quantization, run on a circuit from `sdt locate circuitkit`."""
    import circuitkit as ck

    model = load_hf_model(config.model_path(source))
    circuit = ck.load_scores(circuit_path)
    plan = ck.quantize(model, circuit, bits=4, high_fraction=high_fraction, backend="llmcompressor")
    write_json(config.run_path("compress", f"{source}-circuitkit-plan.json"), {"plan": str(plan)})
