"""Circuit patch (problem #10 angle): restore from Global only the components phase 3 located.

A head h in layer l = its query rows in q_proj and its columns in o_proj (key/value projections are
shared across a query group, so they are left alone). An MLP = all of that layer's MLP weights.
"""

from __future__ import annotations

import logging

from sdt.config import Config
from sdt.io import read_json
from sdt.repair.common import (
    StateDict,
    components_path,
    fraction_changed,
    gpu_timer,
    load_state_dict,
    record_repair,
    repair_name,
    save_model,
)

log = logging.getLogger(__name__)

LAYER_PREFIX = "model.layers.{layer}."


def circuit_patch(
    drifted: StateDict, aligned: StateDict, components: list[dict], hd: int
) -> StateDict:
    """Copies the located heads' and MLPs' weights from `aligned` into a copy of `drifted`."""
    out = dict(drifted)
    for comp in components:
        prefix = LAYER_PREFIX.format(layer=comp["layer"])
        if comp["kind"] == "mlp":
            for key in drifted:
                if key.startswith(prefix + "mlp."):
                    out[key] = aligned[key].to(drifted[key].dtype)
            continue
        rows = slice(comp["index"] * hd, (comp["index"] + 1) * hd)
        q_key, o_key = prefix + "self_attn.q_proj.weight", prefix + "self_attn.o_proj.weight"
        q, o = out[q_key].clone(), out[o_key].clone()
        q[rows, :] = aligned[q_key][rows, :].to(q.dtype)
        o[:, rows] = aligned[o_key][:, rows].to(o.dtype)
        out[q_key], out[o_key] = q, o
    return out


def run(config: Config, top_k: int, source: str = "drifted") -> str:
    from transformers import AutoConfig

    name = repair_name(source, f"repair-circuit-patch-{top_k}")
    components = read_json(components_path(config, source))[:top_k]
    model_cfg = AutoConfig.from_pretrained(config.model_path(source))
    hd = (
        getattr(model_cfg, "head_dim", None)
        or model_cfg.hidden_size // model_cfg.num_attention_heads
    )
    with gpu_timer() as timer:
        drifted = load_state_dict(config, source)
        patched = circuit_patch(drifted, load_state_dict(config, "global"), components, hd)
        save_model(config, patched, name, skeleton=source)
    record_repair(
        config,
        name,
        fraction_changed(drifted, patched),
        timer["minutes"],
        method="circuit_patch",
        top_k=top_k,
        source=source,
        components=[c["key"] for c in components],
    )
    return name
