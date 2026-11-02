"""Loads Hugging Face models, including 4-bit specs from `sdt compress` and steering specs."""

from __future__ import annotations

import contextlib
import logging
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from sdt.config import Config
from sdt.io import read_json

log = logging.getLogger(__name__)

QUANT_SPEC = "quantization.json"
STEER_SPEC = "steer.json"


def spec_file(config: Config, name: str, filename: str) -> Path | None:
    path = config.runs_dir / "models" / name / filename
    return path if path.exists() else None


def needs_hf_backend(config: Config, name: str) -> bool:
    """4-bit and steered models are generated with transformers, not vLLM."""
    return any(spec_file(config, name, f) for f in (QUANT_SPEC, STEER_SPEC))


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"  # batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_hf_model(
    path: str, device_map: str | None = "auto", dtype: str = "bfloat16", **kwargs: Any
):
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=getattr(torch, dtype), device_map=device_map, **kwargs
    )


def load_for_generation(config: Config, name: str) -> tuple[Any, Any, AbstractContextManager]:
    """Returns (model, tokenizer, context) where generation must run inside `context`.

    The context is a no-op except for steered models, where it switches the steering hooks on.
    """
    quant = spec_file(config, name, QUANT_SPEC)
    steer = spec_file(config, name, STEER_SPEC)
    if quant:
        spec = read_json(quant)
        source = config.model_path(spec["source"])
        model = load_hf_model(source, quantization_config=bnb_config(spec))
        return model, load_tokenizer(source), contextlib.nullcontext()
    if steer:
        from sdt.repair.steering import build_steering_context

        spec = read_json(steer)
        source = config.model_path(spec["source"])
        model, tok = load_hf_model(source), load_tokenizer(source)
        return model, tok, build_steering_context(config, model, tok, spec)
    path = config.model_path(name)
    return load_hf_model(path), load_tokenizer(path), contextlib.nullcontext()


def bnb_config(spec: dict[str, Any]):
    """bitsandbytes 4-bit config; `skip_modules` stay in bf16 (mixed precision)."""
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=spec.get("quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=spec.get("skip_modules") or None,
    )


def decoder_layers(model) -> list:
    """The transformer blocks of a decoder-only HF model (Cohere2: model.model.layers)."""
    for attr in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    raise AttributeError(f"Cannot find decoder layers on {type(model).__name__}")


def chat_prompt(tokenizer, user_text: str) -> str:
    """The tokenizer's default chat template with one user turn and no extra system prompt."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True
    )
