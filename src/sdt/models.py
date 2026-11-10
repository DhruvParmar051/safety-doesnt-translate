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


def default_device() -> str:
    """'cuda' on NVIDIA GPUs, 'mps' on Apple Silicon, else 'cpu'."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def release_memory() -> None:
    """Returns cached accelerator memory so the next model (e.g. the judge) fits."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_hf_model(
    path: str, device_map: str | dict | None = "auto", dtype: str = "bfloat16", **kwargs: Any
):
    """Loads a causal LM; `device_map="auto"` means all GPUs on CUDA, the GPU on Apple Silicon."""
    import torch
    from transformers import AutoModelForCausalLM

    if device_map == "auto" and default_device() != "cuda":
        device_map = {"": default_device()}
    return AutoModelForCausalLM.from_pretrained(
        path, dtype=getattr(torch, dtype), device_map=device_map, **kwargs
    )


def load_quantized(config: Config, spec: dict[str, Any]):
    """A 4-bit model from a `sdt compress` spec: bitsandbytes on CUDA, optimum-quanto elsewhere."""
    source = config.model_path(spec["source"])
    if spec.get("backend") == "none":  # the bf16 reference, generated the same way as 4-bit
        return load_hf_model(source)
    if spec.get("backend", "bnb") == "bnb":
        return load_hf_model(source, quantization_config=bnb_config(spec))
    from optimum.quanto import freeze, qint4, quantize

    model = load_hf_model(source)
    quantize(model, weights=qint4, exclude=quanto_patterns(spec.get("skip_modules") or []))
    freeze(model)
    return model


def quanto_patterns(skip_modules: list[str]) -> list[str]:
    """optimum-quanto matches `exclude` with fnmatch, so a module prefix needs a trailing '*'."""
    return [name if name == "lm_head" else f"{name}*" for name in skip_modules]


def load_for_generation(config: Config, name: str) -> tuple[Any, Any, AbstractContextManager]:
    """Returns (model, tokenizer, context) where generation must run inside `context`.

    The context is a no-op except for steered models, where it switches the steering hooks on.
    """
    quant = spec_file(config, name, QUANT_SPEC)
    steer = spec_file(config, name, STEER_SPEC)
    if quant:
        spec = read_json(quant)
        model = load_quantized(config, spec)
        return model, load_tokenizer(config.model_path(spec["source"])), contextlib.nullcontext()
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
