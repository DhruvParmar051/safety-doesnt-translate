"""Generates replies with one of three backends.

- vllm: the GPU box (CUDA only).
- mlx: Apple Silicon, through Apple's MLX (mlx-lm); the Mac profile's default.
- hf: transformers; always used for 4-bit and steered models, which need PyTorch hooks.
"""

from __future__ import annotations

import logging
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

from sdt.config import Config
from sdt.models import (
    load_for_generation,
    load_hf_model,
    load_tokenizer,
    needs_hf_backend,
    release_memory,
)
from sdt.schema import GenerationRecord, PromptRecord

log = logging.getLogger(__name__)

Conversation = list[dict[str, str]]


@dataclass
class GenerationStats:
    n_prompts: int
    n_new_tokens: int
    seconds: float
    backend: str = ""  # the one that actually ran: 4-bit and steered models force transformers

    @property
    def tokens_per_second(self) -> float:
        return self.n_new_tokens / self.seconds if self.seconds else 0.0


def length_order(texts: list[str]) -> list[int]:
    """Indices that sort texts longest first, so each batch pads to similar lengths."""
    return sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)


def chat(
    config: Config,
    model_path: str,
    conversations: list[Conversation],
    max_new_tokens: int | None = None,
    stats: list[GenerationStats] | None = None,
) -> list[str]:
    """Greedy chat completions for helper models (translator, judges) on the configured backend."""
    backend = config["generation"]["backend"]
    if backend == "vllm":
        return vllm_chat(config, model_path, conversations, max_new_tokens, stats)
    if backend == "mlx":
        return mlx_chat(config, model_path, conversations, max_new_tokens, stats)
    model, tokenizer = load_hf_model(model_path), load_tokenizer(model_path)
    try:
        return hf_chat(config, model, tokenizer, conversations, max_new_tokens, stats)
    finally:
        del model
        release_memory()


def chat_token_ids(tokenizer, conversation: Conversation) -> list[int]:
    """The chat template's token ids. transformers 5 returns a dict unless asked for a list."""
    return list(
        tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_dict=False)
    )


def vllm_chat(
    config: Config,
    model_path: str,
    conversations: list[Conversation],
    max_new_tokens: int | None = None,
    stats: list[GenerationStats] | None = None,
) -> list[str]:
    """Greedy chat completions for many conversations with one vLLM engine."""
    from vllm import LLM, SamplingParams

    gen = config["generation"]
    llm = LLM(
        model=model_path, dtype="bfloat16", max_model_len=gen["max_model_len"], seed=config.seed
    )
    tokenizer = llm.get_tokenizer()
    # Token ids, not text: the chat template already adds BOS, and vLLM would add a second one
    # when tokenizing a string. The other backends also start from the template's own ids.
    prompts = [{"prompt_token_ids": chat_token_ids(tokenizer, conv)} for conv in conversations]
    params = SamplingParams(
        temperature=gen["temperature"],
        max_tokens=max_new_tokens or gen["max_new_tokens"],
        seed=config.seed,
    )
    start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    seconds = time.perf_counter() - start
    if stats is not None:
        n_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        stats.append(GenerationStats(len(prompts), n_tokens, seconds))
    # The engine runs in a child process that `del` does not always stop; on a small GPU its
    # leftover memory would stop the next engine (e.g. the judge) from starting.
    engine_core = getattr(llm.llm_engine, "engine_core", None)
    if hasattr(engine_core, "shutdown"):
        engine_core.shutdown()
    del llm
    release_memory()
    return [o.outputs[0].text for o in outputs]


def hf_chat(
    config: Config,
    model,
    tokenizer,
    conversations: list[Conversation],
    max_new_tokens: int | None = None,
    stats: list[GenerationStats] | None = None,
    context: AbstractContextManager | None = None,
) -> list[str]:
    """Batched greedy generation with transformers, inside `context` (e.g. steering hooks)."""
    import torch

    gen = config["generation"]
    prompts = [
        tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for conv in conversations
    ]
    order = length_order(prompts)
    replies: list[str] = [""] * len(prompts)
    n_tokens, start = 0, time.perf_counter()
    model.eval()
    with context or nullcontext(), torch.no_grad():
        for b in range(0, len(order), gen["batch_size"]):
            idx = order[b : b + gen["batch_size"]]
            inputs = tokenizer(
                [prompts[i] for i in idx],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(model.device)
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or gen["max_new_tokens"],
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            new = out[:, inputs["input_ids"].shape[1] :]
            n_tokens += int((new != tokenizer.pad_token_id).sum())
            for i, text in zip(
                idx, tokenizer.batch_decode(new, skip_special_tokens=True), strict=True
            ):
                replies[i] = text
            log.debug("Generated %d / %d", min(b + gen["batch_size"], len(order)), len(order))
    if stats is not None:
        stats.append(GenerationStats(len(prompts), n_tokens, time.perf_counter() - start))
    return replies


def mlx_chat(
    config: Config,
    model_path: str,
    conversations: list[Conversation],
    max_new_tokens: int | None = None,
    stats: list[GenerationStats] | None = None,
) -> list[str]:
    """Greedy chat completions on Apple Silicon with MLX; loads Hub ids and local HF checkpoints."""
    import mlx.core as mx
    from mlx_lm import batch_generate, load

    gen = config["generation"]
    model, tokenizer = load(model_path)
    prompts = [chat_token_ids(tokenizer, conv) for conv in conversations]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
    replies: list[str] = [""] * len(prompts)
    n_tokens, start = 0, time.perf_counter()
    try:
        for b in range(0, len(order), gen["batch_size"]):
            idx = order[b : b + gen["batch_size"]]
            # The default sampler is greedy (temperature 0), matching vLLM and transformers.
            out = batch_generate(
                model,
                tokenizer,
                [prompts[i] for i in idx],
                max_tokens=max_new_tokens or gen["max_new_tokens"],
                return_token_ids=True,
            )
            for i, text, ids in zip(idx, out.texts, out.token_ids, strict=True):
                replies[i] = text
                n_tokens += len(ids)
            log.debug("Generated %d / %d", min(b + gen["batch_size"], len(order)), len(order))
    finally:
        del model
        mx.clear_cache()
    if stats is not None:
        stats.append(GenerationStats(len(prompts), n_tokens, time.perf_counter() - start))
    return replies


def generate_for_model(
    config: Config, name: str, records: list[PromptRecord], backend: str | None = None
) -> tuple[list[GenerationRecord], GenerationStats]:
    """Replies from model `name` to every record; 4-bit and steered models always use transformers."""
    backend = backend or config["generation"]["backend"]
    if needs_hf_backend(config, name):
        backend = "hf"
    stats: list[GenerationStats] = []
    conversations = [[{"role": "user", "content": r.prompt}] for r in records]
    log.info("Generating %d replies from %s with %s", len(records), name, backend)
    if backend == "vllm":
        replies = vllm_chat(config, config.model_path(name), conversations, stats=stats)
    elif backend == "mlx":
        replies = mlx_chat(config, config.model_path(name), conversations, stats=stats)
    else:
        model, tokenizer, context = load_for_generation(config, name)
        try:
            replies = hf_chat(config, model, tokenizer, conversations, stats=stats, context=context)
        finally:
            del model
            release_memory()
    stats[0].backend = backend
    rows = [
        GenerationRecord(r.id, name, r.lang, r.kind, r.prompt, reply.strip(), source=r.source)
        for r, reply in zip(records, replies, strict=True)
    ]
    return rows, stats[0]
