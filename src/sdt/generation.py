"""Generates replies with vLLM (fast) or transformers (4-bit and steered models)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from sdt.config import Config
from sdt.models import chat_prompt, load_for_generation, needs_hf_backend
from sdt.schema import GenerationRecord, PromptRecord

log = logging.getLogger(__name__)


@dataclass
class GenerationStats:
    n_prompts: int
    n_new_tokens: int
    seconds: float

    @property
    def tokens_per_second(self) -> float:
        return self.n_new_tokens / self.seconds if self.seconds else 0.0


def vllm_chat(
    config: Config,
    model_path: str,
    conversations: list[list[dict[str, str]]],
    max_new_tokens: int | None = None,
    stats: list[GenerationStats] | None = None,
) -> list[str]:
    """Greedy chat completions for many conversations with one vLLM engine."""
    from vllm import LLM, SamplingParams

    gen = config["generation"]
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=gen["max_model_len"],
        seed=config.seed,
    )
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for conv in conversations
    ]
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
    _release(llm)
    return [o.outputs[0].text for o in outputs]


def _release(llm) -> None:
    """Frees GPU memory so the next engine (e.g. the judge) fits."""
    import gc

    import torch

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def hf_generate(
    config: Config, name: str, texts: list[str], stats: list[GenerationStats] | None = None
) -> list[str]:
    """Batched greedy generation with transformers, inside the model's context (steering hooks)."""
    import torch

    gen = config["generation"]
    model, tokenizer, context = load_for_generation(config, name)
    model.eval()
    replies: list[str] = []
    n_tokens, start = 0, time.perf_counter()
    with context, torch.no_grad():
        for i in range(0, len(texts), gen["batch_size"]):
            batch = [chat_prompt(tokenizer, t) for t in texts[i : i + gen["batch_size"]]]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
            inputs = inputs.to(model.device)
            out = model.generate(
                **inputs,
                max_new_tokens=gen["max_new_tokens"],
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            new = out[:, inputs["input_ids"].shape[1] :]
            n_tokens += int((new != tokenizer.pad_token_id).sum())
            replies += tokenizer.batch_decode(new, skip_special_tokens=True)
    if stats is not None:
        stats.append(GenerationStats(len(texts), n_tokens, time.perf_counter() - start))
    return replies


def generate_for_model(
    config: Config, name: str, records: list[PromptRecord], backend: str | None = None
) -> tuple[list[GenerationRecord], GenerationStats]:
    """Replies from model `name` to every record; picks the HF backend when vLLM cannot serve it."""
    backend = backend or config["generation"]["backend"]
    if needs_hf_backend(config, name):
        backend = "hf"
    stats: list[GenerationStats] = []
    texts = [r.prompt for r in records]
    log.info("Generating %d replies from %s with %s", len(texts), name, backend)
    if backend == "vllm":
        replies = vllm_chat(
            config,
            config.model_path(name),
            [[{"role": "user", "content": t}] for t in texts],
            stats=stats,
        )
    else:
        replies = hf_generate(config, name, texts, stats=stats)
    rows = [
        GenerationRecord(r.id, name, r.lang, r.kind, r.prompt, reply.strip())
        for r, reply in zip(records, replies, strict=True)
    ]
    return rows, stats[0]
