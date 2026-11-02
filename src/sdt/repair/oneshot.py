"""One-shot saliency patch: copy back from Global only the weights that matter most for refusal.

Saliency of a weight = |∂L/∂θ · (θ_global - θ_drifted)|, the first-order estimate of how much
restoring that weight lowers L, where L is the drifted model's loss on Global's own refusals to
calibration-split harmful prompts.
"""

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
    save_model,
)

log = logging.getLogger(__name__)

SAMPLE_SIZE = 5_000_000


def global_threshold(
    saliency: StateDict, fraction: float, sample_size: int = SAMPLE_SIZE, seed: int = 0
) -> float:
    """Value above which `fraction` of all entries lie; exact for small models, sampled otherwise."""
    import torch

    total = sum(t.numel() for t in saliency.values())
    if total <= sample_size:
        flat = torch.cat([t.flatten().float() for t in saliency.values()])
    else:
        generator = torch.Generator().manual_seed(seed)
        parts = []
        for t in saliency.values():
            k = max(1, round(sample_size * t.numel() / total))
            idx = torch.randint(t.numel(), (k,), generator=generator)
            parts.append(t.flatten()[idx].float())
        flat = torch.cat(parts)
    k = max(1, round(fraction * flat.numel()))
    return float(torch.topk(flat, k).values.min())


def oneshot_patch(
    drifted: StateDict, aligned: StateDict, saliency: StateDict, fraction: float
) -> StateDict:
    """Replaces the top-`fraction` most salient entries with Global's values."""
    import torch

    threshold = global_threshold(saliency, fraction)
    out = {}
    for key, value in drifted.items():
        if key in saliency and is_patchable(value):
            mask = saliency[key] >= threshold
            out[key] = torch.where(mask, aligned[key].to(value.dtype), value)
        else:
            out[key] = value
    return out


def refusal_pairs(config: Config, per_language: int | None = None) -> list[dict]:
    """(calibration harmful prompt, Global's refusal) as chat messages, from the judged calib run."""
    from sdt.evaluate.pipeline import (
        generations_path,
        judgements_path,
        load_generations,
        load_judgements,
    )
    from sdt.schema import Kind, Label, Split

    gens_file = generations_path(config, "global", Split.CALIB)
    judge_file = judgements_path(config, "global", config["judge"]["primary"], Split.CALIB)
    if not judge_file.exists():
        raise FileNotFoundError(
            "Run `sdt generate global --split calib` and `sdt judge global --split calib` first"
        )
    refused = {j.id for j in load_judgements(judge_file) if j.label == Label.REFUSAL}
    counts: dict[str, int] = {}
    rows = []
    for g in load_generations(gens_file):
        if g.kind != Kind.HARMFUL or g.id not in refused:
            continue
        if per_language is not None and counts.get(g.lang, 0) >= per_language:
            continue
        counts[g.lang] = counts.get(g.lang, 0) + 1
        rows.append(
            {
                "lang": g.lang,
                "messages": [
                    {"role": "user", "content": g.prompt},
                    {"role": "assistant", "content": g.response},
                ],
            }
        )
    return rows


def compute_saliency(config: Config, aligned: StateDict) -> StateDict:
    """|grad · (θ_global - θ_drifted)| for every weight, from the refusal loss on the drifted model."""
    import torch

    from sdt.models import load_hf_model, load_tokenizer

    path = config.model_path("drifted")
    model, tok = load_hf_model(path), load_tokenizer(path)
    model.train()
    tok.padding_side = "right"
    pairs = refusal_pairs(config)
    for i in range(0, len(pairs), 4):
        chunk = pairs[i : i + 4]
        prompts = [
            tok.apply_chat_template(p["messages"][:1], tokenize=False, add_generation_prompt=True)
            for p in chunk
        ]
        fulls = [tok.apply_chat_template(p["messages"], tokenize=False) for p in chunk]
        enc = tok(fulls, return_tensors="pt", padding=True, add_special_tokens=False).to(
            model.device
        )
        labels = enc["input_ids"].clone()
        for row, prompt in enumerate(prompts):  # loss only on the refusal tokens
            labels[row, : len(tok(prompt, add_special_tokens=False).input_ids)] = -100
        labels[enc["attention_mask"] == 0] = -100
        loss = model(**enc, labels=labels).loss * len(chunk) / len(pairs)
        loss.backward()

    saliency = {}
    with torch.no_grad():
        for key, param in model.named_parameters():
            if param.grad is None or key not in aligned:
                continue
            delta = aligned[key].to(param.device, torch.float32) - param.float()
            saliency[key] = (param.grad.float() * delta).abs().cpu()
    return saliency


def run(config: Config, fraction: float) -> str:
    name = f"repair-oneshot-{fraction:g}"
    with gpu_timer() as timer:
        aligned = load_state_dict(config, "global")
        saliency = compute_saliency(config, aligned)
        drifted = load_state_dict(config, "drifted")
        patched = oneshot_patch(drifted, aligned, saliency, fraction)
        save_model(config, patched, name)
    record_repair(
        config,
        name,
        fraction_changed(drifted, patched),
        timer["minutes"],
        method="oneshot",
        fraction=fraction,
    )
    return name
