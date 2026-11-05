"""Refusal directions per language and layer (difference in means, Arditi et al. 2024)."""

from __future__ import annotations

import logging
import random
from collections import defaultdict

from sdt.config import Config
from sdt.data.build import load_prompts
from sdt.io import write_json
from sdt.models import chat_prompt, decoder_layers, load_hf_model, load_tokenizer
from sdt.schema import Kind, Split

log = logging.getLogger(__name__)


def mean_last_token_states(model, tokenizer, texts: list[str], batch_size: int = 16):
    """Mean residual stream after every decoder layer at the last prompt token: tensor [layers, d]."""
    import torch

    layers = decoder_layers(model)
    sums: list = [None] * len(layers)
    captured: dict[int, object] = {}

    def hook(index):
        def fn(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[index] = hidden[:, -1, :].float()  # left padding: -1 is the real last token

        return fn

    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]
    try:
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = [chat_prompt(tokenizer, t) for t in texts[i : i + batch_size]]
                inputs = tokenizer(
                    batch, return_tensors="pt", padding=True, add_special_tokens=False
                )
                model(**inputs.to(model.device), logits_to_keep=1)  # only hidden states are read
                for j in range(len(layers)):
                    total = captured[j].sum(dim=0).cpu()
                    sums[j] = total if sums[j] is None else sums[j] + total
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack(sums) / len(texts)


def cosine_matrix(directions: dict[str, object], layer: int) -> list[list[float]]:
    """Cosine similarity between languages' directions at one layer."""
    import torch

    langs = list(directions)
    stacked = torch.stack([directions[lang][layer] for lang in langs])
    unit = stacked / stacked.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (unit @ unit.T).tolist()


def refusal_strength(diff, reference) -> list[float]:
    """Per layer: projection of a model's harmful-minus-harmless gap on the reference direction,
    relative to the reference's own length (1.0 = as strong as the reference model)."""
    ref_norm = reference.norm(dim=-1).clamp_min(1e-8)
    unit = reference / ref_norm.unsqueeze(-1)
    return ((diff * unit).sum(dim=-1) / ref_norm).tolist()


def calib_prompts(config: Config) -> dict[str, dict[Kind, list[str]]]:
    """A seeded random sample of calibration harmful and harmless prompts per language.

    Sampling (not the first n in file order) mixes every source a language has, so directions
    are not built from whichever dataset happens to come first.
    """
    n = config["locate"]["calib_per_language"]
    pools: dict[str, dict[Kind, list]] = defaultdict(lambda: defaultdict(list))
    for p in load_prompts(config):
        if p.split == Split.CALIB and p.kind in (Kind.HARMFUL, Kind.HARMLESS):
            pools[p.lang][p.kind].append(p)
    out: dict[str, dict[Kind, list[str]]] = defaultdict(lambda: defaultdict(list))
    for lang, kinds in pools.items():
        for kind, rows in kinds.items():
            rows = sorted(rows, key=lambda r: r.id)
            random.Random(f"{config.seed}-calib-{lang}-{kind.value}").shuffle(rows)
            out[lang][kind] = [r.prompt for r in rows[:n]]
    return out


def language_directions(config: Config, model_name: str) -> dict[str, object]:
    """{lang: tensor [layers, d]} = mean(harmful) - mean(harmless) for one model."""
    path = config.model_path(model_name)
    model, tokenizer = load_hf_model(path), load_tokenizer(path)
    model.eval()
    directions = {}
    for lang, kinds in calib_prompts(config).items():
        harmful = mean_last_token_states(model, tokenizer, kinds[Kind.HARMFUL])
        harmless = mean_last_token_states(model, tokenizer, kinds[Kind.HARMLESS])
        directions[lang] = harmful - harmless
    del model
    return directions


def run_directions(config: Config, models: list[str]) -> None:
    """Writes directions.pt, similarity.json (reference model) and layer_strength.json."""
    import torch

    reference, *others = models
    all_dirs = {name: language_directions(config, name) for name in models}
    torch.save(all_dirs, config.run_path("locate", "directions.pt"))

    ref = all_dirs[reference]
    langs = [lang for lang in config.languages if lang in ref]
    n_layers = ref[langs[0]].shape[0]
    write_json(
        config.run_path("locate", "similarity.json"),
        {
            "model": reference,
            "languages": langs,
            "by_layer": {
                str(i): cosine_matrix({k: ref[k] for k in langs}, i) for i in range(n_layers)
            },
        },
    )
    strength = {
        name: {
            lang: refusal_strength(all_dirs[name][lang], ref[lang])
            for lang in langs
            if lang in all_dirs[name]
        }
        for name in others
    }
    write_json(config.run_path("locate", "layer_strength.json"), strength)
    log.info("Directions for %s over %d layers", models, n_layers)
