"""Ranks attention heads and MLPs by how much restoring them (Global -> drifted) restores refusal.

Attribution patching (Nanda 2023): effect of patching component c ≈ (a_c^global - a_c^drifted) · ∂m/∂a_c,
with the gradient taken in the drifted model. The metric m is the drifted model's last-token residual
at the direction layer, projected on Global's pooled refusal direction. Only the last prompt token is
patched, which is where the refusal decision is read out. The top components are then confirmed with
real activation patching.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

from sdt.config import Config
from sdt.io import write_json
from sdt.locate.directions import calib_prompts
from sdt.models import chat_prompt, decoder_layers, load_hf_model, load_tokenizer
from sdt.repair.common import components_path
from sdt.schema import Kind

log = logging.getLogger(__name__)


@dataclass
class Component:
    layer: int
    kind: str  # "head" or "mlp"
    index: int  # head index; 0 for mlp
    attribution: float
    patch_effect: float | None = None

    @property
    def key(self) -> str:
        return f"L{self.layer}.{self.kind}{self.index if self.kind == 'head' else ''}"


def head_dim(model) -> int:
    cfg = model.config
    return getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads


class _Recorder:
    """Hooks that record (and optionally overwrite) last-token o_proj inputs and MLP outputs."""

    def __init__(self, model, max_layer: int, keep_grad: bool = False):
        self.layers = decoder_layers(model)[:max_layer]
        self.keep_grad = keep_grad
        self.attn: dict[int, object] = {}
        self.mlp: dict[int, object] = {}
        self.patch: tuple[str, int, int, object] | None = None  # (kind, layer, index, value [B, w])
        self.hd = head_dim(model)
        self.handles = []

    def __enter__(self):
        for i, layer in enumerate(self.layers):
            self.handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(self._attn_hook(i))
            )
            self.handles.append(layer.mlp.register_forward_hook(self._mlp_hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()

    def _store(self, store, i, tensor):
        if self.keep_grad:
            tensor.retain_grad()
        store[i] = tensor

    def _attn_hook(self, i):
        def fn(_module, inputs):
            x = inputs[0]
            if self.patch and self.patch[0] == "head" and self.patch[1] == i:
                _, _, h, value = self.patch
                x = x.clone()
                x[:, -1, h * self.hd : (h + 1) * self.hd] = value.to(x.dtype)
                self._store(self.attn, i, x)
                return (x,)
            self._store(self.attn, i, x)
            return None

        return fn

    def _mlp_hook(self, i):
        def fn(_module, _inputs, output):
            if self.patch and self.patch[0] == "mlp" and self.patch[1] == i:
                output = output.clone()
                output[:, -1, :] = self.patch[3].to(output.dtype)
                self._store(self.mlp, i, output)
                return output
            self._store(self.mlp, i, output)
            return None

        return fn


def _metric(model, inputs, layer: int, direction, embeds=None):
    """Mean projection of the last-token residual after `layer` on the unit refusal direction."""
    captured = {}

    def grab(_m, _i, output):
        captured["h"] = (output[0] if isinstance(output, tuple) else output)[:, -1, :]

    handle = decoder_layers(model)[layer].register_forward_hook(grab)
    try:
        if embeds is not None:
            model(inputs_embeds=embeds, attention_mask=inputs["attention_mask"], logits_to_keep=1)
        else:
            model(**inputs, logits_to_keep=1)  # the hook reads the residual, not logits
    finally:
        handle.remove()
    return (captured["h"].float() @ direction.to(captured["h"].device)).mean()


def attribution_scores(
    attn_global: dict, mlp_global: dict, attn_drift: dict, mlp_drift: dict, hd: int, n_heads: int
) -> dict[tuple[str, int, int], float]:
    """Sum over the batch of (a_global - a_drift) · grad, per head and per MLP, at the last token."""
    scores: dict[tuple[str, int, int], float] = {}
    for layer, a_d in attn_drift.items():
        delta = attn_global[layer] - a_d[:, -1].detach().float()
        grad = a_d.grad[:, -1].float()
        for h in range(n_heads):
            sl = slice(h * hd, (h + 1) * hd)
            scores[("head", layer, h)] = float((delta[:, sl] * grad[:, sl]).sum())
    for layer, m_d in mlp_drift.items():
        delta = mlp_global[layer] - m_d[:, -1].detach().float()
        scores[("mlp", layer, 0)] = float((delta * m_d.grad[:, -1].float()).sum())
    return scores


def rank_components(
    config: Config, reference: str = "global", drifted: str = "drifted"
) -> list[Component]:
    """Attribution patching over all heads and MLPs below the direction layer, then activation patching."""
    import torch

    spec = config["locate"]
    layer_star = spec["direction_layer"]
    directions = torch.load(config.runs_dir / "locate" / "directions.pt")[reference]
    pooled = torch.stack([d[layer_star] for d in directions.values()]).mean(dim=0)
    unit = pooled / pooled.norm()

    texts = [t for kinds in calib_prompts(config).values() for t in kinds[Kind.HARMFUL]]
    tok = load_tokenizer(config.model_path(drifted))
    ref_model = load_hf_model(config.model_path(reference))
    model = load_hf_model(config.model_path(drifted))
    for m in (ref_model, model):
        m.eval()
        m.requires_grad_(False)
    hd, n_heads = head_dim(model), model.config.num_attention_heads

    totals: dict[tuple[str, int, int], float] = {}
    batches = []
    size = spec["batch_size"]
    for i in range(0, len(texts), size):
        batch = [chat_prompt(tok, t) for t in texts[i : i + size]]
        inputs = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(
            model.device
        )
        with torch.no_grad(), _Recorder(ref_model, layer_star + 1) as rec_g:
            ref_model(**inputs, logits_to_keep=1)
        attn_g = {k: v[:, -1].float() for k, v in rec_g.attn.items()}
        mlp_g = {k: v[:, -1].float() for k, v in rec_g.mlp.items()}
        batches.append((inputs, attn_g, mlp_g))

        embeds = model.get_input_embeddings()(inputs["input_ids"]).detach().requires_grad_(True)
        with _Recorder(model, layer_star + 1, keep_grad=True) as rec_d:
            metric = _metric(model, inputs, layer_star, unit, embeds=embeds)
            metric.backward()
        for key, value in attribution_scores(
            attn_g, mlp_g, rec_d.attn, rec_d.mlp, hd, n_heads
        ).items():
            totals[key] = totals.get(key, 0.0) + value
        model.zero_grad(set_to_none=True)

    ranked = sorted(
        (Component(layer, kind, idx, score) for (kind, layer, idx), score in totals.items()),
        key=lambda c: c.attribution,
        reverse=True,
    )
    _confirm_with_patching(model, ranked[: spec["activation_patch_top"]], batches, layer_star, unit)
    top = ranked[: max(spec["top_components"], spec["activation_patch_top"])]
    write_json(components_path(config, drifted), [asdict(c) | {"key": c.key} for c in top])
    log.info("Top components: %s", [c.key for c in top[:10]])
    return top


def _confirm_with_patching(
    model, components: list[Component], batches, layer_star: int, unit
) -> None:
    """Real activation patching: metric change when one component gets Global's last-token value."""
    import torch

    with torch.no_grad():
        base = [_metric(model, inputs, layer_star, unit) for inputs, _, _ in batches]
        for comp in components:
            effects = []
            for (inputs, attn_g, mlp_g), before in zip(batches, base, strict=True):
                if comp.kind == "head":
                    hd = head_dim(model)
                    value = attn_g[comp.layer][:, comp.index * hd : (comp.index + 1) * hd]
                else:
                    value = mlp_g[comp.layer]
                with _Recorder(model, layer_star + 1) as rec:
                    rec.patch = (comp.kind, comp.layer, comp.index, value)
                    after = _metric(model, inputs, layer_star, unit)
                effects.append(float(after - before))
            comp.patch_effect = sum(effects) / len(effects)
