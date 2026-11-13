"""Tables and charts for the write-up, built only from runs/metrics and runs/locate."""

from __future__ import annotations

import logging
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

from sdt.config import Config
from sdt.io import read_json, write_json

log = logging.getLogger(__name__)

# Reference categorical palette (fixed order, validated for colour-blind separation on bars).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
VARIANT_MODELS = ["global", "fire", "water", "earth"]
# Suffixes `sdt compress` gives its outputs (sdt.compress.quantize.compress), in chart order.
COMPRESSED_SUFFIXES = ["-4bit-uniform", "-4bit-mixed"]
# `sdt compress bf16`: the same weights on the 4-bit models' backend, the fair reference.
REFERENCE_SUFFIX = "-bf16-hf"


def load_metrics(config: Config, name: str) -> dict[str, Any] | None:
    path = config.runs_dir / "metrics" / f"{name}.json"
    return read_json(path) if path.exists() else None


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "font.size": 10,
            "legend.frameon": False,
        }
    )
    return plt


def _sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("seq", SEQUENTIAL)


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    log.info("Wrote %s", out)
    return out


def _asr_grid(config: Config, models: list[str]) -> tuple[list[str], list[list[float | None]]]:
    present, grid = [], []
    for name in models:
        m = load_metrics(config, name)
        if m is None:
            log.warning("No metrics for %s; skipped", name)
            continue
        present.append(name)
        grid.append([m["languages"][lang]["asr"] for lang in config.languages])
    return present, grid


def drift_heatmap(config: Config, models: list[str], out: Path) -> Path:
    """RQ 1a: harmful-compliance rate, language x model, values printed in each cell."""
    import numpy as np

    plt = _style()
    names, grid = _asr_grid(config, models)
    data = np.array([[np.nan if v is None else v for v in row] for row in grid])
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(config.languages), 0.9 + 0.55 * len(names)))
    ax.grid(False)
    image = ax.imshow(
        data, cmap=_sequential_cmap(), vmin=0, vmax=max(0.05, np.nanmax(data)), aspect="auto"
    )
    ax.set_xticks(
        range(len(config.languages)), [config.language(c)["name"] for c in config.languages]
    )
    ax.set_yticks(range(len(names)), names)
    threshold = np.nanmax(data) * 0.55
    for i in range(len(names)):
        for j in range(len(config.languages)):
            if not np.isnan(data[i, j]):
                colour = SURFACE if data[i, j] > threshold else INK
                ax.text(
                    j, i, f"{data[i, j]:.0%}", ha="center", va="center", fontsize=9, color=colour
                )
    fig.colorbar(image, ax=ax, label="Harmful compliance (ASR)", format=lambda x, _: f"{x:.0%}")
    ax.set_title("Harmful compliance by language", loc="left", fontsize=12)
    return _save(fig, out)


def datashare_chart(config: Config, out: Path) -> Path:
    """RQ 1b: ASR per language for Global and the three regional variants; favoured variant outlined."""
    import numpy as np

    plt = _style()
    names, grid = _asr_grid(config, VARIANT_MODELS)
    langs = config.languages
    x = np.arange(len(langs))
    width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(1.2 + 1.0 * len(langs), 3.6))
    for k, (name, row) in enumerate(zip(names, grid, strict=True)):
        values = [np.nan if v is None else v for v in row]
        favoured = [config.language(lang).get("variant") == name for lang in langs]
        bars = ax.bar(
            x + (k - (len(names) - 1) / 2) * width,
            values,
            width * 0.9,
            color=SERIES[k],
            label=name,
            edgecolor=SURFACE,
            linewidth=1,
        )
        for bar, fav in zip(bars, favoured, strict=True):
            if fav:  # secondary encoding, so the highlight never relies on colour alone
                bar.set_edgecolor(INK)
                bar.set_linewidth(1.6)
    ax.set_xticks(x, [config.language(c)["name"] for c in langs])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("Harmful compliance (ASR)")
    # Explicit handles: otherwise a series whose first bar is outlined shows an outline in the key.
    from matplotlib.patches import Patch

    handles = [Patch(color=SERIES[k], label=name) for k, name in enumerate(names)]
    ax.legend(handles=handles, ncols=len(names), loc="upper left", bbox_to_anchor=(0, 1.12))
    ax.set_title("Does more data in a language buy safety in it?", loc="left", fontsize=12, pad=28)
    ax.text(
        0,
        -0.2,
        "Outlined bar: the variant whose training mix favours that language.",
        transform=ax.transAxes,
        color=INK_2,
        fontsize=8.5,
    )
    return _save(fig, out)


def similarity_matrix(config: Config, out: Path) -> Path:
    """RQ 2: cosine similarity between per-language refusal directions at the chosen layer."""
    import numpy as np

    plt = _style()
    sim = read_json(config.runs_dir / "locate" / "similarity.json")
    layer = str(config["locate"]["direction_layer"])
    langs = sim["languages"]
    data = np.array(sim["by_layer"][layer])
    fig, ax = plt.subplots(figsize=(1.2 + 0.7 * len(langs), 0.8 + 0.7 * len(langs)))
    ax.grid(False)
    image = ax.imshow(data, cmap=_sequential_cmap(), vmin=0, vmax=1)
    names = [config.language(c)["name"] for c in langs]
    ax.set_xticks(range(len(langs)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(langs)), names)
    for i in range(len(langs)):
        for j in range(len(langs)):
            ax.text(
                j,
                i,
                f"{data[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=SURFACE if data[i, j] > 0.6 else INK,
            )
    fig.colorbar(image, ax=ax, label="Cosine similarity")
    ax.set_title(f"Refusal directions across languages (layer {layer})", loc="left", fontsize=12)
    return _save(fig, out)


def _short_name(name: str) -> str:
    """A repair's method and value, for charts of repairs that all share a source model."""
    return name.removeprefix("repair-harmful-").removeprefix("repair-")


def _sides(
    values: list[float],
    min_gap: float,
    xs: list[float] | None = None,
    x_decades: float = 0.3,
) -> list[str]:
    """Which side of its point each label goes: neighbours closer than `min_gap` take turns.

    With `xs` (a log-scale axis), points more than `x_decades` apart horizontally do not collide.
    """
    sides = ["right"] * len(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    for prev, i in pairwise(order):
        far = xs is not None and abs(math.log10(xs[i] / xs[prev])) > x_decades
        if values[i] - values[prev] < min_gap and sides[prev] == "right" and not far:
            sides[i] = "left"
    return sides


def tradeoff_chart(config: Config, models: list[str], out: Path) -> Path:
    """RQ 3: the two ways safety fails, per model: answers too much, or refuses everything."""
    plt = _style()
    rows = []
    for name in models:
        m = load_metrics(config, name)
        if m is None or m["mean_asr"] is None:
            log.warning("No metrics for %s; skipped", name)
            continue
        rows.append((name, m["mean_over_refusal"], m["mean_asr"]))
    highlight = config["report"]["tradeoff_highlight"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for (name, refused, asr), side in zip(
        rows, _sides([r[2] for r in rows], min_gap=0.025, xs=[r[1] for r in rows]), strict=True
    ):
        ax.scatter(
            refused,
            asr,
            s=90 if name == highlight else 56,
            color=SERIES[0] if name == highlight else INK_2,
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
        )
        ax.annotate(
            _short_name(name),
            (refused, asr),
            xytext=(7 if side == "right" else -7, 0),
            textcoords="offset points",
            ha="left" if side == "right" else "right",
            va="center",
            fontsize=8.5,
            fontweight="bold" if name == highlight else "normal",
        )
    ax.set_xscale("log")
    ax.set_xlim(0.007, 3)
    ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1], ["1%", "3%", "10%", "30%", "100%"])
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Over-refusal: safe prompts refused (log scale), lower is better")
    ax.set_ylabel("Mean harmful compliance, lower is better")
    ax.set_title("Repairs: answering too much against refusing too much", loc="left", fontsize=12)
    return _save(fig, out)


def via_english_chart(config: Config, name: str, out: Path) -> Path:
    """Judge reliability: one judge on the original reply and on its translation, per language."""
    plt = _style()
    report = read_json(config.runs_dir / "metrics" / f"{name}.via_english.json")
    controls = config["judge"]["via_english_controls"]
    change = {k: v["asr_via_english"] - v["asr_native"] for k, v in report.items()}
    langs = sorted((k for k in report if k not in controls), key=change.get)
    langs = [k for k in controls if k in report] + langs  # controls at the bottom
    secondary = config["judge"]["secondary"].replace("_", " ").title()
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.55 * len(langs)))
    ax.grid(axis="y", visible=False)
    for y, lang in enumerate(langs):
        v = report[lang]
        ax.annotate(
            "",
            (v["asr_via_english"], y),
            (v["asr_native"], y),
            arrowprops={"arrowstyle": "-|>", "color": INK_2, "linewidth": 1.4},
            zorder=2,
        )
        kw = {"edgecolor": SURFACE, "linewidth": 1.5, "zorder": 3}
        ax.scatter(v["asr_native"], y, s=64, color=SERIES[0], **kw)
        ax.scatter(v["asr_via_english"], y, s=64, color=SERIES[1], **kw)
        ax.scatter(v["asr_secondary"], y, s=46, color=INK, marker="D", **kw)
    labels = [config.language(k)["name"] + (" (control)" if k in controls else "") for k in langs]
    ax.set_yticks(range(len(langs)), labels)
    ax.set_ylim(-0.6, len(langs) - 0.4)
    ax.set_xlim(0, None)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel(f"Harmful compliance (ASR) of {name}")
    from matplotlib.lines import Line2D

    def key(colour: str, marker: str, label: str) -> Line2D:
        return Line2D([], [], color=colour, marker=marker, linestyle="", label=label)

    ax.legend(
        handles=[
            key(SERIES[0], "o", "Aya on the original reply"),
            key(SERIES[1], "o", "Aya on its English translation"),
            key(INK, "D", f"{secondary} on the original reply"),
        ],
        ncols=3,
        loc="lower left",
        bbox_to_anchor=(-0.02, 1.0),
        fontsize=8.5,
    )
    ax.set_title(
        f"Translating the reply changes the verdict most in {config.language(langs[-1])['name']}",
        loc="left",
        fontsize=12,
        pad=26,
    )
    return _save(fig, out)


def _compression_label(name: str) -> str:
    """The precision alone: every bar on the chart is the same source model."""
    if name.endswith(REFERENCE_SUFFIX):
        return "bfloat16"
    for suffix in COMPRESSED_SUFFIXES:
        if name.endswith(suffix):
            return suffix.removeprefix("-").replace("bit-", "-bit ")
    return name


def compression_chart(config: Config, models: list[str], out: Path) -> Path:
    """RQ 4: ASR per language before and after 4-bit, uniform vs mixed precision."""
    import numpy as np

    plt = _style()
    names, grid = _asr_grid(config, models)
    if len(names) > len(SERIES):
        raise ValueError(
            f"compression chart takes at most {len(SERIES)} models, got {len(names)}: pass --models"
        )
    langs = config.languages
    x = np.arange(len(langs))
    width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(1.2 + 1.0 * len(langs), 3.6))
    for k, (name, row) in enumerate(zip(names, grid, strict=True)):
        size = (load_metrics(config, name) or {}).get("size_gb")
        short = _compression_label(name)
        label = f"{short} ({size:.1f} GB)" if size else short
        ax.bar(
            x + (k - (len(names) - 1) / 2) * width,
            [np.nan if v is None else v for v in row],
            width * 0.9,
            color=SERIES[k],
            label=label,
            edgecolor=SURFACE,
            linewidth=1,
        )
    ax.set_xticks(x, [config.language(c)["name"] for c in langs])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("Harmful compliance (ASR)")
    ax.legend(ncols=len(names), loc="upper left", bbox_to_anchor=(0, 1.12))
    source = next(
        (n.removesuffix(REFERENCE_SUFFIX) for n in names if n.endswith(REFERENCE_SUFFIX)), None
    )
    subject = _short_name(source) if source else "the repair"
    ax.set_title(f"Does {subject} survive 4-bit?", loc="left", fontsize=12, pad=28)
    return _save(fig, out)


def drop_layer(values: list[list[float]], threshold: float = 0.9) -> int | None:
    """First layer from which the mean strength over languages stays below `threshold`.

    None when the last layer is at or above it, i.e. there is no lasting drop to report.
    """
    means = [sum(col) / len(col) for col in zip(*values, strict=True)]
    above = [i for i, mean in enumerate(means) if mean >= threshold]
    if not above:
        return 0
    return None if above[-1] == len(means) - 1 else above[-1] + 1


def layer_drift_chart(config: Config, model: str, out: Path) -> Path:
    """RQ 2: refusal strength relative to Global per layer, one line per language."""
    plt = _style()
    strength = read_json(config.runs_dir / "locate" / "layer_strength.json")[model]
    langs = [lang for lang in config.languages if lang in strength]
    n_layers = len(strength[langs[0]])
    steer_layer = config["locate"]["direction_layer"]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for k, lang in enumerate(langs):
        ax.plot(
            range(n_layers),
            strength[lang],
            color=SERIES[k % len(SERIES)],
            linewidth=1.8,
            label=config.language(lang)["name"],
        )
    ax.axhline(1.0, color=INK_2, linestyle="--", linewidth=1.2, zorder=1)
    ax.annotate(  # in the right margin, clear of the lines
        "Global",
        (1, 1.0),
        xycoords=("axes fraction", "data"),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        color=INK_2,
        fontsize=8.5,
        annotation_clip=False,
    )
    ax.axvline(steer_layer, color=INK_2, linestyle=":", linewidth=1.2, zorder=1)
    ax.annotate(
        f"layer {steer_layer}: steering and attribution",
        (steer_layer, 0),
        xycoords=("data", "axes fraction"),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        color=INK_2,
        fontsize=8.5,
    )
    ax.set_xlim(0, n_layers - 1)
    # Fixed floor at 0 (no signal) and room above 1.0, so small wiggles don't read as drift
    # and charts of different models compare at a glance.
    values = [v for lang in langs for v in strength[lang]]
    ax.set_ylim(min(0.0, min(values)) - 0.02, max(1.2, max(values) + 0.05))
    ax.set_xlabel("Layer")
    ax.set_ylabel("Refusal strength (Global = 1.0)")
    ax.legend(ncols=4, loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=9)

    drop = drop_layer([strength[lang] for lang in langs])
    title = (
        f"{model.capitalize()} model loses refusal signal from layer {drop} onward"
        if drop is not None
        else f"{model.capitalize()} model: refusal strength by layer, relative to Global"
    )
    ax.set_title(title, loc="left", fontsize=12, pad=48)
    return _save(fig, out)


def compression_groups(config: Config) -> dict[str, list[str]]:
    """Source model -> [reference, source-4bit-uniform, source-4bit-mixed], only those with metrics.

    A source appears only if at least one compressed version of it has metrics. The reference is
    `source-bf16-hf` when it exists, so the chart does not also compare vLLM with transformers.
    """
    present = set(available_models(config))
    groups: dict[str, list[str]] = {}
    for name in sorted(present):
        for suffix in COMPRESSED_SUFFIXES:
            if name.endswith(suffix):
                groups.setdefault(name.removesuffix(suffix), [])
    for source, models in groups.items():
        reference = source + REFERENCE_SUFFIX if source + REFERENCE_SUFFIX in present else source
        candidates = [reference, *(source + s for s in COMPRESSED_SUFFIXES)]
        models.extend(name for name in candidates if name in present)
    return groups


def mean_asr_ci(metrics: dict[str, Any]) -> tuple[float, float] | None:
    """95% normal interval of the mean over languages, each language an independent binomial."""
    rows = [m for m in metrics["languages"].values() if m["asr"] is not None and m["n_harmful"]]
    if not rows:
        return None
    mean = sum(m["asr"] for m in rows) / len(rows)
    variance = sum(m["asr"] * (1 - m["asr"]) / m["n_harmful"] for m in rows) / len(rows) ** 2
    half = 1.96 * variance**0.5
    return max(mean - half, 0.0), min(mean + half, 1.0)


def paired_tests(config: Config, pairs: list[list[str]]) -> list[dict[str, Any]]:
    """Exact McNemar test of each [before, after] pair on harmful prompts, with both judges."""
    from sdt.evaluate.pipeline import judgements_path, load_judgements
    from sdt.judge.agreement import binary
    from sdt.judge.via_english import exact_mcnemar
    from sdt.schema import Kind, Label

    def labels(model: str, judge: str) -> dict[str, str]:
        rows = load_judgements(judgements_path(config, model, judge))
        keep = (r for r in rows if r.kind == Kind.HARMFUL and r.label != Label.INVALID)
        return {r.id: binary(r.label) for r in keep}

    results = []
    for before, after in pairs:
        for judge in (config["judge"]["primary"], config["judge"]["secondary"]):
            a, b = labels(before, judge), labels(after, judge)
            ids = sorted(a.keys() & b.keys())
            fell = sum(a[i] == "harmful" != b[i] for i in ids)
            rose = sum(b[i] == "harmful" != a[i] for i in ids)
            results.append(
                {
                    "before": before,
                    "after": after,
                    "judge": judge,
                    "n": len(ids),
                    "stopped_being_harmful": fell,
                    "became_harmful": rose,
                    "p": exact_mcnemar(fell, rose),
                }
            )
    write_json(config.run_path("metrics", "paired_tests.json"), results)
    return results


def paired_tests_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Before | After | Judge | n | Stopped being harmful | Became harmful | McNemar p |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r['before']} | {r['after']} | {r['judge']} | {r['n']} "
            f"| {r['stopped_being_harmful']} | {r['became_harmful']} | {r['p']:.3g} |"
        )
    return "\n".join(lines) + "\n"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def metrics_table(config: Config, models: list[str]) -> str:
    """Markdown before/after table: the seven reported metrics, one row per model."""
    header = (
        "| Model | Mean ASR | 95% CI | Over-refusal | Gap (all sources) | Gap (parallel XSTest) | Task F1 "
        "| Capability | Weights changed | GPU min | Judge invalid |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    lines = [header]
    for name in models:
        m = load_metrics(config, name)
        if m is None:
            continue
        task = [v["task_score"] for v in m["languages"].values() if v["task_score"] is not None]
        cap = [v["capability"] for v in m["languages"].values() if v["capability"] is not None]
        mean_task = f"{sum(task) / len(task):.3f}" if task else "—"
        mean_cap = _pct(sum(cap) / len(cap)) if cap else "—"
        weights = _pct(m["weights_changed"]) if m["weights_changed"] is not None else "—"
        minutes = f"{m['repair_gpu_minutes']:.1f}" if m["repair_gpu_minutes"] is not None else "—"
        ci = mean_asr_ci(m)
        interval = f"{ci[0]:.1%} to {ci[1]:.1%}" if ci else "—"
        lines.append(
            f"| {name} | {_pct(m['mean_asr'])} | {interval} | {_pct(m['mean_over_refusal'])} "
            f"| {_pct(m['safety_gap'])} "
            f"| {_pct(m.get('parallel_safety_gap'))} "
            f"| {mean_task} | {mean_cap} | {weights} | {minutes} | {_pct(m.get('invalid_rate'))} |"
        )
    return "\n".join(lines) + "\n"


def per_language_table(config: Config, models: list[str]) -> str:
    """Markdown table of ASR per language (rows) and model (columns)."""
    present = [(n, m) for n in models if (m := load_metrics(config, n))]
    lines = [
        "| Language | " + " | ".join(n for n, _ in present) + " |",
        "| --- |" + " --- |" * len(present),
    ]
    for lang in config.languages:
        cells = [_pct(m["languages"][lang]["asr"]) for _, m in present]
        lines.append(f"| {config.language(lang)['name']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def available_models(config: Config) -> list[str]:
    """Models with primary-judge metrics (names may contain dots, e.g. repair-oneshot-0.005)."""
    folder = config.runs_dir / "metrics"
    if not folder.exists():
        return []
    not_models = (
        f".{config['judge']['secondary']}",
        ".via_english",
        ".same_content",
        ".prompt_check",
    )
    return sorted(
        p.stem
        for p in folder.glob("*.json")
        if not p.stem.endswith(not_models)
        and p.stem not in ("judge_agreement", "paired_tests", "translation_check")
    )
