"""Tables and charts for the write-up, built only from runs/metrics and runs/locate."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sdt.config import Config
from sdt.io import read_json

log = logging.getLogger(__name__)

# Reference categorical palette (fixed order, validated for colour-blind separation on bars).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
VARIANT_MODELS = ["global", "fire", "water", "earth"]
# Suffixes `sdt compress` gives its outputs (sdt.compress.quantize.compress), in chart order.
COMPRESSED_SUFFIXES = ["-4bit-uniform", "-4bit-mixed"]


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


def tradeoff_chart(config: Config, models: list[str], out: Path) -> Path:
    """RQ 3: mean ASR against mean task score, one labelled point per model or repair."""
    plt = _style()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k, name in enumerate(models):
        m = load_metrics(config, name)
        if m is None or m["mean_asr"] is None:
            continue
        tasks = [v["task_score"] for v in m["languages"].values() if v["task_score"] is not None]
        if not tasks:
            continue
        task = sum(tasks) / len(tasks)
        colour = SERIES[k % len(SERIES)] if k < len(SERIES) else INK_2
        ax.scatter(
            task, m["mean_asr"], s=64, color=colour, edgecolor=SURFACE, linewidth=2, zorder=3
        )
        ax.annotate(
            name, (task, m["mean_asr"]), xytext=(6, 4), textcoords="offset points", fontsize=8.5
        )
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Task score (F1), higher is better")
    ax.set_ylabel("Mean harmful compliance, lower is better")
    ax.set_title("Repairs: safety against task skill", loc="left", fontsize=12)
    return _save(fig, out)


def compression_chart(config: Config, models: list[str], out: Path) -> Path:
    """RQ 4: ASR per language before and after 4-bit, uniform vs mixed precision."""
    import numpy as np

    plt = _style()
    names, grid = _asr_grid(config, models)
    langs = config.languages
    x = np.arange(len(langs))
    width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(1.2 + 1.0 * len(langs), 3.6))
    for k, (name, row) in enumerate(zip(names, grid, strict=True)):
        size = (load_metrics(config, name) or {}).get("size_gb")
        label = f"{name} ({size:.1f} GB)" if size else name
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
    ax.set_title("Does the repair survive 4-bit?", loc="left", fontsize=12, pad=28)
    return _save(fig, out)


def compression_groups(config: Config) -> dict[str, list[str]]:
    """Source model -> [source, source-4bit-uniform, source-4bit-mixed], only those with metrics.

    A source appears only if at least one compressed version of it has metrics.
    """
    present = set(available_models(config))
    groups: dict[str, list[str]] = {}
    for name in sorted(present):
        for suffix in COMPRESSED_SUFFIXES:
            if name.endswith(suffix):
                groups.setdefault(name.removesuffix(suffix), [])
    for source, models in groups.items():
        candidates = [source, *(source + s for s in COMPRESSED_SUFFIXES)]
        models.extend(name for name in candidates if name in present)
    return groups


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def metrics_table(config: Config, models: list[str]) -> str:
    """Markdown before/after table: the seven reported metrics, one row per model."""
    header = (
        "| Model | Mean ASR | Over-refusal | Gap (all sources) | Gap (parallel XSTest) | Task F1 "
        "| Capability | Weights changed | GPU min | Judge invalid |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
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
        lines.append(
            f"| {name} | {_pct(m['mean_asr'])} | {_pct(m['mean_over_refusal'])} | {_pct(m['safety_gap'])} "
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
    secondary = f".{config['judge']['secondary']}"
    return sorted(
        p.stem
        for p in folder.glob("*.json")
        if not p.stem.endswith(secondary) and p.stem != "judge_agreement"
    )
