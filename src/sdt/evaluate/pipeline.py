"""`sdt evaluate <model>`: generate -> judge (both judges) -> metrics -> capability -> task score."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

from sdt.config import Config, save_run_config
from sdt.data.build import load_prompts
from sdt.generation import GenerationStats, generate_for_model
from sdt.io import read_json, read_jsonl, write_json, write_jsonl
from sdt.judge import run_judge
from sdt.metrics import model_metrics
from sdt.schema import GenerationRecord, JudgeRecord, Kind, Label, ModelMetrics, Split

log = logging.getLogger(__name__)

INVALID_WARN = 0.05


def generations_path(config: Config, name: str, split: Split = Split.TEST) -> Path:
    suffix = "" if split == Split.TEST else f".{split.value}"
    return config.run_path("generations", f"{name}{suffix}.jsonl")


def judgements_path(config: Config, name: str, judge: str, split: Split = Split.TEST) -> Path:
    suffix = "" if split == Split.TEST else f".{split.value}"
    return config.run_path("judgements", f"{name}{suffix}.{judge}.jsonl")


def metrics_path(config: Config, name: str, judge: str) -> Path:
    primary = config["judge"]["primary"]
    return config.run_path(
        "metrics", f"{name}.json" if judge == primary else f"{name}.{judge}.json"
    )


def load_generations(path: Path) -> list[GenerationRecord]:
    return [GenerationRecord(**{**r, "kind": Kind(r["kind"])}) for r in read_jsonl(path)]


def load_judgements(path: Path) -> list[JudgeRecord]:
    return [
        JudgeRecord(**{**r, "kind": Kind(r["kind"]), "label": Label(r["label"])})
        for r in read_jsonl(path)
    ]


def generate(
    config: Config, name: str, split: Split = Split.TEST, backend: str | None = None
) -> GenerationStats:
    """Replies to every harmful and XSTest-safe prompt of one split."""
    records = [
        p
        for p in load_prompts(config)
        if p.split == split and p.kind in (Kind.HARMFUL, Kind.XSTEST_SAFE)
    ]
    rows, stats = generate_for_model(config, name, records, backend)
    path = generations_path(config, name, split)
    write_jsonl(path, rows)
    stats_info = asdict(stats) | {
        "tokens_per_second": stats.tokens_per_second,
        "backend": backend or config["generation"]["backend"],
    }
    write_json(path.with_suffix(".stats.json"), stats_info)
    return stats


def judge(
    config: Config, name: str, judge_name: str, split: Split = Split.TEST
) -> list[JudgeRecord]:
    rows = load_generations(generations_path(config, name, split))
    labels = run_judge(config, judge_name, rows)
    write_jsonl(judgements_path(config, name, judge_name, split), labels)
    return labels


def score(config: Config, name: str, judge_name: str, extras: dict | None = None) -> ModelMetrics:
    """Safety metrics from saved labels, plus capability/task/size/repair info when available."""
    labels = load_judgements(judgements_path(config, name, judge_name))
    metrics = model_metrics(name, judge_name, labels, config.languages)
    if extras is None:  # `sdt metrics` after a split run: reuse what `evaluate` measured
        saved = config.runs_dir / "metrics" / name / "extras.json"
        extras = read_json(saved) if saved.exists() else {}
    for lang, value in extras.get("capability", {}).items():
        metrics.languages[lang].capability = value
    for lang, value in extras.get("task_score", {}).items():
        metrics.languages[lang].task_score = value["f1"] if isinstance(value, dict) else value
    if metrics.invalid_rate and metrics.invalid_rate > INVALID_WARN:
        log.warning(
            "%s/%s: judge could not label %.1f%% of replies; they are excluded from every rate. "
            "Check runs/judgements for the raw outputs before reporting.",
            name,
            judge_name,
            100 * metrics.invalid_rate,
        )

    stats_file = generations_path(config, name).with_suffix(".stats.json")
    if stats_file.exists():
        metrics.tokens_per_second = read_json(stats_file)["tokens_per_second"]
    repair_file = config.runs_dir / "repair" / f"{name}.json"
    if repair_file.exists():
        info = read_json(repair_file)
        metrics.weights_changed = info.get("weights_changed")
        metrics.repair_gpu_minutes = info.get("gpu_minutes")
    size_file = config.runs_dir / "models" / name / "size.json"
    if size_file.exists():
        metrics.size_gb = read_json(size_file)["size_gb"]

    write_json(metrics_path(config, name, judge_name), asdict(metrics))
    return metrics


def evaluate(
    config: Config,
    name: str,
    backend: str | None = None,
    with_capability: bool = True,
    with_task: bool = True,
    judges: list[str] | None = None,
) -> ModelMetrics:
    """The full evaluation of one model; returns the first judge's metrics.

    `judges` defaults to both. A GPU too small for the primary judge runs the secondary only;
    `sdt judge <model> --judge <primary>` and `sdt metrics <model>` finish it on a larger GPU.
    """
    run_dir = config.runs_dir / "metrics" / name
    save_run_config(config, run_dir, step="evaluate", model=name, backend=backend, judges=judges)
    generate(config, name, backend=backend)
    judges = judges or [config["judge"]["primary"], config["judge"]["secondary"]]
    for judge_name in judges:
        judge(config, name, judge_name)

    extras: dict = {}
    if with_capability:
        from sdt.evaluate.capability import capability_by_language

        extras["capability"] = capability_by_language(config, name)
    if with_task:
        from sdt.evaluate.task_score import task_score_by_language

        extras["task_score"] = task_score_by_language(config, name)
    write_json(run_dir / "extras.json", extras)

    result, *_ = [score(config, name, judge_name, extras) for judge_name in judges]
    log.info("%s: mean ASR %s, gap %s", name, result.mean_asr, result.safety_gap)
    return result
