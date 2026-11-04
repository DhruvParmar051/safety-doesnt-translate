"""`sdt` command line: one subcommand per pipeline step (see IMPLEMENTATION_PLAN.md)."""

from __future__ import annotations

import argparse
import logging
import sys

from sdt.config import load_config
from sdt.schema import Split


def _data(cfg, args) -> None:
    from sdt.data import build

    if args.step == "sources":
        build.fetch_sources(cfg)
    elif args.step == "translate":
        build.translate(cfg)
    elif args.step == "build":
        build.build(cfg)
    elif args.step == "task":
        from sdt.data.task_data import build_task_data

        print("task rows (train, eval):", build_task_data(cfg))
    elif args.step == "harmful":
        from sdt.data.harmful_data import build_harmful_data

        print("harmful rows:", build_harmful_data(cfg))


def _finetune(cfg, args) -> None:
    from sdt.finetune import train_drifted

    print("Saved", train_drifted(cfg, args.variant))


def _generate(cfg, args) -> None:
    from sdt.evaluate.pipeline import generate

    stats = generate(cfg, args.model, Split(args.split), args.backend)
    print(f"{stats.n_prompts} replies, {stats.tokens_per_second:.0f} tokens/s")


def _judge(cfg, args) -> None:
    from sdt.evaluate.pipeline import judge

    for name in args.judge or [cfg["judge"]["primary"], cfg["judge"]["secondary"]]:
        judge(cfg, args.model, name, Split(args.split))


def _metrics(cfg, args) -> None:
    from sdt.evaluate.pipeline import score

    for name in [cfg["judge"]["primary"], cfg["judge"]["secondary"]]:
        m = score(cfg, args.model, name)
        print(
            f"{name}: mean ASR {m.mean_asr}, gap {m.safety_gap}, over-refusal {m.mean_over_refusal}"
        )


def _evaluate(cfg, args) -> None:
    from sdt.evaluate.pipeline import evaluate

    evaluate(cfg, args.model, args.backend, not args.no_capability, not args.no_task, args.judge)


def _hand_labels(cfg, args) -> None:
    from sdt.data.build import load_prompts
    from sdt.evaluate.pipeline import (
        generations_path,
        judgements_path,
        load_generations,
        load_judgements,
    )
    from sdt.judge.agreement import hand_agreement, hand_label_sheet, judge_agreement

    sheet = cfg.data_dir / "hand_labels" / "sheet.csv"
    primary, secondary = cfg["judge"]["primary"], cfg["judge"]["secondary"]
    if args.action == "sheet":
        labels, replies = [], {}
        for model in args.models:
            labels += load_judgements(judgements_path(cfg, model, primary))
            replies |= {
                (g.model, g.id): g.response for g in load_generations(generations_path(cfg, model))
            }
        prompts = {p.id: p.prompt for p in load_prompts(cfg)}
        n = hand_label_sheet(
            labels, replies, prompts, cfg["judge"]["hand_label_per_language"], cfg.seed, sheet
        )
        print(f"Wrote {n} rows to {sheet}; fill in human_label (refusal / harmful / safe)")
        return
    from sdt.io import write_json

    report = {}
    for model in args.models:
        first = load_judgements(judgements_path(cfg, model, primary))
        second = load_judgements(judgements_path(cfg, model, secondary))
        report[model] = {
            "judges": judge_agreement(first, second),
            "hand": hand_agreement(first, sheet),
        }
    write_json(cfg.run_path("metrics", "judge_agreement.json"), report)
    print(report)


def _locate(cfg, args) -> None:
    if args.step == "directions":
        from sdt.locate.directions import run_directions

        run_directions(cfg, args.models)
    elif args.step == "components":
        from sdt.locate.components import rank_components

        reference, drifted, *_ = args.models
        rank_components(cfg, reference, drifted)
    elif args.step == "circuitkit":
        from sdt.locate.circuitkit_adapter import discover_refusal_circuit

        discover_refusal_circuit(cfg)


def _repair(cfg, args) -> None:
    r = cfg.repair
    names: list[str] = []
    if args.method == "task-arithmetic":
        from sdt.repair import task_arithmetic

        for scale in [args.value] if args.value is not None else r["task_arithmetic"]["scales"]:
            names.append(task_arithmetic.run(cfg, scale, args.source))
    elif args.method == "oneshot":
        from sdt.repair import oneshot

        for fraction in [args.value] if args.value is not None else r["oneshot"]["fractions"]:
            names.append(oneshot.run(cfg, fraction, args.source))
    elif args.method == "circuit-patch":
        from sdt.repair import circuit_patch

        for k in [int(args.value)] if args.value is not None else r["circuit_patch"]["top_k"]:
            names.append(circuit_patch.run(cfg, k, args.source))
    elif args.method == "resta":
        from sdt.repair.safetune_methods import run_resta

        alpha = args.value if args.value is not None else r["resta"]["alpha"]
        names.append(run_resta(cfg, alpha, args.source))
    elif args.method == "safetune":
        from sdt.repair.safetune_methods import run_cli

        for algo in [args.algo] if args.algo else r["safetune_cli"]:
            names.append(run_cli(cfg, algo, args.source))
    elif args.method == "steer":
        from sdt.repair.steering import create_steered_model

        alpha = args.value if args.value is not None else r["steer"]["alpha"]
        names.append(create_steered_model(cfg, alpha, r["steer"]["layer"], args.source))
    elif args.method == "safety-ft":
        from sdt.repair import safety_finetune

        names.append(safety_finetune.run(cfg, args.source))
    print("Created:", ", ".join(names), "\nNext: sdt evaluate <name>")


def _compress(cfg, args) -> None:
    from sdt.compress.quantize import compress, measure_size

    measure_size(cfg, args.model)
    print("Created:", compress(cfg, args.model, args.mode))


def _report(cfg, args) -> None:
    from sdt.evaluate import report

    out = cfg.results_dir
    models = args.models or report.available_models(cfg)
    if args.what in ("drift", "all"):
        report.drift_heatmap(
            cfg, models if args.models else ["global", "drifted"], out / "drift_heatmap.png"
        )
    if args.what in ("datashare", "all"):
        report.datashare_chart(cfg, out / "datashare.png")
    if (
        args.what in ("similarity", "all")
        and (cfg.runs_dir / "locate" / "similarity.json").exists()
    ):
        report.similarity_matrix(cfg, out / "similarity.png")
    if args.what in ("tradeoff", "all"):
        report.tradeoff_chart(cfg, models, out / "tradeoff.png")
    if args.what == "compression":
        report.compression_chart(cfg, models, out / "compression.png")
    if args.what in ("table", "all"):
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics_table.md").write_text(report.metrics_table(cfg, models))
        (out / "asr_by_language.md").write_text(report.per_language_table(cfg, models))
        print(report.metrics_table(cfg, models))


def _publish(cfg, args) -> None:
    from sdt import publish

    private = not args.public
    if args.what == "dataset":
        publish.push_dataset(cfg, args.repo, private)
    elif args.what == "model":
        publish.push_model(cfg, args.model, args.repo, private)
    elif args.what == "logs":
        publish.push_logs(cfg, args.repo, private)


def _doctor(cfg, args) -> None:
    from sdt.doctor import run_doctor

    sys.exit(0 if run_doctor(cfg) else 1)


def _smoke(cfg, args) -> None:
    from sdt.locate.circuitkit_adapter import parity_check

    parity_check(cfg, args.model)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sdt", description=__doc__)
    p.add_argument("--config", default=None, help="path to project.yaml")
    p.add_argument(
        "--profile", default=None, help="config overlay, e.g. 'mac' (or set SDT_PROFILE)"
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("doctor", help="check device, packages and model access on this machine")
    s.set_defaults(fn=_doctor)

    s = sub.add_parser("smoke", help="CircuitKIT parity check on a model")
    s.add_argument("--model", default="global")
    s.set_defaults(fn=_smoke)

    s = sub.add_parser("data", help="phase 1")
    s.add_argument("step", choices=["sources", "translate", "build", "task", "harmful"])
    s.set_defaults(fn=_data)

    s = sub.add_parser("finetune", help="LoRA on the harmless task -> runs/models/drifted")
    s.add_argument(
        "--variant",
        choices=["harmless", "harmful"],
        default="harmless",
        help="harmful: + a few harmful English examples -> runs/models/drifted-harmful",
    )
    s.set_defaults(fn=_finetune)

    for cmd, fn in (("generate", _generate), ("judge", _judge)):
        s = sub.add_parser(cmd)
        s.add_argument("model")
        s.add_argument("--split", default="test", choices=["test", "calib"])
        if cmd == "generate":
            s.add_argument("--backend", choices=["vllm", "mlx", "hf"])
        else:
            s.add_argument("--judge", action="append", choices=["aya", "llama_guard"])
        s.set_defaults(fn=fn)

    s = sub.add_parser("metrics")
    s.add_argument("model")
    s.set_defaults(fn=_metrics)

    s = sub.add_parser("evaluate", help="generate + judge + metrics + capability + task score")
    s.add_argument("model")
    s.add_argument("--backend", choices=["vllm", "mlx", "hf"])
    s.add_argument("--no-capability", action="store_true")
    s.add_argument("--no-task", action="store_true")
    s.add_argument("--judge", action="append", choices=["aya", "llama_guard"], help="default: both")
    s.set_defaults(fn=_evaluate)

    s = sub.add_parser("hand-labels", help="hand-label sheet and judge agreement")
    s.add_argument("action", choices=["sheet", "agreement"])
    s.add_argument("--models", nargs="+", default=["global", "drifted"])
    s.set_defaults(fn=_hand_labels)

    s = sub.add_parser("locate", help="phase 3")
    s.add_argument("step", choices=["directions", "components", "circuitkit"])
    s.add_argument(
        "--models",
        nargs="+",
        default=["global", "drifted", "fire", "water", "earth"],
        help="first model is the reference; `components` compares it with the second",
    )
    s.set_defaults(fn=_locate)

    s = sub.add_parser("repair", help="phase 4")
    s.add_argument(
        "method",
        choices=[
            "task-arithmetic",
            "oneshot",
            "circuit-patch",
            "resta",
            "safetune",
            "steer",
            "safety-ft",
        ],
    )
    s.add_argument(
        "--value",
        type=float,
        help="scale / fraction / top-k / alpha; default: sweep from repair.yaml",
    )
    s.add_argument("--algo", help="SafeTune CLI algorithm (for `safetune`)")
    s.add_argument(
        "--source", default="drifted", help="drifted model to repair, e.g. drifted-harmful"
    )
    s.set_defaults(fn=_repair)

    s = sub.add_parser("compress", help="phase 5")
    s.add_argument("mode", choices=["uniform", "mixed", "bf16"])
    s.add_argument("--model", required=True)
    s.set_defaults(fn=_compress)

    s = sub.add_parser("report", help="charts and tables into results/")
    s.add_argument(
        "what",
        choices=["drift", "datashare", "similarity", "tradeoff", "compression", "table", "all"],
    )
    s.add_argument("--models", nargs="+")
    s.set_defaults(fn=_report)

    s = sub.add_parser("publish", help="upload to the Hugging Face Hub (private unless --public)")
    s.add_argument("what", choices=["dataset", "model", "logs"])
    s.add_argument("--repo", required=True)
    s.add_argument("--model")
    s.add_argument("--public", action="store_true")
    s.set_defaults(fn=_publish)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = (
        load_config(args.config, profile=args.profile)
        if args.config
        else load_config(profile=args.profile)
    )
    args.fn(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
