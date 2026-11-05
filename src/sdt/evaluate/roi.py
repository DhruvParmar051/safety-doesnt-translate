"""ROI: cost of each repair against retraining, and self-hosting Tiny Aya against a closed API.

Prices come only from the `roi:` section of the config. While a price is null, no dollar figure
is computed or written; the tables show minutes, hours and ratios instead.
"""

from __future__ import annotations

import logging
from typing import Any

from sdt.config import Config
from sdt.io import read_json

log = logging.getLogger(__name__)

# Name `sdt repair safety-ft` saves under (sdt.repair.safety_finetune.run): the retraining baseline.
BASELINE = "repair-safety-ft"
PRICE_KEYS = ["gpu_hour_usd", "api_usd_per_million_tokens"]
# Only throughput measured with vLLM on the H200 stands for a real deployment.
HOSTING_BACKEND = "vllm"


def _is_baseline(name: str) -> bool:
    """`repair-safety-ft`, or `repair-<tag>-safety-ft` for another drifted model."""
    return name.endswith(BASELINE.removeprefix("repair"))


def _source(row: dict[str, Any]) -> str:
    return row.get("source", "drifted")


def missing_prices(config: Config) -> list[str]:
    """Names of the `roi:` prices that are still null."""
    roi = config["roi"]
    return [key for key in PRICE_KEYS if roi.get(key) is None]


def repair_costs(config: Config) -> list[dict[str, Any]]:
    """Per repair: GPU minutes, dollars (None while unpriced) and cost relative to retraining."""
    folder = config.runs_dir / "repair"
    rows = [read_json(p) for p in sorted(folder.glob("*.json"))] if folder.exists() else []
    rows = [r for r in rows if "name" in r]  # best.<source>.json holds a choice, not a repair
    hour_usd = config["roi"].get("gpu_hour_usd")
    # Each drifted model has its own retraining run; a repair is compared with its own source's.
    baselines = {_source(r): r["gpu_minutes"] for r in rows if _is_baseline(r["name"])}
    out = []
    for row in rows:
        minutes = row.get("gpu_minutes")
        if minutes is None:  # steering edits no weights, so it has no repair cost to compare
            log.info("%s has no GPU minutes; left out of the ROI table", row["name"])
            continue
        out.append(
            {
                "name": row["name"],
                "gpu_minutes": minutes,
                "usd": None if hour_usd is None else minutes / 60 * hour_usd,
                # Same hourly price on both sides, so the dollar ratio is the minutes ratio.
                "ratio_vs_retraining": minutes / baselines[_source(row)]
                if baselines.get(_source(row))
                else None,
            }
        )
    return sorted(out, key=lambda r: (_is_baseline(r["name"]), r["name"]))


def hosting_cost(config: Config, model: str) -> dict[str, Any]:
    """Monthly cost of serving `model` on our GPU against the same volume on a closed API."""
    roi = config["roi"]
    stats_file = config.runs_dir / "generations" / f"{model}.stats.json"
    if not stats_file.exists():
        raise FileNotFoundError(f"No measured throughput: run `sdt evaluate {model}` first")
    stats = read_json(stats_file)
    if stats.get("backend") != HOSTING_BACKEND:
        raise ValueError(
            f"{stats_file} was measured with {stats.get('backend')}; hosting cost needs the "
            f"{HOSTING_BACKEND} run on the H200"
        )
    tokens = roi["monthly_requests"] * roi["tokens_per_request"]
    gpu_hours = tokens / stats["tokens_per_second"] / 3600
    hour_usd, api_price = roi.get("gpu_hour_usd"), roi.get("api_usd_per_million_tokens")
    self_usd = None if hour_usd is None else gpu_hours * hour_usd
    api_usd = None if api_price is None else tokens / 1e6 * api_price
    return {
        "model": model,
        "tokens_per_second": stats["tokens_per_second"],
        "tokens_per_month": tokens,
        "gpu_hours_per_month": gpu_hours,
        "self_hosted_usd": self_usd,
        "api_usd": api_usd,
        "ratio_vs_api": self_usd / api_usd if self_usd is not None and api_usd else None,
    }


def _usd(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:,.2f}" if value < 100 else f"${value:,.0f}"


def _price(value: float | None) -> str:
    return "not set" if value is None else _usd(value)


def _ratio(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}x"


def roi_markdown(config: Config, model: str | None = None) -> str:
    """Two tables (repairs vs retraining, self-hosting vs API) and the assumptions behind them."""
    roi = config["roi"]
    model = model or roi["hosting_model"]
    missing = missing_prices(config)
    lines = ["## Repair vs retraining", ""]
    repairs = repair_costs(config)
    if repairs:
        lines += [
            "| Repair | GPU minutes | Cost | Cost vs retraining |",
            "| --- | --- | --- | --- |",
        ]
        for r in repairs:
            name = f"{r['name']} (retraining baseline)" if _is_baseline(r["name"]) else r["name"]
            lines.append(
                f"| {name} | {r['gpu_minutes']:.1f} | {_usd(r['usd'])} "
                f"| {_ratio(r['ratio_vs_retraining'])} |"
            )
        if not any(_is_baseline(r["name"]) for r in repairs):
            lines += ["", f"No `{BASELINE}` run yet, so there is no retraining cost to compare."]
    else:
        lines.append("No repair runs yet.")

    lines += ["", "## Self-hosted Tiny Aya vs a closed API", ""]
    try:
        h = hosting_cost(config, model)
    except (FileNotFoundError, ValueError) as err:
        lines.append(f"Not computed: {err}.")
    else:
        lines += [
            "| Option | Tokens/s | GPU hours / month | Cost / month |",
            "| --- | --- | --- | --- |",
            f"| Self-hosted {model} | {h['tokens_per_second']:,.0f} "
            f"| {h['gpu_hours_per_month']:,.1f} | {_usd(h['self_hosted_usd'])} |",
            f"| Closed API | — | — | {_usd(h['api_usd'])} |",
        ]
        if h["ratio_vs_api"] is not None:
            lines += ["", f"Self-hosting costs {_ratio(h['ratio_vs_api'])} the API price."]

    lines += [
        "",
        f"Assumptions: GPU price per hour {_price(roi.get('gpu_hour_usd'))}; "
        f"{roi['monthly_requests']:,} requests/month of {roi['tokens_per_request']:,} tokens each "
        f"(prompt + reply); API price per million tokens "
        f"{_price(roi.get('api_usd_per_million_tokens'))}; throughput from the {HOSTING_BACKEND} "
        f"run of {model}; repair cost = GPU minutes / 60 * hourly price; self-hosting hours = "
        "tokens / tokens per second / 3600.",
    ]
    if missing:
        lines += [
            "",
            "Dollar figures withheld until set in the config: " + ", ".join(missing) + ".",
        ]
    return "\n".join(lines) + "\n"
