import dataclasses

import pytest

from sdt.cli import main
from sdt.config import DEFAULT_CONFIG, load_config
from sdt.evaluate import report, roi
from sdt.io import write_json


@pytest.fixture
def cfg(tmp_path):
    """The project config with runs/ and results/ under a temporary folder."""
    return dataclasses.replace(load_config(), root=tmp_path)


def fake_metrics(cfg, name: str, asr: float = 0.1) -> None:
    langs = {lang: {"asr": asr} for lang in cfg.languages}
    write_json(cfg.runs_dir / "metrics" / f"{name}.json", {"name": name, "languages": langs})


def with_roi(cfg, **values):
    raw = {**cfg.raw, "roi": {**cfg["roi"], **values}}
    return dataclasses.replace(cfg, raw=raw)


# Task 1: compression chart in `sdt report all`


def test_compression_groups_order_source_uniform_mixed(cfg):
    for name in ("repair-oneshot-4bit-mixed", "repair-oneshot", "repair-oneshot-4bit-uniform"):
        fake_metrics(cfg, name)
    fake_metrics(cfg, "global")  # no compressed version: no chart
    assert report.compression_groups(cfg) == {
        "repair-oneshot": [
            "repair-oneshot",
            "repair-oneshot-4bit-uniform",
            "repair-oneshot-4bit-mixed",
        ]
    }


def test_compression_groups_only_existing(cfg):
    fake_metrics(cfg, "drifted-4bit-mixed")
    assert report.compression_groups(cfg) == {"drifted": ["drifted-4bit-mixed"]}
    fake_metrics(cfg, "drifted.llama_guard")  # secondary-judge metrics are not models
    assert report.compression_groups(cfg) == {"drifted": ["drifted-4bit-mixed"]}


def test_compression_groups_empty(cfg):
    fake_metrics(cfg, "global")
    assert report.compression_groups(cfg) == {}


def test_compression_chart_written_per_source(cfg):
    for name in ("x", "x-4bit-uniform", "x-4bit-mixed"):
        fake_metrics(cfg, name)
    for source, group in report.compression_groups(cfg).items():
        report.compression_chart(cfg, group, cfg.results_dir / f"compression_{source}.png")
    assert (cfg.results_dir / "compression_x.png").exists()


# Task 2: per-layer drift chart


def test_drop_layer():
    # mean over languages per layer: 1.0, 0.95, 0.6, 0.5 -> stays below 0.9 from layer 2
    assert report.drop_layer([[1.0, 1.0, 0.7, 0.6], [1.0, 0.9, 0.5, 0.4]]) == 2
    # a dip that recovers is not a lasting drop
    assert report.drop_layer([[1.0, 0.5, 1.0, 1.0]]) is None
    assert report.drop_layer([[0.5, 0.5]]) == 0


def test_layer_drift_chart_one_png_per_model(cfg, tmp_path):
    write_json(
        cfg.runs_dir / "locate" / "layer_strength.json",
        {
            "drifted": {"en": [1.0, 0.9, 0.6, 0.4], "hi": [1.0, 0.8, 0.5, 0.3]},
            "fire": {"en": [1.0, 1.0, 1.1, 1.0], "hi": [0.9, 1.0, 1.0, 1.05]},
        },
    )
    # A config file under tmp_path/configs makes tmp_path the project root for the CLI.
    config_file = tmp_path / "configs" / "project.yaml"
    config_file.parent.mkdir()
    config_file.write_text(DEFAULT_CONFIG.read_text())
    main(["--config", str(config_file), "report", "layer"])
    assert sorted(p.name for p in cfg.results_dir.glob("layer_drift_*.png")) == [
        "layer_drift_drifted.png",
        "layer_drift_fire.png",
    ]


# Task 3: ROI


def fake_repair(cfg, name: str, minutes: float | None) -> None:
    write_json(cfg.runs_dir / "repair" / f"{name}.json", {"name": name, "gpu_minutes": minutes})


def fake_stats(cfg, model: str, tokens_per_second: float, backend: str = "vllm") -> None:
    write_json(
        cfg.runs_dir / "generations" / f"{model}.stats.json",
        {"tokens_per_second": tokens_per_second, "backend": backend},
    )


def test_repair_costs_hand_computed(cfg):
    cfg = with_roi(cfg, gpu_hour_usd=4.0)
    fake_repair(cfg, "repair-oneshot-0.005", 6.0)
    fake_repair(cfg, "repair-safety-ft", 30.0)
    fake_repair(cfg, "repair-steer-0.3", None)  # no weights edited: left out
    costs = {r["name"]: r for r in roi.repair_costs(cfg)}
    assert set(costs) == {"repair-oneshot-0.005", "repair-safety-ft"}
    assert costs["repair-oneshot-0.005"]["usd"] == pytest.approx(0.40)
    assert costs["repair-safety-ft"]["usd"] == pytest.approx(2.00)
    assert costs["repair-oneshot-0.005"]["ratio_vs_retraining"] == pytest.approx(0.2)


def test_hosting_cost_hand_computed(cfg):
    # 1,000 requests x 300 tokens = 300,000 tokens; at 100 tokens/s = 3,000 s = 5/6 hour
    cfg = with_roi(
        cfg,
        gpu_hour_usd=4.0,
        monthly_requests=1000,
        tokens_per_request=300,
        api_usd_per_million_tokens=10.0,
    )
    fake_stats(cfg, "global", 100.0)
    h = roi.hosting_cost(cfg, "global")
    assert h["gpu_hours_per_month"] == pytest.approx(5 / 6)
    assert h["self_hosted_usd"] == pytest.approx(10 / 3)
    assert h["api_usd"] == pytest.approx(3.0)  # 0.3 million tokens x $10


def test_hosting_cost_refuses_non_vllm_throughput(cfg):
    fake_stats(cfg, "global", 100.0, backend="mlx")
    with pytest.raises(ValueError, match="vllm"):
        roi.hosting_cost(cfg, "global")


def test_roi_markdown_without_prices_has_no_dollars(cfg):
    assert roi.missing_prices(cfg) == ["gpu_hour_usd", "api_usd_per_million_tokens"]
    fake_repair(cfg, "repair-oneshot-0.005", 6.0)
    fake_repair(cfg, "repair-safety-ft", 30.0)
    fake_stats(cfg, "global", 100.0)
    text = roi.roi_markdown(cfg)
    assert "$" not in text
    assert "0.20x" in text
    assert "gpu_hour_usd, api_usd_per_million_tokens" in text


def test_roi_markdown_with_prices(cfg):
    cfg = with_roi(cfg, gpu_hour_usd=4.0, api_usd_per_million_tokens=10.0)
    fake_repair(cfg, "repair-oneshot-0.005", 6.0)
    fake_repair(cfg, "repair-safety-ft", 30.0)
    text = roi.roi_markdown(cfg)
    assert "| repair-oneshot-0.005 | 6.0 | $0.40 | 0.20x |" in text
    assert "| repair-safety-ft (retraining baseline) | 30.0 | $2.00 | 1.00x |" in text
    assert "Not computed" in text  # no throughput measured yet
    assert "withheld" not in text


# Fitting the reports to the real run folders


def test_only_model_metrics_count_as_models(cfg):
    fake_metrics(cfg, "drifted")
    fake_metrics(cfg, "drifted.llama_guard")
    write_json(cfg.runs_dir / "metrics" / "drifted.via_english.json", {"sw": {"n": 1}})
    assert report.available_models(cfg) == ["drifted"]


def test_compression_reference_is_on_the_same_backend(cfg):
    for name in ("x", "x-bf16-hf", "x-4bit-uniform", "x-4bit-mixed"):
        fake_metrics(cfg, name)
    assert report.compression_groups(cfg) == {"x": ["x-bf16-hf", "x-4bit-uniform", "x-4bit-mixed"]}


def test_each_repair_is_compared_with_its_own_retraining(cfg):
    def repair(name, minutes, source):
        write_json(
            cfg.runs_dir / "repair" / f"{name}.json",
            {"name": name, "gpu_minutes": minutes, "source": source},
        )

    repair("repair-safety-ft", 40.0, "drifted")
    repair("repair-harmful-safety-ft", 20.0, "drifted-harmful")
    repair("repair-oneshot-0.01", 4.0, "drifted")
    repair("repair-harmful-oneshot-0.01", 4.0, "drifted-harmful")
    write_json(cfg.runs_dir / "repair" / "best.drifted-harmful.json", {"best": "x"})  # not a repair
    costs = {r["name"]: r["ratio_vs_retraining"] for r in roi.repair_costs(cfg)}
    assert len(costs) == 4
    assert costs["repair-oneshot-0.01"] == pytest.approx(0.1)
    assert costs["repair-harmful-oneshot-0.01"] == pytest.approx(0.2)
    assert costs["repair-harmful-safety-ft"] == pytest.approx(1.0)
