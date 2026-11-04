import dataclasses

import pytest

from sdt.cli import main
from sdt.config import DEFAULT_CONFIG, load_config
from sdt.evaluate import report
from sdt.io import write_json


@pytest.fixture
def cfg(tmp_path):
    """The project config with runs/ and results/ under a temporary folder."""
    return dataclasses.replace(load_config(), root=tmp_path)


def fake_metrics(cfg, name: str, asr: float = 0.1) -> None:
    langs = {lang: {"asr": asr} for lang in cfg.languages}
    write_json(cfg.runs_dir / "metrics" / f"{name}.json", {"name": name, "languages": langs})


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
