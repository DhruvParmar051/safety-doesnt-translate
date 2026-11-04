import dataclasses

import pytest

from sdt.config import load_config
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
