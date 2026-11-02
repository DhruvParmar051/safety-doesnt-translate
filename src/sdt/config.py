"""Loads configs/*.yaml and resolves model names and run paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "project.yaml"
DEFAULT_REPAIR_CONFIG = REPO_ROOT / "configs" / "repair.yaml"


@dataclass(frozen=True)
class Config:
    """Project settings plus helpers to locate files; `raw` is the parsed YAML."""

    raw: dict[str, Any]
    repair: dict[str, Any] = field(default_factory=dict)
    root: Path = REPO_ROOT

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 0))

    @property
    def languages(self) -> list[str]:
        """Language codes in the order the config lists them."""
        return list(self.raw["languages"])

    def language(self, code: str) -> dict[str, Any]:
        return self.raw["languages"][code]

    @property
    def data_dir(self) -> Path:
        return self.root / self.raw["paths"]["data_dir"]

    @property
    def runs_dir(self) -> Path:
        return self.root / self.raw["paths"]["runs_dir"]

    @property
    def results_dir(self) -> Path:
        return self.root / self.raw["paths"]["results_dir"]

    def run_path(self, *parts: str) -> Path:
        """A path under runs/, with parent folders created."""
        path = self.runs_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def prompts_path(self) -> Path:
        return self.data_dir / "processed" / "prompts.jsonl"

    def model_path(self, name: str) -> str:
        """Hub id for a configured model, else the local checkpoint runs/models/<name>."""
        models = self.raw["models"]
        if name in models:
            return str(models[name])
        local = self.runs_dir / "models" / name
        if local.exists():
            return str(local)
        raise KeyError(f"Unknown model '{name}': not in configs/project.yaml and no {local}")


def load_config(
    path: Path | str = DEFAULT_CONFIG, repair_path: Path | str = DEFAULT_REPAIR_CONFIG
) -> Config:
    """Read the project config and, if present, the repair config."""
    raw = yaml.safe_load(Path(path).read_text())
    repair_file = Path(repair_path)
    repair = yaml.safe_load(repair_file.read_text()) if repair_file.exists() else {}
    return Config(raw=raw, repair=repair or {}, root=Path(path).resolve().parents[1])


def save_run_config(config: Config, out_dir: Path, **extra: Any) -> None:
    """Write the config used for a run next to its outputs, so every number is traceable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"project": config.raw, "repair": config.repair, "args": extra}
    (out_dir / "run_config.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
