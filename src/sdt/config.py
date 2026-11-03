"""Loads configs/*.yaml (plus an optional profile such as `mac`) and resolves models and paths."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "project.yaml"
DEFAULT_REPAIR_CONFIG = REPO_ROOT / "configs" / "repair.yaml"
PROFILE_ENV = "SDT_PROFILE"


@dataclass(frozen=True)
class Config:
    """Project settings plus helpers to locate files; `raw` is the parsed YAML."""

    raw: dict[str, Any]
    repair: dict[str, Any] = field(default_factory=dict)
    root: Path = REPO_ROOT
    profile: str | None = None

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


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Copy of `base` with `override` merged in; nested dicts merge, everything else replaces."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(
    path: Path | str = DEFAULT_CONFIG,
    repair_path: Path | str = DEFAULT_REPAIR_CONFIG,
    profile: str | None = None,
) -> Config:
    """Read the project and repair configs, then apply `configs/<profile>.yaml` on top.

    The profile comes from the argument or the SDT_PROFILE environment variable. A profile file may
    override any project key, and its `repair:` section overrides configs/repair.yaml.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    repair_file = Path(repair_path)
    repair = (yaml.safe_load(repair_file.read_text()) if repair_file.exists() else {}) or {}

    profile = profile or os.environ.get(PROFILE_ENV) or None
    if profile:
        profile_file = path.parent / f"{profile}.yaml"
        if not profile_file.exists():
            raise FileNotFoundError(f"Profile '{profile}' needs {profile_file}")
        overlay = yaml.safe_load(profile_file.read_text()) or {}
        repair = deep_merge(repair, overlay.pop("repair", {}))
        raw = deep_merge(raw, overlay)
    return Config(raw=raw, repair=repair, root=path.resolve().parents[1], profile=profile)


LEXSI_LIBRARIES = ("curatorkit", "safetune", "circuitkit", "auditkit")


def library_versions() -> dict[str, str | None]:
    """Installed Lexsi Labs library versions (None if missing); without one, its step falls back."""
    import importlib.metadata

    versions: dict[str, str | None] = {}
    for name in LEXSI_LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def save_run_config(config: Config, out_dir: Path, **extra: Any) -> None:
    """Write the config used for a run next to its outputs, so every number is traceable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile": config.profile,
        "project": config.raw,
        "repair": config.repair,
        "args": extra,
        "libraries": library_versions(),
    }
    (out_dir / "run_config.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
