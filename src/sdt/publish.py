"""Uploads datasets, models and logs to the Hugging Face Hub (private by default)."""

from __future__ import annotations

import logging

from sdt.config import Config

log = logging.getLogger(__name__)


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def push_dataset(config: Config, repo_id: str, private: bool = True) -> None:
    """The cleaned prompt set with calibration/test splits, plus the curation rejects."""
    api = _api()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=str(config.data_dir / "processed")
    )
    log.info("Pushed dataset to %s", repo_id)


def push_model(config: Config, name: str, repo_id: str, private: bool = True) -> None:
    """A repaired checkpoint (or the drifted model's LoRA adapter) with its run config."""
    api = _api()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(config.runs_dir / "models" / name))
    log.info("Pushed %s to %s", name, repo_id)


def push_logs(config: Config, repo_id: str, private: bool = True) -> None:
    """Metrics, judge labels, repair records and locate outputs. Raw replies stay private."""
    api = _api()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    for folder in ("metrics", "judgements", "repair", "locate", "finetune"):
        path = config.runs_dir / folder
        if path.exists():
            api.upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(path),
                path_in_repo=folder,
                ignore_patterns=["*.pt"],
            )
    log.info("Pushed logs to %s", repo_id)
