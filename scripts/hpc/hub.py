"""The Hub repos the GPU jobs read, taken from configs/ (GPU nodes are offline, the login node is not).

python scripts/hpc/hub.py download [model names]   # login node: models (default: all), dataset
python scripts/hpc/hub.py folders [model names]   # hub folders a job copies (default: all)
"""

from __future__ import annotations

import sys

from sdt.config import Config, load_config

# Duplicate weight formats some repos ship next to the safetensors.
SKIP = ["original/*", "metal/*", "*.bin", "*.pth", "*.gguf"]


def repos(config: Config, names: list[str] | None = None) -> tuple[list[str], list[str]]:
    """Hub ids of the named models (default: all) and of the datasets the jobs read."""
    chosen = {k: v for k, v in config["models"].items() if not names or k in names}
    models = {v for v in chosen.values() if "/" in str(v)}
    if not names or "translator" in names:  # the seq2seq translators go with the translator
        models |= {c["model"] for c in config["translation"]["seq2seq_candidates"]}
    models = sorted(models)
    return models, [config["capability"]["dataset"], config["finetune_harmful"]["dataset"]]


def skip_patterns(repo: str) -> list[str]:
    """SKIP, but keep *.bin for repos that ship no safetensors (e.g. NLLB-200)."""
    from huggingface_hub import list_repo_files

    has_safetensors = any(f.endswith(".safetensors") for f in list_repo_files(repo))
    return SKIP if has_safetensors else [p for p in SKIP if p != "*.bin"]


def main() -> None:
    config = load_config()
    if sys.argv[1:2] == ["folders"]:
        models, datasets = repos(config, sys.argv[2:])
        for repo in models:
            print("hub/models--" + repo.replace("/", "--"))
        for repo in datasets:
            print("hub/datasets--" + repo.replace("/", "--"))
    elif sys.argv[1:2] == ["download"]:
        from huggingface_hub import snapshot_download

        models, datasets = repos(config, sys.argv[2:])
        for repo in models:
            print(snapshot_download(repo, ignore_patterns=skip_patterns(repo)))
        for repo in datasets:
            print(snapshot_download(repo, repo_type="dataset"))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
