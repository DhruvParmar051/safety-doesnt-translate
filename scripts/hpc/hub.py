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
    models = sorted({v for v in chosen.values() if "/" in str(v)})
    return models, [config["capability"]["dataset"]]


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
            print(snapshot_download(repo, ignore_patterns=SKIP))
        for repo in datasets:
            print(snapshot_download(repo, repo_type="dataset"))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
