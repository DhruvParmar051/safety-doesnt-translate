"""GPU-node checks for scripts/hpc/smoke.sbatch: offline dataset, then two vLLM engines in turn.

A file, not a heredoc: vLLM starts its engine with `spawn`, which re-imports the main module.
"""

from __future__ import annotations

import sys


def main(model: str, dataset: str) -> None:
    import torch
    from datasets import load_dataset

    from sdt.config import load_config
    from sdt.evaluate.capability import split_files
    from sdt.generation import chat

    config = load_config()
    print("== torch", torch.__version__, torch.cuda.get_device_name(0))
    files = split_files(dataset, config["capability"]["split"])
    rows = load_dataset("parquet", data_files=files["en"], split="train")
    print(f"== offline dataset: {len(files)} languages, {len(rows)} English rows")

    conversations = [[{"role": "user", "content": "Reply with one word: hello"}]] * 4
    for attempt in (1, 2):
        free, total = torch.cuda.mem_get_info()
        print(f"== engine {attempt}: {free / 2**30:.1f} of {total / 2**30:.1f} GiB free at start")
        replies = chat(config, model, conversations, max_new_tokens=16)
        print(f"== engine {attempt}: {replies[0]!r}")
    free, _ = torch.cuda.mem_get_info()
    print(f"== after both engines: {free / 2**30:.1f} GiB free")


if __name__ == "__main__":
    main(*sys.argv[1:])
