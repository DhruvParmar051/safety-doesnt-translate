"""Uploads the Space (app, judge rubric, optional curated examples) to a Hugging Face Space repo.

Usage:
  python space/deploy.py <user>/<space> --left <user>/<drifted-repo> --right <user>/<repaired-repo>

The Space is created private. After the first upload, in the Space settings:
  1. Hardware: choose ZeroGPU.
  2. Secrets: add HF_TOKEN (read access to the gated Tiny Aya / Aya Expanse models and our repos).
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_FILES = ["app.py", "display.py", "requirements.txt", "README.md", "space_results.jsonl"]
# Only what the app imports: the judge rubric and what it needs.
SDT_FILES = ["__init__.py", "config.py", "schema.py", "judge/__init__.py", "judge/rubric.py"]


def stage(folder: Path) -> list[str]:
    """Copies the Space's files into `folder`; returns the relative paths staged."""
    staged = []
    for name in SPACE_FILES:
        src = ROOT / "space" / name
        if src.exists():
            shutil.copy2(src, folder / name)
            staged.append(name)
    for name in SDT_FILES:
        dst = folder / "sdt" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "src" / "sdt" / name, dst)
        staged.append(f"sdt/{name}")
    return staged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("space", help="<user>/<space>")
    parser.add_argument("--left", required=True, help="drifted model repo")
    parser.add_argument("--right", required=True, help="repaired model repo")
    parser.add_argument("--judge", default="CohereLabs/aya-expanse-8b")
    parser.add_argument("--public", action="store_true")
    parser.add_argument(
        "--live",
        action="store_true",
        help="add live prompts; refused with --public, since the judge can miss harm",
    )
    args = parser.parse_args()

    if args.live and args.public:
        parser.error("--live is for private Spaces: the live judge can miss harm in some languages")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(
        args.space, repo_type="space", space_sdk="gradio", private=not args.public, exist_ok=True
    )
    with tempfile.TemporaryDirectory() as tmp:
        staged = stage(Path(tmp))
        api.upload_folder(repo_id=args.space, repo_type="space", folder_path=tmp)
    for key, value in {
        "SDT_LEFT_MODEL": args.left,
        "SDT_RIGHT_MODEL": args.right,
        "SDT_JUDGE_MODEL": args.judge,
        "SDT_LIVE": "1" if args.live else "0",
    }.items():
        api.add_space_variable(args.space, key, value)
    print(f"Uploaded {len(staged)} files to https://huggingface.co/spaces/{args.space}")
    print("Next: Settings -> Hardware: ZeroGPU; Settings -> Secrets: HF_TOKEN.")


if __name__ == "__main__":
    main()
