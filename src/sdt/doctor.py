"""`sdt doctor`: checks this machine can run the pipeline before any long job starts."""

from __future__ import annotations

import importlib.metadata
import platform
import shutil

from sdt.config import Config

PACKAGES = [
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "trl",
    "datasets",
    "huggingface_hub",
    "vllm",
    "bitsandbytes",
    "mlx",
    "mlx-lm",
    "optimum-quanto",
    "safetune",
    "circuitkit",
    "curatorkit",
    "auditkit",
]
# Approximate bf16 weight sizes, to compare with this machine's memory.
MODEL_GB = {
    "tiny-aya": 6.7,
    "aya-expanse-8b": 16.0,
    "aya-expanse-32b": 64.0,
    "Llama-Guard-3-8B": 16.0,
    "Llama-Guard-3-1B": 2.5,
}


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _memory_gb() -> float | None:
    try:
        import os

        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return None


def model_access(repo: str) -> str:
    """'ok', 'gated: accept the licence', 'not logged in', or the error text."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    try:
        HfApi().auth_check(repo)
        return "ok"
    except GatedRepoError:
        return f"gated: accept the licence at https://huggingface.co/{repo}"
    except RepositoryNotFoundError:
        return "not found (or not logged in)"
    except Exception as exc:  # network, auth: report, don't crash the check
        return f"error: {exc.__class__.__name__}: {exc}"


def run_doctor(config: Config) -> bool:
    """Prints a checklist; returns True when nothing blocks a run."""
    ok = True
    print(
        f"Profile: {config.profile or 'default'}   Python {platform.python_version()} on {platform.platform()}"
    )
    memory = _memory_gb()
    print(f"Memory: {memory:.0f} GB" if memory else "Memory: unknown")
    free = shutil.disk_usage(config.root).free / 1e9
    print(f"Free disk: {free:.0f} GB" + ("  (models + data need ~100 GB)" if free < 100 else ""))

    try:
        from sdt.models import default_device

        print(f"Device: {default_device()}")
    except ImportError:
        print("Device: torch not installed")
        ok = False

    print("\nPackages:")
    for package in PACKAGES:
        version = _version(package)
        print(f"  {package:<16} {version or '-'}")
    backend = config["generation"]["backend"]
    if backend == "mlx" and not _version("mlx-lm"):
        print("  ! generation.backend is mlx but mlx-lm is missing (uv sync --extra mlx)")
        ok = False
    if backend == "vllm" and not _version("vllm"):
        print(
            "  ! generation.backend is vllm but vLLM is missing (on Apple Silicon set generation.backend: mlx)"
        )
        ok = False

    print("\nModel access (Hugging Face):")
    if not _version("huggingface_hub"):
        print("  huggingface_hub not installed")
        return False
    for name, repo in config["models"].items():
        status = model_access(repo)
        size = next((gb for key, gb in MODEL_GB.items() if key in repo), None)
        fits = (
            ""
            if size is None or memory is None or size < 0.8 * memory
            else "  ! too large for this machine"
        )
        print(f"  {name:<16} {repo:<40} {status}{fits}")
        ok &= status == "ok"
    print("\nAll checks passed." if ok else "\nFix the items above before running the pipeline.")
    return ok
