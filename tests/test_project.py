import subprocess
from pathlib import Path

import pytest

from sdt.cli import build_parser
from sdt.config import REPO_ROOT, load_config
from sdt.data.text_checks import EXPECTED_SCRIPT

HOOK = REPO_ROOT / ".githooks" / "commit-msg"


def test_config_languages_are_complete():
    cfg = load_config()
    assert len(cfg.languages) == 8
    for lang in cfg.languages:
        assert lang in EXPECTED_SCRIPT
        assert cfg.language(lang)["variant"] in {"fire", "water", "earth"}
        assert lang in cfg["task"]["language_codes"]
    # every regional variant has at least one language it favours (RQ 1b)
    assert {cfg.language(lang)["variant"] for lang in cfg.languages} == {"fire", "water", "earth"}


def test_model_path_resolution(tmp_path):
    cfg = load_config()
    assert cfg.model_path("global") == "CohereLabs/tiny-aya-global"
    with pytest.raises(KeyError):
        cfg.model_path("does-not-exist")


def test_cli_parses_every_command():
    parser = build_parser()
    for argv in (
        ["data", "build"],
        ["evaluate", "drifted", "--no-task"],
        ["repair", "task-arithmetic", "--value", "0.5"],
        ["compress", "mixed", "--model", "x"],
        ["report", "all"],
        ["generate", "global", "--split", "calib"],
    ):
        assert callable(parser.parse_args(argv).fn)


def run_hook(tmp_path: Path, message: str) -> int:
    msg = tmp_path / "MSG"
    msg.write_text(message)
    return subprocess.run(["bash", str(HOOK), str(msg)], capture_output=True).returncode


@pytest.mark.parametrize(
    "message",
    [
        "feat(repair): add one-shot saliency patch",
        "fix(judge): parse labels in code fences",
        "docs: explain AI safety metrics",
        "feat(judge): LLM judge prompt for refusals",
        "# Please enter the commit message\n# claude\nfeat: x",  # comment lines are ignored
    ],
)
def test_hook_allows_normal_messages(tmp_path, message):
    assert run_hook(tmp_path, message) == 0


@pytest.mark.parametrize(
    "message",
    [
        "feat: add repair\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
        "fix: written with Claude",
        "feat: x\n\nGenerated with some tool",
        "chore: AI-generated config",
        "feat: ChatGPT suggested this",
        "feat: x\n\nCo-authored-by: copilot-bot <bot@example.com>",
    ],
)
def test_hook_rejects_ai_attribution(tmp_path, message):
    assert run_hook(tmp_path, message) == 1
