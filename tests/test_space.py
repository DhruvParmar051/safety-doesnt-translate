import subprocess
import sys
from pathlib import Path

import pytest

SPACE = Path(__file__).resolve().parents[1] / "space"
sys.path.insert(0, str(SPACE))

from deploy import stage  # noqa: E402
from display import EXCERPT_CHARS, clean_prompt, present  # noqa: E402


def test_safe_and_refusal_shown_in_full():
    long = "x" * 500
    assert long in present("safe", long)
    assert long in present("refusal", long)


@pytest.mark.parametrize("label", ["harmful", "invalid", "something-else"])
def test_harmful_or_unlabelled_is_cut(label):
    shown = present(label, "y" * 500)
    assert "y" * (EXCERPT_CHARS + 1) not in shown
    assert "[rest hidden]" in shown


def test_clean_prompt():
    assert clean_prompt("  hi  ") == "hi"
    assert len(clean_prompt("a" * 5000)) == 1000
    with pytest.raises(ValueError):
        clean_prompt("   ")


def test_staged_space_imports_the_rubric_on_its_own(tmp_path):
    staged = stage(tmp_path)
    assert {"app.py", "display.py", "requirements.txt", "README.md"} <= set(staged)
    code = 'import sys; sys.path.insert(0, sys.argv[1]); from sdt.judge.rubric import parse_label; print(parse_label(\'{"label": "safe"}\').value)'
    out = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, cwd=tmp_path
    )
    assert out.stdout.strip() == "safe", out.stderr


def test_static_page_cuts_harmful_replies():
    from static_site import shown

    long = "y" * 500
    assert shown("safe", long)["text"] == long
    for label in ("harmful", "invalid"):
        assert len(shown(label, long)["text"]) < EXCERPT_CHARS + 20


def test_static_links_open_outside_the_space_frame(tmp_path):
    from static_site import build

    build(tmp_path, None, {"Code": "https://github.com/x/y"})
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert 'href="https://github.com/x/y" target="_blank"' in page
