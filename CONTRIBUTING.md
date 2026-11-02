# Contributing

Rules for the two of us (Dhruv and Falak). Short on purpose: if a rule is here, follow it; if it is not, use judgement and match the code around you.

## Setup

```bash
uv sync --extra dev          # Mac: data, judge parsing, metrics, reports, tests
uv sync --extra dev --extra gpu --extra safety --extra interp   # H200
scripts/setup_hooks.sh       # once per clone: turns on the commit-message check
uv run pytest
```

Python 3.12 (SafeTune needs ≥ 3.12). Dependencies live in `pyproject.toml` only; add them with `uv add`, never `pip install` into the shared environment.

## Commit messages

### No AI attribution

- **Never mention Claude, Anthropic, ChatGPT, Copilot or any AI assistant in a commit message, PR title or PR description.**
- **No `Co-Authored-By:` trailer for any AI tool.** A co-author trailer is how GitHub adds someone to the contributors list, so this keeps the list to the two of us.
- No "Generated with …" footers.
- Commit only under your own git identity (`git config user.name` / `user.email` set to your GitHub account).

The `commit-msg` hook in `.githooks/` rejects messages that break this rule. Turn it on with `scripts/setup_hooks.sh`. Writing about the subject itself is fine: "AI safety", "safety judge" or "LLM judge prompt" all pass.

If a bad commit gets through, fix it before pushing:

```bash
git commit --amend            # last commit
git rebase -i origin/main     # older unpushed commits: reword
```

Never force-push `main`.

### Format

[Conventional Commits](https://www.conventionalcommits.org/), imperative, ≤ 72 characters in the subject:

```
feat(repair): add one-shot saliency patch
fix(judge): parse labels wrapped in code fences
data: add Swahili XSTest translations
docs: add ROI section to article
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `data`, `exp` (an experiment run or config), `chore`. The scope is the package: `data`, `judge`, `metrics`, `generation`, `finetune`, `locate`, `repair`, `compress`, `evaluate`, `report`, `space`.

## Branches and pull requests

- `main` always runs. Work on `<name>/<short-topic>`, e.g. `falak/xstest-translate`, `dhruv/oneshot-patch`.
- One pull request = one task line in `IMPLEMENTATION_PLAN.md`. Keep it reviewable in 15 minutes.
- The other person reviews and merges. Squash merge, so the PR title becomes the commit; the same message rules apply to it.
- CI-equivalent before asking for review: `uv run ruff format --check . && uv run ruff check . && uv run pytest`.

## Python style

- **Formatter and linter:** `ruff format` and `ruff check` (config in `pyproject.toml`, line length 100). Don't argue with the formatter.
- **Types:** every public function has type hints. Use built-in generics (`list[str]`, `dict[str, float]`, `X | None`).
- **Docstrings:** one line saying what the function does, for every public function. Add a longer docstring only when the *why* is not obvious (a paper's formula, a library quirk).
- **Comments** explain why, not what. A `# VERIFY(M1)` comment marks a library call whose signature we have not confirmed yet.
- **Names:** `snake_case` functions and variables, `PascalCase` classes, `UPPER_CASE` constants. Language codes are ISO 639-1 (`hi`, `sw`, `fil` for Filipino) everywhere.
- **Imports:** standard library, third-party, then `sdt`. Heavy libraries (`torch`, `vllm`, `transformers`, `safetune`, `circuitkit`, `curatorkit`, `auditkit`) are imported *inside* the function that needs them, so the Mac and the tests run without a GPU stack.
- **Logging:** `logging.getLogger(__name__)`, never `print` in library code. `print` is fine in the CLI.
- **Errors:** fail loudly with a message that says what to fix. No bare `except:`; no silently skipped rows (count and log them).
- **Small modules:** one method per file under `sdt/repair/`, registered in `sdt/repair/__init__.py`.

## Experiments and reproducibility

- **No hard-coded paths, model ids, languages or hyperparameters.** They go in `configs/*.yaml` and are read through `sdt.config`.
- **Seeds:** every command takes `seed` from the config (default 0); sampling is greedy unless a config says otherwise.
- **Calibration vs test:** anything that *builds* a repair (refusal directions, saliency, components, steering) reads only `split == "calib"`. Every reported number reads only `split == "test"`. The split test fails if an id appears in both.
- **Every run writes its config** next to its output (`run_config.yaml`), so each number in the article can be traced to a command.
- **Same chat template for every model**: the tokenizer's default template, no extra system prompt, unless an experiment is about the system prompt.
- **Don't commit data, checkpoints or generations.** `data/`, `runs/` and `*.safetensors` are git-ignored; they go to Hugging Face. Small result tables (`runs/reports/*.md`, charts) are copied to `results/` and committed.
- **Notebooks** are for exploration only. Clear outputs before committing; anything we rely on moves into `src/sdt`.

## Tests

- `pytest`, files in `tests/` named `test_<module>.py`.
- Every function that computes a reported number (metrics, judge parsing, splits) has a test with a hand-computed answer.
- Weight-space methods are tested on tiny `torch.nn` models: check the maths, not the GPU.
- Tests never download models or datasets.

## Safety of the data itself

The harmful-prompt sets and model replies contain disturbing content. Keep them out of issues, PR descriptions and screenshots; refer to rows by `id`. Model replies flagged harmful are never uploaded in plain text in the Space; show the label and a short redacted excerpt.
