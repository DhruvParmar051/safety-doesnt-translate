# Instructions for AI coding assistants

Before changing anything in this repository, read:

1. [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md): what the project is, the models, data, metrics, pipeline, and the correctness rules that must not be undone.
2. [CONTRIBUTING.md](CONTRIBUTING.md): coding standard and commit rules.
3. [docs/TASKS_FALAK.md](docs/TASKS_FALAK.md): the open code tasks, with acceptance criteria.

Hard rules:
- Commit messages, PR titles and PR descriptions must not mention Claude, Anthropic, ChatGPT, Copilot or any AI assistant, and must not carry a `Co-Authored-By` trailer for an AI tool.
- Do not hard-code model ids, paths, languages or hyperparameters; read them from `configs/`.
- Anything that builds a repair uses only the `calib` split; anything reported uses only the `test` split.
- Run `uv run ruff format --check . && uv run ruff check . && uv run pytest` before proposing a change.
