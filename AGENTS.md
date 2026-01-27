# Repository Guidelines

## Project Structure & Module Organization
- `lmms_eval/`: evaluation harness, CLI entrypoint, tasks, metrics, and model adapters.
- `token_compressor/`: VidCom2 and baseline compression methods (e.g., IPCV, iLLaVA, CDPruner) plus model patches.
- `examples/` and `scripts/`: runnable examples and batch scripts for evaluations.
- `docs/`: usage notes and reference docs.
- `miscs/` and `tools/`: utilities and lightweight checks (look for `miscs/test_*.py`).

## Build, Test, and Development Commands
- `uv sync` to create/update the environment from `uv.lock`.
- Run evaluations with `python -m lmms_eval ...` (single GPU) or `accelerate launch -m lmms_eval ...` (multi-GPU).
- Tooling: `uv run ruff format .`, `uv run ruff check .`, `uv run pyright`, `uv run pytest`.

## Coding Style & Naming Conventions
- Follow PEP 8 naming (snake_case functions, PascalCase classes).
- Line length: 88 chars (per `CLAUDE.md`); use ruff for formatting.
- Add type hints and docstrings for public APIs; keep functions focused and small.

## Testing Guidelines
- Primary framework: `uv run pytest`.
- For quick sanity checks, run a small eval with `--limit` and `--batch_size 1`.
- Add regression tests for bug fixes; keep tests minimal and fast.

## Commit & Pull Request Guidelines
- Use short, sentence-style commit messages.
- Add trailers when applicable: `Reported-by:<name>` or `Github-Issue:#<num>` (see `CLAUDE.md`).
- PRs should explain the problem, solution, and key commands/results.

## Configuration & Runtime Tips
- Compression is controlled via env vars (e.g., `COMPRESSOR`, `R_RATIO`, `COMPRESS_IMAGE`, method-specific knobs).
- Keep changes scoped to the task; avoid unrelated refactors.
