# Repository Guidelines
用中文回复
## Project Structure & Module Organization
- `lmms_eval/`: core evaluation harness, CLI entrypoint, models, tasks, and metrics.
- `token_compressor/`: VidCom2 implementation plus baseline compressors and model adapters.
- `examples/`: runnable scripts (e.g., `examples/models/*.sh`) and usage templates.
- `docs/`: command reference, model/task guides, and versioned docs.
- `tools/` and `miscs/`: dataset utilities, experiments, and ad-hoc scripts (see `miscs/test_*.py`).

## Build, Test, and Development Commands
- `pip install -e .` installs the package in editable mode (per `README.md`).
- `python -m lmms_eval --help` shows CLI flags; `lmms-eval` is the console entrypoint.
- Example run (short sanity):  
  `python -m lmms_eval --model qwen3_vl --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct --tasks videomme --batch_size 1 --limit 8 --output_path ./logs`
- `accelerate launch -m lmms_eval ...` is used for multi-GPU runs (see `README.md` examples).

## Coding Style & Naming Conventions
- Python project; follow PEP 8 naming (snake_case functions, PascalCase classes).
- Formatting: Black with `--line-length=240`, import sorting via isort (`--profile black`) per `.pre-commit-config.yaml`.
- Prefer type hints and docstrings for public APIs; keep functions focused and small (see `CLAUDE.md`).

## Testing Guidelines
- There is no dedicated `tests/` tree; lightweight checks live in `miscs/test_*.py`.
- For fast regressions, run an eval with `--limit` and a small batch size.
- If you add tests, keep them minimal and runnable via `pytest` (`uv run pytest` per `CLAUDE.md`).

## Commit & Pull Request Guidelines
- Recent commits use short, sentence-style messages without prefixes or ticket IDs (e.g., “Support FastV...”).
- If your work is tied to a user report or GitHub issue, add trailers like `Reported-by:` or `Github-Issue:#` (per `CLAUDE.md`).
- PRs should explain the problem and the high-level solution; include key commands or results when relevant.

## Agent-Specific Notes
- `CLAUDE.md` contains stricter workflow rules (uv-first tooling, type checks, linting). Follow it when working in that toolchain.
