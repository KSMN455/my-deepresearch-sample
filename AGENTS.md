# Repository Guidelines

## Project Structure & Module Organization
This repo centers on `main.py`, a Flask-powered Deep Research chat UI and agent pipeline. Keep
route handlers minimal and delegate branching logic to helpers so the four-stage agent flow remains
easy to test. The `tests/` directory holds pytest cases such as `tests/test_sample.py` that assert API
key loading, request validation, and HTTP responses. Project configuration lives in `pyproject.toml`
and the locked dependency graph in `uv.lock`. Runtime secrets belong in environment variables or
`openai_api_key.txt`; never hardcode them in source.

## Build, Test, and Development Commands
Use uv for reproducible environments: `uv sync` installs dependencies, and `uv run python main.py`
starts the Flask UI at <http://127.0.0.1:5000/>. Execute tests with `uv run pytest -q`. When uv is not
available, create a venv via `python -m venv .venv && source .venv/bin/activate` then `pip install -e .`. Lint with `ruff check .`, format via `ruff format .`, and optionally type-check using `uv run mypy .`.

## Coding Style & Naming Conventions
Target Python 3.13, four-space indentation, and a 100-character line limit. Prefer double quotes and
ruff-managed import ordering. Functions and modules use `snake_case`; classes `PascalCase`; constants
`UPPER_SNAKE_CASE`. Keep Flask views slim, return explicit values, and add type hints where they aid
clarity. Centralize configuration in `pyproject.toml` or environment variables instead of ad-hoc
module globals.

## Testing Guidelines
Write pytest suites under `tests/` using `test_*.py` files and descriptive `test_*` functions. Mock
OpenAI calls with fixtures or `monkeypatch` to avoid external traffic. Cover API key discovery, input
sanitization, and the multi-agent handoff through `_triage_agent`, `_clarifier_agent`, `_instruction_builder_agent`, and `_research_agent`. Run `uv run pytest -q` before raising a PR.

## Commit & Pull Request Guidelines
Use Conventional Commit subjects (e.g., `feat: add clarifier retries`). Scope each commit to a single
concern and include matching tests or docs. PRs should describe the change, link issues, list the
commands you ran (tests, lint, format), and attach screenshots when the UI shifts. Confirm `ruff`
checks, formatting, and pytest all succeed locally.

## Security & Configuration Tips
Set `OPENAI_API_KEY` via environment variables or `openai_api_key.txt`, which is ignored by Git. Review
`DEEP_RESEARCH_TOOLS` JSON before enabling new integrations. Avoid committing secrets, production
logs, or raw API payloads; scrub test fixtures and redact run logs.
