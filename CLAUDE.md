# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development
- **Run the application**: `uv run python main.py`
- **Install dependencies**: `uv sync`
- **Run tests**: `uv run pytest -q`
- **Run linting**: `uv run ruff check`
- **Run formatting**: `uv run ruff format`
- **Type checking**: `uv run mypy main.py` (mypy is in dev dependencies)

### Environment Setup
- OpenAI API Key must be set via `OPENAI_API_KEY` environment variable or placed in `openai_api_key.txt` file
- Optional: Configure `DEEP_RESEARCH_TOOLS` as JSON string (defaults to `[{"type": "web_search_preview"}]`)
- Optional: Set model overrides via `TRIAGE_MODEL`, `CLARIFIER_MODEL`, `INSTRUCTION_MODEL`, `RESEARCH_MODEL`
- Optional: Set `OPENAI_MAX_RETRIES` for API retry behavior

## Architecture

This is a Flask-based web application that demonstrates OpenAI's Deep Research API (`o4-mini-deep-research`) through a simple chat interface.

### Four-Stage Agent Pipeline
The core research flow follows a four-stage pipeline in `main.py`:
1. **Triage Agent** (`_triage_agent`): Analyzes input to determine if it's ready for research or needs clarification
2. **Clarifier Agent** (`_clarifier_agent`): Asks follow-up questions when the input is insufficient
3. **Instruction Builder Agent** (`_instruction_builder_agent`): Creates structured research briefs from complete inputs
4. **Research Agent** (`_research_agent`): Executes the actual deep research using the OpenAI Deep Research API

### Key Components
- **`run_deep_research()`**: Main entry point that orchestrates the agent pipeline
- **`get_openai_client()`**: LRU-cached OpenAI client factory with API key management
- **Flask Routes**:
  - `/`: Serves the chat UI (single-page HTML template)
  - `/chat`: JSON API endpoint that accepts conversation history
  - `/healthz`: Health check endpoint

### Data Structures
- **`InstructionBrief`**: TypedDict containing structured research instructions
- **`TraceLog`**: List of trace entries for debugging the agent pipeline
- Message format follows OpenAI's input structure with role/content patterns

### OpenAI Integration
- Uses the latest OpenAI Python SDK with the `client.responses.create()` API
- Supports streaming responses for the Deep Research model
- Includes rate limiting awareness and error handling for OpenAI API errors
- Tools integration for web search capabilities (`web_search_preview` by default)

### Testing
Tests in `tests/test_app.py` use pytest with monkeypatching to mock OpenAI API calls and validate the HTTP endpoints and core functionality.