# KVGuard

Catastrophe-aware KV-cache compression controller for reasoning models.

## Setup

```bash
# Install all dependencies (including dev tools)
uv sync --group dev

# Verify everything works
make check
```

## Usage

```bash
# Run a single experiment
uv run kvguard run --press streaming_llm --compression-ratio 0.875 --num-prompts 10

# Run the full sweep (baseline + 5 ratios x 3 methods)
uv run kvguard sweep --num-prompts 50

# Analyze results
uv run kvguard analyze --num-prompts 50
```

## Quality gates

```bash
make check    # runs all checks: fmt, lint, typecheck, test
make fmt      # ruff format
make lint     # ruff check
make typecheck  # mypy
make test     # pytest
```

All gates must pass before committing. See `CLAUDE.md` for full conventions.
