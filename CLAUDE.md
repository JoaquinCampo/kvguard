# Technologies

- `from __future__ import annotations` is NOT used — we target Python 3.12+.
- Prefer `list`, `dict`, `tuple` over `typing.List`, `typing.Dict`, `typing.Tuple`.
- Use `X | None` instead of `Optional[X]`.
- Pydantic models for structured data. Keep them flat and simple.
- No fancy logic in Pydantic validators unless absolutely necessary.
- Functions over classes. Use classes only when state management is genuinely needed.
- Keep imports sorted (ruff handles this).
- Line length: 100 characters (configured in `pyproject.toml`).

## What NOT to do

- Don't modify `pyproject.toml` without coordinating — dependency changes affect all branches.
- Don't commit results/ data — it's gitignored for a reason.
- Don't push to `main` directly — use branches and PRs.
- Don't add heavy dependencies without justification.
- Don't create subpackages under `src/kvguard/` — keep the flat structure.

## Worktree setup

Each worktree gets its own virtualenv. After the worktree is created:

```bash
uv sync --group dev
```

This installs all dependencies including dev tools (pytest, ruff, mypy).

## Key references (inside this repo)

- `docs/architecture.md` — module details and data flow
- `docs/experiments/001-reproduce-failures.md` — Milestone A results
- `docs/analysis/002-sota-to-experiment-mapping.md` — SOTA predictions vs experimental evidence
- `docs/findings.md` — plain-English summary of what we've found
- `docs/decisions.jsonl` — decision log (append-only)


## Quality gates

Before committing, run:

```bash
make check
```

This runs: `ruff format` + `ruff check` + `mypy` + `pytest`. All must pass.
