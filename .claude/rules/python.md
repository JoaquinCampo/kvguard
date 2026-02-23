---
paths:
  - "src/**/*.py"
---

# Python Source Conventions

- Python 3.12+. Use native union syntax `X | None`, native generics `list[str]`, `dict[str, Any]`.
- No `from __future__ import annotations`. No `typing.Optional`, `typing.List`, etc.
- Pydantic models for structured data. Keep flat and simple — no nested validators.
- Functions over classes. Use classes only when state management is genuinely needed.
- Line length: 100 characters (ruff enforces).
- Flat package: all modules directly under `src/kvguard/`. No subpackages.
- Imports sorted by ruff (isort-compatible).
- Type annotations on all public functions. `mypy --strict` must pass.
- When adding Alembic migrations with foreign keys: add column first, then `op.create_foreign_key()` separately. In downgrade: `op.drop_constraint()` before `op.drop_column()`.
