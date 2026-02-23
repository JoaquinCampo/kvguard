.PHONY: check fmt lint typecheck test

check: fmt lint typecheck test

fmt:
	uv run ruff format src/ tests/

lint:
	uv run ruff check src/ tests/

typecheck:
	uv run mypy src/

test:
	uv run pytest -x -q
