.PHONY: check fmt lint typecheck test setup-server download-models phase2 status

check: fmt lint typecheck test

fmt:
	uv run ruff format src/ tests/

lint:
	uv run ruff check src/ tests/

typecheck:
	uv run mypy src/

test:
	uv run pytest -x -q

# Phase 2 targets
setup-server:
	bash scripts/setup_server.sh

download-models:
	bash scripts/download_models.sh

phase2:
	bash scripts/run_phase2.sh

status:
	bash scripts/check_status.sh
