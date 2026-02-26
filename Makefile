.PHONY: check fmt lint typecheck test setup-server download-models phase2 status paper figures ablations

check: fmt lint typecheck test

fmt:
	uv run ruff format src/ tests/ scripts/

lint:
	uv run ruff check src/ tests/ scripts/

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

# Paper targets
paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex

figures:
	uv run python scripts/generate_figures.py

ablations:
	uv run python scripts/run_ablations.py
