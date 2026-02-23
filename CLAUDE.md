# KVGuard

Catastrophe-aware KV-cache compression controller. Wraps any kvpress compressor, monitors per-token logit signals during generation, and dynamically relaxes compression when catastrophic failure is imminent.

**Thesis**: Heavy KV-cache compression causes dramatic, detectable failures (looping, non-termination, instruction amnesia). A trained hazard predictor watching logit features can prevent these before they manifest.

## Architecture

@docs/architecture.md

Single flat package in `src/kvguard/` — no subpackages.

**Key modules**: config.py (Pydantic models) → prompts.py (GSM8K loading) → signals.py (per-token extraction) → detectors.py (catastrophe detection) → experiment.py (generation + sweep runner) → features.py (ML feature engineering) → labeling.py (hazard labels) → train.py (XGBoost training) → controller.py (3-mode state machine) → evaluate_controller.py (offline simulation)

**Data flow**: GSM8K → format_prompt → model.generate(output_scores=True) inside press(model) context → per-token extract_signals → detect_all → RunResult → JSON

**CLI entry point**: `kvguard` via Typer (6 commands: run, sweep, analyze, verify, train, eval-controller)

## Research context

@docs/findings.md
@docs/v2-plan.md
@docs/related-work-positioning.md

Current status: Phase 1 code fixes complete, Phase 2 GPU sweep scripts ready. Awaiting 5090 server for 2-model × 500-prompt × 16-config sweep (~48-60h GPU time).

## Commands

```bash
make check              # REQUIRED before commits: fmt + lint + typecheck + test
make fmt                # ruff format src/ tests/
make lint               # ruff check src/ tests/
make typecheck          # mypy src/ (strict mode)
make test               # pytest -x -q
make phase2             # Full GPU sweep pipeline
make status             # Check sweep progress
```

**Single experiment**: `uv run kvguard run --press streaming_llm --compression-ratio 0.875 --num-prompts 10`
**Full sweep**: `uv run kvguard sweep --num-prompts 500 --model "Qwen/Qwen2.5-7B-Instruct"`
**Train predictor**: `uv run kvguard train --results-dir results --num-prompts 500`
**Evaluate controller**: `uv run kvguard eval-controller --results-dir results --model-path models/hazard_predictor.json`

## Conventions

- Python 3.12+. No `from __future__ import annotations`.
- `list`, `dict`, `tuple` — not `typing.List`, `typing.Dict`, `typing.Tuple`.
- `X | None` — not `Optional[X]`.
- Pydantic models for structured data. Keep flat and simple.
- Functions over classes. Classes only when state management is genuinely needed.
- Line length: 100 characters (ruff enforces).
- Imports sorted by ruff.
- Flat package structure — no subpackages under `src/kvguard/`.

## Boundaries

### Always safe (no approval needed)
- Run `make check`, `pytest`, `ruff`, `mypy`
- Read any file in the repo
- Edit source files in `src/` and `tests/`
- Run `uv run kvguard` commands

### Ask first
- Modify `pyproject.toml` (dependency changes affect all branches)
- Push to `main` (use branches and PRs)
- Add new dependencies
- Modify sweep scripts that affect GPU runs

### Never
- Commit `results/` data (gitignored for a reason)
- Create subpackages under `src/kvguard/`
- Add `from __future__ import annotations`

## Known failures and gotchas

@docs/failures.md

## Key references

- `docs/decisions.jsonl` — decision log (append-only JSONL, 8 entries)
- `docs/experiments/001-reproduce-failures.md` — Milestone A results
- `docs/analysis/002-sota-to-experiment-mapping.md` — SOTA vs experimental evidence
- `models/hazard_predictor.json` — trained XGBoost (invalidated by Phase 1 fixes, retrain after Phase 2)
