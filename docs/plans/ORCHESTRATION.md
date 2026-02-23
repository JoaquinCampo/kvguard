# KVGuard v2 Phase 1: Agent Team Orchestration

> **For a fresh Claude Code session:** Read this file first. It tells you what to do, how to parallelize, and where all context lives.

## What happened

A systematic audit found 48 issues in the kvguard codebase (see `VALIDATION_REPORT.md` at repo root). The detailed implementation plan is at `docs/plans/2026-02-22-v2-validation-fixes.md`. The higher-level multi-phase roadmap is at `docs/v2-plan.md`.

Phase 1 is **CPU-only code fixes** — no GPU needed. 16 tasks. Many are parallelizable.

## Essential context files

Every agent should read these before starting any task:

| File | Why |
|------|-----|
| `CLAUDE.md` | Project conventions (no `__future__` annotations, Python 3.12+, `make check`) |
| `VALIDATION_REPORT.md` | Full issue catalog — every task references issues by number |
| `docs/plans/2026-02-22-v2-validation-fixes.md` | Detailed per-task implementation steps with exact code |

## Quality gate

Every agent must run before considering a task done:

```bash
cd /Users/joaquincamponario/Documents/INCO/RESEARCH/kvguard
make check   # runs: ruff format + ruff check + mypy + pytest
```

## The 16 tasks

### Group A — Independent (can all run in parallel)

| Task | Summary | Files to modify | Files to test | Est. |
|------|---------|----------------|---------------|------|
| **T2** | Remove `rank_of_chosen` from features (always 0 under greedy) | `features.py:24-36,65-93`, `controller.py:44-59,120-161` | `test_features.py`, `test_controller.py` | 10m |
| **T3** | Fix `token_position` to use `max_new_tokens` denominator (prevents future leakage) | `features.py:135-172` | `test_features.py` | 5m |
| **T6** | Fix GSM8K answer extraction: `re.search` → `re.findall[-1]` (take last `####`) | `detectors.py:79-96` | `test_detectors.py` | 5m |
| **T9** | Remove `from __future__ import annotations` from 6 source files | `controller.py`, `features.py`, `train.py`, `evaluate_controller.py`, `verify.py`, `anchors.py` | `make check` | 3m |
| **T10** | Remove unused `wandb` dependency from `pyproject.toml` | `pyproject.toml:21` | `uv sync && make check` | 2m |
| **T14** | Add `numpy` as explicit dependency in `pyproject.toml` | `pyproject.toml` | `uv sync && make check` | 2m |
| **T15** | Add `model: str = ""` field to `TraceMeta`, populate in `build_dataset()` | `features.py:180-190,250-287` | `test_features.py` | 5m |
| **T16** | Add CUDA cleanup alongside MPS in experiment runner | `experiment.py:296-299,453-455` | N/A (no GPU tests) | 3m |

### Group B — Requires T2 first (feature schema change)

| Task | Summary | Files to modify | Files to test | Est. |
|------|---------|----------------|---------------|------|
| **T1** | Rewrite `split_by_trace()` → `split_by_prompt()` (prompt-level split) | `train.py:63-119,284-334,~395` | `test_train.py` | 15m |
| **T4** | Remove RECOVERY transitions + delete `anchors.py` (dead code) | `controller.py:30-36,169-209,212-332`, `evaluate_controller.py:137-211`, delete `anchors.py` + `test_anchors.py` | `test_controller.py` | 15m |
| **T12** | Add note that `delta_h_valid` is near-constant | `features.py:24-36` | N/A | 1m |

### Group C — Requires T1 (uses split_by_prompt)

| Task | Summary | Files to modify | Files to test | Est. |
|------|---------|----------------|---------------|------|
| **T5** | Add `Dataset.drop_features()` + feature ablation infra | `features.py:193-201`, `run_ablations.py` | `test_train.py` | 10m |
| **T7** | Add `holdout_prompt_ids` to controller eval + save `split_info.json` | `evaluate_controller.py:331-445`, `train.py:353-429`, `__init__.py` | `test_evaluate_controller.py` (new) | 15m |

### Group D — Requires T4 (RECOVERY removed)

| Task | Summary | Files to modify | Files to test | Est. |
|------|---------|----------------|---------------|------|
| **T8** | Remove 8 unused imports from `evaluate_controller.py` | `evaluate_controller.py:13-36` | `make check` | 3m |
| **T11** | Fix `ControllerConfig` defaults to match paper's Balanced config | `controller.py:77-86`, `__init__.py:136-185` | `test_controller.py` | 5m |
| **T13** | Add cross-validation test: both state machine implementations agree | `test_evaluate_controller.py` (new) | itself | 10m |

## Recommended team structure

### Option A: 3 agents (recommended)

```
Agent 1 ("features"): T2 → T3 → T15 → T12 → T5
Agent 2 ("controller"): T4 → T8 → T11 → T13
Agent 3 ("pipeline"): T6 → T1 → T7 → T16

Leader: T9, T10, T14 (trivial), then merge + verify
```

### Option B: 4 agents (max parallelism)

```
Agent 1 ("features"): T2 → T3 → T15 → T12
Agent 2 ("controller"): T4 → T8 → T11 → T13
Agent 3 ("splits"): T1 → T5 → T7
Agent 4 ("fixes"): T6 → T9 → T10 → T14 → T16

Leader: merge + final make check
```

## Critical implementation details (read before coding)

### T2: rank_of_chosen removal
- Remove from `BASE_FEATURE_NAMES` list (features.py:28) — the line `"rank_of_chosen",`
- Remove `float(sig["rank_of_chosen"]),` from `flatten_token()` (features.py:83)
- N_BASE goes 30→29, total features 41→40
- Remove from `_DEFAULT_WEIGHTS` and `_DEFAULT_NORMS` in controller.py
- Remove rank variable from `compute_risk_score()` (lines 148, 156)
- **Keep** `rank_of_chosen` in `TokenSignals` (config.py) and `extract_signals()` (signals.py) — still recorded in JSON traces, just not an ML feature

### T1: prompt-level split
- Rewrite the **internals** of `split_by_trace()` — keep the same function name and signature
- Group by `prompt_id` from `ds.traces`, not by `trace_idx`
- A prompt "has catastrophe" if **any** of its traces do
- All trace_idxs for a prompt go to the same partition
- `_make_synthetic_dataset()` in test_train.py needs updating: create shared prompt_ids across presses (e.g., 10 prompts × 3 presses × 3 ratios = 90 traces)
- For `leave_one_out_cv()`: prompt overlap across compressor folds is **intentional** (testing compressor generalization). Add docstring noting this. No code change.

### T4: RECOVERY removal
- **Keep** `Mode.RECOVERY = 3` in the enum to avoid breaking serialized mode_history data in existing traces
- Make RECOVERY **unreachable**: remove SAFE→RECOVERY escalation from `_decide_mode()` (lines 198-199) and RECOVERY→SAFE de-escalation (lines 202-203)
- Same in `_run_predictor_state_machine()` (evaluate_controller.py lines 188-189, 192)
- Add docstring: "RECOVERY mode is reserved for future KV recomputation; not reachable via current state machine."
- Delete `src/kvguard/anchors.py` and `tests/test_anchors.py`
- Remove `rep_count_recovery` from `ControllerConfig` or default to large sentinel value

### T3: token_position fix
- Change `add_rolling_features()` signature: add `max_new_tokens: int = 512` parameter
- Replace `np.arange(n) / (n - 1)` with `np.arange(n) / max(max_new_tokens, 1)`
- Update `build_dataset()` to pass `run.max_new_tokens` to `add_rolling_features()`

### T6: answer extraction
- Change `re.search(r"####\s*([\d,.\-]+)", text)` → `re.findall(r"####\s*([\d,.\-]+)", text)` then take `matches[-1]`
- Same for `\\boxed{}` pattern

### T7: controller holdout
- Add `filter_traces_by_prompts()` function to evaluate_controller.py
- Add `holdout_prompt_ids: set[str] | None = None` param to `evaluate_controller()`
- In `run_training()`, save `split_info.json` with val/train prompt_id lists
- In CLI `eval_controller` command, load split_info.json and pass holdout set

### T11: config defaults
- `tau_high`: 0.6 → 0.7
- `k_escalate`: 3 → 8
- `safe_compression_ratio`: 0.5 → 0.0
- Update CLI defaults in `__init__.py` to match

## Merge strategy

All agents work in **worktrees** branching from the same commit. After all agents finish:

1. Merge agent branches one at a time, resolving any conflicts in features.py (most likely conflict point since T2, T3, T15, T12 all touch it)
2. Run `make check` after each merge
3. Final `make check` on the merged result

## What comes after Phase 1

Phase 1 produces a **clean codebase** but **no new results**. The existing model and metrics are invalidated by the feature schema change and split fix. Phases 2-5 (in `docs/v2-plan.md`) require GPU:

- Phase 2: Run sweeps on 500 prompts × 2 models (Qwen 7B + Llama 8B) — needs 5090
- Phase 3: Train, evaluate, ablate with new data
- Phase 4: Live token-by-token validation (stretch)
- Phase 5: Paper revision
