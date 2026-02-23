# KVGuard Systematic Validation Report

**Date:** 2026-02-22
**Scope:** Full codebase + paper audit across 6 dimensions: data pipeline, features, controller, experiments, tests/dead code, statistical claims.

**Triage update:** 2026-02-22 — Pre-Phase-2 audit fixes applied. Status tags added below.

Status legend:
- **FIXED** — resolved in this audit round
- **PREVIOUSLY FIXED** — resolved in Phase 1
- **PHASE 2 (GPU)** — requires GPU sweep data to address
- **PHASE 3 (paper)** — paper-level correction, no code change
- **WON'T FIX** — accepted limitation or by-design

---

## TIER 1: Results-Invalidating Issues

These issues mean the headline numbers in the paper may not survive correction.

### 1.1 Controller evaluated on predictor's training data — **PREVIOUSLY FIXED**
**Files:** `evaluate_controller.py`, `train.py`
**Impact:** ALL controller numbers (43.2% CFR reduction, 63.9% excl. extreme, 6.9% FP rate)

`evaluate_controller()` accepts `holdout_prompt_ids` to restrict evaluation to held-out prompts. `run_training()` saves `split_info.json` with val prompt IDs. Phase 2 GPU sweeps will use this properly.

### 1.2 Threshold selection on test data — **PHASE 2 (GPU)**
**Files:** `scripts/run_ablations.py`, `evaluate_controller.py`
**Impact:** The "balanced" operating point (tau_low=0.3, tau_high=0.7, k=8)

Must re-run ablation sweep on held-out prompts only. Now possible with `holdout_prompt_ids` parameter.

### 1.3 Prompt-level data leakage in train/test split — **PREVIOUSLY FIXED**
**Files:** `train.py` (`split_by_prompt`)
**Impact:** AUROC 0.945, all predictor metrics

`split_by_prompt()` was added and is now the default split in `run_training()`. All configs for a prompt go to the same partition. Edge cases (1 positive prompt, all positive) now handled with warnings.

### 1.4 Same leakage in leave-one-compressor-out CV — **WON'T FIX**
**Files:** `train.py` (`leave_one_out_cv`)
**Impact:** CV mean AUROC 0.898

By design: CV tests cross-compressor generalization, not cross-prompt. The main split now uses `split_by_prompt()` for prompt-level isolation. CV is supplementary.

### 1.5 Token position feature leaks future information — **PREVIOUSLY FIXED**
**Files:** `features.py`
**Impact:** AUROC and controller performance

Now uses `t / max_new_tokens` (known at inference time) instead of `t / (n - 1)`.

### 1.6 Labels include post-onset tokens, inflating predictor metrics — **FIXED**
**Files:** `labeling.py`, `features.py`, `train.py`
**Impact:** AUROC 0.945, recall numbers

Added `compute_onset_position()`, `Dataset.onset_positions`, and pre-onset evaluation metrics (`pre_onset_recall`, `pre_onset_auroc`) to `EvalMetrics`. `run_training()` now computes and saves pre-onset metrics separately in `metrics.json`. Post-onset tokens are still labeled (needed for training) but eval now distinguishes pre- vs post-onset performance.

---

## TIER 2: Methodology Issues

These don't necessarily invalidate results but represent flawed or misleading methodology.

### 2.1 Offline simulation is an upper bound, not an estimate
**Files:** `evaluate_controller.py:410-416`
**Impact:** All CFR reduction numbers

Prevention is counted when: (a) controller triggers before onset, AND (b) the same prompt at r_safe=0.0 succeeds. This assumes switching mid-generation perfectly recovers baseline quality. In reality, tokens already generated under aggressive compression may have irreversibly corrupted the reasoning state. The paper acknowledges this but buries it in §3.7 and §4.6 rather than qualifying the headline numbers.

### 2.2 CFR metric mixes ordinary wrong answers with catastrophic behaviors
**Impact:** The narrative "average metrics hide catastrophes"

CFR includes wrong_answer alongside looping and non-termination. But wrong_answer is *exactly* what accuracy measures — it's not hidden by average metrics. Only looping and non-termination are the genuinely hidden catastrophic *behaviors*. The paper's framing ("average metrics conceal a dangerous failure pattern") is overstated because the metric includes the thing average metrics already capture.

### 2.3 FP rate denominator changes silently between tables
**Files:** Paper Tables 4 and 5
**Impact:** Reported FP rates (6.9% vs 8.1%)

Table 4 reports FP rate as 6.9% (52/750 total prompts). Table 5 reports 8.1% (52/639 non-catastrophe prompts). Same balanced config, same 52 false triggers, different denominators. The main narrative uses 6.9% — the more flattering number with the less standard denominator.

### 2.4 No confidence intervals on any reported metric
**Impact:** All per-config claims

With n=50 prompts per configuration, a CFR of 34% (17/50) has a 95% binomial CI of approximately [21%, 49%]. The difference between "43.2% CFR reduction" and "47.7% CFR reduction" (balanced vs aggressive) could easily be noise. No statistical significance tests, no bootstrap CIs, no error bars anywhere.

### 2.5 Train/eval onset mismatch for non-termination — **FIXED**
**Files:** `evaluate_controller.py`
**Impact:** Non-termination prevention numbers

`TraceInfo.cfr_onset` now uses `int(nt_onset_frac * max_new_tokens)` for non_termination, matching the proxy used in labeling. `nt_onset_frac` is threaded through `load_all_traces()` and `evaluate_controller()`.

### 2.6 "none" baseline fold inflates CV mean
**Files:** `train.py:302`, CV results in `models/metrics.json`
**Impact:** CV mean AUROC 0.898

The "none" (uncompressed) baseline fold has AUROC 0.970 with only 320 positive tokens out of 15,932 (2% positive rate). This is an easy classification task that inflates the CV mean. The predictor is never used on uncompressed traces in practice — this fold tests a scenario that doesn't arise at deployment.

---

## TIER 3: Feature & Predictor Issues

### 3.1 compression_ratio makes the predictor a partial lookup table
**Files:** `features.py:169-170`
**Impact:** Feature importance claims, AUROC

compression_ratio is the #1 feature by XGBoost gain (1645) and is a static per-trace constant taking one of 6 values. The predictor partially learns "ratio 0.875 = danger." No ablation tests performance without this feature. The "zero-cost logit features predict catastrophe" narrative is unvalidated — we don't know how much AUROC comes from logit features vs. the compression ratio lookup.

### 3.2 rep_count is detection, not forecasting
**Files:** `signals.py:49-74`
**Impact:** "Forecast catastrophe within the next 32 tokens" claim

rep_count spikes at/after loop onset, not before. For the pre-onset hazard window (tokens t*-32 to t*-1), rep_count is typically 0 because the loop hasn't started yet. rep_count is the 2nd-3rd most important feature — meaning the predictor partly succeeds by detecting loops already in progress (post-onset tokens labeled 1 where rep_count > 0), not by forecasting them.

### 3.3 rank_of_chosen is always 0 — dead feature — **PREVIOUSLY FIXED**
**Files:** `features.py`
**Impact:** 1 of 41 features is a constant zero

`rank_of_chosen` was removed from `BASE_FEATURE_NAMES` in Phase 1. Feature count is now 29 base (was 30).

### 3.4 Feature count is 41, not 42 — **PREVIOUSLY FIXED → now 40**
**Files:** `features.py`
**Impact:** Paper claims "42-dimensional"

After removing `rank_of_chosen`: 29 base + 8 rolling + 1 rep_count_sum + 1 token_position + 1 compression_ratio = **40**. Paper needs updating. — **PHASE 3 (paper)**

### 3.5 h_alts description in paper is wrong
**Files:** `signals.py:118-127`
**Impact:** Paper §3.3 and §2.3

Paper says: "h_alts: average log-probability of the top-20 alternatives." Code computes: entropy of the probability distribution excluding the top-1 token (renormalized). These are different quantities. The code computes competitor entropy, not average log-probability.

### 3.6 logprob_0 is redundant with top1_prob
**Files:** `signals.py:108-113`
**Impact:** Minor (XGBoost handles it, but it's 2 features encoding the same information)

`logprob_0 = log(top1_prob)` by construction since top-20 are sorted by probability.

### 3.7 delta_h_valid is near-constant
**Files:** `features.py:74-76`
**Impact:** Minor (1 wasted feature dimension)

This is 0 for exactly the first token of each trace and 1 for all others. Effectively constant.

### 3.8 Missing feature ablations — **PREVIOUSLY FIXED**
**Files:** `scripts/run_ablations.py`
**Impact:** Cannot validate which features actually matter

`ablation_feature_importance()` was added in Phase 1, with `FEATURE_ABLATIONS` dict covering: full, no_compression_ratio, no_rep, logit_only. Uses `Dataset.drop_features()` to remove columns before retraining.

---

## TIER 4: Dead Code & Paper Claims for Unimplemented Features

### 4.1 Anchor-aware token protection: described in paper, never used
**Files:** `anchors.py` (274 lines), paper §3.6
**Impact:** Paper §3.6 describes a feature that doesn't exist in the system

anchors.py is fully implemented and tested (96 lines of tests) but never imported by experiment.py, evaluate_controller.py, or any production code. The paper's §3.6 describes receiver head calibration, vertical attention scores, and runtime anchor identification as part of the system. None of this is wired in. The controller's `protect_thinking_tokens` flag is set based on mode alone — no actual tokens are identified or protected.

### 4.2 RECOVERY mode: defined but functionally identical to SAFE
**Files:** `controller.py:198-199, 309, 331-332`
**Impact:** Paper claims 4-mode state machine; only 3 modes are operative

RECOVERY sets `trigger_recomputation=True` and `compression_ratio=0.0`. But nothing consumes `trigger_recomputation`. Since `safe_compression_ratio` defaults to 0.0 in evaluation, RECOVERY and SAFE produce the same behavior. The paper describes "selective KV recomputation" — this is never implemented.

### 4.3 RiskController class unused in evaluation
**Files:** `controller.py:212-333`
**Impact:** The "main" controller API is dead code

evaluate_controller.py reimplements the state machine in `_run_predictor_state_machine()` rather than using `RiskController.step()`. Two implementations of the same logic with no test verifying they produce identical results.

### 4.4 compute_risk_score() unused in evaluation
**Files:** `controller.py:120-161`
**Impact:** Rule-based risk scoring is dead code

The evaluation uses XGBoost output directly. The hand-crafted risk score function (with weights for entropy, h_alts, delta_h, rank_of_chosen, rep_count) is never called.

---

## TIER 5: Code Quality & Robustness Issues

### 5.1 GSM8K answer extraction takes first #### match, not last
**Files:** `detectors.py:87`
**Impact:** Correct answers could be marked wrong

`re.search` finds the first `####` in the output. Standard GSM8K evaluation uses the last `####`. If the model mentions `####` in intermediate reasoning, the first (wrong) value is extracted.

### 5.2 Failed prompts silently dropped
**Files:** `experiment.py:278-282`
**Impact:** Potential accuracy bias

Prompts that crash are skipped with `continue`. Accuracy is computed over successful prompts only. If hard prompts systematically crash (e.g., OOM on long sequences), accuracy is biased upward.

### 5.3 No early stopping in XGBoost — **FIXED**
**Files:** `train.py`
**Impact:** Moderate overfitting (train AUROC 0.984 vs val 0.945)

`train_predictor()` now accepts optional `X_val`/`y_val` and uses `early_stopping_rounds=20` when provided. `run_training()` passes validation data through automatically.

### 5.4 Checkpoint writes are not atomic — **WON'T FIX (documented)**
**Files:** `experiment.py`
**Impact:** Crash during write corrupts checkpoint, blocking resume

Non-atomicity documented in code comment. Acceptable for research prototype.

### 5.5 MPS memory not cleaned between model groups in Python sweep
**Files:** `experiment.py:453-455`
**Impact:** Only checks for CUDA, not MPS. MPS memory leaks between attention groups.

### 5.6 No verification that compression actually occurred
**Files:** `experiment.py:131-132, 199`
**Impact:** If kvpress silently fails, results are mislabeled.

### 5.7 Timeout timer starts at object construction, not generation start
**Files:** `experiment.py:98-111`
**Impact:** Model loading / compression setup time counts against the 300s timeout.

### 5.8 Controller default config matches NO paper operating point
**Files:** `controller.py:77-86`, `__init__.py:144`
**Impact:** Anyone using default `ControllerConfig()` or CLI gets tau_high=0.6, k_escalate=3, safe_ratio=0.5 — none of these match Conservative, Balanced, or Aggressive.

### 5.9 Weak stratification test
**Files:** `tests/test_train.py:205-212`
**Impact:** Test asserts `train_has_pos or val_has_pos` instead of `and`. Doesn't actually verify stratification.

### 5.10 No tests for evaluate_controller.py — **FIXED**
**Impact:** The module producing all headline results has zero test coverage.

Added `TestTraceInfo`, `TestCfrOnsetProxy`, `TestSimulateControllerOnTrace`, `TestFormatEvalTable`, `test_safe_key_missing_warns`, and state machine tests in `tests/test_evaluate_controller.py`.

### 5.11 No tests for analyze.py, run_ablations.py, prompts.py
**Impact:** Analysis and ablation scripts are untested.

### 5.12 wandb dependency declared but never imported
**Files:** `pyproject.toml:21`

### 5.13 6+ unused imports in evaluate_controller.py — **PREVIOUSLY FIXED**
Unused imports cleaned up in Phase 1.

### 5.14 `from __future__ import annotations` in 6 files despite project convention against it — **FIXED**
All `from __future__` imports removed from src/ and scripts/.

### 5.15 Duplicated test helpers between test_features.py and test_train.py
**Files:** `tests/test_features.py:26-111`, `tests/test_train.py:27-110`

---

## TIER 6: Paper-Specific Inconsistencies

### 6.1 42 vs 41 dimensions — abstract/intro/conclusion say 42, §4.2 says 41
### 6.2 §4.2 lists "25 HALT logit features" but base features are 30 (includes non-HALT items)
### 6.3 FP rate: Table 4 says 6.9%, Table 5 says 8.1% for same config
### 6.4 h_alts described as "average log-probability" — code computes entropy
### 6.5 §3.6 describes anchor protection as implemented — it's not wired in
### 6.6 No figures in a 12-page paper
### 6.7 §5.1 repeats §4.5, §5.4 repeats §4.6 almost verbatim
### 6.8 Formal notation (π, s_t, r_t) in §3.1 is introduced and never used again
### 6.9 Only 15 citations for a 12-page paper

---

## Summary Statistics

| Tier | Total | Fixed/Prev-Fixed | Phase 2 (GPU) | Phase 3 (paper) | Won't Fix | Open |
|------|-------|-------------------|---------------|-----------------|-----------|------|
| 1 — Results-invalidating | 6 | 4 | 1 | 0 | 1 | 0 |
| 2 — Methodology | 6 | 1 | 0 | 4 | 0 | 1 |
| 3 — Features/Predictor | 8 | 3 | 0 | 1 | 0 | 4 |
| 4 — Dead code/Unimplemented | 4 | 0 | 0 | 4 | 0 | 0 |
| 5 — Code quality | 15 | 4 | 0 | 0 | 1 | 10 |
| 6 — Paper inconsistencies | 9 | 0 | 0 | 9 | 0 | 0 |
| **Total** | **48** | **12** | **1** | **18** | **2** | **15** |

---

## The Three Questions That Matter

1. **What is the AUROC after fixing prompt-level splits, removing post-onset tokens from evaluation, and dropping compression_ratio + rep_count?** If it drops below ~0.8, the "zero-cost logit features predict catastrophe" story collapses.

2. **What is the controller's CFR reduction when evaluated only on held-out traces with properly selected thresholds?** If the 43.2% drops substantially, the practical value proposition weakens.

3. **Does switching compression ratio mid-generation actually recover quality, or is the offline simulation systematically wrong?** This requires one live experiment — even 10 prompts would bound the gap.
