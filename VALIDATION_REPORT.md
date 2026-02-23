# KVGuard Systematic Validation Report

**Date:** 2026-02-22
**Scope:** Full codebase + paper audit across 6 dimensions: data pipeline, features, controller, experiments, tests/dead code, statistical claims.

---

## TIER 1: Results-Invalidating Issues

These issues mean the headline numbers in the paper may not survive correction.

### 1.1 Controller evaluated on predictor's training data
**Files:** `evaluate_controller.py:361-362`, `train.py:63-119`
**Impact:** ALL controller numbers (43.2% CFR reduction, 63.9% excl. extreme, 6.9% FP rate)

The controller evaluation loads all 800 traces and runs the trained predictor on all of them. ~80% of these traces were used to train the predictor. The predictor's predictions on training traces are overconfident, directly inflating the controller's evaluated effectiveness. There is no holdout separation between predictor training data and controller evaluation data.

### 1.2 Threshold selection on test data
**Files:** `scripts/run_ablations.py:278-280`, `evaluate_controller.py`
**Impact:** The "balanced" operating point (tau_low=0.3, tau_high=0.7, k=8)

The 45-configuration threshold sweep was run on the same 800 traces used for controller evaluation (which also overlap with training data). The "balanced" point was selected for best prevention-to-FP ratio from this sweep and then reported as the main result. This is hyperparameter selection on the test set — the reported numbers are optimistic by construction.

### 1.3 Prompt-level data leakage in train/test split
**Files:** `train.py:63-119` (`split_by_trace`)
**Impact:** AUROC 0.945, all predictor metrics

The split is by trace (prompt × compressor × ratio), not by prompt. The same GSM8K prompt appears in both train and validation under different compressor/ratio configs. With only 50 unique prompts, the predictor can learn prompt-specific logit signatures. The paper claims "We split at the trace level to prevent data leakage" — this is literally true but the relevant leakage unit is the prompt, not the trace.

### 1.4 Same leakage in leave-one-compressor-out CV
**Files:** `train.py:284-334` (`leave_one_out_cv`)
**Impact:** CV mean AUROC 0.898

When SnapKV is held out, the training set includes all 50 prompts from StreamingLLM and ObservedAttention. The held-out SnapKV fold has those same 50 prompts. Every prompt in test also appears in training. The CV tests cross-compressor generalization but NOT cross-prompt generalization.

### 1.5 Token position feature leaks future information
**Files:** `features.py:163-167`
**Impact:** AUROC and controller performance

`token_position = t / (n - 1)` where `n` is total tokens in the trace. At training time, every token knows the final sequence length — information unavailable at inference time. Non-termination traces have longer sequences, so this feature encodes outcome information. At deployment, you'd need `t / max_new_tokens` instead.

### 1.6 Labels include post-onset tokens, inflating predictor metrics
**Files:** `labeling.py:49-52`
**Impact:** AUROC 0.945, recall numbers

Hazard labels are set to 1 for all tokens from `t* - H` through end of trace. Tokens after onset are *inside* the catastrophe (looping, repetition already happening) and are trivially classifiable. The reported AUROC and recall include these easy post-onset tokens. True *predictive* performance (pre-onset only) would be lower.

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

### 2.5 Train/eval onset mismatch for non-termination
**Files:** `labeling.py:41-43`, `detectors.py:69-71`, `evaluate_controller.py:410`
**Impact:** Non-termination prevention numbers

The predictor trains with proxy onset at 0.75 × max_new_tokens = token 384. The evaluation uses actual onset = last token (token 511). The predictor learns to fire around tokens 352-384, which incidentally provides lead time for the evaluation criterion, but the two are disconnected by design rather than intentionally aligned.

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

### 3.3 rank_of_chosen is always 0 — dead feature
**Files:** `experiment.py:140` (`do_sample=False`), `signals.py:133-135`
**Impact:** 1 of 41 features is a constant zero

Under greedy decoding, the chosen token is always rank 0. This feature carries zero information. Additionally, `compute_risk_score()` in controller.py allocates 10% weight to it, wasting weight budget.

### 3.4 Feature count is 41, not 42
**Files:** `features.py:24-57`
**Impact:** Paper claims "42-dimensional" (abstract, introduction, §3.3, conclusion)

Exact count: 30 base + 8 rolling stats + rep_count_sum_8 + token_position + compression_ratio = **41**. The paper says 42 in the abstract, introduction, and conclusion. Section 4.2 says 41 but then lists "25 HALT logit features" which only accounts for 36 of the 41.

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

### 3.8 Missing feature ablations
**Files:** `scripts/run_ablations.py`
**Impact:** Cannot validate which features actually matter

The ablation suite tests controller components (always-safe, random predictor, hysteresis, threshold sweep) but never removes features from the predictor. No ablation for: dropping compression_ratio, dropping rep_count, dropping rolling features, logit-only features, etc. This is the most standard ML ablation and it's absent.

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

### 5.3 No early stopping in XGBoost
**Files:** `train.py:126-167`
**Impact:** Moderate overfitting (train AUROC 0.984 vs val 0.945)

200 trees with no early stopping and no eval_set. The 0.04 AUROC gap suggests moderate overfitting.

### 5.4 Checkpoint writes are not atomic
**Files:** `experiment.py:41-46`
**Impact:** Crash during write corrupts checkpoint, blocking resume

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

### 5.10 No tests for evaluate_controller.py
**Impact:** The module producing all headline results has zero test coverage.

### 5.11 No tests for analyze.py, run_ablations.py, prompts.py
**Impact:** Analysis and ablation scripts are untested.

### 5.12 wandb dependency declared but never imported
**Files:** `pyproject.toml:21`

### 5.13 6+ unused imports in evaluate_controller.py
**Files:** `evaluate_controller.py:15,23-31` (json, RunResult, RiskController, N_BASE, ROLLING_COL_INDICES, ROLLING_COLS, ROLLING_STATS, _rolling_stat)

### 5.14 `from __future__ import annotations` in 6 files despite project convention against it
**Files:** controller.py, features.py, train.py, evaluate_controller.py, verify.py, anchors.py

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

| Tier | Count | Description |
|------|-------|-------------|
| 1 — Results-invalidating | 6 | Would change headline numbers if corrected |
| 2 — Methodology | 6 | Flawed or misleading methodology |
| 3 — Features/Predictor | 8 | Feature engineering and predictor issues |
| 4 — Dead code/Unimplemented | 4 | Paper claims for code that doesn't function |
| 5 — Code quality | 15 | Bugs, missing tests, robustness |
| 6 — Paper inconsistencies | 9 | Text-level errors and mismatches |
| **Total** | **48** | |

---

## The Three Questions That Matter

1. **What is the AUROC after fixing prompt-level splits, removing post-onset tokens from evaluation, and dropping compression_ratio + rep_count?** If it drops below ~0.8, the "zero-cost logit features predict catastrophe" story collapses.

2. **What is the controller's CFR reduction when evaluated only on held-out traces with properly selected thresholds?** If the 43.2% drops substantially, the practical value proposition weakens.

3. **Does switching compression ratio mid-generation actually recover quality, or is the offline simulation systematically wrong?** This requires one live experiment — even 10 prompts would bound the gap.
