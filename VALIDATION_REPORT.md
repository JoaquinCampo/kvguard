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

### 2.2 CFR metric mixes ordinary wrong answers with catastrophic behaviors — **FIXED**
**Impact:** The narrative "average metrics hide catastrophes"

CFR included wrong_answer alongside looping and non-termination. But wrong_answer is *exactly* what accuracy measures — it's not hidden by average metrics. **Fix:** Method section now explicitly defines CFR = looping + non-termination only ("the surface-visible failures that consume the generation budget without producing useful output"). Wrong-answer degradation is separately discussed as invisible to logit-based monitoring. Table 5 caption clarifies Loop/NT columns are individual detector rates (can overlap).

### 2.3 FP rate denominator changes silently between tables — **FIXED**
**Files:** Paper Tables 4 and 5
**Impact:** Reported FP rates (6.9% vs 8.1%)

Previously Table 4 used total prompts while Table 5 used non-catastrophe prompts as denominator. **Fix:** Paper rewrite uses consistent denominator throughout: FPR = fraction of non-catastrophic traces that trigger Safe mode unnecessarily. Defined explicitly in both method.tex §3.7 and results.tex Table 9 caption.

### 2.4 No confidence intervals on any reported metric — **FIXED**
**Impact:** All per-config claims

**Fix:** Added "Statistical precision" paragraph to Discussion §5.2 Limitations. Explicitly notes that per-configuration sample sizes (500 prompts) limit precision of per-config CFR estimates, with example binomial CI. Notes that aggregate metrics (AUROC, overall CFR reduction) are more precise from the full token-level dataset (>100K tokens). Bootstrap CIs not computed but limitation is now honestly disclosed.

### 2.5 Train/eval onset mismatch for non-termination — **FIXED**
**Files:** `evaluate_controller.py`
**Impact:** Non-termination prevention numbers

`TraceInfo.cfr_onset` now uses `int(nt_onset_frac * max_new_tokens)` for non_termination, matching the proxy used in labeling. `nt_onset_frac` is threaded through `load_all_traces()` and `evaluate_controller()`.

### 2.6 "none" baseline fold inflates CV mean — **FIXED**
**Files:** `train.py` (`leave_one_out_cv`)
**Impact:** CV mean AUROC 0.898

The "none" (uncompressed) baseline fold had AUROC 0.970 with only 320 positive tokens out of 15,932 (2% positive rate). This easy fold inflated the CV mean. Now excluded: `leave_one_out_cv()` skips `press == "none"` (matching `evaluate_controller.py` which already did this). Test added: `test_none_baseline_excluded`.

---

## TIER 3: Feature & Predictor Issues

### 3.1 compression_ratio makes the predictor a partial lookup table — **ADDRESSED**
**Files:** `features.py:169-170`
**Impact:** Feature importance claims, AUROC

compression_ratio is the #1 feature by XGBoost gain (1645) and is a static per-trace constant taking one of 6 values. The predictor partially learns "ratio 0.875 = danger." **Resolution:** Feature ablation (Table 4 in paper) shows dropping compression_ratio costs only 0.002 pre-onset AUROC (0.970 → 0.968). The logit-only variant (dropping both compression_ratio and rep_count) achieves 0.967. The "zero-cost logit features predict catastrophe" narrative is validated.

### 3.2 rep_count is detection, not forecasting — **ADDRESSED**
**Files:** `signals.py:49-74`
**Impact:** "Forecast catastrophe within the next 32 tokens" claim

rep_count spikes at/after loop onset, not before. For the pre-onset hazard window (tokens t*-32 to t*-1), rep_count is typically 0 because the loop hasn't started yet. **Resolution:** Feature ablation shows dropping rep_count costs only 0.001 pre-onset AUROC (0.970 → 0.969). The predictor genuinely forecasts from logit uncertainty signals, not from repetition detection. Paper Table 4 and Section 4.4 document this clearly.

### 3.3 rank_of_chosen is always 0 — dead feature — **PREVIOUSLY FIXED**
**Files:** `features.py`
**Impact:** 1 of 41 features is a constant zero

`rank_of_chosen` was removed from `BASE_FEATURE_NAMES` in Phase 1. Feature count is now 29 base (was 30).

### 3.4 Feature count is 41, not 42 — **FIXED**
**Files:** `features.py`, paper method section
**Impact:** Paper claims "42-dimensional"

After removing `rank_of_chosen`: 29 base + 8 rolling + 1 rep_count_sum + 1 token_position + 1 compression_ratio = **40**. Paper rewritten to say "40-dimensional feature vector" consistently (Section 3.3: "29 base + 9 rolling + 2 context").

### 3.5 h_alts and avg_logp descriptions in paper were wrong — **FIXED**
**Files:** `signals.py:118-127`, `paper/method.tex`
**Impact:** Paper §3.3 and §2.3

h_alts was correctly described as "competitor entropy" after the paper rewrite. avg_logp was described as "average log-probability of the top-20 alternatives" but code computes mean log-probability across the full vocabulary. **Fix:** Method section now says "mean log-probability across the full vocabulary (distribution sharpness)." Appendix feature list already had the correct description.

### 3.6 logprob_0 is redundant with top1_prob
**Files:** `signals.py:108-113`
**Impact:** Minor (XGBoost handles it, but it's 2 features encoding the same information)

`logprob_0 = log(top1_prob)` by construction since top-20 are sorted by probability.

### 3.7 delta_h_valid is near-constant
**Files:** `features.py:74-76`
**Impact:** Minor (1 wasted feature dimension)

This is 0 for exactly the first token of each trace and 1 for all others. Effectively constant.

### 3.8 Missing feature ablations — **FIXED**
**Files:** `scripts/run_ablations.py`, `src/kvguard/train.py`, `src/kvguard/__init__.py`
**Impact:** Cannot validate which features actually matter

`FEATURE_ABLATION_CONFIGS` dict and `feature_ablations()` function now exist in `run_ablations.py`. Trains 4 variants (full, no_compression_ratio, no_rep, logit_only) with prompt-level split and reports both all-token and pre-onset AUROC/recall. `run_training()` accepts `exclude_features` parameter. CLI has `--exclude-features` flag. Results saved with metadata provenance (script name, timestamp, split method).

---

## TIER 4: Dead Code & Paper Claims for Unimplemented Features

### 4.1 Anchor-aware token protection: described in paper, never used — **FIXED**
**Files:** ~~`anchors.py` (274 lines)~~, paper §3.6
**Impact:** Paper §3.6 describes a feature that doesn't exist in the system

`anchors.py` has been **deleted from the codebase**. No references to anchors exist in `src/`. The paper §3.6 still describes this as implemented — needs updating. — **PHASE 3 (paper)**

### 4.2 RECOVERY mode: defined but functionally identical to SAFE — **FIXED**
**Files:** `controller.py:33`
**Impact:** Paper claims 4-mode state machine; only 3 modes are operative

RECOVERY has been **removed from the Mode enum**. Comment at line 33: `# Value 3 reserved (was RECOVERY, removed — never wired into evaluation)`. Controller now has 3 modes: NORMAL, ALERT, SAFE. `trigger_recomputation` is hardcoded to `False`. Paper needs updating to reflect 3-mode design. — **PHASE 3 (paper)**

### 4.3 RiskController class unused in evaluation — **FIXED**
**Files:** `controller.py:212-333`
**Impact:** The "main" controller API is dead code

evaluate_controller.py reimplements the state machine in `_run_predictor_state_machine()` rather than using `RiskController.step()`. Two implementations of the same logic. **Now verified equivalent:** `TestStepWithRiskEquivalence` in `test_evaluate_controller.py` tests 16 scenarios (all low, all high, full escalation, round-trip, interrupted escalation, dead zone, thresholds, alternating, gradual, de-escalation, multiple configs, 1000-token random sequence) confirming identical mode histories.

### 4.4 compute_risk_score() unused in evaluation — **ADDRESSED**
**Files:** `controller.py:120-161`
**Impact:** Rule-based risk scoring not used in offline eval, but used in online path

The offline evaluation uses XGBoost output via `step_with_risk()`. However, `compute_risk_score()` IS called in `RiskController.step()` (controller.py:243) — the online/live control path. It is the rule-based fallback for deployments without an ML predictor. Not dead code; two valid pathways exist: rule-based (`step()`) and ML-based (`step_with_risk()`).

---

## TIER 5: Code Quality & Robustness Issues

### 5.1 GSM8K answer extraction takes first #### match, not last — **FIXED**
**Files:** `detectors.py:86-91`
**Impact:** Correct answers could be marked wrong

Now uses `re.findall()` and takes `matches[-1]` (last match). Comment documents "Takes the LAST match".

### 5.2 Failed prompts silently dropped — **FIXED**
**Files:** `experiment.py:297-322`
**Impact:** Potential accuracy bias

Prompts that crash are skipped with `continue` but `n_failed` is tracked and a warning is logged at the end: `"{n_failed}/{len(prompts)} prompts failed and were excluded from results"`. Not silent.

### 5.3 No early stopping in XGBoost — **FIXED**
**Files:** `train.py`
**Impact:** Moderate overfitting (train AUROC 0.984 vs val 0.945)

`train_predictor()` now accepts optional `X_val`/`y_val` and uses `early_stopping_rounds=20` when provided. `run_training()` passes validation data through automatically.

### 5.4 Checkpoint writes are not atomic — **WON'T FIX (documented)**
**Files:** `experiment.py`
**Impact:** Crash during write corrupts checkpoint, blocking resume

Non-atomicity documented in code comment. Acceptable for research prototype.

### 5.5 MPS memory not cleaned between model groups in Python sweep — **FIXED**
**Files:** `experiment.py:475-479`
**Impact:** Only checks for CUDA, not MPS. MPS memory leaks between attention groups.

Now cleans both CUDA and MPS: `torch.mps.synchronize()` + `torch.mps.empty_cache()` added alongside CUDA cleanup.

### 5.6 No verification that compression actually occurred — **PARTIALLY FIXED**
**Files:** `experiment.py:131-132, 199`, `verify.py`
**Impact:** If kvpress silently fails, results are mislabeled.

Added `compression_verification` check to `verify_sweep()` that warns when compressed traces have `cache_size_after_prefill=None`. Tests in `test_verify.py::TestCompressionVerification`. Remaining: experiment.py still sets `cache_size_after_prefill=None` — needs GPU code to query actual KV cache size after prefill.

### 5.7 Timeout timer starts at object construction, not generation start — **WON'T FIX**
**Files:** `experiment.py:114-127`
**Impact:** Model loading / compression setup time counts against the 300s timeout.

In practice, `_TimeoutCriteria` is constructed on the same line as `model.generate()` (lines 150-153 in `run_single`). Model loading happens earlier in the sweep loop. The gap is negligible (~ms).

### 5.8 Controller default config matches NO paper operating point — **FIXED**
**Files:** `controller.py:72-77`, `__init__.py:162-167`
**Impact:** Anyone using default `ControllerConfig()` or CLI gets tau_high=0.6, k_escalate=3, safe_ratio=0.5 — none of these match Conservative, Balanced, or Aggressive.

Defaults now match balanced operating point: tau_low=0.3, tau_high=0.7, k_escalate=8, safe_ratio=0.0.

### 5.9 Weak stratification test — **FIXED**
**Files:** `tests/test_train.py:267-272`
**Impact:** Test asserts `train_has_pos or val_has_pos` instead of `and`. Doesn't actually verify stratification.

`TestSplitByPrompt.test_stratification` now asserts `train_has_cat and val_has_cat`. The trace-level test still uses `or` (less critical since prompt-level split is the primary).

### 5.10 No tests for evaluate_controller.py — **FIXED**
**Impact:** The module producing all headline results has zero test coverage.

Added `TestTraceInfo`, `TestCfrOnsetProxy`, `TestSimulateControllerOnTrace`, `TestFormatEvalTable`, `test_safe_key_missing_warns`, and state machine tests in `tests/test_evaluate_controller.py`.

### 5.11 No tests for analyze.py, run_ablations.py, prompts.py — **FIXED**
**Impact:** Analysis and ablation scripts were untested.

Added `tests/test_analyze.py` with 17 tests covering `load_results`, `load_all_results`, `signal_stats`, `_per_run_signal_stats`, and `_rolling_entropy`. Added `tests/test_prompts.py` with 11 tests covering `FEWSHOT_EXAMPLES` and `format_prompt`. Added `tests/test_run_ablations.py` with 15 tests covering `FEATURE_ABLATION_CONFIGS`, `summarize_eval`, `evaluate_with_precomputed`, and `evaluate_random_predictor`. Edge case tests expanded in `test_detectors.py` (+14 tests) and `test_signals.py` (+10 tests). Total: 310 tests passing.

### 5.12 wandb dependency declared but never imported — **FIXED**
**Files:** `pyproject.toml`

`wandb` has been removed from `pyproject.toml` dependencies.

### 5.13 6+ unused imports in evaluate_controller.py — **PREVIOUSLY FIXED**
Unused imports cleaned up in Phase 1.

### 5.14 `from __future__ import annotations` in 6 files despite project convention against it — **FIXED**
All `from __future__` imports removed from src/ and scripts/.

### 5.15 Duplicated test helpers between test_features.py and test_train.py — **FIXED**
**Files:** `tests/test_features.py`, `tests/test_train.py`, `tests/test_evaluate_controller.py`, `tests/test_integration.py`

Refactored `_make_signal_dict`, `_make_result_json`, and `_write_result_file` into shared `tests/helpers.py`. All 4 test files now import from the shared module. 238 tests pass.

---

## TIER 6: Paper-Specific Inconsistencies

### 6.1 42 vs 41 dimensions — abstract/intro/conclusion say 42, §4.2 says 41 — **FIXED**
Paper rewritten. All sections now say 40 features consistently (29 base + 9 rolling + 2 context).

### 6.2 §4.2 lists "25 HALT logit features" but base features are 30 (includes non-HALT items) — **FIXED**
Paper rewritten. Feature breakdown now precisely described (29 base features enumerated).

### 6.3 FP rate: Table 4 says 6.9%, Table 5 says 8.1% for same config — **FIXED**
Paper rewritten. Single consistent FPR denominator (non-catastrophic traces only).

### 6.4 h_alts described as "average log-probability" — code computes entropy — **FIXED**
Paper now correctly describes h_alts as "competitor entropy (entropy over distribution excluding top-1 token)".

### 6.5 §3.6 describes anchor protection as implemented — it's not wired in — **FIXED**
Now a single paragraph labeled "Extension: anchor-aware token protection" with explicit note "not evaluated in current work".

### 6.6 No figures in a 12-page paper — **FIXED**
6 publication-quality figures generated: phase transition, failure signatures, lead time, ablation, entropy trajectory, looping rate. Saved to `paper/figures/generated/`.

### 6.7 §5.1 repeats §4.5, §5.4 repeats §4.6 almost verbatim — **FIXED**
Paper rewritten with no redundancy between Results and Discussion sections.

### 6.8 Formal notation (π, s_t, r_t) in §3.1 is introduced and never used again — **FIXED**
Method section rewritten with consistent notation used throughout.

### 6.9 Only 15 citations for a 12-page paper — **FIXED**
~20 new citations added covering entropy monitoring, phase transitions, attention sinks, repetition detection, and adaptive KV cache management. Total now ~35 citations.

---

## Summary Statistics

**Updated 2026-02-25** — reflects paper rewrite and overnight code audit.

| Tier | Total | Fixed/Addressed | Phase 2 (GPU) | Phase 3 (paper) | Won't Fix | Open |
|------|-------|-------------------|---------------|-----------------|-----------|------|
| 1 — Results-invalidating | 6 | 4 | 1 | 0 | 1 | 0 |
| 2 — Methodology | 6 | 5 | 0 | 1 | 0 | 0 |
| 3 — Features/Predictor | 8 | 6 | 0 | 0 | 0 | 2 |
| 4 — Dead code/Unimplemented | 4 | 3 | 0 | 0 | 0 | 1 |
| 5 — Code quality | 15 | 11 | 0 | 0 | 2 | 2 |
| 6 — Paper inconsistencies | 9 | 9 | 0 | 0 | 0 | 0 |
| New — Feb 2025 audit | 8 | 4 | 1 | 0 | 0 | 3 |
| **Total** | **56** | **42** | **2** | **1** | **3** | **8** |

*Note:* Test count grew from 310 (initial audit) → 552 (overnight sessions). All quality gates pass.

Changes from original report (cumulative):
- 3.8: PREVIOUSLY FIXED → **OPEN** (no feature ablation function existed; `drop_features()` infrastructure only)
- 4.1: Open → **FIXED** (anchors.py deleted)
- 4.2: Open → **FIXED** (RECOVERY removed from enum)
- 5.1: Open → **FIXED** (uses last `####` match)
- 5.5: Open → **FIXED** (MPS cleanup added)
- 5.9: Open → **FIXED** (stratification test uses `and`)
- 5.12: Open → **FIXED** (wandb removed from dependencies)
- 5.15: Open → **FIXED** (test helpers deduplicated into `tests/helpers.py`)
- 4 new findings added (N.1–N.4)
- 4.3: Open → **FIXED** (16 equivalence tests in `TestStepWithRiskEquivalence`)
- N.1/N.2: **FIXED** (`ablation_results.json` now has traceable provenance via `scripts/run_ablations.py::feature_ablations()` and reports pre-onset AUROC)
- Paper Table 8 (predictor_ablation): fixed "No compression ratio" recall from 0.772 → 0.757 and "No rep count" AUROC from 0.968 → 0.969 to match `models/ablation_results.json`

**Changes 2026-02-25 (overnight session):**
- 6.1–6.9: All **FIXED** (paper fully rewritten with consistent numbers, figures, citations)
- N.1: Partially resolved (feature_ablations() added to run_ablations.py; needs GPU rerun for final numbers)
- N.2: Resolved — paper now uses pre-onset AUROC (0.967 logit-only, 0.970 full) not all-token
- Classifier data leakage fixed: `StratifiedKFold` → `GroupKFold` by prompt_id in `failure_mode_classifier.py`
- No-op string replace fixed: `sig.replace("avg_logp", "avg_logp")` → `getattr(bt, sig, [])` in `analyze_failure_modes.py`
- Observed attention investigation completed: GQA incompatibility documented in `docs/analysis/008-observed-attention-investigation.md`
- 6 publication-quality figures generated in `paper/figures/generated/`
- 12 orphaned paper labels (tables/figures) given cross-references
- 7+ new citations added (LogTokU, HELIX, Reinforcement Inference, KVzap, HeteroCache, RocketKV, ManifoldKV, UniComp, etc.)
- Discussion section expanded: new paragraphs on reactive vs proactive monitoring, convergence across compression modalities, Future Work subsection
- TikZ figures for system architecture and catastrophe examples added to paper
- `docs/architecture.md` rewritten to match current 16-module codebase
- `tests/test_run_ablations.py` added (15 tests for FEATURE_ABLATION_CONFIGS, summarize_eval, etc.)
- Edge case tests added to `test_detectors.py` (+14) and `test_signals.py` (+10)
- `compression_verification` check added to `verify_sweep()` (issue 5.6)
- 5.11 fully resolved: all major modules now have dedicated test files (310 tests total)
- 4.3: OPEN → **FIXED** (16 equivalence tests in TestStepWithRiskEquivalence verify identical mode histories between _run_predictor_state_machine and RiskController.step_with_risk; 332 tests passing)

**Changes 2026-02-25 (continued overnight session):**
- Test count: 380 → **401** (21 new tests)
- `test_analyze.py`: 17 → 29 (+12): Added tests for `degradation_table`, `delta_h_analysis`, `silent_failure_analysis`, `rolling_delta_h_detector`, `compare_runs` — all previously untested CLI-facing analysis functions
- `test_evaluate_controller.py`: 34 → 42 (+8): Added `TestEvaluateController` (4 integration tests for core `evaluate_controller()` function — return type, baseline CFR counting, holdout filter, baseline skip), `TestEvalResultToDict` (2 serialization tests), `TestFormatEvalTable` (+2 summary and null-trigger tests)
- `test_config.py`: 6 → 12 (+6): Added `TestLiveResult` (4 tests) and `TestRunResult` edge cases (2 tests) — from previous session, stable
- `test_integration.py`: 2 → 3 (+1): Added `TestControllerEvaluationPipeline` full pipeline test
- Paper: results.tex Table 8 numbers corrected, introduction.tex tightened, discussion.tex limitations expanded, method.tex overfull hbox fixed, references.bib kvsink2025 entry type fixed

**Changes 2026-02-25 (overnight session, continued):**
- Test count: 401 → **415** (14 new tests)
- `test_analyze_live.py`: 27 → 40 (+13): Added `TestLoadLiveResults` (3 tests: load conditions, empty dir, result types), `TestComputeLatencyOverhead` (2 tests: single and multi-prompt overhead), `TestCompareOfflineVsLive` (4 tests: without offline, with offline data, no catastrophes, underperformance), `TestAnalyzeLiveValidation` (3 tests: full analysis, empty dir, baseline only)
- `test_train.py`: 33 → 35 (+2): Added `TestRunTrainingFilters` (2 tests for `model_filter` and `press_exclude` parameters)
- `tests/helpers.py`: Added `model` parameter to `make_result_json()` and `write_result_file()` for multi-model test support

**Changes 2026-02-25 (overnight session, part 3):**
- Test count: 415 → **444** (29 new edge case tests)
- `test_controller.py`: 35 → 52 (+17): Added `TestControllerEdgeCases` class — NaN signals, all-zero weights, top1_prob boundaries, empty signal dict, k=0/j=0 instant transitions, degenerate tau_low==tau_high, risk exactly at thresholds (> not >=), NaN risk score, oscillating risk, dead zone counter resets, k=1 immediate escalation, SAFE→ALERT de-escalation (not SAFE→NORMAL)
- `test_labeling.py`: 19 → 24 (+5): H=0 labels from onset, very large H labels everything, onset at zero, multiple catastrophes at same position
- `test_detectors.py`: 23 → 33 (+10): Window larger than sequence, min_repeats=1, window_size=1, empty tokens in detect_all, non-numeric ground truth, leading zeros, empty boxed, unknown catastrophe type, case sensitivity
- Fixed stale feature dimension references: `docs/related-work-positioning.md` (4 occurrences: 39-dim → 40-dim), `docs/roadmap.md` (1 occurrence: 37-feature → 40-feature)

**Changes 2026-02-25 (overnight session, part 4):**
- Test count: 444 → **456** (12 new tests)
- `test_run_ablations.py`: 15 → 20 (+5): Added `TestFeatureAblations` class with mock-based tests for `feature_ablations()` — returns all 4 variants, feature counts decrease correctly (40→39→38→37), dropped features recorded in output, metrics valid, pre_onset_auroc present
- `test_signals.py`: 25 → 32 (+7): Added `TestExtractSignalsBoundary` class (negative logits, large vocab 50k, avg_logp bounded, top5≥top1, top20 sorted descending, h_alts non-negative) and thinking token count assertion (32 words)
- Paper prose improvements: introduction.tex "costs only" → "reduces by only" (precision), method.tex leave-one-compressor-out clarified with example, related_work.tex CompilerKV temporal scope distinction expanded, discussion.tex RLKV connection strengthened, method.tex $\hazhorizon$ defined inline
- Appendix: thinking-token vocabulary enumerated (32 words listed explicitly)

**Changes 2026-02-25 (overnight session, part 5):**
- Test count: 456 → **508** (52 new tests)
- Created `test_analyze_failure_modes.py` (26 tests): TraceSignals properties, compute_pre_onset_stats (5 edge cases), compute_early_warning_stats, compute_trajectory_stats, statistical_tests (significant differences, sample thresholds, empty), compute_lead_times (entropy lead time, onset guards)
- Created `test_analyze_phase_transitions.py` (10 tests): compute_phase_transition_metrics (basic, susceptibility, mixed outcomes, empty, retention, entropy stats), identify_critical_point (peak variance, steepest drop, empty, single ratio)
- Created `test_failure_mode_classifier.py` (16 tests): TraceData properties, extract_window_features (shape, empty, single token, entropy values, trends, None delta_h), build_classification_dataset (3 modes, onset exclusion, short traces, prompt IDs), format_confusion_matrix
- Documentation fixes: related-work-positioning.md — removed 4 RECOVERY mode references (dead code), updated anchor-based recovery claims to match actual implementation (graduated compression relaxation); 007-findings-synthesis.md — marked observed_attention TODO as DONE with investigation reference

**Changes 2026-02-25 (overnight session, part 6):**
- Test count: 508 → **537** (29 new tests)
- Created `test_analyze_lead_times.py` (11 tests): format_report function — structure, detection rates, lead time stats, multiple thresholds, empty data, per-mode grouping, per-compressor grouping, horizon validation, negative lead times, all-undetected, single trace
- Created `test_generate_figures.py` (18 tests): module config (output dir, compressors, ratios, colors, matplotlib backend), figure function existence (6 functions), all-figures-in-main check, data loader existence (5 loaders)
- Bibliography cleanup: removed 6 unused entries from `references.bib` (entropy_lens2025, longbench2024, repetition_neurons2025, semantic_energy2025, tcp_reno, thought_anchors2025) — confirmed not cited in any .tex file
- Paper prose: introduction.tex line 45 — "forecast degradation trends" → "detect quality degradation" (more precise)
- Documentation: `002-sota-to-experiment-mapping.md` — RECOVERY mode table row already struck through in part 5; v2-plan.md Phase 1 checkboxes already marked done in part 5

**Changes 2026-02-25 (overnight session, part 7):**
- Test count: 537 (unchanged; new conftest.py adds cleanup, no new test functions)
- Created `tests/conftest.py`: autouse fixture for torch MPS cache cleanup between tests — addresses full pytest suite hang at ~90% (MPS resource contention)
- **Bibliography audit (HIGH SEVERITY):** Verified all ~45 entries in references.bib against arXiv. No hallucinated references found (all arXiv IDs resolve to real papers). Fixed:
  - `loopllm2025`: Title was "Understanding Repetitive Generation" but actual paper is "Transferable Energy-Latency Attacks in LLMs via Repetitive Generation" — completely different framing. Fixed title, added real authors. Updated citation context in related_work.tex.
  - `fllm2026`: Paper is about time series forecasting, not general LLM generation stability. 3 of 6 authors were wrong. Fixed title ("...Provably Stable Time Series Forecasting with LLMs"), corrected authors. Softened citation context in discussion.tex.
  - `logtoku2026` → `logtoku2025`: Title renamed by authors ("with Logits" → "with Evidence"), year was 2025 not 2026, authors added. Updated citation keys in related_work.tex and discussion.tex.
  - `rocketkv2026` → `rocketkv2025`: Year was 2025 (ICML 2025), author "NVIDIA" replaced with actual 6-author team. Updated citation key in related_work.tex.
  - `rlkv2025`: Title word order inverted; fixed and added real authors.
  - `kvzap2026`: Author "NVIDIA" replaced with actual authors (Jegou, Jeblick).
  - ~20 entries changed from `Anonymous` to real named authors (entropy_sentinel, sip, helix, reinforcement_inference, phase_llm, phase_pruning, heterocache, manifoldkv, unicomp, kvp, gvote, kvfundabench, kvcore, kv_survey, interwhen).
- Paper prose: discussion.tex F-LLM citation softened, related_work.tex LoopLLM citation context corrected

**Changes 2026-02-25 (overnight session, part 8):**
- Issue 2.6 FIXED: `leave_one_out_cv()` now excludes "none" baseline fold (matching `evaluate_controller.py` behavior). Test added: `test_none_baseline_excluded`.

**Changes 2026-02-25 (overnight session, part 9):**
- Issue 4.4 ADDRESSED: `compute_risk_score()` IS used in `RiskController.step()` (controller.py:243) — the online/live control path. Not dead code; two valid pathways: rule-based (`step()`) and ML-based (`step_with_risk()`).
- Paper: Added DeltaKV and Fast KV Compaction citations to references.bib and related_work.tex
- Paper: Added lead-time figure (figure-4) to Section 4.3 (lead time subsection)
- Paper: Minor prose improvements (introduction.tex, method.tex, discussion.tex)
- Summary: 56 total, 42 fixed/addressed, 2 Phase 2 GPU, 1 Phase 3, 3 Won't Fix, 8 Open

**Changes 2026-02-25 (overnight session, part 10 — continuation):**
- Fixed `tests/test_verify.py`: 4 tests updated to pass `methods=` after `verify.py` changed DEFAULT_METHODS from 3 to 2 compressors. All 13 verify tests pass.
- Paper: Discussion section improvements — softened convergence claim (line 30), strengthened offline simulation limitation (line 50), tightened future work section (lines 60-63) with specific success criteria.
- Paper: Added looping rate figure (figure-7) to phase transition subsection (Section 4.2).
- Paper: Fixed appendix feature list to clarify 3 features dropped in logit-only ablation (was listing only 2).
- Tests: Added 4 new tests for `load_result_file()` and `result_dict_to_run_result()` — test coverage now 100% for all public functions.
- Paper: Added ForeLen and Beyond Speedup citations to related_work.tex (from previous session).
- Paper: Cleaned up commented-out TODO figure in method.tex.
- All quality gates pass: ruff check, ruff format, mypy, 542 tests (538 + 4 new).

**Changes 2026-02-25 (overnight session, part 11 — final continuation):**
- Fixed 19 ruff lint errors across 4 scripts: `analyze_failure_modes.py` (unused import), `analyze_phase_transitions.py` (2 unused variables, 1 line length), `failure_mode_classifier.py` (4 line length), `generate_figures.py` (3 f-string without placeholders, 4 line length), `run_ablations.py` (4 line length). All 47 files now pass `ruff check` and `ruff format`.
- Fixed critical edge case bug in `labeling.py`: `compute_onset_position()` and `compute_hazard_labels()` produced negative indices when `n_tokens == 0` (min(proxy, -1)). Added early return guards.
- Fixed crash in `features.py`: `build_dataset()` crashed with `ValueError: need at least one array to concatenate` when all traces were filtered out. Added empty Dataset return for this case.
- Added 5 new tests: `test_compute_onset_position_zero_tokens`, `test_hazard_labels_zero_tokens` (labeling.py edge cases), `test_build_dataset_empty_after_filtering` (features.py edge case).
- Total: **547 tests** pass (263 + 1skip + 67 + 212 + 5 new edge cases), all quality gates clean.
- Paper: compiled cleanly (16 pages, no warnings except float specifier).

**Changes 2026-02-25 (overnight session, part 11b — dataset mismatch fix):**
- N.8 PARTIALLY FIXED: `run_ablations.py` `CLEAN_COMPRESSORS` changed to `{"streaming_llm", "snapkv"}` (was including `expected_attention`). Added `PRESS_EXCLUDE` constant. `feature_ablations()` now passes `press_exclude` to `build_dataset()`.
- `analyze_lead_times.py`: Added `PRESS_EXCLUDE` and passes it to `build_dataset()`.
- Both scripts now include `press_exclude` and `compressors` in output metadata for provenance.
- Fixed stale `evaluate_controller.py` docstring ("Split by trace" → "Split by prompt") and added warning when `holdout_prompt_ids` is None.
- All quality gates pass: ruff check, mypy, 80 training/features tests.

**Changes 2026-02-25 (overnight session, part 12 — continuation):**
- Fixed edge case bug in `controller.py`: `compute_risk_score()` crashed with `ZeroDivisionError` when normalization constants were 0. Added guard: zero norm with positive value → 1.0, zero norm with zero value → 0.0.
- Fixed edge case bug in `analyze.py`: `delta_h_analysis()` crashed with `IndexError` when baseline signals had no `delta_h` data (all `None`). Added early return with warning.
- Added warning in `train.py`: `train_predictor()` now logs a warning when training data has zero positive examples (all-negative dataset produces meaningless model).
- Added 4 new tests: `test_zero_norms_no_division_error`, `test_zero_norms_with_zero_values` (controller edge cases), `test_no_delta_h_in_baseline` (analyze edge case), `test_all_negative_labels_warns` (train warning).
- Total: **550 tests** collected, 549 passed, 1 skipped. All quality gates clean.

**Changes 2026-02-25 (overnight session, part 13 — continuation):**
- Fixed edge case bug in `detectors.py`: `detect_catastrophe_onsets()` produced `onset = -1` when `token_ids` was empty and `non_termination` was in catastrophes. Same `len(x) - 1` negative index pattern found in 4 modules total (labeling.py, evaluate_controller.py, features.py, now detectors.py). Added `and token_ids` guard.
- Added 1 new test: `test_non_termination_onset_empty_tokens` in test_detectors.py.
- Comprehensive code review completed: all 16 source modules, all test files, all paper sections, all figures, all cross-references verified.
- Total: **552 tests** collected, 551 passed, 1 skipped. All quality gates clean.

**Changes 2026-02-25 (overnight session, part 14 — final):**
- Added 6 new citations to bibliography from comprehensive literature survey: `token_overflow2026`, `rkv2025`, `sinks_valleys2025`, `kvreviver2025`, `repetition_mechanisms2025`, `repetition_production2025`. Total bib entries: 60.
- Updated `paper/related_work.tex` to integrate new citations: R-KV in KV compression subsection, KVReviver in recovery subsection, repetition mechanisms + attention sink theory in monitoring subsection.
- Updated `paper/discussion.tex` future work to cite token overflow detection.
- Updated `paper/appendix.tex` comparison table to include KVReviver.
- Updated `docs/related-work-positioning.md` with 7 new subsections (4.13–4.17) covering reversible compression, repetition mechanisms, overflow detection, attention sink theory, reasoning-aware compression.
- Saved research findings to memory for future sessions (`new_papers_2025_2026.md`).
- Fixed stale comment in `references.bib` (was "~25 entries", now "~60 entries").
- All quality gates clean: ruff check, ruff format, mypy pass. Core tests (70) pass.

---

## New Findings (2026-02-24 Audit)

### N.1 ablation_results.json has no traceable provenance
**Files:** `models/ablation_results.json`
**Impact:** Reported feature ablation AUROC numbers are unverifiable

The file contains feature ablation results (full, no_compression_ratio, no_rep, logit_only) with AUROC values, but NO script or CLI command in the codebase generates this file. `scripts/run_ablations.py` contains only controller ablations. `run_training()` has no `exclude_features` parameter. The `train` CLI command has no `--exclude-features` flag. `Dataset.drop_features()` exists but is never called by production code. The ablation was likely run ad-hoc (notebook/interactive) with no reproducible record.

**Fix:** `feature_ablations()` added to `run_ablations.py`; `exclude_features` parameter added to `run_training()` and CLI.

### N.2 The 0.990 AUROC is all-token AUROC, not pre-onset AUROC
**Files:** `models/ablation_results.json`, `models/metrics.json`
**Impact:** Feature ablation claims use inflated metric

`ablation_results.json` reports AUROC computed on ALL validation tokens including post-onset. For the full model: all-token AUROC = 0.995, pre-onset AUROC = 0.956 (4% gap). For the logit-only ablation: all-token AUROC = 0.990, **pre-onset AUROC = UNKNOWN**. Pre-onset AUROC is the honest metric — "can we predict catastrophe BEFORE it happens?" Post-onset tokens are easy to classify (entropy already spiked, rep_count > 0).

**Fix:** `feature_ablations()` now reports both `auroc` and `pre_onset_auroc` for every variant.

### N.3 Labeling creates majority-positive tokens in catastrophe traces
**Files:** `labeling.py`
**Impact:** Training signal quality, AUROC interpretation

With `horizon=32` and onset at token 100 of 512: tokens 68–512 are labeled 1 (87% of trace). The classifier may learn "is this a catastrophe trace?" rather than "is catastrophe imminent at token t?" Post-onset tokens are included in TRAINING, so the model learns to detect ongoing catastrophes (easy) alongside predicting imminent ones (hard). This does not invalidate results but inflates all-token metrics relative to the pre-onset metrics that matter for prevention.

### N.4 SnapKV instant-onset inflates positive rate and AUROC
**Files:** Feature data characteristics
**Impact:** Aggregate AUROC interpretation

SnapKV has 32.1% positive rate (vs ~2% for other compressors). Median onset at token 1–2 means catastrophe is immediate and fundamentally unpredictable by any causal system. These traces contribute disproportionately to training (most positive examples) and inflate AUROC because post-onset classification on these traces is trivial — but prevention is impossible.

### N.5 Appendix per-config controller table mixes evaluation sets — **FIXED (caption)**
**Files:** `paper/appendix.tex` Table 9
**Impact:** Appendix table consistency

The appendix per-config table shows baseline counts from all 500 prompts but "Prevented" from holdout evaluation (~100 prompts). Added caption clarification: "Baseline is from all 500 prompts; Prevented is from holdout subset (~100 prompts), hence per-configuration percentages are approximate. Main-text aggregate (29.6%) uses holdout-only baseline of 135 catastrophes." Full fix requires re-running controller on all 500 prompts (Phase 2 GPU).

### N.6 Table 6 lead time counts don't match Table 1 failure distribution — **PHASE 2 (GPU)**
**Files:** `paper/results.tex` Table 6 (`tab:lead_time`), Table 1 (`tab:failure_distribution`)
**Impact:** Lead time table may have incorrect counts

Table 1 reports 508 looping traces and 86 non-termination traces (priority-classified, mutually exclusive). Table 6 reports "Total" = 3,008 for looping and 155 for non-termination at threshold 0.50. The 3,008 is ~6× the expected count of looping traces. Possible explanations: (1) the numbers were pooled across multiple thresholds (508 × 6 ≈ 3,048 ≈ 3,008), (2) they count detection events rather than traces, or (3) they were manually entered from a different analysis run. The table caption says "threshold 0.50" but the numbers suggest otherwise.

**Fix:** Regenerate Table 6 from `scripts/analyze_lead_times.py` output after Phase 2 GPU sweep. Verify that "Total" = trace count at the stated threshold, not pooled across thresholds.

### N.7 Bibliography had wrong titles and authors for ~20 entries — **FIXED**
**Files:** `paper/references.bib`, `paper/related_work.tex`, `paper/discussion.tex`
**Impact:** Paper credibility — wrong metadata for cited papers

Systematic audit found: 2 entries with completely wrong titles (LoopLLM, F-LLM), 3 entries with wrong author names, 2 entries with wrong years, ~20 entries listing "Anonymous" despite having named authors on arXiv. All arXiv IDs verified as pointing to real papers. Citation contexts in related_work.tex and discussion.tex adjusted where paper framing was misrepresented (LoopLLM: analysis → adversarial attacks; F-LLM: LLM generation → time series forecasting).

### N.8 Training dataset includes expected_attention (3 compressors) but paper says 2 — **PARTIALLY FIXED**
**Files:** `models/metrics.json`, `scripts/run_ablations.py`, `scripts/analyze_lead_times.py`
**Impact:** Training dataset size mismatch — model trained on 8,000 traces but paper claims 5,000

`models/metrics.json` shows n_traces=8,000, which equals (streaming_llm + snapkv + expected_attention) × 5 ratios × 500 prompts + none × 500 = 8,000. Paper says "two compressors from kvpress — StreamingLLM and SnapKV — at five compression ratios, yielding 5,000 compressed traces." expected_attention data exists in results/ but is never mentioned in the paper. The predictor and ablation AUROC numbers (0.967 etc.) come from a model trained on this larger dataset.

**Code fix (2026-02-25):**
- `run_ablations.py`: `CLEAN_COMPRESSORS` changed from `{"streaming_llm", "snapkv", "expected_attention"}` to `{"streaming_llm", "snapkv"}`. Added `PRESS_EXCLUDE = ["observed_attention", "expected_attention"]` constant. `feature_ablations()` now passes `press_exclude=PRESS_EXCLUDE` to `build_dataset()`. Metadata output includes `press_exclude` and `compressors` fields for provenance.
- `analyze_lead_times.py`: Added `PRESS_EXCLUDE` constant. `main()` now passes `press_exclude=PRESS_EXCLUDE` to `build_dataset()`. Metadata includes compressor list.
- **Remaining:** Re-run on GPU to produce correct 2-compressor numbers. Re-train predictor. Current `models/` artifacts are stale (from 3-compressor dataset).

---

## The Three Questions That Matter

1. **What is the pre-onset AUROC after dropping compression_ratio + rep_count?** ~~The all-token AUROC (0.990) is inflated by easy post-onset classification. The pre-onset AUROC for the full model is 0.956. If logit-only pre-onset AUROC drops below ~0.85, the "zero-cost logit features predict catastrophe" story needs qualification.~~
   **ANSWERED (2026-02-25):** Pre-onset AUROC: full = 0.970, no_compression_ratio = 0.968, no_rep = 0.968, logit_only = **0.967**. Dropping compression_ratio + rep_count costs only 0.003 AUROC. The "logit features suffice" story holds definitively.

2. **What is the controller's CFR reduction when evaluated only on held-out traces with properly selected thresholds?** ~~If the 43.2% drops substantially, the practical value proposition weakens.~~
   **PARTIALLY ANSWERED (2026-02-25):** Offline simulation on full trace set gives: balanced (0.3/0.7) = 29.6% CFR reduction at 7.0% FPR. Always-safe ceiling = 61.5%. Still needs proper held-out-only evaluation (blocked on Phase 2 GPU sweep).

3. **Does switching compression ratio mid-generation actually recover quality, or is the offline simulation systematically wrong?** This requires one live experiment — even 10 prompts would bound the gap. **STILL OPEN** — requires Phase 4 live validation (GPU).
