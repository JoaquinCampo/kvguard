# KVGuard v2: Addressing Methodological Issues

**Created:** 2026-02-22
**Status:** In progress
**Goal:** Fix the real problems identified in the v1 code/methodology, scale up experiments on 5090 GPU, produce results that survive peer review.

---

## Context: What's Wrong with v1

Post-review analysis identified five substantive problems in the v1 codebase and results:

| # | Issue | Severity | File(s) |
|---|-------|----------|---------|
| 1 | **Prompt-level data leakage** — train/test split is by trace, not by prompt. Same GSM8K prompt appears in both partitions under different compressor/ratio configs. | High | `train.py` |
| 2 | **rep_count is detection, not forecasting** — the feature spikes at/after looping onset, not before. Inflates AUROC for "forecast H=32 tokens ahead" claim. | High | `signals.py`, `features.py` |
| 3 | **compression_ratio as top feature** — static per-trace constant that turns the predictor into a partial lookup table. No ablation exists. | Medium | `features.py`, `scripts/run_ablations.py` |
| 4 | **RECOVERY mode + anchors.py are dead code** — defined, tested in isolation, never wired into experiment or evaluation pipeline. | High | `controller.py`, `anchors.py`, `evaluate_controller.py` |
| 5 | **Offline simulation causality** — "prevention" assumes switching to r_safe=0 mid-generation recovers the same behavior as never compressing. Untested. | High | `evaluate_controller.py` |

Secondary issues:
- Feature vector is 41-dim, paper says 42 in some places
- `rank_of_chosen` is always 0 under greedy decoding (dead feature)
- n=50 prompts is too small for credible per-compressor statistics
- Single model (Qwen 3B) — generalization untested

---

## Plan

### Phase 1: Code Fixes (CPU-only, do now)

These are all code changes that don't require GPU compute.

#### P1-01: Implement prompt-level train/test split
- **File:** `src/kvguard/train.py`
- **Change:** Rewrite `split_by_trace()` to group by `prompt_id`. All traces for a given prompt go to the same partition. Stratify by whether any of the prompt's traces have catastrophe.
- **Also fix:** `leave_one_out_cv()` — hold out compressor AND ensure prompt-level integrity within each fold.
- **Tests:** Update `tests/test_train.py` — add test that no `prompt_id` appears in both train and val.
- [ ] Done

#### P1-02: Add feature ablation support to training
- **File:** `src/kvguard/train.py`, `src/kvguard/features.py`
- **Change:** Add `exclude_features: list[str] | None` parameter to `train_predictor()` and `build_dataset()`. When set, drop those columns from X before training.
- **Predefined ablation sets:**
  - `no_compression_ratio` — drop `compression_ratio`
  - `no_rep` — drop `rep_count`, `rep_count_sum_8`
  - `logit_only` — drop both of the above
  - `logit_only_minimal` — also drop `rank_of_chosen` (dead under greedy)
- **Tests:** Add tests that excluded features are actually absent from X.
- [ ] Done

#### P1-03: Drop rank_of_chosen from default feature set
- **File:** `src/kvguard/features.py`
- **Change:** Remove `rank_of_chosen` from the base feature list. It's always 0 under greedy decoding.
- **Update:** Feature count goes from 41 → 40. Update all references.
- **Tests:** Update feature count assertions.
- [ ] Done

#### P1-04: Decide on RECOVERY mode and anchors.py
- **Options:**
  - (A) **Cut it** — remove RECOVERY from the state machine, delete `anchors.py`, simplify controller to 3 modes (NORMAL, ALERT, SAFE). Honest about what's actually evaluated.
  - (B) **Wire it up** — integrate anchors into the evaluation pipeline so RECOVERY is actually tested. More work, but preserves the contribution.
- **Recommendation:** Option A for now. We can add it back in Phase 3 if we implement live integration.
- [ ] Decision made
- [ ] Done

#### P1-05: Add multi-model support to experiment pipeline
- **File:** `src/kvguard/experiment.py`, `src/kvguard/config.py`
- **Change:** Parameterize model name/path. Currently hardcoded to Qwen2.5-3B-Instruct. Need to support at least:
  - `Qwen/Qwen2.5-7B-Instruct`
  - `meta-llama/Llama-3.1-8B-Instruct`
- **Also:** Add `model_name` field to trace output JSON and `TraceMeta`.
- **Tests:** Update config tests, feature extraction tests.
- [ ] Done

#### P1-06: Scale prompt sampling to 500
- **File:** `src/kvguard/prompts.py`
- **Change:** Support configurable prompt count (currently hardcoded to 50). Add `--n-prompts` CLI arg. Use deterministic seed for reproducible sampling from GSM8K test set (1,319 problems).
- **Tests:** Verify deterministic sampling, no duplicate prompts.
- [ ] Done

#### P1-07: Add feature ablation to run_ablations.py
- **File:** `scripts/run_ablations.py`
- **Change:** Add ablation configs:
  - Full features (baseline)
  - Without `compression_ratio`
  - Without `rep_count` + `rep_count_sum_8`
  - Without both (logit-only)
- Each retrains the predictor and re-runs controller evaluation.
- [ ] Done

#### P1-08: Add CUDA support / device selection
- **File:** `src/kvguard/experiment.py`
- **Change:** Auto-detect device (CUDA > MPS > CPU). Add `--device` CLI override. Ensure model and KV cache land on the correct device.
- **Tests:** Unit test for device selection logic.
- [ ] Done

---

### Phase 2: GPU Sweep (requires 5090)

All compute-intensive work. Run on 5090 with CUDA.

#### P2-01: Run sweep — Qwen2.5-7B-Instruct
- **Config:** 500 prompts × 16 configs (3 compressors × 5 ratios + baseline)
- **Output:** 8,000 traces
- **Est. time:** ~12-18h on 5090 (8B model, CUDA, ~8-12s/prompt)
- [ ] Done

#### P2-02: Run sweep — Llama-3.1-8B-Instruct
- **Config:** Same 500 prompts × 16 configs
- **Output:** 8,000 traces
- **Est. time:** ~12-18h on 5090
- [ ] Done

#### P2-03: Verify dataset integrity
- **Run:** `kvguard verify` on both model datasets
- **Check:** Row counts, label distributions, feature completeness, no NaN/Inf
- [ ] Done

---

### Phase 3: Train, Evaluate, Ablate (GPU, fast)

All XGBoost training — minutes, not hours.

#### P3-01: Train predictor with prompt-level split (per model)
- Train on each model's dataset separately
- Report AUROC, F1, recall@5%FPR, trace detection rate, lead time
- Compare to v1 numbers — how much does prompt-level split hurt?
- [ ] Done

#### P3-02: Feature ablation suite
- For each model, train 4 predictor variants:
  - Full features
  - Without `compression_ratio`
  - Without `rep_count` + `rep_count_sum_8`
  - Logit-only (without both)
- Report AUROC for each. This answers: "does the predictor actually forecast from logit signals?"
- [ ] Done

#### P3-03: Leave-one-compressor-out CV (prompt-aware)
- Proper CV with prompt-level integrity
- Report per-compressor AUROC
- [ ] Done

#### P3-04: Cross-model transfer
- Train on Qwen traces, evaluate on Llama (and vice versa)
- This is the strongest generalization test — if it works, failure signatures are model-universal
- [ ] Done

#### P3-05: Controller evaluation with new predictor
- Run offline simulation on both models
- Report CFR reduction, FP rate, per-compressor breakdown
- Compare balanced/conservative/aggressive operating points
- [ ] Done

#### P3-06: Updated ablation suite
- Always-safe ceiling, random predictor, hysteresis, threshold sweep
- On both models, with prompt-level split
- [ ] Done

---

### Phase 4: Live Validation (stretch, GPU)

#### P4-01: Implement live token-by-token controller
- Integrate `RiskController` into the autoregressive generation loop
- At each token: extract features → predict → controller.step() → adjust compression ratio
- Start with StreamingLLM (simplest to modify dynamically)
- [ ] Done

#### P4-02: Run live validation (small scale)
- 50 prompts, 1 compressor (StreamingLLM), 2 ratios (0.75, 0.875)
- Compare: static compression vs. kvguard-controlled
- Measure: actual CFR reduction, memory savings retained, latency overhead
- This bounds the offline simulation gap
- [ ] Done

---

### Phase 5: Paper Revision

#### P5-01: Redefine metrics
- Split CFR into: **Catastrophic Behavior Rate (CBR)** = looping + non-termination, and **task accuracy** = correct answers
- Reframe narrative: "compressors with similar accuracy can have wildly different CBR"
- [ ] Done

#### P5-02: Add figures
- System architecture (4-mode state machine)
- CBR vs. compression ratio curves (baseline vs. controlled, per compressor)
- Pareto frontier from threshold sweep
- Example trace: hazard probability rising before catastrophe
- Feature ablation comparison bar chart
- [ ] Done

#### P5-03: Update all sections with new results
- New tables with 500-prompt, 2-model, prompt-split results
- Feature ablation results
- Cross-model transfer results
- Consolidate duplicate content between Results and Discussion
- [ ] Done

#### P5-04: Recompile and verify
- [ ] Done

---

## Dependency Graph

```
P1-01 through P1-08 (parallel, no GPU needed)
        │
        ▼
P2-01, P2-02 (parallel, need 5090)
        │
        ▼
      P2-03
        │
        ▼
P3-01 through P3-06 (mostly parallel, fast)
        │
        ▼
P4-01 → P4-02 (stretch)
        │
        ▼
P5-01 through P5-04
```

## Resource Estimates

| Phase | Hardware | Time |
|-------|----------|------|
| Phase 1 | MacBook (CPU only) | 1-2 days coding |
| Phase 2 | 5090 GPU | ~24-36h sweep time |
| Phase 3 | 5090 GPU (XGBoost, fast) | 2-4h |
| Phase 4 | 5090 GPU | 4-8h |
| Phase 5 | Any | 1-2 days writing |
