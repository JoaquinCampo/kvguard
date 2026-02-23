# Known Failures and Gotchas

Documented failure patterns from development and experiments. Read this before starting new work to avoid repeating mistakes.

---

### MPS hangs during long sweeps
**When**: Running >30 prompts on Apple Silicon MPS
**What happened**: PyTorch MPS backend stops responding mid-generation. Process appears running but produces no output. GPU memory is not released.
**Root cause**: MPS memory management is less mature than CUDA. Long-running inference accumulates fragmented GPU memory.
**Fix/workaround**: Use process isolation — run each config in a separate process (see `scripts/sweep.sh`). Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7`. Add 15s cooldown between configs. For Phase 2, use CUDA on the 5090 instead.

### SnapKV causes 100% looping at 87.5% compression
**When**: Running SnapKV with compression_ratio=0.875 on any model
**What happened**: Model outputs "ToToToTo..." for 100+ tokens, then partially recovers and solves a different problem (instruction amnesia). 10/10 prompts affected.
**Root cause**: SnapKV's attention-based voting selects tokens once at prefill. The selection becomes catastrophically stale during long generation at extreme compression.
**Fix/workaround**: This is expected behavior — it's what KVGuard is designed to detect and prevent. The entropy spike at onset (3.5→5.9 nats over 6 tokens) is the predictive signal.

### rank_of_chosen is always 0 under greedy decoding
**When**: Extracting per-token signals with `do_sample=False`
**What happened**: The feature was always 0, contributing nothing to the predictor.
**Root cause**: Under greedy decoding, the chosen token is always the top-1 token, so its rank is always 0.
**Fix/workaround**: Removed from the feature set in Phase 1 (P1-03).

### v1 data leakage: prompt-level split violation
**When**: Training the hazard predictor in v1
**What happened**: Train/test split was by trace, not by prompt_id. The same GSM8K prompt appeared in both partitions under different compressor/ratio configs. AUROC was inflated.
**Root cause**: `split_by_trace()` treated each (prompt, compressor, ratio) as independent. But traces from the same prompt share the same question text and correct answer.
**Fix/workaround**: Fixed in Phase 1 (P1-01). Split is now by prompt_id — all traces for a given prompt go to the same partition.

### compression_ratio dominates feature importance
**When**: Analyzing XGBoost feature importances in v1
**What happened**: `compression_ratio` was the most important feature. The predictor was partially a lookup table (high ratio → high risk).
**Root cause**: compression_ratio is a static per-trace constant. It's highly correlated with catastrophe occurrence by construction.
**Fix/workaround**: Feature ablation added in Phase 1 (P1-02). Must demonstrate that logit-only features (without compression_ratio) still achieve useful AUROC.

### rep_count is concurrent, not predictive
**When**: Evaluating feature contribution to hazard prediction
**What happened**: `rep_count` spikes at/after looping onset, not before. It inflates AUROC for "forecast H=32 tokens ahead" claim.
**Root cause**: The feature counts n-gram repetitions in a trailing window — by definition, it detects looping after it has started.
**Fix/workaround**: Acknowledged in Phase 1. Feature ablation (no_rep variant) isolates the contribution of genuinely predictive logit signals.

### Answer extraction takes first #### match
**When**: Parsing model output for GSM8K answer
**What happened**: If the model mentions `####` in its reasoning (e.g., referencing the format), the first match is used, which may not be the final answer.
**Root cause**: `re.search(r"####\s*([\d,.-]+)", text)` returns the first match.
**Fix/workaround**: Use the last match instead. Addressed in Phase 1 fixes.

### Offline simulation assumes instant recovery
**When**: Evaluating controller CFR reduction
**What happened**: The simulation assumes that switching to r_safe=0 mid-generation produces the same output as never compressing. This is untested.
**Root cause**: Offline evaluation cross-references the same prompt at safe_compression_ratio. But the generation path would diverge after the switch point.
**Fix/workaround**: Acknowledged as a limitation. Phase 4 (live validation) bounds this gap with actual token-by-token controller integration.

### XGBoost training OOM on MPS with large datasets
**When**: Training on >50k token-level samples on Apple Silicon
**What happened**: XGBoost exhausts GPU memory on MPS backend.
**Root cause**: XGBoost's GPU tree method allocates large temporary buffers.
**Fix/workaround**: Use `tree_method="hist"` (CPU) for local development. CUDA handles larger datasets without issue.
