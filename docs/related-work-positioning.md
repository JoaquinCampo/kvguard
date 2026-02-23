# Related Work Positioning: kvguard vs. Existing Systems

**Date:** 2026-02-20
**Purpose:** Articulate how kvguard differs from systems that partially address runtime detection, recovery, or signal integration for KV-cache compression. Required before paper writing.

---

## 1. Summary Comparison

| Dimension | ASR-KF-EGR | RefreshKV | ERGO | ThinKV | Rethinking KV | DefensiveKV | UNComp | **kvguard** |
|---|---|---|---|---|---|---|---|---|
| **Architecture** | Own compressor (soft freeze) | Attention alternation | Context rewriter | Hybrid eviction+quant | Pre-generation router | Static prefill defense | Entropy-driven policy | **Wrapper around any compressor** |
| **Runtime detection** | Proposed (unimplemented) | None | Entropy threshold | None | Pre-generation only | None | None | **Trained hazard predictor** |
| **Signals used** | Vague "entropy spikes" | Attention scores | Rolling entropy delta | Attention sparsity | Complexity estimator | Importance stability | Matrix entropy | **39-dim HALT+logit features + receiver heads** |
| **Recovery mechanism** | Proposed 4-level (unimplemented) | Full-cache refresh | Full context reset | None | Route to full cache | None | None | **Selective KV recomputation** |
| **Memory savings** | 55-67% active KV | None (full cache maintained) | N/A (resets) | Yes | Varies (routing) | Yes | Yes | **Preserves compressor savings** |
| **Compressor-agnostic** | No (is the compressor) | No (is the compressor) | Yes (but not cache-aware) | No | Partial (routing) | No | No | **Yes** |
| **Reasoning evaluation** | None (passkey, QA) | Summarization, retrieval | MATH, HotpotQA | GSM8K, MATH | Not specified | LongBench | LongBench | **GSM8K with CFR metric** |
| **Overhead** | 5x slowdown | ~0 (interleaved) | Reset cost | ~0 | Evaluator cost | ~0 | ~0 | **~0 (logit signals are free)** |

---

## 2. Per-System Analysis

### 2.1 ASR-KF-EGR (arXiv:2512.11221) — Closest Superficial Match

**What they propose:** A reversible soft-freeze mechanism that suspends KV updates for low-importance tokens (moved to CPU, not permanently evicted), with sublinear freeze scheduling (d_j = floor(sqrt(c_j)/k)) and a 4-level entropy-guided recovery system:
- Soft Reset (SR): unfreeze tokens with d > 1
- Window Reset (WR): unfreeze last N steps
- Full Reset (FR): clear all freeze durations
- Rewalk Regeneration (RR): regenerate last k tokens after FR

**Why it looks similar:** 4-level escalation parallels kvguard's NORMAL/ALERT/SAFE/RECOVERY. Both motivated by catastrophic compression failures. Both mention entropy as a detection signal.

**Critical differences:**

1. **The recovery system is unimplemented.** Section 3.6 is labeled "Future Work." No entropy thresholds are defined, no signal computation is specified, no triggering mechanism is implemented, no evaluation is performed. The 4-level system exists only as a bullet-point proposal.

2. **ASR-KF-EGR is its own compressor, not a wrapper.** It implements soft-freeze as the compression mechanism. kvguard wraps StreamingLLM, SnapKV, H2O, or any other kvpress-compatible compressor. This is a fundamental architectural difference: ASR-KF-EGR competes with compressors; kvguard makes any compressor safer.

3. **No reasoning task evaluation.** Evaluated on: 500-token generation (memory test), passkey retrieval (~1500 tokens), and a quantum physics explanation. No GSM8K, no MATH, no multi-step reasoning. The paper cannot speak to catastrophe prevention in reasoning.

4. **5x overhead from CPU-GPU transfers.** The soft-freeze mechanism requires moving KV pairs between GPU and CPU memory. Their Table 1 shows 38.96s vs 7.55s baseline (5.2x slowdown). kvguard's HALT logit features are a byproduct of decoding — essentially zero additional cost.

5. **No trained predictor.** ASR-KF-EGR proposes (but doesn't implement) threshold-based entropy monitoring. kvguard trains an XGBoost hazard predictor on a 39-dimensional feature vector with hysteresis-based mode transitions, calibrated on actual catastrophe data.

6. **Reversibility vs. recovery.** ASR-KF-EGR's soft freeze is reversible by design (tokens are on CPU, not deleted). This is elegant but requires maintaining all tokens in CPU memory — the "savings" are in active GPU memory only. kvguard operates with compressors that permanently evict tokens, and provides selective KV recomputation through receiver-head-identified anchors when recovery is needed.

**What we learn from ASR-KF-EGR:** The 4-level escalation idea validates our multi-mode controller design. The soft-freeze reversibility principle is sound — but the fact that their recovery is unimplemented and unevaluated means kvguard would be the first to actually build, evaluate, and validate a multi-level failure response system for KV-cache compression.

### 2.2 RefreshKV (arXiv:2411.05787) — Genuine Recovery, Zero Memory Savings

**What they do:** Alternate between partial-cache attention (top-K tokens) and full-cache attention, refreshing the partial cache's token selection based on the latest full-attention step. Refresh frequency is controlled by query similarity (cosine threshold, adaptive per layer).

**Key strength:** Genuine recovery — tokens are never evicted, so any "mistake" in partial-cache selection is corrected at the next full-attention step. Recovers 52% of lost performance on HTML-to-TSV (Llama-3.1-8B) versus 0% for SnapKV/H2O.

**Critical differences:**

1. **No memory savings.** RefreshKV maintains the full KV cache throughout inference. Memory footprint equals vanilla attention. The savings are in compute (fewer attention operations on partial steps), not memory. kvguard preserves the compressor's memory savings.

2. **Not failure-triggered.** RefreshKV refreshes on a schedule (every S steps) or when query similarity drops below a threshold. It does not detect *failures* — it periodically recalibrates regardless of whether anything went wrong. This means it pays the refresh cost even when compression is working perfectly.

3. **No failure detection.** No concept of catastrophe, no signal monitoring, no risk score. The full-attention steps are defensive but blind.

**What we learn from RefreshKV:** Periodic recalibration is a viable strategy, and the query-similarity threshold for triggering refresh is an interesting adaptive mechanism. However, the zero-memory-savings constraint makes this approach fundamentally different from kvguard's problem domain (making compression safe, not replacing it).

### 2.3 ERGO (arXiv:2510.14077) — Entropy-Triggered, Sledgehammer Response

**What they do:** Monitor rolling entropy delta between generation windows. When ΔH > τ (calibrated at 85-90th percentile on ~80 examples, model-specific: 0.03 for Llama-8B to 0.3 for GPT-4o), trigger a full context reset: rewrite the prompt and regenerate from scratch.

**Key strength:** Simple, effective. 56.6% average performance gain, 35.3% unreliability reduction. Proves that entropy signals reliably detect generation quality degradation.

**Critical differences:**

1. **Full context reset, not surgical intervention.** ERGO destroys all generated text and starts over. kvguard implements graduated response — ALERT (protect thinking tokens), SAFE (relax compression ratio), RECOVERY (selective KV recomputation for anchors) — before ever considering a reset.

2. **Not cache-aware.** ERGO operates at the generation level, not the cache management level. It doesn't adjust compression ratios, protect specific tokens, or modify cache behavior. It's a generation-quality safety net, not a compression controller.

3. **Single signal (entropy delta).** kvguard uses a 39-dimensional feature vector (HALT logit features, rolling entropy, repetition signals, thinking-token flags) trained into a hazard predictor, versus ERGO's single threshold on rolling ΔH.

4. **No lead time.** ERGO detects degradation after it has occurred (entropy has already shifted). kvguard's hazard predictor aims to predict catastrophe H tokens *before* onset, providing lead time for intervention.

**What we learn from ERGO:** Entropy delta works. The calibration approach (percentile-based, model-specific) is practical. ERGO validates the signal — kvguard refines the response.

### 2.4 ThinKV (arXiv:2510.01290) — Mode-Aware But No Detection

**What they do:** Classify reasoning into three modes via attention sparsity metrics (computed over 4 optimal layers, classified every 128 tokens using KDE-calibrated thresholds):
- Reasoning (R): high attention density → conservative compression (4-bit quant)
- Transition (T): medium → moderate compression
- Execution (E): low density → aggressive compression (2-bit quant)

**Key strength:** First method to tie compression policy to reasoning phase. Acknowledges that different phases tolerate different compression.

**Critical differences:**

1. **No failure detection.** ThinKV classifies reasoning mode but cannot detect when its classification is wrong. If a token is misclassified as Execution and aggressively compressed, there is no correction.

2. **No recovery.** ThinKV explicitly acknowledges that "complete eviction causes reasoning loops" but provides no mechanism to detect loops or recover from them.

3. **Own compressor, not a wrapper.** ThinKV implements its own hybrid eviction+quantization scheme. It cannot be applied on top of other compressors.

4. **128-token granularity.** Mode classification happens every 128 tokens. kvguard's per-token risk scoring detects transitions at single-token granularity.

**What we learn from ThinKV:** Mode-aware compression is the right idea. The R/T/E classification scheme informed our NORMAL/ALERT/SAFE/RECOVERY design. But ThinKV's approach is open-loop (classify → compress, no verification), while kvguard closes the loop (compress → monitor → intervene if failing).

### 2.5 Rethinking KV-Cache Compression (arXiv:2503.24000) — Pre-Generation Routing

**What they do:** Use a pre-generation complexity evaluator to route inputs to either compressed or full-cache inference paths. The routing decision is made *before* generation begins.

**Critical differences:**

1. **Pre-generation, not runtime.** The routing decision is static: once a prompt is classified, the compression policy is fixed for the entire generation. No monitoring during generation, no adaptation.

2. **Binary routing.** Compressed or full — no graduated response. kvguard has four modes with continuous risk scoring.

**What we learn:** Pre-generation routing addresses a real problem (some inputs are more sensitive than others) but misses within-generation dynamics. Compression failures often emerge mid-generation as the model's reasoning state evolves.

### 2.6 DefensiveKV (arXiv:2510.13334) — Passive Static Insurance

**What they do:** Max-based aggregation with adaptive prior correction applied at prefill. Instead of using latest attention scores for importance, use max-over-history to prevent premature eviction of tokens that were important at any point.

**Key strength:** Proves the stability assumption is false (retained importance drops to 0.34 worst case). 4.3x quality loss reduction on LongBench.

**Critical differences:**

1. **Static, applied once at prefill.** DefensiveKV does not monitor generation, does not detect failures, does not adapt during inference. It is passive insurance.

2. **No recovery.** If the max-aggregated importance is still wrong (it can be — 0.34 worst case), there is no correction.

**What we learn:** The empirical evidence that importance is non-stationary (retained importance drops to 0.34) is the strongest motivation for closed-loop monitoring. If importance scoring is fundamentally unstable, the only robust approach is continuous monitoring and intervention.

### 2.7 UNComp (arXiv:2410.03090) — Entropy-Driven Static Policy

**What they do:** Use matrix entropy to drive inter-layer and inter-head compression decisions. Layers/heads with lower entropy (more concentrated attention) can be compressed more aggressively.

**Critical difference:** Static policy optimization using entropy as input, not closed-loop monitoring. Entropy informs the initial compression decision but does not trigger runtime intervention.

---

## 3. kvguard's Differentiating Claims

### Claim 1: First validated closed-loop controller for KV-cache compression

No existing system monitors compressed generation quality and feeds that information back into cache management decisions. ASR-KF-EGR proposes this but doesn't implement or evaluate it. kvguard implements a complete detect → decide → act → monitor loop.

### Claim 2: Compressor-agnostic wrapper architecture

kvguard wraps any kvpress-compatible compressor (StreamingLLM, SnapKV, H2O, ObservedAttention) without modifying the compressor. This is orthogonal to compression research — every improvement in compression methods automatically benefits from kvguard's safety layer. No other system provides this.

### Claim 3: Trained hazard predictor, not threshold heuristics

ERGO uses a single entropy threshold. ASR-KF-EGR proposes (but doesn't implement) entropy monitoring. kvguard trains an XGBoost classifier on a 39-dimensional per-token feature vector (HALT logit features, rolling entropy, repetition signals, thinking-token flags) to predict catastrophe within H tokens. The predictor is evaluated with precision/recall/lead-time metrics and validated with leave-one-compressor-out cross-validation.

### Claim 4: Selective recovery via receiver-head-identified anchors

ERGO resets everything. RefreshKV maintains the full cache (no memory savings). kvguard identifies anchor tokens through receiver-head attention concentration (arXiv:2506.19143) and performs selective KV recomputation only for those anchors. This is the first selective, failure-triggered recovery mechanism that preserves memory savings.

### Claim 5: Catastrophic Failure Rate as primary metric

Standard evaluation uses average accuracy/perplexity, which hides catastrophic failures (95% average accuracy can mean 0% on the 5% where compression causes catastrophe). kvguard introduces CFR vs. budget curves as the primary evaluation metric, directly measuring what matters: does the controller prevent the compressor from failing catastrophically?

---

## 4. Positioning Statement (for paper introduction)

> Existing KV-cache compression methods are open-loop: they make eviction decisions without verifying whether those decisions degraded generation quality. Recent work has begun to recognize this gap — ASR-KF-EGR (2512.11221) proposes but does not implement an entropy-guided recovery system, RefreshKV (2411.05787) maintains the full cache for periodic recalibration at the cost of zero memory savings, and ERGO (2510.14077) demonstrates that entropy signals can trigger corrective action but applies only a full context reset. We present kvguard, the first validated closed-loop controller that wraps any existing KV-cache compressor, monitors generation quality through zero-cost logit features, and dynamically adjusts compression aggressiveness to prevent catastrophic failures before they become irreversible.

---

## 5. What We Must Not Overclaim

1. **ASR-KF-EGR's reversibility is genuinely novel.** Their soft-freeze mechanism (move to CPU, not delete) is a cleaner approach to avoiding irreversible eviction. We should cite this fairly and note that our approach addresses a different problem: making irreversible compressors safer, not replacing them with reversible ones.

2. **RefreshKV's recovery works.** 52% recovery on HTML-to-TSV is real. Our selective recovery approach has not yet been validated — it is a design, not a result. Until T-007 integration testing, this is a proposed advantage, not a demonstrated one.

3. **ERGO's simplicity is a feature.** A single entropy threshold with full reset is simple, robust, and effective (56.6% gain). Our multi-signal, multi-mode approach may be better but is also more complex. We must demonstrate that the added complexity provides sufficient improvement.

4. **ThinKV's mode classification is related.** Their R/T/E modes and our NORMAL/ALERT/SAFE/RECOVERY modes share the insight that different reasoning phases need different compression. We differentiate on closed-loop monitoring, not the multi-mode concept itself.
