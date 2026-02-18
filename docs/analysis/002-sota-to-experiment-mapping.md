# Analysis 002: Mapping SOTA Predictions to Experimental Evidence

**Date:** 2026-02-17
**Purpose:** Connect findings from Experiment 001 to the 68-paper SOTA review, extract the detector design, and identify what signals to implement next.

---

## 1. Failure modes: predicted vs. observed

The SOTA (Section 4) catalogs five failure modes. Our experiment at 87.5% compression confirmed four of them on 10 GSM8K prompts.

| Failure mode | SOTA prediction | Our observation | Confirmed? |
|---|---|---|---|
| **Looping** | Softmax collapses to small token set when context lost (Closing the Curious Case, 2310.01693; ThinKV acknowledges loops) | SnapKV 87.5%: 10/10 looping. Model stuck on "To" token. Entropy drops to <0.05 during loops (matches LoopLLM's finding). | Yes |
| **Non-termination** | Occurs when problem-statement context evicted (SOTA §4.1) | SnapKV: 7/10, StreamingLLM: 1/10 hit max_tokens. Directionless generation without arriving at an answer. | Yes |
| **Instruction amnesia** | System prompt / task tokens evicted → model reverts to base behavior (Pitfalls of KV Cache Compression, 2510.00231) | SnapKV gsm8k_0: after recovering from loop, model solved a *completely different problem* — asked about ages summing to 30 instead of the original question. | Yes |
| **Reasoning path corruption** | Lost premises cause wrong conclusions; output appears fluent (SOTA §4.1 — "hardest to detect") | StreamingLLM 87.5%: plausible math steps arriving at wrong answers without looping (10/10 wrong, only 1 loop). Fluent but wrong. | Yes |
| **Structured collapse** | JSON/code format violations after structural context evicted | Not tested (GSM8K is free-form text, not structured output). | Untested |

### Key insight: method-dependent failure signatures

The SOTA treats failure modes as generic. Our data reveals they are **method-dependent**:

- **StreamingLLM** (sliding window): mostly reasoning corruption (silent failure). Low looping (1/10). The model keeps recent context and loses distant premises quietly.
- **SnapKV** (attention-based voting): mostly looping + non-termination (loud failure). The voting-window selects tokens once at prefill, then that selection becomes catastrophically stale during long generation.

This distinction matters for detection: silent failures need different signals than loud ones.

---

## 2. Signal analysis: what the SOTA says we should measure

The SOTA (Section 5.6) defines a signal stack ordered by cost. Here is each signal mapped to our experimental evidence and implementation status.

### Layer 0: Thinking-token vocabulary match — O(1)

**Paper:** Demystifying Reasoning Dynamics (2506.02867)
**What it does:** Identifies tokens at MI peaks — "So", "Wait", "Therefore", "Hmm", "Let", "First", "But", etc. These 1-5% of tokens carry disproportionate reasoning content. Suppressing them degrades reasoning; suppressing random tokens doesn't.
**Relevance:** These tokens' KV entries must be protected from eviction. A vocabulary lookup is O(1) per token.
**Our status:** Not implemented. But our looping failures show the model losing exactly these transitional markers.
**Action:** Add vocabulary-based thinking-token detection to identify KV entries for eviction immunity.

**New paper:** Deep-Thinking Tokens (2602.13517) adds "settling depth" — tokens where intermediate layers disagree with the final layer (measured by JSD). These are mechanistically grounded thinking tokens, not just vocabulary heuristics. Settling depth requires intermediate-layer logits, adding moderate cost.

### Layer 1: HALT 25-dim logit features — ~zero cost

**Paper:** HALT (2602.02888)
**What it does:** Extracts 25 features per token from top-20 log-probabilities:
- Top-20 log-probs (20 features)
- H_overall: total entropy
- H_alts: entropy excluding top choice (competitor disagreement)
- AvgLogP: distribution sharpness
- RankProxy: distance between sampled and greedy-optimal token
- DeltaH_dec: temporal entropy change between consecutive tokens (confidence shocks)

**Relevance:** DeltaH_dec directly captures compression-induced "confidence shocks." RankProxy spikes when the compressed cache produces different top-k rankings.
**HALT achieves 67% Macro-F1** on hallucination detection with a 5M-parameter GRU classifier.
**Limitation:** HALT uses a *bidirectional* GRU — cannot work causally during generation without adaptation to a causal variant.
**Our status:** We extract 4 of these 25 features (entropy, top1_prob, top5_prob, rank). Missing: top-20 log-probs, H_alts, AvgLogP, DeltaH_dec, RankProxy.
**Action:** Expand `signals.py` to extract the full HALT feature vector. DeltaH_dec (entropy delta) is trivially computed from consecutive tokens. All 25 features are zero marginal cost.

### Layer 2: Rolling entropy ΔH — ~zero cost

**Paper:** ERGO (2510.14077)
**What it does:** Computes rolling average entropy over turns. When ΔH exceeds threshold τ, triggers corrective action.
**Calibration:** τ set at 65th-90th percentile on ~80 held-out examples (varies by model — 0.03 for Llama-8B, 0.3 for GPT-4o).
**Results:** 56.6% performance gain, 35.3% unreliability reduction.
**ERGO's response is a sledgehammer:** full context reset (rewrite prompt + stateless regeneration). Our response should be surgical — mode switch (relax compression, protect anchors).
**Our status:** We don't compute rolling ΔH yet, but we have per-token entropy. Computing ΔH is trivial.
**Action:** Implement rolling ΔH as a trigger signal, calibrate threshold on our data.

### Layer 3: Lookback ratio (attention) — low cost

**Paper:** Lookback Lens (2407.07071)
**What it does:** Per attention head, computes ratio of attention mass on context tokens vs. generated tokens:

```
LR = A(context) / (A(context) + A(new))
```

A 1025-parameter logistic regression classifier on the concatenated lookback ratios across all heads/layers detects hallucinations.
**Key properties:**
- Works causally (per-token during generation)
- 1025 parameters for 32-layer, 32-head model
- Transfers across model sizes without retraining (7B → 13B)
- Can do classifier-guided decoding (sample N candidates, pick highest-factuality one)

**Relevance:** Directly measures whether the model is still attending to the original context (problem statement, few-shot examples) vs. its own generation. Under compression, evicted tokens reduce context attention → lookback ratio drops → hallucination signal rises.
**Our status:** Not implemented. Requires access to attention weights during generation.
**Cost:** Extracting attention weights adds moderate overhead. kvpress already hooks into attention layers — we could piggyback.
**Action:** This is a strong candidate for Stage 2 signals. Could detect instruction amnesia (model stops looking back at the question).

### Layer 4: Spectral entropy (attention) — moderate cost

**Paper:** Geometry of Reason (2601.00791)
**What it does:** Converts attention matrices to graphs via symmetrization, computes the graph Laplacian, extracts spectral features:
- Spectral entropy: disorder of attention structure
- Fiedler value: algebraic connectivity
- HFER: high-frequency energy ratio

**Spectral entropy at Layer 0 alone achieves >93% accuracy** distinguishing valid from invalid reasoning (Qwen-0.5B).
**Limitation:** Post-hoc only. Requires full sequence attention matrices and eigendecomposition. Not designed for online use during generation.
**Cost:** O(N² · k) for k eigenvalues. Negligible for N < 1000 but not trivially online.
**Our status:** Not implemented.
**Action:** Useful as an offline analysis tool for our dataset, not a runtime signal.

### Offline: TokUR epistemic uncertainty — 5x cost

**Paper:** TokUR (2505.11737)
**What it does:** Decomposes per-token uncertainty into aleatoric (data noise) and epistemic (model knowledge gap) via M=5 perturbed forward passes.
**EU achieves 74% AUROC** for detecting incorrect reasoning paths on GSM8K.
**Key insight for us:** KV-cache eviction is conceptually an *induced knowledge gap* — removing cached info makes the model behave as if it lacks knowledge it had. EU should spike precisely where compression causes quality loss.
**Limitation:** 5x decoding overhead. Runtime-impractical.
**Action:** Use as offline labeling oracle. Run traces with and without compression, label divergence points using TokUR EU, then train a lightweight predictor (using HALT features) to approximate these labels at runtime.

---

## 3. Critical insights from the SOTA that change our approach

### 3.1 Token-level importance prediction is a dead end

**Paper:** Limits of Learned Importance Scoring (2601.14279)
**Finding:** KV representations carry only 0.12 bits of MI about future attention, vs 0.31 bits from position alone. A 1.7M-parameter MLP doesn't beat position heuristics for scoring token importance.
**Why:** "Circular dependence" — future importance depends on future queries that are unknown at scoring time.
**Implication for us:** Don't try to predict which tokens are important (that's what compressors already fail at). Instead predict *when the system is failing* from observable symptoms. System-level failure prediction from generation signals is more tractable than token-level importance prediction from KV representations.

### 3.2 Low-entropy tokens are the most vulnerable

**Paper:** ForesightKV (2602.03203)
**Finding:** Low-entropy tokens (model is confident) suffer +147% loss increase from eviction, vs +52% for high-entropy tokens. Counter-intuitive.
**Explanation:** Low-entropy tokens encode *resolved* reasoning steps — the model is confident *because* it successfully processed the information. Removing their KV entries retroactively destabilizes the resolution.
**Implication:** Entropy is not just a failure detection signal — it's also an eviction-protection signal. The tokens that look safest to evict are the most dangerous to lose. This is a direct input to the controller's protection policy.

### 3.3 The stability assumption is empirically false

**Paper:** DefensiveKV (2510.13334)
**Finding:** Retained importance drops to 0.34 at worst case. Average stability is ~0.92 correlation, but importance shifts abruptly in specific intervals (e.g., generation steps 150-320).
**Implication:** Any eviction policy can be wrong at any time. This isn't a bug in specific compressors — it's inherent to the problem. A controller that monitors and intervenes is the appropriate response to fundamentally unreliable eviction.

### 3.4 LoopLLM confirms our entropy observation

**Paper:** LoopLLM (2511.07876)
**Finding:** Loop onset is preceded by rapid entropy convergence to low values (<0.05). The token distribution becomes concentrated on a "cyclic segment."
**Match to our data:** Our SnapKV failures show entropy spiking to 8.6 nats (confusion) before settling to near-zero (loop lock-in). The spike is the *onset*; the near-zero is the *established loop*. Both signals are useful: the spike for early warning, the near-zero for confirmation.

---

## 4. Detector architecture — informed by SOTA + experiments

### 4.1 Signal stack to implement (ordered by priority)

| Priority | Signal | Cost | Source | What it detects | Implementation effort |
|---|---|---|---|---|---|
| **P0** | DeltaH_dec (entropy delta) | ~0 | HALT / ERGO | Confidence shocks, degradation onset | Trivial — diff of existing entropy |
| **P0** | Full top-20 log-probs | ~0 | HALT | Distribution shape changes | Small — extend `signals.py` |
| **P0** | H_alts (competitor entropy) | ~0 | HALT | Uncertainty among alternatives | Small — entropy excluding top-1 |
| **P1** | Thinking-token vocabulary flag | O(1) | Demystifying Reasoning | KV entries to protect from eviction | Small — vocabulary lookup |
| **P1** | Rolling ΔH with threshold | ~0 | ERGO | Trigger for mode switching | Small — windowed average |
| **P2** | Lookback ratio (attention) | Low | Lookback Lens | Instruction amnesia, context loss | Moderate — requires attention weights |
| **P3** | Spectral entropy | O(N²·k) | Geometry of Reason | Reasoning structure fragmentation | Offline analysis only |
| **P3** | TokUR EU | 5x | TokUR | Ground-truth failure labels | Offline labeling only |

### 4.2 Detector design

**Input:** Per-token feature vector (HALT 25-dim, extended with thinking-token flag and rolling ΔH).

**Architecture options (from cheapest to most powerful):**

1. **Threshold on rolling ΔH** (ERGO-style): simplest, no training. Calibrate threshold on our data. This alone may be sufficient for loud failures (looping, non-termination).

2. **Causal GRU/LSTM on HALT features**: adapt HALT's classifier from bidirectional to causal. 5M parameters. Can predict failure risk at each token. Needs training data — use existing experiment results + expanded dataset.

3. **Threshold + Lookback ratio**: combine entropy trigger with attention-based context tracking. Catches both loud failures (entropy) and silent failures (instruction amnesia via lookback ratio drop).

### 4.3 Controller response modes

| Mode | Trigger | Action | SOTA basis |
|---|---|---|---|
| **NORMAL** | risk < τ_low | Aggressive compression (current press settings) | Default operation |
| **ALERT** | τ_low < risk < τ_high | Protect thinking-token KV entries from eviction | Demystifying Reasoning, ForesightKV |
| **SAFE** | risk > τ_high sustained K tokens | Relax compression ratio (e.g., 87.5% → 50%) | ERGO trigger concept, but surgical |
| **RECOVERY** | Confirmed loop or non-termination | Selective KV recomputation for anchor tokens, or stop + re-generate from checkpoint | ERGO full reset adapted to selective action |

---

## 5. What we do NOT yet know (open questions for expanded experiments)

1. **Does DeltaH_dec give earlier warning than raw entropy?** We saw entropy spikes at t=108. Would ΔH flag at t=105?
2. **What threshold separates normal from catastrophic?** Our data: baseline max entropy ≤ 3.4, compressed max ≥ 5.3. The threshold is somewhere in 3.5-5.0 range — need more data points.
3. **Does the lookback ratio drop before entropy spikes?** If the model stops attending to the question before entropy rises, lookback ratio gives more lead time.
4. **Does reasoning corruption (silent failure) have a logit signature?** StreamingLLM 87.5% had 10/10 wrong answers but only 1/10 looping. The silent failures may need attention signals, not just logit signals.
5. **Do these findings generalize beyond Qwen2.5-3B?** The 3B model is small. Larger models may be more robust to compression, shifting the cliff point.

---

## 6. Recommended next steps

1. **Expand `signals.py` to extract the full HALT 25-feature vector** — zero cost, immediate value.
2. **Add DeltaH_dec computation and rolling ΔH trigger** — test ERGO's threshold approach on our existing data.
3. **Run 50 prompts × 6 compression ratios × 3 methods** — fill in the degradation curve, find the cliff point precisely.
4. **Build the simplest possible detector** (rolling ΔH threshold) and measure offline precision/recall.
5. **If logit signals are insufficient for silent failures**, add lookback ratio as a Stage 2 signal.
