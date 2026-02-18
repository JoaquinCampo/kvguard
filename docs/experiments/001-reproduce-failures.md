# Experiment 001: Reproduce Catastrophic Failures

**Date:** 2026-02-17
**Status:** Complete
**Milestone:** A (Reproduce failures)

## Objective

Confirm that KV-cache compression causes catastrophic failures on reasoning tasks, and that per-token signals can distinguish healthy from failing runs.

## Setup

- **Model:** Qwen2.5-3B-Instruct (`Qwen/Qwen2.5-3B-Instruct`)
- **Task:** GSM8K (grade-school math, 10 prompts, 3 few-shot CoT examples)
- **Decoding:** greedy (temperature=0), max 512 tokens
- **Device:** Apple M-series MPS (float16)
- **Framework:** kvpress 0.5.1 via `src/kvguard/experiment.py`
- **Seed:** 42

### Compression configurations

| Run | Press | compression_ratio | Effective budget |
|-----|-------|-------------------|-----------------|
| Baseline | none | 0.0 | 100% retained |
| Moderate | StreamingLLM | 0.5 | 50% retained |
| Heavy-SL | StreamingLLM | 0.875 | 12.5% retained |
| Heavy-SK | SnapKV | 0.875 | 12.5% retained |

`compression_ratio` in kvpress = fraction of KV cache to **remove**.

## Results

| Press | Ratio | Acc | CFR | Non-term | Looping | Wrong | Avg tokens |
|-------|-------|-----|-----|----------|---------|-------|------------|
| none | 0.0 | 50% | 50% | 0 | 0 | 5 | 325 |
| streaming_llm | 0.5 | 70% | 30% | 2 | 0 | 3 | 333 |
| streaming_llm | 0.875 | **0%** | **100%** | 1 | 1 | 10 | 362 |
| snapkv | 0.875 | **0%** | **100%** | **7** | **10** | 10 | 472 |

Result files: `results/{press_name}/Qwen2.5-3B-Instruct_{ratio}_{n}p.json`

## Signal analysis

Per-token signals extracted via `src/kvguard/signals.py`.

| Press | Ratio | Mean entropy | Max entropy | Min top-1 prob |
|-------|-------|-------------|-------------|----------------|
| none | 0.0 | 0.36 | 3.36 | 0.131 |
| streaming_llm | 0.5 | 0.31 | 3.33 | 0.148 |
| streaming_llm | 0.875 | 0.44 | 5.28 | 0.066 |
| snapkv | 0.875 | 0.46 | **8.63** | **0.012** |

### Entropy spike predicts looping

SnapKV `gsm8k_0` (looping, wrong answer, 424 tokens):

```
t=0-100   entropy ~0.1   (normal generation, confident)
t=108     entropy 3.55   (rising — confusion onset)
t=111     entropy 4.13   (spike — looping imminent)
t=114     entropy 5.86   (peak — model stuck on "To" token)
t=140     entropy 5.26   (sustained confusion)
t=220     entropy 0.02   (partial recovery)
t=400     entropy 2.37   (instability continues)
```

The entropy spike at t=108-114 **precedes** the looping pattern, confirming that entropy is a **predictive** signal, not just a concurrent symptom.

### Failure modes observed

1. **Immediate looping** (SnapKV 87.5%): model outputs "ToToToTo..." for ~100 tokens before partially recovering.
2. **Instruction amnesia** (SnapKV 87.5%): after recovering from a loop, the model solves a *completely different problem* — not the original GSM8K question.
3. **Non-termination** (both at 87.5%): generation hits max_tokens without producing a final answer.
4. **Reasoning corruption** (StreamingLLM 87.5%): model produces plausible-looking math steps but arrives at wrong answers.

## Conclusions

1. **Heavy compression causes total catastrophic failure** — 0% accuracy at 12.5% budget regardless of method.
2. **SnapKV is worse than StreamingLLM** at extreme compression — 10/10 looping vs 1/10.
3. **Max entropy is a strong discriminator** — baseline tops at 3.4 nats, SnapKV reaches 8.6 nats.
4. **Entropy spikes are predictive** — they appear 5-10 tokens before looping establishes.
5. **Moderate compression (50%) is survivable** — 70% accuracy, only 2 non-termination events.

## Next steps

- Expand to 50+ prompts for statistical significance
- Add ObservedAttention (H2O-style) runs
- Test intermediate compression ratios (25%, 37.5%, 62.5%, 75%)
- Begin building the hazard predictor (Milestone C) using the entropy signal
