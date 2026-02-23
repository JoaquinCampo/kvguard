---
name: analyze-results
description: Analyze sweep results after completion. Computes accuracy, CFR, signal statistics, and generates comparison tables. Use after a sweep completes to understand the data.
argument-hint: [num-prompts]
disable-model-invocation: true
allowed-tools: Read, Bash, Grep, Glob
---

# Analyze Sweep Results

Run a structured analysis of sweep results.

## Steps

1. **Verify data completeness**:
   ```bash
   uv run kvguard verify --output-dir results --num-prompts ${0:-500}
   ```

2. **Run built-in analysis**:
   ```bash
   uv run kvguard analyze --output-dir results --num-prompts ${0:-500}
   ```

3. **Generate comparison tables** by reading result files:

   For each result file in `results/`, extract from the JSON:
   - `summary.accuracy` — correct answer rate
   - `summary.cfr` — catastrophic failure rate (looping + non-termination)
   - Per-compressor breakdown by ratio

4. **Key comparisons to produce**:

   **Table 1: Accuracy × Compression Budget**
   | Model | Press | Ratio | Accuracy | CFR | Non-term | Looping |
   |-------|-------|-------|----------|-----|----------|---------|

   **Table 2: Signal Statistics**
   | Model | Press | Ratio | Mean Entropy | Max Entropy | Min Top-1 Prob |
   |-------|-------|-------|-------------|-------------|----------------|

   **Table 3: Model Comparison (same press × ratio)**
   | Press | Ratio | Qwen Acc | Llama Acc | Qwen CFR | Llama CFR |
   |-------|-------|----------|-----------|----------|-----------|

5. **Flag anomalies**:
   - Any config with 0% accuracy at moderate compression (≤0.5)
   - Any config with >50% CFR at light compression (≤0.25)
   - Configs where one model fails but the other doesn't

## Output

Present findings as structured Markdown tables. Highlight the "cliff point" where accuracy drops sharply for each model × compressor combination.
