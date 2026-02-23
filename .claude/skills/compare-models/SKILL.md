---
name: compare-models
description: Compare results across models (Qwen vs Llama) at matching compression configurations. Generates side-by-side tables and identifies cross-model patterns.
argument-hint: [num-prompts]
allowed-tools: Read, Grep, Glob, Bash
---

# Compare Models

Generate a comprehensive cross-model comparison for all matching compression configurations.

## Steps

1. **Find matching result files** for both models:
   ```bash
   find results -name "*Qwen*_${0:-500}p.json" | sort
   find results -name "*Llama*_${0:-500}p.json" | sort
   ```

2. **For each press × ratio**, load both models' results and extract:
   - Accuracy (% correct GSM8K answers)
   - CFR (looping + non-termination rate)
   - Individual catastrophe rates (looping, non-termination, wrong_answer)
   - Mean/max entropy, min top-1 probability
   - Average tokens generated

3. **Generate comparison tables**:

   **Table: Accuracy Comparison**
   | Press | Ratio | Qwen 7B | Llama 8B | Delta |
   |-------|-------|---------|----------|-------|

   **Table: CFR Comparison**
   | Press | Ratio | Qwen CFR | Llama CFR | Delta |
   |-------|-------|----------|-----------|-------|

   **Table: Failure Mode Comparison**
   | Press | Ratio | Qwen Loop% | Llama Loop% | Qwen NT% | Llama NT% |
   |-------|-------|------------|-------------|----------|-----------|

4. **Identify patterns**:
   - Which model is more robust to compression overall?
   - Do they fail at the same compression ratios or different ones?
   - Are failure modes the same (both loop) or different (one loops, one non-terminates)?
   - Where does the "cliff" occur for each model × compressor?

5. **Cross-model transfer implications**:
   - If failure signatures are similar across models → predictor may transfer
   - If failure modes differ → per-model predictors may be needed
   - This directly informs Phase 3 (P3-04: Cross-model transfer experiment)
