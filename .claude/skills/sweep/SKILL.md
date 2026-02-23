---
name: sweep
description: Prepare and run a KV-cache compression sweep. Loads sweep configuration, validates prerequisites, and provides the exact commands needed. Use before starting any GPU experiment.
argument-hint: [model-name] [num-prompts]
disable-model-invocation: true
allowed-tools: Read, Bash, Grep, Glob
---

# Sweep Preparation and Execution

Before running a sweep, verify prerequisites and provide the correct commands.

## Steps

1. **Check environment**:
   - Run `uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"` to verify GPU access
   - Run `make check` to ensure code is clean

2. **Check existing results**:
   - Count files: `find results -name "*_$1p.json" 2>/dev/null | wc -l`
   - List what's done: `find results -name "*_$1p.json" -exec basename {} \;`
   - Check checkpoints: `find results -name "*.ckpt.jsonl" -exec wc -l {} \;`

3. **Validate model access**:
   - Check if model is cached: `ls ~/.cache/huggingface/hub/models--$(echo "$0" | tr '/' '--')/ 2>/dev/null`
   - If not cached, suggest: `bash scripts/download_models.sh`

4. **Provide the sweep command**:
   ```bash
   uv run kvguard sweep \
     --num-prompts ${1:-500} \
     --model "${0:-Qwen/Qwen2.5-7B-Instruct}" \
     --output-dir results \
     --max-new-tokens 512
   ```

5. **For the full Phase 2 pipeline** (both models + train + eval):
   ```bash
   nohup bash scripts/run_phase2.sh &
   # Monitor with: bash scripts/check_status.sh
   ```

## Sweep configuration

16 configs per model:
- **Baseline**: none @ 0.0
- **StreamingLLM**: 0.25, 0.5, 0.625, 0.75, 0.875
- **SnapKV**: 0.25, 0.5, 0.625, 0.75, 0.875
- **ObservedAttention**: 0.25, 0.5, 0.625, 0.75, 0.875

**Expected output**: `results/{press}/{ModelShort}_{ratio}_{num_prompts}p.json`

**Checkpoint files**: `results/{press}/{ModelShort}_{ratio}.ckpt.jsonl` (auto-resume on restart)

## Time estimates (5090 GPU)
- 500 prompts x 16 configs ≈ 12-18h per model
- Full Phase 2 (2 models) ≈ 48-60h total including train/eval
