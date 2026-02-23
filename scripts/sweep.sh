#!/bin/bash
# Robust sweep runner: one process per config for MPS stability.
# Each config runs in its own kvguard run invocation — full process isolation.
# Checkpointing handles resume within a config on crash.
set -euo pipefail

cd "$(dirname "$0")/.."

# Cap MPS memory to 70% of system RAM — prevents macOS OOM kills.
# The remaining 30% stays available for the OS, browser, etc.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5

NUM_PROMPTS=50
OUTPUT_DIR=results
MAX_RETRIES=3

# All configs: (press, ratio)
CONFIGS=(
    "none 0.0"
    "streaming_llm 0.25"
    "streaming_llm 0.5"
    "streaming_llm 0.625"
    "streaming_llm 0.75"
    "streaming_llm 0.875"
    "snapkv 0.25"
    "snapkv 0.5"
    "snapkv 0.625"
    "snapkv 0.75"
    "snapkv 0.875"
    "observed_attention 0.25"
    "observed_attention 0.5"
    "observed_attention 0.625"
    "observed_attention 0.75"
    "observed_attention 0.875"
)

TOTAL=${#CONFIGS[@]}
DONE=0
FAILED=0

result_file() {
    local press=$1 ratio=$2
    printf "%s/%s/Qwen2.5-3B-Instruct_%.3f_%dp.json" "$OUTPUT_DIR" "$press" "$ratio" "$NUM_PROMPTS"
}

echo "=== KVGuard Sweep: $TOTAL configs × $NUM_PROMPTS prompts ==="
echo "=== Started: $(date) ==="

for entry in "${CONFIGS[@]}"; do
    read -r press ratio <<< "$entry"
    DONE=$((DONE + 1))

    # Skip if result file exists
    rf=$(result_file "$press" "$ratio")
    if [ -f "$rf" ]; then
        echo "[$DONE/$TOTAL] SKIP $press @ $ratio — result exists"
        continue
    fi

    echo ""
    echo "[$DONE/$TOTAL] RUN $press @ $ratio"
    echo "    Started: $(date)"

    success=false
    for attempt in $(seq 1 $MAX_RETRIES); do
        if uv run kvguard run \
            --press "$press" \
            --compression-ratio "$ratio" \
            --num-prompts "$NUM_PROMPTS" \
            --output-dir "$OUTPUT_DIR" \
            --max-new-tokens 512; then
            echo "    DONE in attempt $attempt: $(date)"
            success=true
            break
        else
            echo "    CRASHED attempt $attempt/$MAX_RETRIES: $(date)"
            echo "    Checkpoint preserved — will resume on retry"
            sleep 10  # longer cooldown for MPS memory to fully release
        fi
    done

    if ! $success; then
        echo "    FAILED after $MAX_RETRIES attempts"
        FAILED=$((FAILED + 1))
    fi

    # Cool down between configs — let MPS memory fully release before next model load
    echo "    Cooling down (15s) to release MPS memory..."
    sleep 15
done

echo ""
echo "=== Sweep complete: $(date) ==="
echo "=== Configs: $TOTAL total, $FAILED failed ==="
