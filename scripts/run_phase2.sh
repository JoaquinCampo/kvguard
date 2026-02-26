#!/usr/bin/env bash
# Master orchestration script for Phase 2 GPU sweep.
#
# Runs the full pipeline: sweep both models, verify, train, eval, ablations.
# Logs to logs/phase2_YYYYMMDD_HHMMSS.log (tee'd to stdout).
# Writes results/phase2_status.json after each step for remote monitoring.
#
# Usage:
#   nohup bash scripts/run_phase2.sh &
#   # or inside tmux/screen
set -euo pipefail

cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/phase2_${TIMESTAMP}.log"
STATUS_FILE="results/phase2_status.json"
NUM_PROMPTS=500
MAX_NEW_TOKENS=512
STARTED_AT=$(date -u +"%Y-%m-%dT%H:%M:%S")

mkdir -p "$LOG_DIR" results

# ---------------------------------------------------------------------------
# Status tracking
# ---------------------------------------------------------------------------

STEPS_COMPLETED='[]'
ERRORS='[]'

update_status() {
    local step="$1" model="$2" status="$3"
    cat > "$STATUS_FILE" <<EOJSON
{
  "phase": "phase2",
  "started_at": "$STARTED_AT",
  "current_step": "$step",
  "current_model": "$model",
  "status": "$status",
  "steps_completed": $STEPS_COMPLETED,
  "errors": $ERRORS,
  "last_updated": "$(date -u +"%Y-%m-%dT%H:%M:%S")"
}
EOJSON
}

mark_completed() {
    local label="$1"
    STEPS_COMPLETED=$(echo "$STEPS_COMPLETED" | python3 -c "
import sys, json
arr = json.load(sys.stdin)
arr.append('$label')
print(json.dumps(arr))
")
}

record_error() {
    local label="$1"
    ERRORS=$(echo "$ERRORS" | python3 -c "
import sys, json
arr = json.load(sys.stdin)
arr.append('$label')
print(json.dumps(arr))
")
}

echo "=== Phase 2 GPU Sweep ==="
echo "=== Started: $(date) ==="
echo "=== Log: $LOG_FILE ==="

# ---------------------------------------------------------------------------
# Step 1: Sweep Qwen 7B
# ---------------------------------------------------------------------------

echo ""
echo "=== Step 1/8: Sweep Qwen/Qwen2.5-7B-Instruct ==="
update_status "sweep" "Qwen/Qwen2.5-7B-Instruct" "running"

uv run kvguard sweep \
    --num-prompts "$NUM_PROMPTS" \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --output-dir results \
    --max-new-tokens "$MAX_NEW_TOKENS" 2>&1 | tee -a "$LOG_FILE"

mark_completed "sweep:Qwen/Qwen2.5-7B-Instruct"
update_status "sweep" "Qwen/Qwen2.5-7B-Instruct" "completed"

# ---------------------------------------------------------------------------
# Step 2: Sweep Llama 8B
# ---------------------------------------------------------------------------

echo ""
echo "=== Step 2/8: Sweep meta-llama/Llama-3.1-8B-Instruct ==="
update_status "sweep" "meta-llama/Llama-3.1-8B-Instruct" "running"

uv run kvguard sweep \
    --num-prompts "$NUM_PROMPTS" \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --output-dir results \
    --max-new-tokens "$MAX_NEW_TOKENS" 2>&1 | tee -a "$LOG_FILE"

mark_completed "sweep:meta-llama/Llama-3.1-8B-Instruct"
update_status "sweep" "meta-llama/Llama-3.1-8B-Instruct" "completed"

# ---------------------------------------------------------------------------
# Steps 3-6: Post-processing (failures logged but don't abort)
# ---------------------------------------------------------------------------
set +e

# Step 3: Verify data integrity
echo ""
echo "=== Step 3/8: Verify data integrity ==="
update_status "verify" "all" "running"

if uv run kvguard verify \
    --output-dir results \
    --num-prompts "$NUM_PROMPTS" 2>&1 | tee -a "$LOG_FILE"; then
    mark_completed "verify"
    update_status "verify" "all" "completed"
else
    record_error "verify failed"
    update_status "verify" "all" "failed"
    echo "WARNING: Verification failed — check results before training"
fi

# Step 4: Train predictor
# Exclude observed_attention (GQA incompatibility, see Appendix A)
# Exclude expected_attention (not reported in paper — paper uses 2 compressors: StreamingLLM, SnapKV)
echo ""
echo "=== Step 4/8: Train hazard predictor ==="
update_status "train" "all" "running"

if uv run kvguard train \
    --results-dir results \
    --num-prompts "$NUM_PROMPTS" \
    --output-dir models \
    --press-exclude observed_attention,expected_attention 2>&1 | tee -a "$LOG_FILE"; then
    mark_completed "train"
    update_status "train" "all" "completed"
else
    record_error "train failed"
    update_status "train" "all" "failed"
fi

# Step 5: Evaluate controller
echo ""
echo "=== Step 5/8: Evaluate controller ==="
update_status "eval" "all" "running"

if uv run kvguard eval-controller \
    --results-dir results \
    --model-path models/hazard_predictor.json \
    --num-prompts "$NUM_PROMPTS" \
    --output-path results/controller_eval.json 2>&1 | tee -a "$LOG_FILE"; then
    mark_completed "eval-controller"
    update_status "eval" "all" "completed"
else
    record_error "eval-controller failed"
    update_status "eval" "all" "failed"
fi

# Step 6: Run ablations
echo ""
echo "=== Step 6/8: Run ablations ==="
update_status "ablations" "all" "running"

if uv run python scripts/run_ablations.py 2>&1 | tee -a "$LOG_FILE"; then
    mark_completed "ablations"
    update_status "ablations" "all" "completed"
else
    record_error "ablations failed"
    update_status "ablations" "all" "failed"
fi

# Step 7: Generate figures for paper
echo ""
echo "=== Step 7/8: Generate figures ==="
update_status "figures" "all" "running"

if uv run python scripts/generate_figures.py 2>&1 | tee -a "$LOG_FILE"; then
    mark_completed "figures"
    update_status "figures" "all" "completed"
else
    record_error "figures failed"
    update_status "figures" "all" "failed"
fi

# Step 8: Phase transition analysis
echo ""
echo "=== Step 8/8: Phase transition analysis ==="
update_status "phase_transitions" "all" "running"

if uv run python scripts/analyze_phase_transitions.py 2>&1 | tee -a "$LOG_FILE"; then
    mark_completed "phase_transitions"
    update_status "phase_transitions" "all" "completed"
else
    record_error "phase_transitions failed"
    update_status "phase_transitions" "all" "failed"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

update_status "done" "all" "Phase 2 complete"

echo ""
echo "=== Phase 2 complete: $(date) ==="
echo "=== Log: $LOG_FILE ==="
echo "=== Status: $STATUS_FILE ==="
