#!/usr/bin/env bash
# Quick Phase 2 progress check. Run from SSH to see current state.
cd "$(dirname "$0")/.."

echo "=== Phase 2 Status ==="
if [ -f results/phase2_status.json ]; then
    python3 -m json.tool results/phase2_status.json
else
    echo "No status file yet"
fi

echo ""
echo "=== Result files ==="
FOUND=$(find results -name "*_500p.json" 2>/dev/null | wc -l | tr -d ' ')
echo "$FOUND of 32 expected (2 models x 16 configs)"

echo ""
echo "=== Checkpoint files (in-progress) ==="
find results -name "*.ckpt.jsonl" -exec wc -l {} \; 2>/dev/null || echo "None"

echo ""
echo "=== Last 5 log lines ==="
# shellcheck disable=SC2012
LATEST_LOG=$(ls -t logs/phase2_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    tail -5 "$LATEST_LOG"
else
    echo "No log files yet"
fi
