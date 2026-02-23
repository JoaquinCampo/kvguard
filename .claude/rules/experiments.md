---
paths:
  - "scripts/**"
  - "results/**"
---

# Experiment and Script Conventions

- **Never edit result JSON files** — they are experiment output, not source code.
- Shell scripts must use `set -euo pipefail` and `cd "$(dirname "$0")/.."` for robustness.
- All scripts should be `chmod +x` and have a shebang line.
- Sweep results are saved to `results/{press}/{ModelShort}_{ratio}_{num_prompts}p.json`.
- Checkpoint files are `.ckpt.jsonl` — these allow automatic resume on crash.
- The sweep CLI handles checkpointing internally — don't add external checkpoint logic.
- `scripts/run_ablations.py` uses `sys.path.insert` to import from `src/` — this is intentional for standalone execution.
- MPS (Apple Silicon) is unstable for long sweeps — process isolation per config is required (see `scripts/sweep.sh`).
- CUDA is the target for Phase 2. MPS is for local development/testing only.
