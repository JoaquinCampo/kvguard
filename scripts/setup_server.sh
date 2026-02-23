#!/usr/bin/env bash
# One-time server setup after rsync. Installs deps, downloads models, verifies CUDA.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Installing uv ==="
if command -v uv &>/dev/null; then
    echo "uv already installed: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo ""
echo "=== Syncing dependencies ==="
uv sync --group dev

echo ""
echo "=== Running quality gate ==="
make check

echo ""
echo "=== Pre-downloading models ==="
bash scripts/download_models.sh

echo ""
echo "=== Verifying CUDA ==="
uv run python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('WARNING: CUDA not available — sweep will be very slow on CPU')
"

echo ""
echo "=== Ready for Phase 2 ==="
echo "Run: nohup bash scripts/run_phase2.sh &"
echo "Monitor: bash scripts/check_status.sh"
