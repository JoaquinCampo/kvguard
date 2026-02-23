#!/usr/bin/env bash
# Pre-download HuggingFace models to local cache.
# Run this before starting the sweep to avoid network issues during GPU time.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Downloading models to HF cache ==="

uv run python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer

models = [
    'Qwen/Qwen2.5-7B-Instruct',
    'meta-llama/Llama-3.1-8B-Instruct',
]

for model_name in models:
    print(f'Downloading {model_name}...')
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto')
    print(f'  Done: {model_name}')

print('All models downloaded.')
"
