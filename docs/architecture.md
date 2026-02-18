# Architecture

Code lives in `src/kvguard/`. Single flat package, no subpackages.

## Modules

### `config.py` — Data models

- `ExperimentConfig`: experiment parameters (model, press, ratio, prompts, device). Built on `pydantic-settings`.
- `TokenSignals`: per-token signal snapshot (entropy, top-1/top-5 prob, rank of chosen token).
- `RunResult`: full output of a single prompt run — text, labels, signals, catastrophe tags.

### `prompts.py` — Prompt loading

- `FEWSHOT_EXAMPLES`: 5 hardcoded GSM8K few-shot CoT examples.
- `load_gsm8k(num_prompts, seed)`: loads and shuffles GSM8K test split via HuggingFace `datasets`.
- `format_prompt(question, num_fewshot)`: wraps a question with few-shot examples in `Q:/A:` format.

### `signals.py` — Per-token signal extraction

- `extract_signals(logits, chosen_token_id, tokenizer)`: computes from a single decoding step's logit distribution:
  - **Entropy**: `H = -sum(p * log(p))` — higher = more uncertain.
  - **Top-1 probability**: confidence in the greedy choice.
  - **Top-5 probability**: mass in the top 5 tokens.
  - **Rank of chosen token**: always 0 under greedy decoding.

### `detectors.py` — Catastrophe detectors

- `detect_non_termination(stop_reason)`: hit `max_tokens` without EOS.
- `detect_looping(token_ids, window_size=20, min_repeats=3)`: sliding-window n-gram repetition.
- `parse_gsm8k_answer(text)`: extracts numeric answer from `#### N` or `\boxed{N}` format.
- `detect_answer_failure(generated_text, ground_truth)`: checks if predicted answer matches ground truth.
- `detect_all(...)`: runs all detectors, returns list of catastrophe type strings.

### `experiment.py` — Experiment runner

- `get_press(name, compression_ratio)`: creates a kvpress `Press` object (`StreamingLLMPress`, `ObservedAttentionPress`, `SnapKVPress`, or `None` for baseline).
- `load_model(config)`: loads model + tokenizer onto device. Uses `attn_implementation="eager"` for ObservedAttention.
- `run_single(...)`: runs one prompt — tokenizes via chat template, applies press context manager, generates with `output_scores=True`, extracts per-token signals, runs detectors.
- `run_experiment(config)`: iterates prompts, collects `RunResult` list, prints summary.
- `save_results(results, config)`: writes JSON with config, summary stats, and per-prompt results.
- `summarize(results)`: computes accuracy, CFR, catastrophe counts, avg tokens.

### `analyze.py` — Results analysis

- `compare_runs(result_dir)`: loads all result JSONs and prints a comparison table with signal statistics (mean entropy, max entropy, min top-1 prob).

### `__init__.py` — CLI entry point

- Typer app with a single `run` command exposing all `ExperimentConfig` fields as CLI options.
- Entry point: `kvguard = "kvguard:main"` (in `pyproject.toml`).

## Data flow

```
GSM8K dataset
    → format_prompt() → chat_template → tokenize
    → model.generate() inside press(model) context
    → per-token: extract_signals(logits)
    → detect_all(text, tokens, stop_reason, ground_truth)
    → RunResult → JSON
```

## Dependencies

| Package | Role |
|---------|------|
| `kvpress` | KV-cache compression (NVIDIA, 32 methods) |
| `transformers` | Model loading and generation |
| `torch` | Tensor operations, MPS/CUDA backend |
| `datasets` | GSM8K loading from HuggingFace Hub |
| `pydantic` / `pydantic-settings` | Config and result schemas |
| `typer` | CLI |
| `loguru` | Logging |
