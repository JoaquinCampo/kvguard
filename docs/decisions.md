# Decision Log

Tracks choices made during the project with rationale. Append-only.

| # | Decision | Options considered | Chosen | Rationale | Date |
|---|----------|-------------------|--------|-----------|------|
| 1 | Primary framework | kvpress, KVCache-Factory, Cold Compress, from-scratch | **kvpress** | 32 methods as swappable presses, NVIDIA maintained, HF Transformers native, our controller wraps a `Press` directly | 2026-02-09 |
| 2 | First model | Llama-3-8B, Qwen2.5-7B, Qwen2.5-3B, Mistral-7B | **Qwen2.5-3B-Instruct** | No license gate, kvpress compatible, fits on MPS (float16), fast iteration during prototyping | 2026-02-17 |
| 3 | First task | GSM8K, MATH, AIME, LongBench | **GSM8K** | Multi-step CoT reasoning, canonical math benchmark, answer extraction is trivial, datasets library support | 2026-02-17 |
| 4 | Budget normalization | Retained tokens/layer, total KV bytes | **compression_ratio (kvpress)** | kvpress defines ratio as fraction removed; simple, matches the library API directly | 2026-02-17 |
| 5 | Answer format parsing | `####` only, `\boxed{}` only, both | **Both** | Instruction-tuned models prefer `\boxed{}` despite few-shot examples using `####`; supporting both avoids false negatives in accuracy measurement | 2026-02-17 |
| 6 | Signal extraction method | Manual generation loop, `output_scores=True` | **`output_scores=True`** | HF `model.generate()` with `return_dict_in_generate=True` gives per-step logits without reimplementing the generation loop | 2026-02-17 |
