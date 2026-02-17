"""Main experiment runner: load model, compress KV cache, generate, detect catastrophes."""

import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvguard.config import ExperimentConfig, RunResult
from kvguard.detectors import detect_all, parse_gsm8k_answer
from kvguard.prompts import format_prompt, load_gsm8k
from kvguard.signals import extract_signals


def get_press(name: str, compression_ratio: float):  # noqa: ANN201
    """Create a kvpress Press object (or None for baseline)."""
    if name == "none":
        return None

    from kvpress import ObservedAttentionPress, SnapKVPress, StreamingLLMPress

    presses = {
        "streaming_llm": StreamingLLMPress,
        "observed_attention": ObservedAttentionPress,
        "snapkv": SnapKVPress,
    }
    if name not in presses:
        raise ValueError(f"Unknown press: {name}. Available: {list(presses.keys())}")
    return presses[name](compression_ratio=compression_ratio)


def load_model(
    config: ExperimentConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    """Load model and tokenizer onto the target device."""
    device = config.resolve_device()
    logger.info(f"Loading {config.model_name} on {device}...")

    # ObservedAttention requires eager attention implementation
    kwargs: dict = {"dtype": torch.float16}
    if config.press_name == "observed_attention":
        kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded. Parameters: {param_count:,}")
    return model, tokenizer, device


def run_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    prompt_data: dict,
    config: ExperimentConfig,
    press: object | None,
) -> RunResult:
    """Run generation for a single prompt and extract signals."""
    prompt_text = format_prompt(prompt_data["question"], config.num_fewshot)

    # Use chat template for instruction-tuned models
    messages = [{"role": "user", "content": prompt_text}]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Compress KV cache during prefill via press context manager
    ctx = press(model) if press is not None else nullcontext()  # type: ignore[operator]

    with torch.no_grad(), ctx:
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # Decode generated tokens (excluding prompt)
    generated_ids = outputs.sequences[0, input_len:].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Determine stop reason
    eos_id = tokenizer.eos_token_id
    hit_max = len(generated_ids) >= config.max_new_tokens
    hit_eos = len(generated_ids) > 0 and generated_ids[-1] == eos_id
    stop_reason = "max_tokens" if (hit_max and not hit_eos) else "eos"

    # Extract per-token signals from generation scores
    signals = []
    for score, token_id in zip(outputs.scores, generated_ids):
        logits = score[0]  # remove batch dim
        sig = extract_signals(logits, token_id, tokenizer)
        signals.append(sig)

    # Run catastrophe detectors
    catastrophes = detect_all(
        generated_text, generated_ids, stop_reason, prompt_data["ground_truth"]
    )

    predicted = parse_gsm8k_answer(generated_text)
    try:
        correct = float(predicted) == float(prompt_data["ground_truth"]) if predicted else None
    except ValueError:
        correct = None

    return RunResult(
        prompt_id=prompt_data["id"],
        prompt_text=prompt_text,
        model=config.model_name,
        press=config.press_name,
        compression_ratio=config.compression_ratio,
        max_new_tokens=config.max_new_tokens,
        seed=config.seed,
        generated_text=generated_text,
        ground_truth=prompt_data["ground_truth"],
        predicted_answer=predicted,
        correct=correct,
        stop_reason=stop_reason,
        catastrophes=catastrophes,
        num_tokens_generated=len(generated_ids),
        cache_size_after_prefill=None,
        signals=signals,
    )


def summarize(results: list[RunResult]) -> dict:
    """Compute summary statistics over a batch of results."""
    n = len(results)
    if n == 0:
        return {"total": 0}

    correct = sum(1 for r in results if r.correct)
    has_catastrophe = sum(1 for r in results if r.catastrophes)

    cat_counts: dict[str, int] = {}
    for r in results:
        for c in r.catastrophes:
            cat_counts[c] = cat_counts.get(c, 0) + 1

    return {
        "total": n,
        "correct": correct,
        "accuracy": round(correct / n, 4),
        "catastrophic_failure_rate": round(has_catastrophe / n, 4),
        "catastrophe_counts": cat_counts,
        "avg_tokens": round(sum(r.num_tokens_generated for r in results) / n, 1),
    }


def save_results(results: list[RunResult], config: ExperimentConfig) -> Path:
    """Save results + summary to a JSON file."""
    output_dir = config.output_dir / config.press_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model_short = config.model_name.split("/")[-1]
    ratio_str = f"{config.compression_ratio:.3f}"
    filename = f"{model_short}_{ratio_str}_{config.num_prompts}p.json"
    path = output_dir / filename

    data = {
        "config": config.model_dump(mode="json"),
        "summary": summarize(results),
        "results": [r.model_dump(mode="json") for r in results],
    }
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info(f"Results saved to {path}")
    return path


def run_experiment(config: ExperimentConfig) -> list[RunResult]:
    """Run the full experiment: load model, iterate prompts, collect results."""
    logger.info(f"Experiment: {config.press_name} @ compression_ratio={config.compression_ratio}")

    model, tokenizer, device = load_model(config)
    press = get_press(config.press_name, config.compression_ratio)
    prompts = load_gsm8k(config.num_prompts, config.seed)

    results: list[RunResult] = []
    for i, prompt_data in enumerate(prompts):
        logger.info(f"[{i + 1}/{len(prompts)}] {prompt_data['id']}")
        t0 = time.time()

        try:
            result = run_single(model, tokenizer, device, prompt_data, config, press)
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            continue

        elapsed = time.time() - t0
        status = "CORRECT" if result.correct else "WRONG"
        cats = ", ".join(result.catastrophes) if result.catastrophes else "none"
        logger.info(
            f"  {status} | tokens={result.num_tokens_generated} "
            f"| catastrophes=[{cats}] | {elapsed:.1f}s"
        )
        results.append(result)

    # Print summary
    s = summarize(results)
    logger.info("=" * 60)
    if s["total"] > 0:
        logger.info(
            f"SUMMARY: {s['total']} prompts | accuracy={s['accuracy']:.1%}"
            f" | CFR={s['catastrophic_failure_rate']:.1%}"
        )
        if s.get("catastrophe_counts"):
            for cat, count in s["catastrophe_counts"].items():
                logger.info(f"  {cat}: {count}/{s['total']}")
    else:
        logger.warning("No results collected — all prompts failed.")
    logger.info("=" * 60)

    return results
