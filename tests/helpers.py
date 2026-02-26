"""Shared test helpers for synthetic signal/result generation.

Used by test_features.py, test_train.py, test_evaluate_controller.py,
and test_integration.py to avoid duplicating factory functions.
"""

import json
from pathlib import Path


def make_signal_dict(
    *,
    entropy: float = 1.0,
    top1_prob: float = 0.5,
    top5_prob: float = 0.9,
    rank_of_chosen: int = 0,
    top20_logprobs: list[float] | None = None,
    h_alts: float = 0.3,
    avg_logp: float = -2.0,
    delta_h: float | None = 0.1,
    rep_count: int = 0,
    is_thinking_token: bool = False,
) -> dict:
    """Build a single token's signal dictionary."""
    return {
        "entropy": entropy,
        "top1_prob": top1_prob,
        "top5_prob": top5_prob,
        "top1_token": "a",
        "rank_of_chosen": rank_of_chosen,
        "top20_logprobs": top20_logprobs if top20_logprobs is not None else [-0.5] * 20,
        "h_alts": h_alts,
        "avg_logp": avg_logp,
        "delta_h": delta_h,
        "rep_count": rep_count,
        "is_thinking_token": is_thinking_token,
    }


def make_result_json(
    *,
    n_tokens: int = 10,
    press: str = "none",
    compression_ratio: float = 0.0,
    catastrophes: list[str] | None = None,
    catastrophe_onsets: dict[str, int] | None = None,
    prompt_id: str = "gsm8k_0",
    correct: bool = True,
    predicted_answer: str = "42",
    stop_reason: str = "eos",
    model: str = "test-model",
) -> dict:
    """Build a full result dict as it would appear in sweep JSON."""
    sigs = []
    onset = min(catastrophe_onsets.values()) if catastrophe_onsets else n_tokens
    for t in range(n_tokens):
        is_post_onset = catastrophes and catastrophe_onsets and t >= onset
        if is_post_onset:
            sigs.append(
                make_signal_dict(
                    entropy=4.0 + 0.1 * t,
                    rep_count=t - onset + 1,
                    delta_h=1.5,
                )
            )
        else:
            sigs.append(
                make_signal_dict(
                    entropy=1.0 + 0.01 * t,
                    delta_h=0.01 if t > 0 else None,
                    rep_count=max(0, t - 8),
                )
            )
    return {
        "prompt_id": prompt_id,
        "prompt_text": "test prompt",
        "model": model,
        "press": press,
        "compression_ratio": compression_ratio,
        "max_new_tokens": 512,
        "seed": 42,
        "generated_text": "test output",
        "ground_truth": "42",
        "predicted_answer": predicted_answer,
        "correct": correct,
        "stop_reason": stop_reason,
        "catastrophes": catastrophes or [],
        "num_tokens_generated": n_tokens,
        "cache_size_after_prefill": None,
        "catastrophe_onsets": catastrophe_onsets or {},
        "signals": sigs,
    }


def write_result_file(
    tmpdir: Path,
    press: str,
    ratio: float,
    results: list[dict],
    n_prompts: int = 50,
    model: str = "test-model",
) -> Path:
    """Write a sweep result JSON to a temp directory."""
    subdir = tmpdir / press
    subdir.mkdir(parents=True, exist_ok=True)
    fname = f"{model}_{ratio:.3f}_{n_prompts}p.json"
    path = subdir / fname
    data = {
        "config": {
            "model_name": model,
            "press_name": press,
            "compression_ratio": ratio,
        },
        "summary": {},
        "results": results,
    }
    path.write_text(json.dumps(data))
    return path
