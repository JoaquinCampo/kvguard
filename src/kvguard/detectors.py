"""Catastrophe detectors for compressed generation output."""

import re


def detect_non_termination(stop_reason: str) -> bool:
    """Generation hit max_tokens without producing an EOS."""
    return stop_reason == "max_tokens"


def detect_looping(token_ids: list[int], window_size: int = 20, min_repeats: int = 3) -> bool:
    """Detect repeated token windows in the output.

    Returns True if any window of `window_size` tokens appears >= `min_repeats` times.
    """
    if len(token_ids) < window_size * min_repeats:
        return False

    window_counts: dict[tuple[int, ...], int] = {}
    for i in range(len(token_ids) - window_size + 1):
        window = tuple(token_ids[i : i + window_size])
        window_counts[window] = window_counts.get(window, 0) + 1
        if window_counts[window] >= min_repeats:
            return True
    return False


def parse_gsm8k_answer(text: str) -> str | None:
    """Extract the final numeric answer from model output.

    Supports two formats:
    - GSM8K canonical: #### NUMBER
    - LaTeX boxed (instruction-tuned models): \\boxed{NUMBER}
    """
    # Try #### format first
    match = re.search(r"####\s*([\d,.\-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()

    # Try \\boxed{NUMBER} format (common in instruction-tuned models)
    match = re.search(r"\\boxed\{([\d,.\-]+)\}", text)
    if match:
        return match.group(1).replace(",", "").strip()

    return None


def detect_answer_failure(generated_text: str, ground_truth: str) -> bool:
    """Check if the model's answer matches ground truth."""
    predicted = parse_gsm8k_answer(generated_text)
    if predicted is None:
        return True  # no answer found at all
    try:
        return float(predicted) != float(ground_truth)
    except ValueError:
        return True


def detect_all(
    generated_text: str,
    token_ids: list[int],
    stop_reason: str,
    ground_truth: str,
) -> list[str]:
    """Run all detectors and return list of detected catastrophe types."""
    catastrophes = []
    if detect_non_termination(stop_reason):
        catastrophes.append("non_termination")
    if detect_looping(token_ids):
        catastrophes.append("looping")
    if detect_answer_failure(generated_text, ground_truth):
        catastrophes.append("wrong_answer")
    return catastrophes
