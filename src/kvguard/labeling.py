"""Hazard labeling for per-token catastrophe prediction."""

from kvguard.config import RunResult

# Default horizons for sweeping
HORIZON_SWEEP = (16, 32, 64, 128)


def compute_hazard_labels(
    result: RunResult,
    horizon: int = 32,
    nt_onset_frac: float = 0.75,
) -> list[int]:
    """Compute per-token binary hazard labels for a single trace.

    y_t = 1 if token t is within ``horizon`` tokens of (or past) the earliest
    applicable catastrophe onset.  Only **looping** and **non_termination** are
    targeted — ``wrong_answer`` has no token-level onset.

    For non_termination the true onset (last token) is useless for prediction,
    so a proxy onset at ``nt_onset_frac * max_new_tokens`` is used instead.

    Args:
        result: A RunResult with catastrophe detection already applied.
        horizon: H — how many tokens before onset to start labeling.
        nt_onset_frac: Fraction of max_new_tokens used as proxy onset
            for non_termination (default 0.75 = token 384 of 512).

    Returns:
        List of 0/1 labels, one per generated token.
    """
    n_tokens = result.num_tokens_generated
    labels = [0] * n_tokens

    # Collect applicable onset positions
    onsets: list[int] = []

    if "looping" in result.catastrophe_onsets:
        onsets.append(result.catastrophe_onsets["looping"])

    if "non_termination" in result.catastrophes:
        proxy = int(nt_onset_frac * result.max_new_tokens)
        proxy = min(proxy, n_tokens - 1)
        onsets.append(proxy)

    if not onsets:
        return labels

    earliest = min(onsets)
    start = max(0, earliest - horizon)
    for t in range(start, n_tokens):
        labels[t] = 1

    return labels
