"""Tests for analyze_phase_transitions.py helper functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from analyze_phase_transitions import (  # noqa: E402
    compute_phase_transition_metrics,
    identify_critical_point,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prompt(
    *,
    correct: bool = True,
    catastrophes: list[str] | None = None,
    entropy: float = 1.0,
    n_tokens: int = 100,
) -> dict:
    """Create a minimal per-prompt result dict."""
    return {
        "correct": correct,
        "catastrophes": catastrophes or [],
        "num_tokens_generated": n_tokens,
        "signals": [{"entropy": entropy} for _ in range(n_tokens)],
    }


# ---------------------------------------------------------------------------
# Tests: compute_phase_transition_metrics
# ---------------------------------------------------------------------------


class TestComputePhaseTransitionMetrics:
    def test_basic_metrics(self) -> None:
        per_prompt = {
            "streaming_llm": {
                0.5: [_prompt(correct=True) for _ in range(10)],
                0.875: [_prompt(correct=False, catastrophes=["looping"]) for _ in range(10)],
            },
        }
        metrics = compute_phase_transition_metrics(per_prompt)
        assert "streaming_llm" in metrics
        assert len(metrics["streaming_llm"]) == 2

        # At 0.5: all correct, no catastrophes
        m05 = metrics["streaming_llm"][0]
        assert m05["ratio"] == 0.5
        assert m05["accuracy"] == 1.0
        assert m05["cfr"] == 0.0

        # At 0.875: all wrong, all looping
        m875 = metrics["streaming_llm"][1]
        assert m875["ratio"] == 0.875
        assert m875["accuracy"] == 0.0
        assert m875["cfr"] == 1.0
        assert m875["looping_rate"] == 1.0

    def test_susceptibility_computed(self) -> None:
        per_prompt = {
            "test": {
                0.25: [_prompt(correct=True) for _ in range(20)],
                0.5: [_prompt(correct=True) for _ in range(20)],
                0.75: [_prompt(correct=True) for _ in range(20)],
                # Mix of catastrophes and clean — gives nonzero variance
                0.875: [_prompt(correct=False, catastrophes=["looping"])] * 10
                + [_prompt(correct=True)] * 10,
            },
        }
        metrics = compute_phase_transition_metrics(per_prompt)
        # CFR variance at 0.875 should be > 0 (mixed outcomes), while prior ratios have 0
        last = metrics["test"][-1]
        assert "susceptibility_cfr" in last
        assert last["cfr_variance"] > 0
        assert last["susceptibility_cfr"] > 0  # variance jumped from 0 to positive

    def test_mixed_outcomes(self) -> None:
        per_prompt = {
            "test": {
                0.75: [_prompt(correct=True)] * 8 + [_prompt(correct=False)] * 2,
            },
        }
        metrics = compute_phase_transition_metrics(per_prompt)
        m = metrics["test"][0]
        assert abs(m["accuracy"] - 0.8) < 0.01
        assert m["acc_variance"] > 0  # not all same

    def test_empty_compressor(self) -> None:
        per_prompt: dict[str, dict[float, list[dict]]] = {"test": {}}
        metrics = compute_phase_transition_metrics(per_prompt)
        assert metrics["test"] == []

    def test_retention_calculated(self) -> None:
        per_prompt = {
            "test": {
                0.75: [_prompt(correct=True) for _ in range(5)],
            },
        }
        metrics = compute_phase_transition_metrics(per_prompt)
        assert abs(metrics["test"][0]["retention"] - 0.25) < 0.01

    def test_entropy_stats_included(self) -> None:
        per_prompt = {
            "test": {
                0.5: [_prompt(correct=True, entropy=2.0) for _ in range(10)],
            },
        }
        metrics = compute_phase_transition_metrics(per_prompt)
        m = metrics["test"][0]
        assert abs(m["mean_max_entropy"] - 2.0) < 0.01
        assert abs(m["mean_mean_entropy"] - 2.0) < 0.01


# ---------------------------------------------------------------------------
# Tests: identify_critical_point
# ---------------------------------------------------------------------------


class TestIdentifyCriticalPoint:
    def test_finds_peak_variance(self) -> None:
        metrics = [
            {"ratio": 0.25, "accuracy": 0.9, "cfr_variance": 0.01},
            {"ratio": 0.5, "accuracy": 0.85, "cfr_variance": 0.02},
            {"ratio": 0.75, "accuracy": 0.7, "cfr_variance": 0.2},  # peak variance
            {"ratio": 0.875, "accuracy": 0.1, "cfr_variance": 0.05},
        ]
        cp = identify_critical_point(metrics)
        assert cp["critical_ratio_variance"] == 0.75
        assert abs(cp["peak_cfr_variance"] - 0.2) < 0.01

    def test_finds_steepest_drop(self) -> None:
        metrics = [
            {"ratio": 0.25, "accuracy": 0.9, "cfr_variance": 0.01},
            {"ratio": 0.5, "accuracy": 0.85, "cfr_variance": 0.02},
            {"ratio": 0.75, "accuracy": 0.8, "cfr_variance": 0.03},
            {"ratio": 0.875, "accuracy": 0.1, "cfr_variance": 0.05},  # biggest drop
        ]
        cp = identify_critical_point(metrics)
        assert cp["steepest_drop_ratio"] == 0.875
        assert abs(cp["steepest_drop_magnitude"] - 0.7) < 0.01

    def test_empty_metrics(self) -> None:
        cp = identify_critical_point([])
        assert cp["critical_ratio"] is None
        assert cp["peak_variance"] == 0.0

    def test_single_ratio(self) -> None:
        metrics = [{"ratio": 0.5, "accuracy": 0.9, "cfr_variance": 0.05}]
        cp = identify_critical_point(metrics)
        assert cp["critical_ratio_variance"] == 0.5
        assert cp["steepest_drop_ratio"] is None  # no drop possible with 1 point
