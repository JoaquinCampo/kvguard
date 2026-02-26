"""Tests for analyze_failure_modes.py helper functions."""

import sys
from pathlib import Path

import numpy as np

# Must insert before importing the script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from analyze_failure_modes import (  # noqa: E402
    TraceSignals,
    compute_early_warning_stats,
    compute_lead_times,
    compute_pre_onset_stats,
    compute_trajectory_stats,
    statistical_tests,
)

# ---------------------------------------------------------------------------
# Helper to build synthetic TraceSignals
# ---------------------------------------------------------------------------


def _trace(
    *,
    prompt_id: str = "p0",
    press: str = "streaming_llm",
    ratio: float = 0.875,
    catastrophes: list[str] | None = None,
    onsets: dict[str, int] | None = None,
    n_tokens: int = 100,
    correct: bool = True,
    entropy_val: float = 1.0,
    top1_val: float = 0.8,
) -> TraceSignals:
    """Build a synthetic TraceSignals with uniform signal values."""
    ts = TraceSignals(
        prompt_id=prompt_id,
        press=press,
        compression_ratio=ratio,
        catastrophes=catastrophes or [],
        onsets=onsets or {},
        n_tokens=n_tokens,
        correct=correct,
    )
    for i in range(n_tokens):
        ts.entropy.append(entropy_val)
        ts.top1_prob.append(top1_val)
        ts.top5_prob.append(min(1.0, top1_val + 0.1))
        ts.h_alts.append(entropy_val * 0.5)
        ts.delta_h.append(0.1 if i > 0 else None)
        ts.rep_count.append(0)
        ts.avg_logp.append(-2.0)
    return ts


# ---------------------------------------------------------------------------
# Tests: TraceSignals
# ---------------------------------------------------------------------------


class TestTraceSignals:
    def test_failure_mode_looping(self) -> None:
        t = _trace(catastrophes=["looping"])
        assert t.failure_mode == "looping"

    def test_failure_mode_non_termination(self) -> None:
        t = _trace(catastrophes=["non_termination"])
        assert t.failure_mode == "non_termination"

    def test_failure_mode_wrong_answer(self) -> None:
        t = _trace(catastrophes=["wrong_answer"])
        assert t.failure_mode == "wrong_answer"

    def test_failure_mode_correct(self) -> None:
        t = _trace(catastrophes=[])
        assert t.failure_mode == "correct"

    def test_failure_mode_priority_looping_over_nt(self) -> None:
        t = _trace(catastrophes=["non_termination", "looping"])
        assert t.failure_mode == "looping"

    def test_onset_token_single(self) -> None:
        t = _trace(onsets={"looping": 50})
        assert t.onset_token == 50

    def test_onset_token_multiple(self) -> None:
        t = _trace(onsets={"looping": 50, "non_termination": 30})
        assert t.onset_token == 30  # min of onsets

    def test_onset_token_none(self) -> None:
        t = _trace(onsets={})
        assert t.onset_token is None


# ---------------------------------------------------------------------------
# Tests: compute_pre_onset_stats
# ---------------------------------------------------------------------------


class TestComputePreOnsetStats:
    def test_basic(self) -> None:
        traces = [
            _trace(
                catastrophes=["looping"],
                onsets={"looping": 80},
                entropy_val=5.0,
            ),
        ]
        stats = compute_pre_onset_stats(traces, window=32)
        assert "looping" in stats
        assert len(stats["looping"]["entropy"]) == 32

    def test_no_onset_skipped(self) -> None:
        traces = [_trace(catastrophes=["wrong_answer"], onsets={})]
        stats = compute_pre_onset_stats(traces, window=32)
        assert len(stats) == 0

    def test_small_window(self) -> None:
        traces = [
            _trace(
                catastrophes=["looping"],
                onsets={"looping": 10},
                n_tokens=20,
            ),
        ]
        stats = compute_pre_onset_stats(traces, window=5)
        assert len(stats["looping"]["entropy"]) == 5

    def test_onset_at_start(self) -> None:
        traces = [
            _trace(
                catastrophes=["looping"],
                onsets={"looping": 0},
                n_tokens=20,
            ),
        ]
        stats = compute_pre_onset_stats(traces, window=32)
        # end <= start, so no data
        assert "looping" not in stats

    def test_multiple_modes(self) -> None:
        traces = [
            _trace(catastrophes=["looping"], onsets={"looping": 80}, entropy_val=5.0),
            _trace(
                catastrophes=["non_termination"],
                onsets={"non_termination": 80},
                entropy_val=1.0,
            ),
        ]
        stats = compute_pre_onset_stats(traces, window=32)
        assert "looping" in stats
        assert "non_termination" in stats
        # Looping should have higher entropy values
        loop_mean = np.mean(stats["looping"]["entropy"])
        nt_mean = np.mean(stats["non_termination"]["entropy"])
        assert loop_mean > nt_mean


# ---------------------------------------------------------------------------
# Tests: compute_early_warning_stats
# ---------------------------------------------------------------------------


class TestComputeEarlyWarningStats:
    def test_basic(self) -> None:
        traces = [_trace(n_tokens=100)]
        stats = compute_early_warning_stats(traces)
        assert "correct" in stats
        assert len(stats["correct"]["entropy"]) == 50  # first 50 tokens

    def test_short_trace(self) -> None:
        traces = [_trace(n_tokens=20)]
        stats = compute_early_warning_stats(traces)
        assert len(stats["correct"]["entropy"]) == 20

    def test_groups_by_mode(self) -> None:
        traces = [
            _trace(catastrophes=[], entropy_val=1.0),
            _trace(catastrophes=["looping"], entropy_val=5.0),
        ]
        stats = compute_early_warning_stats(traces)
        assert "correct" in stats
        assert "looping" in stats


# ---------------------------------------------------------------------------
# Tests: compute_trajectory_stats
# ---------------------------------------------------------------------------


class TestComputeTrajectoryStats:
    def test_basic(self) -> None:
        traces = [_trace(n_tokens=50, entropy_val=2.0)]
        stats = compute_trajectory_stats(traces)
        assert "correct" in stats
        assert abs(stats["correct"]["mean_entropy"]["mean"] - 2.0) < 0.01

    def test_empty_trace_skipped(self) -> None:
        t = _trace(n_tokens=0)
        t.entropy = []
        t.top1_prob = []
        t.top5_prob = []
        t.h_alts = []
        t.delta_h = []
        t.rep_count = []
        t.avg_logp = []
        stats = compute_trajectory_stats([t])
        assert len(stats) == 0

    def test_multiple_modes(self) -> None:
        traces = [
            _trace(catastrophes=[], entropy_val=1.0),
            _trace(catastrophes=["looping"], entropy_val=5.0),
        ]
        stats = compute_trajectory_stats(traces)
        assert stats["looping"]["mean_entropy"]["mean"] > stats["correct"]["mean_entropy"]["mean"]


# ---------------------------------------------------------------------------
# Tests: statistical_tests
# ---------------------------------------------------------------------------


class TestStatisticalTests:
    def test_significant_difference(self) -> None:
        mode_signals = {
            "looping": {
                "entropy": [5.0 + np.random.randn() * 0.1 for _ in range(100)],
                "top1_prob": [0.3 + np.random.randn() * 0.05 for _ in range(100)],
                "h_alts": [3.0 + np.random.randn() * 0.2 for _ in range(100)],
                "delta_h": [0.5 + np.random.randn() * 0.1 for _ in range(100)],
                "avg_logp": [-5.0 + np.random.randn() * 0.3 for _ in range(100)],
            },
            "non_termination": {
                "entropy": [1.0 + np.random.randn() * 0.1 for _ in range(100)],
                "top1_prob": [0.9 + np.random.randn() * 0.02 for _ in range(100)],
                "h_alts": [0.5 + np.random.randn() * 0.1 for _ in range(100)],
                "delta_h": [0.1 + np.random.randn() * 0.05 for _ in range(100)],
                "avg_logp": [-2.0 + np.random.randn() * 0.1 for _ in range(100)],
            },
        }
        results = statistical_tests(mode_signals)
        assert "entropy" in results
        key = "looping_vs_non_termination"
        assert key in results["entropy"]
        assert results["entropy"][key]["p"] < 0.01

    def test_too_few_samples_skipped(self) -> None:
        mode_signals = {
            "looping": {"entropy": [1.0, 2.0]},
            "non_termination": {"entropy": [3.0, 4.0]},
        }
        results = statistical_tests(mode_signals)
        # Fewer than 10 samples — should skip
        assert "looping_vs_non_termination" not in results.get("entropy", {})

    def test_empty_modes(self) -> None:
        results = statistical_tests({})
        for sig in ["entropy", "top1_prob", "h_alts", "delta_h", "avg_logp"]:
            assert results[sig] == {}


# ---------------------------------------------------------------------------
# Tests: compute_lead_times
# ---------------------------------------------------------------------------


class TestComputeLeadTimes:
    def test_entropy_lead_time(self) -> None:
        # Trace with entropy spike before onset
        t = _trace(
            catastrophes=["looping"],
            onsets={"looping": 80},
            entropy_val=0.5,  # normal entropy
            n_tokens=100,
        )
        # Create a clear spike at token 70 (10 tokens before onset)
        t.entropy[70] = 10.0
        # Need correct traces for baseline
        correct_trace = _trace(catastrophes=[], entropy_val=0.5, n_tokens=100)
        traces = [t, correct_trace]
        lead_times = compute_lead_times(traces)
        assert "looping_entropy" in lead_times
        assert lead_times["looping_entropy"][0] == 10  # 80 - 70

    def test_no_onset_skipped(self) -> None:
        traces = [_trace(catastrophes=[], entropy_val=1.0)]
        lead_times = compute_lead_times(traces)
        # No catastrophe traces → no lead times
        assert "looping_entropy" not in lead_times

    def test_no_correct_traces(self) -> None:
        # All traces are catastrophic → no baseline for thresholds
        traces = [_trace(catastrophes=["looping"], onsets={"looping": 50})]
        lead_times = compute_lead_times(traces)
        assert lead_times == {}

    def test_onset_too_early_skipped(self) -> None:
        traces = [
            _trace(catastrophes=["looping"], onsets={"looping": 2}, n_tokens=100),
            _trace(catastrophes=[], entropy_val=0.5),
        ]
        lead_times = compute_lead_times(traces)
        # Onset < 5 → skipped
        assert "looping_entropy" not in lead_times
