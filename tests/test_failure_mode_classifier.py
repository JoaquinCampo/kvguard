"""Tests for failure_mode_classifier.py helper functions."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from failure_mode_classifier import (  # noqa: E402
    TraceData,
    build_classification_dataset,
    extract_window_features,
    format_confusion_matrix,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(
    *,
    entropy: float = 1.0,
    top1_prob: float = 0.8,
    top5_prob: float = 0.9,
    h_alts: float = 0.5,
    avg_logp: float = -2.0,
    rep_count: int = 0,
    delta_h: float | None = 0.1,
) -> dict:
    """Create a minimal signal dict."""
    return {
        "entropy": entropy,
        "top1_prob": top1_prob,
        "top5_prob": top5_prob,
        "h_alts": h_alts,
        "avg_logp": avg_logp,
        "rep_count": rep_count,
        "delta_h": delta_h,
    }


def _make_trace(
    *,
    prompt_id: str = "p0",
    catastrophes: list[str] | None = None,
    onsets: dict[str, int] | None = None,
    n_tokens: int = 100,
    correct: bool = True,
    entropy: float = 1.0,
) -> TraceData:
    """Create a synthetic TraceData."""
    return TraceData(
        prompt_id=prompt_id,
        press="streaming_llm",
        compression_ratio=0.875,
        catastrophes=catastrophes or [],
        onsets=onsets or {},
        n_tokens=n_tokens,
        correct=correct,
        signals=[_make_signal(entropy=entropy) for _ in range(n_tokens)],
    )


# ---------------------------------------------------------------------------
# Tests: TraceData
# ---------------------------------------------------------------------------


class TestTraceData:
    def test_failure_mode(self) -> None:
        assert _make_trace(catastrophes=["looping"]).failure_mode == "looping"
        assert _make_trace(catastrophes=["non_termination"]).failure_mode == "non_termination"
        assert _make_trace(catastrophes=["wrong_answer"]).failure_mode == "wrong_answer"
        assert _make_trace(catastrophes=[]).failure_mode == "correct"

    def test_onset_token(self) -> None:
        t = _make_trace(onsets={"looping": 50, "non_termination": 30})
        assert t.onset_token == 30


# ---------------------------------------------------------------------------
# Tests: extract_window_features
# ---------------------------------------------------------------------------


class TestExtractWindowFeatures:
    def test_output_shape(self) -> None:
        signals = [_make_signal() for _ in range(50)]
        features = extract_window_features(signals, 0, 50)
        # 6 signals × 5 stats + 2 trends + 2 delta_h = 34
        assert features.shape == (34,)

    def test_empty_window(self) -> None:
        features = extract_window_features([], 0, 0)
        assert features.shape == (34,)
        assert np.all(features == 0)

    def test_single_token(self) -> None:
        signals = [_make_signal(entropy=3.0)]
        features = extract_window_features(signals, 0, 1)
        # mean entropy should be 3.0 (first feature)
        assert abs(features[0] - 3.0) < 0.01

    def test_entropy_values(self) -> None:
        signals = [_make_signal(entropy=float(i)) for i in range(10)]
        features = extract_window_features(signals, 0, 10)
        # Mean entropy = mean(0..9) = 4.5
        assert abs(features[0] - 4.5) < 0.01
        # Std entropy > 0
        assert features[1] > 0
        # Min entropy = 0
        assert abs(features[2] - 0.0) < 0.01
        # Max entropy = 9
        assert abs(features[3] - 9.0) < 0.01
        # Last entropy = 9
        assert abs(features[4] - 9.0) < 0.01

    def test_trend_positive(self) -> None:
        signals = [_make_signal(entropy=float(i)) for i in range(10)]
        features = extract_window_features(signals, 0, 10)
        # Entropy trend (slope) should be positive — index 30
        assert features[30] > 0

    def test_none_delta_h(self) -> None:
        signals = [_make_signal(delta_h=None) for _ in range(10)]
        features = extract_window_features(signals, 0, 10)
        # Delta-H features (last 2) should be 0 when all None
        assert features[-1] == 0.0
        assert features[-2] == 0.0


# ---------------------------------------------------------------------------
# Tests: build_classification_dataset
# ---------------------------------------------------------------------------


class TestBuildClassificationDataset:
    def test_pre_onset_mode(self) -> None:
        traces = [
            _make_trace(catastrophes=["looping"], onsets={"looping": 80}, n_tokens=100),
            _make_trace(catastrophes=["non_termination"], onsets={"non_termination": 80}),
        ]
        X, y, pids = build_classification_dataset(traces, mode="pre_onset", window=32)
        assert X.shape[0] == 2
        assert set(y) == {"looping", "non_termination"}
        assert len(pids) == 2

    def test_pre_onset_no_onset_excluded(self) -> None:
        traces = [
            _make_trace(catastrophes=["wrong_answer"], onsets={}),
        ]
        X, y, pids = build_classification_dataset(traces, mode="pre_onset")
        assert X.shape[0] == 0

    def test_early_mode(self) -> None:
        traces = [
            _make_trace(catastrophes=[], n_tokens=100),
            _make_trace(catastrophes=["looping"], onsets={"looping": 80}, n_tokens=100),
        ]
        X, y, pids = build_classification_dataset(traces, mode="early")
        assert X.shape[0] == 2
        assert "correct" in y
        assert "looping" in y

    def test_trace_agg_mode(self) -> None:
        traces = [_make_trace(n_tokens=50)]
        X, y, pids = build_classification_dataset(traces, mode="trace_agg")
        assert X.shape[0] == 1
        assert X.shape[1] == 34  # feature count

    def test_short_traces_excluded(self) -> None:
        traces = [_make_trace(n_tokens=5)]  # < 10 threshold
        X, y, pids = build_classification_dataset(traces, mode="early")
        assert X.shape[0] == 0

    def test_prompt_ids_preserved(self) -> None:
        traces = [
            _make_trace(prompt_id="q1", catastrophes=["looping"], onsets={"looping": 50}),
            _make_trace(prompt_id="q2", catastrophes=["looping"], onsets={"looping": 50}),
        ]
        _, _, pids = build_classification_dataset(traces, mode="pre_onset")
        assert list(pids) == ["q1", "q2"]


# ---------------------------------------------------------------------------
# Tests: format_confusion_matrix
# ---------------------------------------------------------------------------


class TestFormatConfusionMatrix:
    def test_basic_format(self) -> None:
        cm = [[10, 2], [1, 8]]
        classes = ["looping", "correct"]
        result = format_confusion_matrix(cm, classes)
        assert "Pred: looping" in result
        assert "True: looping" in result
        assert "10" in result
        assert "8" in result

    def test_three_classes(self) -> None:
        cm = [[5, 1, 0], [2, 8, 1], [0, 0, 3]]
        classes = ["a", "b", "c"]
        result = format_confusion_matrix(cm, classes)
        lines = result.strip().split("\n")
        assert len(lines) == 5  # header + sep + 3 rows
