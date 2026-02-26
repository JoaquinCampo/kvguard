"""Tests for the analyze module."""

import json
from pathlib import Path

from kvguard.analyze import (
    _per_run_signal_stats,
    _rolling_entropy,
    compare_runs,
    degradation_table,
    delta_h_analysis,
    load_all_results,
    load_results,
    rolling_delta_h_detector,
    signal_stats,
    silent_failure_analysis,
)
from tests.helpers import make_result_json, make_signal_dict, write_result_file

# ---------------------------------------------------------------------------
# Tests: load_results
# ---------------------------------------------------------------------------


class TestLoadResults:
    def test_loads_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "test.json"
        data = {"config": {"model_name": "test"}, "results": []}
        path.write_text(json.dumps(data))
        loaded = load_results(path)
        assert loaded["config"]["model_name"] == "test"

    def test_loads_nested_structure(self, tmp_path: Path) -> None:
        write_result_file(tmp_path, "none", 0.0, [make_result_json(n_tokens=5)])
        path = next((tmp_path / "none").glob("*.json"))
        loaded = load_results(path)
        assert "config" in loaded
        assert "results" in loaded
        assert len(loaded["results"]) == 1


# ---------------------------------------------------------------------------
# Tests: load_all_results
# ---------------------------------------------------------------------------


class TestLoadAllResults:
    def test_loads_all_files(self, tmp_path: Path) -> None:
        write_result_file(tmp_path, "none", 0.0, [make_result_json()])
        write_result_file(tmp_path, "streaming_llm", 0.5, [make_result_json()])
        results = load_all_results(tmp_path)
        assert len(results) == 2

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        results = load_all_results(tmp_path)
        assert results == []

    def test_prompt_filter(self, tmp_path: Path) -> None:
        """prompt_filter matches config.num_prompts field."""
        # Write a file whose config has num_prompts=50
        subdir = tmp_path / "none"
        subdir.mkdir()
        data = {
            "config": {
                "model_name": "test-model",
                "press_name": "none",
                "compression_ratio": 0.0,
                "num_prompts": 50,
            },
            "summary": {},
            "results": [],
        }
        (subdir / "test.json").write_text(json.dumps(data))

        # Write another with num_prompts=100
        data2 = dict(data)
        data2["config"] = dict(data["config"], num_prompts=100)
        (subdir / "test2.json").write_text(json.dumps(data2))

        # Filter for 50 only
        results = load_all_results(tmp_path, prompt_filter=50)
        assert len(results) == 1
        assert results[0]["config"]["num_prompts"] == 50


# ---------------------------------------------------------------------------
# Tests: signal_stats
# ---------------------------------------------------------------------------


class TestSignalStats:
    def test_basic_stats(self) -> None:
        results = [
            {
                "signals": [
                    make_signal_dict(entropy=2.0, top1_prob=0.8, delta_h=0.1, h_alts=0.5),
                    make_signal_dict(entropy=3.0, top1_prob=0.6, delta_h=0.2, h_alts=0.7),
                ]
            }
        ]
        stats = signal_stats(results)
        assert stats["mean_entropy"] == 2.5
        assert stats["max_entropy"] == 3.0
        assert stats["mean_top1"] == 0.7
        assert stats["min_top1"] == 0.6
        assert "mean_abs_delta_h" in stats
        assert "mean_h_alts" in stats

    def test_empty_results(self) -> None:
        stats = signal_stats([])
        assert stats == {}

    def test_empty_signals(self) -> None:
        stats = signal_stats([{"signals": []}])
        assert stats == {}

    def test_multiple_results(self) -> None:
        results = [
            {"signals": [make_signal_dict(entropy=1.0)]},
            {"signals": [make_signal_dict(entropy=5.0)]},
        ]
        stats = signal_stats(results)
        assert stats["max_entropy"] == 5.0
        assert stats["median_max_entropy"] == 3.0  # median of [1.0, 5.0]

    def test_thinking_token_pct(self) -> None:
        results = [
            {
                "signals": [
                    make_signal_dict(is_thinking_token=True),
                    make_signal_dict(is_thinking_token=False),
                    make_signal_dict(is_thinking_token=True),
                    make_signal_dict(is_thinking_token=False),
                ]
            }
        ]
        stats = signal_stats(results)
        assert stats["thinking_token_pct"] == 50.0

    def test_delta_h_none_excluded(self) -> None:
        """Tokens with delta_h=None should be excluded from delta_h stats."""
        results = [
            {
                "signals": [
                    make_signal_dict(delta_h=None),
                    make_signal_dict(delta_h=0.5),
                ]
            }
        ]
        stats = signal_stats(results)
        assert stats["max_abs_delta_h"] == 0.5


# ---------------------------------------------------------------------------
# Tests: _per_run_signal_stats
# ---------------------------------------------------------------------------


class TestPerRunSignalStats:
    def test_basic(self) -> None:
        sigs = [
            make_signal_dict(entropy=1.0, top1_prob=0.9, delta_h=0.1, h_alts=0.2),
            make_signal_dict(entropy=4.0, top1_prob=0.3, delta_h=1.5, h_alts=2.0),
        ]
        stats = _per_run_signal_stats(sigs)
        assert stats["max_entropy"] == 4.0
        assert stats["mean_entropy"] == 2.5
        assert stats["min_top1"] == 0.3
        assert stats["max_abs_delta_h"] == 1.5
        assert stats["max_h_alts"] == 2.0

    def test_thinking_pct(self) -> None:
        sigs = [
            make_signal_dict(is_thinking_token=True),
            make_signal_dict(is_thinking_token=True),
            make_signal_dict(is_thinking_token=False),
        ]
        stats = _per_run_signal_stats(sigs)
        assert abs(stats["think_pct"] - 66.67) < 0.1


# ---------------------------------------------------------------------------
# Tests: _rolling_entropy
# ---------------------------------------------------------------------------


class TestRollingEntropy:
    def test_constant_signal(self) -> None:
        sigs = [make_signal_dict(entropy=3.0) for _ in range(10)]
        rolling = _rolling_entropy(sigs, window=5)
        assert len(rolling) == 6  # 10 - 5 + 1
        assert all(abs(v - 3.0) < 1e-6 for v in rolling)

    def test_short_signal(self) -> None:
        """Signals shorter than window return raw entropies."""
        sigs = [make_signal_dict(entropy=float(i)) for i in range(3)]
        rolling = _rolling_entropy(sigs, window=5)
        assert len(rolling) == 3
        assert rolling == [0.0, 1.0, 2.0]

    def test_window_1(self) -> None:
        sigs = [make_signal_dict(entropy=float(i)) for i in range(5)]
        rolling = _rolling_entropy(sigs, window=1)
        assert len(rolling) == 5
        assert rolling == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_increasing_signal(self) -> None:
        sigs = [make_signal_dict(entropy=float(i)) for i in range(6)]
        rolling = _rolling_entropy(sigs, window=3)
        # rolling[0] = mean(0,1,2)=1.0, rolling[1]=mean(1,2,3)=2.0, etc.
        assert len(rolling) == 4
        assert abs(rolling[0] - 1.0) < 1e-6
        assert abs(rolling[1] - 2.0) < 1e-6
        assert abs(rolling[2] - 3.0) < 1e-6
        assert abs(rolling[3] - 4.0) < 1e-6


# ---------------------------------------------------------------------------
# Helpers for analysis function tests
# ---------------------------------------------------------------------------


def _write_analysis_file(
    tmpdir: Path,
    press: str,
    ratio: float,
    results: list[dict],
    n_prompts: int = 50,
) -> Path:
    """Write a sweep result JSON with summary and num_prompts for analysis tests."""
    subdir = tmpdir / press
    subdir.mkdir(parents=True, exist_ok=True)
    fname = f"test-model_{ratio:.3f}_{n_prompts}p.json"
    path = subdir / fname

    # Compute summary from results
    total = len(results)
    correct_count = sum(1 for r in results if r.get("correct", False))
    cats: dict[str, int] = {}
    has_catastrophe = 0
    for r in results:
        if r["catastrophes"]:
            has_catastrophe += 1
        for c in r["catastrophes"]:
            cats[c] = cats.get(c, 0) + 1

    data = {
        "config": {
            "model_name": "test-model",
            "press_name": press,
            "compression_ratio": ratio,
            "num_prompts": n_prompts,
        },
        "summary": {
            "total": total,
            "accuracy": correct_count / total if total else 0,
            "catastrophic_failure_rate": has_catastrophe / total if total else 0,
            "catastrophe_counts": cats,
            "avg_tokens": sum(r["num_tokens_generated"] for r in results) / total if total else 0,
        },
        "results": results,
    }
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# Tests: degradation_table
# ---------------------------------------------------------------------------


class TestDegradationTable:
    def test_prints_table(self, tmp_path: Path, capsys: object) -> None:
        """degradation_table prints a formatted table with press/ratio rows."""
        import _pytest.capture

        capsys_fixture: _pytest.capture.CaptureFixture[str] = capsys  # type: ignore[assignment]
        # Baseline: correct, no catastrophes
        baseline_results = [
            make_result_json(n_tokens=20, correct=True, catastrophes=[]),
            make_result_json(n_tokens=20, correct=True, catastrophes=[], prompt_id="gsm8k_1"),
        ]
        _write_analysis_file(tmp_path, "none", 0.0, baseline_results)

        # Compressed: one correct, one failing
        compressed_results = [
            make_result_json(n_tokens=20, correct=True, catastrophes=[]),
            make_result_json(
                n_tokens=20,
                correct=False,
                catastrophes=["looping"],
                catastrophe_onsets={"looping": 10},
                prompt_id="gsm8k_1",
            ),
        ]
        _write_analysis_file(tmp_path, "streaming_llm", 0.875, compressed_results)

        degradation_table(tmp_path, num_prompts=50)
        captured = capsys_fixture.readouterr()
        assert "streaming_llm" in captured.out
        assert "none" in captured.out
        assert "0.875" in captured.out

    def test_no_results_warns(self, tmp_path: Path) -> None:
        """degradation_table handles empty directory gracefully."""
        # Should not raise, just log a warning
        degradation_table(tmp_path, num_prompts=50)


# ---------------------------------------------------------------------------
# Tests: silent_failure_analysis
# ---------------------------------------------------------------------------


class TestSilentFailureAnalysis:
    def test_classifies_failures(self, tmp_path: Path, capsys: object) -> None:
        """silent_failure_analysis categorizes runs into correct/silent/loud."""
        import _pytest.capture

        capsys_fixture: _pytest.capture.CaptureFixture[str] = capsys  # type: ignore[assignment]
        results = [
            # Correct run
            make_result_json(n_tokens=20, correct=True, catastrophes=[]),
            # Silent failure (wrong answer only)
            make_result_json(
                n_tokens=20,
                correct=False,
                catastrophes=["wrong_answer"],
                prompt_id="gsm8k_1",
                predicted_answer="99",
            ),
            # Loud failure (looping)
            make_result_json(
                n_tokens=20,
                correct=False,
                catastrophes=["looping"],
                catastrophe_onsets={"looping": 10},
                prompt_id="gsm8k_2",
            ),
        ]
        _write_analysis_file(tmp_path, "streaming_llm", 0.875, results)

        silent_failure_analysis(tmp_path, num_prompts=50)
        captured = capsys_fixture.readouterr()
        assert "Correct runs: 1" in captured.out
        assert "Silent failures (wrong only): 1" in captured.out
        assert "Loud failures (loop/non-term): 1" in captured.out

    def test_no_failures(self, tmp_path: Path, capsys: object) -> None:
        """No failures produces appropriate message."""
        import _pytest.capture

        capsys_fixture: _pytest.capture.CaptureFixture[str] = capsys  # type: ignore[assignment]
        results = [make_result_json(n_tokens=20, correct=True, catastrophes=[])]
        _write_analysis_file(tmp_path, "streaming_llm", 0.5, results)

        silent_failure_analysis(tmp_path, num_prompts=50)
        captured = capsys_fixture.readouterr()
        assert "No failures to analyze" in captured.out

    def test_only_baseline_skipped(self, tmp_path: Path, capsys: object) -> None:
        """Only baseline runs are skipped (press_name == 'none')."""
        import _pytest.capture

        capsys_fixture: _pytest.capture.CaptureFixture[str] = capsys  # type: ignore[assignment]
        results = [make_result_json(n_tokens=20, correct=True)]
        _write_analysis_file(tmp_path, "none", 0.0, results)

        silent_failure_analysis(tmp_path, num_prompts=50)
        captured = capsys_fixture.readouterr()
        # Only baseline => no compressed runs => no output about counts
        assert "Correct runs: 0" in captured.out


# ---------------------------------------------------------------------------
# Tests: delta_h_analysis
# ---------------------------------------------------------------------------


class TestDeltaHAnalysis:
    def _setup_data(self, tmp_path: Path) -> None:
        """Create baseline + compressed data with catastrophes."""
        # Baseline: low entropy, small delta_h
        baseline_sigs = [
            make_signal_dict(entropy=1.0, delta_h=None),  # first token
            make_signal_dict(entropy=1.1, delta_h=0.1),
            make_signal_dict(entropy=1.0, delta_h=-0.1),
            make_signal_dict(entropy=1.2, delta_h=0.2),
            make_signal_dict(entropy=1.1, delta_h=-0.1),
        ]
        baseline_result = {
            "prompt_id": "gsm8k_0",
            "prompt_text": "test",
            "model": "test-model",
            "press": "none",
            "compression_ratio": 0.0,
            "max_new_tokens": 512,
            "seed": 42,
            "generated_text": "#### 42",
            "ground_truth": "42",
            "predicted_answer": "42",
            "correct": True,
            "stop_reason": "eos",
            "catastrophes": [],
            "catastrophe_onsets": {},
            "num_tokens_generated": 5,
            "cache_size_after_prefill": None,
            "signals": baseline_sigs,
        }
        _write_analysis_file(tmp_path, "none", 0.0, [baseline_result])

        # Compressed: entropy spike + catastrophe
        compressed_sigs = [
            make_signal_dict(entropy=1.0, delta_h=None),
            make_signal_dict(entropy=1.1, delta_h=0.1),
            make_signal_dict(entropy=5.0, delta_h=3.9),  # big spike
            make_signal_dict(entropy=6.0, delta_h=1.0),
            make_signal_dict(entropy=7.0, delta_h=1.0),
        ]
        compressed_result = {
            "prompt_id": "gsm8k_0",
            "prompt_text": "test",
            "model": "test-model",
            "press": "streaming_llm",
            "compression_ratio": 0.875,
            "max_new_tokens": 512,
            "seed": 42,
            "generated_text": "So So So So",
            "ground_truth": "42",
            "predicted_answer": None,
            "correct": False,
            "stop_reason": "max_tokens",
            "catastrophes": ["looping"],
            "catastrophe_onsets": {"looping": 3},
            "num_tokens_generated": 5,
            "cache_size_after_prefill": None,
            "signals": compressed_sigs,
        }
        _write_analysis_file(tmp_path, "streaming_llm", 0.875, [compressed_result])

    def test_runs_with_baseline_and_compressed(self, tmp_path: Path, capsys: object) -> None:
        """delta_h_analysis calibrates thresholds from baseline and reports results."""
        import _pytest.capture

        capsys_fixture: _pytest.capture.CaptureFixture[str] = capsys  # type: ignore[assignment]
        self._setup_data(tmp_path)
        delta_h_analysis(tmp_path, num_prompts=50)
        captured = capsys_fixture.readouterr()
        assert "Entropy threshold" in captured.out
        assert "DeltaH" in captured.out
        assert "Failing runs analyzed: 1" in captured.out

    def test_no_baseline_warns(self, tmp_path: Path) -> None:
        """delta_h_analysis handles missing baseline gracefully."""
        results = [
            make_result_json(
                n_tokens=10,
                catastrophes=["looping"],
                catastrophe_onsets={"looping": 5},
            )
        ]
        _write_analysis_file(tmp_path, "streaming_llm", 0.875, results)
        # Should not raise
        delta_h_analysis(tmp_path, num_prompts=50)

    def test_no_delta_h_in_baseline(self, tmp_path: Path) -> None:
        """delta_h_analysis handles baseline with no delta_h data gracefully."""
        # Baseline with only entropy (no delta_h)
        baseline_sigs = [
            make_signal_dict(entropy=1.0, delta_h=None),
            make_signal_dict(entropy=1.1, delta_h=None),
        ]
        baseline_result = {
            "prompt_id": "gsm8k_0",
            "prompt_text": "test",
            "model": "test-model",
            "press": "none",
            "compression_ratio": 0.0,
            "max_new_tokens": 512,
            "seed": 42,
            "generated_text": "#### 42",
            "ground_truth": "42",
            "predicted_answer": "42",
            "correct": True,
            "stop_reason": "eos",
            "catastrophes": [],
            "catastrophe_onsets": {},
            "num_tokens_generated": 2,
            "cache_size_after_prefill": None,
            "signals": baseline_sigs,
        }
        _write_analysis_file(tmp_path, "none", 0.0, [baseline_result])
        # Should not raise (returns early with warning)
        delta_h_analysis(tmp_path, num_prompts=50)


# ---------------------------------------------------------------------------
# Tests: rolling_delta_h_detector
# ---------------------------------------------------------------------------


class TestRollingDeltaHDetector:
    def test_reports_precision_recall(self, tmp_path: Path, capsys: object) -> None:
        """rolling_delta_h_detector reports P/R/F1 at multiple thresholds."""
        import _pytest.capture

        capsys_fixture: _pytest.capture.CaptureFixture[str] = capsys  # type: ignore[assignment]
        # Baseline: stable entropy
        baseline_sigs = [make_signal_dict(entropy=1.0) for _ in range(20)]
        baseline_result = {
            "prompt_id": "gsm8k_0",
            "prompt_text": "test",
            "model": "test-model",
            "press": "none",
            "compression_ratio": 0.0,
            "max_new_tokens": 512,
            "seed": 42,
            "generated_text": "#### 42",
            "ground_truth": "42",
            "predicted_answer": "42",
            "correct": True,
            "stop_reason": "eos",
            "catastrophes": [],
            "catastrophe_onsets": {},
            "num_tokens_generated": 20,
            "cache_size_after_prefill": None,
            "signals": baseline_sigs,
        }
        _write_analysis_file(tmp_path, "none", 0.0, [baseline_result])

        # Compressed: entropy spike in catastrophe
        compressed_sigs = [make_signal_dict(entropy=1.0) for _ in range(10)]
        compressed_sigs += [make_signal_dict(entropy=5.0 + i) for i in range(10)]
        catastrophe_result = {
            "prompt_id": "gsm8k_1",
            "prompt_text": "test",
            "model": "test-model",
            "press": "streaming_llm",
            "compression_ratio": 0.875,
            "max_new_tokens": 512,
            "seed": 42,
            "generated_text": "So So So",
            "ground_truth": "42",
            "predicted_answer": None,
            "correct": False,
            "stop_reason": "max_tokens",
            "catastrophes": ["looping"],
            "catastrophe_onsets": {"looping": 10},
            "num_tokens_generated": 20,
            "cache_size_after_prefill": None,
            "signals": compressed_sigs,
        }
        # Non-catastrophe compressed run
        clean_sigs = [make_signal_dict(entropy=1.0 + 0.01 * i) for i in range(20)]
        clean_result = {
            "prompt_id": "gsm8k_2",
            "prompt_text": "test",
            "model": "test-model",
            "press": "streaming_llm",
            "compression_ratio": 0.875,
            "max_new_tokens": 512,
            "seed": 42,
            "generated_text": "#### 42",
            "ground_truth": "42",
            "predicted_answer": "42",
            "correct": True,
            "stop_reason": "eos",
            "catastrophes": [],
            "catastrophe_onsets": {},
            "num_tokens_generated": 20,
            "cache_size_after_prefill": None,
            "signals": clean_sigs,
        }
        _write_analysis_file(tmp_path, "streaming_llm", 0.875, [catastrophe_result, clean_result])

        rolling_delta_h_detector(tmp_path, num_prompts=50, window=5)
        captured = capsys_fixture.readouterr()
        assert "Baseline rolling" in captured.out
        # Should report for each percentile threshold
        assert "P90" in captured.out
        assert "P95" in captured.out
        assert "P99" in captured.out
        # Should contain P=, R=, F1= metrics
        assert "P=" in captured.out
        assert "R=" in captured.out
        assert "F1=" in captured.out

    def test_no_baseline_warns(self, tmp_path: Path) -> None:
        """rolling_delta_h_detector handles missing baseline gracefully."""
        results = [make_result_json(n_tokens=20)]
        _write_analysis_file(tmp_path, "streaming_llm", 0.5, results)
        # Should not raise
        rolling_delta_h_detector(tmp_path, num_prompts=50)


# ---------------------------------------------------------------------------
# Tests: compare_runs
# ---------------------------------------------------------------------------


class TestCompareRuns:
    def test_prints_comparison(self, tmp_path: Path, capsys: object) -> None:
        """compare_runs outputs a formatted comparison table."""
        import _pytest.capture

        capsys_fixture: _pytest.capture.CaptureFixture[str] = capsys  # type: ignore[assignment]
        results1 = [make_result_json(n_tokens=20, correct=True)]
        results2 = [
            make_result_json(
                n_tokens=20,
                correct=False,
                catastrophes=["looping"],
                catastrophe_onsets={"looping": 10},
            )
        ]
        _write_analysis_file(tmp_path, "none", 0.0, results1)
        _write_analysis_file(tmp_path, "streaming_llm", 0.875, results2)

        compare_runs(tmp_path)
        captured = capsys_fixture.readouterr()
        assert "none" in captured.out
        assert "streaming_llm" in captured.out
        assert "0.875" in captured.out

    def test_num_prompts_filter(self, tmp_path: Path, capsys: object) -> None:
        """compare_runs respects num_prompts filter."""
        import _pytest.capture

        capsys_fixture: _pytest.capture.CaptureFixture[str] = capsys  # type: ignore[assignment]
        results = [make_result_json(n_tokens=10)]
        _write_analysis_file(tmp_path, "none", 0.0, results, n_prompts=50)
        _write_analysis_file(tmp_path, "streaming_llm", 0.5, results, n_prompts=100)

        compare_runs(tmp_path, num_prompts=50)
        captured = capsys_fixture.readouterr()
        assert "none" in captured.out
        # streaming_llm has n_prompts=100, should be filtered out
        assert "streaming_llm" not in captured.out

    def test_empty_dir(self, tmp_path: Path) -> None:
        """compare_runs handles empty directory gracefully."""
        compare_runs(tmp_path)
