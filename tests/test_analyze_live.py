"""Tests for live validation analysis module."""

import json
from pathlib import Path

from kvguard.analyze_live import (
    analyze_live_validation,
    compare_offline_vs_live,
    compute_accuracy,
    compute_cbr_reduction,
    compute_false_trigger_rate,
    compute_hazard_calibration,
    compute_latency_overhead,
    compute_lead_times,
    compute_memory_savings,
    compute_mode_transitions,
    fmt_ci,
    load_live_results,
    wilson_ci,
)
from kvguard.config import LiveResult

# ---------------------------------------------------------------------------
# Helper to build minimal LiveResult
# ---------------------------------------------------------------------------


def _lr(
    *,
    prompt_id: str = "p0",
    correct: bool | None = True,
    catastrophes: list[str] | None = None,
    catastrophe_onsets: dict[str, int] | None = None,
    safe_trigger_token: int | None = None,
    mode_history: list[int] | None = None,
    hazard_probs: list[float] | None = None,
    cache_sizes: list[int] | None = None,
    num_tokens: int = 100,
    generation_time: float = 1.0,
) -> LiveResult:
    """Build a minimal LiveResult for testing."""
    return LiveResult(
        prompt_id=prompt_id,
        prompt_text="test",
        model="test-model",
        compression_ratio=0.875,
        max_new_tokens=512,
        seed=42,
        generated_text="test output",
        ground_truth="42",
        predicted_answer="42" if correct else "0",
        correct=correct,
        stop_reason="eos",
        catastrophes=catastrophes or [],
        catastrophe_onsets=catastrophe_onsets or {},
        num_tokens_generated=num_tokens,
        signals=[],
        controlled=True,
        mode_history=mode_history or [],
        hazard_probs=hazard_probs or [],
        eviction_history=[],
        cache_sizes=cache_sizes or [],
        safe_trigger_token=safe_trigger_token,
        generation_time_seconds=generation_time,
    )


# ---------------------------------------------------------------------------
# Tests: wilson_ci and fmt_ci
# ---------------------------------------------------------------------------


class TestWilsonCI:
    def test_zero_n(self) -> None:
        lo, hi = wilson_ci(0, 0)
        assert lo == 0.0
        assert hi == 0.0

    def test_zero_successes(self) -> None:
        lo, hi = wilson_ci(0, 100)
        assert lo == 0.0
        assert hi > 0.0  # upper bound is non-zero

    def test_all_successes(self) -> None:
        lo, hi = wilson_ci(100, 100)
        assert lo > 0.0
        assert hi > 0.96  # upper bound is near 1.0

    def test_half(self) -> None:
        lo, hi = wilson_ci(50, 100)
        assert 0.39 < lo < 0.42
        assert 0.58 < hi < 0.61

    def test_fmt_ci_format(self) -> None:
        result = fmt_ci(50, 100)
        assert "50/100" in result
        assert "%" in result


# ---------------------------------------------------------------------------
# Tests: compute_accuracy
# ---------------------------------------------------------------------------


class TestComputeAccuracy:
    def test_all_correct(self) -> None:
        results = [_lr(correct=True) for _ in range(10)]
        acc = compute_accuracy(results)
        assert acc["correct"] == 10
        assert acc["n"] == 10

    def test_mixed(self) -> None:
        results = [_lr(correct=True)] * 3 + [_lr(correct=False)] * 7
        acc = compute_accuracy(results)
        assert acc["correct"] == 3
        assert acc["n"] == 10

    def test_none_correct(self) -> None:
        results = [_lr(correct=None)] * 5
        acc = compute_accuracy(results)
        assert acc["correct"] == 0


# ---------------------------------------------------------------------------
# Tests: compute_cbr_reduction
# ---------------------------------------------------------------------------


class TestComputeCbrReduction:
    def test_no_catastrophes(self) -> None:
        static = [_lr(catastrophes=[]) for _ in range(10)]
        ctrl = [_lr(catastrophes=[]) for _ in range(10)]
        cbr = compute_cbr_reduction(static, ctrl)
        assert cbr["static_cbr_count"] == 0
        assert cbr["controlled_cbr_count"] == 0
        assert cbr["reduction_pct"] == 0.0

    def test_full_prevention(self) -> None:
        static = [_lr(catastrophes=["looping"])] * 5 + [_lr(catastrophes=[])] * 5
        ctrl = [_lr(catastrophes=[])] * 10
        cbr = compute_cbr_reduction(static, ctrl)
        assert cbr["static_cbr_count"] == 5
        assert cbr["controlled_cbr_count"] == 0
        assert cbr["reduction_pct"] == 100.0

    def test_partial_prevention(self) -> None:
        static = [_lr(catastrophes=["looping"])] * 4
        ctrl = [_lr(catastrophes=["looping"])] * 2 + [_lr(catastrophes=[])] * 2
        cbr = compute_cbr_reduction(static, ctrl)
        assert cbr["static_cbr_count"] == 4
        assert cbr["controlled_cbr_count"] == 2
        assert cbr["reduction_pct"] == 50.0

    def test_non_termination_counted(self) -> None:
        static = [_lr(catastrophes=["non_termination"])]
        ctrl = [_lr(catastrophes=[])]
        cbr = compute_cbr_reduction(static, ctrl)
        assert cbr["static_cbr_count"] == 1

    def test_wrong_answer_not_counted(self) -> None:
        static = [_lr(catastrophes=["wrong_answer"])]
        ctrl = [_lr(catastrophes=["wrong_answer"])]
        cbr = compute_cbr_reduction(static, ctrl)
        assert cbr["static_cbr_count"] == 0


# ---------------------------------------------------------------------------
# Tests: compute_lead_times
# ---------------------------------------------------------------------------


class TestComputeLeadTimes:
    def test_trigger_before_onset(self) -> None:
        results = [
            _lr(
                safe_trigger_token=10,
                catastrophes=["looping"],
                catastrophe_onsets={"looping": 50},
            )
        ]
        lt = compute_lead_times(results)
        assert lt["count_triggered_before_onset"] == 1
        assert lt["count_triggered_after_onset"] == 0
        assert lt["mean"] == 40.0

    def test_trigger_after_onset(self) -> None:
        results = [
            _lr(
                safe_trigger_token=60,
                catastrophes=["looping"],
                catastrophe_onsets={"looping": 50},
            )
        ]
        lt = compute_lead_times(results)
        assert lt["count_triggered_before_onset"] == 0
        assert lt["count_triggered_after_onset"] == 1

    def test_no_trigger(self) -> None:
        results = [_lr(safe_trigger_token=None, catastrophes=["looping"])]
        lt = compute_lead_times(results)
        assert lt["count_triggered_before_onset"] == 0
        assert lt["count_triggered_after_onset"] == 0

    def test_wrong_answer_ignored(self) -> None:
        results = [
            _lr(
                safe_trigger_token=10,
                catastrophes=["wrong_answer"],
                catastrophe_onsets={"wrong_answer": 50},
            )
        ]
        lt = compute_lead_times(results)
        assert lt["count_triggered_before_onset"] == 0


# ---------------------------------------------------------------------------
# Tests: compute_false_trigger_rate
# ---------------------------------------------------------------------------


class TestComputeFalseTriggerRate:
    def test_true_positive(self) -> None:
        static = [_lr(prompt_id="p0", catastrophes=["looping"])]
        ctrl = [_lr(prompt_id="p0", safe_trigger_token=10)]
        ft = compute_false_trigger_rate(static, ctrl)
        assert ft["total_triggers"] == 1
        assert ft["false_triggers"] == 0

    def test_false_positive(self) -> None:
        static = [_lr(prompt_id="p0", catastrophes=[])]
        ctrl = [_lr(prompt_id="p0", safe_trigger_token=10)]
        ft = compute_false_trigger_rate(static, ctrl)
        assert ft["total_triggers"] == 1
        assert ft["false_triggers"] == 1

    def test_no_triggers(self) -> None:
        static = [_lr(prompt_id="p0")]
        ctrl = [_lr(prompt_id="p0", safe_trigger_token=None)]
        ft = compute_false_trigger_rate(static, ctrl)
        assert ft["total_triggers"] == 0
        assert ft["false_triggers"] == 0


# ---------------------------------------------------------------------------
# Tests: compute_memory_savings
# ---------------------------------------------------------------------------


class TestComputeMemorySavings:
    def test_static_savings(self) -> None:
        baseline = [_lr(prompt_id="p0", cache_sizes=[100])]
        static = [_lr(prompt_id="p0", cache_sizes=[20])]
        ctrl = [_lr(prompt_id="p0", cache_sizes=[50])]
        mem = compute_memory_savings(baseline, static, ctrl)
        assert mem["static_mean_savings_pct"] == 80.0  # 1 - 20/100
        assert mem["controlled_mean_savings_pct"] == 50.0  # 1 - 50/100

    def test_empty_cache_sizes(self) -> None:
        baseline = [_lr(prompt_id="p0", cache_sizes=[])]
        static = [_lr(prompt_id="p0", cache_sizes=[])]
        ctrl = [_lr(prompt_id="p0", cache_sizes=[])]
        mem = compute_memory_savings(baseline, static, ctrl)
        assert "static_mean_savings_pct" not in mem


# ---------------------------------------------------------------------------
# Tests: compute_hazard_calibration
# ---------------------------------------------------------------------------


class TestComputeHazardCalibration:
    def test_catastrophe_higher_hazard(self) -> None:
        results = [
            _lr(
                prompt_id="p0",
                catastrophes=["looping"],
                catastrophe_onsets={"looping": 2},
                hazard_probs=[0.1, 0.5, 0.9, 0.95],
            ),
            _lr(
                prompt_id="p1",
                catastrophes=[],
                hazard_probs=[0.05, 0.1, 0.08, 0.03],
            ),
        ]
        haz = compute_hazard_calibration(results)
        assert haz["mean_max_hazard_catastrophe"] > haz["mean_max_hazard_clean"]
        assert haz["max_hazard_separation"] > 0

    def test_hazard_at_onset(self) -> None:
        results = [
            _lr(
                catastrophes=["looping"],
                catastrophe_onsets={"looping": 2},
                hazard_probs=[0.1, 0.5, 0.8, 0.9],
            ),
        ]
        haz = compute_hazard_calibration(results)
        assert haz["mean_hazard_at_onset"] == 0.8  # index 2

    def test_empty_hazard_probs(self) -> None:
        results = [_lr(hazard_probs=[])]
        haz = compute_hazard_calibration(results)
        assert "mean_max_hazard_catastrophe" not in haz


# ---------------------------------------------------------------------------
# Tests: compute_mode_transitions
# ---------------------------------------------------------------------------


class TestComputeModeTransitions:
    def test_transitions_counted(self) -> None:
        results = [
            _lr(
                mode_history=[0, 0, 1, 1, 2, 2, 1, 0],
                safe_trigger_token=4,
            )
        ]
        mt = compute_mode_transitions(results)
        assert mt["traces_reaching_safe"] == 1
        assert mt["transitions"]["NORMAL→ALERT"] == 1
        assert mt["transitions"]["ALERT→SAFE"] == 1
        assert mt["transitions"]["SAFE→ALERT"] == 1
        assert mt["transitions"]["ALERT→NORMAL"] == 1

    def test_no_transitions(self) -> None:
        results = [_lr(mode_history=[0, 0, 0, 0])]
        mt = compute_mode_transitions(results)
        assert mt["traces_reaching_safe"] == 0
        assert mt["transitions"] == {}

    def test_empty_mode_history(self) -> None:
        results = [_lr(mode_history=[])]
        mt = compute_mode_transitions(results)
        assert mt["traces_reaching_safe"] == 0


# ---------------------------------------------------------------------------
# Helper to write condition JSON files
# ---------------------------------------------------------------------------


def _write_condition(
    output_dir: Path,
    condition: str,
    results: list[LiveResult],
    num_prompts: int = 10,
) -> None:
    data = {
        "condition": condition,
        "model": "test-model",
        "num_prompts": num_prompts,
        "results": [r.model_dump() for r in results],
    }
    fpath = output_dir / f"{condition}_{num_prompts}p.json"
    fpath.write_text(json.dumps(data, indent=2, default=str))


# ---------------------------------------------------------------------------
# Tests: load_live_results
# ---------------------------------------------------------------------------


class TestLoadLiveResults:
    def test_loads_conditions(self, tmp_path: Path) -> None:
        r = _lr(prompt_id="p0")
        _write_condition(tmp_path, "baseline", [r])
        _write_condition(tmp_path, "static_0.75", [r])

        conds = load_live_results(tmp_path)
        assert "baseline" in conds
        assert "static_0.75" in conds
        assert len(conds["baseline"]) == 1

    def test_empty_dir(self, tmp_path: Path) -> None:
        assert load_live_results(tmp_path) == {}

    def test_result_types(self, tmp_path: Path) -> None:
        _write_condition(tmp_path, "baseline", [_lr()])
        conds = load_live_results(tmp_path)
        assert isinstance(conds["baseline"][0], LiveResult)


# ---------------------------------------------------------------------------
# Tests: compute_latency_overhead
# ---------------------------------------------------------------------------


class TestComputeLatencyOverhead:
    def test_overhead_computed(self) -> None:
        baseline = [_lr(prompt_id="p0", generation_time=1.0)]
        static = [_lr(prompt_id="p0", generation_time=0.8)]
        ctrl = [_lr(prompt_id="p0", generation_time=0.9)]
        lat = compute_latency_overhead(baseline, static, ctrl)
        assert lat["mean_baseline_time_s"] == 1.0
        assert lat["mean_static_time_s"] == 0.8
        assert lat["mean_controlled_time_s"] == 0.9
        # overhead vs static = (0.9 - 0.8) / 0.8 * 100 = 12.5%
        assert abs(lat["overhead_vs_static_pct"] - 12.5) < 0.1

    def test_multiple_prompts(self) -> None:
        baseline = [
            _lr(prompt_id="p0", generation_time=1.0),
            _lr(prompt_id="p1", generation_time=2.0),
        ]
        static = [
            _lr(prompt_id="p0", generation_time=0.8),
            _lr(prompt_id="p1", generation_time=1.6),
        ]
        ctrl = [
            _lr(prompt_id="p0", generation_time=0.9),
            _lr(prompt_id="p1", generation_time=1.8),
        ]
        lat = compute_latency_overhead(baseline, static, ctrl)
        assert lat["mean_baseline_time_s"] == 1.5
        assert lat["mean_static_time_s"] == 1.2
        assert lat["mean_controlled_time_s"] == 1.35


# ---------------------------------------------------------------------------
# Tests: compare_offline_vs_live
# ---------------------------------------------------------------------------


class TestCompareOfflineVsLive:
    def test_without_offline(self) -> None:
        static = [_lr(prompt_id="p0", catastrophes=["looping"])]
        ctrl = [_lr(prompt_id="p0")]
        result = compare_offline_vs_live(None, static, ctrl, 0.75)
        assert result["live_reduction_pct"] == 100.0
        assert "offline_reduction_pct" not in result

    def test_with_offline_data(self, tmp_path: Path) -> None:
        offline_path = tmp_path / "controller_eval.json"
        offline_data = {
            "budgets": [
                {
                    "press": "streaming_llm",
                    "compression_ratio": 0.75,
                    "cfr_reduction_pct": 50.0,
                    "baseline_cfr_count": 10,
                    "n_prompts": 100,
                },
            ]
        }
        offline_path.write_text(json.dumps(offline_data))

        static = [
            _lr(prompt_id="p0", catastrophes=["looping"]),
            _lr(prompt_id="p1"),
        ]
        ctrl = [_lr(prompt_id="p0"), _lr(prompt_id="p1")]
        result = compare_offline_vs_live(offline_path, static, ctrl, 0.75)
        assert result["live_reduction_pct"] == 100.0
        assert result["offline_reduction_pct"] == 50.0
        assert result["simulation_gap_pp"] == 50.0
        assert "exceeded" in result["interpretation"]

    def test_no_catastrophes(self) -> None:
        static = [_lr(prompt_id="p0")]
        ctrl = [_lr(prompt_id="p0")]
        result = compare_offline_vs_live(None, static, ctrl, 0.75)
        assert result["live_reduction_pct"] == 0.0

    def test_underperformance(self, tmp_path: Path) -> None:
        offline_path = tmp_path / "controller_eval.json"
        offline_data = {
            "budgets": [
                {
                    "press": "streaming_llm",
                    "compression_ratio": 0.875,
                    "cfr_reduction_pct": 80.0,
                    "baseline_cfr_count": 20,
                    "n_prompts": 100,
                },
            ]
        }
        offline_path.write_text(json.dumps(offline_data))

        # Live: 2 cat static, 1 cat controlled = 50% reduction (< 80% offline)
        static = [
            _lr(prompt_id="p0", catastrophes=["looping"]),
            _lr(prompt_id="p1", catastrophes=["looping"]),
        ]
        ctrl = [
            _lr(prompt_id="p0"),
            _lr(prompt_id="p1", catastrophes=["looping"]),
        ]
        result = compare_offline_vs_live(offline_path, static, ctrl, 0.875)
        assert result["live_reduction_pct"] == 50.0
        assert "underperformed" in result["interpretation"]


# ---------------------------------------------------------------------------
# Tests: analyze_live_validation (integration)
# ---------------------------------------------------------------------------


class TestAnalyzeLiveValidation:
    def test_full_analysis(self, tmp_path: Path) -> None:
        baseline = [
            _lr(prompt_id="p0", correct=True, generation_time=1.0, cache_sizes=[100]),
            _lr(prompt_id="p1", correct=True, generation_time=1.2, cache_sizes=[110]),
        ]
        static = [
            _lr(
                prompt_id="p0",
                correct=False,
                catastrophes=["looping"],
                catastrophe_onsets={"looping": 30},
                generation_time=0.8,
                cache_sizes=[50],
            ),
            _lr(prompt_id="p1", correct=True, generation_time=0.9, cache_sizes=[55]),
        ]
        controlled = [
            _lr(
                prompt_id="p0",
                correct=True,
                safe_trigger_token=20,
                mode_history=[0, 0, 1, 1, 2, 2],
                hazard_probs=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9],
                generation_time=0.85,
                cache_sizes=[70],
            ),
            _lr(
                prompt_id="p1",
                correct=True,
                mode_history=[0, 0, 0, 0],
                hazard_probs=[0.05, 0.1, 0.08, 0.06],
                generation_time=0.95,
                cache_sizes=[56],
            ),
        ]

        _write_condition(tmp_path, "baseline", baseline, num_prompts=2)
        _write_condition(tmp_path, "static_0.75", static, num_prompts=2)
        _write_condition(tmp_path, "controlled_0.75", controlled, num_prompts=2)

        report = analyze_live_validation(tmp_path)
        assert "conditions" in report
        assert "accuracy" in report
        assert "ratio_0.75" in report

        ratio_rpt = report["ratio_0.75"]
        assert "cbr_reduction" in ratio_rpt
        assert "lead_times" in ratio_rpt
        assert "false_triggers" in ratio_rpt
        assert "memory_savings" in ratio_rpt
        assert "hazard_calibration" in ratio_rpt
        assert "mode_transitions" in ratio_rpt
        assert "latency" in ratio_rpt
        assert "offline_comparison" in ratio_rpt

    def test_empty_dir(self, tmp_path: Path) -> None:
        report = analyze_live_validation(tmp_path)
        assert report == {}

    def test_baseline_only(self, tmp_path: Path) -> None:
        _write_condition(tmp_path, "baseline", [_lr()])
        report = analyze_live_validation(tmp_path)
        assert "conditions" in report
        assert not any(k.startswith("ratio_") for k in report)
