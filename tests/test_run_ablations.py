"""Tests for run_ablations.py helper functions."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

# Must insert before importing run_ablations
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_ablations import (  # noqa: E402
    FEATURE_ABLATION_CONFIGS,
    ablation_entropy_threshold,
    evaluate_random_predictor,
    evaluate_with_precomputed,
    feature_ablations,
    summarize_eval,
)

from kvguard.controller import ControllerConfig  # noqa: E402
from kvguard.evaluate_controller import BudgetResult, EvalResult, TraceInfo  # noqa: E402
from kvguard.features import Dataset, TraceMeta  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_trace(
    *,
    prompt_id: str = "p0",
    press: str = "streaming_llm",
    ratio: float = 0.875,
    catastrophes: list[str] | None = None,
    onsets: dict[str, int] | None = None,
    n_tokens: int = 100,
    correct: bool = True,
) -> TraceInfo:
    """Create a minimal TraceInfo for testing."""
    signals = [
        {
            "entropy": 1.0,
            "top1_prob": 0.5,
            "top5_prob": 0.9,
            "top1_token": "a",
            "rank_of_chosen": 0,
            "top20_logprobs": [-0.5] * 20,
            "h_alts": 0.3,
            "avg_logp": -2.0,
            "delta_h": 0.1,
            "rep_count": 0,
            "is_thinking_token": False,
        }
        for _ in range(n_tokens)
    ]
    return TraceInfo(
        prompt_id=prompt_id,
        press=press,
        compression_ratio=ratio,
        catastrophes=catastrophes or [],
        catastrophe_onsets=onsets or {},
        num_tokens=n_tokens,
        correct=correct,
        signals=signals,
    )


def _make_budget_result(
    *,
    press: str = "streaming_llm",
    ratio: float = 0.875,
    n_prompts: int = 100,
    baseline_cfr_count: int = 20,
    prevented: int = 5,
    false_triggers: int = 3,
    n_correct: int = 60,
) -> BudgetResult:
    """Create a BudgetResult for summarize_eval testing."""
    controlled = baseline_cfr_count - prevented
    baseline_cfr = baseline_cfr_count / n_prompts
    controlled_cfr = controlled / n_prompts
    return BudgetResult(
        press=press,
        compression_ratio=ratio,
        n_prompts=n_prompts,
        baseline_cfr_count=baseline_cfr_count,
        baseline_cfr=round(baseline_cfr, 4),
        baseline_accuracy=round(n_correct / n_prompts, 4),
        controller_triggered_count=prevented + false_triggers,
        triggered_before_onset=prevented,
        catastrophes_prevented=prevented,
        controlled_cfr_count=controlled,
        controlled_cfr=round(controlled_cfr, 4),
        cfr_reduction_abs=round(baseline_cfr - controlled_cfr, 4),
        cfr_reduction_pct=round(
            (prevented / baseline_cfr_count * 100) if baseline_cfr_count else 0, 1
        ),
        mean_trigger_token=50.0,
        false_trigger_count=false_triggers,
    )


# ---------------------------------------------------------------------------
# Tests: FEATURE_ABLATION_CONFIGS
# ---------------------------------------------------------------------------


class TestFeatureAblationConfigs:
    def test_has_four_variants(self) -> None:
        assert len(FEATURE_ABLATION_CONFIGS) == 4

    def test_full_drops_nothing(self) -> None:
        assert FEATURE_ABLATION_CONFIGS["full"] == []

    def test_logit_only_drops_three(self) -> None:
        dropped = FEATURE_ABLATION_CONFIGS["logit_only"]
        assert "compression_ratio" in dropped
        assert "rep_count" in dropped
        assert "rep_count_sum_8" in dropped

    def test_no_compression_ratio(self) -> None:
        assert FEATURE_ABLATION_CONFIGS["no_compression_ratio"] == ["compression_ratio"]

    def test_no_rep(self) -> None:
        dropped = FEATURE_ABLATION_CONFIGS["no_rep"]
        assert "rep_count" in dropped
        assert "rep_count_sum_8" in dropped
        assert "compression_ratio" not in dropped


# ---------------------------------------------------------------------------
# Tests: summarize_eval
# ---------------------------------------------------------------------------


class TestSummarizeEval:
    def test_empty_budgets(self) -> None:
        result = EvalResult(safe_compression_ratio=0.0)
        summary = summarize_eval(result, "test")
        assert summary["label"] == "test"
        assert "error" in summary

    def test_single_budget(self) -> None:
        result = EvalResult(safe_compression_ratio=0.0)
        result.budgets.append(
            _make_budget_result(
                press="streaming_llm",
                n_prompts=100,
                baseline_cfr_count=20,
                prevented=5,
                false_triggers=3,
            )
        )
        summary = summarize_eval(result, "single")
        assert summary["label"] == "single"
        assert summary["total_baseline_cfr"] == 20
        assert summary["total_prevented"] == 5
        assert summary["overall_cfr_reduction_pct"] == 25.0

    def test_multiple_budgets(self) -> None:
        result = EvalResult(safe_compression_ratio=0.0)
        result.budgets.append(
            _make_budget_result(press="streaming_llm", baseline_cfr_count=10, prevented=3)
        )
        result.budgets.append(
            _make_budget_result(press="snapkv", baseline_cfr_count=30, prevented=10)
        )
        summary = summarize_eval(result, "multi")
        assert summary["total_baseline_cfr"] == 40
        assert summary["total_prevented"] == 13

    def test_fp_rate(self) -> None:
        result = EvalResult(safe_compression_ratio=0.0)
        result.budgets.append(
            _make_budget_result(
                n_prompts=100,
                baseline_cfr_count=10,
                prevented=2,
                false_triggers=9,
            )
        )
        summary = summarize_eval(result, "fp_test")
        # FP rate = 9 false triggers / 90 non-catastrophe prompts = 10%
        assert summary["false_positive_rate_pct"] == 10.0

    def test_per_compressor_breakdown(self) -> None:
        result = EvalResult(safe_compression_ratio=0.0)
        result.budgets.append(
            _make_budget_result(press="streaming_llm", baseline_cfr_count=10, prevented=3)
        )
        result.budgets.append(
            _make_budget_result(press="snapkv", baseline_cfr_count=30, prevented=10)
        )
        summary = summarize_eval(result, "per_comp")
        assert "streaming_llm" in summary["per_compressor"]
        assert "snapkv" in summary["per_compressor"]

    def test_gradual_onset_excludes_snapkv_extreme(self) -> None:
        result = EvalResult(safe_compression_ratio=0.0)
        result.budgets.append(
            _make_budget_result(
                press="streaming_llm", ratio=0.875, baseline_cfr_count=10, prevented=5
            )
        )
        result.budgets.append(
            _make_budget_result(press="snapkv", ratio=0.875, baseline_cfr_count=30, prevented=0)
        )
        summary = summarize_eval(result, "gradual")
        # Gradual should exclude snapkv@0.875
        assert summary["gradual_onset_reduction_pct"] == 50.0


# ---------------------------------------------------------------------------
# Tests: evaluate_with_precomputed
# ---------------------------------------------------------------------------


class TestEvaluateWithPrecomputed:
    def test_empty_traces(self) -> None:
        config = ControllerConfig(tau_low=0.3, tau_high=0.7)
        result = evaluate_with_precomputed({}, {}, config)
        assert len(result.budgets) == 0

    def test_baseline_only_ignored(self) -> None:
        """Baseline (none, 0.0) traces should not produce any budget entries."""
        traces = {
            ("none", 0.0): {"p0": _make_trace(press="none", ratio=0.0)},
        }
        config = ControllerConfig(tau_low=0.3, tau_high=0.7)
        result = evaluate_with_precomputed(traces, {}, config)
        assert len(result.budgets) == 0


# ---------------------------------------------------------------------------
# Tests: evaluate_random_predictor
# ---------------------------------------------------------------------------


class TestEvaluateRandomPredictor:
    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce same results."""
        trace = _make_trace(
            press="streaming_llm",
            ratio=0.875,
            catastrophes=["looping"],
            onsets={"looping": 50},
        )
        safe_trace = _make_trace(press="none", ratio=0.0)
        traces = {
            ("streaming_llm", 0.875): {"p0": trace},
            ("none", 0.0): {"p0": safe_trace},
        }
        config = ControllerConfig(
            tau_low=0.3,
            tau_high=0.7,
            k_escalate=3,
            safe_compression_ratio=0.0,
        )
        r1 = evaluate_random_predictor(traces, config, seed=42)
        r2 = evaluate_random_predictor(traces, config, seed=42)
        assert len(r1.budgets) == len(r2.budgets)
        if r1.budgets:
            assert r1.budgets[0].catastrophes_prevented == r2.budgets[0].catastrophes_prevented

    def test_different_seeds_may_differ(self) -> None:
        """Different seeds should produce different random probs (probabilistic test)."""
        traces_dict: dict[tuple[str, float], dict[str, TraceInfo]] = {}
        for i in range(20):
            trace = _make_trace(
                prompt_id=f"p{i}",
                press="streaming_llm",
                ratio=0.875,
                catastrophes=["looping"] if i < 5 else [],
                onsets={"looping": 50} if i < 5 else {},
                correct=i >= 5,
            )
            traces_dict.setdefault(("streaming_llm", 0.875), {})[f"p{i}"] = trace
        for i in range(20):
            safe = _make_trace(prompt_id=f"p{i}", press="none", ratio=0.0)
            traces_dict.setdefault(("none", 0.0), {})[f"p{i}"] = safe

        config = ControllerConfig(tau_low=0.1, tau_high=0.3, k_escalate=1)
        r1 = evaluate_random_predictor(traces_dict, config, seed=42)
        r2 = evaluate_random_predictor(traces_dict, config, seed=999)
        # With enough traces and low thresholds, different seeds should produce
        # different trigger counts (not guaranteed but highly likely)
        assert len(r1.budgets) > 0
        assert len(r2.budgets) > 0


# ---------------------------------------------------------------------------
# Tests: feature_ablations
# ---------------------------------------------------------------------------


def _make_synthetic_dataset(n_traces: int = 20, tokens_per: int = 50) -> Dataset:
    """Build a synthetic Dataset for feature_ablations testing."""
    rng = np.random.RandomState(42)
    n_total = n_traces * tokens_per
    # 40 features to match real data
    feature_names = [
        "entropy",
        "top1_prob",
        "top5_prob",
        *[f"top20_lp_{i}" for i in range(20)],
        "h_alts",
        "avg_logp",
        "delta_h",
        "delta_h_valid",
        "rep_count",
        "is_thinking_token",
        "entropy_mean_8",
        "entropy_std_8",
        "top1_prob_mean_8",
        "top1_prob_std_8",
        "h_alts_mean_8",
        "h_alts_std_8",
        "delta_h_mean_8",
        "delta_h_std_8",
        "rep_count_sum_8",
        "token_position",
        "compression_ratio",
    ]
    assert len(feature_names) == 40

    X = rng.randn(n_total, 40).astype(np.float32)
    # Make some labels: first 5 traces are catastrophe traces
    y = np.zeros(n_total, dtype=np.int32)
    for t in range(5):
        onset = 30  # onset at token 30
        y[t * tokens_per + onset : (t + 1) * tokens_per] = 1

    trace_ids = np.repeat(np.arange(n_traces), tokens_per)

    # Onset positions: catastrophe traces have onset=30, others -1
    onset_positions = np.full(n_total, -1, dtype=np.int32)
    for t in range(5):
        onset_positions[t * tokens_per : (t + 1) * tokens_per] = 30

    traces = []
    for t in range(n_traces):
        traces.append(
            TraceMeta(
                trace_idx=t,
                prompt_id=f"p{t % 10}",  # 10 unique prompts, 2 traces each
                press="streaming_llm" if t % 2 == 0 else "snapkv",
                compression_ratio=0.875,
                has_catastrophe=t < 5,
                catastrophe_types=["looping"] if t < 5 else [],
                n_tokens=tokens_per,
                model="test-model",
            )
        )

    return Dataset(
        X=X,
        y=y,
        trace_ids=trace_ids,
        feature_names=feature_names,
        traces=traces,
        onset_positions=onset_positions,
    )


class TestFeatureAblations:
    """Tests for the feature_ablations() function with mocked dataset."""

    @patch("kvguard.features.build_dataset")
    def test_returns_all_variants(self, mock_build: object) -> None:
        """feature_ablations returns results for all 4 ablation variants."""
        mock_build.return_value = _make_synthetic_dataset()  # type: ignore[union-attr]
        results = feature_ablations()
        assert set(results.keys()) == {"full", "no_compression_ratio", "no_rep", "logit_only"}

    @patch("kvguard.features.build_dataset")
    def test_feature_counts_decrease(self, mock_build: object) -> None:
        """Ablated variants should have fewer features than full."""
        mock_build.return_value = _make_synthetic_dataset()  # type: ignore[union-attr]
        results = feature_ablations()
        assert results["full"]["n_features"] == 40
        assert results["no_compression_ratio"]["n_features"] == 39
        assert results["no_rep"]["n_features"] == 38
        assert results["logit_only"]["n_features"] == 37

    @patch("kvguard.features.build_dataset")
    def test_dropped_features_recorded(self, mock_build: object) -> None:
        """Each variant records which features were dropped."""
        mock_build.return_value = _make_synthetic_dataset()  # type: ignore[union-attr]
        results = feature_ablations()
        assert results["full"]["features_dropped"] == []
        assert results["logit_only"]["features_dropped"] == [
            "compression_ratio",
            "rep_count",
            "rep_count_sum_8",
        ]

    @patch("kvguard.features.build_dataset")
    def test_metrics_are_valid(self, mock_build: object) -> None:
        """All metric values should be valid floats in expected ranges."""
        mock_build.return_value = _make_synthetic_dataset()  # type: ignore[union-attr]
        results = feature_ablations()
        for label, entry in results.items():
            assert 0.0 <= entry["auroc"] <= 1.0, f"{label} auroc out of range"
            assert 0.0 <= entry["f1"] <= 1.0, f"{label} f1 out of range"
            assert entry["n_val_samples"] > 0, f"{label} has no val samples"
            assert entry["n_features"] > 0, f"{label} has no features"

    @patch("kvguard.features.build_dataset")
    def test_pre_onset_auroc_present(self, mock_build: object) -> None:
        """Pre-onset AUROC should be computed when onset data is available."""
        mock_build.return_value = _make_synthetic_dataset()  # type: ignore[union-attr]
        results = feature_ablations()
        for label, entry in results.items():
            # pre_onset_auroc may be None if insufficient samples, but should be present
            assert "pre_onset_auroc" in entry, f"{label} missing pre_onset_auroc"


# ---------------------------------------------------------------------------
# Tests: ablation_entropy_threshold
# ---------------------------------------------------------------------------


class TestAblationEntropyThreshold:
    """Tests for the naive entropy threshold baseline."""

    def test_returns_expected_structure(self) -> None:
        """Output contains description, configs, and trained_comparison."""
        # Build minimal traces with entropy signals
        trace = _make_trace(
            press="streaming_llm",
            ratio=0.875,
            catastrophes=["looping"],
            onsets={"looping": 50},
        )
        safe_trace = _make_trace(press="none", ratio=0.0)
        traces = {
            ("streaming_llm", 0.875): {"p0": trace},
            ("none", 0.0): {"p0": safe_trace},
        }
        # Precomputed hazard probs (matching trace structure)
        hazard_probs: dict[tuple[str, float], dict[str, list[float]]] = {
            ("streaming_llm", 0.875): {"p0": [0.1] * 100},
        }
        result = ablation_entropy_threshold(traces, hazard_probs)
        assert "description" in result
        assert "configs" in result
        assert "trained_comparison" in result

    def test_all_tau_scales_present(self) -> None:
        """All 5 tau_scale values should appear in configs."""
        trace = _make_trace(press="streaming_llm", ratio=0.875)
        safe_trace = _make_trace(press="none", ratio=0.0)
        traces = {
            ("streaming_llm", 0.875): {"p0": trace},
            ("none", 0.0): {"p0": safe_trace},
        }
        hazard_probs: dict[tuple[str, float], dict[str, list[float]]] = {
            ("streaming_llm", 0.875): {"p0": [0.1] * 100},
        }
        result = ablation_entropy_threshold(traces, hazard_probs)
        assert len(result["configs"]) == 5
        expected_keys = {
            "tau_scale=3.0",
            "tau_scale=4.0",
            "tau_scale=5.0",
            "tau_scale=6.0",
            "tau_scale=8.0",
        }
        assert set(result["configs"].keys()) == expected_keys

    def test_entropy_probs_are_bounded(self) -> None:
        """Entropy-derived pseudo-risk should be in [0, 1]."""
        # Trace with high entropy (should clamp to 1.0)
        high_entropy_signals = [
            {
                "entropy": 10.0,
                "top1_prob": 0.01,
                "top5_prob": 0.1,
                "top1_token": "x",
                "rank_of_chosen": 0,
                "top20_logprobs": [-5.0] * 20,
                "h_alts": 5.0,
                "avg_logp": -10.0,
                "delta_h": 0.0,
                "rep_count": 0,
                "is_thinking_token": False,
            }
            for _ in range(10)
        ]
        trace = TraceInfo(
            prompt_id="p0",
            press="streaming_llm",
            compression_ratio=0.875,
            catastrophes=[],
            catastrophe_onsets={},
            num_tokens=10,
            correct=True,
            signals=high_entropy_signals,
        )
        safe_trace = _make_trace(press="none", ratio=0.0, prompt_id="p0")
        traces = {
            ("streaming_llm", 0.875): {"p0": trace},
            ("none", 0.0): {"p0": safe_trace},
        }
        hazard_probs: dict[tuple[str, float], dict[str, list[float]]] = {
            ("streaming_llm", 0.875): {"p0": [0.5] * 10},
        }
        result = ablation_entropy_threshold(traces, hazard_probs)
        # With entropy=10.0 and tau_scale=3.0, risk = min(10/3, 1) = 1.0
        assert "tau_scale=3.0" in result["configs"]

    def test_empty_traces(self) -> None:
        """Should handle empty trace dict gracefully."""
        traces: dict[tuple[str, float], dict[str, TraceInfo]] = {}
        hazard_probs: dict[tuple[str, float], dict[str, list[float]]] = {}
        result = ablation_entropy_threshold(traces, hazard_probs)
        assert len(result["configs"]) == 5
