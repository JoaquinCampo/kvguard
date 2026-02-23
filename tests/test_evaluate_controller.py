"""Tests for controller evaluation module."""

import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from kvguard.controller import ControllerConfig, Mode, RiskController, compute_risk_score
from kvguard.evaluate_controller import (
    BudgetResult,
    EvalResult,
    TraceInfo,
    _run_predictor_state_machine,
    evaluate_controller,
    filter_traces_by_prompts,
    format_eval_table,
    simulate_controller_on_trace,
)
from kvguard.features import feature_names

# ---------------------------------------------------------------------------
# Tests: filter_traces_by_prompts
# ---------------------------------------------------------------------------


def test_filter_traces_by_prompts() -> None:
    """Only traces whose prompt_id is in the allowed set are returned."""
    all_traces = {
        ("snapkv", 0.75): {
            "gsm8k_0": "trace_a",
            "gsm8k_1": "trace_b",
            "gsm8k_2": "trace_c",
        }
    }
    allowed = {"gsm8k_0", "gsm8k_2"}
    filtered = filter_traces_by_prompts(all_traces, allowed)
    assert set(filtered[("snapkv", 0.75)].keys()) == {"gsm8k_0", "gsm8k_2"}


def test_filter_preserves_structure() -> None:
    all_traces = {
        ("snapkv", 0.5): {"gsm8k_0": "t1", "gsm8k_1": "t2"},
        ("snapkv", 0.75): {"gsm8k_0": "t3", "gsm8k_1": "t4"},
    }
    allowed = {"gsm8k_1"}
    filtered = filter_traces_by_prompts(all_traces, allowed)
    assert len(filtered) == 2
    for key in filtered:
        assert list(filtered[key].keys()) == ["gsm8k_1"]


# ---------------------------------------------------------------------------
# Tests: cross-validation of duplicate state machines (T13)
# ---------------------------------------------------------------------------


class TestStateMachinesAgree:
    """Verify _run_predictor_state_machine and RiskController produce same mode history.

    Both implement the same hysteresis-based state machine. The standalone
    version uses hazard_prob directly as the risk score, while RiskController
    computes risk from raw signals via compute_risk_score(). This means they
    can only be compared when using equivalent risk inputs.
    """

    def test_standalone_transitions_correctly(self) -> None:
        """_run_predictor_state_machine escalates and de-escalates as expected."""
        config = ControllerConfig(tau_low=0.3, tau_high=0.7, k_escalate=2, j_deescalate=2)
        hazard_probs = [
            0.5,
            0.5,  # 2 above tau_low → ALERT
            0.8,
            0.8,  # 2 above tau_high → SAFE
            0.1,
            0.1,  # 2 below tau_high → ALERT
            0.1,
            0.1,  # 2 below tau_low → NORMAL
        ]

        mode_history, safe_trigger, recovery_trigger = _run_predictor_state_machine(
            hazard_probs, config
        )

        assert mode_history[0] == Mode.NORMAL
        assert mode_history[1] == Mode.ALERT
        assert Mode.SAFE in mode_history
        assert safe_trigger is not None
        assert recovery_trigger is None  # RECOVERY removed

    def test_risk_controller_transitions_match_standalone(self) -> None:
        """RiskController mode transitions match standalone when fed same risk scores.

        We feed RiskController signals that produce the exact same risk scores
        as the hazard_probs fed to _run_predictor_state_machine, verifying that
        the two state machines agree on mode transitions.
        """
        config = ControllerConfig(tau_low=0.3, tau_high=0.7, k_escalate=3, j_deescalate=3)
        # Use risk scores that are easy to reproduce through compute_risk_score
        # Risk = 0.30*(entropy/5) + 0.25*(delta_h/3) + 0.30*(rep/3) + 0.15*(1-top1)
        # For simplicity, only use entropy: risk ≈ 0.30 * entropy/5
        # entropy=0.5 → risk=0.03, entropy=3.0 → risk=0.18, entropy=8.0 → risk=0.30*1.0=0.30
        # We need a richer signal mix to reach above 0.7

        # Strategy: compute risk scores from signals, then use those as hazard_probs
        signal_sequences = [
            {"entropy": 0.5, "delta_h": 0.0, "rep_count": 0, "top1_prob": 0.9},  # low
            {"entropy": 0.5, "delta_h": 0.0, "rep_count": 0, "top1_prob": 0.9},  # low
            {"entropy": 3.0, "delta_h": 1.5, "rep_count": 1, "top1_prob": 0.4},  # med
            {"entropy": 3.0, "delta_h": 1.5, "rep_count": 1, "top1_prob": 0.4},  # med
            {"entropy": 3.0, "delta_h": 1.5, "rep_count": 1, "top1_prob": 0.4},  # med
            {"entropy": 5.0, "delta_h": 3.0, "rep_count": 3, "top1_prob": 0.1},  # high
            {"entropy": 5.0, "delta_h": 3.0, "rep_count": 3, "top1_prob": 0.1},  # high
            {"entropy": 5.0, "delta_h": 3.0, "rep_count": 3, "top1_prob": 0.1},  # high
            {"entropy": 0.5, "delta_h": 0.0, "rep_count": 0, "top1_prob": 0.9},  # low
            {"entropy": 0.5, "delta_h": 0.0, "rep_count": 0, "top1_prob": 0.9},  # low
            {"entropy": 0.5, "delta_h": 0.0, "rep_count": 0, "top1_prob": 0.9},  # low
            {"entropy": 0.5, "delta_h": 0.0, "rep_count": 0, "top1_prob": 0.9},  # low
        ]

        # Compute exact risk scores
        hazard_probs = [
            compute_risk_score(s, weights=config.weights, norms=config.norms)
            for s in signal_sequences
        ]

        # Run standalone
        mode_history_sm, _, _ = _run_predictor_state_machine(hazard_probs, config)

        # Run through RiskController
        ctrl = RiskController(config)
        mode_history_ctrl = []
        for s in signal_sequences:
            action = ctrl.step(s)
            mode_history_ctrl.append(action.mode.value)

        assert mode_history_sm == mode_history_ctrl, (
            f"State machines diverge!\n"
            f"  standalone: {mode_history_sm}\n"
            f"  controller: {mode_history_ctrl}"
        )

    def test_empty_sequence(self) -> None:
        """Empty hazard_probs produces empty history."""
        config = ControllerConfig()
        mode_history, safe_trigger, recovery_trigger = _run_predictor_state_machine([], config)
        assert mode_history == []
        assert safe_trigger is None
        assert recovery_trigger is None

    def test_all_low_risk_stays_normal(self) -> None:
        """Consistently low-risk sequence stays in NORMAL."""
        config = ControllerConfig(tau_low=0.3, tau_high=0.7, k_escalate=3)
        hazard_probs = [0.1] * 20

        mode_history, safe_trigger, _ = _run_predictor_state_machine(hazard_probs, config)

        assert all(m == Mode.NORMAL for m in mode_history)
        assert safe_trigger is None

    def test_sustained_high_risk_reaches_safe(self) -> None:
        """Sustained high risk reaches SAFE mode."""
        config = ControllerConfig(tau_low=0.3, tau_high=0.7, k_escalate=3, j_deescalate=3)
        hazard_probs = [0.9] * 20

        mode_history, safe_trigger, _ = _run_predictor_state_machine(hazard_probs, config)

        assert Mode.SAFE in mode_history
        assert safe_trigger is not None


# ---------------------------------------------------------------------------
# Tests: safe_key missing warns (Phase 2C)
# ---------------------------------------------------------------------------


def _make_signal_dict(
    *,
    entropy: float = 1.0,
    top1_prob: float = 0.5,
    top5_prob: float = 0.9,
) -> dict:
    return {
        "entropy": entropy,
        "top1_prob": top1_prob,
        "top5_prob": top5_prob,
        "top1_token": "a",
        "rank_of_chosen": 0,
        "top20_logprobs": [-0.5] * 20,
        "h_alts": 0.3,
        "avg_logp": -2.0,
        "delta_h": 0.1,
        "rep_count": 0,
        "is_thinking_token": False,
    }


def _make_result_json(
    *,
    n_tokens: int = 10,
    press: str = "streaming_llm",
    compression_ratio: float = 0.875,
    prompt_id: str = "gsm8k_0",
) -> dict:
    sigs = [_make_signal_dict() for _ in range(n_tokens)]
    return {
        "prompt_id": prompt_id,
        "prompt_text": "test",
        "model": "test-model",
        "press": press,
        "compression_ratio": compression_ratio,
        "max_new_tokens": 512,
        "seed": 42,
        "generated_text": "out",
        "ground_truth": "42",
        "predicted_answer": "42",
        "correct": True,
        "stop_reason": "eos",
        "catastrophes": [],
        "num_tokens_generated": n_tokens,
        "cache_size_after_prefill": None,
        "catastrophe_onsets": {},
        "signals": sigs,
    }


def _write_result_file(
    tmpdir: Path,
    press: str,
    ratio: float,
    results: list[dict],
    n_prompts: int = 50,
) -> Path:
    subdir = tmpdir / press
    subdir.mkdir(parents=True, exist_ok=True)
    fname = f"test-model_{ratio:.3f}_{n_prompts}p.json"
    path = subdir / fname
    data = {
        "config": {
            "model_name": "test-model",
            "press_name": press,
            "compression_ratio": ratio,
        },
        "summary": {},
        "results": results,
    }
    path.write_text(json.dumps(data))
    return path


def test_safe_key_missing_warns(tmp_path: Path) -> None:
    """evaluate_controller skips budget when safe_key is missing."""
    # Only write streaming_llm at ratio=0.875 (no "none" at 0.0 baseline)
    results = [
        _make_result_json(n_tokens=20, prompt_id="p0"),
        _make_result_json(n_tokens=20, prompt_id="p1"),
    ]
    _write_result_file(tmp_path, "streaming_llm", 0.875, results)

    # Train a minimal predictor
    n_features = len(feature_names())
    rng = np.random.RandomState(42)
    X = rng.randn(100, n_features).astype(np.float32)
    y = (rng.random(100) > 0.5).astype(np.float32)
    predictor = xgb.XGBClassifier(n_estimators=5, max_depth=2, verbosity=0, random_state=42)
    predictor.fit(X, y)

    # safe_compression_ratio=0.0 means safe_key=("none", 0.0) which is missing
    config = ControllerConfig(safe_compression_ratio=0.0)
    result = evaluate_controller(
        tmp_path,
        predictor,
        num_prompts=50,
        controller_config=config,
    )

    assert isinstance(result, EvalResult)
    # No budgets produced since safe_key is missing for the only (press, ratio)
    assert len(result.budgets) == 0


# ---------------------------------------------------------------------------
# Tests: cfr_onset uses proxy for non_termination (Phase 1B)
# ---------------------------------------------------------------------------


class TestCfrOnsetProxy:
    def test_cfr_onset_uses_proxy_for_non_termination(self) -> None:
        """cfr_onset uses nt_onset_frac * max_new_tokens for non_termination,
        not the raw catastrophe_onsets value."""
        trace = TraceInfo(
            prompt_id="test",
            press="snapkv",
            compression_ratio=0.75,
            catastrophes=["non_termination"],
            # The raw onset for non_termination would be the last token (511),
            # but cfr_onset should use the proxy formula instead.
            catastrophe_onsets={"non_termination": 511},
            num_tokens=512,
            correct=None,
            signals=[],
            max_new_tokens=512,
            nt_onset_frac=0.75,
        )
        # proxy = min(int(0.75 * 512), 511) = min(384, 511) = 384
        assert trace.cfr_onset == 384

    def test_cfr_onset_looping_unchanged(self) -> None:
        """Looping onset is still taken directly from catastrophe_onsets."""
        trace = TraceInfo(
            prompt_id="test",
            press="snapkv",
            compression_ratio=0.75,
            catastrophes=["looping"],
            catastrophe_onsets={"looping": 100},
            num_tokens=512,
            correct=None,
            signals=[],
            max_new_tokens=512,
            nt_onset_frac=0.75,
        )
        assert trace.cfr_onset == 100

    def test_cfr_onset_both_uses_min(self) -> None:
        """With both catastrophes, uses min of looping onset and proxy."""
        trace = TraceInfo(
            prompt_id="test",
            press="snapkv",
            compression_ratio=0.75,
            catastrophes=["looping", "non_termination"],
            catastrophe_onsets={"looping": 50, "non_termination": 511},
            num_tokens=512,
            correct=None,
            signals=[],
            max_new_tokens=512,
            nt_onset_frac=0.75,
        )
        # looping=50, nt_proxy=384 → min=50
        assert trace.cfr_onset == 50

    def test_cfr_onset_proxy_clamped_to_sequence(self) -> None:
        """Proxy onset is clamped to num_tokens - 1."""
        trace = TraceInfo(
            prompt_id="test",
            press="snapkv",
            compression_ratio=0.75,
            catastrophes=["non_termination"],
            catastrophe_onsets={},
            num_tokens=200,
            correct=None,
            signals=[],
            max_new_tokens=512,
            nt_onset_frac=0.75,
        )
        # proxy = min(int(0.75 * 512), 199) = min(384, 199) = 199
        assert trace.cfr_onset == 199


# ---------------------------------------------------------------------------
# Tests: TraceInfo properties (has_cfr_catastrophe, cfr_onset)
# ---------------------------------------------------------------------------


class TestTraceInfo:
    """Tests for TraceInfo properties."""

    def test_has_cfr_catastrophe_looping(self) -> None:
        """looping counts as CFR catastrophe."""
        trace = TraceInfo(
            prompt_id="test",
            press="snapkv",
            compression_ratio=0.75,
            catastrophes=["looping"],
            catastrophe_onsets={"looping": 50},
            num_tokens=100,
            correct=False,
            signals=[],
        )
        assert trace.has_cfr_catastrophe is True

    def test_has_cfr_catastrophe_wrong_answer(self) -> None:
        """wrong_answer does NOT count as CFR catastrophe."""
        trace = TraceInfo(
            prompt_id="test",
            press="snapkv",
            compression_ratio=0.75,
            catastrophes=["wrong_answer"],
            catastrophe_onsets={},
            num_tokens=100,
            correct=False,
            signals=[],
        )
        assert trace.has_cfr_catastrophe is False

    def test_cfr_onset_earliest(self) -> None:
        """cfr_onset returns the earliest onset across CFR catastrophe types."""
        trace = TraceInfo(
            prompt_id="test",
            press="snapkv",
            compression_ratio=0.75,
            catastrophes=["looping", "non_termination"],
            catastrophe_onsets={"looping": 50},
            num_tokens=512,
            correct=False,
            signals=[],
            max_new_tokens=512,
            nt_onset_frac=0.75,
        )
        # looping onset = 50, non_termination proxy = int(0.75*512)=384
        # clamped to min(384, 511) = 384; looping is earlier
        assert trace.cfr_onset == 50


# ---------------------------------------------------------------------------
# Tests: simulate_controller_on_trace
# ---------------------------------------------------------------------------


class _MockPredictor:
    """Mock predictor that returns constant probabilities."""

    def __init__(self, prob: float = 0.0) -> None:
        self.prob = prob

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        return np.column_stack([np.full(n, 1 - self.prob), np.full(n, self.prob)])


class TestSimulateControllerOnTrace:
    """Tests for simulate_controller_on_trace."""

    def test_no_trigger_on_clean_trace(self) -> None:
        """Controller should not trigger SAFE on a non-catastrophe trace with low risk."""
        n_tokens = 30
        signals = [_make_signal_dict() for _ in range(n_tokens)]
        trace = TraceInfo(
            prompt_id="clean",
            press="snapkv",
            compression_ratio=0.75,
            catastrophes=[],
            catastrophe_onsets={},
            num_tokens=n_tokens,
            correct=True,
            signals=signals,
        )
        predictor = _MockPredictor(prob=0.0)
        config = ControllerConfig(tau_low=0.3, tau_high=0.7, k_escalate=3, j_deescalate=3)
        ct = simulate_controller_on_trace(trace, predictor, config)
        assert ct.controller_triggered_safe is False
        assert ct.safe_trigger_token is None

    def test_trigger_before_onset_gives_lead_time(self) -> None:
        """When controller triggers before cfr_onset, lead_time should be positive."""
        n_tokens = 30
        signals = [_make_signal_dict() for _ in range(n_tokens)]
        trace = TraceInfo(
            prompt_id="looping",
            press="snapkv",
            compression_ratio=0.75,
            catastrophes=["looping"],
            catastrophe_onsets={"looping": 25},
            num_tokens=n_tokens,
            correct=False,
            signals=signals,
        )
        # High probability predictor should trigger SAFE quickly
        predictor = _MockPredictor(prob=0.95)
        config = ControllerConfig(tau_low=0.3, tau_high=0.7, k_escalate=2, j_deescalate=2)
        ct = simulate_controller_on_trace(trace, predictor, config)
        assert ct.controller_triggered_safe is True
        assert ct.safe_trigger_token is not None
        assert ct.safe_trigger_token < 25
        assert ct.lead_time is not None
        assert ct.lead_time > 0

    def test_empty_trace(self) -> None:
        """Empty trace returns empty results without crash."""
        trace = TraceInfo(
            prompt_id="empty",
            press="snapkv",
            compression_ratio=0.75,
            catastrophes=[],
            catastrophe_onsets={},
            num_tokens=0,
            correct=None,
            signals=[],
        )
        predictor = _MockPredictor(prob=0.5)
        config = ControllerConfig()
        ct = simulate_controller_on_trace(trace, predictor, config)
        assert ct.mode_history == []
        assert ct.hazard_probs == []
        assert ct.controller_triggered_safe is False
        assert ct.lead_time is None


# ---------------------------------------------------------------------------
# Tests: format_eval_table
# ---------------------------------------------------------------------------


class TestFormatEvalTable:
    def test_produces_non_empty_string(self) -> None:
        """format_eval_table should return a non-empty string."""
        result = EvalResult(
            safe_compression_ratio=0.0,
            budgets=[
                BudgetResult(
                    press="snapkv",
                    compression_ratio=0.75,
                    n_prompts=10,
                    baseline_cfr_count=3,
                    baseline_cfr=0.3,
                    baseline_accuracy=0.7,
                    controller_triggered_count=5,
                    triggered_before_onset=2,
                    catastrophes_prevented=2,
                    controlled_cfr_count=1,
                    controlled_cfr=0.1,
                    cfr_reduction_abs=0.2,
                    cfr_reduction_pct=66.7,
                    mean_trigger_token=100.0,
                    false_trigger_count=1,
                ),
            ],
        )
        table = format_eval_table(result)
        assert len(table) > 0
        assert "snapkv" in table
