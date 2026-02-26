"""Tests for controller evaluation module."""

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
from tests.helpers import make_result_json, make_signal_dict, write_result_file

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

        mode_history, safe_trigger = _run_predictor_state_machine(hazard_probs, config)

        assert mode_history[0] == Mode.NORMAL
        assert mode_history[1] == Mode.ALERT
        assert Mode.SAFE in mode_history
        assert safe_trigger is not None

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
        mode_history_sm, _ = _run_predictor_state_machine(hazard_probs, config)

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
        mode_history, safe_trigger = _run_predictor_state_machine([], config)
        assert mode_history == []
        assert safe_trigger is None

    def test_all_low_risk_stays_normal(self) -> None:
        """Consistently low-risk sequence stays in NORMAL."""
        config = ControllerConfig(tau_low=0.3, tau_high=0.7, k_escalate=3)
        hazard_probs = [0.1] * 20

        mode_history, safe_trigger = _run_predictor_state_machine(hazard_probs, config)

        assert all(m == Mode.NORMAL for m in mode_history)
        assert safe_trigger is None

    def test_sustained_high_risk_reaches_safe(self) -> None:
        """Sustained high risk reaches SAFE mode."""
        config = ControllerConfig(tau_low=0.3, tau_high=0.7, k_escalate=3, j_deescalate=3)
        hazard_probs = [0.9] * 20

        mode_history, safe_trigger = _run_predictor_state_machine(hazard_probs, config)

        assert Mode.SAFE in mode_history
        assert safe_trigger is not None


# ---------------------------------------------------------------------------
# Tests: step_with_risk vs _run_predictor_state_machine equivalence
# ---------------------------------------------------------------------------


def _collect_step_with_risk_history(
    hazard_probs: list[float],
    config: ControllerConfig,
) -> tuple[list[int], int | None]:
    """Run RiskController.step_with_risk and collect mode history + safe trigger."""
    ctrl = RiskController(config)
    modes: list[int] = []
    safe_trigger: int | None = None
    for t, p in enumerate(hazard_probs):
        action = ctrl.step_with_risk(p)
        modes.append(int(action.mode))
        if action.mode >= Mode.SAFE and safe_trigger is None:
            safe_trigger = t
    return modes, safe_trigger


class TestStepWithRiskEquivalence:
    """Verify _run_predictor_state_machine and RiskController.step_with_risk
    produce identical mode histories for the same hazard probability sequences.

    This is the key cross-validation for validation issue 4.3: two implementations
    of the same state machine with no test verifying equivalence.
    """

    CFG = ControllerConfig(k_escalate=3, j_deescalate=3, tau_low=0.3, tau_high=0.7)

    def _assert_equivalent(
        self, hazard_probs: list[float], config: ControllerConfig | None = None
    ) -> None:
        """Assert both implementations produce identical mode histories and safe triggers."""
        cfg = config or self.CFG
        sm_modes, sm_trigger = _run_predictor_state_machine(hazard_probs, cfg)
        ctrl_modes, ctrl_trigger = _collect_step_with_risk_history(hazard_probs, cfg)
        assert sm_modes == ctrl_modes, (
            f"Mode histories differ.\n  state_machine: {sm_modes}\n  controller:    {ctrl_modes}"
        )
        assert sm_trigger == ctrl_trigger, (
            f"Safe triggers differ: state_machine={sm_trigger}, controller={ctrl_trigger}"
        )

    def test_all_low_risk(self) -> None:
        """Both stay NORMAL for sustained low risk."""
        self._assert_equivalent([0.1] * 20)

    def test_all_high_risk(self) -> None:
        """Both escalate identically with sustained high risk."""
        self._assert_equivalent([0.9] * 20)

    def test_escalation_to_alert(self) -> None:
        """Both escalate NORMAL → ALERT after k consecutive above tau_low."""
        self._assert_equivalent([0.5] * 5 + [0.1] * 10)

    def test_full_escalation(self) -> None:
        """NORMAL → ALERT → SAFE with increasing risk."""
        self._assert_equivalent([0.5, 0.5, 0.5, 0.8, 0.8, 0.8] + [0.1] * 10)

    def test_full_round_trip(self) -> None:
        """NORMAL → ALERT → SAFE → ALERT → NORMAL."""
        probs = (
            [0.5] * 3  # → ALERT
            + [0.8] * 3  # → SAFE
            + [0.1] * 5  # → ALERT (j=3 below tau_high)
            + [0.1] * 5  # → NORMAL (j=3 below tau_low)
        )
        self._assert_equivalent(probs)

    def test_interrupted_escalation(self) -> None:
        """Low-risk token interrupts escalation, resets consecutive counter."""
        self._assert_equivalent([0.5, 0.5, 0.1, 0.5, 0.5, 0.5] + [0.1] * 5)

    def test_dead_zone_values(self) -> None:
        """Values exactly at tau_low reset both counters when in NORMAL."""
        self._assert_equivalent([0.3] * 10)

    def test_at_thresholds(self) -> None:
        """Exact threshold values: > not >= for escalation."""
        self._assert_equivalent([0.3] * 10 + [0.7] * 10)

    def test_empty_sequence(self) -> None:
        """Empty hazard prob list produces empty mode history."""
        self._assert_equivalent([])

    def test_single_token(self) -> None:
        """Single token doesn't escalate (needs k=3 consecutive)."""
        self._assert_equivalent([0.9])

    def test_alternating_risk(self) -> None:
        """Alternating high/low prevents escalation due to counter resets."""
        self._assert_equivalent([0.9, 0.1] * 20)

    def test_gradual_increase(self) -> None:
        """Gradually increasing risk crosses thresholds at same token."""
        self._assert_equivalent([i / 100.0 for i in range(100)])

    def test_deescalation_from_safe(self) -> None:
        """De-escalation from SAFE requires j consecutive below tau_high."""
        cfg = ControllerConfig(k_escalate=2, j_deescalate=3, tau_low=0.3, tau_high=0.7)
        probs = (
            [0.5] * 3  # → ALERT
            + [0.8] * 3  # → SAFE
            + [0.5] * 5  # below tau_high → de-escalate to ALERT
            + [0.1] * 5  # below tau_low → de-escalate to NORMAL
        )
        self._assert_equivalent(probs, config=cfg)

    def test_various_configs(self) -> None:
        """Equivalence holds across different config parameters."""
        probs = [0.5] * 5 + [0.9] * 10 + [0.05] * 20
        configs = [
            ControllerConfig(k_escalate=1, j_deescalate=1, tau_low=0.2, tau_high=0.5),
            ControllerConfig(k_escalate=5, j_deescalate=5, tau_low=0.4, tau_high=0.8),
            ControllerConfig(k_escalate=2, j_deescalate=10, tau_low=0.1, tau_high=0.9),
        ]
        for cfg in configs:
            self._assert_equivalent(probs, config=cfg)

    def test_long_random_sequence(self) -> None:
        """Equivalence holds on a 1000-token random sequence."""
        import random

        rng = random.Random(42)
        probs = [rng.random() for _ in range(1000)]
        self._assert_equivalent(probs)


# ---------------------------------------------------------------------------
# Tests: safe_key missing warns (Phase 2C)
# ---------------------------------------------------------------------------


def test_safe_key_missing_warns(tmp_path: Path) -> None:
    """evaluate_controller skips budget when safe_key is missing."""
    # Only write streaming_llm at ratio=0.875 (no "none" at 0.0 baseline)
    results = [
        make_result_json(
            n_tokens=20, press="streaming_llm", compression_ratio=0.875, prompt_id="p0"
        ),
        make_result_json(
            n_tokens=20, press="streaming_llm", compression_ratio=0.875, prompt_id="p1"
        ),
    ]
    write_result_file(tmp_path, "streaming_llm", 0.875, results)

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

    def test_cfr_onset_zero_tokens(self) -> None:
        """cfr_onset returns None when num_tokens is 0 (avoids negative index)."""
        trace = TraceInfo(
            prompt_id="test",
            press="snapkv",
            compression_ratio=0.75,
            catastrophes=["non_termination"],
            catastrophe_onsets={},
            num_tokens=0,
            correct=False,
            signals=[],
        )
        assert trace.cfr_onset is None


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
        signals = [make_signal_dict() for _ in range(n_tokens)]
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
        signals = [make_signal_dict() for _ in range(n_tokens)]
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

    def test_summary_line(self) -> None:
        """format_eval_table includes a summary line with overall stats."""
        result = EvalResult(
            safe_compression_ratio=0.0,
            budgets=[
                BudgetResult(
                    press="streaming_llm",
                    compression_ratio=0.875,
                    n_prompts=50,
                    baseline_cfr_count=20,
                    baseline_cfr=0.4,
                    baseline_accuracy=0.6,
                    controller_triggered_count=15,
                    triggered_before_onset=10,
                    catastrophes_prevented=8,
                    controlled_cfr_count=12,
                    controlled_cfr=0.24,
                    cfr_reduction_abs=0.16,
                    cfr_reduction_pct=40.0,
                    mean_trigger_token=50.0,
                    false_trigger_count=3,
                ),
            ],
        )
        table = format_eval_table(result)
        assert "OVERALL" in table
        assert "8/20" in table

    def test_no_trigger_token(self) -> None:
        """format_eval_table handles None mean_trigger_token."""
        result = EvalResult(
            safe_compression_ratio=0.0,
            budgets=[
                BudgetResult(
                    press="snapkv",
                    compression_ratio=0.75,
                    n_prompts=10,
                    baseline_cfr_count=0,
                    baseline_cfr=0.0,
                    baseline_accuracy=1.0,
                    controller_triggered_count=0,
                    triggered_before_onset=0,
                    catastrophes_prevented=0,
                    controlled_cfr_count=0,
                    controlled_cfr=0.0,
                    cfr_reduction_abs=0.0,
                    cfr_reduction_pct=0.0,
                    mean_trigger_token=None,
                    false_trigger_count=0,
                ),
            ],
        )
        table = format_eval_table(result)
        assert "-" in table  # None renders as "-"


# ---------------------------------------------------------------------------
# Tests: eval_result_to_dict
# ---------------------------------------------------------------------------


class TestEvalResultToDict:
    def test_serialization_round_trip(self) -> None:
        """eval_result_to_dict produces a dict matching the EvalResult structure."""
        from kvguard.evaluate_controller import eval_result_to_dict

        result = EvalResult(
            safe_compression_ratio=0.0,
            budgets=[
                BudgetResult(
                    press="snapkv",
                    compression_ratio=0.875,
                    n_prompts=100,
                    baseline_cfr_count=40,
                    baseline_cfr=0.4,
                    baseline_accuracy=0.6,
                    controller_triggered_count=30,
                    triggered_before_onset=25,
                    catastrophes_prevented=20,
                    controlled_cfr_count=20,
                    controlled_cfr=0.2,
                    cfr_reduction_abs=0.2,
                    cfr_reduction_pct=50.0,
                    mean_trigger_token=45.3,
                    false_trigger_count=5,
                ),
            ],
        )
        d = eval_result_to_dict(result)
        assert d["safe_compression_ratio"] == 0.0
        assert len(d["budgets"]) == 1
        b = d["budgets"][0]
        assert b["press"] == "snapkv"
        assert b["catastrophes_prevented"] == 20
        assert b["cfr_reduction_pct"] == 50.0
        assert b["mean_trigger_token"] == 45.3

    def test_empty_budgets(self) -> None:
        """eval_result_to_dict with no budgets."""
        from kvguard.evaluate_controller import eval_result_to_dict

        result = EvalResult(safe_compression_ratio=0.5)
        d = eval_result_to_dict(result)
        assert d["safe_compression_ratio"] == 0.5
        assert d["budgets"] == []


# ---------------------------------------------------------------------------
# Tests: evaluate_controller (core function)
# ---------------------------------------------------------------------------


class TestEvaluateController:
    """Integration tests for the evaluate_controller function."""

    def _write_traces(self, tmp_path: Path) -> None:
        """Write baseline + compressed traces for controller evaluation."""
        # Baseline (none, 0.0) — all correct, no catastrophes
        baseline_results = [
            make_result_json(n_tokens=30, correct=True, catastrophes=[], prompt_id=f"p{i}")
            for i in range(5)
        ]
        write_result_file(tmp_path, "none", 0.0, baseline_results)

        # Compressed (streaming_llm, 0.875) — mix of correct and catastrophe
        # NOTE: must set press= and compression_ratio= on each result dict
        # because load_all_traces reads them from the RunResult, not the config.
        compressed_results = [
            make_result_json(
                n_tokens=30,
                correct=True,
                catastrophes=[],
                prompt_id="p0",
                press="streaming_llm",
                compression_ratio=0.875,
            ),
            make_result_json(
                n_tokens=30,
                correct=True,
                catastrophes=[],
                prompt_id="p1",
                press="streaming_llm",
                compression_ratio=0.875,
            ),
            make_result_json(
                n_tokens=30,
                correct=False,
                catastrophes=["looping"],
                catastrophe_onsets={"looping": 20},
                prompt_id="p2",
                press="streaming_llm",
                compression_ratio=0.875,
            ),
            make_result_json(
                n_tokens=30,
                correct=False,
                catastrophes=["looping"],
                catastrophe_onsets={"looping": 20},
                prompt_id="p3",
                press="streaming_llm",
                compression_ratio=0.875,
            ),
            make_result_json(
                n_tokens=30,
                correct=False,
                catastrophes=["wrong_answer"],
                prompt_id="p4",
                predicted_answer="99",
                press="streaming_llm",
                compression_ratio=0.875,
            ),
        ]
        write_result_file(tmp_path, "streaming_llm", 0.875, compressed_results)

    def _make_predictor(self) -> xgb.XGBClassifier:
        """Create a minimal trained predictor."""
        n_features = len(feature_names())
        rng = np.random.RandomState(42)
        X = rng.randn(200, n_features).astype(np.float32)
        y = (rng.random(200) > 0.5).astype(np.float32)
        predictor = xgb.XGBClassifier(n_estimators=5, max_depth=2, verbosity=0, random_state=42)
        predictor.fit(X, y)
        return predictor

    def test_returns_eval_result(self, tmp_path: Path) -> None:
        """evaluate_controller returns EvalResult with budget entries."""
        self._write_traces(tmp_path)
        predictor = self._make_predictor()
        config = ControllerConfig(safe_compression_ratio=0.0)
        result = evaluate_controller(tmp_path, predictor, num_prompts=50, controller_config=config)
        assert isinstance(result, EvalResult)
        assert result.safe_compression_ratio == 0.0
        # Should have 1 budget: streaming_llm at 0.875
        assert len(result.budgets) == 1
        b = result.budgets[0]
        assert b.press == "streaming_llm"
        assert b.compression_ratio == 0.875

    def test_baseline_cfr_count(self, tmp_path: Path) -> None:
        """Baseline CFR count matches number of CFR catastrophe traces."""
        self._write_traces(tmp_path)
        predictor = self._make_predictor()
        config = ControllerConfig(safe_compression_ratio=0.0)
        result = evaluate_controller(tmp_path, predictor, num_prompts=50, controller_config=config)
        b = result.budgets[0]
        # 2 looping traces out of 5 total → baseline_cfr_count = 2
        # (wrong_answer doesn't count toward CFR)
        assert b.baseline_cfr_count == 2
        assert b.n_prompts == 5

    def test_holdout_filter(self, tmp_path: Path) -> None:
        """Holdout prompt IDs restricts evaluation to those prompts only."""
        self._write_traces(tmp_path)
        predictor = self._make_predictor()
        config = ControllerConfig(safe_compression_ratio=0.0)
        result = evaluate_controller(
            tmp_path,
            predictor,
            num_prompts=50,
            controller_config=config,
            holdout_prompt_ids={"p0", "p2"},
        )
        b = result.budgets[0]
        assert b.n_prompts == 2
        # Only p2 has CFR catastrophe in the holdout set
        assert b.baseline_cfr_count == 1

    def test_skips_baseline_press(self, tmp_path: Path) -> None:
        """evaluate_controller skips 'none' press (no compression to control)."""
        self._write_traces(tmp_path)
        predictor = self._make_predictor()
        config = ControllerConfig(safe_compression_ratio=0.0)
        result = evaluate_controller(tmp_path, predictor, num_prompts=50, controller_config=config)
        # Only streaming_llm budget, not "none"
        presses = [b.press for b in result.budgets]
        assert "none" not in presses
