"""Tests for controller evaluation module."""

from kvguard.controller import ControllerConfig, Mode, RiskController, compute_risk_score
from kvguard.evaluate_controller import _run_predictor_state_machine, filter_traces_by_prompts

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
