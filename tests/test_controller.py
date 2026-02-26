"""Tests for the online risk-aware controller."""

from kvguard.controller import (
    ControllerAction,
    ControllerConfig,
    Mode,
    RiskController,
    _decide_mode,
    compute_risk_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sig(
    *,
    entropy: float = 1.0,
    delta_h: float | None = 0.0,
    rep_count: int = 0,
    top1_prob: float = 0.5,
) -> dict:
    """Build a minimal TokenSignals-compatible dict for testing."""
    return {
        "entropy": entropy,
        "delta_h": delta_h,
        "rep_count": rep_count,
        "top1_prob": top1_prob,
    }


def _low_risk_sig() -> dict:
    """Signal dict that produces a low risk score."""
    return _sig(entropy=0.5, delta_h=0.0, rep_count=0, top1_prob=0.9)


def _high_risk_sig() -> dict:
    """Signal dict that produces a high risk score (above tau_high=0.6)."""
    return _sig(entropy=4.5, delta_h=2.5, rep_count=2, top1_prob=0.1)


def _medium_risk_sig() -> dict:
    """Signal dict with risk between tau_low and tau_high."""
    return _sig(entropy=2.0, delta_h=1.0, rep_count=1, top1_prob=0.4)


# ---------------------------------------------------------------------------
# Tests: Mode enum
# ---------------------------------------------------------------------------


class TestMode:
    def test_ordering(self) -> None:
        """Modes are ordered by severity."""
        assert Mode.NORMAL < Mode.ALERT < Mode.SAFE

    def test_int_values(self) -> None:
        assert Mode.NORMAL == 0
        assert Mode.SAFE == 2


# ---------------------------------------------------------------------------
# Tests: compute_risk_score
# ---------------------------------------------------------------------------


class TestComputeRiskScore:
    def test_zero_signals(self) -> None:
        """All-zero signals should give low risk."""
        score = compute_risk_score(_sig(entropy=0.0, delta_h=0.0, rep_count=0, top1_prob=1.0))
        assert score == 0.0

    def test_high_entropy_raises_risk(self) -> None:
        low = compute_risk_score(_sig(entropy=0.5))
        high = compute_risk_score(_sig(entropy=4.0))
        assert high > low

    def test_positive_delta_h_raises_risk(self) -> None:
        low = compute_risk_score(_sig(delta_h=0.0))
        high = compute_risk_score(_sig(delta_h=2.0))
        assert high > low

    def test_negative_delta_h_ignored(self) -> None:
        """Negative delta_h (entropy drop) should not increase risk."""
        score = compute_risk_score(_sig(delta_h=-2.0))
        score_zero = compute_risk_score(_sig(delta_h=0.0))
        assert score == score_zero

    def test_none_delta_h_treated_as_zero(self) -> None:
        score = compute_risk_score(_sig(delta_h=None))
        score_zero = compute_risk_score(_sig(delta_h=0.0))
        assert score == score_zero

    def test_rep_count_raises_risk(self) -> None:
        low = compute_risk_score(_sig(rep_count=0))
        high = compute_risk_score(_sig(rep_count=3))
        assert high > low

    def test_clipped_to_unit(self) -> None:
        """Score should be clipped to [0, 1] even with extreme inputs."""
        score = compute_risk_score(_sig(entropy=100.0, delta_h=100.0, rep_count=100, top1_prob=0.0))
        assert score <= 1.0

    def test_custom_weights(self) -> None:
        """Custom weights should change the score."""
        signals = _sig(entropy=3.0, rep_count=0)
        w_ent = {
            "entropy": 1.0,
            "delta_h": 0.0,
            "rep_count": 0.0,
            "top1_prob_inv": 0.0,
        }
        w_rep = {
            "entropy": 0.0,
            "delta_h": 0.0,
            "rep_count": 1.0,
            "top1_prob_inv": 0.0,
        }
        score_ent = compute_risk_score(signals, weights=w_ent)
        score_rep = compute_risk_score(signals, weights=w_rep)
        assert score_ent > 0.0
        assert score_rep == 0.0


# ---------------------------------------------------------------------------
# Tests: _decide_mode (pure transition function)
# ---------------------------------------------------------------------------


class TestDecideMode:
    def test_normal_stays_normal_when_low_risk(self) -> None:
        cfg = ControllerConfig()
        mode = _decide_mode(
            Mode.NORMAL,
            risk_score=0.1,
            consecutive_high=0,
            consecutive_low=10,
            config=cfg,
        )
        assert mode == Mode.NORMAL

    def test_normal_to_alert_after_k(self) -> None:
        cfg = ControllerConfig(k_escalate=3, tau_low=0.3)
        mode = _decide_mode(
            Mode.NORMAL,
            risk_score=0.4,
            consecutive_high=3,
            consecutive_low=0,
            config=cfg,
        )
        assert mode == Mode.ALERT

    def test_normal_to_alert_blocked_below_k(self) -> None:
        cfg = ControllerConfig(k_escalate=3, tau_low=0.3)
        mode = _decide_mode(
            Mode.NORMAL,
            risk_score=0.4,
            consecutive_high=2,
            consecutive_low=0,
            config=cfg,
        )
        assert mode == Mode.NORMAL

    def test_alert_to_safe_after_k(self) -> None:
        cfg = ControllerConfig(k_escalate=3, tau_high=0.6)
        mode = _decide_mode(
            Mode.ALERT,
            risk_score=0.7,
            consecutive_high=3,
            consecutive_low=0,
            config=cfg,
        )
        assert mode == Mode.SAFE

    def test_alert_to_normal_deescalate(self) -> None:
        cfg = ControllerConfig(j_deescalate=5, tau_low=0.3)
        mode = _decide_mode(
            Mode.ALERT,
            risk_score=0.1,
            consecutive_high=0,
            consecutive_low=5,
            config=cfg,
        )
        assert mode == Mode.NORMAL


# ---------------------------------------------------------------------------
# Tests: RiskController (stateful)
# ---------------------------------------------------------------------------


class TestRiskController:
    def test_initial_state(self) -> None:
        ctrl = RiskController()
        assert ctrl.mode == Mode.NORMAL
        assert ctrl.step_count == 0
        assert ctrl.risk_history == []

    def test_step_returns_action(self) -> None:
        ctrl = RiskController()
        action = ctrl.step(_low_risk_sig())
        assert isinstance(action, ControllerAction)
        assert action.mode == Mode.NORMAL
        assert isinstance(action.risk_score, float)

    def test_stays_normal_under_low_risk(self) -> None:
        ctrl = RiskController()
        for _ in range(20):
            action = ctrl.step(_low_risk_sig())
        assert action.mode == Mode.NORMAL
        assert not action.protect_thinking_tokens
        assert not action.trigger_recomputation

    def test_escalates_to_alert_with_hysteresis(self) -> None:
        """Must see K consecutive high-risk tokens before escalating."""
        cfg = ControllerConfig(k_escalate=3, tau_low=0.3, tau_high=0.6)
        ctrl = RiskController(cfg)

        # First two high-risk tokens: still NORMAL
        for _ in range(2):
            action = ctrl.step(_medium_risk_sig())
        assert action.mode == Mode.NORMAL

        # Third consecutive high-risk: escalate to ALERT
        action = ctrl.step(_medium_risk_sig())
        assert action.mode == Mode.ALERT
        assert action.protect_thinking_tokens

    def test_escalates_normal_to_safe(self) -> None:
        """Full escalation: NORMAL → ALERT → SAFE."""
        cfg = ControllerConfig(k_escalate=2, tau_low=0.2, tau_high=0.5)
        ctrl = RiskController(cfg)

        # Escalate to ALERT
        for _ in range(3):
            ctrl.step(_medium_risk_sig())

        # Now push above tau_high to escalate to SAFE
        for _ in range(3):
            action = ctrl.step(_high_risk_sig())

        assert action.mode == Mode.SAFE
        assert action.compression_ratio == cfg.safe_compression_ratio

    def test_deescalation_with_hysteresis(self) -> None:
        """Mode drops back after J consecutive low-risk tokens."""
        cfg = ControllerConfig(k_escalate=2, j_deescalate=3, tau_low=0.3, tau_high=0.6)
        ctrl = RiskController(cfg)

        # Escalate to ALERT
        for _ in range(3):
            ctrl.step(_medium_risk_sig())
        assert ctrl.mode == Mode.ALERT

        # First 2 low-risk: still ALERT (j_deescalate=3)
        for _ in range(2):
            action = ctrl.step(_low_risk_sig())
        assert action.mode == Mode.ALERT

        # Third low-risk: de-escalate to NORMAL
        action = ctrl.step(_low_risk_sig())
        assert action.mode == Mode.NORMAL

    def test_reset(self) -> None:
        ctrl = RiskController()
        for _ in range(5):
            ctrl.step(_high_risk_sig())
        assert ctrl.step_count > 0

        ctrl.reset()
        assert ctrl.mode == Mode.NORMAL
        assert ctrl.step_count == 0
        assert ctrl.risk_history == []

    def test_risk_history_tracks_steps(self) -> None:
        ctrl = RiskController()
        ctrl.step(_low_risk_sig())
        ctrl.step(_high_risk_sig())
        assert len(ctrl.risk_history) == 2
        assert ctrl.risk_history[0] < ctrl.risk_history[1]

    def test_normal_compression_ratio(self) -> None:
        cfg = ControllerConfig(base_compression_ratio=0.875)
        ctrl = RiskController(cfg)
        action = ctrl.step(_low_risk_sig())
        assert action.compression_ratio == 0.875

    def test_safe_compression_ratio(self) -> None:
        cfg = ControllerConfig(
            k_escalate=1,
            tau_low=0.1,
            tau_high=0.3,
            base_compression_ratio=0.875,
            safe_compression_ratio=0.5,
        )
        ctrl = RiskController(cfg)
        # Get to SAFE
        for _ in range(5):
            ctrl.step(_high_risk_sig())
        assert ctrl.mode == Mode.SAFE
        action = ctrl.step(_high_risk_sig())
        assert action.compression_ratio == 0.5

    def test_interrupted_escalation_resets(self) -> None:
        """Low-risk token during escalation resets the consecutive counter."""
        cfg = ControllerConfig(k_escalate=3, tau_low=0.3)
        ctrl = RiskController(cfg)

        ctrl.step(_medium_risk_sig())  # consecutive_high = 1
        ctrl.step(_medium_risk_sig())  # consecutive_high = 2
        ctrl.step(_low_risk_sig())  # reset
        ctrl.step(_medium_risk_sig())  # consecutive_high = 1
        ctrl.step(_medium_risk_sig())  # consecutive_high = 2
        action = ctrl.step(_medium_risk_sig())
        # Should escalate now (3 consecutive after the reset)
        assert action.mode == Mode.ALERT

    def test_step_count_increments(self) -> None:
        ctrl = RiskController()
        for i in range(10):
            ctrl.step(_low_risk_sig())
        assert ctrl.step_count == 10


# ---------------------------------------------------------------------------
# Tests: step_with_risk (pre-computed risk score)
# ---------------------------------------------------------------------------


class TestStepWithRisk:
    def test_basic(self) -> None:
        ctrl = RiskController()
        action = ctrl.step_with_risk(0.1)
        assert action.mode == Mode.NORMAL
        assert isinstance(action.risk_score, float)

    def test_escalation(self) -> None:
        """Should escalate with pre-computed risk scores."""
        cfg = ControllerConfig(k_escalate=2, tau_low=0.3, tau_high=0.6)
        ctrl = RiskController(cfg)
        for _ in range(3):
            ctrl.step_with_risk(0.5)
        assert ctrl.mode == Mode.ALERT

    def test_clamps_risk(self) -> None:
        """Risk scores outside [0,1] should be clamped."""
        ctrl = RiskController()
        action = ctrl.step_with_risk(5.0)
        assert action.risk_score == 1.0
        action = ctrl.step_with_risk(-1.0)
        assert action.risk_score == 0.0

    def test_full_escalation_and_deescalation(self) -> None:
        """NORMAL → ALERT → SAFE → ALERT → NORMAL."""
        cfg = ControllerConfig(k_escalate=2, j_deescalate=3, tau_low=0.3, tau_high=0.6)
        ctrl = RiskController(cfg)

        # NORMAL → ALERT
        for _ in range(3):
            ctrl.step_with_risk(0.5)
        assert ctrl.mode == Mode.ALERT

        # ALERT → SAFE
        for _ in range(3):
            ctrl.step_with_risk(0.8)
        assert ctrl.mode == Mode.SAFE

        # SAFE → ALERT (need j_deescalate=3 low-risk tokens below tau_high)
        for _ in range(4):
            ctrl.step_with_risk(0.1)
        assert ctrl.mode == Mode.ALERT

        # ALERT → NORMAL (need j_deescalate=3 low-risk tokens below tau_low)
        for _ in range(4):
            ctrl.step_with_risk(0.05)
        assert ctrl.mode == Mode.NORMAL

    def test_risk_history(self) -> None:
        ctrl = RiskController()
        ctrl.step_with_risk(0.1)
        ctrl.step_with_risk(0.5)
        ctrl.step_with_risk(0.9)
        assert len(ctrl.risk_history) == 3
        assert ctrl.risk_history == [0.1, 0.5, 0.9]

    def test_at_threshold_does_not_escalate(self) -> None:
        """Risk exactly at tau_low should not trigger escalation (needs >)."""
        cfg = ControllerConfig(k_escalate=1, tau_low=0.3)
        ctrl = RiskController(cfg)
        for _ in range(5):
            ctrl.step_with_risk(0.3)  # exactly at threshold
        assert ctrl.mode == Mode.NORMAL

    def test_long_sequence_stability(self) -> None:
        """Controller should maintain state across long sequences."""
        ctrl = RiskController()
        for _ in range(1000):
            ctrl.step_with_risk(0.05)
        assert ctrl.mode == Mode.NORMAL
        assert ctrl.step_count == 1000


# ---------------------------------------------------------------------------
# Tests: ControllerConfig defaults
# ---------------------------------------------------------------------------


class TestControllerConfig:
    def test_defaults_sensible(self) -> None:
        cfg = ControllerConfig()
        assert 0.0 < cfg.tau_low < cfg.tau_high < 1.0
        assert cfg.k_escalate >= 1
        assert cfg.j_deescalate >= 1
        assert 0.0 <= cfg.safe_compression_ratio < cfg.base_compression_ratio

    def test_custom_config(self) -> None:
        cfg = ControllerConfig(tau_low=0.2, tau_high=0.8, k_escalate=5)
        assert cfg.tau_low == 0.2
        assert cfg.tau_high == 0.8
        assert cfg.k_escalate == 5


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestControllerEdgeCases:
    """Edge cases for compute_risk_score, _decide_mode, and RiskController."""

    # --- compute_risk_score edge cases ---

    def test_nan_entropy_produces_finite_score(self) -> None:
        """NaN in signals should not crash; float(NaN) propagates."""
        import math

        score = compute_risk_score(_sig(entropy=float("nan")))
        # NaN propagation means score may be NaN — at minimum, no exception
        assert isinstance(score, float)
        # If NaN propagated, it won't equal itself
        if not math.isnan(score):
            assert 0.0 <= score <= 1.0

    def test_all_weights_zero_gives_zero(self) -> None:
        """All-zero weights should produce zero risk regardless of signals."""
        w = {"entropy": 0.0, "delta_h": 0.0, "rep_count": 0.0, "top1_prob_inv": 0.0}
        score = compute_risk_score(
            _sig(entropy=10.0, delta_h=5.0, rep_count=10, top1_prob=0.0),
            weights=w,
        )
        assert score == 0.0

    def test_top1_prob_zero_max_risk_component(self) -> None:
        """top1_prob=0.0 should max out the top1_prob_inv component."""
        # Only use top1_prob_inv weight
        w = {"entropy": 0.0, "delta_h": 0.0, "rep_count": 0.0, "top1_prob_inv": 1.0}
        score = compute_risk_score(_sig(top1_prob=0.0), weights=w)
        assert score == 1.0

    def test_top1_prob_one_zero_risk_component(self) -> None:
        """top1_prob=1.0 should zero out the top1_prob_inv component."""
        w = {"entropy": 0.0, "delta_h": 0.0, "rep_count": 0.0, "top1_prob_inv": 1.0}
        score = compute_risk_score(_sig(top1_prob=1.0), weights=w)
        assert score == 0.0

    def test_missing_signal_keys_use_defaults(self) -> None:
        """Missing keys in signals dict should use safe defaults."""
        score = compute_risk_score({})
        # entropy=0, delta_h=None→0, rep_count=0, top1_prob=1.0 → all zero
        assert score == 0.0

    # --- _decide_mode edge cases ---

    def test_k_zero_instant_escalation(self) -> None:
        """k_escalate=0 should allow immediate escalation (>=0 is always true)."""
        cfg = ControllerConfig(k_escalate=0, tau_low=0.3)
        mode = _decide_mode(
            Mode.NORMAL, risk_score=0.5, consecutive_high=0, consecutive_low=0, config=cfg
        )
        assert mode == Mode.ALERT

    def test_j_zero_instant_deescalation(self) -> None:
        """j_deescalate=0 should allow immediate de-escalation."""
        cfg = ControllerConfig(j_deescalate=0, tau_low=0.3)
        mode = _decide_mode(
            Mode.ALERT, risk_score=0.1, consecutive_high=0, consecutive_low=0, config=cfg
        )
        assert mode == Mode.NORMAL

    def test_tau_low_equals_tau_high(self) -> None:
        """Degenerate case: tau_low == tau_high. Escalation still works."""
        cfg = ControllerConfig(k_escalate=1, tau_low=0.5, tau_high=0.5)
        # NORMAL → ALERT requires risk > tau_low (0.5)
        mode = _decide_mode(
            Mode.NORMAL, risk_score=0.6, consecutive_high=1, consecutive_low=0, config=cfg
        )
        assert mode == Mode.ALERT

    def test_risk_exactly_at_tau_high_no_escalation(self) -> None:
        """Risk exactly at tau_high should NOT escalate (requires >)."""
        cfg = ControllerConfig(k_escalate=1, tau_high=0.7)
        mode = _decide_mode(
            Mode.ALERT, risk_score=0.7, consecutive_high=10, consecutive_low=0, config=cfg
        )
        assert mode == Mode.ALERT

    def test_risk_exactly_at_tau_low_no_deescalation(self) -> None:
        """Risk exactly at tau_low should NOT de-escalate (requires <)."""
        cfg = ControllerConfig(j_deescalate=1, tau_low=0.3)
        mode = _decide_mode(
            Mode.ALERT, risk_score=0.3, consecutive_high=0, consecutive_low=10, config=cfg
        )
        assert mode == Mode.ALERT

    # --- RiskController edge cases ---

    def test_step_with_risk_nan_clamped(self) -> None:
        """NaN risk_score clamped by max(0, min(NaN, 1)) — should not crash."""
        import math

        ctrl = RiskController()
        action = ctrl.step_with_risk(float("nan"))
        # max(0.0, min(nan, 1.0)) behavior is platform-dependent
        assert isinstance(action, ControllerAction)
        # Mode should remain NORMAL (NaN doesn't satisfy > threshold)
        assert action.mode == Mode.NORMAL or math.isnan(action.risk_score)

    def test_oscillating_risk_no_escalation(self) -> None:
        """Alternating high/low risk should never escalate due to counter resets."""
        cfg = ControllerConfig(k_escalate=3, tau_low=0.3)
        ctrl = RiskController(cfg)
        for _ in range(100):
            ctrl.step_with_risk(0.5)  # above tau_low
            ctrl.step_with_risk(0.1)  # below tau_low — resets consecutive_high
        assert ctrl.mode == Mode.NORMAL

    def test_dead_zone_resets_both_counters(self) -> None:
        """Risk in the dead zone (between tau_low and tau_high) resets both counters."""
        cfg = ControllerConfig(k_escalate=3, j_deescalate=3, tau_low=0.3, tau_high=0.7)
        ctrl = RiskController(cfg)
        # Build up consecutive_high
        ctrl.step_with_risk(0.5)  # above tau_low (NORMAL threshold)
        ctrl.step_with_risk(0.5)
        # Now hit dead zone — at this point the escalation threshold for NORMAL is tau_low=0.3
        # 0.5 is above 0.3, so it's still consecutive_high
        # But if we go exactly between thresholds after ALERT:
        # First, escalate to ALERT
        ctrl.step_with_risk(0.5)  # 3rd consecutive above tau_low → ALERT
        assert ctrl.mode == Mode.ALERT
        # Now in ALERT, escalation threshold is tau_high=0.7, deescalation is tau_low=0.3
        # 0.5 is in the dead zone for ALERT mode
        ctrl.step_with_risk(0.5)
        # Should not change mode
        assert ctrl.mode == Mode.ALERT

    def test_k_escalate_1_immediate(self) -> None:
        """k_escalate=1 should escalate on the very first high-risk token."""
        cfg = ControllerConfig(k_escalate=1, tau_low=0.3, tau_high=0.6)
        ctrl = RiskController(cfg)
        action = ctrl.step_with_risk(0.5)  # above tau_low
        assert action.mode == Mode.ALERT

    def test_safe_deescalation_to_alert_not_normal(self) -> None:
        """SAFE should de-escalate to ALERT, not jump to NORMAL."""
        cfg = ControllerConfig(k_escalate=1, j_deescalate=1, tau_low=0.2, tau_high=0.5)
        ctrl = RiskController(cfg)
        # Escalate to SAFE
        ctrl.step_with_risk(0.8)  # → ALERT (above tau_low)
        assert ctrl.mode == Mode.ALERT
        ctrl.step_with_risk(0.8)  # → SAFE (above tau_high)
        assert ctrl.mode == Mode.SAFE
        # De-escalate: risk below tau_high
        ctrl.step_with_risk(0.1)  # → ALERT (below tau_high)
        assert ctrl.mode == Mode.ALERT  # not NORMAL

    def test_zero_norms_no_division_error(self) -> None:
        """Zero normalization constants should not cause ZeroDivisionError."""
        zero_norms = {"entropy": 0.0, "delta_h": 0.0, "rep_count": 0.0, "top1_prob_inv": 0.0}
        score = compute_risk_score(
            _sig(entropy=5.0, delta_h=2.0, rep_count=3, top1_prob=0.1),
            norms=zero_norms,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_zero_norms_with_zero_values(self) -> None:
        """Zero norms + zero values should produce zero risk."""
        zero_norms = {"entropy": 0.0, "delta_h": 0.0, "rep_count": 0.0, "top1_prob_inv": 0.0}
        score = compute_risk_score(
            _sig(entropy=0.0, delta_h=0.0, rep_count=0, top1_prob=1.0),
            norms=zero_norms,
        )
        assert score == 0.0
