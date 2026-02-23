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
