"""Online risk-aware controller for KV-cache compression.

Inspired by TCP congestion control: monitor generation health via
near-zero-cost signals and dynamically adjust compression aggressiveness
to prevent catastrophic failures before they become irreversible.

The controller wraps any KV-cache compressor and operates as a state
machine with three modes:

    NORMAL  → aggressive compression (default press settings)
    ALERT   → protect thinking-token KV entries from eviction
    SAFE    → relax compression ratio

Mode transitions use hysteresis (require K consecutive high-risk or
J consecutive low-risk tokens) to prevent flapping on noisy signals.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

# ---------------------------------------------------------------------------
# Mode enum (IntEnum for easy comparison / ordering)
# ---------------------------------------------------------------------------


class Mode(IntEnum):
    """Controller operating modes, ordered by severity."""

    NORMAL = 0
    ALERT = 1
    SAFE = 2
    # Value 3 reserved (was RECOVERY, removed — never wired into evaluation)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default risk-score weights (tuned heuristically; will be refined by data)
_DEFAULT_WEIGHTS: dict[str, float] = {
    "entropy": 0.30,
    "delta_h": 0.25,
    "rep_count": 0.30,
    "top1_prob_inv": 0.15,
}

# Normalization constants — map raw signal values to roughly [0, 1]
_DEFAULT_NORMS: dict[str, float] = {
    "entropy": 5.0,  # entropy of ~5 nats is high for Qwen-3B
    "delta_h": 3.0,  # delta_h of ~3 is a large jump
    "rep_count": 3.0,  # 3 window repeats = established loop
    "top1_prob_inv": 1.0,  # 1 - top1_prob is already in [0, 1]
}


@dataclass(frozen=True)
class ControllerConfig:
    """Tunable parameters for the risk controller.

    Attributes:
        tau_low: Risk threshold for NORMAL → ALERT transition.
        tau_high: Risk threshold for ALERT → SAFE transition.
        k_escalate: Consecutive high-risk tokens required to escalate.
        j_deescalate: Consecutive low-risk tokens required to de-escalate.
        base_compression_ratio: Compression ratio in NORMAL mode (fraction removed).
        safe_compression_ratio: Relaxed compression ratio in SAFE mode.
        weights: Signal weights for risk score computation.
        norms: Normalization constants for risk score inputs.
    """

    tau_low: float = 0.3
    tau_high: float = 0.7
    k_escalate: int = 8
    j_deescalate: int = 5
    base_compression_ratio: float = 0.875
    safe_compression_ratio: float = 0.0
    weights: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))
    norms: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_NORMS))


# ---------------------------------------------------------------------------
# Controller action (output of each step)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControllerAction:
    """Action output by the controller at each generation step.

    Attributes:
        mode: Current operating mode.
        compression_ratio: Recommended compression ratio (fraction to remove).
        protect_thinking_tokens: Whether to flag thinking-token KV entries
            for eviction immunity.
        trigger_recomputation: Whether to trigger selective KV recomputation
            for anchor tokens.
        risk_score: The computed risk score for this step.
    """

    mode: Mode
    compression_ratio: float
    protect_thinking_tokens: bool
    trigger_recomputation: bool
    risk_score: float


# ---------------------------------------------------------------------------
# Risk score computation
# ---------------------------------------------------------------------------


def compute_risk_score(
    signals: dict[str, Any],
    *,
    weights: dict[str, float] | None = None,
    norms: dict[str, float] | None = None,
) -> float:
    """Compute a scalar risk score from per-token signals.

    This is a lightweight rule-based scorer (not the ML predictor).
    Combines entropy, delta_h, rep_count, and inverse top-1 probability
    into a weighted sum in [0, 1].

    Args:
        signals: TokenSignals dict (or any dict with the expected keys).
        weights: Signal component weights (default: balanced heuristic).
        norms: Normalization constants mapping raw values to ~[0, 1].

    Returns:
        Risk score in [0, 1] (clipped).
    """
    w = weights if weights is not None else _DEFAULT_WEIGHTS
    n = norms if norms is not None else _DEFAULT_NORMS

    # Extract raw values with safe defaults
    entropy = float(signals.get("entropy", 0.0))
    delta_h_raw = signals.get("delta_h")
    delta_h = max(float(delta_h_raw), 0.0) if delta_h_raw is not None else 0.0
    rep_count = float(signals.get("rep_count", 0))
    top1_prob = float(signals.get("top1_prob", 1.0))

    # Normalize to roughly [0, 1] (guard against zero norms)
    def _norm(value: float, norm: float) -> float:
        return min(value / norm, 1.0) if norm > 0 else (1.0 if value > 0 else 0.0)

    components = {
        "entropy": _norm(entropy, n["entropy"]),
        "delta_h": _norm(delta_h, n["delta_h"]),
        "rep_count": _norm(rep_count, n["rep_count"]),
        "top1_prob_inv": _norm(1.0 - top1_prob, n["top1_prob_inv"]),
    }

    score = sum(w.get(k, 0.0) * v for k, v in components.items())
    return max(0.0, min(score, 1.0))


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


def _decide_mode(
    current_mode: Mode,
    risk_score: float,
    consecutive_high: int,
    consecutive_low: int,
    config: ControllerConfig,
) -> Mode:
    """Pure function: decide the next mode given current state.

    Transition rules (with hysteresis):
        Escalation (requires ``k_escalate`` consecutive tokens above threshold):
            NORMAL  → ALERT:    risk > τ_low  for K tokens
            ALERT   → SAFE:     risk > τ_high for K tokens

        De-escalation (requires ``j_deescalate`` consecutive tokens below threshold):
            SAFE     → ALERT:   risk < τ_high for J tokens
            ALERT    → NORMAL:  risk < τ_low  for J tokens
    """
    k = config.k_escalate
    j = config.j_deescalate

    # --- Escalation ---
    if current_mode == Mode.NORMAL and consecutive_high >= k and risk_score > config.tau_low:
        return Mode.ALERT
    if current_mode == Mode.ALERT and consecutive_high >= k and risk_score > config.tau_high:
        return Mode.SAFE

    # --- De-escalation ---
    if current_mode == Mode.SAFE and consecutive_low >= j and risk_score < config.tau_high:
        return Mode.ALERT
    if current_mode == Mode.ALERT and consecutive_low >= j and risk_score < config.tau_low:
        return Mode.NORMAL

    return current_mode


class RiskController:
    """Online risk-aware controller for KV-cache compression.

    Maintains a state machine that monitors per-token signals and
    decides operating mode + actions at each generation step.

    Usage::

        ctrl = RiskController(config)
        for token_signals in generation_stream:
            action = ctrl.step(token_signals)
            # Use action.compression_ratio, action.protect_thinking_tokens, etc.
    """

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()
        self._mode: Mode = Mode.NORMAL
        self._step_count: int = 0
        self._consecutive_high: int = 0  # tokens above current escalation threshold
        self._consecutive_low: int = 0  # tokens below current de-escalation threshold
        self._risk_history: list[float] = []

    # -- Properties --

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def risk_history(self) -> list[float]:
        return list(self._risk_history)

    # -- Core --

    def step(self, signals: dict[str, Any]) -> ControllerAction:
        """Process one token's signals and return the controller action.

        Args:
            signals: Per-token signal dict (TokenSignals-compatible).

        Returns:
            ControllerAction with mode, compression ratio, and flags.
        """
        risk = compute_risk_score(
            signals,
            weights=self.config.weights,
            norms=self.config.norms,
        )
        self._risk_history.append(risk)

        # Update consecutive counters based on current mode thresholds
        escalation_threshold = (
            self.config.tau_high if self._mode >= Mode.ALERT else self.config.tau_low
        )
        deescalation_threshold = (
            self.config.tau_low if self._mode <= Mode.ALERT else self.config.tau_high
        )

        if risk > escalation_threshold:
            self._consecutive_high += 1
            self._consecutive_low = 0
        elif risk < deescalation_threshold:
            self._consecutive_low += 1
            self._consecutive_high = 0
        else:
            # In the dead zone between thresholds — reset both
            self._consecutive_high = 0
            self._consecutive_low = 0

        # Decide mode transition
        new_mode = _decide_mode(
            self._mode,
            risk,
            self._consecutive_high,
            self._consecutive_low,
            self.config,
        )

        # If mode changed, reset counters
        if new_mode != self._mode:
            self._consecutive_high = 0
            self._consecutive_low = 0
            self._mode = new_mode

        self._step_count += 1

        return ControllerAction(
            mode=self._mode,
            compression_ratio=self._action_compression_ratio(),
            protect_thinking_tokens=self._mode >= Mode.ALERT,
            trigger_recomputation=False,
            risk_score=risk,
        )

    def step_with_risk(self, risk_score: float) -> ControllerAction:
        """Process one token using a pre-computed risk score.

        Same state machine logic as :meth:`step`, but accepts a risk score
        directly (e.g., hazard probability from the ML predictor) instead of
        computing it from raw signals via :func:`compute_risk_score`.

        Args:
            risk_score: Pre-computed risk in [0, 1] (e.g., XGBoost hazard_prob).

        Returns:
            ControllerAction with mode, compression ratio, and flags.
        """
        risk = max(0.0, min(risk_score, 1.0))
        self._risk_history.append(risk)

        # Update consecutive counters based on current mode thresholds
        escalation_threshold = (
            self.config.tau_high if self._mode >= Mode.ALERT else self.config.tau_low
        )
        deescalation_threshold = (
            self.config.tau_low if self._mode <= Mode.ALERT else self.config.tau_high
        )

        if risk > escalation_threshold:
            self._consecutive_high += 1
            self._consecutive_low = 0
        elif risk < deescalation_threshold:
            self._consecutive_low += 1
            self._consecutive_high = 0
        else:
            self._consecutive_high = 0
            self._consecutive_low = 0

        new_mode = _decide_mode(
            self._mode,
            risk,
            self._consecutive_high,
            self._consecutive_low,
            self.config,
        )

        if new_mode != self._mode:
            self._consecutive_high = 0
            self._consecutive_low = 0
            self._mode = new_mode

        self._step_count += 1

        return ControllerAction(
            mode=self._mode,
            compression_ratio=self._action_compression_ratio(),
            protect_thinking_tokens=self._mode >= Mode.ALERT,
            trigger_recomputation=False,
            risk_score=risk,
        )

    def reset(self) -> None:
        """Reset controller to initial state (for a new generation)."""
        self._mode = Mode.NORMAL
        self._step_count = 0
        self._consecutive_high = 0
        self._consecutive_low = 0
        self._risk_history.clear()

    # -- Internal helpers --

    def _action_compression_ratio(self) -> float:
        """Return compression ratio for the current mode."""
        if self._mode == Mode.NORMAL:
            return self.config.base_compression_ratio
        if self._mode == Mode.ALERT:
            return self.config.base_compression_ratio
        # SAFE: relaxed compression
        return self.config.safe_compression_ratio
