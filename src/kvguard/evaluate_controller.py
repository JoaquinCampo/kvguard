"""Offline controller evaluation: simulate hazard-predictor + controller on existing traces.

For each trace at aggressive compression that has a catastrophe, this module:
1. Runs the hazard predictor per-token to get P(catastrophe in next H tokens)
2. Feeds predictions to the controller state machine
3. Checks if the controller triggers SAFE mode before catastrophe onset
4. Cross-references the same prompt at the controller's safe_compression_ratio
   to determine if the catastrophe would have been prevented

This produces CFR-reduction curves: controlled-CFR vs compressor-only-CFR at each budget level.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from loguru import logger

from kvguard.controller import ControllerConfig, Mode
from kvguard.features import (
    ROLLING_WINDOW,
    add_rolling_features,
    flatten_signals,
    load_result_file,
    result_dict_to_run_result,
)

# ---------------------------------------------------------------------------
# Data loading: build a prompt-level lookup table
# ---------------------------------------------------------------------------


@dataclass
class TraceInfo:
    """Per-trace metadata for controller evaluation."""

    prompt_id: str
    press: str
    compression_ratio: float
    catastrophes: list[str]
    catastrophe_onsets: dict[str, int]
    num_tokens: int
    correct: bool | None
    signals: list[dict[str, Any]]
    max_new_tokens: int = 512
    nt_onset_frac: float = 0.75

    @property
    def has_cfr_catastrophe(self) -> bool:
        """Catastrophes that count toward CFR (looping, non_termination)."""
        return any(c in ("looping", "non_termination") for c in self.catastrophes)

    @property
    def cfr_onset(self) -> int | None:
        """Earliest onset of a CFR-relevant catastrophe.

        For non_termination, uses a proxy onset at ``nt_onset_frac * max_new_tokens``
        (clamped to sequence length) instead of the raw catastrophe_onsets value,
        matching the proxy logic in :func:`kvguard.labeling.compute_hazard_labels`.
        """
        if self.num_tokens == 0:
            return None

        onsets: list[int] = []

        if "looping" in self.catastrophe_onsets:
            onsets.append(self.catastrophe_onsets["looping"])

        if "non_termination" in self.catastrophes:
            proxy = min(
                int(self.nt_onset_frac * self.max_new_tokens),
                self.num_tokens - 1,
            )
            onsets.append(proxy)

        return min(onsets) if onsets else None


def load_all_traces(
    results_dir: Path,
    *,
    num_prompts: int = 50,
    nt_onset_frac: float = 0.75,
    model_filter: str | None = None,
) -> dict[tuple[str, float], dict[str, TraceInfo]]:
    """Load all traces indexed by (press, ratio) -> {prompt_id: TraceInfo}.

    Args:
        results_dir: Directory with sweep result JSONs.
        num_prompts: Filter to files matching this prompt count.
        nt_onset_frac: Non-termination proxy onset fraction, passed to TraceInfo
            for consistent cfr_onset computation.
        model_filter: If set, only include traces where ``run.model`` matches.

    Returns:
        Nested dict: traces[(press, ratio)][prompt_id] = TraceInfo
    """
    pattern = f"*_{num_prompts}p.json"
    files = sorted(results_dir.rglob(pattern))

    traces: dict[tuple[str, float], dict[str, TraceInfo]] = {}

    for fpath in files:
        _config, results_list = load_result_file(fpath)
        for rd in results_list:
            run = result_dict_to_run_result(rd)
            if model_filter and run.model != model_filter:
                continue
            key = (run.press, run.compression_ratio)
            if key not in traces:
                traces[key] = {}

            sigs = [s if isinstance(s, dict) else s.model_dump() for s in rd.get("signals", [])]

            traces[key][run.prompt_id] = TraceInfo(
                prompt_id=run.prompt_id,
                press=run.press,
                compression_ratio=run.compression_ratio,
                catastrophes=run.catastrophes,
                catastrophe_onsets=run.catastrophe_onsets,
                num_tokens=run.num_tokens_generated,
                correct=run.correct,
                signals=sigs[: run.num_tokens_generated],
                max_new_tokens=run.max_new_tokens,
                nt_onset_frac=nt_onset_frac,
            )

    return traces


def filter_traces_by_prompts(
    all_traces: dict[tuple[str, float], dict[str, Any]],
    prompt_ids: set[str],
) -> dict[tuple[str, float], dict[str, Any]]:
    """Filter traces to only include specified prompt_ids.

    Used to evaluate the controller only on held-out prompts that were
    not used for predictor training.
    """
    return {
        key: {pid: trace for pid, trace in traces.items() if pid in prompt_ids}
        for key, traces in all_traces.items()
    }


# ---------------------------------------------------------------------------
# Per-trace controller simulation
# ---------------------------------------------------------------------------


@dataclass
class ControllerTrace:
    """Result of running the controller on one trace."""

    prompt_id: str
    press: str
    compression_ratio: float
    has_cfr_catastrophe: bool
    cfr_onset: int | None
    controller_triggered_safe: bool
    safe_trigger_token: int | None
    lead_time: int | None  # tokens before onset that controller triggered
    mode_history: list[int]  # Mode values per token
    hazard_probs: list[float]  # predictor probabilities per token


def _run_predictor_state_machine(
    hazard_probs: list[float],
    config: ControllerConfig,
) -> tuple[list[int], int | None]:
    """Run the controller state machine using hazard probabilities directly.

    Instead of feeding synthetic signals through compute_risk_score(),
    this uses hazard_prob as the risk score directly — it's the ML
    predictor's calibrated P(catastrophe within H tokens).

    Transition logic mirrors RiskController._decide_mode():
    - Escalation: K consecutive tokens with hazard_prob > threshold
    - De-escalation: J consecutive tokens with hazard_prob < threshold

    Returns:
        (mode_history, safe_trigger_token)
    """
    mode = Mode.NORMAL
    consecutive_high = 0
    consecutive_low = 0
    mode_history: list[int] = []
    safe_trigger: int | None = None

    k = config.k_escalate
    j = config.j_deescalate

    for t, p in enumerate(hazard_probs):
        # Determine thresholds for current mode
        esc_thresh = config.tau_high if mode >= Mode.ALERT else config.tau_low
        deesc_thresh = config.tau_low if mode <= Mode.ALERT else config.tau_high

        # Update consecutive counters
        if p > esc_thresh:
            consecutive_high += 1
            consecutive_low = 0
        elif p < deesc_thresh:
            consecutive_low += 1
            consecutive_high = 0
        else:
            consecutive_high = 0
            consecutive_low = 0

        # Mode transitions (escalation)
        new_mode = mode
        if mode == Mode.NORMAL and consecutive_high >= k and p > config.tau_low:
            new_mode = Mode.ALERT
        elif mode == Mode.ALERT and consecutive_high >= k and p > config.tau_high:
            new_mode = Mode.SAFE

        # Mode transitions (de-escalation)
        if mode == Mode.SAFE and consecutive_low >= j and p < config.tau_high:
            new_mode = Mode.ALERT
        elif mode == Mode.ALERT and consecutive_low >= j and p < config.tau_low:
            new_mode = Mode.NORMAL

        if new_mode != mode:
            consecutive_high = 0
            consecutive_low = 0
            mode = new_mode

        mode_history.append(mode)

        if mode >= Mode.SAFE and safe_trigger is None:
            safe_trigger = t

    return mode_history, safe_trigger


def simulate_controller_on_trace(
    trace: TraceInfo,
    predictor: xgb.XGBClassifier,
    controller_config: ControllerConfig,
    *,
    rolling_window: int = ROLLING_WINDOW,
) -> ControllerTrace:
    """Run hazard predictor + controller on a single trace's signals.

    Builds features causally and uses predictor's P(catastrophe) as the
    risk score for the controller's mode-transition state machine.

    Args:
        trace: Pre-collected trace with per-token signals.
        predictor: Trained XGBoost hazard predictor.
        controller_config: Controller configuration.
        rolling_window: Window size for rolling features.

    Returns:
        ControllerTrace with full mode history and trigger info.
    """
    n = trace.num_tokens
    if n == 0:
        return ControllerTrace(
            prompt_id=trace.prompt_id,
            press=trace.press,
            compression_ratio=trace.compression_ratio,
            has_cfr_catastrophe=trace.has_cfr_catastrophe,
            cfr_onset=trace.cfr_onset,
            controller_triggered_safe=False,
            safe_trigger_token=None,
            lead_time=None,
            mode_history=[],
            hazard_probs=[],
        )

    # Build full feature matrix for the trace (causal rolling features)
    X_base = flatten_signals(trace.signals)
    X_full = add_rolling_features(
        X_base,
        window=rolling_window,
        compression_ratio=trace.compression_ratio,
    )

    # Batch prediction — get hazard probabilities for all tokens
    hazard_probs = predictor.predict_proba(X_full)[:, 1].tolist()

    # Run state machine with hazard probabilities as risk scores
    mode_history, safe_trigger = _run_predictor_state_machine(hazard_probs, controller_config)

    # Compute lead time (tokens before onset that controller triggered)
    onset = trace.cfr_onset
    lead_time = None
    if onset is not None and safe_trigger is not None and safe_trigger < onset:
        lead_time = onset - safe_trigger

    return ControllerTrace(
        prompt_id=trace.prompt_id,
        press=trace.press,
        compression_ratio=trace.compression_ratio,
        has_cfr_catastrophe=trace.has_cfr_catastrophe,
        cfr_onset=onset,
        controller_triggered_safe=safe_trigger is not None,
        safe_trigger_token=safe_trigger,
        lead_time=lead_time,
        mode_history=mode_history,
        hazard_probs=hazard_probs,
    )


# ---------------------------------------------------------------------------
# Controller evaluation: simulate across all traces, compute CFR reduction
# ---------------------------------------------------------------------------


@dataclass
class BudgetResult:
    """Evaluation result for one (compressor, ratio) budget level."""

    press: str
    compression_ratio: float
    n_prompts: int
    # Baseline (compressor-only)
    baseline_cfr_count: int
    baseline_cfr: float
    baseline_accuracy: float
    # Controller
    controller_triggered_count: int  # traces where controller entered SAFE+
    triggered_before_onset: int  # triggered before catastrophe onset
    catastrophes_prevented: int  # onset preceded by trigger AND safe-ratio is clean
    controlled_cfr_count: int
    controlled_cfr: float
    cfr_reduction_abs: float
    cfr_reduction_pct: float
    # Cost
    mean_trigger_token: float | None  # average token position of SAFE trigger
    false_trigger_count: int  # triggered on non-catastrophe traces


@dataclass
class EvalResult:
    """Full controller evaluation result."""

    safe_compression_ratio: float
    budgets: list[BudgetResult] = field(default_factory=list)


def evaluate_controller(
    results_dir: Path,
    predictor: xgb.XGBClassifier,
    *,
    num_prompts: int = 50,
    controller_config: ControllerConfig | None = None,
    rolling_window: int = ROLLING_WINDOW,
    holdout_prompt_ids: set[str] | None = None,
    nt_onset_frac: float = 0.75,
    model_filter: str | None = None,
) -> EvalResult:
    """Run full offline controller evaluation.

    For each (compressor, ratio) pair at aggressive compression:
    1. Simulate controller on every trace
    2. For traces where controller triggers SAFE before onset,
       check if the same prompt at safe_compression_ratio is catastrophe-free
    3. Compute controlled CFR vs baseline CFR

    Args:
        results_dir: Directory with sweep result JSONs.
        predictor: Trained XGBoost hazard predictor.
        num_prompts: Filter to files matching this prompt count.
        controller_config: Controller tuning parameters.
        rolling_window: Feature window size.
        nt_onset_frac: Non-termination proxy onset fraction.
        model_filter: If set, only include traces where ``run.model`` matches.

    Returns:
        EvalResult with per-budget-level CFR comparisons.
    """
    config = controller_config or ControllerConfig()
    safe_ratio = config.safe_compression_ratio

    # Load all traces
    all_traces = load_all_traces(
        results_dir,
        num_prompts=num_prompts,
        nt_onset_frac=nt_onset_frac,
        model_filter=model_filter,
    )

    if holdout_prompt_ids is not None:
        all_traces = filter_traces_by_prompts(all_traces, holdout_prompt_ids)
    else:
        logger.warning(
            "No holdout_prompt_ids provided — evaluating on ALL prompts. "
            "Pass holdout_prompt_ids from split_info.json to avoid data leakage."
        )

    eval_result = EvalResult(safe_compression_ratio=safe_ratio)

    # Evaluate each (press, ratio) pair where ratio > safe_ratio
    # (controller only helps when starting from more aggressive compression)
    for (press, ratio), prompt_traces in sorted(all_traces.items()):
        if press == "none":
            continue  # baseline has no compression to control
        if ratio <= safe_ratio:
            continue  # controller can only help if we start more aggressive

        # Get safe-ratio traces for cross-reference.
        # When safe_ratio=0.0, fall back to baseline ("none", 0.0).
        if safe_ratio == 0.0:
            safe_key = ("none", 0.0)
        else:
            safe_key = (press, safe_ratio)

        if safe_key not in all_traces:
            logger.warning(
                f"Safe key {safe_key} not in traces. "
                f"Available: {sorted(all_traces.keys())}. "
                f"Skipping ({press}, {ratio})."
            )
            continue
        safe_traces = all_traces.get(safe_key, {})

        n_prompts_actual = len(prompt_traces)
        baseline_cfr_count = 0
        controller_triggered = 0
        triggered_before_onset = 0
        prevented = 0
        false_triggers = 0
        trigger_tokens: list[int] = []
        n_correct = 0

        for prompt_id, trace in prompt_traces.items():
            # Run controller simulation
            ct = simulate_controller_on_trace(
                trace, predictor, config, rolling_window=rolling_window
            )

            if trace.correct:
                n_correct += 1

            if trace.has_cfr_catastrophe:
                baseline_cfr_count += 1

            if ct.controller_triggered_safe:
                controller_triggered += 1
                if ct.safe_trigger_token is not None:
                    trigger_tokens.append(ct.safe_trigger_token)

                if not trace.has_cfr_catastrophe:
                    false_triggers += 1

                if trace.has_cfr_catastrophe and ct.lead_time is not None and ct.lead_time > 0:
                    triggered_before_onset += 1

                    # Would the safe ratio have prevented this catastrophe?
                    safe_trace = safe_traces.get(prompt_id)
                    if safe_trace is not None and not safe_trace.has_cfr_catastrophe:
                        prevented += 1

        controlled_cfr_count = baseline_cfr_count - prevented
        baseline_cfr = baseline_cfr_count / n_prompts_actual if n_prompts_actual > 0 else 0.0
        controlled_cfr = controlled_cfr_count / n_prompts_actual if n_prompts_actual > 0 else 0.0

        reduction_abs = baseline_cfr - controlled_cfr
        reduction_pct = (reduction_abs / baseline_cfr * 100) if baseline_cfr > 0 else 0.0

        eval_result.budgets.append(
            BudgetResult(
                press=press,
                compression_ratio=ratio,
                n_prompts=n_prompts_actual,
                baseline_cfr_count=baseline_cfr_count,
                baseline_cfr=round(baseline_cfr, 4),
                baseline_accuracy=round(n_correct / n_prompts_actual, 4)
                if n_prompts_actual > 0
                else 0.0,
                controller_triggered_count=controller_triggered,
                triggered_before_onset=triggered_before_onset,
                catastrophes_prevented=prevented,
                controlled_cfr_count=controlled_cfr_count,
                controlled_cfr=round(controlled_cfr, 4),
                cfr_reduction_abs=round(reduction_abs, 4),
                cfr_reduction_pct=round(reduction_pct, 1),
                mean_trigger_token=round(float(np.mean(trigger_tokens)), 1)
                if trigger_tokens
                else None,
                false_trigger_count=false_triggers,
            )
        )

    return eval_result


def format_eval_table(result: EvalResult) -> str:
    """Format evaluation results as a human-readable table."""
    lines = [
        f"Controller Evaluation (safe_ratio={result.safe_compression_ratio})",
        "=" * 100,
        f"{'Press':<22} {'Ratio':>5} {'Base CFR':>9} {'Ctrl CFR':>9} {'Reduced':>8} "
        f"{'Prevent':>7} {'Trigger':>7} {'FP':>4} {'MeanTok':>8}",
        "-" * 100,
    ]

    for b in result.budgets:
        mt = f"{b.mean_trigger_token:.0f}" if b.mean_trigger_token is not None else "-"
        lines.append(
            f"{b.press:<22} {b.compression_ratio:>5.3f} "
            f"{b.baseline_cfr_count:>3}/{b.n_prompts:<3} "
            f"{b.controlled_cfr_count:>3}/{b.n_prompts:<3} "
            f"{b.cfr_reduction_pct:>6.1f}% "
            f"{b.catastrophes_prevented:>5}/{b.baseline_cfr_count:<3} "
            f"{b.controller_triggered_count:>5}/{b.n_prompts:<3} "
            f"{b.false_trigger_count:>3} "
            f"{mt:>8}"
        )

    lines.append("-" * 100)

    # Summary
    total_baseline = sum(b.baseline_cfr_count for b in result.budgets)
    total_controlled = sum(b.controlled_cfr_count for b in result.budgets)
    total_prevented = sum(b.catastrophes_prevented for b in result.budgets)
    total_prompts = sum(b.n_prompts for b in result.budgets)

    if total_baseline > 0:
        overall_reduction = (total_baseline - total_controlled) / total_baseline * 100
        lines.append(
            f"OVERALL: {total_prevented}/{total_baseline} catastrophes prevented "
            f"({overall_reduction:.1f}% CFR reduction) across {total_prompts} prompt-configs"
        )
    lines.append("=" * 100)

    return "\n".join(lines)


def eval_result_to_dict(result: EvalResult) -> dict[str, Any]:
    """Serialize evaluation result to a JSON-friendly dict."""
    return {
        "safe_compression_ratio": result.safe_compression_ratio,
        "budgets": [
            {
                "press": b.press,
                "compression_ratio": b.compression_ratio,
                "n_prompts": b.n_prompts,
                "baseline_cfr_count": b.baseline_cfr_count,
                "baseline_cfr": b.baseline_cfr,
                "baseline_accuracy": b.baseline_accuracy,
                "controller_triggered_count": b.controller_triggered_count,
                "triggered_before_onset": b.triggered_before_onset,
                "catastrophes_prevented": b.catastrophes_prevented,
                "controlled_cfr_count": b.controlled_cfr_count,
                "controlled_cfr": b.controlled_cfr,
                "cfr_reduction_abs": b.cfr_reduction_abs,
                "cfr_reduction_pct": b.cfr_reduction_pct,
                "mean_trigger_token": b.mean_trigger_token,
                "false_trigger_count": b.false_trigger_count,
            }
            for b in result.budgets
        ],
    }
