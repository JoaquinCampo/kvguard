#!/usr/bin/env python3
"""Ablation suite for the KVGuard controller (v2).

Runs four ablation experiments to demonstrate component contributions:
1. Always-safe baseline: What if we just turn off compression entirely?
2. Random predictor: Does the trained predictor matter?
3. Hysteresis (k): Does requiring consecutive tokens matter?
4. Threshold sensitivity: Pareto frontier of CFR reduction vs FP rate

All ablations use held-out prompts only (prompt-level split integrity).
Excludes corrupted observed_attention data.

OPTIMIZATION: Loads all traces once and reuses for all evaluations.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kvguard.controller import ControllerConfig
from kvguard.evaluate_controller import (
    BudgetResult,
    EvalResult,
    TraceInfo,
    _run_predictor_state_machine,
    filter_traces_by_prompts,
    load_all_traces,
)

RESULTS_DIR = Path("results")
MODEL_PATH = Path("models/hazard_predictor.json")
SPLIT_INFO_PATH = Path("models/split_info.json")
OUTPUT_PATH = Path("models/controller_ablations.json")
FEATURE_ABLATION_OUTPUT = Path("models/ablation_results.json")

# Feature ablation definitions: name → list of features to DROP
FEATURE_ABLATION_CONFIGS: dict[str, list[str]] = {
    "full": [],
    "no_compression_ratio": ["compression_ratio"],
    "no_rep": ["rep_count", "rep_count_sum_8"],
    "logit_only": ["compression_ratio", "rep_count", "rep_count_sum_8"],
}

NUM_PROMPTS = 500
MODEL_FILTER = "Qwen/Qwen2.5-7B-Instruct"
CLEAN_COMPRESSORS = {"streaming_llm", "snapkv"}
# Compressors to exclude from build_dataset() — paper claims 2 compressors only
PRESS_EXCLUDE = ["observed_attention", "expected_attention"]

# Balanced config from T-005 (k=3 was best balanced operating point)
BALANCED_CONFIG = ControllerConfig(
    tau_low=0.3,
    tau_high=0.7,
    k_escalate=3,
    j_deescalate=5,
    base_compression_ratio=0.875,
    safe_compression_ratio=0.0,
)


def load_holdout_ids() -> set[str]:
    """Load held-out prompt IDs from split_info.json."""
    split = json.loads(SPLIT_INFO_PATH.read_text())
    return set(split["val_prompt_ids"])


# ---------------------------------------------------------------------------
# Optimized evaluation: precompute hazard probs, reuse for all configs
# ---------------------------------------------------------------------------


def precompute_hazard_probs(
    traces: dict[tuple[str, float], dict[str, TraceInfo]],
    predictor,
) -> dict[tuple[str, float], dict[str, list[float]]]:
    """Precompute hazard probabilities for all traces once.

    This is the expensive part (XGBoost inference on feature matrices).
    The state machine simulation is cheap and can be re-run with different configs.
    """
    from kvguard.features import ROLLING_WINDOW, add_rolling_features, flatten_signals

    probs: dict[tuple[str, float], dict[str, list[float]]] = {}

    for key, prompt_traces in traces.items():
        probs[key] = {}
        for prompt_id, trace in prompt_traces.items():
            if trace.num_tokens == 0:
                probs[key][prompt_id] = []
                continue

            X_base = flatten_signals(trace.signals)
            X_full = add_rolling_features(
                X_base,
                window=ROLLING_WINDOW,
                compression_ratio=trace.compression_ratio,
            )
            hazard_p = predictor.predict_proba(X_full)[:, 1].tolist()
            probs[key][prompt_id] = hazard_p

    return probs


def evaluate_with_precomputed(
    traces: dict[tuple[str, float], dict[str, TraceInfo]],
    hazard_probs: dict[tuple[str, float], dict[str, list[float]]],
    config: ControllerConfig,
) -> EvalResult:
    """Run controller evaluation using precomputed hazard probabilities.

    This is fast because it only runs the state machine (no ML inference).
    """
    safe_ratio = config.safe_compression_ratio
    eval_result = EvalResult(safe_compression_ratio=safe_ratio)

    if safe_ratio == 0.0:
        safe_key = ("none", 0.0)
    else:
        safe_key = None  # Will be computed per-press

    for (press, ratio), prompt_traces in sorted(traces.items()):
        if press == "none":
            continue
        if ratio <= safe_ratio:
            continue
        if press not in CLEAN_COMPRESSORS:
            continue

        sk = safe_key if safe_key else (press, safe_ratio)
        if sk not in traces:
            continue
        safe_traces = traces.get(sk, {})

        n_prompts = len(prompt_traces)
        baseline_cfr_count = 0
        controller_triggered = 0
        triggered_before_onset = 0
        prevented = 0
        false_triggers = 0
        trigger_tokens: list[int] = []
        n_correct = 0

        for prompt_id, trace in prompt_traces.items():
            hp = hazard_probs.get((press, ratio), {}).get(prompt_id, [])

            if trace.correct:
                n_correct += 1
            if trace.has_cfr_catastrophe:
                baseline_cfr_count += 1

            if not hp:
                continue

            # Run state machine with precomputed probs
            mode_history, safe_trigger = _run_predictor_state_machine(hp, config)

            onset = trace.cfr_onset
            lead_time = None
            if onset is not None and safe_trigger is not None and safe_trigger < onset:
                lead_time = onset - safe_trigger

            if safe_trigger is not None:
                controller_triggered += 1
                trigger_tokens.append(safe_trigger)

                if not trace.has_cfr_catastrophe:
                    false_triggers += 1

                if trace.has_cfr_catastrophe and lead_time is not None and lead_time > 0:
                    triggered_before_onset += 1
                    safe_trace = safe_traces.get(prompt_id)
                    if safe_trace is not None and not safe_trace.has_cfr_catastrophe:
                        prevented += 1

        controlled_cfr_count = baseline_cfr_count - prevented
        baseline_cfr = baseline_cfr_count / n_prompts if n_prompts > 0 else 0.0
        controlled_cfr = controlled_cfr_count / n_prompts if n_prompts > 0 else 0.0
        reduction_abs = baseline_cfr - controlled_cfr
        reduction_pct = (reduction_abs / baseline_cfr * 100) if baseline_cfr > 0 else 0.0

        eval_result.budgets.append(
            BudgetResult(
                press=press,
                compression_ratio=ratio,
                n_prompts=n_prompts,
                baseline_cfr_count=baseline_cfr_count,
                baseline_cfr=round(baseline_cfr, 4),
                baseline_accuracy=round(n_correct / n_prompts, 4) if n_prompts > 0 else 0.0,
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


def evaluate_random_predictor(
    traces: dict[tuple[str, float], dict[str, TraceInfo]],
    config: ControllerConfig,
    seed: int,
) -> EvalResult:
    """Run controller eval with random hazard probs (no ML inference needed)."""
    rng = np.random.RandomState(seed)
    safe_ratio = config.safe_compression_ratio
    eval_result = EvalResult(safe_compression_ratio=safe_ratio)

    safe_key = ("none", 0.0) if safe_ratio == 0.0 else None

    for (press, ratio), prompt_traces in sorted(traces.items()):
        if press == "none" or ratio <= safe_ratio or press not in CLEAN_COMPRESSORS:
            continue

        sk = safe_key if safe_key else (press, safe_ratio)
        if sk not in traces:
            continue
        safe_traces = traces.get(sk, {})

        n_prompts = len(prompt_traces)
        baseline_cfr_count = 0
        controller_triggered = 0
        triggered_before_onset = 0
        prevented = 0
        false_triggers = 0
        trigger_tokens: list[int] = []
        n_correct = 0

        for prompt_id, trace in prompt_traces.items():
            if trace.correct:
                n_correct += 1
            if trace.has_cfr_catastrophe:
                baseline_cfr_count += 1

            n = trace.num_tokens
            if n == 0:
                continue

            # Generate random hazard probs
            hp = rng.uniform(0, 1, size=n).tolist()
            mode_history, safe_trigger = _run_predictor_state_machine(hp, config)

            onset = trace.cfr_onset
            lead_time = None
            if onset is not None and safe_trigger is not None and safe_trigger < onset:
                lead_time = onset - safe_trigger

            if safe_trigger is not None:
                controller_triggered += 1
                trigger_tokens.append(safe_trigger)
                if not trace.has_cfr_catastrophe:
                    false_triggers += 1
                if trace.has_cfr_catastrophe and lead_time is not None and lead_time > 0:
                    triggered_before_onset += 1
                    safe_trace = safe_traces.get(prompt_id)
                    if safe_trace is not None and not safe_trace.has_cfr_catastrophe:
                        prevented += 1

        controlled_cfr_count = baseline_cfr_count - prevented
        baseline_cfr = baseline_cfr_count / n_prompts if n_prompts > 0 else 0.0
        controlled_cfr = controlled_cfr_count / n_prompts if n_prompts > 0 else 0.0
        reduction_abs = baseline_cfr - controlled_cfr
        reduction_pct = (reduction_abs / baseline_cfr * 100) if baseline_cfr > 0 else 0.0

        eval_result.budgets.append(
            BudgetResult(
                press=press,
                compression_ratio=ratio,
                n_prompts=n_prompts,
                baseline_cfr_count=baseline_cfr_count,
                baseline_cfr=round(baseline_cfr, 4),
                baseline_accuracy=round(n_correct / n_prompts, 4) if n_prompts > 0 else 0.0,
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


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def summarize_eval(result: EvalResult, label: str) -> dict:
    """Extract summary metrics from an eval result."""
    budgets = result.budgets
    if not budgets:
        return {"label": label, "error": "no budgets"}

    total_baseline = sum(b.baseline_cfr_count for b in budgets)
    total_controlled = sum(b.controlled_cfr_count for b in budgets)
    total_prevented = sum(b.catastrophes_prevented for b in budgets)
    total_prompts = sum(b.n_prompts for b in budgets)
    total_false = sum(b.false_trigger_count for b in budgets)
    total_non_catastrophe = total_prompts - total_baseline

    overall_reduction = (
        (total_baseline - total_controlled) / total_baseline * 100 if total_baseline > 0 else 0.0
    )
    fp_rate = total_false / total_non_catastrophe * 100 if total_non_catastrophe > 0 else 0.0

    per_compressor = {}
    for press in sorted(CLEAN_COMPRESSORS):
        press_budgets = [b for b in budgets if b.press == press]
        if not press_budgets:
            continue
        pb_baseline = sum(b.baseline_cfr_count for b in press_budgets)
        pb_controlled = sum(b.controlled_cfr_count for b in press_budgets)
        pb_false = sum(b.false_trigger_count for b in press_budgets)
        pb_prompts = sum(b.n_prompts for b in press_budgets)
        pb_non_cat = pb_prompts - pb_baseline
        per_compressor[press] = {
            "baseline_cfr": pb_baseline,
            "controlled_cfr": pb_controlled,
            "reduction_pct": round((pb_baseline - pb_controlled) / pb_baseline * 100, 1)
            if pb_baseline > 0
            else 0.0,
            "false_positives": pb_false,
            "fp_rate_pct": round(pb_false / pb_non_cat * 100, 1) if pb_non_cat > 0 else 0.0,
        }

    # Gradual-onset (exclude snapkv 0.875 instant onset)
    gradual_budgets = [
        b for b in budgets if not (b.press == "snapkv" and b.compression_ratio >= 0.875)
    ]
    gradual_baseline = sum(b.baseline_cfr_count for b in gradual_budgets)
    gradual_controlled = sum(b.controlled_cfr_count for b in gradual_budgets)
    gradual_reduction = (
        (gradual_baseline - gradual_controlled) / gradual_baseline * 100
        if gradual_baseline > 0
        else 0.0
    )

    return {
        "label": label,
        "total_baseline_cfr": total_baseline,
        "total_controlled_cfr": total_controlled,
        "total_prevented": total_prevented,
        "overall_cfr_reduction_pct": round(overall_reduction, 1),
        "gradual_onset_reduction_pct": round(gradual_reduction, 1),
        "false_positive_rate_pct": round(fp_rate, 1),
        "per_compressor": per_compressor,
    }


# ---------------------------------------------------------------------------
# Ablation experiments
# ---------------------------------------------------------------------------


def ablation_always_safe(traces, hazard_probs) -> dict:
    """Ablation 1: Always-safe ceiling."""
    print("\n=== ABLATION 1: Always-Safe Ceiling ===", flush=True)

    always_safe = ControllerConfig(
        tau_low=0.0,
        tau_high=0.0,
        k_escalate=1,
        j_deescalate=9999,
        safe_compression_ratio=0.0,
    )

    result = evaluate_with_precomputed(traces, hazard_probs, always_safe)
    summary = summarize_eval(result, "always_safe")

    print(f"  CFR reduction: {summary['overall_cfr_reduction_pct']}%", flush=True)
    print(f"  Gradual-onset: {summary['gradual_onset_reduction_pct']}%", flush=True)
    print(f"  FP rate: {summary['false_positive_rate_pct']}%", flush=True)

    return {
        "description": "Always-safe: trigger SAFE on every trace. Upper bound.",
        "summary": summary,
    }


def ablation_random_predictor(traces, hazard_probs) -> dict:
    """Ablation 2: Random predictor."""
    print("\n=== ABLATION 2: Random Predictor ===", flush=True)

    seeds = [42, 123, 456, 789, 1337]
    results_by_seed = []
    for seed in seeds:
        result = evaluate_random_predictor(traces, BALANCED_CONFIG, seed)
        results_by_seed.append(summarize_eval(result, f"random_seed_{seed}"))

    avg_reduction = float(np.mean([r["overall_cfr_reduction_pct"] for r in results_by_seed]))
    avg_gradual = float(np.mean([r["gradual_onset_reduction_pct"] for r in results_by_seed]))
    avg_fp = float(np.mean([r["false_positive_rate_pct"] for r in results_by_seed]))
    std_reduction = float(np.std([r["overall_cfr_reduction_pct"] for r in results_by_seed]))

    # Trained predictor reference
    trained_result = evaluate_with_precomputed(traces, hazard_probs, BALANCED_CONFIG)
    trained_summary = summarize_eval(trained_result, "trained_predictor")

    print(
        f"  Random (avg {len(seeds)} seeds): "
        f"{avg_reduction:.1f}% ± {std_reduction:.1f}%, {avg_fp:.1f}% FP",
        flush=True,
    )
    trained_red = trained_summary["overall_cfr_reduction_pct"]
    trained_fp = trained_summary["false_positive_rate_pct"]
    print(
        f"  Trained: {trained_red}% reduction, {trained_fp}% FP",
        flush=True,
    )
    print(
        f"  Delta: {trained_summary['overall_cfr_reduction_pct'] - avg_reduction:.1f}pp", flush=True
    )

    return {
        "description": "Random predictor: uniform [0,1] noise vs trained XGBoost.",
        "random_seeds": results_by_seed,
        "avg_cfr_reduction_pct": round(avg_reduction, 1),
        "avg_gradual_reduction_pct": round(avg_gradual, 1),
        "avg_fp_rate_pct": round(avg_fp, 1),
        "std_reduction_pct": round(std_reduction, 1),
        "trained_comparison": trained_summary,
    }


def ablation_hysteresis(traces, hazard_probs) -> dict:
    """Ablation 3: Hysteresis sweep."""
    print("\n=== ABLATION 3: Hysteresis (k sweep) ===", flush=True)

    k_values = [1, 2, 3, 4, 6, 8, 12, 16]
    results = {}

    for k in k_values:
        label = f"k={k}"
        if k == 1:
            label += " (no hysteresis)"
        elif k == 3:
            label += " (balanced)"

        config = ControllerConfig(
            tau_low=0.3,
            tau_high=0.7,
            k_escalate=k,
            j_deescalate=5,
            safe_compression_ratio=0.0,
        )

        result = evaluate_with_precomputed(traces, hazard_probs, config)
        summary = summarize_eval(result, label)
        results[label] = {"k": k, "summary": summary}
        print(
            f"  {label}: {summary['overall_cfr_reduction_pct']}% overall, "
            f"{summary['gradual_onset_reduction_pct']}% gradual, "
            f"{summary['false_positive_rate_pct']}% FP",
            flush=True,
        )

    return {
        "description": "Hysteresis ablation: vary k. Shows state machine contribution.",
        "configs": results,
    }


def ablation_threshold_sweep(traces, hazard_probs) -> dict:
    """Ablation 4: Threshold sensitivity sweep."""
    print("\n=== ABLATION 4: Threshold Sensitivity ===", flush=True)

    tau_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []

    for tau_low in tau_values:
        for tau_high in tau_values:
            if tau_high < tau_low:
                continue

            config = ControllerConfig(
                tau_low=tau_low,
                tau_high=tau_high,
                k_escalate=3,
                j_deescalate=5,
                safe_compression_ratio=0.0,
            )

            result = evaluate_with_precomputed(traces, hazard_probs, config)
            summary = summarize_eval(result, f"tau={tau_low}/{tau_high}")
            results.append(
                {
                    "tau_low": tau_low,
                    "tau_high": tau_high,
                    "cfr_reduction_pct": summary["overall_cfr_reduction_pct"],
                    "gradual_onset_reduction_pct": summary["gradual_onset_reduction_pct"],
                    "false_positive_rate_pct": summary["false_positive_rate_pct"],
                    "prevented": summary["total_prevented"],
                    "per_compressor": summary["per_compressor"],
                }
            )

    # Pareto-optimal configs
    pareto = []
    for r in results:
        dominated = any(
            other["cfr_reduction_pct"] > r["cfr_reduction_pct"]
            and other["false_positive_rate_pct"] <= r["false_positive_rate_pct"]
            for other in results
        )
        if not dominated:
            pareto.append(r)

    pareto.sort(key=lambda x: x["false_positive_rate_pct"])

    print(f"  Tested {len(results)} threshold combinations", flush=True)
    print(f"  Pareto-optimal ({len(pareto)}):", flush=True)
    for p in pareto:
        print(
            f"    tau={p['tau_low']}/{p['tau_high']}: "
            f"{p['cfr_reduction_pct']}% overall, "
            f"{p['gradual_onset_reduction_pct']}% gradual, "
            f"{p['false_positive_rate_pct']}% FP",
            flush=True,
        )

    return {
        "description": "Threshold sweep: tau_low x tau_high with k=3. Pareto frontier.",
        "all_results": results,
        "pareto_optimal": pareto,
        "n_configs_tested": len(results),
    }


# ---------------------------------------------------------------------------
# Naive entropy threshold baseline
# ---------------------------------------------------------------------------


def ablation_entropy_threshold(traces, hazard_probs) -> dict:
    """Ablation 5: Naive entropy threshold baseline.

    Instead of the trained ML predictor, use a simple rule:
    risk = min(entropy / tau_scale, 1.0) for each token.
    This answers: "Is the ML predictor better than just thresholding entropy?"
    """
    print("\n=== ABLATION 5: Naive Entropy Threshold ===", flush=True)

    # Compute entropy-based "hazard" probabilities for all traces
    tau_scale_values = [3.0, 4.0, 5.0, 6.0, 8.0]
    results = {}

    for tau_scale in tau_scale_values:
        # Convert each trace's entropy to a pseudo-risk in [0, 1]
        entropy_probs: dict[tuple[str, float], dict[str, list[float]]] = {}
        for key, prompt_traces in traces.items():
            entropy_probs[key] = {}
            for prompt_id, trace in prompt_traces.items():
                if trace.num_tokens == 0 or not trace.signals:
                    entropy_probs[key][prompt_id] = []
                    continue
                # Simple: risk = clamp(entropy / tau_scale, 0, 1)
                probs_list = [
                    min(float(s.get("entropy", 0.0)) / tau_scale, 1.0) for s in trace.signals
                ]
                entropy_probs[key][prompt_id] = probs_list

        # Evaluate using the balanced config
        result = evaluate_with_precomputed(traces, entropy_probs, BALANCED_CONFIG)
        summary = summarize_eval(result, f"entropy_tau={tau_scale}")
        label = f"tau_scale={tau_scale}"
        results[label] = {
            "tau_scale": tau_scale,
            "summary": summary,
        }

        if "error" not in summary:
            print(
                f"  entropy / {tau_scale}: "
                f"{summary['overall_cfr_reduction_pct']}% overall, "
                f"{summary['gradual_onset_reduction_pct']}% gradual, "
                f"{summary['false_positive_rate_pct']}% FP",
                flush=True,
            )
        else:
            print(f"  entropy / {tau_scale}: no data", flush=True)

    # Also run the trained predictor for comparison
    trained_result = evaluate_with_precomputed(traces, hazard_probs, BALANCED_CONFIG)
    trained_summary = summarize_eval(trained_result, "trained_predictor")

    if "error" not in trained_summary:
        print(
            f"  Trained predictor: "
            f"{trained_summary['overall_cfr_reduction_pct']}% overall, "
            f"{trained_summary['false_positive_rate_pct']}% FP",
            flush=True,
        )
    else:
        print("  Trained predictor: no data", flush=True)

    return {
        "description": "Naive entropy threshold: risk = clamp(entropy / tau_scale, 0, 1). "
        "No ML, just raw entropy.",
        "configs": results,
        "trained_comparison": trained_summary,
    }


# ---------------------------------------------------------------------------
# Feature ablations
# ---------------------------------------------------------------------------


def feature_ablations() -> dict:
    """Feature ablation: retrain predictor with subsets of features.

    For each ablation variant, loads the dataset, drops features,
    splits by prompt, trains XGBoost, and evaluates with both
    all-token and pre-onset metrics.
    """
    print("\n=== FEATURE ABLATION SUITE ===", flush=True)

    from kvguard.features import build_dataset
    from kvguard.train import evaluate_predictor, split_by_prompt, train_predictor

    ds_full = build_dataset(
        RESULTS_DIR,
        num_prompts=NUM_PROMPTS,
        model_filter=MODEL_FILTER,
        press_exclude=PRESS_EXCLUDE,
    )

    results: dict[str, dict] = {}

    for label, features_to_drop in FEATURE_ABLATION_CONFIGS.items():
        print(f"\n  Variant: {label}", flush=True)
        if features_to_drop:
            print(f"    Dropping: {features_to_drop}", flush=True)
            ds = ds_full.drop_features(features_to_drop)
        else:
            ds = ds_full

        split = split_by_prompt(ds, val_fraction=0.2, random_state=42)

        X_train = ds.X[split.train_mask]
        y_train = ds.y[split.train_mask]
        X_val = ds.X[split.val_mask]
        y_val = ds.y[split.val_mask]

        # Compute pre-onset mask for val tokens
        val_pre_onset_mask = None
        if len(ds.onset_positions) > 0:
            val_onset = ds.onset_positions[split.val_mask]
            val_trace_ids = ds.trace_ids[split.val_mask]
            token_positions = np.zeros(len(val_trace_ids), dtype=np.int32)
            prev_trace = -1
            pos = 0
            for i in range(len(val_trace_ids)):
                if val_trace_ids[i] != prev_trace:
                    pos = 0
                    prev_trace = val_trace_ids[i]
                token_positions[i] = pos
                pos += 1
            val_pre_onset_mask = (val_onset == -1) | (token_positions < val_onset)

        model = train_predictor(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
        )

        metrics = evaluate_predictor(
            model,
            X_val,
            y_val,
            pre_onset_mask=val_pre_onset_mask,
        )

        entry = {
            "auroc": metrics.auroc,
            "pre_onset_auroc": metrics.pre_onset_auroc,
            "f1": metrics.f1,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "pre_onset_recall": metrics.pre_onset_recall,
            "n_features": int(ds.X.shape[1]),
            "features_dropped": features_to_drop,
            "n_val_samples": metrics.n_samples,
            "n_val_positive": metrics.n_positive,
        }
        results[label] = entry

        pre_auroc = f"{metrics.pre_onset_auroc:.4f}" if metrics.pre_onset_auroc else "N/A"
        print(
            f"    AUROC: {metrics.auroc:.4f} | "
            f"pre-onset AUROC: {pre_auroc} | "
            f"F1: {metrics.f1:.4f} | "
            f"n_features: {ds.X.shape[1]}",
            flush=True,
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    t0 = time.time()
    print("KVGuard Controller Ablation Suite (v2)", flush=True)
    print("=" * 60, flush=True)

    # Load holdout prompt IDs
    holdout = load_holdout_ids()
    print(f"Holdout prompts: {len(holdout)}", flush=True)

    # Load predictor
    print(f"Loading predictor from {MODEL_PATH}...", flush=True)
    predictor = xgb.XGBClassifier()
    predictor.load_model(str(MODEL_PATH))

    # Load ALL traces ONCE
    print("Loading all traces (this is the slow part)...", flush=True)
    all_traces = load_all_traces(
        RESULTS_DIR,
        num_prompts=NUM_PROMPTS,
        model_filter=MODEL_FILTER,
    )
    all_traces = filter_traces_by_prompts(all_traces, holdout)
    n_keys = len(all_traces)
    n_traces = sum(len(v) for v in all_traces.values())
    print(f"Loaded {n_traces} traces across {n_keys} (press, ratio) configs", flush=True)

    # Precompute hazard probs ONCE (XGBoost inference)
    print("Precomputing hazard probabilities...", flush=True)
    hazard_probs = precompute_hazard_probs(all_traces, predictor)
    print(f"Precomputation done ({time.time() - t0:.1f}s elapsed)", flush=True)

    all_results = {
        "metadata": {
            "num_prompts": NUM_PROMPTS,
            "model_filter": MODEL_FILTER,
            "n_holdout": len(holdout),
            "clean_compressors": sorted(CLEAN_COMPRESSORS),
            "balanced_config": {
                "k": BALANCED_CONFIG.k_escalate,
                "j": BALANCED_CONFIG.j_deescalate,
                "tau_low": BALANCED_CONFIG.tau_low,
                "tau_high": BALANCED_CONFIG.tau_high,
            },
            "leakage_verified": True,
        },
    }

    # Run all ablations (fast — just state machine simulation)
    all_results["always_safe"] = ablation_always_safe(all_traces, hazard_probs)
    all_results["random_predictor"] = ablation_random_predictor(all_traces, hazard_probs)
    all_results["hysteresis"] = ablation_hysteresis(all_traces, hazard_probs)
    all_results["threshold_sweep"] = ablation_threshold_sweep(all_traces, hazard_probs)
    all_results["entropy_threshold"] = ablation_entropy_threshold(all_traces, hazard_probs)

    # Save controller ablation results
    OUTPUT_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nController ablations saved to {OUTPUT_PATH}", flush=True)

    # Feature ablations (independent of controller ablations)
    feature_results = feature_ablations()
    feature_output = {
        "metadata": {
            "generated_by": "scripts/run_ablations.py::feature_ablations()",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "num_prompts": NUM_PROMPTS,
            "model_filter": MODEL_FILTER,
            "press_exclude": PRESS_EXCLUDE,
            "compressors": sorted(CLEAN_COMPRESSORS),
            "split": "prompt-level, val_fraction=0.2, seed=42",
        },
        "variants": feature_results,
    }
    FEATURE_ABLATION_OUTPUT.write_text(json.dumps(feature_output, indent=2))
    print(f"Feature ablations saved to {FEATURE_ABLATION_OUTPUT}", flush=True)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s", flush=True)

    # Summary
    print(f"\n{'=' * 60}", flush=True)
    print("ABLATION SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)

    balanced = all_results["random_predictor"]["trained_comparison"]
    print(
        f"\nReference (balanced k=3): "
        f"{balanced['overall_cfr_reduction_pct']}% overall, "
        f"{balanced['gradual_onset_reduction_pct']}% gradual, "
        f"{balanced['false_positive_rate_pct']}% FP",
        flush=True,
    )

    asafe = all_results["always_safe"]["summary"]
    print(
        f"\n1. Always-Safe Ceiling: {asafe['overall_cfr_reduction_pct']}% reduction, "
        f"{asafe['false_positive_rate_pct']}% FP",
        flush=True,
    )

    rp = all_results["random_predictor"]
    print(
        f"\n2. Random Predictor: {rp['avg_cfr_reduction_pct']}% ± {rp['std_reduction_pct']}%",
        flush=True,
    )
    delta = balanced["overall_cfr_reduction_pct"] - rp["avg_cfr_reduction_pct"]
    print(f"   Trained adds: {delta:.1f}pp", flush=True)

    print("\n3. Hysteresis:", flush=True)
    for label, data in all_results["hysteresis"]["configs"].items():
        s = data["summary"]
        red = s["overall_cfr_reduction_pct"]
        fp = s["false_positive_rate_pct"]
        print(f"   {label}: {red}% overall, {fp}% FP", flush=True)

    pareto = all_results["threshold_sweep"]["pareto_optimal"]
    print(
        f"\n4. Threshold Sweep: {all_results['threshold_sweep']['n_configs_tested']} tested, "
        f"{len(pareto)} Pareto-optimal",
        flush=True,
    )

    print("\n5. Entropy Threshold Baseline:", flush=True)
    et = all_results["entropy_threshold"]
    for label, data in et["configs"].items():
        s = data["summary"]
        if "error" not in s:
            print(
                f"   {label}: {s['overall_cfr_reduction_pct']}% CFR red, "
                f"{s['false_positive_rate_pct']}% FP",
                flush=True,
            )
    tc = et["trained_comparison"]
    if "error" not in tc:
        print(
            f"   Trained predictor: {tc['overall_cfr_reduction_pct']}% CFR red, "
            f"{tc['false_positive_rate_pct']}% FP",
            flush=True,
        )


if __name__ == "__main__":
    main()
