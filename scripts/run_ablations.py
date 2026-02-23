#!/usr/bin/env python3
"""Ablation suite for the KVGuard controller.

Runs four ablation experiments to demonstrate component contributions:
1. Always-safe baseline: What if we just turn off compression entirely?
2. Random predictor: Does the trained predictor matter?
3. Hysteresis (k): Does requiring consecutive tokens matter? (k=1 vs k=8)
4. Threshold sensitivity: Pareto frontier of CFR reduction vs FP rate

All ablations reuse the same loaded traces and predictor for efficiency.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import xgboost as xgb

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kvguard.controller import ControllerConfig
from kvguard.evaluate_controller import (
    EvalResult,
    eval_result_to_dict,
    evaluate_controller,
    load_all_traces,
    simulate_controller_on_trace,
)


RESULTS_DIR = Path("results")
MODEL_PATH = Path("models/hazard_predictor.json")
OUTPUT_DIR = Path("results/ablations")

# Balanced config from T-007 (our best config)
BALANCED_CONFIG = ControllerConfig(
    tau_low=0.3,
    tau_high=0.7,
    k_escalate=8,
    j_deescalate=5,
    base_compression_ratio=0.875,
    safe_compression_ratio=0.0,
)


def summarize_eval(result: EvalResult, label: str) -> dict:
    """Extract summary metrics from an eval result."""
    budgets = result.budgets
    if not budgets:
        return {"label": label, "error": "no budgets"}

    total_baseline = sum(b.baseline_cfr_count for b in budgets)
    total_controlled = sum(b.controlled_cfr_count for b in budgets)
    total_prevented = sum(b.catastrophes_prevented for b in budgets)
    total_prompts = sum(b.n_prompts for b in budgets)
    total_triggers = sum(b.controller_triggered_count for b in budgets)
    total_false = sum(b.false_trigger_count for b in budgets)
    total_non_catastrophe = total_prompts - total_baseline

    overall_reduction = (
        (total_baseline - total_controlled) / total_baseline * 100
        if total_baseline > 0
        else 0.0
    )
    fp_rate = (
        total_false / total_non_catastrophe * 100
        if total_non_catastrophe > 0
        else 0.0
    )

    # Per-compressor breakdown
    per_compressor = {}
    for press in ["observed_attention", "snapkv", "streaming_llm"]:
        press_budgets = [b for b in budgets if b.press == press]
        if not press_budgets:
            continue
        pb_baseline = sum(b.baseline_cfr_count for b in press_budgets)
        pb_controlled = sum(b.controlled_cfr_count for b in press_budgets)
        per_compressor[press] = {
            "baseline_cfr": pb_baseline,
            "controlled_cfr": pb_controlled,
            "reduction_pct": round(
                (pb_baseline - pb_controlled) / pb_baseline * 100, 1
            )
            if pb_baseline > 0
            else 0.0,
        }

    return {
        "label": label,
        "total_baseline_cfr": total_baseline,
        "total_controlled_cfr": total_controlled,
        "total_prevented": total_prevented,
        "overall_cfr_reduction_pct": round(overall_reduction, 1),
        "false_positive_rate_pct": round(fp_rate, 1),
        "per_compressor": per_compressor,
    }


def ablation_always_safe(predictor: xgb.XGBClassifier) -> dict:
    """Ablation 1: Always-safe baseline.

    What if instead of a controller, we just always use no compression?
    This tells us whether the controller adds value over simply using
    the safe fallback universally. The controller should beat this by
    maintaining compression most of the time while still preventing
    catastrophes.
    """
    print("\n=== ABLATION 1: Always-Safe Baseline ===")
    print("Config: Controller triggers immediately on every trace (k=1, tau_low=0.0)")

    # Config that triggers SAFE on literally every token
    always_safe = ControllerConfig(
        tau_low=0.0,  # any hazard prob > 0 triggers
        tau_high=0.0,
        k_escalate=1,  # single token triggers
        j_deescalate=9999,  # never de-escalate
        safe_compression_ratio=0.0,
    )

    result = evaluate_controller(
        RESULTS_DIR, predictor, controller_config=always_safe
    )
    summary = summarize_eval(result, "always_safe")

    # The "always safe" should prevent ~100% of catastrophes but at 100% FP rate
    print(f"  CFR reduction: {summary['overall_cfr_reduction_pct']}%")
    print(f"  FP rate: {summary['false_positive_rate_pct']}%")
    print(f"  (This is the 'turn off compression' upper bound)")

    return {
        "description": "Always-safe: trigger SAFE immediately on every trace (k=1, tau=0.0). Upper bound on prevention, but compression never active.",
        "summary": summary,
        "full_results": eval_result_to_dict(result),
    }


def ablation_random_predictor(predictor: xgb.XGBClassifier) -> dict:
    """Ablation 2: Random predictor.

    Replace the trained XGBoost with random uniform [0, 1] predictions.
    This shows whether the trained predictor is essential or if the
    state machine alone (with any noise input) could achieve similar results.
    """
    print("\n=== ABLATION 2: Random Predictor ===")
    print("Config: Replace trained predictor with uniform random noise")

    # We need to monkey-patch the predictor to output random probabilities.
    # The cleanest way: create a wrapper class that mimics predict_proba.
    class RandomPredictor:
        def __init__(self, seed: int = 42):
            self.rng = np.random.RandomState(seed)

        def predict_proba(self, X):
            n = X.shape[0]
            probs = self.rng.uniform(0, 1, size=n)
            return np.column_stack([1 - probs, probs])

    # Run with 3 different seeds and average
    results_by_seed = []
    for seed in [42, 123, 456]:
        random_pred = RandomPredictor(seed=seed)
        result = evaluate_controller(
            RESULTS_DIR, random_pred, controller_config=BALANCED_CONFIG
        )
        results_by_seed.append(summarize_eval(result, f"random_seed_{seed}"))

    # Average across seeds
    avg_reduction = np.mean(
        [r["overall_cfr_reduction_pct"] for r in results_by_seed]
    )
    avg_fp = np.mean([r["false_positive_rate_pct"] for r in results_by_seed])

    print(f"  Avg CFR reduction (3 seeds): {avg_reduction:.1f}%")
    print(f"  Avg FP rate (3 seeds): {avg_fp:.1f}%")

    # Compare with trained
    trained_result = evaluate_controller(
        RESULTS_DIR, predictor, controller_config=BALANCED_CONFIG
    )
    trained_summary = summarize_eval(trained_result, "trained_predictor")
    print(f"  Trained predictor: {trained_summary['overall_cfr_reduction_pct']}% reduction, {trained_summary['false_positive_rate_pct']}% FP")
    print(f"  Delta: {trained_summary['overall_cfr_reduction_pct'] - avg_reduction:.1f}pp better reduction")

    return {
        "description": "Random predictor: uniform [0,1] noise instead of trained XGBoost. Tests whether state machine alone explains results.",
        "random_seeds": results_by_seed,
        "avg_cfr_reduction_pct": round(float(avg_reduction), 1),
        "avg_fp_rate_pct": round(float(avg_fp), 1),
        "trained_comparison": trained_summary,
        "trained_full_results": eval_result_to_dict(trained_result),
    }


def ablation_hysteresis(predictor: xgb.XGBClassifier) -> dict:
    """Ablation 3: Hysteresis (k=1 vs k=8).

    Tests whether requiring k consecutive high-risk tokens matters.
    k=1 means instant trigger (no state machine benefit).
    k=8 is our balanced config (requires sustained signal).
    """
    print("\n=== ABLATION 3: Hysteresis (k=1 vs k=8) ===")

    configs = {
        "k=1 (no hysteresis)": ControllerConfig(
            tau_low=0.3,
            tau_high=0.7,
            k_escalate=1,
            j_deescalate=5,
            safe_compression_ratio=0.0,
        ),
        "k=2": ControllerConfig(
            tau_low=0.3,
            tau_high=0.7,
            k_escalate=2,
            j_deescalate=5,
            safe_compression_ratio=0.0,
        ),
        "k=4": ControllerConfig(
            tau_low=0.3,
            tau_high=0.7,
            k_escalate=4,
            j_deescalate=5,
            safe_compression_ratio=0.0,
        ),
        "k=8 (balanced)": BALANCED_CONFIG,
        "k=16": ControllerConfig(
            tau_low=0.3,
            tau_high=0.7,
            k_escalate=16,
            j_deescalate=5,
            safe_compression_ratio=0.0,
        ),
        "k=32": ControllerConfig(
            tau_low=0.3,
            tau_high=0.7,
            k_escalate=32,
            j_deescalate=5,
            safe_compression_ratio=0.0,
        ),
    }

    results = {}
    for label, config in configs.items():
        result = evaluate_controller(
            RESULTS_DIR, predictor, controller_config=config
        )
        summary = summarize_eval(result, label)
        results[label] = {
            "summary": summary,
            "full_results": eval_result_to_dict(result),
        }
        print(f"  {label}: {summary['overall_cfr_reduction_pct']}% reduction, {summary['false_positive_rate_pct']}% FP")

    return {
        "description": "Hysteresis ablation: vary k (consecutive tokens required to escalate). Shows state machine contribution.",
        "configs": results,
    }


def ablation_threshold_sensitivity(predictor: xgb.XGBClassifier) -> dict:
    """Ablation 4: Threshold sensitivity sweep.

    Systematic sweep of tau_low and tau_high to show Pareto frontier
    of CFR reduction vs false positive rate. Demonstrates robustness
    to threshold selection.
    """
    print("\n=== ABLATION 4: Threshold Sensitivity ===")

    tau_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []

    for tau_low in tau_values:
        for tau_high in tau_values:
            if tau_high < tau_low:
                continue  # tau_high must be >= tau_low

            config = ControllerConfig(
                tau_low=tau_low,
                tau_high=tau_high,
                k_escalate=8,
                j_deescalate=5,
                safe_compression_ratio=0.0,
            )

            result = evaluate_controller(
                RESULTS_DIR, predictor, controller_config=config
            )
            summary = summarize_eval(result, f"tau={tau_low}/{tau_high}")
            results.append({
                "tau_low": tau_low,
                "tau_high": tau_high,
                "cfr_reduction_pct": summary["overall_cfr_reduction_pct"],
                "false_positive_rate_pct": summary["false_positive_rate_pct"],
                "prevented": summary["total_prevented"],
                "per_compressor": summary["per_compressor"],
            })

    # Find Pareto-optimal configs
    pareto = []
    for r in results:
        dominated = False
        for other in results:
            if (
                other["cfr_reduction_pct"] > r["cfr_reduction_pct"]
                and other["false_positive_rate_pct"] <= r["false_positive_rate_pct"]
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(r)

    pareto.sort(key=lambda x: x["false_positive_rate_pct"])

    print(f"  Tested {len(results)} threshold combinations")
    print(f"  Pareto-optimal configs ({len(pareto)}):")
    for p in pareto:
        print(f"    tau={p['tau_low']}/{p['tau_high']}: {p['cfr_reduction_pct']}% reduction, {p['false_positive_rate_pct']}% FP")

    return {
        "description": "Threshold sensitivity: sweep tau_low x tau_high with k=8 fixed. Shows Pareto frontier of prevention vs false positives.",
        "all_results": results,
        "pareto_optimal": pareto,
        "n_configs_tested": len(results),
    }


def main():
    print("KVGuard Controller Ablation Suite")
    print("=" * 60)

    # Load predictor once
    print(f"Loading predictor from {MODEL_PATH}...")
    predictor = xgb.XGBClassifier()
    predictor.load_model(str(MODEL_PATH))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Run all ablations
    all_results["always_safe"] = ablation_always_safe(predictor)
    all_results["random_predictor"] = ablation_random_predictor(predictor)
    all_results["hysteresis"] = ablation_hysteresis(predictor)
    all_results["threshold_sensitivity"] = ablation_threshold_sensitivity(predictor)

    # Save combined results
    output_path = OUTPUT_DIR / "ablation_results.json"
    output_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n{'=' * 60}")
    print(f"All results saved to {output_path}")

    # Print summary table
    print(f"\n{'=' * 60}")
    print("ABLATION SUMMARY")
    print(f"{'=' * 60}")

    # Balanced config reference
    balanced = all_results["random_predictor"]["trained_comparison"]
    print(f"\nReference (balanced config): {balanced['overall_cfr_reduction_pct']}% CFR reduction, {balanced['false_positive_rate_pct']}% FP")

    # Always-safe
    asafe = all_results["always_safe"]["summary"]
    print(f"\n1. Always-Safe Baseline:")
    print(f"   {asafe['overall_cfr_reduction_pct']}% CFR reduction, {asafe['false_positive_rate_pct']}% FP")
    print(f"   → Controller saves {asafe['false_positive_rate_pct'] - balanced['false_positive_rate_pct']:.1f}pp FP by being selective")

    # Random predictor
    print(f"\n2. Random Predictor:")
    print(f"   {all_results['random_predictor']['avg_cfr_reduction_pct']}% CFR reduction (avg 3 seeds)")
    print(f"   → Trained predictor adds {balanced['overall_cfr_reduction_pct'] - all_results['random_predictor']['avg_cfr_reduction_pct']:.1f}pp")

    # Hysteresis
    print(f"\n3. Hysteresis (k=1 vs k=8):")
    k1 = all_results["hysteresis"]["configs"]["k=1 (no hysteresis)"]["summary"]
    k8 = all_results["hysteresis"]["configs"]["k=8 (balanced)"]["summary"]
    print(f"   k=1: {k1['overall_cfr_reduction_pct']}% reduction, {k1['false_positive_rate_pct']}% FP")
    print(f"   k=8: {k8['overall_cfr_reduction_pct']}% reduction, {k8['false_positive_rate_pct']}% FP")
    print(f"   → Hysteresis reduces FP by {k1['false_positive_rate_pct'] - k8['false_positive_rate_pct']:.1f}pp with {k8['overall_cfr_reduction_pct'] - k1['overall_cfr_reduction_pct']:.1f}pp prevention change")

    # Threshold sensitivity
    pareto = all_results["threshold_sensitivity"]["pareto_optimal"]
    print(f"\n4. Threshold Sensitivity:")
    print(f"   {all_results['threshold_sensitivity']['n_configs_tested']} configs tested, {len(pareto)} Pareto-optimal")
    print(f"   Range: {min(r['cfr_reduction_pct'] for r in pareto)}-{max(r['cfr_reduction_pct'] for r in pareto)}% reduction")


# ---------------------------------------------------------------------------
# Feature ablation sets
# ---------------------------------------------------------------------------

FEATURE_ABLATIONS: dict[str, list[str]] = {
    "full": [],
    "no_compression_ratio": ["compression_ratio"],
    "no_rep": ["rep_count", "rep_count_sum_8"],
    "logit_only": ["compression_ratio", "rep_count", "rep_count_sum_8"],
}


def ablation_feature_importance(results_dir: Path, output_dir: Path) -> list[dict]:
    """Retrain predictor with feature subsets and compare AUROC."""
    from kvguard.features import build_dataset
    from kvguard.train import evaluate_predictor, split_by_prompt, train_predictor

    ds = build_dataset(results_dir)
    results = []

    for ablation_name, drop_features in FEATURE_ABLATIONS.items():
        ds_ablated = ds.drop_features(drop_features) if drop_features else ds
        split = split_by_prompt(ds_ablated, val_fraction=0.2)

        X_tr = ds_ablated.X[split.train_mask]
        y_tr = ds_ablated.y[split.train_mask]
        X_va = ds_ablated.X[split.val_mask]
        y_va = ds_ablated.y[split.val_mask]

        model = train_predictor(X_tr, y_tr)
        metrics = evaluate_predictor(model, X_va, y_va)

        results.append({
            "ablation": ablation_name,
            "dropped_features": drop_features,
            "n_features": ds_ablated.X.shape[1],
            "auroc": metrics.auroc,
            "f1": metrics.f1,
            "recall": metrics.recall,
        })

    return results


if __name__ == "__main__":
    main()
