"""Post-hoc analysis of Phase 4 live validation results.

Loads condition JSONs from the live validation output directory
and computes comprehensive metrics:

  - CBR reduction per ratio (with Wilson CIs)
  - Lead time distribution (how early controller triggers before onset)
  - False trigger rate (controller activated on non-catastrophe traces)
  - Memory savings retained (effective compression under control)
  - Hazard probability calibration (predictor quality at onset)
  - Mode transition statistics (state machine behavior)
  - Latency overhead (wall-clock cost of online prediction)
  - Offline vs live comparison (simulation gap)
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from kvguard.config import LiveResult

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

CBR_CATS = ("looping", "non_termination")


def load_live_results(output_dir: Path) -> dict[str, list[LiveResult]]:
    """Load all condition results from the live validation output directory."""
    conditions: dict[str, list[LiveResult]] = {}
    for fpath in sorted(output_dir.glob("*_*p.json")):
        data = json.loads(fpath.read_text())
        cond_name = data["condition"]
        conditions[cond_name] = [LiveResult(**r) for r in data["results"]]
        logger.info(f"Loaded {cond_name}: {len(conditions[cond_name])} results from {fpath.name}")
    return conditions


# ---------------------------------------------------------------------------
# Wilson CI helper
# ---------------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5 / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def fmt_ci(k: int, n: int) -> str:
    """Format count/total with CI."""
    lo, hi = wilson_ci(k, n)
    return f"{k}/{n} ({lo:.1%}–{hi:.1%})"


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------


def compute_cbr_reduction(
    static: list[LiveResult],
    controlled: list[LiveResult],
) -> dict[str, Any]:
    """CBR reduction from static to controlled, with CIs."""
    n = len(static)
    static_cbr = sum(1 for r in static if any(c in CBR_CATS for c in r.catastrophes))
    ctrl_cbr = sum(1 for r in controlled if any(c in CBR_CATS for c in r.catastrophes))

    reduction_pct = (static_cbr - ctrl_cbr) / static_cbr * 100 if static_cbr > 0 else 0.0

    return {
        "n_prompts": n,
        "static_cbr": fmt_ci(static_cbr, n),
        "controlled_cbr": fmt_ci(ctrl_cbr, n),
        "reduction_pct": round(reduction_pct, 1),
        "static_cbr_count": static_cbr,
        "controlled_cbr_count": ctrl_cbr,
    }


def compute_accuracy(results: list[LiveResult]) -> dict[str, Any]:
    """Accuracy with CI."""
    n = len(results)
    correct = sum(1 for r in results if r.correct)
    return {
        "accuracy": fmt_ci(correct, n),
        "correct": correct,
        "n": n,
    }


def compute_lead_times(controlled: list[LiveResult]) -> dict[str, Any]:
    """Lead time: how many tokens before catastrophe onset did controller trigger?"""
    lead_times: list[int] = []
    triggered_after: list[int] = []

    for r in controlled:
        if r.safe_trigger_token is None:
            continue
        for cat_type, onset in r.catastrophe_onsets.items():
            if cat_type not in CBR_CATS:
                continue
            lt = onset - r.safe_trigger_token
            if lt > 0:
                lead_times.append(lt)
            elif lt <= 0:
                triggered_after.append(-lt)

    result: dict[str, Any] = {
        "count_triggered_before_onset": len(lead_times),
        "count_triggered_after_onset": len(triggered_after),
    }
    if lead_times:
        arr = np.array(lead_times)
        result.update(
            {
                "mean": round(float(arr.mean()), 1),
                "median": round(float(np.median(arr)), 1),
                "min": int(arr.min()),
                "max": int(arr.max()),
                "q25": round(float(np.quantile(arr, 0.25)), 1),
                "q75": round(float(np.quantile(arr, 0.75)), 1),
            }
        )
    return result


def compute_false_trigger_rate(
    static: list[LiveResult],
    controlled: list[LiveResult],
) -> dict[str, Any]:
    """False triggers: controller activated on traces that had no catastrophe in static."""
    static_by_pid = {r.prompt_id: r for r in static}

    total_triggers = 0
    false_triggers = 0

    for r in controlled:
        if r.safe_trigger_token is None:
            continue
        total_triggers += 1

        s = static_by_pid.get(r.prompt_id)
        if s is None:
            continue

        # False positive: controller triggered, but no catastrophe in static condition
        static_has_cat = any(c in CBR_CATS for c in s.catastrophes)
        if not static_has_cat:
            false_triggers += 1

    return {
        "total_triggers": total_triggers,
        "false_triggers": false_triggers,
        "false_trigger_rate": fmt_ci(false_triggers, total_triggers),
    }


def compute_memory_savings(
    baseline: list[LiveResult],
    static: list[LiveResult],
    controlled: list[LiveResult],
) -> dict[str, Any]:
    """Memory savings: how much compression is retained under control."""
    baseline_by_pid = {r.prompt_id: r for r in baseline}

    static_savings: list[float] = []
    ctrl_savings: list[float] = []

    for r in static:
        b = baseline_by_pid.get(r.prompt_id)
        if b is None or not r.cache_sizes or not b.cache_sizes:
            continue
        baseline_cache = b.cache_sizes[-1] if b.cache_sizes else b.num_tokens_generated
        if baseline_cache > 0:
            static_savings.append(1.0 - r.cache_sizes[-1] / baseline_cache)

    for r in controlled:
        b = baseline_by_pid.get(r.prompt_id)
        if b is None or not r.cache_sizes or not b.cache_sizes:
            continue
        baseline_cache = b.cache_sizes[-1] if b.cache_sizes else b.num_tokens_generated
        if baseline_cache > 0:
            ctrl_savings.append(1.0 - r.cache_sizes[-1] / baseline_cache)

    result: dict[str, Any] = {}
    if static_savings:
        result["static_mean_savings_pct"] = round(float(np.mean(static_savings) * 100), 1)
    if ctrl_savings:
        result["controlled_mean_savings_pct"] = round(float(np.mean(ctrl_savings) * 100), 1)
    if static_savings and ctrl_savings:
        # How much of static's savings does controlled retain?
        s_mean = np.mean(static_savings)
        c_mean = np.mean(ctrl_savings)
        if s_mean > 0:
            result["savings_retained_pct"] = round(float(c_mean / s_mean * 100), 1)

    return result


def compute_hazard_calibration(controlled: list[LiveResult]) -> dict[str, Any]:
    """How well does the hazard predictor correlate with actual catastrophes?"""
    max_hazard_cat: list[float] = []  # max hazard on catastrophe traces
    max_hazard_nocat: list[float] = []  # max hazard on clean traces
    hazard_at_onset: list[float] = []

    for r in controlled:
        if not r.hazard_probs:
            continue
        has_cat = any(c in CBR_CATS for c in r.catastrophes)
        max_h = max(r.hazard_probs)

        if has_cat:
            max_hazard_cat.append(max_h)
            for cat_type, onset in r.catastrophe_onsets.items():
                if cat_type in CBR_CATS and 0 <= onset < len(r.hazard_probs):
                    hazard_at_onset.append(r.hazard_probs[onset])
        else:
            max_hazard_nocat.append(max_h)

    result: dict[str, Any] = {}
    if max_hazard_cat:
        result["mean_max_hazard_catastrophe"] = round(float(np.mean(max_hazard_cat)), 3)
    if max_hazard_nocat:
        result["mean_max_hazard_clean"] = round(float(np.mean(max_hazard_nocat)), 3)
    if hazard_at_onset:
        result["mean_hazard_at_onset"] = round(float(np.mean(hazard_at_onset)), 3)
    if max_hazard_cat and max_hazard_nocat:
        # Separation: how distinguishable are catastrophe vs clean traces?
        result["max_hazard_separation"] = round(
            float(np.mean(max_hazard_cat) - np.mean(max_hazard_nocat)), 3
        )
    return result


def compute_mode_transitions(controlled: list[LiveResult]) -> dict[str, Any]:
    """Controller state machine behavior statistics."""
    mode_names = {0: "NORMAL", 1: "ALERT", 2: "SAFE"}
    transition_counts: dict[str, int] = {}
    safe_tokens: list[int] = []
    traces_reaching_safe = 0

    for r in controlled:
        if not r.mode_history:
            continue
        if r.safe_trigger_token is not None:
            traces_reaching_safe += 1
            safe_tokens.append(r.safe_trigger_token)

        for i in range(1, len(r.mode_history)):
            prev, curr = r.mode_history[i - 1], r.mode_history[i]
            if prev != curr:
                key = f"{mode_names.get(prev, str(prev))}→{mode_names.get(curr, str(curr))}"
                transition_counts[key] = transition_counts.get(key, 0) + 1

    result: dict[str, Any] = {
        "traces_reaching_safe": traces_reaching_safe,
        "transitions": transition_counts,
    }
    if safe_tokens:
        arr = np.array(safe_tokens)
        result["safe_trigger_token_mean"] = round(float(arr.mean()), 1)
        result["safe_trigger_token_median"] = round(float(np.median(arr)), 1)
    return result


def compute_latency_overhead(
    baseline: list[LiveResult],
    static: list[LiveResult],
    controlled: list[LiveResult],
) -> dict[str, Any]:
    """Wall-clock overhead of online prediction and control."""
    baseline_by_pid = {r.prompt_id: r for r in baseline}
    static_by_pid = {r.prompt_id: r for r in static}

    # Controlled vs baseline (total overhead including compression + prediction)
    ctrl_vs_base: list[float] = []
    # Controlled vs static (overhead of prediction alone, since both compress)
    ctrl_vs_static: list[float] = []

    for r in controlled:
        b = baseline_by_pid.get(r.prompt_id)
        s = static_by_pid.get(r.prompt_id)
        if b and b.generation_time_seconds > 0:
            ctrl_vs_base.append(
                (r.generation_time_seconds - b.generation_time_seconds)
                / b.generation_time_seconds
                * 100
            )
        if s and s.generation_time_seconds > 0:
            ctrl_vs_static.append(
                (r.generation_time_seconds - s.generation_time_seconds)
                / s.generation_time_seconds
                * 100
            )

    result: dict[str, Any] = {
        "mean_baseline_time_s": round(
            float(np.mean([r.generation_time_seconds for r in baseline])), 2
        ),
        "mean_static_time_s": round(float(np.mean([r.generation_time_seconds for r in static])), 2),
        "mean_controlled_time_s": round(
            float(np.mean([r.generation_time_seconds for r in controlled])), 2
        ),
    }
    if ctrl_vs_base:
        result["overhead_vs_baseline_pct"] = round(float(np.mean(ctrl_vs_base)), 1)
    if ctrl_vs_static:
        result["overhead_vs_static_pct"] = round(float(np.mean(ctrl_vs_static)), 1)
    return result


def compare_offline_vs_live(
    offline_eval_path: Path | None,
    static: list[LiveResult],
    controlled: list[LiveResult],
    ratio: float,
) -> dict[str, Any]:
    """Compare offline-predicted CBR reduction vs live-observed."""
    n = len(static)
    static_cbr = sum(1 for r in static if any(c in CBR_CATS for c in r.catastrophes))
    ctrl_cbr = sum(1 for r in controlled if any(c in CBR_CATS for c in r.catastrophes))
    live_reduction = (static_cbr - ctrl_cbr) / static_cbr * 100 if static_cbr > 0 else 0.0

    result: dict[str, Any] = {
        "live_static_cbr": f"{static_cbr}/{n}",
        "live_controlled_cbr": f"{ctrl_cbr}/{n}",
        "live_reduction_pct": round(live_reduction, 1),
    }

    if offline_eval_path and offline_eval_path.exists():
        offline = json.loads(offline_eval_path.read_text())
        for b in offline.get("budgets", []):
            if b["press"] == "streaming_llm" and abs(b["compression_ratio"] - ratio) < 0.01:
                result["offline_reduction_pct"] = b["cfr_reduction_pct"]
                result["offline_baseline_cbr"] = f"{b['baseline_cfr_count']}/{b['n_prompts']}"
                gap = abs(live_reduction - b["cfr_reduction_pct"])
                result["simulation_gap_pp"] = round(gap, 1)
                if live_reduction >= b["cfr_reduction_pct"]:
                    result["interpretation"] = "Live met or exceeded offline prediction"
                else:
                    result["interpretation"] = f"Live underperformed by {gap:.1f}pp"
                break

    return result


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------


def analyze_live_validation(
    output_dir: Path,
    offline_eval_path: Path | None = None,
) -> dict[str, Any]:
    """Run full post-hoc analysis on live validation results.

    Args:
        output_dir: Directory containing condition JSON files.
        offline_eval_path: Path to offline controller_eval.json for comparison.

    Returns:
        Nested dict with all metrics, also printed to console.
    """
    conditions = load_live_results(output_dir)

    if not conditions:
        logger.error(f"No results found in {output_dir}")
        return {}

    baseline = conditions.get("baseline", [])
    report: dict[str, Any] = {"conditions": list(conditions.keys())}

    # Per-condition accuracy
    print("\n" + "=" * 85)
    print("LIVE VALIDATION ANALYSIS")
    print("=" * 85)

    print("\n--- Accuracy ---")
    acc_report: dict[str, Any] = {}
    for cond_name, results in conditions.items():
        acc = compute_accuracy(results)
        acc_report[cond_name] = acc
        print(f"  {cond_name:<20} {acc['accuracy']}")
    report["accuracy"] = acc_report

    # Find all ratios tested
    ratios: list[float] = []
    for cond_name in conditions:
        if cond_name.startswith("static_"):
            try:
                ratios.append(float(cond_name.removeprefix("static_")))
            except ValueError:
                pass
    ratios.sort()

    # Per-ratio analysis
    for ratio in ratios:
        r_str = str(ratio)
        static = conditions.get(f"static_{r_str}", [])
        controlled = conditions.get(f"controlled_{r_str}", [])
        if not static or not controlled:
            continue

        ratio_report: dict[str, Any] = {}
        print(f"\n{'=' * 85}")
        print(f"RATIO = {ratio}")
        print(f"{'=' * 85}")

        # CBR reduction
        cbr = compute_cbr_reduction(static, controlled)
        ratio_report["cbr_reduction"] = cbr
        print("\n--- CBR Reduction ---")
        print(f"  Static:     {cbr['static_cbr']}")
        print(f"  Controlled: {cbr['controlled_cbr']}")
        print(f"  Reduction:  {cbr['reduction_pct']}%")

        # Lead times
        lt = compute_lead_times(controlled)
        ratio_report["lead_times"] = lt
        print("\n--- Lead Times ---")
        print(f"  Triggered before onset: {lt['count_triggered_before_onset']}")
        print(f"  Triggered after onset:  {lt['count_triggered_after_onset']}")
        if "mean" in lt:
            print(f"  Mean: {lt['mean']} tokens, Median: {lt['median']}")
            print(f"  Range: [{lt['min']}, {lt['max']}]")

        # False triggers
        ft = compute_false_trigger_rate(static, controlled)
        ratio_report["false_triggers"] = ft
        print("\n--- False Trigger Rate ---")
        print(f"  Total triggers: {ft['total_triggers']}")
        print(f"  False triggers: {ft['false_trigger_rate']}")

        # Memory savings
        mem = compute_memory_savings(baseline, static, controlled)
        ratio_report["memory_savings"] = mem
        print("\n--- Memory Savings ---")
        for k, v in mem.items():
            print(f"  {k}: {v}%")

        # Hazard calibration
        haz = compute_hazard_calibration(controlled)
        ratio_report["hazard_calibration"] = haz
        print("\n--- Hazard Calibration ---")
        for k, v in haz.items():
            print(f"  {k}: {v}")

        # Mode transitions
        mt = compute_mode_transitions(controlled)
        ratio_report["mode_transitions"] = mt
        print("\n--- Mode Transitions ---")
        print(f"  Traces reaching SAFE: {mt['traces_reaching_safe']}")
        if "safe_trigger_token_mean" in mt:
            print(
                f"  Mean SAFE token: {mt['safe_trigger_token_mean']}, "
                f"Median: {mt['safe_trigger_token_median']}"
            )
        if mt["transitions"]:
            for t, count in sorted(mt["transitions"].items()):
                print(f"    {t}: {count}")

        # Latency
        lat = compute_latency_overhead(baseline, static, controlled)
        ratio_report["latency"] = lat
        print("\n--- Latency ---")
        print(f"  Baseline: {lat['mean_baseline_time_s']}s")
        print(f"  Static:   {lat['mean_static_time_s']}s")
        print(f"  Controlled: {lat['mean_controlled_time_s']}s")
        if "overhead_vs_static_pct" in lat:
            print(f"  Overhead vs static: {lat['overhead_vs_static_pct']}%")

        # Offline comparison
        offline = compare_offline_vs_live(offline_eval_path, static, controlled, ratio)
        ratio_report["offline_comparison"] = offline
        print("\n--- Offline vs Live ---")
        print(f"  Live CBR reduction: {offline['live_reduction_pct']}%")
        if "offline_reduction_pct" in offline:
            print(f"  Offline prediction: {offline['offline_reduction_pct']}%")
            print(f"  Simulation gap:     {offline['simulation_gap_pp']}pp")
            print(f"  {offline['interpretation']}")
        else:
            print("  (No offline eval data available for comparison)")

        report[f"ratio_{r_str}"] = ratio_report

    print("\n" + "=" * 85)

    return report
