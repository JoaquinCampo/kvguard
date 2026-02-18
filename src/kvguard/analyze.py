"""Analyze experiment results and per-token signals.

Supports both old (5-field) and new (11-field HALT) signal formats.
"""

import json
import statistics
from pathlib import Path

from loguru import logger


def load_results(path: Path) -> dict:
    """Load a results JSON file."""
    return json.loads(path.read_text())


def load_all_results(
    result_dir: Path = Path("results"),
    prompt_filter: int | None = None,
) -> list[dict]:
    """Load all result files, optionally filtering by prompt count."""
    files = sorted(result_dir.rglob("*.json"))
    out = []
    for f in files:
        data = load_results(f)
        if prompt_filter and data["config"]["num_prompts"] != prompt_filter:
            continue
        out.append(data)
    return out


# ---------------------------------------------------------------------------
# Per-config signal statistics
# ---------------------------------------------------------------------------

def signal_stats(results: list[dict]) -> dict:
    """Compute aggregate signal statistics across all results in one config."""
    all_entropy: list[float] = []
    all_top1: list[float] = []
    all_delta_h: list[float] = []
    all_h_alts: list[float] = []
    max_entropies: list[float] = []
    min_top1s: list[float] = []
    max_abs_delta_h: list[float] = []
    thinking_counts: list[int] = []
    total_tokens: list[int] = []

    for r in results:
        sigs = r["signals"]
        if not sigs:
            continue

        entropies = [s["entropy"] for s in sigs]
        top1s = [s["top1_prob"] for s in sigs]

        all_entropy.extend(entropies)
        all_top1.extend(top1s)
        max_entropies.append(max(entropies))
        min_top1s.append(min(top1s))

        # New HALT features (backward-compatible)
        dhs = [s["delta_h"] for s in sigs if s.get("delta_h") is not None]
        if dhs:
            all_delta_h.extend(dhs)
            max_abs_delta_h.append(max(abs(d) for d in dhs))

        halts = [s["h_alts"] for s in sigs if "h_alts" in s]
        if halts:
            all_h_alts.extend(halts)

        think = sum(1 for s in sigs if s.get("is_thinking_token", False))
        thinking_counts.append(think)
        total_tokens.append(len(sigs))

    if not all_entropy:
        return {}

    stats = {
        "mean_entropy": round(statistics.mean(all_entropy), 3),
        "median_max_entropy": round(statistics.median(max_entropies), 3),
        "max_entropy": round(max(max_entropies), 3),
        "mean_top1": round(statistics.mean(all_top1), 3),
        "median_min_top1": round(statistics.median(min_top1s), 3),
        "min_top1": round(min(min_top1s), 3),
    }

    # HALT-expanded stats
    if all_delta_h:
        stats["mean_abs_delta_h"] = round(statistics.mean(abs(d) for d in all_delta_h), 4)
        stats["max_abs_delta_h"] = round(max(abs(d) for d in all_delta_h), 3)
        stats["p95_abs_delta_h"] = round(
            sorted(abs(d) for d in all_delta_h)[int(0.95 * len(all_delta_h))], 3
        )
    if all_h_alts:
        stats["mean_h_alts"] = round(statistics.mean(all_h_alts), 3)
        stats["max_h_alts"] = round(max(all_h_alts), 3)
    if thinking_counts:
        total_think = sum(thinking_counts)
        total_tok = sum(total_tokens)
        stats["thinking_token_pct"] = round(total_think / total_tok * 100, 1) if total_tok else 0

    return stats


# ---------------------------------------------------------------------------
# Degradation curve: accuracy & CFR vs compression ratio
# ---------------------------------------------------------------------------

def degradation_table(result_dir: Path = Path("results"), num_prompts: int = 50) -> None:
    """Print degradation curve table: accuracy & CFR by method × ratio."""
    all_data = load_all_results(result_dir, prompt_filter=num_prompts)
    if not all_data:
        logger.warning(f"No results with {num_prompts} prompts found in {result_dir}")
        return

    # Sort by method then ratio
    rows = []
    for data in all_data:
        cfg = data["config"]
        s = data["summary"]
        sig = signal_stats(data["results"])
        rows.append({
            "press": cfg["press_name"],
            "ratio": cfg["compression_ratio"],
            "n": s["total"],
            "accuracy": s.get("accuracy", 0),
            "cfr": s.get("catastrophic_failure_rate", 0),
            "non_term": s.get("catastrophe_counts", {}).get("non_termination", 0),
            "looping": s.get("catastrophe_counts", {}).get("looping", 0),
            "wrong": s.get("catastrophe_counts", {}).get("wrong_answer", 0),
            "avg_tokens": s.get("avg_tokens", 0),
            **sig,
        })

    rows.sort(key=lambda r: (r["press"], r["ratio"]))

    # Print table
    header = (
        f"{'Press':<18s} {'Ratio':>5s} {'Acc':>5s} {'CFR':>5s} "
        f"{'NT':>3s} {'Loop':>4s} {'Wrong':>5s} "
        f"{'maxH':>5s} {'maxDH':>6s} {'Think%':>6s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['press']:<18s} {r['ratio']:>5.3f} "
            f"{r['accuracy']:>5.1%} {r['cfr']:>5.1%} "
            f"{r['non_term']:>3d} {r['looping']:>4d} {r['wrong']:>5d} "
            f"{r.get('max_entropy', 0):>5.2f} "
            f"{r.get('max_abs_delta_h', 0):>6.2f} "
            f"{r.get('thinking_token_pct', 0):>6.1f}"
        )


# ---------------------------------------------------------------------------
# DeltaH early warning analysis
# ---------------------------------------------------------------------------

def delta_h_analysis(result_dir: Path = Path("results"), num_prompts: int = 50) -> None:
    """Analyze whether DeltaH gives earlier warning than raw entropy.

    For each failing run, find:
    - First token where entropy > threshold (entropy-based warning)
    - First token where |delta_h| > threshold (delta_h-based warning)
    - Compare which fires first
    """
    all_data = load_all_results(result_dir, prompt_filter=num_prompts)

    # Get baseline entropy stats for threshold calibration
    baseline = [d for d in all_data if d["config"]["press_name"] == "none"]
    if not baseline:
        logger.warning("No baseline results found")
        return

    # Compute baseline max entropy as threshold reference
    baseline_max_entropies = []
    baseline_abs_deltas = []
    for data in baseline:
        for r in data["results"]:
            sigs = r["signals"]
            if not sigs:
                continue
            baseline_max_entropies.append(max(s["entropy"] for s in sigs))
            for s in sigs:
                if s.get("delta_h") is not None:
                    baseline_abs_deltas.append(abs(s["delta_h"]))

    if not baseline_max_entropies:
        logger.warning("No baseline signal data")
        return

    # Thresholds: just above baseline maximum
    entropy_threshold = max(baseline_max_entropies) * 1.1
    delta_h_threshold = sorted(baseline_abs_deltas)[int(0.99 * len(baseline_abs_deltas))] * 1.1

    print(f"Entropy threshold: {entropy_threshold:.2f} (1.1 × baseline max)")
    print(f"|DeltaH| threshold: {delta_h_threshold:.2f} (1.1 × baseline p99)")
    print()

    # Analyze compressed runs
    compressed = [d for d in all_data if d["config"]["press_name"] != "none"]
    results_table = []

    for data in compressed:
        cfg = data["config"]
        for r in data["results"]:
            if not r["catastrophes"]:
                continue  # only analyze failing runs

            sigs = r["signals"]
            if not sigs:
                continue

            # Find first entropy exceedance
            entropy_warn = None
            for t, s in enumerate(sigs):
                if s["entropy"] > entropy_threshold:
                    entropy_warn = t
                    break

            # Find first delta_h exceedance
            delta_warn = None
            for t, s in enumerate(sigs):
                if s.get("delta_h") is not None and abs(s["delta_h"]) > delta_h_threshold:
                    delta_warn = t
                    break

            lead_time = None
            if entropy_warn is not None and delta_warn is not None:
                lead_time = entropy_warn - delta_warn  # positive = delta_h warns earlier

            results_table.append({
                "press": cfg["press_name"],
                "ratio": cfg["compression_ratio"],
                "prompt": r["prompt_id"],
                "cats": ",".join(r["catastrophes"]),
                "entropy_warn_t": entropy_warn,
                "delta_warn_t": delta_warn,
                "lead_time": lead_time,
            })

    if not results_table:
        print("No failing runs found in compressed configs.")
        return

    # Summary
    lead_times = [r["lead_time"] for r in results_table if r["lead_time"] is not None]
    delta_earlier = sum(1 for lt in lead_times if lt > 0)
    entropy_earlier = sum(1 for lt in lead_times if lt < 0)
    same = sum(1 for lt in lead_times if lt == 0)
    no_data = sum(1 for r in results_table if r["lead_time"] is None)

    print(f"Failing runs analyzed: {len(results_table)}")
    print(f"DeltaH warns earlier: {delta_earlier}")
    print(f"Entropy warns earlier: {entropy_earlier}")
    print(f"Same token: {same}")
    print(f"No comparable data: {no_data}")
    if lead_times:
        print(f"Mean lead time (DeltaH advantage): {statistics.mean(lead_times):.1f} tokens")
        print(f"Median lead time: {statistics.median(lead_times):.1f} tokens")

    # Per-method breakdown
    print(f"\n{'Press':<18s} {'Ratio':>5s} {'Failures':>8s} {'DH first':>8s} {'H first':>8s}")
    print("-" * 60)
    by_config = {}
    for r in results_table:
        key = (r["press"], r["ratio"])
        by_config.setdefault(key, []).append(r)

    for (press, ratio), runs in sorted(by_config.items()):
        lts = [r["lead_time"] for r in runs if r["lead_time"] is not None]
        dh_first = sum(1 for lt in lts if lt > 0)
        h_first = sum(1 for lt in lts if lt < 0)
        print(f"{press:<18s} {ratio:>5.3f} {len(runs):>8d} {dh_first:>8d} {h_first:>8d}")


# ---------------------------------------------------------------------------
# Silent failure analysis (reasoning corruption without looping)
# ---------------------------------------------------------------------------

def silent_failure_analysis(
    result_dir: Path = Path("results"), num_prompts: int = 50,
) -> None:
    """Analyze whether logit signals can distinguish silent failures.

    Silent failures = wrong_answer WITHOUT looping or non_termination.
    Loud failures = looping or non_termination present.
    """
    all_data = load_all_results(result_dir, prompt_filter=num_prompts)
    compressed = [d for d in all_data if d["config"]["press_name"] != "none"]

    silent: list[dict] = []
    loud: list[dict] = []
    correct: list[dict] = []

    for data in compressed:
        for r in data["results"]:
            sigs = r["signals"]
            if not sigs:
                continue

            stats = _per_run_signal_stats(sigs)
            stats["press"] = data["config"]["press_name"]
            stats["ratio"] = data["config"]["compression_ratio"]

            cats = set(r["catastrophes"])
            if not cats:
                correct.append(stats)
            elif cats == {"wrong_answer"}:
                silent.append(stats)
            else:
                loud.append(stats)

    print(f"Correct runs: {len(correct)}")
    print(f"Silent failures (wrong only): {len(silent)}")
    print(f"Loud failures (loop/non-term): {len(loud)}")
    print()

    if not silent and not loud:
        print("No failures to analyze.")
        return

    # Compare signal distributions
    features = ["max_entropy", "mean_entropy", "max_abs_delta_h", "mean_h_alts", "think_pct"]
    header = f"{'Feature':<20s} {'Correct':>10s} {'Silent':>10s} {'Loud':>10s}"
    print(header)
    print("-" * len(header))

    for feat in features:
        vals_c = [s[feat] for s in correct if feat in s] if correct else []
        vals_s = [s[feat] for s in silent if feat in s] if silent else []
        vals_l = [s[feat] for s in loud if feat in s] if loud else []

        def _fmt(vals: list[float]) -> str:
            if not vals:
                return "n/a"
            return f"{statistics.mean(vals):.3f}"

        print(f"{feat:<20s} {_fmt(vals_c):>10s} {_fmt(vals_s):>10s} {_fmt(vals_l):>10s}")


def _per_run_signal_stats(sigs: list[dict]) -> dict:
    """Compute signal stats for a single run's token signals."""
    entropies = [s["entropy"] for s in sigs]
    top1s = [s["top1_prob"] for s in sigs]
    delta_hs = [s["delta_h"] for s in sigs if s.get("delta_h") is not None]
    h_alts_vals = [s["h_alts"] for s in sigs if "h_alts" in s]
    think = sum(1 for s in sigs if s.get("is_thinking_token", False))

    stats: dict = {
        "max_entropy": max(entropies),
        "mean_entropy": statistics.mean(entropies),
        "min_top1": min(top1s),
        "mean_top1": statistics.mean(top1s),
        "think_pct": think / len(sigs) * 100 if sigs else 0,
    }
    if delta_hs:
        stats["max_abs_delta_h"] = max(abs(d) for d in delta_hs)
        stats["mean_abs_delta_h"] = statistics.mean(abs(d) for d in delta_hs)
    if h_alts_vals:
        stats["mean_h_alts"] = statistics.mean(h_alts_vals)
        stats["max_h_alts"] = max(h_alts_vals)
    return stats


# ---------------------------------------------------------------------------
# Rolling DeltaH threshold detector (ERGO-style)
# ---------------------------------------------------------------------------

def rolling_delta_h_detector(
    result_dir: Path = Path("results"),
    num_prompts: int = 50,
    window: int = 10,
) -> None:
    """Evaluate a simple rolling-DeltaH threshold detector offline.

    Computes rolling average entropy over a window, then checks if the
    change exceeds a threshold. Reports precision/recall for detecting
    catastrophic runs.
    """
    all_data = load_all_results(result_dir, prompt_filter=num_prompts)

    # Calibrate threshold from baseline
    baseline = [d for d in all_data if d["config"]["press_name"] == "none"]
    if not baseline:
        logger.warning("No baseline results for calibration")
        return

    baseline_rolling_dhs: list[float] = []
    for data in baseline:
        for r in data["results"]:
            sigs = r["signals"]
            rolling = _rolling_entropy(sigs, window)
            if len(rolling) >= 2:
                for i in range(1, len(rolling)):
                    baseline_rolling_dhs.append(abs(rolling[i] - rolling[i - 1]))

    if not baseline_rolling_dhs:
        print("No baseline rolling DH data")
        return

    # Try multiple threshold percentiles
    percentiles = [90, 95, 99]
    sorted_baseline = sorted(baseline_rolling_dhs)

    print(f"Baseline rolling |DH| (window={window}):")
    print(f"  Mean: {statistics.mean(baseline_rolling_dhs):.4f}")
    print(f"  Max: {max(baseline_rolling_dhs):.4f}")
    for p in percentiles:
        idx = int(p / 100 * len(sorted_baseline))
        print(f"  P{p}: {sorted_baseline[min(idx, len(sorted_baseline)-1)]:.4f}")
    print()

    # Evaluate each threshold on compressed runs
    compressed = [d for d in all_data if d["config"]["press_name"] != "none"]

    for p in percentiles:
        idx = int(p / 100 * len(sorted_baseline))
        threshold = sorted_baseline[min(idx, len(sorted_baseline) - 1)] * 1.5

        tp = fp = fn = tn = 0
        for data in compressed:
            for r in data["results"]:
                sigs = r["signals"]
                rolling = _rolling_entropy(sigs, window)
                triggered = False
                if len(rolling) >= 2:
                    for i in range(1, len(rolling)):
                        if abs(rolling[i] - rolling[i - 1]) > threshold:
                            triggered = True
                            break

                has_catastrophe = bool(r["catastrophes"])

                if triggered and has_catastrophe:
                    tp += 1
                elif triggered and not has_catastrophe:
                    fp += 1
                elif not triggered and has_catastrophe:
                    fn += 1
                else:
                    tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(
            f"Threshold P{p}×1.5 ({threshold:.4f}): "
            f"P={precision:.2f} R={recall:.2f} F1={f1:.2f} "
            f"(TP={tp} FP={fp} FN={fn} TN={tn})"
        )


def _rolling_entropy(sigs: list[dict], window: int) -> list[float]:
    """Compute rolling average entropy over a window."""
    entropies = [s["entropy"] for s in sigs]
    if len(entropies) < window:
        return entropies
    rolling = []
    for i in range(len(entropies) - window + 1):
        rolling.append(statistics.mean(entropies[i : i + window]))
    return rolling


# ---------------------------------------------------------------------------
# CLI-compatible entry points
# ---------------------------------------------------------------------------

def compare_runs(result_dir: Path = Path("results"), num_prompts: int | None = None) -> None:
    """Print a comparison table across all result files."""
    files = sorted(result_dir.rglob("*.json"))
    if not files:
        logger.warning(f"No result files found in {result_dir}")
        return

    rows = []
    for f in files:
        data = load_results(f)
        cfg = data["config"]
        if num_prompts and cfg["num_prompts"] != num_prompts:
            continue
        s = data["summary"]
        sig = signal_stats(data["results"])

        rows.append({
            "press": cfg["press_name"],
            "ratio": cfg["compression_ratio"],
            "n": s["total"],
            "accuracy": s.get("accuracy", 0),
            "cfr": s.get("catastrophic_failure_rate", 0),
            "non_term": s.get("catastrophe_counts", {}).get("non_termination", 0),
            "looping": s.get("catastrophe_counts", {}).get("looping", 0),
            "wrong": s.get("catastrophe_counts", {}).get("wrong_answer", 0),
            "mean_entropy": sig.get("mean_entropy", 0),
            "max_entropy": sig.get("max_entropy", 0),
            "max_abs_dh": sig.get("max_abs_delta_h", 0),
            "think_pct": sig.get("thinking_token_pct", 0),
            "min_top1": sig.get("min_top1", 0),
        })

    rows.sort(key=lambda r: (r["press"], r["ratio"]))

    header = (
        f"{'Press':<18s} {'Ratio':>5s} {'N':>3s} "
        f"{'Acc':>5s} {'CFR':>5s} {'NT':>3s} {'Loop':>4s} "
        f"{'maxH':>5s} {'maxDH':>6s} {'Think%':>6s} {'minP1':>6s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['press']:<18s} {r['ratio']:>5.3f} {r['n']:>3d} "
            f"{r['accuracy']:>5.1%} {r['cfr']:>5.1%} "
            f"{r['non_term']:>3d} {r['looping']:>4d} "
            f"{r.get('max_entropy', 0):>5.2f} "
            f"{r.get('max_abs_dh', 0):>6.2f} "
            f"{r.get('think_pct', 0):>6.1f} "
            f"{r.get('min_top1', 0):>6.3f}"
        )


def full_analysis(result_dir: Path = Path("results"), num_prompts: int = 50) -> None:
    """Run all analyses."""
    print("=" * 70)
    print("DEGRADATION TABLE")
    print("=" * 70)
    degradation_table(result_dir, num_prompts)

    print("\n" + "=" * 70)
    print("DELTA-H EARLY WARNING ANALYSIS")
    print("=" * 70)
    delta_h_analysis(result_dir, num_prompts)

    print("\n" + "=" * 70)
    print("SILENT FAILURE ANALYSIS")
    print("=" * 70)
    silent_failure_analysis(result_dir, num_prompts)

    print("\n" + "=" * 70)
    print("ROLLING DELTA-H DETECTOR (ERGO-style)")
    print("=" * 70)
    rolling_delta_h_detector(result_dir, num_prompts)


if __name__ == "__main__":
    full_analysis()
