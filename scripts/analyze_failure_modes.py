"""S1: Failure Mode Taxonomy — Logit Signature Analysis.

Loads all Qwen 7B sweep results, categorizes traces by failure mode,
extracts per-token signal trajectories, and performs statistical analysis
to determine whether failure modes have distinct, pre-manifestation
logit signatures.

Usage:
    uv run python scripts/analyze_failure_modes.py
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("docs/analysis")
MODEL_FILTER = "Qwen2.5-7B-Instruct"
NUM_PROMPTS = 500

# Compressors to analyze (exclude observed_attention — broken at all ratios)
COMPRESSORS = ["streaming_llm", "snapkv"]
RATIOS = ["0.250", "0.500", "0.625", "0.750", "0.875"]


@dataclass
class TraceSignals:
    """Per-token signal trajectories for a single trace."""

    prompt_id: str
    press: str
    compression_ratio: float
    catastrophes: list[str]
    onsets: dict[str, int]
    n_tokens: int
    correct: bool
    entropy: list[float] = field(default_factory=list)
    top1_prob: list[float] = field(default_factory=list)
    top5_prob: list[float] = field(default_factory=list)
    h_alts: list[float] = field(default_factory=list)
    delta_h: list[float | None] = field(default_factory=list)
    rep_count: list[int] = field(default_factory=list)
    avg_logp: list[float] = field(default_factory=list)

    @property
    def failure_mode(self) -> str:
        """Classify trace into one of 4 failure modes."""
        if "looping" in self.catastrophes:
            return "looping"
        if "non_termination" in self.catastrophes:
            return "non_termination"
        if "wrong_answer" in self.catastrophes:
            return "wrong_answer"
        return "correct"

    @property
    def onset_token(self) -> int | None:
        """Earliest catastrophe onset token."""
        onsets = []
        if "looping" in self.onsets and self.onsets["looping"] is not None:
            onsets.append(self.onsets["looping"])
        if "non_termination" in self.onsets and self.onsets["non_termination"] is not None:
            onsets.append(self.onsets["non_termination"])
        return min(onsets) if onsets else None


def load_traces() -> list[TraceSignals]:
    """Load all 7B traces from streaming_llm and snapkv."""
    traces = []
    for press in COMPRESSORS:
        for ratio in RATIOS:
            fn = RESULTS_DIR / press / f"{MODEL_FILTER}_{ratio}_{NUM_PROMPTS}p.json"
            if not fn.exists():
                print(f"  SKIP {fn} (not found)")
                continue
            with open(fn) as f:
                data = json.load(f)
            for r in data["results"]:
                ts = TraceSignals(
                    prompt_id=r["prompt_id"],
                    press=press,
                    compression_ratio=float(ratio),
                    catastrophes=r["catastrophes"],
                    onsets=r.get("catastrophe_onsets", {}),
                    n_tokens=r["num_tokens_generated"],
                    correct=r["correct"],
                )
                for sig in r["signals"]:
                    ts.entropy.append(sig["entropy"])
                    ts.top1_prob.append(sig["top1_prob"])
                    ts.top5_prob.append(sig["top5_prob"])
                    ts.h_alts.append(sig["h_alts"])
                    ts.delta_h.append(sig["delta_h"])
                    ts.rep_count.append(sig["rep_count"])
                    ts.avg_logp.append(sig["avg_logp"])
                traces.append(ts)
    return traces


def load_baseline_traces() -> list[TraceSignals]:
    """Load baseline (no compression) traces for reference."""
    fn = RESULTS_DIR / "none" / f"{MODEL_FILTER}_0.000_{NUM_PROMPTS}p.json"
    if not fn.exists():
        return []
    traces = []
    with open(fn) as f:
        data = json.load(f)
    for r in data["results"]:
        ts = TraceSignals(
            prompt_id=r["prompt_id"],
            press="none",
            compression_ratio=0.0,
            catastrophes=r["catastrophes"],
            onsets=r.get("catastrophe_onsets", {}),
            n_tokens=r["num_tokens_generated"],
            correct=r["correct"],
        )
        for sig in r["signals"]:
            ts.entropy.append(sig["entropy"])
            ts.top1_prob.append(sig["top1_prob"])
            ts.top5_prob.append(sig["top5_prob"])
            ts.h_alts.append(sig["h_alts"])
            ts.delta_h.append(sig["delta_h"])
            ts.rep_count.append(sig["rep_count"])
            ts.avg_logp.append(sig["avg_logp"])
        traces.append(ts)
    return traces


def compute_pre_onset_stats(
    traces: list[TraceSignals], window: int = 32
) -> dict[str, dict[str, list[float]]]:
    """Extract signal statistics from the pre-onset window for traces with onsets.

    For each trace with a known onset, extract signals from tokens
    [onset - window, onset) — the window just before catastrophe manifests.
    """
    mode_signals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for t in traces:
        onset = t.onset_token
        if onset is None:
            continue
        mode = t.failure_mode
        start = max(0, onset - window)
        end = onset
        if end <= start:
            continue
        # Collect signals from the pre-onset window
        mode_signals[mode]["entropy"].extend(t.entropy[start:end])
        mode_signals[mode]["top1_prob"].extend(t.top1_prob[start:end])
        mode_signals[mode]["top5_prob"].extend(t.top5_prob[start:end])
        mode_signals[mode]["h_alts"].extend(t.h_alts[start:end])
        mode_signals[mode]["avg_logp"].extend(t.avg_logp[start:end])
        # delta_h may have None values
        dh = [v for v in t.delta_h[start:end] if v is not None]
        mode_signals[mode]["delta_h"].extend(dh)
        mode_signals[mode]["rep_count"].extend([float(v) for v in t.rep_count[start:end]])
    return dict(mode_signals)


def compute_early_warning_stats(
    traces: list[TraceSignals],
) -> dict[str, dict[str, list[float]]]:
    """Extract signal statistics from the first 50 tokens of each trace.

    This captures the EARLY behavior before any catastrophe manifests,
    testing whether failure modes are distinguishable from the start.
    """
    mode_signals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for t in traces:
        mode = t.failure_mode
        n = min(50, len(t.entropy))
        mode_signals[mode]["entropy"].extend(t.entropy[:n])
        mode_signals[mode]["top1_prob"].extend(t.top1_prob[:n])
        mode_signals[mode]["top5_prob"].extend(t.top5_prob[:n])
        mode_signals[mode]["h_alts"].extend(t.h_alts[:n])
        mode_signals[mode]["avg_logp"].extend(t.avg_logp[:n])
        dh = [v for v in t.delta_h[:n] if v is not None]
        mode_signals[mode]["delta_h"].extend(dh)
        mode_signals[mode]["rep_count"].extend([float(v) for v in t.rep_count[:n]])
    return dict(mode_signals)


def compute_trajectory_stats(
    traces: list[TraceSignals],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute per-trace aggregate statistics, grouped by failure mode.

    Returns per-mode distributions of trace-level summaries (mean, max,
    std of each signal across each trace).
    """
    mode_aggs: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for t in traces:
        mode = t.failure_mode
        if len(t.entropy) == 0:
            continue
        ent = np.array(t.entropy)
        t1 = np.array(t.top1_prob)
        t5 = np.array(t.top5_prob)
        ha = np.array(t.h_alts)
        alp = np.array(t.avg_logp)
        dh = np.array([v for v in t.delta_h if v is not None])

        mode_aggs[mode]["mean_entropy"].append(float(np.mean(ent)))
        mode_aggs[mode]["max_entropy"].append(float(np.max(ent)))
        mode_aggs[mode]["std_entropy"].append(float(np.std(ent)))
        mode_aggs[mode]["mean_top1"].append(float(np.mean(t1)))
        mode_aggs[mode]["min_top1"].append(float(np.min(t1)))
        mode_aggs[mode]["mean_top5"].append(float(np.mean(t5)))
        mode_aggs[mode]["mean_h_alts"].append(float(np.mean(ha)))
        mode_aggs[mode]["max_h_alts"].append(float(np.max(ha)))
        mode_aggs[mode]["mean_avg_logp"].append(float(np.mean(alp)))
        if len(dh) > 0:
            mode_aggs[mode]["max_abs_delta_h"].append(float(np.max(np.abs(dh))))
            mode_aggs[mode]["mean_abs_delta_h"].append(float(np.mean(np.abs(dh))))
        mode_aggs[mode]["max_rep_count"].append(float(max(t.rep_count)))
        mode_aggs[mode]["n_tokens"].append(float(t.n_tokens))

    # Compute summary stats for each aggregate
    result: dict[str, dict[str, dict[str, float]]] = {}
    for mode, aggs in mode_aggs.items():
        result[mode] = {}
        for metric, values in aggs.items():
            arr = np.array(values)
            result[mode][metric] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "n": len(values),
            }
    return result


def statistical_tests(
    mode_signals: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Run pairwise Mann-Whitney U tests between failure modes for each signal.

    Returns U-statistic, p-value, and effect size (rank-biserial correlation)
    for each pair of modes for each signal.
    """
    modes = sorted(mode_signals.keys())
    signals = ["entropy", "top1_prob", "h_alts", "delta_h", "avg_logp"]
    results: dict[str, dict[str, dict[str, float]]] = {}

    for sig in signals:
        results[sig] = {}
        for i, m1 in enumerate(modes):
            for m2 in modes[i + 1 :]:
                v1 = mode_signals.get(m1, {}).get(sig, [])
                v2 = mode_signals.get(m2, {}).get(sig, [])
                if len(v1) < 10 or len(v2) < 10:
                    continue
                a1, a2 = np.array(v1), np.array(v2)
                u_stat, p_val = scipy_stats.mannwhitneyu(a1, a2, alternative="two-sided")
                # Rank-biserial correlation as effect size
                n1, n2 = len(a1), len(a2)
                rbc = 1 - (2 * u_stat) / (n1 * n2)
                results[sig][f"{m1}_vs_{m2}"] = {
                    "U": float(u_stat),
                    "p": float(p_val),
                    "effect_size_rbc": float(rbc),
                    "n1": n1,
                    "n2": n2,
                    "mean1": float(np.mean(a1)),
                    "mean2": float(np.mean(a2)),
                }
    return results


def compute_lead_times(traces: list[TraceSignals]) -> dict[str, list[int]]:
    """For each trace with an onset, compute how many tokens before onset
    key signal thresholds were crossed.

    Uses entropy > baseline_p95 and top1_prob < baseline_p5 as thresholds.
    """
    # First pass: compute baseline (correct traces) signal distributions
    correct_entropies = []
    correct_top1 = []
    for t in traces:
        if t.failure_mode == "correct":
            correct_entropies.extend(t.entropy)
            correct_top1.extend(t.top1_prob)

    if not correct_entropies:
        return {}

    ent_p95 = float(np.percentile(correct_entropies, 95))
    top1_p5 = float(np.percentile(correct_top1, 5))

    lead_times: dict[str, list[int]] = defaultdict(list)

    for t in traces:
        onset = t.onset_token
        if onset is None or onset < 5:
            continue
        mode = t.failure_mode

        # Find earliest token before onset where entropy exceeds threshold
        earliest_ent = None
        for i in range(onset):
            if t.entropy[i] > ent_p95:
                earliest_ent = onset - i
                break

        # Find earliest token before onset where top1_prob drops below threshold
        earliest_top1 = None
        for i in range(onset):
            if t.top1_prob[i] < top1_p5:
                earliest_top1 = onset - i
                break

        if earliest_ent is not None:
            lead_times[f"{mode}_entropy"].append(earliest_ent)
        if earliest_top1 is not None:
            lead_times[f"{mode}_top1_prob"].append(earliest_top1)

    return dict(lead_times)


def format_report(
    traces: list[TraceSignals],
    baseline_traces: list[TraceSignals],
    mode_counts: dict[str, int],
    trajectory_stats: dict[str, dict[str, dict[str, float]]],
    pre_onset_tests: dict[str, dict[str, dict[str, float]]],
    early_tests: dict[str, dict[str, dict[str, float]]],
    pre_onset_signals: dict[str, dict[str, list[float]]],
    early_signals: dict[str, dict[str, list[float]]],
    lead_times: dict[str, list[int]],
) -> str:
    """Format the analysis results as a markdown report."""
    lines = []
    lines.append("# S1: Failure Mode Taxonomy — Logit Signature Analysis")
    lines.append("")
    lines.append("**Generated:** 2026-02-24")
    lines.append(
        f"**Data:** Qwen2.5-7B-Instruct, {NUM_PROMPTS} prompts, "
        f"compressors: {', '.join(COMPRESSORS)}"
    )
    lines.append(f"**Compression ratios:** {', '.join(RATIOS)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 1: Trace distribution
    lines.append("## 1. Trace Distribution by Failure Mode")
    lines.append("")
    total = sum(mode_counts.values())
    lines.append(f"Total traces analyzed: {total}")
    lines.append("")
    lines.append("| Failure Mode | Count | Fraction |")
    lines.append("|---|---|---|")
    for mode in ["looping", "non_termination", "wrong_answer", "correct"]:
        c = mode_counts.get(mode, 0)
        lines.append(f"| {mode} | {c} | {c / total:.1%} |")
    lines.append("")

    # Per-config breakdown
    lines.append("### Per-config breakdown")
    lines.append("")
    lines.append("| Press | Ratio | Looping | Non-term | Wrong Answer | Correct |")
    lines.append("|---|---|---|---|---|---|")
    config_counts: dict[tuple[str, float], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t in traces:
        config_counts[(t.press, t.compression_ratio)][t.failure_mode] += 1
    for press in COMPRESSORS:
        for ratio in RATIOS:
            key = (press, float(ratio))
            cc = config_counts.get(key, {})
            lines.append(
                f"| {press} | {ratio} | {cc.get('looping', 0)} | "
                f"{cc.get('non_termination', 0)} | "
                f"{cc.get('wrong_answer', 0)} | {cc.get('correct', 0)} |"
            )
    lines.append("")

    # Section 2: Trajectory statistics
    lines.append("## 2. Per-Trace Signal Statistics by Failure Mode")
    lines.append("")
    lines.append(
        "Each row shows the **distribution of per-trace aggregate statistics** "
        "across all traces of that failure mode. For example, 'mean_entropy' shows "
        "the distribution of each trace's mean entropy value."
    )
    lines.append("")

    key_metrics = [
        "mean_entropy",
        "max_entropy",
        "std_entropy",
        "mean_top1",
        "min_top1",
        "max_abs_delta_h",
        "max_rep_count",
        "n_tokens",
    ]
    for metric in key_metrics:
        lines.append(f"### {metric}")
        lines.append("")
        lines.append("| Mode | Mean | Median | Std | P25 | P75 | N |")
        lines.append("|---|---|---|---|---|---|---|")
        for mode in ["looping", "non_termination", "wrong_answer", "correct"]:
            if mode in trajectory_stats and metric in trajectory_stats[mode]:
                s = trajectory_stats[mode][metric]
                lines.append(
                    f"| {mode} | {s['mean']:.4f} | {s['median']:.4f} | "
                    f"{s['std']:.4f} | {s['p25']:.4f} | {s['p75']:.4f} | "
                    f"{int(s['n'])} |"
                )
        lines.append("")

    # Section 3: Pre-onset window analysis
    lines.append("## 3. Pre-Onset Signal Analysis (32 tokens before catastrophe)")
    lines.append("")
    lines.append(
        "Signals extracted from the 32-token window immediately before "
        "catastrophe onset. This is the prediction target — can we see "
        "it coming in this window?"
    )
    lines.append("")

    for sig in ["entropy", "top1_prob", "h_alts", "avg_logp"]:
        lines.append(f"### {sig} (pre-onset window)")
        lines.append("")
        lines.append("| Mode | Mean | Std | N tokens |")
        lines.append("|---|---|---|---|")
        for mode in ["looping", "non_termination"]:
            vals = pre_onset_signals.get(mode, {}).get(sig, [])
            if vals:
                arr = np.array(vals)
                lines.append(f"| {mode} | {np.mean(arr):.4f} | {np.std(arr):.4f} | {len(arr)} |")
        # Add correct traces for comparison
        correct_vals = []
        for bt in baseline_traces:
            if bt.failure_mode == "correct":
                correct_vals.extend(getattr(bt, sig, []))
        if not correct_vals:
            for t in traces:
                if t.failure_mode == "correct":
                    vals_list = {
                        "entropy": t.entropy,
                        "top1_prob": t.top1_prob,
                        "h_alts": t.h_alts,
                        "avg_logp": t.avg_logp,
                    }
                    correct_vals.extend(vals_list[sig][:50])
        if correct_vals:
            arr = np.array(correct_vals[:10000])  # cap for display
            lines.append(f"| correct (ref) | {np.mean(arr):.4f} | {np.std(arr):.4f} | {len(arr)} |")
        lines.append("")

    # Section 4: Statistical tests
    lines.append("## 4. Statistical Distinguishability")
    lines.append("")
    lines.append(
        "Mann-Whitney U tests between failure modes. Effect size is "
        "rank-biserial correlation (|rbc| > 0.3 = medium, > 0.5 = large)."
    )
    lines.append("")

    lines.append("### Pre-onset window (32 tokens before catastrophe)")
    lines.append("")
    lines.append("| Signal | Comparison | Mean₁ | Mean₂ | Effect Size | p-value |")
    lines.append("|---|---|---|---|---|---|")
    for sig in ["entropy", "top1_prob", "h_alts", "avg_logp"]:
        if sig in pre_onset_tests:
            for comp, vals in pre_onset_tests[sig].items():
                p_str = f"{vals['p']:.2e}" if vals["p"] < 0.001 else f"{vals['p']:.4f}"
                lines.append(
                    f"| {sig} | {comp} | {vals['mean1']:.4f} | "
                    f"{vals['mean2']:.4f} | {vals['effect_size_rbc']:.3f} | "
                    f"{p_str} |"
                )
    lines.append("")

    lines.append("### Early tokens (first 50 tokens)")
    lines.append("")
    lines.append("| Signal | Comparison | Mean₁ | Mean₂ | Effect Size | p-value |")
    lines.append("|---|---|---|---|---|---|")
    for sig in ["entropy", "top1_prob", "h_alts", "avg_logp"]:
        if sig in early_tests:
            for comp, vals in early_tests[sig].items():
                p_str = f"{vals['p']:.2e}" if vals["p"] < 0.001 else f"{vals['p']:.4f}"
                lines.append(
                    f"| {sig} | {comp} | {vals['mean1']:.4f} | "
                    f"{vals['mean2']:.4f} | {vals['effect_size_rbc']:.3f} | "
                    f"{p_str} |"
                )
    lines.append("")

    # Section 5: Lead times
    lines.append("## 5. Lead Time Analysis")
    lines.append("")
    lines.append(
        "How many tokens before catastrophe onset do signal thresholds get crossed? "
        "Entropy threshold = P95 of correct traces. Top-1 prob threshold = P5 of correct traces."
    )
    lines.append("")
    lines.append("| Mode × Signal | Mean | Median | P25 | P75 | N traces |")
    lines.append("|---|---|---|---|---|---|")
    for key in sorted(lead_times.keys()):
        vals = lead_times[key]
        if len(vals) < 5:
            continue
        arr = np.array(vals)
        lines.append(
            f"| {key} | {np.mean(arr):.1f} | {np.median(arr):.1f} | "
            f"{np.percentile(arr, 25):.1f} | {np.percentile(arr, 75):.1f} | "
            f"{len(arr)} |"
        )
    lines.append("")

    # Section 6: Key findings
    lines.append("## 6. Key Findings")
    lines.append("")

    # Determine if looping vs non-termination are distinguishable
    ent_test = pre_onset_tests.get("entropy", {}).get("looping_vs_non_termination", {})
    if ent_test:
        es = abs(ent_test.get("effect_size_rbc", 0))
        strength = "large" if es > 0.5 else "medium" if es > 0.3 else "small"
        lines.append(
            f"1. **Looping vs. non-termination pre-onset entropy**: "
            f"effect size = {es:.3f} ({strength}). "
            f"Looping mean = {ent_test.get('mean1', 0):.4f}, "
            f"non-termination mean = {ent_test.get('mean2', 0):.4f}."
        )
    else:
        lines.append("1. **Looping vs. non-termination**: insufficient data for comparison.")
    lines.append("")

    # Top-1 probability analysis
    t1_test = pre_onset_tests.get("top1_prob", {}).get("looping_vs_non_termination", {})
    if t1_test:
        es = abs(t1_test.get("effect_size_rbc", 0))
        strength = "large" if es > 0.5 else "medium" if es > 0.3 else "small"
        lines.append(
            f"2. **Looping vs. non-termination pre-onset top-1 prob**: "
            f"effect size = {es:.3f} ({strength}). "
            f"Looping mean = {t1_test.get('mean1', 0):.4f}, "
            f"non-termination mean = {t1_test.get('mean2', 0):.4f}."
        )
    lines.append("")

    # Lead time summary
    loop_lt = lead_times.get("looping_entropy", [])
    nt_lt = lead_times.get("non_termination_entropy", [])
    if loop_lt:
        lines.append(
            f"3. **Looping lead time (entropy)**: median {np.median(loop_lt):.0f} "
            f"tokens before onset (N={len(loop_lt)} traces)."
        )
    if nt_lt:
        lines.append(
            f"4. **Non-termination lead time (entropy)**: median "
            f"{np.median(nt_lt):.0f} tokens before onset (N={len(nt_lt)} traces)."
        )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    print("Loading traces...")
    traces = load_traces()
    baseline = load_baseline_traces()
    print(f"  Loaded {len(traces)} compressed traces, {len(baseline)} baseline traces")

    # Count by failure mode
    mode_counts: dict[str, int] = defaultdict(int)
    for t in traces:
        mode_counts[t.failure_mode] += 1
    print("\nFailure mode distribution:")
    for mode, count in sorted(mode_counts.items()):
        print(f"  {mode}: {count}")

    # Compute trajectory statistics
    print("\nComputing trajectory statistics...")
    traj_stats = compute_trajectory_stats(traces)

    # Compute pre-onset window signals
    print("Computing pre-onset window signals...")
    pre_onset_signals = compute_pre_onset_stats(traces, window=32)
    for mode, sigs in pre_onset_signals.items():
        n_tokens = len(sigs.get("entropy", []))
        print(f"  {mode}: {n_tokens} pre-onset tokens")

    # Compute early-warning signals
    print("Computing early-warning signals (first 50 tokens)...")
    early_signals = compute_early_warning_stats(traces)

    # Statistical tests
    print("Running statistical tests...")
    pre_onset_tests = statistical_tests(pre_onset_signals)
    early_tests = statistical_tests(early_signals)

    # Lead time analysis
    print("Computing lead times...")
    lead_times = compute_lead_times(traces)

    # Generate report
    print("\nGenerating report...")
    report = format_report(
        traces=traces,
        baseline_traces=baseline,
        mode_counts=dict(mode_counts),
        trajectory_stats=traj_stats,
        pre_onset_tests=pre_onset_tests,
        early_tests=early_tests,
        pre_onset_signals=pre_onset_signals,
        early_signals=early_signals,
        lead_times=lead_times,
    )

    # Save report
    output_path = OUTPUT_DIR / "003-failure-mode-taxonomy.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {output_path}")

    # Save raw statistics as JSON for further analysis
    json_path = Path("models") / "failure_mode_analysis.json"
    json_data = {
        "metadata": {
            "generated_by": "scripts/analyze_failure_modes.py",
            "model": MODEL_FILTER,
            "num_prompts": NUM_PROMPTS,
            "compressors": COMPRESSORS,
            "ratios": RATIOS,
        },
        "mode_counts": dict(mode_counts),
        "trajectory_stats": traj_stats,
        "lead_times": {
            k: {
                "mean": float(np.mean(v)),
                "median": float(np.median(v)),
                "std": float(np.std(v)),
                "n": len(v),
            }
            for k, v in lead_times.items()
            if len(v) >= 5
        },
        "pre_onset_tests": pre_onset_tests,
        "early_tests": early_tests,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Raw data saved to {json_path}")


if __name__ == "__main__":
    main()
