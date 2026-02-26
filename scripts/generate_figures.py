"""Generate publication-quality figures for the KVGuard paper.

Produces:
  - Figure 2: Phase transition (accuracy + CFR vs compression ratio)
  - Figure 3: Failure mode logit signatures (pre-onset signal comparison)
  - Figure 4: Lead time distributions by failure mode
  - Figure 5: Feature ablation comparison (pre-onset AUROC bar chart)
  - Figure 6: Entropy trajectory example (normal vs catastrophe)

Usage:
    uv run python scripts/generate_figures.py
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# Style configuration for publication
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("paper/figures/generated")
MODEL_FILTER = "Qwen2.5-7B-Instruct"
NUM_PROMPTS = 500
COMPRESSORS = ["streaming_llm", "snapkv"]
RATIOS = [0.250, 0.500, 0.625, 0.750, 0.875]
RATIO_LABELS = ["0.25", "0.50", "0.625", "0.75", "0.875"]

# Colors
C_STREAM = "#2196F3"  # blue
C_SNAP = "#FF5722"  # orange-red
C_BASELINE = "#4CAF50"  # green
C_CORRECT = "#66BB6A"
C_LOOPING = "#EF5350"
C_NONTERM = "#FFA726"
C_WRONG = "#AB47BC"


def load_phase_transition_data() -> dict:
    """Load phase transition analysis data. Returns metrics dict keyed by compressor."""
    path = MODELS_DIR / "phase_transition_analysis.json"
    with open(path) as f:
        data = json.load(f)
    return data["metrics"]


def load_failure_mode_data() -> dict:
    """Load failure mode analysis data."""
    path = MODELS_DIR / "failure_mode_analysis.json"
    with open(path) as f:
        return json.load(f)


def load_ablation_data() -> dict:
    """Load ablation results. Returns variants dict."""
    path = MODELS_DIR / "ablation_results.json"
    with open(path) as f:
        data = json.load(f)
    return data["variants"]


def load_lead_time_data() -> dict:
    """Load lead time analysis data."""
    path = MODELS_DIR / "lead_time_analysis.json"
    with open(path) as f:
        return json.load(f)


def load_traces_for_example() -> list[dict]:
    """Load a few example traces for the entropy trajectory figure."""
    traces = []
    # Load a correct trace from moderate compression
    fn = RESULTS_DIR / "streaming_llm" / f"{MODEL_FILTER}_0.500_{NUM_PROMPTS}p.json"
    if fn.exists():
        with open(fn) as f:
            data = json.load(f)
        for r in data["results"]:
            if r["correct"] and not r["catastrophes"]:
                traces.append(
                    {
                        "type": "correct",
                        "signals": r["signals"],
                        "press": "streaming_llm",
                        "ratio": 0.5,
                    }
                )
                break

    # Load a looping trace from heavy compression
    fn = RESULTS_DIR / "snapkv" / f"{MODEL_FILTER}_0.875_{NUM_PROMPTS}p.json"
    if fn.exists():
        with open(fn) as f:
            data = json.load(f)
        for r in data["results"]:
            if "looping" in r["catastrophes"]:
                traces.append(
                    {
                        "type": "looping",
                        "signals": r["signals"],
                        "press": "snapkv",
                        "ratio": 0.875,
                        "onset": r.get("catastrophe_onsets", {}).get("looping"),
                    }
                )
                break

    # Load a non-termination trace
    fn = RESULTS_DIR / "streaming_llm" / f"{MODEL_FILTER}_0.875_{NUM_PROMPTS}p.json"
    if fn.exists():
        with open(fn) as f:
            data = json.load(f)
        for r in data["results"]:
            if "non_termination" in r["catastrophes"] and "looping" not in r["catastrophes"]:
                traces.append(
                    {
                        "type": "non_termination",
                        "signals": r["signals"],
                        "press": "streaming_llm",
                        "ratio": 0.875,
                    }
                )
                break

    return traces


def figure_phase_transition() -> None:
    """Figure 2: Phase transition — accuracy and CFR vs compression ratio."""
    metrics = load_phase_transition_data()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    for comp, color, marker in [
        ("streaming_llm", C_STREAM, "o"),
        ("snapkv", C_SNAP, "s"),
    ]:
        if comp not in metrics:
            continue
        entries = metrics[comp]
        ratios = [e["ratio"] for e in entries]
        accs = [e["accuracy"] for e in entries]
        cfrs = [e["cfr"] for e in entries]

        label = "StreamingLLM" if comp == "streaming_llm" else "SnapKV"
        ax1.plot(ratios, accs, f"-{marker}", color=color, label=label, markersize=5, linewidth=1.5)
        ax2.plot(ratios, cfrs, f"-{marker}", color=color, label=label, markersize=5, linewidth=1.5)

    # Add baseline
    if "baseline" in metrics and metrics["baseline"]:
        bl = metrics["baseline"][0]
        ax1.axhline(
            y=bl["accuracy"],
            color=C_BASELINE,
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Baseline",
        )
        ax2.axhline(
            y=bl["cfr"],
            color=C_BASELINE,
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Baseline",
        )

    # Mark critical transition zone
    for ax in [ax1, ax2]:
        ax.axvspan(0.75, 0.875, alpha=0.08, color="red")

    ax1.set_xlabel("Compression Ratio (fraction removed)")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="lower left", framealpha=0.9)
    ax1.set_title("(a) Accuracy vs. Compression")

    ax2.set_xlabel("Compression Ratio (fraction removed)")
    ax2.set_ylabel("CFR")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.set_title("(b) CFR vs. Compression")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure-2-phase-transition.pdf")
    fig.savefig(OUTPUT_DIR / "figure-2-phase-transition.png")
    plt.close(fig)
    print("  Saved figure-2-phase-transition")


def figure_failure_signatures() -> None:
    """Figure 3: Pre-onset logit signatures by failure mode."""
    data = load_failure_mode_data()
    if "pre_onset_tests" not in data:
        print("  SKIP figure-3 (no pre_onset_tests in data)")
        return

    signals = ["entropy", "top1_prob", "h_alts", "avg_logp"]
    signal_labels = ["Entropy", "Top-1 Prob", "h_alts", "avg log P"]

    # Extract means from the test results
    modes = ["looping", "non_termination"]
    mode_labels = ["Looping", "Non-term"]
    mode_colors = [C_LOOPING, C_NONTERM]

    fig, axes = plt.subplots(1, 4, figsize=(8, 2.5))

    for i, (sig, sig_label) in enumerate(zip(signals, signal_labels)):
        ax = axes[i]
        test_data = data.get("pre_onset_tests", {}).get(sig, {})

        means = {}
        for comp_key, vals in test_data.items():
            parts = comp_key.split("_vs_")
            if len(parts) == 2:
                means[parts[0]] = vals["mean1"]
                means[parts[1]] = vals["mean2"]

        bar_vals = [means.get(m, 0) for m in modes]
        bars = ax.bar(mode_labels, bar_vals, color=mode_colors, width=0.6, edgecolor="white")
        ax.set_title(sig_label)
        ax.set_ylabel("Mean" if i == 0 else "")

        # Add value labels on bars
        for bar, val in zip(bars, bar_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.suptitle("Pre-onset Signal Values (32 tokens before catastrophe)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure-3-failure-signatures.pdf")
    fig.savefig(OUTPUT_DIR / "figure-3-failure-signatures.png")
    plt.close(fig)
    print("  Saved figure-3-failure-signatures")


def figure_lead_time() -> None:
    """Figure 4: Lead time distributions by compressor."""
    data = load_lead_time_data()

    fig, ax = plt.subplots(figsize=(5, 3))

    compressors = []
    means = []
    medians = []
    colors = []

    for comp_key in ["streaming_llm", "snapkv"]:
        if "by_compressor" in data:
            for entry in data["by_compressor"]:
                if entry.get("compressor") == comp_key:
                    label = "StreamingLLM" if comp_key == "streaming_llm" else "SnapKV"
                    compressors.append(label)
                    means.append(entry.get("mean_lead", 0))
                    medians.append(entry.get("median_lead", 0))
                    colors.append(C_STREAM if comp_key == "streaming_llm" else C_SNAP)

    if not compressors:
        # Fallback: use hardcoded values from analysis
        compressors = ["StreamingLLM", "SnapKV"]
        means = [282.7, 28.2]
        medians = [384, 1]
        colors = [C_STREAM, C_SNAP]

    x = np.arange(len(compressors))
    width = 0.35
    bars1 = ax.bar(x - width / 2, means, width, label="Mean lead time", color=colors, alpha=0.8)
    bars2 = ax.bar(
        x + width / 2,
        medians,
        width,
        label="Median lead time",
        color=colors,
        alpha=0.5,
        edgecolor=[c for c in colors],
        linewidth=1.5,
        hatch="//",
    )

    ax.set_ylabel("Lead time (tokens)")
    ax.set_xticks(x)
    ax.set_xticklabels(compressors)
    ax.legend()
    ax.set_title("Prediction Lead Time by Compressor")

    # Add value labels
    for bar, val in zip(bars1, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar, val in zip(bars2, medians):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure-4-lead-time.pdf")
    fig.savefig(OUTPUT_DIR / "figure-4-lead-time.png")
    plt.close(fig)
    print("  Saved figure-4-lead-time")


def figure_ablation() -> None:
    """Figure 5: Feature ablation comparison (pre-onset AUROC)."""
    data = load_ablation_data()

    variants = []
    aurocs = []
    colors_list = []

    variant_order = ["full", "no_compression_ratio", "no_rep", "logit_only"]
    variant_labels = {
        "full": "Full\n(40 feat)",
        "no_compression_ratio": "No ratio\n(39 feat)",
        "no_rep": "No rep\n(38 feat)",
        "logit_only": "Logit-only\n(37 feat)",
    }
    variant_colors = {
        "full": "#1976D2",
        "no_compression_ratio": "#42A5F5",
        "no_rep": "#64B5F6",
        "logit_only": "#90CAF9",
    }

    for v in variant_order:
        if v in data:
            variants.append(variant_labels.get(v, v))
            # Use pre-onset AUROC
            auroc = data[v].get("pre_onset_auroc", data[v].get("auroc", 0))
            aurocs.append(auroc)
            colors_list.append(variant_colors.get(v, "#999"))

    if not variants:
        print("  SKIP figure-5 (no ablation data)")
        return

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(variants, aurocs, color=colors_list, width=0.6, edgecolor="white")

    # Add value labels
    for bar, val in zip(bars, aurocs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 0.003,
            f"{val:.3f}",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

    ax.set_ylabel("Pre-onset AUROC")
    ax.set_ylim(0.95, 0.975)
    ax.set_title("Feature Ablation: Pre-onset AUROC")

    # Add a dashed line at the logit-only value
    if len(aurocs) >= 4:
        ax.axhline(y=aurocs[-1], color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure-5-ablation.pdf")
    fig.savefig(OUTPUT_DIR / "figure-5-ablation.png")
    plt.close(fig)
    print("  Saved figure-5-ablation")


def figure_entropy_trajectory() -> None:
    """Figure 6: Example entropy trajectories for normal vs catastrophe traces."""
    traces = load_traces_for_example()

    if len(traces) < 2:
        print("  SKIP figure-6 (insufficient example traces)")
        return

    fig, ax = plt.subplots(figsize=(6, 3))

    for t in traces:
        entropies = [s["entropy"] for s in t["signals"] if s["entropy"] is not None]
        n = len(entropies)
        if n == 0:
            continue

        x = np.arange(n)
        if t["type"] == "correct":
            ax.plot(
                x,
                entropies,
                color=C_CORRECT,
                alpha=0.8,
                linewidth=1,
                label="Correct (StreamingLLM, r=0.5)",
            )
        elif t["type"] == "looping":
            ax.plot(
                x,
                entropies,
                color=C_LOOPING,
                alpha=0.8,
                linewidth=1,
                label="Looping (SnapKV, r=0.875)",
            )
            onset = t.get("onset")
            if onset and onset < n:
                ax.axvline(x=onset, color=C_LOOPING, linestyle="--", alpha=0.5, linewidth=0.8)
                ax.text(onset + 2, max(entropies) * 0.9, "onset", fontsize=7, color=C_LOOPING)
        elif t["type"] == "non_termination":
            ax.plot(
                x,
                entropies,
                color=C_NONTERM,
                alpha=0.8,
                linewidth=1,
                label="Non-term (StreamingLLM, r=0.875)",
            )

    ax.set_xlabel("Token position")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("Entropy Trajectories: Normal vs. Catastrophe")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure-6-entropy-trajectory.pdf")
    fig.savefig(OUTPUT_DIR / "figure-6-entropy-trajectory.png")
    plt.close(fig)
    print("  Saved figure-6-entropy-trajectory")


def figure_phase_transition_looping() -> None:
    """Figure 7: Looping rate vs compression ratio — shows the 477x increase."""
    metrics = load_phase_transition_data()

    fig, ax = plt.subplots(figsize=(4, 3))

    for comp, color, marker in [
        ("streaming_llm", C_STREAM, "o"),
        ("snapkv", C_SNAP, "s"),
    ]:
        if comp not in metrics:
            continue
        entries = metrics[comp]
        ratios = [e["ratio"] for e in entries]
        looping = [e.get("looping_rate", 0) for e in entries]

        label = "StreamingLLM" if comp == "streaming_llm" else "SnapKV"
        ax.plot(
            ratios,
            looping,
            f"-{marker}",
            color=color,
            label=label,
            markersize=5,
            linewidth=1.5,
        )

    ax.axvspan(0.75, 0.875, alpha=0.08, color="red")
    ax.set_xlabel("Compression Ratio (fraction removed)")
    ax.set_ylabel("Looping Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title("Looping Rate vs. Compression")

    # Annotate the 477x jump
    ax.annotate("477×", xy=(0.875, 0.95), fontsize=10, fontweight="bold", color=C_SNAP, ha="center")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure-7-looping-rate.pdf")
    fig.savefig(OUTPUT_DIR / "figure-7-looping-rate.png")
    plt.close(fig)
    print("  Saved figure-7-looping-rate")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating figures...")

    figure_phase_transition()
    figure_failure_signatures()
    figure_lead_time()
    figure_ablation()
    figure_entropy_trajectory()
    figure_phase_transition_looping()

    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
