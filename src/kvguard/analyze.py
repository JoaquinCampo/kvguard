"""Analyze experiment results and per-token signals."""

import json
import statistics
from pathlib import Path

from loguru import logger


def load_results(path: Path) -> dict:
    """Load a results JSON file."""
    return json.loads(path.read_text())


def signal_stats(results: list[dict]) -> dict:
    """Compute aggregate signal statistics across all results."""
    all_entropy = []
    all_top1 = []
    max_entropies = []
    min_top1s = []

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

    if not all_entropy:
        return {}

    return {
        "mean_entropy": round(statistics.mean(all_entropy), 3),
        "median_max_entropy": round(statistics.median(max_entropies), 3),
        "max_entropy": round(max(max_entropies), 3),
        "mean_top1": round(statistics.mean(all_top1), 3),
        "median_min_top1": round(statistics.median(min_top1s), 3),
        "min_top1": round(min(min_top1s), 3),
    }


def compare_runs(result_dir: Path = Path("results")) -> None:
    """Print a comparison table across all result files."""
    files = sorted(result_dir.rglob("*.json"))
    if not files:
        logger.warning(f"No result files found in {result_dir}")
        return

    rows = []
    for f in files:
        data = load_results(f)
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
            "mean_entropy": sig.get("mean_entropy", 0),
            "max_entropy": sig.get("max_entropy", 0),
            "min_top1": sig.get("min_top1", 0),
        })

    # Print table
    header = (
        f"{'Press':<16s} {'Ratio':>5s} {'N':>3s} "
        f"{'Acc':>5s} {'CFR':>5s} {'NT':>3s} {'Loop':>4s} "
        f"{'μH':>5s} {'maxH':>5s} {'minP1':>6s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['press']:<16s} {r['ratio']:>5.3f} {r['n']:>3d} "
            f"{r['accuracy']:>5.1%} {r['cfr']:>5.1%} "
            f"{r['non_term']:>3d} {r['looping']:>4d} "
            f"{r['mean_entropy']:>5.2f} {r['max_entropy']:>5.2f} "
            f"{r['min_top1']:>6.3f}"
        )


if __name__ == "__main__":
    compare_runs()
