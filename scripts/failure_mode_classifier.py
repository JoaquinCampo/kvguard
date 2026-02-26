"""S1 Part 2: Failure Mode Classifier.

Trains a classifier to distinguish between failure modes (looping,
non-termination, wrong_answer, correct) using per-token logit features.

Tests two key questions:
1. Can we classify failure mode from tokens in the pre-onset window?
2. Can we classify failure mode from EARLY tokens (first 50)?

Usage:
    uv run python scripts/failure_mode_classifier.py
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

RESULTS_DIR = Path("results")
MODEL_FILTER = "Qwen2.5-7B-Instruct"
NUM_PROMPTS = 500
COMPRESSORS = ["streaming_llm", "snapkv"]
RATIOS = ["0.250", "0.500", "0.625", "0.750", "0.875"]

# Features to use for classification (per-token)
SIGNAL_KEYS = [
    "entropy",
    "top1_prob",
    "top5_prob",
    "h_alts",
    "avg_logp",
    "rep_count",
]


@dataclass
class TraceData:
    prompt_id: str
    press: str
    compression_ratio: float
    catastrophes: list[str]
    onsets: dict[str, int]
    n_tokens: int
    correct: bool
    signals: list[dict]

    @property
    def failure_mode(self) -> str:
        if "looping" in self.catastrophes:
            return "looping"
        if "non_termination" in self.catastrophes:
            return "non_termination"
        if "wrong_answer" in self.catastrophes:
            return "wrong_answer"
        return "correct"

    @property
    def onset_token(self) -> int | None:
        onsets = []
        if "looping" in self.onsets and self.onsets["looping"] is not None:
            onsets.append(self.onsets["looping"])
        if "non_termination" in self.onsets and self.onsets["non_termination"] is not None:
            onsets.append(self.onsets["non_termination"])
        return min(onsets) if onsets else None


def load_traces() -> list[TraceData]:
    traces = []
    for press in COMPRESSORS:
        for ratio in RATIOS:
            fn = RESULTS_DIR / press / f"{MODEL_FILTER}_{ratio}_{NUM_PROMPTS}p.json"
            if not fn.exists():
                continue
            with open(fn) as f:
                data = json.load(f)
            for r in data["results"]:
                traces.append(
                    TraceData(
                        prompt_id=r["prompt_id"],
                        press=press,
                        compression_ratio=float(ratio),
                        catastrophes=r["catastrophes"],
                        onsets=r.get("catastrophe_onsets", {}),
                        n_tokens=r["num_tokens_generated"],
                        correct=r["correct"],
                        signals=r["signals"],
                    )
                )
    return traces


def extract_window_features(signals: list[dict], start: int, end: int) -> np.ndarray:
    """Extract aggregate features from a window of tokens.

    Returns a feature vector with:
    - mean, std, min, max, last for each signal
    - trend (slope) for entropy and top1_prob
    """
    if end <= start or len(signals) == 0:
        return np.zeros(len(SIGNAL_KEYS) * 5 + 4)

    window = signals[start:end]
    features = []

    for key in SIGNAL_KEYS:
        vals = []
        for s in window:
            v = s.get(key)
            if v is not None:
                vals.append(float(v))
        if vals:
            arr = np.array(vals)
            features.extend(
                [
                    np.mean(arr),
                    np.std(arr),
                    np.min(arr),
                    np.max(arr),
                    arr[-1],  # last value
                ]
            )
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    # Trend features: linear slope of entropy and top1_prob
    for key in ["entropy", "top1_prob"]:
        vals = [float(s[key]) for s in window if s.get(key) is not None]
        if len(vals) > 2:
            x = np.arange(len(vals))
            slope = np.polyfit(x, vals, 1)[0]
            features.append(float(slope))
        else:
            features.append(0.0)

    # Delta-H features: max absolute and mean absolute
    dh_vals = [float(s["delta_h"]) for s in window if s.get("delta_h") is not None]
    if dh_vals:
        dh_arr = np.abs(np.array(dh_vals))
        features.extend([float(np.max(dh_arr)), float(np.mean(dh_arr))])
    else:
        features.extend([0.0, 0.0])

    return np.array(features)


def build_classification_dataset(
    traces: list[TraceData], mode: str = "pre_onset", window: int = 32
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a dataset for failure-mode classification.

    mode:
      - "pre_onset": use 32 tokens before onset (only for traces with onsets)
      - "early": use first 50 tokens (all traces)
      - "trace_agg": use full trace aggregate features

    Returns: (X, y_labels, prompt_ids) where prompt_ids is used for grouped CV.
    """
    X_list = []
    y_list: list[str] = []
    pid_list: list[str] = []

    for t in traces:
        fm = t.failure_mode

        if mode == "pre_onset":
            onset = t.onset_token
            if onset is None:
                continue
            start = max(0, onset - window)
            end = onset
            if end <= start:
                continue
            feats = extract_window_features(t.signals, start, end)

        elif mode == "early":
            n = min(50, len(t.signals))
            if n < 10:
                continue
            feats = extract_window_features(t.signals, 0, n)

        elif mode == "trace_agg":
            if len(t.signals) < 10:
                continue
            feats = extract_window_features(t.signals, 0, len(t.signals))

        else:
            raise ValueError(f"Unknown mode: {mode}")

        X_list.append(feats)
        y_list.append(fm)
        pid_list.append(t.prompt_id)

    return np.array(X_list), np.array(y_list), np.array(pid_list)


def evaluate_classifier(
    X: np.ndarray,
    y: np.ndarray,
    prompt_ids: np.ndarray,
    n_splits: int = 5,
) -> dict:
    """Evaluate a failure-mode classifier using prompt-level grouped k-fold CV.

    Splits by prompt_id so that all traces from the same prompt go to the
    same fold. This prevents data leakage (same GSM8K question in train+test).
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_.tolist()

    gkf = GroupKFold(n_splits=n_splits)
    all_y_true = []
    all_y_pred = []
    fold_accuracies = []

    for train_idx, test_idx in gkf.split(X, y_encoded, groups=prompt_ids):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        fold_accuracies.append(float(np.mean(y_pred == y_test)))

    # Decode back to labels
    all_y_true_labels = le.inverse_transform(all_y_true)
    all_y_pred_labels = le.inverse_transform(all_y_pred)

    report = classification_report(all_y_true_labels, all_y_pred_labels, output_dict=True)
    cm = confusion_matrix(all_y_true_labels, all_y_pred_labels, labels=classes)

    return {
        "classes": classes,
        "accuracy": float(np.mean(fold_accuracies)),
        "accuracy_std": float(np.std(fold_accuracies)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y),
    }


def format_confusion_matrix(cm: list[list[int]], classes: list[str]) -> str:
    """Format a confusion matrix as markdown."""
    # Header
    header = "| | " + " | ".join(f"Pred: {c}" for c in classes) + " |"
    sep = "|---|" + "|".join(["---"] * len(classes)) + "|"
    rows = [header, sep]
    for i, c in enumerate(classes):
        row_vals = " | ".join(str(v) for v in cm[i])
        rows.append(f"| True: {c} | {row_vals} |")
    return "\n".join(rows)


def main() -> None:
    print("Loading traces...")
    traces = load_traces()
    print(f"  Loaded {len(traces)} traces")

    results = {}

    # Experiment 1: Pre-onset window classification (looping vs non-termination)
    print("\n=== Experiment 1: Pre-onset window (32 tokens before onset) ===")
    X_pre, y_pre, pids_pre = build_classification_dataset(traces, mode="pre_onset", window=32)
    print(f"  Samples: {len(y_pre)}")
    counts = defaultdict(int)
    for label in y_pre:
        counts[label] += 1
    for label, count in sorted(counts.items()):
        print(f"    {label}: {count}")
    if len(set(y_pre)) >= 2:
        res = evaluate_classifier(X_pre, y_pre, pids_pre)
        results["pre_onset"] = res
        print(f"  Accuracy: {res['accuracy']:.3f} ± {res['accuracy_std']:.3f}")
        for cls in res["classes"]:
            r = res["classification_report"].get(cls, {})
            print(
                f"    {cls}: P={r.get('precision', 0):.3f} "
                f"R={r.get('recall', 0):.3f} F1={r.get('f1-score', 0):.3f}"
            )

    # Experiment 2: Early tokens classification (all failure modes)
    print("\n=== Experiment 2: Early tokens (first 50) ===")
    X_early, y_early, pids_early = build_classification_dataset(traces, mode="early")
    print(f"  Samples: {len(y_early)}")
    counts = defaultdict(int)
    for label in y_early:
        counts[label] += 1
    for label, count in sorted(counts.items()):
        print(f"    {label}: {count}")
    if len(set(y_early)) >= 2:
        res = evaluate_classifier(X_early, y_early, pids_early)
        results["early_tokens"] = res
        print(f"  Accuracy: {res['accuracy']:.3f} ± {res['accuracy_std']:.3f}")
        for cls in res["classes"]:
            r = res["classification_report"].get(cls, {})
            print(
                f"    {cls}: P={r.get('precision', 0):.3f} "
                f"R={r.get('recall', 0):.3f} F1={r.get('f1-score', 0):.3f}"
            )

    # Experiment 3: Full trace aggregate (all failure modes)
    print("\n=== Experiment 3: Full trace aggregate ===")
    X_agg, y_agg, pids_agg = build_classification_dataset(traces, mode="trace_agg")
    print(f"  Samples: {len(y_agg)}")
    counts = defaultdict(int)
    for label in y_agg:
        counts[label] += 1
    for label, count in sorted(counts.items()):
        print(f"    {label}: {count}")
    if len(set(y_agg)) >= 2:
        res = evaluate_classifier(X_agg, y_agg, pids_agg)
        results["trace_aggregate"] = res
        print(f"  Accuracy: {res['accuracy']:.3f} ± {res['accuracy_std']:.3f}")
        for cls in res["classes"]:
            r = res["classification_report"].get(cls, {})
            print(
                f"    {cls}: P={r.get('precision', 0):.3f} "
                f"R={r.get('recall', 0):.3f} F1={r.get('f1-score', 0):.3f}"
            )

    # Experiment 4: Binary looping-vs-rest (early tokens)
    print("\n=== Experiment 4: Binary looping detection (early 50 tokens) ===")
    y_binary = np.array(["looping" if y == "looping" else "other" for y in y_early])
    if len(set(y_binary)) >= 2:
        res = evaluate_classifier(X_early, y_binary, pids_early)
        results["binary_looping_early"] = res
        print(f"  Accuracy: {res['accuracy']:.3f} ± {res['accuracy_std']:.3f}")
        for cls in res["classes"]:
            r = res["classification_report"].get(cls, {})
            print(
                f"    {cls}: P={r.get('precision', 0):.3f} "
                f"R={r.get('recall', 0):.3f} F1={r.get('f1-score', 0):.3f}"
            )

    # Generate report
    report_lines = []
    report_lines.append("# S1 Part 2: Failure Mode Classifier Results")
    report_lines.append("")
    report_lines.append("**Generated:** 2026-02-24")
    report_lines.append("")

    for exp_name, exp_key in [
        ("Pre-onset window (32 tokens before onset)", "pre_onset"),
        ("Early tokens (first 50)", "early_tokens"),
        ("Full trace aggregate", "trace_aggregate"),
        ("Binary looping detection (early 50 tokens)", "binary_looping_early"),
    ]:
        if exp_key not in results:
            continue
        res = results[exp_key]
        report_lines.append(f"## {exp_name}")
        report_lines.append("")
        report_lines.append(
            f"**Accuracy:** {res['accuracy']:.3f} ± {res['accuracy_std']:.3f} "
            f"(5-fold CV, N={res['n_samples']})"
        )
        report_lines.append("")

        # Per-class metrics
        report_lines.append("| Class | Precision | Recall | F1 | Support |")
        report_lines.append("|---|---|---|---|---|")
        for cls in res["classes"]:
            r = res["classification_report"].get(cls, {})
            report_lines.append(
                f"| {cls} | {r.get('precision', 0):.3f} | "
                f"{r.get('recall', 0):.3f} | {r.get('f1-score', 0):.3f} | "
                f"{int(r.get('support', 0))} |"
            )
        report_lines.append("")

        # Confusion matrix
        report_lines.append("### Confusion Matrix")
        report_lines.append("")
        report_lines.append(format_confusion_matrix(res["confusion_matrix"], res["classes"]))
        report_lines.append("")

    # Save report
    output_path = Path("docs/analysis/004-failure-mode-classifier.md")
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved to {output_path}")

    # Save raw results
    json_path = Path("models/failure_mode_classifier.json")
    # Convert numpy arrays to lists for JSON serialization
    json_results: dict = {
        "metadata": {
            "generated_by": "scripts/failure_mode_classifier.py",
            "model": MODEL_FILTER,
            "num_prompts": NUM_PROMPTS,
            "compressors": COMPRESSORS,
            "cv_folds": 5,
            "cv_method": "GroupKFold by prompt_id (no prompt-level leakage)",
            "classifier": "GradientBoostingClassifier(n_estimators=200, max_depth=4)",
            "note": "Traces with < 10 signals are excluded from all experiments",
        },
    }
    for k, v in results.items():
        json_results[k] = {
            "accuracy": v["accuracy"],
            "accuracy_std": v["accuracy_std"],
            "classes": v["classes"],
            "n_samples": v["n_samples"],
            "per_class": {
                cls: {
                    "precision": v["classification_report"][cls]["precision"],
                    "recall": v["classification_report"][cls]["recall"],
                    "f1": v["classification_report"][cls]["f1-score"],
                    "support": v["classification_report"][cls]["support"],
                }
                for cls in v["classes"]
            },
            "confusion_matrix": v["confusion_matrix"],
        }
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Raw results saved to {json_path}")


if __name__ == "__main__":
    main()
