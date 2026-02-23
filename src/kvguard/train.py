"""Hazard predictor training: Dataset → XGBoost binary classifier.

Provides functions for trace-level stratified splitting, XGBoost training
with automatic class-imbalance handling, evaluation, leave-one-compressor-out
cross-validation, and optional wandb logging.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from kvguard.features import Dataset, build_dataset

# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------


def load_all_results(
    results_dir: Path,
    *,
    num_prompts: int = 50,
    horizon: int = 32,
    nt_onset_frac: float = 0.75,
    rolling_window: int = 8,
) -> Dataset:
    """Load all sweep results into a flat ML dataset.

    Thin wrapper around :func:`kvguard.features.build_dataset` for a
    consistent training-module entry point.
    """
    return build_dataset(
        results_dir,
        num_prompts=num_prompts,
        horizon=horizon,
        nt_onset_frac=nt_onset_frac,
        rolling_window=rolling_window,
    )


@dataclass
class SplitResult:
    """Indices for a train/val split."""

    train_traces: list[int]
    val_traces: list[int]
    train_mask: np.ndarray  # boolean mask over tokens
    val_mask: np.ndarray


def split_by_trace(
    ds: Dataset,
    *,
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> SplitResult:
    """Split dataset so no trace appears in both train and val.

    Stratifies by ``has_catastrophe`` to ensure both splits contain
    positive examples.  Falls back to random split if stratification
    is impossible (e.g., only one catastrophe trace).

    Args:
        ds: Dataset from :func:`load_all_results`.
        val_fraction: Fraction of *traces* to hold out.
        random_state: RNG seed for reproducibility.

    Returns:
        SplitResult with trace indices and boolean token masks.
    """
    rng = np.random.RandomState(random_state)

    # Group trace indices by catastrophe status
    pos_traces = [t.trace_idx for t in ds.traces if t.has_catastrophe]
    neg_traces = [t.trace_idx for t in ds.traces if not t.has_catastrophe]

    def _sample_val(indices: list[int], frac: float) -> list[int]:
        n_val = max(1, int(len(indices) * frac))
        n_val = min(n_val, len(indices) - 1)  # keep at least 1 for train
        if n_val <= 0 or len(indices) <= 1:
            return []
        perm = rng.permutation(len(indices))
        return [indices[i] for i in perm[:n_val]]

    val_pos = _sample_val(pos_traces, val_fraction)
    val_neg = _sample_val(neg_traces, val_fraction)
    val_set = set(val_pos + val_neg)

    # Fallback: if stratification produced empty val, just take random traces
    if not val_set:
        all_idx = list(range(len(ds.traces)))
        n_val = max(1, int(len(all_idx) * val_fraction))
        perm = rng.permutation(len(all_idx))
        val_set = {all_idx[i] for i in perm[:n_val]}

    train_traces = sorted(i for i in range(len(ds.traces)) if i not in val_set)
    val_traces = sorted(val_set)

    train_mask = np.isin(ds.trace_ids, train_traces)
    val_mask = np.isin(ds.trace_ids, val_traces)

    return SplitResult(
        train_traces=train_traces,
        val_traces=val_traces,
        train_mask=train_mask,
        val_mask=val_mask,
    )


def split_by_prompt(
    ds: Dataset,
    *,
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> SplitResult:
    """Split dataset by prompt_id — all configs for a prompt go to the same partition.

    Stratifies by whether a prompt has *any* catastrophe trace, ensuring both
    partitions contain catastrophe examples.

    Args:
        ds: Dataset from :func:`load_all_results`.
        val_fraction: Fraction of *prompts* to hold out.
        random_state: RNG seed for reproducibility.

    Returns:
        SplitResult with trace indices and boolean token masks.
    """
    rng = np.random.RandomState(random_state)

    # Group traces by prompt_id
    prompt_to_traces: dict[str, list[int]] = {}
    for t in ds.traces:
        prompt_to_traces.setdefault(t.prompt_id, []).append(t.trace_idx)

    # A prompt "has catastrophe" if any of its traces do
    prompt_ids = sorted(prompt_to_traces.keys())
    prompt_has_cat = {
        pid: any(ds.traces[ti].has_catastrophe for ti in prompt_to_traces[pid])
        for pid in prompt_ids
    }

    pos_prompts = [p for p in prompt_ids if prompt_has_cat[p]]
    neg_prompts = [p for p in prompt_ids if not prompt_has_cat[p]]

    def _sample_val(prompts: list[str], frac: float) -> list[str]:
        n_val = max(1, int(len(prompts) * frac))
        n_val = min(n_val, len(prompts) - 1)
        if n_val <= 0 or len(prompts) <= 1:
            return []
        perm = rng.permutation(len(prompts))
        return [prompts[i] for i in perm[:n_val]]

    val_prompts = set(
        _sample_val(pos_prompts, val_fraction) + _sample_val(neg_prompts, val_fraction)
    )

    if not val_prompts:
        n_val = max(1, int(len(prompt_ids) * val_fraction))
        perm = rng.permutation(len(prompt_ids))
        val_prompts = {prompt_ids[i] for i in perm[:n_val]}

    val_traces = sorted(ti for pid in val_prompts for ti in prompt_to_traces[pid])
    train_traces = sorted(
        ti for pid in prompt_ids if pid not in val_prompts for ti in prompt_to_traces[pid]
    )

    train_mask = np.isin(ds.trace_ids, train_traces)
    val_mask = np.isin(ds.trace_ids, val_traces)

    return SplitResult(
        train_traces=train_traces,
        val_traces=val_traces,
        train_mask=train_mask,
        val_mask=val_mask,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "verbosity": 0,
    "random_state": 42,
}


def train_predictor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    xgb_params: dict[str, Any] | None = None,
    auto_scale_pos_weight: bool = True,
) -> xgb.XGBClassifier:
    """Train an XGBoost binary classifier for hazard prediction.

    Args:
        X_train: (N, D) feature matrix.
        y_train: (N,) binary labels.
        xgb_params: Override default XGBoost parameters.
        auto_scale_pos_weight: If True, set scale_pos_weight = n_neg / n_pos.

    Returns:
        Trained XGBClassifier.
    """
    params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}

    if auto_scale_pos_weight and "scale_pos_weight" not in params:
        n_pos = float(np.sum(y_train == 1))
        n_neg = float(np.sum(y_train == 0))
        if n_pos > 0:
            params["scale_pos_weight"] = n_neg / n_pos

    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train)
    return clf


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvalMetrics:
    """Evaluation metrics for the hazard predictor."""

    precision: float
    recall: float
    f1: float
    auroc: float
    n_samples: int
    n_positive: int
    threshold: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auroc": self.auroc,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "threshold": self.threshold,
        }


def evaluate_predictor(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    threshold: float = 0.5,
) -> EvalMetrics:
    """Evaluate trained predictor on a test set.

    Args:
        model: Trained XGBClassifier.
        X_test: (N, D) feature matrix.
        y_test: (N,) binary labels.
        threshold: Decision threshold for binary predictions.

    Returns:
        EvalMetrics with precision, recall, F1, AUROC.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    n_pos = int(np.sum(y_test == 1))

    # Handle degenerate cases (all same class)
    if n_pos == 0 or n_pos == len(y_test):
        auroc = 0.0
    else:
        auroc = float(roc_auc_score(y_test, y_prob))

    return EvalMetrics(
        precision=float(precision_score(y_test, y_pred, zero_division=0.0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0.0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0.0)),
        auroc=auroc,
        n_samples=len(y_test),
        n_positive=n_pos,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Leave-one-compressor-out cross-validation
# ---------------------------------------------------------------------------


@dataclass
class CVFold:
    """Results from one fold of leave-one-compressor-out CV."""

    held_out_press: str
    train_presses: list[str]
    metrics: EvalMetrics


@dataclass
class CVResult:
    """Aggregate cross-validation results."""

    folds: list[CVFold] = field(default_factory=list)

    @property
    def mean_auroc(self) -> float:
        vals = [f.metrics.auroc for f in self.folds]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def mean_f1(self) -> float:
        vals = [f.metrics.f1 for f in self.folds]
        return float(np.mean(vals)) if vals else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_auroc": self.mean_auroc,
            "mean_f1": self.mean_f1,
            "folds": [
                {
                    "held_out_press": f.held_out_press,
                    "train_presses": f.train_presses,
                    "metrics": f.metrics.to_dict(),
                }
                for f in self.folds
            ],
        }


def leave_one_out_cv(
    ds: Dataset,
    *,
    xgb_params: dict[str, Any] | None = None,
) -> CVResult:
    """Leave-one-compressor-out cross-validation.

    For each unique compressor (press), train on all other compressors
    and evaluate on the held-out one.  This tests whether the predictor
    generalises across compression methods.

    Note: prompt overlap between folds is intentional here — the purpose
    is to measure cross-compressor generalisation, not cross-prompt
    generalisation. Prompt-level isolation is enforced only in the main
    train/val split (see :func:`split_by_prompt`).

    Args:
        ds: Full dataset.
        xgb_params: Override default XGBoost parameters.

    Returns:
        CVResult with per-fold and aggregate metrics.
    """
    presses = sorted({t.press for t in ds.traces})
    result = CVResult()

    for held_out in presses:
        held_out_traces = {t.trace_idx for t in ds.traces if t.press == held_out}
        train_traces = {t.trace_idx for t in ds.traces if t.press != held_out}

        if not held_out_traces or not train_traces:
            continue

        val_mask = np.isin(ds.trace_ids, sorted(held_out_traces))
        train_mask = np.isin(ds.trace_ids, sorted(train_traces))

        X_train, y_train = ds.X[train_mask], ds.y[train_mask]
        X_val, y_val = ds.X[val_mask], ds.y[val_mask]

        # Skip folds with degenerate labels
        if len(np.unique(y_train)) < 2:
            continue

        model = train_predictor(X_train, y_train, xgb_params=xgb_params)
        metrics = evaluate_predictor(model, X_val, y_val)

        train_presses = sorted({t.press for t in ds.traces if t.press != held_out})
        result.folds.append(
            CVFold(
                held_out_press=held_out,
                train_presses=train_presses,
                metrics=metrics,
            )
        )

    return result


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------


@dataclass
class TrainOutput:
    """Output of a full training run."""

    model: xgb.XGBClassifier
    train_metrics: EvalMetrics
    val_metrics: EvalMetrics
    cv_result: CVResult | None
    split: SplitResult


def run_training(
    results_dir: Path,
    *,
    num_prompts: int = 50,
    horizon: int = 32,
    nt_onset_frac: float = 0.75,
    val_fraction: float = 0.2,
    random_state: int = 42,
    xgb_params: dict[str, Any] | None = None,
    run_cv: bool = True,
    output_dir: Path | None = None,
) -> TrainOutput:
    """End-to-end training pipeline.

    1. Load dataset from sweep results.
    2. Split by trace (stratified).
    3. Train XGBoost predictor.
    4. Evaluate on train and val sets.
    5. Optionally run leave-one-compressor-out CV.
    6. Optionally save model and metrics.

    Args:
        results_dir: Directory with sweep result JSONs.
        num_prompts: Filter to files matching this prompt count.
        horizon: Hazard label horizon H.
        nt_onset_frac: Non-termination proxy onset fraction.
        val_fraction: Fraction of traces for validation.
        random_state: RNG seed.
        xgb_params: Override XGBoost parameters.
        run_cv: Whether to run leave-one-compressor-out CV.
        output_dir: If set, save model + metrics here.

    Returns:
        TrainOutput with model, metrics, and split info.
    """
    ds = load_all_results(
        results_dir,
        num_prompts=num_prompts,
        horizon=horizon,
        nt_onset_frac=nt_onset_frac,
    )

    split = split_by_prompt(ds, val_fraction=val_fraction, random_state=random_state)

    X_train = ds.X[split.train_mask]
    y_train = ds.y[split.train_mask]
    X_val = ds.X[split.val_mask]
    y_val = ds.y[split.val_mask]

    model = train_predictor(X_train, y_train, xgb_params=xgb_params)
    train_metrics = evaluate_predictor(model, X_train, y_train)
    val_metrics = evaluate_predictor(model, X_val, y_val)

    cv_result = leave_one_out_cv(ds, xgb_params=xgb_params) if run_cv else None

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(str(output_dir / "hazard_predictor.json"))
        metrics_out = {
            "horizon": horizon,
            "nt_onset_frac": nt_onset_frac,
            "train": train_metrics.to_dict(),
            "val": val_metrics.to_dict(),
            "cv": cv_result.to_dict() if cv_result else None,
            "n_traces": len(ds.traces),
            "n_tokens": len(ds.y),
            "positive_rate": float(np.mean(ds.y)),
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2))

        # Save prompt partition for controller evaluation holdout
        val_prompt_ids = {ds.traces[t].prompt_id for t in split.val_traces}
        split_info = {
            "val_prompt_ids": sorted(val_prompt_ids),
            "train_prompt_ids": sorted({ds.traces[t].prompt_id for t in split.train_traces}),
            "n_val_traces": len(split.val_traces),
            "n_train_traces": len(split.train_traces),
        }
        (output_dir / "split_info.json").write_text(json.dumps(split_info, indent=2))

    return TrainOutput(
        model=model,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        cv_result=cv_result,
        split=split,
    )
