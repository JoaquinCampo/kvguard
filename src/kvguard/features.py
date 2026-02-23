"""Feature extraction: trace JSON → ML-ready flat feature matrix.

Converts per-token TokenSignals from sweep results into numpy arrays
suitable for training a hazard predictor (XGBoost / logistic regression).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kvguard.config import RunResult
from kvguard.labeling import compute_hazard_labels

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

# 29 base features per token (order matters — must match flatten_token)
BASE_FEATURE_NAMES: list[str] = [
    "entropy",
    "top1_prob",
    "top5_prob",
    # "rank_of_chosen" removed — always 0 under greedy decoding (do_sample=False)
    *[f"logprob_{i}" for i in range(20)],
    "h_alts",
    "avg_logp",
    "delta_h",
    "delta_h_valid",
    "rep_count",
    "is_thinking_token",
]

N_BASE = len(BASE_FEATURE_NAMES)  # 29

# Rolling window features (appended after base)
ROLLING_COLS = ("entropy", "top1_prob", "h_alts", "delta_h")
ROLLING_COL_INDICES = [BASE_FEATURE_NAMES.index(c) for c in ROLLING_COLS]
ROLLING_STATS = ("mean", "std")
# Plus: rep_count rolling sum, and token_position
ROLLING_WINDOW = 8


def feature_names(window: int = ROLLING_WINDOW) -> list[str]:
    """Return full ordered list of feature names (base + rolling + position + ratio)."""
    names = list(BASE_FEATURE_NAMES)
    for col in ROLLING_COLS:
        for stat in ROLLING_STATS:
            names.append(f"{col}_{stat}_{window}")
    names.append(f"rep_count_sum_{window}")
    names.append("token_position")
    names.append("compression_ratio")
    return names


# ---------------------------------------------------------------------------
# Per-token flattening
# ---------------------------------------------------------------------------


def flatten_token(sig: dict[str, Any]) -> np.ndarray:
    """Convert a single TokenSignals dict to a 1-D float array (N_BASE,)."""
    logprobs = sig.get("top20_logprobs", [])
    # Pad or truncate to exactly 20
    if len(logprobs) < 20:
        logprobs = logprobs + [0.0] * (20 - len(logprobs))
    else:
        logprobs = logprobs[:20]

    delta_h = sig.get("delta_h")
    delta_h_val = float(delta_h) if delta_h is not None else 0.0
    delta_h_valid = 1.0 if delta_h is not None else 0.0

    return np.array(
        [
            sig["entropy"],
            sig["top1_prob"],
            sig["top5_prob"],
            *logprobs,
            sig.get("h_alts", 0.0),
            sig.get("avg_logp", 0.0),
            delta_h_val,
            delta_h_valid,
            float(sig.get("rep_count", 0)),
            float(sig.get("is_thinking_token", False)),
        ],
        dtype=np.float32,
    )


def flatten_signals(signals: list[dict[str, Any]]) -> np.ndarray:
    """Convert a list of TokenSignals dicts to (N, N_BASE) float matrix."""
    if not signals:
        return np.empty((0, N_BASE), dtype=np.float32)
    return np.stack([flatten_token(s) for s in signals])


# ---------------------------------------------------------------------------
# Rolling window features
# ---------------------------------------------------------------------------


def _rolling_stat(col: np.ndarray, window: int, stat: str) -> np.ndarray:
    """Compute rolling statistic over a 1-D column.

    Uses a causal window (looks back, not forward) so the predictor
    cannot peek into the future.  First ``window-1`` tokens use a
    growing window (minimum 1 observation).
    """
    n = len(col)
    out = np.empty(n, dtype=np.float32)
    cumsum = np.cumsum(col)
    for t in range(n):
        start = max(0, t - window + 1)
        count = t - start + 1
        s = cumsum[t] - (cumsum[start - 1] if start > 0 else 0.0)
        if stat == "mean":
            out[t] = s / count
        elif stat == "sum":
            out[t] = s
        elif stat == "std":
            # Two-pass not needed — use E[X^2] - E[X]^2
            mean_val = s / count
            sq_sum = np.sum(col[start : t + 1] ** 2)
            var = sq_sum / count - mean_val**2
            out[t] = np.sqrt(max(var, 0.0))
    return out


def add_rolling_features(
    X_base: np.ndarray,
    window: int = ROLLING_WINDOW,
    compression_ratio: float = 0.0,
) -> np.ndarray:
    """Append rolling statistics, token_position, and compression_ratio.

    Args:
        X_base: (N, N_BASE) base features for a single trace.
        window: Rolling window size.
        compression_ratio: Fraction of KV-cache removed (0.0 = none).

    Returns:
        (N, N_BASE + n_rolling + 2) augmented feature matrix.
    """
    n = X_base.shape[0]
    extras: list[np.ndarray] = []

    # Rolling mean/std for key columns
    for col_idx in ROLLING_COL_INDICES:
        col = X_base[:, col_idx]
        for stat in ROLLING_STATS:
            extras.append(_rolling_stat(col, window, stat))

    # Rolling sum for rep_count
    rep_idx = BASE_FEATURE_NAMES.index("rep_count")
    extras.append(_rolling_stat(X_base[:, rep_idx], window, "sum"))

    # Normalised token position [0, 1]
    if n > 1:
        extras.append(np.arange(n, dtype=np.float32) / (n - 1))
    else:
        extras.append(np.array([0.0], dtype=np.float32))

    # Compression ratio (constant per trace, known at inference time)
    extras.append(np.full(n, compression_ratio, dtype=np.float32))

    return np.column_stack([X_base, *extras])


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


@dataclass
class TraceMeta:
    """Metadata for one trace (used for stratified splitting later)."""

    trace_idx: int
    prompt_id: str
    press: str
    compression_ratio: float
    has_catastrophe: bool
    catastrophe_types: list[str]
    n_tokens: int


@dataclass
class Dataset:
    """ML-ready dataset from sweep results."""

    X: np.ndarray  # (N_total_tokens, D) feature matrix
    y: np.ndarray  # (N_total_tokens,) binary labels
    trace_ids: np.ndarray  # (N_total_tokens,) trace index per token
    feature_names: list[str]  # ordered feature column names
    traces: list[TraceMeta] = field(default_factory=list)


def load_result_file(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a sweep result JSON, returning (config, results_list)."""
    with open(path) as f:
        data = json.load(f)
    return data["config"], data["results"]


def result_dict_to_run_result(d: dict[str, Any]) -> RunResult:
    """Convert a raw dict from JSON back into a RunResult (for labeling)."""
    return RunResult.model_validate(d)


def build_dataset(
    results_dir: Path,
    *,
    num_prompts: int = 50,
    horizon: int = 32,
    nt_onset_frac: float = 0.75,
    rolling_window: int = ROLLING_WINDOW,
) -> Dataset:
    """Load all sweep result files and build a flat ML dataset.

    Args:
        results_dir: Root results directory (e.g., ``results/``).
        num_prompts: Filter to files matching this prompt count.
        horizon: Hazard label horizon H.
        nt_onset_frac: Non-termination proxy onset fraction.
        rolling_window: Window size for rolling features.

    Returns:
        Dataset with concatenated features, labels, and metadata.
    """
    pattern = f"*_{num_prompts}p.json"
    files = sorted(results_dir.rglob(pattern))
    if not files:
        msg = f"No result files matching {pattern} in {results_dir}"
        raise FileNotFoundError(msg)

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_trace_ids: list[np.ndarray] = []
    traces: list[TraceMeta] = []
    trace_idx = 0

    for fpath in files:
        _config, results_list = load_result_file(fpath)
        for result_dict in results_list:
            run = result_dict_to_run_result(result_dict)
            n_tok = run.num_tokens_generated
            if n_tok == 0:
                continue

            # Features
            sigs = [
                s if isinstance(s, dict) else s.model_dump() for s in result_dict.get("signals", [])
            ]
            X_base = flatten_signals(sigs[:n_tok])
            X_full = add_rolling_features(
                X_base,
                window=rolling_window,
                compression_ratio=run.compression_ratio,
            )

            # Labels
            labels = compute_hazard_labels(run, horizon=horizon, nt_onset_frac=nt_onset_frac)
            y = np.array(labels, dtype=np.float32)

            all_X.append(X_full)
            all_y.append(y)
            all_trace_ids.append(np.full(n_tok, trace_idx, dtype=np.int32))

            cat_types = [c for c in run.catastrophes if c != "wrong_answer"]
            traces.append(
                TraceMeta(
                    trace_idx=trace_idx,
                    prompt_id=run.prompt_id,
                    press=run.press,
                    compression_ratio=run.compression_ratio,
                    has_catastrophe=bool(cat_types),
                    catastrophe_types=cat_types,
                    n_tokens=n_tok,
                )
            )
            trace_idx += 1

    return Dataset(
        X=np.concatenate(all_X, axis=0),
        y=np.concatenate(all_y, axis=0),
        trace_ids=np.concatenate(all_trace_ids, axis=0),
        feature_names=feature_names(rolling_window),
        traces=traces,
    )
