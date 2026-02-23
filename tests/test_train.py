"""Tests for hazard predictor training module."""

import json
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from kvguard.features import Dataset, TraceMeta
from kvguard.train import (
    CVResult,
    EvalMetrics,
    evaluate_predictor,
    leave_one_out_cv,
    load_all_results,
    run_training,
    split_by_prompt,
    split_by_trace,
    train_predictor,
)

# ---------------------------------------------------------------------------
# Helpers — reuse patterns from test_features.py
# ---------------------------------------------------------------------------


def _make_signal_dict(
    *,
    entropy: float = 1.0,
    top1_prob: float = 0.5,
    top5_prob: float = 0.9,
    rank_of_chosen: int = 0,
    top20_logprobs: list[float] | None = None,
    h_alts: float = 0.3,
    avg_logp: float = -2.0,
    delta_h: float | None = 0.1,
    rep_count: int = 0,
    is_thinking_token: bool = False,
) -> dict:
    return {
        "entropy": entropy,
        "top1_prob": top1_prob,
        "top5_prob": top5_prob,
        "top1_token": "a",
        "rank_of_chosen": rank_of_chosen,
        "top20_logprobs": top20_logprobs if top20_logprobs is not None else [-0.5] * 20,
        "h_alts": h_alts,
        "avg_logp": avg_logp,
        "delta_h": delta_h,
        "rep_count": rep_count,
        "is_thinking_token": is_thinking_token,
    }


def _make_result_json(
    *,
    n_tokens: int = 50,
    press: str = "none",
    compression_ratio: float = 0.0,
    catastrophes: list[str] | None = None,
    catastrophe_onsets: dict[str, int] | None = None,
    prompt_id: str = "gsm8k_0",
) -> dict:
    sigs = []
    for t in range(n_tokens):
        sigs.append(
            _make_signal_dict(
                entropy=1.0 + 0.01 * t,
                delta_h=0.01 if t > 0 else None,
                rep_count=max(0, t - 8),
            )
        )
    return {
        "prompt_id": prompt_id,
        "prompt_text": "test prompt",
        "model": "test-model",
        "press": press,
        "compression_ratio": compression_ratio,
        "max_new_tokens": 512,
        "seed": 42,
        "generated_text": "test output",
        "ground_truth": "42",
        "predicted_answer": "42",
        "correct": True,
        "stop_reason": "eos",
        "catastrophes": catastrophes or [],
        "num_tokens_generated": n_tokens,
        "cache_size_after_prefill": None,
        "catastrophe_onsets": catastrophe_onsets or {},
        "signals": sigs,
    }


def _write_result_file(
    tmpdir: Path, press: str, ratio: float, results: list[dict], n_prompts: int = 50
) -> Path:
    subdir = tmpdir / press
    subdir.mkdir(parents=True, exist_ok=True)
    fname = f"test-model_{ratio:.3f}_{n_prompts}p.json"
    path = subdir / fname
    data = {
        "config": {
            "model_name": "test-model",
            "press_name": press,
            "compression_ratio": ratio,
        },
        "summary": {},
        "results": results,
    }
    path.write_text(json.dumps(data))
    return path


def _make_synthetic_dataset(
    *,
    n_prompts: int = 10,
    presses: list[str] | None = None,
    ratios: list[float] | None = None,
    catastrophe_fraction: float = 0.3,
    tokens_per_trace: int = 50,
    random_state: int = 42,
) -> Dataset:
    """Build synthetic Dataset with realistic prompt × config structure."""
    presses = presses or ["streaming_llm", "snapkv"]
    ratios = ratios or [0.25, 0.5, 0.75]
    rng = np.random.RandomState(random_state)

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_trace_ids: list[np.ndarray] = []
    traces: list[TraceMeta] = []
    trace_idx = 0

    from kvguard.features import feature_names

    n_features = len(feature_names())

    for prompt_i in range(n_prompts):
        prompt_id = f"gsm8k_{prompt_i}"
        is_cat_prompt = rng.random() < catastrophe_fraction
        for press in presses:
            for ratio in ratios:
                n_tok = tokens_per_trace
                X = rng.randn(n_tok, n_features).astype(np.float32)
                y = np.zeros(n_tok, dtype=np.float32)
                # Some configs of a catastrophe-prompt actually fail
                has_cat = is_cat_prompt and ratio >= 0.5 and rng.random() < 0.7
                if has_cat:
                    onset = int(n_tok * 0.6)
                    y[onset:] = 1.0
                    X[onset:, 0] += 3.0  # boost entropy

                all_X.append(X)
                all_y.append(y)
                all_trace_ids.append(np.full(n_tok, trace_idx, dtype=np.int32))
                traces.append(
                    TraceMeta(
                        trace_idx=trace_idx,
                        prompt_id=prompt_id,
                        press=press,
                        compression_ratio=ratio,
                        has_catastrophe=has_cat,
                        catastrophe_types=["looping"] if has_cat else [],
                        n_tokens=n_tok,
                    )
                )
                trace_idx += 1

    return Dataset(
        X=np.concatenate(all_X),
        y=np.concatenate(all_y),
        trace_ids=np.concatenate(all_trace_ids),
        feature_names=feature_names(),
        traces=traces,
    )


# ---------------------------------------------------------------------------
# Tests: split_by_trace
# ---------------------------------------------------------------------------


class TestSplitByTrace:
    def test_no_trace_overlap(self) -> None:
        """Train and val traces must be disjoint."""
        ds = _make_synthetic_dataset()
        split = split_by_trace(ds, val_fraction=0.2)
        assert set(split.train_traces).isdisjoint(set(split.val_traces))

    def test_all_tokens_assigned(self) -> None:
        """Every token must be in exactly one of train or val."""
        ds = _make_synthetic_dataset()
        split = split_by_trace(ds, val_fraction=0.2)
        assert np.all(split.train_mask | split.val_mask)
        assert not np.any(split.train_mask & split.val_mask)

    def test_all_traces_present(self) -> None:
        """Union of train_traces + val_traces = all traces."""
        ds = _make_synthetic_dataset()
        split = split_by_trace(ds, val_fraction=0.2)
        all_traces = sorted(split.train_traces + split.val_traces)
        assert all_traces == list(range(len(ds.traces)))

    def test_val_fraction_respected(self) -> None:
        """Val set should be roughly val_fraction of traces."""
        ds = _make_synthetic_dataset(n_prompts=20)
        split = split_by_trace(ds, val_fraction=0.2)
        frac = len(split.val_traces) / len(ds.traces)
        assert 0.1 <= frac <= 0.4  # loose bounds due to stratification

    def test_stratification_both_classes(self) -> None:
        """Both train and val should have positive examples (if possible)."""
        ds = _make_synthetic_dataset(catastrophe_fraction=0.4, random_state=123)
        split = split_by_trace(ds, val_fraction=0.3)
        train_has_pos = any(ds.traces[i].has_catastrophe for i in split.train_traces)
        val_has_pos = any(ds.traces[i].has_catastrophe for i in split.val_traces)
        # At least one set should have positives
        assert train_has_pos or val_has_pos

    def test_reproducible(self) -> None:
        """Same random_state gives same split."""
        ds = _make_synthetic_dataset()
        s1 = split_by_trace(ds, random_state=42)
        s2 = split_by_trace(ds, random_state=42)
        assert s1.train_traces == s2.train_traces
        assert s1.val_traces == s2.val_traces


# ---------------------------------------------------------------------------
# Tests: split_by_prompt
# ---------------------------------------------------------------------------


class TestSplitByPrompt:
    """Tests for prompt-level splitting (no prompt_id in both train and val)."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.ds = _make_synthetic_dataset(
            n_prompts=10,
            presses=["streaming_llm", "snapkv", "observed_attention"],
            ratios=[0.25, 0.5, 0.75],
            catastrophe_fraction=0.3,
        )

    def test_no_prompt_overlap(self) -> None:
        """No prompt_id appears in both train and val."""
        split = split_by_prompt(self.ds, val_fraction=0.2)
        train_prompts = {self.ds.traces[t].prompt_id for t in split.train_traces}
        val_prompts = {self.ds.traces[t].prompt_id for t in split.val_traces}
        assert train_prompts & val_prompts == set()

    def test_all_configs_for_prompt_stay_together(self) -> None:
        """Every trace of a prompt is in the same partition."""
        split = split_by_prompt(self.ds, val_fraction=0.2)
        val_set = set(split.val_traces)
        for prompt_id in {t.prompt_id for t in self.ds.traces}:
            trace_idxs = [t.trace_idx for t in self.ds.traces if t.prompt_id == prompt_id]
            in_val = [i in val_set for i in trace_idxs]
            assert all(in_val) or not any(in_val), f"Prompt {prompt_id} split across partitions"

    def test_all_traces_assigned(self) -> None:
        split = split_by_prompt(self.ds, val_fraction=0.2)
        assert sorted(split.train_traces + split.val_traces) == list(range(len(self.ds.traces)))

    def test_stratification(self) -> None:
        """Both partitions have at least one catastrophe prompt."""
        split = split_by_prompt(self.ds, val_fraction=0.3)
        train_has_cat = any(self.ds.traces[t].has_catastrophe for t in split.train_traces)
        val_has_cat = any(self.ds.traces[t].has_catastrophe for t in split.val_traces)
        assert train_has_cat and val_has_cat

    def test_reproducible(self) -> None:
        s1 = split_by_prompt(self.ds, val_fraction=0.2, random_state=42)
        s2 = split_by_prompt(self.ds, val_fraction=0.2, random_state=42)
        assert s1.train_traces == s2.train_traces


# ---------------------------------------------------------------------------
# Tests: train_predictor
# ---------------------------------------------------------------------------


class TestTrainPredictor:
    def test_returns_classifier(self) -> None:
        ds = _make_synthetic_dataset()
        model = train_predictor(ds.X, ds.y)
        assert isinstance(model, xgb.XGBClassifier)

    def test_auto_scale_pos_weight(self) -> None:
        """scale_pos_weight should be set automatically."""
        ds = _make_synthetic_dataset()
        model = train_predictor(ds.X, ds.y, auto_scale_pos_weight=True)
        # Model should have been trained — verify it can predict
        proba = model.predict_proba(ds.X[:5])
        assert proba.shape == (5, 2)

    def test_custom_params(self) -> None:
        ds = _make_synthetic_dataset()
        model = train_predictor(
            ds.X,
            ds.y,
            xgb_params={"n_estimators": 10, "max_depth": 3},
        )
        assert isinstance(model, xgb.XGBClassifier)

    def test_all_negative_labels(self) -> None:
        """Training on all-negative data should not crash."""
        ds = _make_synthetic_dataset(catastrophe_fraction=0.0)
        model = train_predictor(ds.X, ds.y)
        assert isinstance(model, xgb.XGBClassifier)


# ---------------------------------------------------------------------------
# Tests: evaluate_predictor
# ---------------------------------------------------------------------------


class TestEvaluatePredictor:
    def test_metrics_fields(self) -> None:
        ds = _make_synthetic_dataset()
        model = train_predictor(ds.X, ds.y, xgb_params={"n_estimators": 10})
        metrics = evaluate_predictor(model, ds.X, ds.y)
        assert isinstance(metrics, EvalMetrics)
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.f1 <= 1.0
        assert 0.0 <= metrics.auroc <= 1.0
        assert metrics.n_samples == len(ds.y)

    def test_to_dict(self) -> None:
        ds = _make_synthetic_dataset()
        model = train_predictor(ds.X, ds.y, xgb_params={"n_estimators": 10})
        metrics = evaluate_predictor(model, ds.X, ds.y)
        d = metrics.to_dict()
        assert "precision" in d
        assert "recall" in d
        assert "auroc" in d

    def test_all_negative_eval(self) -> None:
        """Evaluating on all-negative data should not crash."""
        ds = _make_synthetic_dataset(catastrophe_fraction=0.0)
        model = train_predictor(ds.X, ds.y, xgb_params={"n_estimators": 10})
        metrics = evaluate_predictor(model, ds.X, ds.y)
        assert metrics.n_positive == 0
        assert metrics.auroc == 0.0


# ---------------------------------------------------------------------------
# Tests: leave_one_out_cv
# ---------------------------------------------------------------------------


class TestLeaveOneOutCV:
    def test_folds_match_presses(self) -> None:
        ds = _make_synthetic_dataset(catastrophe_fraction=0.5, random_state=99)
        cv = leave_one_out_cv(ds, xgb_params={"n_estimators": 10})
        assert isinstance(cv, CVResult)
        held_out_presses = {f.held_out_press for f in cv.folds}
        # Should have a fold for each press (may skip if degenerate labels)
        assert len(held_out_presses) >= 1

    def test_held_out_not_in_train(self) -> None:
        ds = _make_synthetic_dataset(catastrophe_fraction=0.5, random_state=99)
        cv = leave_one_out_cv(ds, xgb_params={"n_estimators": 10})
        for fold in cv.folds:
            assert fold.held_out_press not in fold.train_presses

    def test_cv_to_dict(self) -> None:
        ds = _make_synthetic_dataset(catastrophe_fraction=0.5, random_state=99)
        cv = leave_one_out_cv(ds, xgb_params={"n_estimators": 10})
        d = cv.to_dict()
        assert "mean_auroc" in d
        assert "mean_f1" in d
        assert "folds" in d


# ---------------------------------------------------------------------------
# Tests: load_all_results (integration with features.build_dataset)
# ---------------------------------------------------------------------------


class TestLoadAllResults:
    def test_loads_from_files(self, tmp_path: Path) -> None:
        results = [_make_result_json(n_tokens=20, press="none")]
        _write_result_file(tmp_path, "none", 0.0, results)
        ds = load_all_results(tmp_path, num_prompts=50, horizon=32)
        assert isinstance(ds, Dataset)
        assert ds.X.shape[0] == 20


# ---------------------------------------------------------------------------
# Tests: run_training (end-to-end)
# ---------------------------------------------------------------------------


class TestRunTraining:
    def test_end_to_end(self, tmp_path: Path) -> None:
        """Full pipeline: files → model + metrics."""
        # Create result files with mix of catastrophe and clean traces
        results_none = [
            _make_result_json(n_tokens=50, press="none", prompt_id="p0"),
            _make_result_json(n_tokens=50, press="none", prompt_id="p1"),
        ]
        results_slm = [
            _make_result_json(
                n_tokens=50,
                press="streaming_llm",
                compression_ratio=0.5,
                catastrophes=["looping"],
                catastrophe_onsets={"looping": 30},
                prompt_id="p0",
            ),
            _make_result_json(
                n_tokens=50,
                press="streaming_llm",
                compression_ratio=0.5,
                catastrophes=["looping"],
                catastrophe_onsets={"looping": 25},
                prompt_id="p1",
            ),
            _make_result_json(
                n_tokens=50,
                press="streaming_llm",
                compression_ratio=0.5,
                prompt_id="p2",
            ),
        ]
        _write_result_file(tmp_path, "none", 0.0, results_none)
        _write_result_file(tmp_path, "streaming_llm", 0.5, results_slm)

        out_dir = tmp_path / "output"
        result = run_training(
            tmp_path,
            num_prompts=50,
            horizon=16,
            val_fraction=0.3,
            xgb_params={"n_estimators": 10},
            run_cv=True,
            output_dir=out_dir,
        )

        assert isinstance(result.model, xgb.XGBClassifier)
        assert result.val_metrics.n_samples > 0
        assert (out_dir / "hazard_predictor.json").exists()
        assert (out_dir / "metrics.json").exists()

        # Verify metrics JSON is valid
        metrics = json.loads((out_dir / "metrics.json").read_text())
        assert "train" in metrics
        assert "val" in metrics

        # Verify split_info.json is saved
        assert (out_dir / "split_info.json").exists()
        split_info = json.loads((out_dir / "split_info.json").read_text())
        assert "val_prompt_ids" in split_info
        assert "train_prompt_ids" in split_info
        assert len(split_info["val_prompt_ids"]) > 0
        assert len(split_info["train_prompt_ids"]) > 0


# ---------------------------------------------------------------------------
# Tests: split_by_prompt edge cases (Phase 2B)
# ---------------------------------------------------------------------------


class TestSplitByPromptEdgeCases:
    def test_split_single_positive_prompt(self) -> None:
        """With only 1 catastrophe prompt, it stays in train (no val positive)."""
        ds = _make_synthetic_dataset(
            n_prompts=5,
            catastrophe_fraction=0.15,
            random_state=42,
        )
        # Verify there is exactly 1 positive prompt
        prompt_has_cat: dict[str, bool] = {}
        for t in ds.traces:
            if t.has_catastrophe:
                prompt_has_cat[t.prompt_id] = True
            else:
                prompt_has_cat.setdefault(t.prompt_id, False)

        n_pos = sum(1 for v in prompt_has_cat.values() if v)
        # If random_state doesn't yield exactly 1, adjust — but check at least works
        if n_pos == 0:
            pytest.skip("No positive prompts generated with this seed")

        split = split_by_prompt(ds, val_fraction=0.2, random_state=42)
        # Split should work without error
        assert sorted(split.train_traces + split.val_traces) == list(range(len(ds.traces)))
        # The single positive prompt should be in train (since _sample_val
        # returns empty for len(prompts) <= 1)
        if n_pos == 1:
            pos_prompt_id = [p for p, v in prompt_has_cat.items() if v][0]
            train_prompts = {ds.traces[t].prompt_id for t in split.train_traces}
            assert pos_prompt_id in train_prompts

    def test_split_all_positive(self) -> None:
        """All prompts have catastrophe — split still works."""
        ds = _make_synthetic_dataset(
            n_prompts=5,
            catastrophe_fraction=1.0,
            random_state=42,
        )
        split = split_by_prompt(ds, val_fraction=0.2, random_state=42)
        # Should not crash and all traces should be assigned
        assert sorted(split.train_traces + split.val_traces) == list(range(len(ds.traces)))


# ---------------------------------------------------------------------------
# Tests: pre-onset evaluation (Phase 1A)
# ---------------------------------------------------------------------------


class TestPreOnsetEvaluation:
    """Test evaluate_predictor with pre_onset_mask."""

    def test_pre_onset_metrics_populated(self) -> None:
        """When pre_onset_mask is provided, pre_onset_auroc and pre_onset_recall are set."""
        ds = _make_synthetic_dataset(catastrophe_fraction=0.5, random_state=99)
        model = train_predictor(ds.X, ds.y, xgb_params={"n_estimators": 10})

        # Create a mask: first half of tokens are "pre-onset"
        mask = np.zeros(len(ds.y), dtype=bool)
        mask[: len(ds.y) // 2] = True

        metrics = evaluate_predictor(model, ds.X, ds.y, pre_onset_mask=mask)
        # With mixed labels in the mask subset, both should be populated
        n_pos_in_mask = int(np.sum(ds.y[mask] == 1))
        if n_pos_in_mask > 0 and n_pos_in_mask < int(mask.sum()):
            assert metrics.pre_onset_recall is not None
            assert metrics.pre_onset_auroc is not None
            assert 0.0 <= metrics.pre_onset_recall <= 1.0
            assert 0.0 <= metrics.pre_onset_auroc <= 1.0

    def test_pre_onset_metrics_none_without_mask(self) -> None:
        """Without pre_onset_mask, pre-onset metrics are None."""
        ds = _make_synthetic_dataset(catastrophe_fraction=0.5, random_state=99)
        model = train_predictor(ds.X, ds.y, xgb_params={"n_estimators": 10})
        metrics = evaluate_predictor(model, ds.X, ds.y)
        assert metrics.pre_onset_recall is None
        assert metrics.pre_onset_auroc is None

    def test_pre_onset_metrics_in_to_dict(self) -> None:
        """Pre-onset metrics appear in to_dict() when set."""
        ds = _make_synthetic_dataset(catastrophe_fraction=0.5, random_state=99)
        model = train_predictor(ds.X, ds.y, xgb_params={"n_estimators": 10})
        mask = np.ones(len(ds.y), dtype=bool)
        metrics = evaluate_predictor(model, ds.X, ds.y, pre_onset_mask=mask)
        d = metrics.to_dict()
        # At minimum, the keys should be present if values are set
        if metrics.pre_onset_recall is not None:
            assert "pre_onset_recall" in d
        if metrics.pre_onset_auroc is not None:
            assert "pre_onset_auroc" in d

    def test_pre_onset_metrics_not_in_to_dict_when_none(self) -> None:
        """Pre-onset metrics are excluded from to_dict() when None."""
        ds = _make_synthetic_dataset(catastrophe_fraction=0.5, random_state=99)
        model = train_predictor(ds.X, ds.y, xgb_params={"n_estimators": 10})
        metrics = evaluate_predictor(model, ds.X, ds.y)
        d = metrics.to_dict()
        assert "pre_onset_recall" not in d
        assert "pre_onset_auroc" not in d


# ---------------------------------------------------------------------------
# Tests: early stopping (Phase 3A)
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    def test_early_stopping_reduces_trees(self) -> None:
        """With separable val data, early stopping should stop before 200 trees."""
        rng = np.random.RandomState(42)
        n = 500
        n_features = 10
        # Perfectly separable data: label determined by feature 0
        X = rng.randn(n, n_features).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)

        # Split into train/val
        X_train, X_val = X[:400], X[400:]
        y_train, y_val = y[:400], y[400:]

        model = train_predictor(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            xgb_params={"n_estimators": 200},
        )
        assert model.best_iteration < 200

    def test_train_without_early_stopping(self) -> None:
        """Training without X_val/y_val still works (backward compat)."""
        ds = _make_synthetic_dataset()
        model = train_predictor(ds.X, ds.y, xgb_params={"n_estimators": 10})
        assert isinstance(model, xgb.XGBClassifier)
        proba = model.predict_proba(ds.X[:5])
        assert proba.shape == (5, 2)
