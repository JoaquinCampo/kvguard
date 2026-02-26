"""Integration test: full pipeline from result files to evaluated model."""

from pathlib import Path

import numpy as np

from kvguard.controller import ControllerConfig
from kvguard.evaluate_controller import EvalResult, evaluate_controller
from kvguard.features import build_dataset
from kvguard.train import evaluate_predictor, split_by_prompt, train_predictor
from tests.helpers import make_result_json, write_result_file


class TestFullPipeline:
    """End-to-end: files -> dataset -> split -> train -> evaluate."""

    def test_pipeline_produces_reasonable_metrics(self, tmp_path: Path) -> None:
        # --- Create diverse results: 4 prompt_ids, 2 (press, ratio) combos ---

        # Combo 1: none / 0.0 — all clean
        write_result_file(
            tmp_path,
            "none",
            0.0,
            [
                make_result_json(prompt_id="p0", press="none", compression_ratio=0.0, n_tokens=60),
                make_result_json(prompt_id="p1", press="none", compression_ratio=0.0, n_tokens=60),
                make_result_json(prompt_id="p2", press="none", compression_ratio=0.0, n_tokens=60),
                make_result_json(prompt_id="p3", press="none", compression_ratio=0.0, n_tokens=60),
            ],
        )

        # Combo 2: streaming_llm / 0.5 — p0 and p1 have catastrophes, p2 and p3 clean
        write_result_file(
            tmp_path,
            "streaming_llm",
            0.500,
            [
                make_result_json(
                    prompt_id="p0",
                    press="streaming_llm",
                    compression_ratio=0.5,
                    n_tokens=80,
                    catastrophes=["looping"],
                    catastrophe_onsets={"looping": 30},
                ),
                make_result_json(
                    prompt_id="p1",
                    press="streaming_llm",
                    compression_ratio=0.5,
                    n_tokens=80,
                    catastrophes=["looping"],
                    catastrophe_onsets={"looping": 25},
                ),
                make_result_json(
                    prompt_id="p2",
                    press="streaming_llm",
                    compression_ratio=0.5,
                    n_tokens=60,
                ),
                make_result_json(
                    prompt_id="p3",
                    press="streaming_llm",
                    compression_ratio=0.5,
                    n_tokens=60,
                ),
            ],
        )

        # --- Build dataset ---
        ds = build_dataset(tmp_path, num_prompts=50, horizon=16)

        # 4*60 (none) + 2*80 + 2*60 (streaming_llm) = 240 + 160 + 120 = 520
        expected_tokens = 4 * 60 + 2 * 80 + 2 * 60
        assert ds.X.shape[0] == expected_tokens
        assert ds.y.shape[0] == expected_tokens
        assert ds.trace_ids.shape[0] == expected_tokens
        assert len(ds.traces) == 8  # 4 + 4

        # Should have both positive and negative labels
        assert ds.y.sum() > 0
        assert (ds.y == 0).sum() > 0

        # --- Split by prompt ---
        split = split_by_prompt(ds, val_fraction=0.3, random_state=42)

        # All tokens assigned, no overlap
        assert np.all(split.train_mask | split.val_mask)
        assert not np.any(split.train_mask & split.val_mask)

        # No prompt in both partitions
        train_prompts = {ds.traces[t].prompt_id for t in split.train_traces}
        val_prompts = {ds.traces[t].prompt_id for t in split.val_traces}
        assert train_prompts & val_prompts == set()

        X_train = ds.X[split.train_mask]
        y_train = ds.y[split.train_mask]
        X_val = ds.X[split.val_mask]
        y_val = ds.y[split.val_mask]

        # --- Train predictor (small for speed) ---
        model = train_predictor(
            X_train,
            y_train,
            xgb_params={"n_estimators": 20},
        )

        # Model should produce probabilities
        proba = model.predict_proba(X_val)
        assert proba.shape == (len(y_val), 2)

        # --- Evaluate on train set (should fit well on separable data) ---
        train_metrics = evaluate_predictor(model, X_train, y_train)
        assert train_metrics.n_samples == len(y_train)

        # On train set with clearly separable signals, AUROC should be > 0.5
        if train_metrics.n_positive > 0:
            assert train_metrics.auroc > 0.5

        assert 0.0 <= train_metrics.precision <= 1.0
        assert 0.0 <= train_metrics.recall <= 1.0

        # --- Evaluate on val set ---
        val_metrics = evaluate_predictor(model, X_val, y_val)
        assert val_metrics.n_samples == len(y_val)
        assert 0.0 <= val_metrics.precision <= 1.0
        assert 0.0 <= val_metrics.recall <= 1.0

        # Val AUROC: with separable signals it should exceed random chance
        if val_metrics.n_positive > 0:
            assert val_metrics.auroc > 0.5

    def test_dataset_to_split_consistency(self, tmp_path: Path) -> None:
        """build_dataset -> split_by_prompt -> verify all tokens assigned, no overlap."""
        # Create results with 4 prompts across 2 configs
        write_result_file(
            tmp_path,
            "none",
            0.0,
            [
                make_result_json(prompt_id="a", press="none", compression_ratio=0.0, n_tokens=30),
                make_result_json(prompt_id="b", press="none", compression_ratio=0.0, n_tokens=30),
                make_result_json(prompt_id="c", press="none", compression_ratio=0.0, n_tokens=30),
                make_result_json(prompt_id="d", press="none", compression_ratio=0.0, n_tokens=30),
            ],
        )
        write_result_file(
            tmp_path,
            "snapkv",
            0.500,
            [
                make_result_json(
                    prompt_id="a",
                    press="snapkv",
                    compression_ratio=0.5,
                    n_tokens=40,
                    catastrophes=["looping"],
                    catastrophe_onsets={"looping": 20},
                ),
                make_result_json(prompt_id="b", press="snapkv", compression_ratio=0.5, n_tokens=40),
                make_result_json(
                    prompt_id="c",
                    press="snapkv",
                    compression_ratio=0.5,
                    n_tokens=40,
                    catastrophes=["looping"],
                    catastrophe_onsets={"looping": 15},
                ),
                make_result_json(prompt_id="d", press="snapkv", compression_ratio=0.5, n_tokens=40),
            ],
        )

        ds = build_dataset(tmp_path, num_prompts=50, horizon=16)
        total_tokens = ds.X.shape[0]

        # Expected: 4*30 + 4*40 = 280
        assert total_tokens == 4 * 30 + 4 * 40

        split = split_by_prompt(ds, val_fraction=0.25, random_state=7)

        # Every token is in exactly one partition
        assert np.all(split.train_mask | split.val_mask), "Some tokens not assigned"
        assert not np.any(split.train_mask & split.val_mask), "Some tokens in both partitions"

        # Token counts match
        assert int(split.train_mask.sum()) + int(split.val_mask.sum()) == total_tokens

        # All trace indices appear in exactly one partition
        all_assigned = sorted(split.train_traces + split.val_traces)
        assert all_assigned == list(range(len(ds.traces)))

        # Prompt-level integrity: all traces for a prompt in the same partition
        val_set = set(split.val_traces)
        for prompt_id in {t.prompt_id for t in ds.traces}:
            trace_idxs = [t.trace_idx for t in ds.traces if t.prompt_id == prompt_id]
            in_val = [i in val_set for i in trace_idxs]
            assert all(in_val) or not any(in_val), f"Prompt {prompt_id} split across partitions"


class TestControllerEvaluationPipeline:
    """End-to-end: files -> dataset -> train -> evaluate_controller."""

    def _write_sweep_data(self, tmp_path: Path) -> None:
        """Create synthetic sweep data with baseline + compressed traces."""
        prompts = [f"p{i}" for i in range(8)]

        # Baseline (none, 0.0) — all correct
        write_result_file(
            tmp_path,
            "none",
            0.0,
            [
                make_result_json(prompt_id=pid, press="none", compression_ratio=0.0, n_tokens=50)
                for pid in prompts
            ],
        )

        # Compressed (streaming_llm, 0.875) — some catastrophes
        compressed = []
        for i, pid in enumerate(prompts):
            if i < 3:
                # Catastrophe traces with clear signal differentiation
                compressed.append(
                    make_result_json(
                        prompt_id=pid,
                        press="streaming_llm",
                        compression_ratio=0.875,
                        n_tokens=50,
                        catastrophes=["looping"],
                        catastrophe_onsets={"looping": 30},
                        correct=False,
                    )
                )
            else:
                # Clean traces
                compressed.append(
                    make_result_json(
                        prompt_id=pid,
                        press="streaming_llm",
                        compression_ratio=0.875,
                        n_tokens=50,
                    )
                )
        write_result_file(tmp_path, "streaming_llm", 0.875, compressed)

    def test_train_then_evaluate_controller(self, tmp_path: Path) -> None:
        """Full pipeline: build dataset -> train predictor -> evaluate controller."""
        self._write_sweep_data(tmp_path)

        # Build dataset and train predictor
        ds = build_dataset(tmp_path, num_prompts=50, horizon=16)
        split = split_by_prompt(ds, val_fraction=0.3, random_state=42)

        X_train = ds.X[split.train_mask]
        y_train = ds.y[split.train_mask]

        predictor = train_predictor(X_train, y_train, xgb_params={"n_estimators": 20})

        # Get holdout prompt IDs for controller evaluation
        holdout_prompts = {ds.traces[i].prompt_id for i in split.val_traces}

        # Evaluate controller
        config = ControllerConfig(
            safe_compression_ratio=0.0,
            tau_low=0.3,
            tau_high=0.7,
            k_escalate=3,
            j_deescalate=5,
        )
        result = evaluate_controller(
            tmp_path,
            predictor,
            num_prompts=50,
            controller_config=config,
            holdout_prompt_ids=holdout_prompts,
        )

        # Basic structural checks
        assert isinstance(result, EvalResult)
        assert result.safe_compression_ratio == 0.0

        # Should have exactly 1 budget (streaming_llm at 0.875)
        assert len(result.budgets) == 1
        b = result.budgets[0]
        assert b.press == "streaming_llm"
        assert b.compression_ratio == 0.875

        # Metrics should be internally consistent
        assert b.n_prompts > 0
        assert b.controlled_cfr_count == b.baseline_cfr_count - b.catastrophes_prevented
        assert 0.0 <= b.baseline_cfr <= 1.0
        assert 0.0 <= b.controlled_cfr <= 1.0
        assert b.false_trigger_count >= 0
        assert b.catastrophes_prevented >= 0
        assert b.catastrophes_prevented <= b.baseline_cfr_count
