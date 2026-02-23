"""Integration test: full pipeline from result files to evaluated model."""

import json
from pathlib import Path

import numpy as np

from kvguard.features import build_dataset
from kvguard.train import evaluate_predictor, split_by_prompt, train_predictor


def _make_signal_dict(*, entropy=1.0, rep_count=0, delta_h=0.1, **kw):
    """Minimal signal dict for integration tests."""
    return {
        "entropy": entropy,
        "top1_prob": kw.get("top1_prob", 0.5),
        "top5_prob": kw.get("top5_prob", 0.9),
        "top1_token": "a",
        "rank_of_chosen": 0,
        "top20_logprobs": [-0.5] * 20,
        "h_alts": kw.get("h_alts", 0.3),
        "avg_logp": kw.get("avg_logp", -2.0),
        "delta_h": delta_h,
        "rep_count": rep_count,
        "is_thinking_token": False,
    }


def _make_result(
    *,
    prompt_id: str,
    press: str = "none",
    compression_ratio: float = 0.0,
    n_tokens: int = 50,
    catastrophes: list[str] | None = None,
    catastrophe_onsets: dict[str, int] | None = None,
) -> dict:
    """Build a result dict as it appears in sweep JSON."""
    sigs = []
    for t in range(n_tokens):
        # Catastrophe traces: high entropy + rep after onset
        is_cat = catastrophes and catastrophe_onsets
        onset = min(catastrophe_onsets.values()) if catastrophe_onsets else n_tokens
        if is_cat and t >= onset:
            sigs.append(
                _make_signal_dict(
                    entropy=4.0 + 0.1 * t,
                    rep_count=t - onset + 1,
                    delta_h=1.5,
                )
            )
        else:
            sigs.append(
                _make_signal_dict(
                    entropy=1.0 + 0.01 * t,
                    rep_count=0,
                    delta_h=0.01 if t > 0 else None,
                )
            )
    return {
        "prompt_id": prompt_id,
        "prompt_text": "test",
        "model": "test-model",
        "press": press,
        "compression_ratio": compression_ratio,
        "max_new_tokens": 512,
        "seed": 42,
        "generated_text": "output",
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


def _write_results(tmp_path: Path, press: str, ratio: float, results: list[dict]) -> None:
    subdir = tmp_path / press
    subdir.mkdir(parents=True, exist_ok=True)
    fname = f"test-model_{ratio:.3f}_50p.json"
    data = {
        "config": {"model_name": "test-model", "press_name": press, "compression_ratio": ratio},
        "summary": {},
        "results": results,
    }
    (subdir / fname).write_text(json.dumps(data))


class TestFullPipeline:
    """End-to-end: files -> dataset -> split -> train -> evaluate."""

    def test_pipeline_produces_reasonable_metrics(self, tmp_path: Path) -> None:
        # --- Create diverse results: 4 prompt_ids, 2 (press, ratio) combos ---

        # Combo 1: none / 0.0 — all clean
        _write_results(
            tmp_path,
            "none",
            0.0,
            [
                _make_result(prompt_id="p0", press="none", compression_ratio=0.0, n_tokens=60),
                _make_result(prompt_id="p1", press="none", compression_ratio=0.0, n_tokens=60),
                _make_result(prompt_id="p2", press="none", compression_ratio=0.0, n_tokens=60),
                _make_result(prompt_id="p3", press="none", compression_ratio=0.0, n_tokens=60),
            ],
        )

        # Combo 2: streaming_llm / 0.5 — p0 and p1 have catastrophes, p2 and p3 clean
        _write_results(
            tmp_path,
            "streaming_llm",
            0.500,
            [
                _make_result(
                    prompt_id="p0",
                    press="streaming_llm",
                    compression_ratio=0.5,
                    n_tokens=80,
                    catastrophes=["looping"],
                    catastrophe_onsets={"looping": 30},
                ),
                _make_result(
                    prompt_id="p1",
                    press="streaming_llm",
                    compression_ratio=0.5,
                    n_tokens=80,
                    catastrophes=["looping"],
                    catastrophe_onsets={"looping": 25},
                ),
                _make_result(
                    prompt_id="p2",
                    press="streaming_llm",
                    compression_ratio=0.5,
                    n_tokens=60,
                ),
                _make_result(
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
        _write_results(
            tmp_path,
            "none",
            0.0,
            [
                _make_result(prompt_id="a", press="none", compression_ratio=0.0, n_tokens=30),
                _make_result(prompt_id="b", press="none", compression_ratio=0.0, n_tokens=30),
                _make_result(prompt_id="c", press="none", compression_ratio=0.0, n_tokens=30),
                _make_result(prompt_id="d", press="none", compression_ratio=0.0, n_tokens=30),
            ],
        )
        _write_results(
            tmp_path,
            "snapkv",
            0.500,
            [
                _make_result(
                    prompt_id="a",
                    press="snapkv",
                    compression_ratio=0.5,
                    n_tokens=40,
                    catastrophes=["looping"],
                    catastrophe_onsets={"looping": 20},
                ),
                _make_result(prompt_id="b", press="snapkv", compression_ratio=0.5, n_tokens=40),
                _make_result(
                    prompt_id="c",
                    press="snapkv",
                    compression_ratio=0.5,
                    n_tokens=40,
                    catastrophes=["looping"],
                    catastrophe_onsets={"looping": 15},
                ),
                _make_result(prompt_id="d", press="snapkv", compression_ratio=0.5, n_tokens=40),
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
