"""Tests for feature extraction pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest

from kvguard.features import (
    BASE_FEATURE_NAMES,
    N_BASE,
    Dataset,
    add_rolling_features,
    build_dataset,
    feature_names,
    flatten_signals,
    flatten_token,
)

# ---------------------------------------------------------------------------
# Helpers
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
    n_tokens: int = 10,
    press: str = "none",
    compression_ratio: float = 0.0,
    catastrophes: list[str] | None = None,
    catastrophe_onsets: dict[str, int] | None = None,
    prompt_id: str = "gsm8k_0",
) -> dict:
    """Build a full result dict as it would appear in JSON."""
    sigs = []
    for t in range(n_tokens):
        sigs.append(
            _make_signal_dict(
                entropy=1.0 + 0.01 * t,
                delta_h=0.01 if t > 0 else None,
                rep_count=max(0, t - 8),  # repetition starts at t=9
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
    """Write a sweep result JSON to a temp directory."""
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


# ---------------------------------------------------------------------------
# Tests: flatten_token
# ---------------------------------------------------------------------------


class TestFlattenToken:
    def test_shape(self) -> None:
        sig = _make_signal_dict()
        vec = flatten_token(sig)
        assert vec.shape == (N_BASE,)
        assert vec.dtype == np.float32

    def test_values(self) -> None:
        sig = _make_signal_dict(entropy=2.5, top1_prob=0.7, rep_count=3)
        vec = flatten_token(sig)
        assert vec[0] == pytest.approx(2.5)  # entropy
        assert vec[1] == pytest.approx(0.7)  # top1_prob
        assert vec[BASE_FEATURE_NAMES.index("rep_count")] == pytest.approx(3.0)

    def test_delta_h_none_fills_zero(self) -> None:
        sig = _make_signal_dict(delta_h=None)
        vec = flatten_token(sig)
        dh_idx = BASE_FEATURE_NAMES.index("delta_h")
        dh_valid_idx = BASE_FEATURE_NAMES.index("delta_h_valid")
        assert vec[dh_idx] == pytest.approx(0.0)
        assert vec[dh_valid_idx] == pytest.approx(0.0)

    def test_delta_h_present(self) -> None:
        sig = _make_signal_dict(delta_h=0.5)
        vec = flatten_token(sig)
        dh_idx = BASE_FEATURE_NAMES.index("delta_h")
        dh_valid_idx = BASE_FEATURE_NAMES.index("delta_h_valid")
        assert vec[dh_idx] == pytest.approx(0.5)
        assert vec[dh_valid_idx] == pytest.approx(1.0)

    def test_short_logprobs_padded(self) -> None:
        sig = _make_signal_dict(top20_logprobs=[-1.0, -2.0])
        vec = flatten_token(sig)
        lp_start = BASE_FEATURE_NAMES.index("logprob_0")
        assert vec[lp_start] == pytest.approx(-1.0)
        assert vec[lp_start + 1] == pytest.approx(-2.0)
        assert vec[lp_start + 2] == pytest.approx(0.0)  # padded

    def test_empty_logprobs_padded(self) -> None:
        sig = _make_signal_dict(top20_logprobs=[])
        vec = flatten_token(sig)
        lp_start = BASE_FEATURE_NAMES.index("logprob_0")
        assert all(vec[lp_start + i] == pytest.approx(0.0) for i in range(20))

    def test_is_thinking_token_flag(self) -> None:
        sig_false = _make_signal_dict(is_thinking_token=False)
        sig_true = _make_signal_dict(is_thinking_token=True)
        idx = BASE_FEATURE_NAMES.index("is_thinking_token")
        assert flatten_token(sig_false)[idx] == pytest.approx(0.0)
        assert flatten_token(sig_true)[idx] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests: flatten_signals
# ---------------------------------------------------------------------------


class TestFlattenSignals:
    def test_shape(self) -> None:
        sigs = [_make_signal_dict() for _ in range(5)]
        X = flatten_signals(sigs)
        assert X.shape == (5, N_BASE)

    def test_empty(self) -> None:
        X = flatten_signals([])
        assert X.shape == (0, N_BASE)

    def test_values_match_individual(self) -> None:
        sigs = [_make_signal_dict(entropy=float(i)) for i in range(3)]
        X = flatten_signals(sigs)
        for i, sig in enumerate(sigs):
            np.testing.assert_array_almost_equal(X[i], flatten_token(sig))


# ---------------------------------------------------------------------------
# Tests: add_rolling_features
# ---------------------------------------------------------------------------


class TestRollingFeatures:
    def test_output_shape(self) -> None:
        X_base = flatten_signals([_make_signal_dict() for _ in range(20)])
        X_full = add_rolling_features(X_base, window=8)
        # 4 cols × 2 stats + 1 rep_count_sum + 1 token_position + 1 compression_ratio = 11 extra
        expected_cols = N_BASE + 4 * 2 + 1 + 1 + 1
        assert X_full.shape == (20, expected_cols)

    def test_single_token(self) -> None:
        X_base = flatten_signals([_make_signal_dict()])
        X_full = add_rolling_features(X_base, window=8)
        assert X_full.shape[0] == 1
        # token_position for single token should be 0 (second-to-last col)
        assert X_full[0, -2] == pytest.approx(0.0)
        # compression_ratio defaults to 0.0 (last col)
        assert X_full[0, -1] == pytest.approx(0.0)

    def test_token_position_range(self) -> None:
        n = 10
        X_base = flatten_signals([_make_signal_dict() for _ in range(n)])
        X_full = add_rolling_features(X_base, window=8)
        pos = X_full[:, -2]  # token_position is second-to-last
        assert pos[0] == pytest.approx(0.0)
        assert pos[-1] == pytest.approx(1.0)
        # compression_ratio is constant across trace
        assert X_full[0, -1] == pytest.approx(0.0)
        assert X_full[-1, -1] == pytest.approx(0.0)

    def test_rolling_mean_constant_signal(self) -> None:
        """If entropy is constant, rolling mean should equal the constant."""
        n = 20
        sigs = [_make_signal_dict(entropy=3.0) for _ in range(n)]
        X_base = flatten_signals(sigs)
        X_full = add_rolling_features(X_base, window=8)
        # First rolling feature after base columns is entropy_mean_8
        entropy_mean_idx = N_BASE
        np.testing.assert_array_almost_equal(X_full[:, entropy_mean_idx], np.full(n, 3.0))

    def test_rolling_std_constant_is_zero(self) -> None:
        """Rolling std of constant signal should be ~0."""
        n = 20
        sigs = [_make_signal_dict(entropy=3.0) for _ in range(n)]
        X_base = flatten_signals(sigs)
        X_full = add_rolling_features(X_base, window=8)
        entropy_std_idx = N_BASE + 1
        np.testing.assert_array_almost_equal(X_full[:, entropy_std_idx], np.zeros(n), decimal=5)


# ---------------------------------------------------------------------------
# Tests: feature_names
# ---------------------------------------------------------------------------


class TestFeatureNames:
    def test_count_matches_data(self) -> None:
        names = feature_names(window=8)
        sigs = [_make_signal_dict() for _ in range(5)]
        X_base = flatten_signals(sigs)
        X_full = add_rolling_features(X_base, window=8)
        assert len(names) == X_full.shape[1]

    def test_base_features_prefix(self) -> None:
        names = feature_names()
        assert names[0] == "entropy"
        assert names[1] == "top1_prob"
        assert "token_position" in names


# ---------------------------------------------------------------------------
# Tests: build_dataset
# ---------------------------------------------------------------------------


class TestBuildDataset:
    def test_basic_loading(self, tmp_path: Path) -> None:
        results = [_make_result_json(n_tokens=10, press="none")]
        _write_result_file(tmp_path, "none", 0.0, results)
        ds = build_dataset(tmp_path, num_prompts=50, horizon=32)
        assert isinstance(ds, Dataset)
        assert ds.X.shape[0] == 10
        assert ds.y.shape[0] == 10
        assert len(ds.traces) == 1
        assert ds.traces[0].press == "none"

    def test_multiple_files(self, tmp_path: Path) -> None:
        r1 = [_make_result_json(n_tokens=10, press="none")]
        r2 = [_make_result_json(n_tokens=15, press="streaming_llm", compression_ratio=0.5)]
        _write_result_file(tmp_path, "none", 0.0, r1)
        _write_result_file(tmp_path, "streaming_llm", 0.5, r2)
        ds = build_dataset(tmp_path, num_prompts=50, horizon=32)
        assert ds.X.shape[0] == 25  # 10 + 15
        assert len(ds.traces) == 2

    def test_trace_ids_correct(self, tmp_path: Path) -> None:
        results = [
            _make_result_json(n_tokens=5, prompt_id="a"),
            _make_result_json(n_tokens=8, prompt_id="b"),
        ]
        _write_result_file(tmp_path, "none", 0.0, results)
        ds = build_dataset(tmp_path, num_prompts=50, horizon=32)
        assert ds.X.shape[0] == 13
        assert np.sum(ds.trace_ids == 0) == 5
        assert np.sum(ds.trace_ids == 1) == 8

    def test_catastrophe_labeling(self, tmp_path: Path) -> None:
        results = [
            _make_result_json(
                n_tokens=100,
                catastrophes=["looping"],
                catastrophe_onsets={"looping": 50},
            )
        ]
        _write_result_file(tmp_path, "snapkv", 0.5, results)
        ds = build_dataset(tmp_path, num_prompts=50, horizon=32)
        # onset=50, H=32 → tokens 18..99 should be 1
        assert ds.y[:18].sum() == 0
        assert ds.y[18:].sum() == 82
        assert ds.traces[0].has_catastrophe is True
        assert ds.traces[0].catastrophe_types == ["looping"]

    def test_no_catastrophe_all_zeros(self, tmp_path: Path) -> None:
        results = [_make_result_json(n_tokens=50)]
        _write_result_file(tmp_path, "none", 0.0, results)
        ds = build_dataset(tmp_path, num_prompts=50, horizon=32)
        assert ds.y.sum() == 0

    def test_wrong_answer_excluded(self, tmp_path: Path) -> None:
        results = [_make_result_json(n_tokens=50, catastrophes=["wrong_answer"])]
        _write_result_file(tmp_path, "none", 0.0, results)
        ds = build_dataset(tmp_path, num_prompts=50, horizon=32)
        assert ds.y.sum() == 0
        assert ds.traces[0].has_catastrophe is False

    def test_model_field_populated(self, tmp_path: Path) -> None:
        results = [_make_result_json(n_tokens=10, press="none")]
        _write_result_file(tmp_path, "none", 0.0, results)
        ds = build_dataset(tmp_path, num_prompts=50, horizon=32)
        assert ds.traces[0].model == "test-model"

    def test_no_files_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            build_dataset(tmp_path, num_prompts=50)

    def test_feature_names_match_columns(self, tmp_path: Path) -> None:
        results = [_make_result_json(n_tokens=20)]
        _write_result_file(tmp_path, "none", 0.0, results)
        ds = build_dataset(tmp_path, num_prompts=50, horizon=32)
        assert len(ds.feature_names) == ds.X.shape[1]


# ---------------------------------------------------------------------------
# Tests: rank_of_chosen removal (T2)
# ---------------------------------------------------------------------------


def test_rank_of_chosen_not_in_base_features():
    """rank_of_chosen is excluded from default feature set (always 0 under greedy)."""
    assert "rank_of_chosen" not in BASE_FEATURE_NAMES


def test_base_feature_count_is_29():
    assert N_BASE == 29


def test_full_feature_count_is_40():
    names = feature_names()
    assert len(names) == 40


# ---------------------------------------------------------------------------
# Tests: token_position fix (T3)
# ---------------------------------------------------------------------------


class TestTokenPositionFix:
    def test_token_position_uses_max_tokens_not_trace_length(self):
        """Token position should be t / max_new_tokens, not t / (n-1)."""
        X_base = np.random.randn(100, N_BASE).astype(np.float32)
        X_full = add_rolling_features(X_base, max_new_tokens=512)
        pos_col = X_full[:, -2]  # token_position is second-to-last column
        # At token 0: 0/512 = 0.0
        assert pos_col[0] == pytest.approx(0.0)
        # At token 99: 99/512 ≈ 0.1934, NOT 99/99 = 1.0
        assert pos_col[99] == pytest.approx(99 / 512, abs=1e-5)
        assert pos_col[99] < 0.2  # definitely not 1.0


# ---------------------------------------------------------------------------
# Tests: feature ablation (T5)
# ---------------------------------------------------------------------------


class TestFeatureExclusion:
    @pytest.fixture(autouse=True)
    def _setup(self):
        """Build a small synthetic dataset for drop_features tests."""
        rng = np.random.RandomState(42)
        from kvguard.features import TraceMeta

        n_tok = 50
        names = feature_names()
        self.ds = Dataset(
            X=rng.randn(n_tok, len(names)).astype(np.float32),
            y=np.zeros(n_tok, dtype=np.float32),
            trace_ids=np.zeros(n_tok, dtype=np.int32),
            feature_names=list(names),
            traces=[
                TraceMeta(
                    trace_idx=0,
                    prompt_id="gsm8k_0",
                    press="none",
                    compression_ratio=0.0,
                    has_catastrophe=False,
                    catastrophe_types=[],
                    n_tokens=n_tok,
                    model="test",
                )
            ],
        )

    def test_drop_compression_ratio(self):
        ds = self.ds.drop_features(["compression_ratio"])
        assert "compression_ratio" not in ds.feature_names
        assert ds.X.shape[1] == self.ds.X.shape[1] - 1

    def test_drop_rep_features(self):
        ds = self.ds.drop_features(["rep_count", "rep_count_sum_8"])
        assert "rep_count" not in ds.feature_names
        assert "rep_count_sum_8" not in ds.feature_names
        assert ds.X.shape[1] == self.ds.X.shape[1] - 2

    def test_drop_nonexistent_raises(self):
        with pytest.raises(KeyError):
            self.ds.drop_features(["nonexistent_feature"])

    def test_drop_preserves_other_data(self):
        ds = self.ds.drop_features(["compression_ratio"])
        assert len(ds.traces) == len(self.ds.traces)
        np.testing.assert_array_equal(ds.y, self.ds.y)
        np.testing.assert_array_equal(ds.trace_ids, self.ds.trace_ids)
