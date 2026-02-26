"""Tests for the live token-by-token controller module."""

import numpy as np
import pytest
import torch
from transformers import DynamicCache

from kvguard.config import TokenSignals
from kvguard.controller import ControllerConfig, Mode, RiskController
from kvguard.features import (
    BASE_FEATURE_NAMES,
    N_BASE,
    ROLLING_COL_INDICES,
    add_rolling_features,
    flatten_token,
)
from kvguard.live import (
    OnlineFeatureBuilder,
    _cache_seq_len,
    evict_streaming_llm_bulk,
    evict_streaming_llm_step,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cache(n_layers: int, n_heads: int, seq_len: int, head_dim: int) -> DynamicCache:
    """Create a DynamicCache filled with identifiable values."""
    cache = DynamicCache()
    for layer in range(n_layers):
        # Use layer * 1000 + position as values so we can verify which tokens survived
        keys = torch.arange(seq_len, dtype=torch.float32).view(1, 1, seq_len, 1)
        keys = keys.expand(1, n_heads, seq_len, head_dim).clone()
        values = keys.clone()
        cache.update(keys, values, layer)
    return cache


def _make_signals(
    *,
    entropy: float = 1.0,
    top1_prob: float = 0.5,
    top5_prob: float = 0.8,
    h_alts: float = 0.3,
    avg_logp: float = -2.0,
    delta_h: float | None = 0.1,
    rep_count: int = 0,
    is_thinking_token: bool = False,
) -> TokenSignals:
    """Build a synthetic TokenSignals for testing."""
    return TokenSignals(
        entropy=entropy,
        top1_prob=top1_prob,
        top5_prob=top5_prob,
        top1_token="the",
        rank_of_chosen=0,
        top20_logprobs=[-0.5 - i * 0.1 for i in range(20)],
        h_alts=h_alts,
        avg_logp=avg_logp,
        delta_h=delta_h,
        rep_count=rep_count,
        is_thinking_token=is_thinking_token,
    )


# ---------------------------------------------------------------------------
# Cache eviction tests
# ---------------------------------------------------------------------------


class TestEvictBulk:
    def test_correct_shape(self) -> None:
        """Bulk eviction produces n_sink + window_size tokens."""
        cache = _make_cache(n_layers=2, n_heads=4, seq_len=100, head_dim=8)
        evict_streaming_llm_bulk(cache, n_sink=4, window_size=8)
        assert _cache_seq_len(cache) == 12  # 4 + 8

    def test_preserves_sinks(self) -> None:
        """First n_sink tokens are preserved after bulk eviction."""
        cache = _make_cache(n_layers=1, n_heads=1, seq_len=50, head_dim=1)
        evict_streaming_llm_bulk(cache, n_sink=4, window_size=6)
        # Sink positions should be 0, 1, 2, 3
        keys = cache.layers[0].keys  # (1, 1, 10, 1)
        sink_values = keys[0, 0, :4, 0].tolist()
        assert sink_values == [0.0, 1.0, 2.0, 3.0]

    def test_preserves_window(self) -> None:
        """Last window_size tokens are preserved after bulk eviction."""
        cache = _make_cache(n_layers=1, n_heads=1, seq_len=50, head_dim=1)
        evict_streaming_llm_bulk(cache, n_sink=4, window_size=6)
        keys = cache.layers[0].keys
        window_values = keys[0, 0, 4:, 0].tolist()
        # Should be positions 44, 45, 46, 47, 48, 49
        assert window_values == [44.0, 45.0, 46.0, 47.0, 48.0, 49.0]

    def test_noop_when_small(self) -> None:
        """No eviction when cache is already small enough."""
        cache = _make_cache(n_layers=1, n_heads=1, seq_len=10, head_dim=1)
        evict_streaming_llm_bulk(cache, n_sink=4, window_size=8)
        assert _cache_seq_len(cache) == 10  # unchanged


class TestEvictStep:
    def test_removes_one_token(self) -> None:
        """Per-step eviction removes exactly one token."""
        cache = _make_cache(n_layers=2, n_heads=4, seq_len=13, head_dim=8)
        evict_streaming_llm_step(cache, n_sink=4)
        assert _cache_seq_len(cache) == 12

    def test_preserves_sinks_after_multiple(self) -> None:
        """Sinks survive multiple per-step evictions."""
        cache = _make_cache(n_layers=1, n_heads=1, seq_len=20, head_dim=1)
        # First do bulk to get to n_sink + window_size
        evict_streaming_llm_bulk(cache, n_sink=4, window_size=6)
        assert _cache_seq_len(cache) == 10

        # Simulate 5 generation steps (each adds 1 token, then evict 1)
        for i in range(5):
            # Simulate adding a token by expanding the cache
            new_key = torch.tensor([[[[100.0 + i]]]]).expand(1, 1, 1, 1)
            new_val = new_key.clone()
            # Manually append
            cache.layers[0].keys = torch.cat([cache.layers[0].keys, new_key], dim=2)
            cache.layers[0].values = torch.cat([cache.layers[0].values, new_val], dim=2)
            evict_streaming_llm_step(cache, n_sink=4)

        # Sinks should still be the original 0, 1, 2, 3
        keys = cache.layers[0].keys
        sink_values = keys[0, 0, :4, 0].tolist()
        assert sink_values == [0.0, 1.0, 2.0, 3.0]
        assert _cache_seq_len(cache) == 10  # back to steady state


# ---------------------------------------------------------------------------
# Online feature builder tests
# ---------------------------------------------------------------------------


class TestOnlineFeatureBuilder:
    def test_output_dimension(self) -> None:
        """Feature vector is 40-dim."""
        builder = OnlineFeatureBuilder(compression_ratio=0.5)
        sig = _make_signals()
        vec = builder.push(sig, token_id=42)
        assert vec.shape == (N_BASE + len(ROLLING_COL_INDICES) * 2 + 1 + 2,)
        assert vec.shape == (40,)

    def test_matches_offline(self) -> None:
        """Online features match offline add_rolling_features for a sequence."""
        n_tokens = 20
        compression_ratio = 0.75
        max_new_tokens = 512

        # Generate a sequence of signals with varying values
        rng = np.random.RandomState(42)
        signals_list: list[TokenSignals] = []
        for i in range(n_tokens):
            sig = _make_signals(
                entropy=float(rng.uniform(0.5, 4.0)),
                top1_prob=float(rng.uniform(0.1, 0.9)),
                h_alts=float(rng.uniform(0.1, 2.0)),
                delta_h=float(rng.uniform(-1.0, 2.0)) if i > 0 else None,
                rep_count=0,
            )
            signals_list.append(sig)

        # Offline path
        sig_dicts = [s.model_dump() for s in signals_list]
        X_base = np.stack([flatten_token(d) for d in sig_dicts])
        X_offline = add_rolling_features(
            X_base, compression_ratio=compression_ratio, max_new_tokens=max_new_tokens
        )

        # Online path
        builder = OnlineFeatureBuilder(
            compression_ratio=compression_ratio,
            max_new_tokens=max_new_tokens,
        )
        online_rows = []
        for sig in signals_list:
            vec = builder.push(sig, token_id=42)
            online_rows.append(vec)
        X_online = np.stack(online_rows)

        np.testing.assert_allclose(
            X_online,
            X_offline,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Online features do not match offline features",
        )

    def test_compression_ratio_update(self) -> None:
        """Compression ratio updates are reflected in subsequent features."""
        builder = OnlineFeatureBuilder(compression_ratio=0.875)
        sig = _make_signals()
        vec1 = builder.push(sig, token_id=1)
        assert vec1[-1] == pytest.approx(0.875)

        builder.update_compression_ratio(0.0)
        vec2 = builder.push(sig, token_id=2)
        assert vec2[-1] == pytest.approx(0.0)

    def test_reset(self) -> None:
        """Reset clears all state."""
        builder = OnlineFeatureBuilder()
        builder.push(_make_signals(), token_id=1)
        builder.push(_make_signals(), token_id=2)
        builder.reset()
        assert builder._step == 0
        assert len(builder._base_history) == 0


# ---------------------------------------------------------------------------
# Controller step_with_risk tests
# ---------------------------------------------------------------------------


class TestStepWithRisk:
    def test_low_risk_stays_normal(self) -> None:
        """Low risk scores keep the controller in NORMAL mode."""
        ctrl = RiskController(ControllerConfig(k_escalate=3))
        for _ in range(10):
            action = ctrl.step_with_risk(0.1)
        assert action.mode == Mode.NORMAL

    def test_high_risk_escalates(self) -> None:
        """Sustained high risk escalates NORMAL → ALERT → SAFE."""
        ctrl = RiskController(ControllerConfig(tau_low=0.3, tau_high=0.7, k_escalate=3))
        # Pass NORMAL → ALERT
        for _ in range(3):
            action = ctrl.step_with_risk(0.5)
        assert action.mode == Mode.ALERT

        # Pass ALERT → SAFE
        for _ in range(3):
            action = ctrl.step_with_risk(0.9)
        assert action.mode == Mode.SAFE

    def test_matches_evaluate_controller_pattern(self) -> None:
        """step_with_risk produces the same mode transitions as _run_predictor_state_machine."""
        from kvguard.evaluate_controller import _run_predictor_state_machine

        config = ControllerConfig(tau_low=0.3, tau_high=0.7, k_escalate=4, j_deescalate=3)

        # Generate a sequence of hazard probabilities
        probs = [0.1] * 5 + [0.5] * 6 + [0.9] * 6 + [0.1] * 5

        # Offline path
        offline_modes, _ = _run_predictor_state_machine(probs, config)

        # Online path via step_with_risk
        ctrl = RiskController(config)
        online_modes = []
        for p in probs:
            action = ctrl.step_with_risk(p)
            online_modes.append(int(action.mode))

        assert online_modes == offline_modes

    def test_risk_clipped(self) -> None:
        """Risk scores outside [0, 1] are clipped."""
        ctrl = RiskController()
        action = ctrl.step_with_risk(1.5)
        assert action.risk_score <= 1.0
        action = ctrl.step_with_risk(-0.5)
        assert action.risk_score >= 0.0


# ---------------------------------------------------------------------------
# OnlineFeatureBuilder edge cases
# ---------------------------------------------------------------------------


class TestOnlineFeatureBuilderEdgeCases:
    def test_single_token(self) -> None:
        """Feature vector is correct for a single token (rolling window = 1)."""
        builder = OnlineFeatureBuilder(compression_ratio=0.5)
        sig = _make_signals()
        vec = builder.push(sig, token_id=42)
        assert vec.shape == (40,)
        # token_position should be 0/512 = 0 for the first token
        assert vec[-2] == pytest.approx(0.0)
        # compression_ratio should be 0.5
        assert vec[-1] == pytest.approx(0.5)

    def test_rolling_std_zero_for_single_token(self) -> None:
        """Rolling std should be 0 when only 1 token in the window."""
        builder = OnlineFeatureBuilder(compression_ratio=0.5)
        sig = _make_signals(entropy=2.0)
        vec = builder.push(sig, token_id=42)
        # Rolling features start at index N_BASE (29)
        # First rolling feature is entropy_mean, second is entropy_std
        assert vec[N_BASE + 1] == pytest.approx(0.0)  # entropy_std should be 0

    def test_token_position_advances(self) -> None:
        """token_position should increase with each push."""
        builder = OnlineFeatureBuilder(max_new_tokens=100)
        for i in range(5):
            vec = builder.push(_make_signals(), token_id=i)
        # After 5 pushes, position = 4/100 = 0.04
        assert vec[-2] == pytest.approx(4.0 / 100.0)

    def test_rep_count_detects_repetition(self) -> None:
        """Online rep_count matches expected behavior for repeated 20-token windows."""
        builder = OnlineFeatureBuilder()
        # Push 20 tokens (first window) → rep_count = 0
        for i in range(20):
            vec = builder.push(_make_signals(), token_id=i)
        # rep_count is in base features at index where "rep_count" is in BASE_FEATURE_NAMES
        rep_idx = BASE_FEATURE_NAMES.index("rep_count")
        assert vec[rep_idx] == 0  # first occurrence

        # Push the same 20 tokens again → rep_count should be 1 at the 20th
        for i in range(20):
            vec = builder.push(_make_signals(), token_id=i)
        assert vec[rep_idx] == 1  # second occurrence of same window

    def test_reset_clears_rep_history(self) -> None:
        """After reset, repetition counts start fresh."""
        builder = OnlineFeatureBuilder()
        for i in range(25):
            builder.push(_make_signals(), token_id=i)
        builder.reset()
        assert len(builder._window_seen) == 0
        assert len(builder._token_ids) == 0

    def test_different_signals_produce_different_features(self) -> None:
        """Different entropy values produce different feature vectors."""
        builder1 = OnlineFeatureBuilder()
        builder2 = OnlineFeatureBuilder()
        vec1 = builder1.push(_make_signals(entropy=0.5), token_id=1)
        vec2 = builder2.push(_make_signals(entropy=4.0), token_id=1)
        assert vec1[0] != vec2[0]  # entropy is first base feature


# ---------------------------------------------------------------------------
# Cache utility edge cases
# ---------------------------------------------------------------------------


class TestCacheUtilEdgeCases:
    def test_cache_seq_len_empty(self) -> None:
        """Empty cache returns 0."""
        cache = DynamicCache()
        assert _cache_seq_len(cache) == 0

    def test_evict_step_noop_when_small(self) -> None:
        """Per-step eviction is a no-op when cache has n_sink+1 or fewer tokens."""
        cache = _make_cache(n_layers=1, n_heads=1, seq_len=5, head_dim=1)
        evict_streaming_llm_step(cache, n_sink=4)
        assert _cache_seq_len(cache) == 5  # unchanged, only had 5 <= 4+1

    def test_evict_bulk_multi_layer(self) -> None:
        """Bulk eviction works correctly across multiple layers."""
        cache = _make_cache(n_layers=4, n_heads=2, seq_len=50, head_dim=8)
        evict_streaming_llm_bulk(cache, n_sink=2, window_size=4)
        for layer_idx in range(4):
            keys = cache.layers[layer_idx].keys
            assert keys is not None
            assert keys.shape[2] == 6  # 2 sink + 4 window


# ---------------------------------------------------------------------------
# live_experiment.py utilities
# ---------------------------------------------------------------------------


class TestWilsonCI:
    """Tests for the Wilson score confidence interval helper."""

    def test_zero_successes(self) -> None:
        from kvguard.live_experiment import _wilson_ci

        result = _wilson_ci(0, 100)
        assert result != "N/A"
        assert "0%" in result

    def test_all_successes(self) -> None:
        from kvguard.live_experiment import _wilson_ci

        result = _wilson_ci(100, 100)
        assert "100%" in result or "96%" in result  # upper bound may be <100%

    def test_zero_n_returns_na(self) -> None:
        from kvguard.live_experiment import _wilson_ci

        result = _wilson_ci(0, 0)
        assert result == "N/A"

    def test_returns_string(self) -> None:
        from kvguard.live_experiment import _wilson_ci

        result = _wilson_ci(50, 100)
        assert isinstance(result, str)
        assert "-" in result  # format is "XX%-YY%"

    def test_half(self) -> None:
        from kvguard.live_experiment import _wilson_ci

        result = _wilson_ci(50, 100)
        # CI should be around 40%-60% for 50/100
        assert "40%" in result or "41%" in result or "39%" in result
