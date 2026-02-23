"""Tests for hazard labeling."""

from kvguard.config import RunResult, TokenSignals
from kvguard.labeling import compute_hazard_labels, compute_onset_position


def _make_result(
    *,
    n_tokens: int = 100,
    max_new_tokens: int = 512,
    catastrophes: list[str] | None = None,
    catastrophe_onsets: dict[str, int] | None = None,
    stop_reason: str = "eos",
) -> RunResult:
    """Helper to build a minimal RunResult for labeling tests."""
    return RunResult(
        prompt_id="test",
        prompt_text="",
        model="test-model",
        press="none",
        compression_ratio=0.0,
        max_new_tokens=max_new_tokens,
        seed=42,
        generated_text="",
        ground_truth="42",
        predicted_answer="42",
        correct=True,
        stop_reason=stop_reason,
        catastrophes=catastrophes or [],
        num_tokens_generated=n_tokens,
        cache_size_after_prefill=None,
        catastrophe_onsets=catastrophe_onsets or {},
        signals=[
            TokenSignals(
                entropy=0.0,
                top1_prob=1.0,
                top5_prob=1.0,
                top1_token="a",
                rank_of_chosen=0,
            )
        ]
        * n_tokens,
    )


class TestNoCatastrophe:
    def test_all_zeros(self) -> None:
        result = _make_result(n_tokens=100)
        labels = compute_hazard_labels(result, horizon=32)
        assert labels == [0] * 100

    def test_wrong_answer_only_is_ignored(self) -> None:
        result = _make_result(n_tokens=100, catastrophes=["wrong_answer"])
        labels = compute_hazard_labels(result, horizon=32)
        assert labels == [0] * 100


class TestLooping:
    def test_basic_looping(self) -> None:
        result = _make_result(
            n_tokens=100,
            catastrophes=["looping"],
            catastrophe_onsets={"looping": 50},
        )
        labels = compute_hazard_labels(result, horizon=32)
        # Tokens 0..17 = 0, tokens 18..99 = 1
        assert all(v == 0 for v in labels[:18])
        assert all(v == 1 for v in labels[18:])

    def test_onset_early_clamps_to_zero(self) -> None:
        result = _make_result(
            n_tokens=100,
            catastrophes=["looping"],
            catastrophe_onsets={"looping": 10},
        )
        labels = compute_hazard_labels(result, horizon=32)
        # onset=10, H=32 → start=max(0, 10-32)=0 → all tokens labeled 1
        assert all(v == 1 for v in labels)

    def test_small_horizon(self) -> None:
        result = _make_result(
            n_tokens=100,
            catastrophes=["looping"],
            catastrophe_onsets={"looping": 50},
        )
        labels = compute_hazard_labels(result, horizon=16)
        # Tokens 0..33 = 0, tokens 34..99 = 1
        assert all(v == 0 for v in labels[:34])
        assert all(v == 1 for v in labels[34:])


class TestNonTermination:
    def test_proxy_onset(self) -> None:
        result = _make_result(
            n_tokens=512,
            max_new_tokens=512,
            catastrophes=["non_termination"],
            stop_reason="max_tokens",
        )
        labels = compute_hazard_labels(result, horizon=32, nt_onset_frac=0.75)
        # proxy_onset = int(0.75 * 512) = 384
        # start = 384 - 32 = 352
        assert all(v == 0 for v in labels[:352])
        assert all(v == 1 for v in labels[352:])

    def test_proxy_onset_clamped_to_sequence(self) -> None:
        # Shorter sequence than max_new_tokens implies
        result = _make_result(
            n_tokens=200,
            max_new_tokens=512,
            catastrophes=["non_termination"],
            stop_reason="max_tokens",
        )
        labels = compute_hazard_labels(result, horizon=32, nt_onset_frac=0.75)
        # proxy_onset = min(384, 199) = 199
        # start = 199 - 32 = 167
        assert all(v == 0 for v in labels[:167])
        assert all(v == 1 for v in labels[167:])

    def test_custom_nt_fraction(self) -> None:
        result = _make_result(
            n_tokens=512,
            max_new_tokens=512,
            catastrophes=["non_termination"],
            stop_reason="max_tokens",
        )
        labels = compute_hazard_labels(result, horizon=32, nt_onset_frac=0.80)
        # proxy = int(0.80 * 512) = 409, start = 409 - 32 = 377
        assert all(v == 0 for v in labels[:377])
        assert all(v == 1 for v in labels[377:])


class TestBothCatastrophes:
    def test_earliest_onset_wins(self) -> None:
        # Looping at 50, non_termination proxy at 384 → looping onset is earlier
        result = _make_result(
            n_tokens=512,
            max_new_tokens=512,
            catastrophes=["looping", "non_termination"],
            catastrophe_onsets={"looping": 50},
            stop_reason="max_tokens",
        )
        labels = compute_hazard_labels(result, horizon=32)
        # earliest = 50, start = 50 - 32 = 18
        assert all(v == 0 for v in labels[:18])
        assert all(v == 1 for v in labels[18:])


class TestHorizonSweep:
    def test_different_horizons(self) -> None:
        result = _make_result(
            n_tokens=200,
            catastrophes=["looping"],
            catastrophe_onsets={"looping": 100},
        )
        for h in (16, 32, 64, 128):
            labels = compute_hazard_labels(result, horizon=h)
            expected_start = max(0, 100 - h)
            assert labels[expected_start] == 1
            if expected_start > 0:
                assert labels[expected_start - 1] == 0


class TestEdgeCases:
    def test_single_token(self) -> None:
        result = _make_result(n_tokens=1)
        labels = compute_hazard_labels(result, horizon=32)
        assert labels == [0]

    def test_onset_at_last_token(self) -> None:
        result = _make_result(
            n_tokens=100,
            catastrophes=["looping"],
            catastrophe_onsets={"looping": 99},
        )
        labels = compute_hazard_labels(result, horizon=32)
        # start = 99 - 32 = 67
        assert all(v == 0 for v in labels[:67])
        assert all(v == 1 for v in labels[67:])

    def test_looping_in_catastrophes_but_no_onset_key(self) -> None:
        # Edge case: looping detected but onset could not be found
        result = _make_result(
            n_tokens=100,
            catastrophes=["looping"],
            catastrophe_onsets={},  # no looping key
        )
        labels = compute_hazard_labels(result, horizon=32)
        # No applicable onset → all zeros
        assert labels == [0] * 100


# ---------------------------------------------------------------------------
# Tests: compute_onset_position
# ---------------------------------------------------------------------------


class TestComputeOnsetPosition:
    def test_compute_onset_position_looping(self) -> None:
        """Looping onset returns that position."""
        result = _make_result(
            n_tokens=100,
            catastrophes=["looping"],
            catastrophe_onsets={"looping": 50},
        )
        assert compute_onset_position(result) == 50

    def test_compute_onset_position_non_termination(self) -> None:
        """Non-termination returns proxy position."""
        result = _make_result(
            n_tokens=512,
            max_new_tokens=512,
            catastrophes=["non_termination"],
            stop_reason="max_tokens",
        )
        # proxy = int(0.75 * 512) = 384
        assert compute_onset_position(result, nt_onset_frac=0.75) == 384

    def test_compute_onset_position_both(self) -> None:
        """Both looping and non_termination returns min of both."""
        result = _make_result(
            n_tokens=512,
            max_new_tokens=512,
            catastrophes=["looping", "non_termination"],
            catastrophe_onsets={"looping": 50},
            stop_reason="max_tokens",
        )
        # looping=50, nt_proxy=384 → min=50
        assert compute_onset_position(result) == 50

    def test_compute_onset_position_none(self) -> None:
        """No applicable catastrophe returns None."""
        result = _make_result(n_tokens=100)
        assert compute_onset_position(result) is None

    def test_compute_onset_position_wrong_answer_ignored(self) -> None:
        """wrong_answer only returns None."""
        result = _make_result(
            n_tokens=100,
            catastrophes=["wrong_answer"],
        )
        assert compute_onset_position(result) is None
