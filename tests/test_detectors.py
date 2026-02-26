"""Tests for catastrophe detectors."""

from kvguard.detectors import (
    detect_all,
    detect_answer_failure,
    detect_catastrophe_onsets,
    detect_looping,
    detect_looping_onset,
    detect_non_termination,
    parse_gsm8k_answer,
)


class TestNonTermination:
    def test_max_tokens(self) -> None:
        assert detect_non_termination("max_tokens") is True

    def test_eos(self) -> None:
        assert detect_non_termination("eos") is False


class TestLooping:
    def test_no_looping_short(self) -> None:
        assert detect_looping([1, 2, 3]) is False

    def test_no_looping_varied(self) -> None:
        assert detect_looping(list(range(100))) is False

    def test_detects_exact_repeats(self) -> None:
        # 20-token window repeated 3 times
        pattern = list(range(20))
        token_ids = pattern * 3
        assert detect_looping(token_ids, window_size=20, min_repeats=3) is True

    def test_below_min_repeats(self) -> None:
        pattern = list(range(20))
        token_ids = pattern * 2  # only 2 repeats, need 3
        assert detect_looping(token_ids, window_size=20, min_repeats=3) is False

    def test_small_window(self) -> None:
        # "ToToToTo" — small repeating pattern
        token_ids = [42, 42, 42, 42, 42, 42, 42, 42, 42, 42]
        assert detect_looping(token_ids, window_size=2, min_repeats=3) is True

    def test_empty(self) -> None:
        assert detect_looping([]) is False


class TestLoopingOnset:
    def test_no_looping(self) -> None:
        assert detect_looping_onset(list(range(100))) is None

    def test_short_sequence(self) -> None:
        assert detect_looping_onset([1, 2, 3]) is None

    def test_exact_triple_repeat(self) -> None:
        # Pattern [0..19] repeated 3 times
        pattern = list(range(20))
        token_ids = pattern * 3
        # Onset should be position 20 (start of second occurrence)
        onset = detect_looping_onset(token_ids, window_size=20, min_repeats=3)
        assert onset == 20

    def test_small_window(self) -> None:
        # [1,2,3] repeated 3 times
        token_ids = [1, 2, 3] * 3
        onset = detect_looping_onset(token_ids, window_size=3, min_repeats=3)
        # Second occurrence starts at position 3
        assert onset == 3

    def test_two_repeats_not_enough(self) -> None:
        pattern = list(range(20))
        token_ids = pattern * 2
        assert detect_looping_onset(token_ids, window_size=20, min_repeats=3) is None


class TestCatastropheOnsets:
    def test_looping_onset(self) -> None:
        # Pattern repeated 3 times — looping detected
        pattern = list(range(20))
        token_ids = pattern * 3
        catastrophes = ["looping"]
        onsets = detect_catastrophe_onsets(token_ids, "eos", catastrophes)
        assert "looping" in onsets
        assert onsets["looping"] == 20

    def test_non_termination_onset(self) -> None:
        token_ids = list(range(100))
        catastrophes = ["non_termination"]
        onsets = detect_catastrophe_onsets(token_ids, "max_tokens", catastrophes)
        assert "non_termination" in onsets
        assert onsets["non_termination"] == 99

    def test_no_catastrophes(self) -> None:
        onsets = detect_catastrophe_onsets([1, 2, 3], "eos", [])
        assert onsets == {}

    def test_wrong_answer_has_no_onset(self) -> None:
        onsets = detect_catastrophe_onsets([1, 2, 3], "eos", ["wrong_answer"])
        assert "wrong_answer" not in onsets


class TestLoopingEdgeCases:
    def test_exact_boundary_length(self) -> None:
        """Sequence of exactly window_size * min_repeats tokens."""
        pattern = list(range(20))
        token_ids = pattern * 3  # exactly 60 tokens
        assert detect_looping(token_ids, window_size=20, min_repeats=3) is True

    def test_pattern_after_prefix(self) -> None:
        """Looping that starts after a non-repeating prefix."""
        prefix = list(range(100, 150))  # 50 unique tokens
        pattern = list(range(20))
        token_ids = prefix + pattern * 3
        assert detect_looping(token_ids, window_size=20, min_repeats=3) is True

    def test_onset_after_prefix(self) -> None:
        """Onset should be at the start of the second occurrence after prefix."""
        prefix = list(range(100, 150))
        pattern = list(range(20))
        token_ids = prefix + pattern * 3
        onset = detect_looping_onset(token_ids, window_size=20, min_repeats=3)
        assert onset == 50 + 20  # prefix(50) + first pattern(20) = 70

    def test_no_false_positives_on_similar_windows(self) -> None:
        """Distinct but structurally similar windows should not false-trigger."""
        # Three segments where each 20-token window is unique
        seg1 = list(range(0, 20))  # [0..19]
        seg2 = list(range(100, 120))  # [100..119]
        seg3 = list(range(200, 220))  # [200..219]
        token_ids = seg1 + seg2 + seg3
        assert detect_looping(token_ids, window_size=20, min_repeats=3) is False

    def test_single_token_repeat(self) -> None:
        """Single token repeated many times."""
        token_ids = [42] * 100
        assert detect_looping(token_ids, window_size=20, min_repeats=3) is True


class TestCatastropheOnsetsCombined:
    def test_looping_and_non_termination(self) -> None:
        """Both looping and non-termination detected together."""
        pattern = list(range(20))
        token_ids = pattern * 3
        catastrophes = ["looping", "non_termination"]
        onsets = detect_catastrophe_onsets(token_ids, "max_tokens", catastrophes)
        assert "looping" in onsets
        assert "non_termination" in onsets
        assert onsets["looping"] == 20
        assert onsets["non_termination"] == 59

    def test_single_token_sequence(self) -> None:
        """Single token in sequence — non_termination onset is index 0."""
        onsets = detect_catastrophe_onsets([42], "max_tokens", ["non_termination"])
        assert onsets["non_termination"] == 0


class TestParseGsm8kAnswer:
    def test_hash_format(self) -> None:
        assert parse_gsm8k_answer("The answer is #### 42") == "42"

    def test_hash_with_comma(self) -> None:
        assert parse_gsm8k_answer("#### 1,234") == "1234"

    def test_boxed_format(self) -> None:
        assert parse_gsm8k_answer("Therefore \\boxed{42}") == "42"

    def test_no_answer(self) -> None:
        assert parse_gsm8k_answer("I don't know the answer") is None

    def test_negative(self) -> None:
        assert parse_gsm8k_answer("#### -5") == "-5"

    def test_decimal(self) -> None:
        assert parse_gsm8k_answer("#### 3.14") == "3.14"

    def test_takes_last_hash_not_first(self) -> None:
        """When model mentions #### in reasoning, take the LAST one."""
        text = "The cost per item is #### 15. For 3 items: #### 45"
        assert parse_gsm8k_answer(text) == "45"

    def test_takes_last_boxed_not_first(self) -> None:
        text = "First part \\boxed{10}, final \\boxed{42}"
        assert parse_gsm8k_answer(text) == "42"

    def test_empty_string(self) -> None:
        assert parse_gsm8k_answer("") is None

    def test_zero_answer(self) -> None:
        assert parse_gsm8k_answer("#### 0") == "0"

    def test_large_number(self) -> None:
        assert parse_gsm8k_answer("#### 1,000,000") == "1000000"

    def test_hash_without_number(self) -> None:
        """#### followed by non-numeric text should not match."""
        assert parse_gsm8k_answer("#### the answer") is None


class TestAnswerFailure:
    def test_correct(self) -> None:
        assert detect_answer_failure("#### 42", "42") is False

    def test_wrong(self) -> None:
        assert detect_answer_failure("#### 99", "42") is True

    def test_no_answer(self) -> None:
        assert detect_answer_failure("rambling text", "42") is True

    def test_float_match(self) -> None:
        assert detect_answer_failure("#### 42.0", "42") is False

    def test_invalid_predicted_value(self) -> None:
        """Non-numeric extracted answer should count as failure."""
        assert detect_answer_failure("#### abc", "42") is True

    def test_zero_correct(self) -> None:
        assert detect_answer_failure("#### 0", "0") is False

    def test_negative_correct(self) -> None:
        assert detect_answer_failure("#### -7", "-7") is False


class TestDetectAll:
    def test_no_catastrophes(self) -> None:
        cats = detect_all("The answer is #### 42", [1, 2, 3], "eos", "42")
        assert cats == []

    def test_non_termination_and_wrong(self) -> None:
        cats = detect_all("no answer here", list(range(100)), "max_tokens", "42")
        assert "non_termination" in cats
        assert "wrong_answer" in cats

    def test_looping(self) -> None:
        token_ids = [1, 2] * 30  # 2-token loop, 30 repeats
        cats = detect_all("#### 42", token_ids, "eos", "42")
        assert "looping" in cats

    def test_all_three_catastrophes(self) -> None:
        """All catastrophe types detected simultaneously."""
        token_ids = [1, 2] * 30  # looping
        cats = detect_all("no answer", token_ids, "max_tokens", "42")
        assert "looping" in cats
        assert "non_termination" in cats
        assert "wrong_answer" in cats


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestDetectorEdgeCases:
    """Edge cases for looping detection and answer parsing."""

    def test_looping_window_larger_than_sequence(self) -> None:
        """Window larger than sequence should return False, not crash."""
        assert detect_looping([1, 2, 3], window_size=100, min_repeats=3) is False

    def test_looping_onset_window_larger_than_sequence(self) -> None:
        """Onset with oversized window should return None."""
        assert detect_looping_onset([1, 2, 3], window_size=100, min_repeats=3) is None

    def test_looping_min_repeats_one(self) -> None:
        """min_repeats=1: every window appears at least once, so any
        sequence >= window_size should trigger."""
        assert detect_looping([1, 2, 3, 4, 5], window_size=3, min_repeats=1) is True

    def test_looping_window_size_one(self) -> None:
        """Window of 1 with repeated single token should detect looping."""
        assert detect_looping([7, 7, 7], window_size=1, min_repeats=3) is True
        assert detect_looping([1, 2, 3], window_size=1, min_repeats=3) is False

    def test_detect_all_empty_tokens(self) -> None:
        """Empty token list should not crash; no looping detected."""
        cats = detect_all("#### 42", [], "eos", "42")
        assert "looping" not in cats
        assert cats == []

    def test_answer_failure_non_numeric_ground_truth(self) -> None:
        """Non-numeric ground truth should count as failure (ValueError)."""
        assert detect_answer_failure("#### 42", "abc") is True

    def test_parse_answer_leading_zeros(self) -> None:
        """Leading zeros should still parse correctly."""
        assert parse_gsm8k_answer("#### 007") == "007"

    def test_parse_answer_empty_boxed(self) -> None:
        """\\boxed{} with no content should not match (regex requires digits)."""
        assert parse_gsm8k_answer("\\boxed{}") is None

    def test_catastrophe_onsets_unknown_type(self) -> None:
        """Unknown catastrophe type is silently ignored."""
        onsets = detect_catastrophe_onsets([1, 2, 3], "eos", ["unknown_type"])
        assert onsets == {}

    def test_non_termination_case_sensitive(self) -> None:
        """stop_reason matching is case-sensitive."""
        assert detect_non_termination("Max_Tokens") is False
        assert detect_non_termination("MAX_TOKENS") is False

    def test_non_termination_onset_empty_tokens(self) -> None:
        """Non-termination with empty token_ids should not produce negative onset."""
        onsets = detect_catastrophe_onsets([], "max_tokens", ["non_termination"])
        assert "non_termination" not in onsets
