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


class TestAnswerFailure:
    def test_correct(self) -> None:
        assert detect_answer_failure("#### 42", "42") is False

    def test_wrong(self) -> None:
        assert detect_answer_failure("#### 99", "42") is True

    def test_no_answer(self) -> None:
        assert detect_answer_failure("rambling text", "42") is True

    def test_float_match(self) -> None:
        assert detect_answer_failure("#### 42.0", "42") is False


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
