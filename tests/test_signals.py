"""Tests for signal extraction — uses CPU tensors, no GPU needed."""

import torch

from kvguard.signals import THINKING_TOKENS, compute_repetition_counts, extract_signals


class FakeTokenizer:
    """Minimal tokenizer stub for testing signal extraction."""

    def __init__(self, vocab: dict[int, str] | None = None) -> None:
        self.vocab = vocab or {0: "the", 1: "so", 2: "cat", 3: "wait"}

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self.vocab.get(t, f"<{t}>") for t in token_ids)


class TestExtractSignals:
    def setup_method(self) -> None:
        self.tokenizer = FakeTokenizer()

    def test_basic_extraction(self) -> None:
        # Sharp distribution: one dominant logit
        logits = torch.zeros(100)
        logits[0] = 10.0  # "the" is highly probable
        sig = extract_signals(logits, chosen_token_id=0, tokenizer=self.tokenizer)

        assert sig.entropy >= 0
        assert sig.top1_prob > 0.5
        assert sig.top5_prob >= sig.top1_prob
        assert sig.rank_of_chosen == 0
        assert sig.delta_h is None  # no prev_entropy
        assert len(sig.top20_logprobs) == 20

    def test_uniform_distribution(self) -> None:
        # Uniform distribution: high entropy
        logits = torch.zeros(1000)
        sig = extract_signals(logits, chosen_token_id=0, tokenizer=self.tokenizer)

        assert sig.entropy > 5  # log(1000) ~ 6.9
        assert sig.top1_prob < 0.01

    def test_delta_h(self) -> None:
        logits = torch.zeros(100)
        logits[0] = 5.0
        sig = extract_signals(
            logits,
            chosen_token_id=0,
            tokenizer=self.tokenizer,
            prev_entropy=1.0,
        )
        assert sig.delta_h is not None
        assert isinstance(sig.delta_h, float)

    def test_h_alts(self) -> None:
        # When top-1 has all the mass, h_alts should be near zero
        logits = torch.full((100,), -100.0)
        logits[0] = 10.0
        sig = extract_signals(logits, chosen_token_id=0, tokenizer=self.tokenizer)
        assert sig.h_alts < 0.1

    def test_thinking_token_detected(self) -> None:
        # Token 1 decodes to "so" which is in THINKING_TOKENS
        logits = torch.zeros(100)
        logits[1] = 10.0
        sig = extract_signals(logits, chosen_token_id=1, tokenizer=self.tokenizer)
        assert sig.is_thinking_token is True

    def test_non_thinking_token(self) -> None:
        # Token 2 decodes to "cat" which is NOT a thinking token
        logits = torch.zeros(100)
        logits[2] = 10.0
        sig = extract_signals(logits, chosen_token_id=2, tokenizer=self.tokenizer)
        assert sig.is_thinking_token is False

    def test_rank_of_non_greedy_choice(self) -> None:
        logits = torch.zeros(100)
        logits[0] = 10.0  # top-1
        logits[5] = 8.0  # top-2
        logits[9] = 6.0  # top-3
        sig = extract_signals(logits, chosen_token_id=5, tokenizer=self.tokenizer)
        assert sig.rank_of_chosen == 1  # second most probable


class TestRepetitionCounts:
    def test_empty(self) -> None:
        assert compute_repetition_counts([]) == []

    def test_short_sequence(self) -> None:
        # Shorter than window — all zeros
        result = compute_repetition_counts([1, 2, 3], window_size=20)
        assert result == [0, 0, 0]

    def test_no_repeats(self) -> None:
        # All unique windows
        token_ids = list(range(40))
        result = compute_repetition_counts(token_ids, window_size=20)
        assert all(c == 0 for c in result)

    def test_exact_repeat(self) -> None:
        # Pattern of 20 tokens repeated twice
        pattern = list(range(20))
        token_ids = pattern + pattern
        result = compute_repetition_counts(token_ids, window_size=20)
        # First occurrence: all zeros (positions 19 through 19)
        assert result[19] == 0
        # Second occurrence: count=1 at the end of second window
        assert result[39] == 1

    def test_triple_repeat(self) -> None:
        pattern = list(range(20))
        token_ids = pattern * 3
        result = compute_repetition_counts(token_ids, window_size=20)
        # End of first occurrence
        assert result[19] == 0
        # End of second occurrence
        assert result[39] == 1
        # End of third occurrence
        assert result[59] == 2

    def test_small_window(self) -> None:
        # Small window for easier reasoning
        token_ids = [1, 2, 3, 1, 2, 3]
        result = compute_repetition_counts(token_ids, window_size=3)
        # Position 2: first [1,2,3] — count=0
        assert result[2] == 0
        # Position 5: second [1,2,3] — count=1
        assert result[5] == 1


class TestThinkingTokens:
    def test_known_thinking_tokens(self) -> None:
        for token in ["so", "wait", "therefore", "hmm", "but", "however"]:
            assert token in THINKING_TOKENS

    def test_non_thinking_tokens(self) -> None:
        for token in ["the", "42", "=", "cat"]:
            assert token not in THINKING_TOKENS
