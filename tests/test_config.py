"""Tests for configuration models."""

from kvguard.config import ExperimentConfig, RunResult, TokenSignals


class TestExperimentConfig:
    def test_defaults(self) -> None:
        cfg = ExperimentConfig()
        assert cfg.model_name == "Qwen/Qwen2.5-3B-Instruct"
        assert cfg.press_name == "streaming_llm"
        assert cfg.compression_ratio == 0.875
        assert cfg.seed == 42

    def test_custom(self) -> None:
        cfg = ExperimentConfig(press_name="snapkv", compression_ratio=0.5)
        assert cfg.press_name == "snapkv"
        assert cfg.compression_ratio == 0.5

    def test_serialization(self) -> None:
        cfg = ExperimentConfig()
        data = cfg.model_dump(mode="json")
        assert data["model_name"] == "Qwen/Qwen2.5-3B-Instruct"
        assert isinstance(data["output_dir"], str)


class TestTokenSignals:
    def test_minimal(self) -> None:
        sig = TokenSignals(
            entropy=1.5,
            top1_prob=0.6,
            top5_prob=0.9,
            top1_token="the",
            rank_of_chosen=0,
        )
        assert sig.entropy == 1.5
        assert sig.delta_h is None
        assert sig.is_thinking_token is False
        assert sig.top20_logprobs == []

    def test_full_halt(self) -> None:
        sig = TokenSignals(
            entropy=1.5,
            top1_prob=0.6,
            top5_prob=0.9,
            top1_token="so",
            rank_of_chosen=0,
            top20_logprobs=[-0.5] * 20,
            h_alts=2.1,
            avg_logp=-8.3,
            delta_h=0.3,
            is_thinking_token=True,
        )
        assert len(sig.top20_logprobs) == 20
        assert sig.delta_h == 0.3
        assert sig.is_thinking_token is True


class TestRunResult:
    def test_creation(self) -> None:
        result = RunResult(
            prompt_id="gsm8k_0",
            prompt_text="Q: test",
            model="test-model",
            press="none",
            compression_ratio=0.0,
            max_new_tokens=512,
            seed=42,
            generated_text="#### 42",
            ground_truth="42",
            predicted_answer="42",
            correct=True,
            stop_reason="eos",
            catastrophes=[],
            num_tokens_generated=10,
            cache_size_after_prefill=None,
            signals=[],
        )
        assert result.correct is True
        assert result.catastrophes == []
