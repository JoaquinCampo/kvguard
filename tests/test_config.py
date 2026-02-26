"""Tests for configuration models."""

from kvguard.config import ExperimentConfig, LiveResult, RunResult, TokenSignals


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

    def test_with_catastrophes(self) -> None:
        result = RunResult(
            prompt_id="gsm8k_1",
            prompt_text="Q: test",
            model="test-model",
            press="snapkv",
            compression_ratio=0.875,
            seed=42,
            generated_text="So So So",
            ground_truth="42",
            predicted_answer=None,
            correct=False,
            stop_reason="max_tokens",
            catastrophes=["looping", "non_termination"],
            catastrophe_onsets={"looping": 50, "non_termination": 512},
            num_tokens_generated=512,
            cache_size_after_prefill=100,
            signals=[],
        )
        assert result.catastrophes == ["looping", "non_termination"]
        assert result.catastrophe_onsets["looping"] == 50
        assert result.cache_size_after_prefill == 100

    def test_round_trip_serialization(self) -> None:
        sig = TokenSignals(
            entropy=1.5,
            top1_prob=0.6,
            top5_prob=0.9,
            top1_token="the",
            rank_of_chosen=0,
        )
        result = RunResult(
            prompt_id="gsm8k_0",
            prompt_text="Q: test",
            model="test-model",
            press="none",
            compression_ratio=0.0,
            seed=42,
            generated_text="#### 42",
            ground_truth="42",
            predicted_answer="42",
            correct=True,
            stop_reason="eos",
            catastrophes=[],
            num_tokens_generated=1,
            cache_size_after_prefill=None,
            signals=[sig],
        )
        data = result.model_dump(mode="json")
        restored = RunResult(**data)
        assert restored.prompt_id == "gsm8k_0"
        assert len(restored.signals) == 1
        assert restored.signals[0].entropy == 1.5


class TestLiveResult:
    def test_creation(self) -> None:
        result = LiveResult(
            prompt_id="p0",
            prompt_text="Q: test",
            model="test-model",
            compression_ratio=0.875,
            seed=42,
            generated_text="#### 42",
            ground_truth="42",
            predicted_answer="42",
            correct=True,
            stop_reason="eos",
            catastrophes=[],
            num_tokens_generated=50,
            signals=[],
            controlled=True,
        )
        assert result.controlled is True
        assert result.safe_trigger_token is None
        assert result.mode_history == []
        assert result.hazard_probs == []

    def test_with_controller_data(self) -> None:
        result = LiveResult(
            prompt_id="p0",
            prompt_text="Q: test",
            model="test-model",
            compression_ratio=0.875,
            seed=42,
            generated_text="output",
            ground_truth="42",
            predicted_answer="42",
            correct=True,
            stop_reason="eos",
            catastrophes=["looping"],
            catastrophe_onsets={"looping": 30},
            num_tokens_generated=100,
            signals=[],
            controlled=True,
            mode_history=[0, 0, 1, 1, 2, 2],
            hazard_probs=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
            eviction_history=[False, False, False, True, True, True],
            cache_sizes=[100, 100, 100, 80, 60, 40],
            safe_trigger_token=4,
            generation_time_seconds=2.5,
        )
        assert result.safe_trigger_token == 4
        assert len(result.mode_history) == 6
        assert result.generation_time_seconds == 2.5

    def test_round_trip_serialization(self) -> None:
        result = LiveResult(
            prompt_id="p0",
            prompt_text="Q: test",
            model="test-model",
            compression_ratio=0.5,
            seed=42,
            generated_text="#### 42",
            ground_truth="42",
            predicted_answer="42",
            correct=True,
            stop_reason="eos",
            catastrophes=[],
            num_tokens_generated=10,
            signals=[],
            controlled=False,
            mode_history=[0, 0, 0],
            hazard_probs=[0.1, 0.2, 0.1],
        )
        data = result.model_dump(mode="json")
        restored = LiveResult(**data)
        assert restored.prompt_id == "p0"
        assert restored.controlled is False
        assert restored.hazard_probs == [0.1, 0.2, 0.1]

    def test_static_run(self) -> None:
        """A static (uncontrolled) run has controlled=False."""
        result = LiveResult(
            prompt_id="p0",
            prompt_text="Q: test",
            model="test-model",
            compression_ratio=0.875,
            seed=42,
            generated_text="output",
            ground_truth="42",
            predicted_answer=None,
            correct=None,
            stop_reason="max_tokens",
            catastrophes=["non_termination"],
            num_tokens_generated=512,
            signals=[],
            controlled=False,
        )
        assert result.correct is None
        assert result.predicted_answer is None
