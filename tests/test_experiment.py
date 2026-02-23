"""Tests for experiment.py — functions that don't require GPU/model loading."""

import json
from pathlib import Path

from kvguard.config import ExperimentConfig, RunResult, TokenSignals
from kvguard.experiment import (
    SWEEP_METHODS,
    _append_checkpoint,
    _clear_checkpoint,
    _load_checkpoint,
    build_sweep_configs,
    result_exists,
    save_results,
    summarize,
)


def _make_signals(n: int = 10) -> list[TokenSignals]:
    return [
        TokenSignals(entropy=1.0, top1_prob=0.5, top5_prob=0.9, top1_token="a", rank_of_chosen=0)
    ] * n


def _make_result(
    prompt_id: str = "test_0",
    correct: bool | None = True,
    catastrophes: list[str] | None = None,
    num_tokens: int = 10,
) -> RunResult:
    return RunResult(
        prompt_id=prompt_id,
        prompt_text="test",
        model="test-model",
        press="none",
        compression_ratio=0.0,
        max_new_tokens=512,
        seed=42,
        generated_text="The answer is 42",
        ground_truth="42",
        predicted_answer="42",
        correct=correct,
        stop_reason="eos",
        catastrophes=catastrophes or [],
        num_tokens_generated=num_tokens,
        cache_size_after_prefill=None,
        catastrophe_onsets={},
        signals=_make_signals(num_tokens),
    )


def _make_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        model_name="test/test-model",
        press_name="none",
        compression_ratio=0.0,
        num_prompts=5,
        seed=42,
        output_dir=tmp_path / "results",
    )


class TestBuildSweepConfigs:
    def test_default_count(self) -> None:
        configs = build_sweep_configs()
        # 1 (none@0.0) + 5 ratios x 3 methods = 16
        assert len(configs) == 16

    def test_baseline_is_none(self) -> None:
        configs = build_sweep_configs()
        assert configs[0].compression_ratio == 0.0
        assert configs[0].press_name == "none"

    def test_all_methods_present(self) -> None:
        configs = build_sweep_configs()
        methods = {c.press_name for c in configs if c.press_name != "none"}
        assert methods == set(SWEEP_METHODS)

    def test_custom_params_propagate(self) -> None:
        configs = build_sweep_configs(num_prompts=100, seed=123)
        for cfg in configs:
            assert cfg.num_prompts == 100
            assert cfg.seed == 123


class TestSaveResults:
    def test_saves_valid_json(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        results = [_make_result(prompt_id=f"p_{i}") for i in range(3)]
        path = save_results(results, config)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "results" in data
        assert len(data["results"]) == 3

    def test_summary_block_present(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        results = [_make_result()]
        path = save_results(results, config)
        data = json.loads(path.read_text())
        assert "summary" in data
        summary = data["summary"]
        assert "total" in summary
        assert "accuracy" in summary


class TestCheckpointing:
    def test_checkpoint_roundtrip(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        for i in range(3):
            _append_checkpoint(_make_result(prompt_id=f"p_{i}"), config)
        loaded = _load_checkpoint(config)
        assert len(loaded) == 3
        assert [r.prompt_id for r in loaded] == ["p_0", "p_1", "p_2"]

    def test_clear_checkpoint(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        _append_checkpoint(_make_result(), config)
        _clear_checkpoint(config)
        from kvguard.experiment import _checkpoint_path

        assert not _checkpoint_path(config).exists()


class TestResultExists:
    def test_exists_true(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        # Write a file at the expected path
        output_dir = config.output_dir / config.press_name
        output_dir.mkdir(parents=True, exist_ok=True)
        model_short = config.model_name.split("/")[-1]
        ratio_str = f"{config.compression_ratio:.3f}"
        filename = f"{model_short}_{ratio_str}_{config.num_prompts}p.json"
        (output_dir / filename).write_text("{}")
        assert result_exists(config) is True

    def test_exists_false(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        assert result_exists(config) is False


class TestSummarize:
    def test_empty_results(self) -> None:
        assert summarize([]) == {"total": 0}

    def test_basic_summarize(self) -> None:
        results = [
            _make_result(prompt_id="p_0", correct=True, catastrophes=[]),
            _make_result(prompt_id="p_1", correct=False, catastrophes=["repetition_loop"]),
            _make_result(prompt_id="p_2", correct=True, catastrophes=[]),
            _make_result(
                prompt_id="p_3", correct=False, catastrophes=["repetition_loop", "truncation"]
            ),
        ]
        summary = summarize(results)
        assert summary["total"] == 4
        assert summary["correct"] == 2
        assert summary["accuracy"] == 0.5
        assert summary["catastrophic_failure_rate"] == 0.5
        assert summary["catastrophe_counts"]["repetition_loop"] == 2
        assert summary["catastrophe_counts"]["truncation"] == 1
        assert summary["avg_tokens"] == 10.0
