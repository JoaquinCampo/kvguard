"""Tests for per-prompt checkpointing in experiment.py."""

from pathlib import Path

from kvguard.config import ExperimentConfig, RunResult
from kvguard.experiment import (
    _append_checkpoint,
    _checkpoint_path,
    _clear_checkpoint,
    _load_checkpoint,
)


def _make_config(tmp_path: Path, press: str = "none", ratio: float = 0.0) -> ExperimentConfig:
    return ExperimentConfig(
        model_name="test/Model",
        press_name=press,
        compression_ratio=ratio,
        num_prompts=5,
        seed=42,
        output_dir=tmp_path / "results",
    )


def _make_result(prompt_id: str = "gsm8k_0", correct: bool = True) -> RunResult:
    return RunResult(
        prompt_id=prompt_id,
        prompt_text="Q: test",
        model="test/Model",
        press="none",
        compression_ratio=0.0,
        max_new_tokens=512,
        seed=42,
        generated_text="#### 42",
        ground_truth="42",
        predicted_answer="42",
        correct=correct,
        stop_reason="eos",
        catastrophes=[],
        num_tokens_generated=10,
        cache_size_after_prefill=None,
        signals=[],
    )


class TestCheckpointPath:
    def test_returns_ckpt_jsonl(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        p = _checkpoint_path(cfg)
        assert p.suffix == ".jsonl"
        assert ".ckpt" in p.name

    def test_path_includes_press_dir(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, press="snapkv", ratio=0.5)
        p = _checkpoint_path(cfg)
        assert "snapkv" in str(p)

    def test_path_includes_ratio(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, press="streaming_llm", ratio=0.875)
        p = _checkpoint_path(cfg)
        assert "0.875" in p.name


class TestLoadCheckpoint:
    def test_empty_when_no_file(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        results = _load_checkpoint(cfg)
        assert results == []

    def test_loads_checkpointed_results(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        r1 = _make_result("gsm8k_0")
        r2 = _make_result("gsm8k_1", correct=False)
        _append_checkpoint(r1, cfg)
        _append_checkpoint(r2, cfg)
        loaded = _load_checkpoint(cfg)
        assert len(loaded) == 2
        assert loaded[0].prompt_id == "gsm8k_0"
        assert loaded[1].prompt_id == "gsm8k_1"
        assert loaded[1].correct is False

    def test_skips_empty_lines(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        r1 = _make_result("gsm8k_0")
        _append_checkpoint(r1, cfg)
        # Manually append an empty line
        ckpt = _checkpoint_path(cfg)
        with ckpt.open("a") as f:
            f.write("\n\n")
        loaded = _load_checkpoint(cfg)
        assert len(loaded) == 1


class TestAppendCheckpoint:
    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, press="snapkv", ratio=0.5)
        r = _make_result()
        _append_checkpoint(r, cfg)
        ckpt = _checkpoint_path(cfg)
        assert ckpt.exists()

    def test_appends_incrementally(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        for i in range(3):
            r = _make_result(f"gsm8k_{i}")
            _append_checkpoint(r, cfg)
        ckpt = _checkpoint_path(cfg)
        lines = [ln for ln in ckpt.read_text().splitlines() if ln.strip()]
        assert len(lines) == 3

    def test_roundtrip_preserves_data(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        original = _make_result("gsm8k_42", correct=False)
        _append_checkpoint(original, cfg)
        loaded = _load_checkpoint(cfg)
        assert len(loaded) == 1
        assert loaded[0].prompt_id == original.prompt_id
        assert loaded[0].correct == original.correct
        assert loaded[0].num_tokens_generated == original.num_tokens_generated


class TestClearCheckpoint:
    def test_removes_file(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        _append_checkpoint(_make_result(), cfg)
        ckpt = _checkpoint_path(cfg)
        assert ckpt.exists()
        _clear_checkpoint(cfg)
        assert not ckpt.exists()

    def test_noop_when_no_file(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        # Should not raise
        _clear_checkpoint(cfg)
