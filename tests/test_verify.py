"""Tests for dataset verification module."""

import json
from pathlib import Path

from kvguard.verify import VerificationReport, verify_sweep

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(*, rep_count: int = 0) -> dict:
    return {
        "entropy": 1.0,
        "top1_prob": 0.5,
        "top5_prob": 0.9,
        "top1_token": "a",
        "rank_of_chosen": 0,
        "top20_logprobs": [-0.5] * 20,
        "h_alts": 0.3,
        "avg_logp": -2.0,
        "delta_h": 0.1,
        "rep_count": rep_count,
        "is_thinking_token": False,
    }


def _make_result(
    *,
    prompt_id: str = "gsm8k_0",
    press: str = "none",
    ratio: float = 0.0,
    n_tokens: int = 20,
    catastrophes: list[str] | None = None,
    catastrophe_onsets: dict[str, int] | None = None,
) -> dict:
    return {
        "prompt_id": prompt_id,
        "prompt_text": "test",
        "model": "test-model",
        "press": press,
        "compression_ratio": ratio,
        "max_new_tokens": 512,
        "seed": 42,
        "generated_text": "output",
        "ground_truth": "42",
        "predicted_answer": "42",
        "correct": True,
        "stop_reason": "eos",
        "catastrophes": catastrophes or [],
        "num_tokens_generated": n_tokens,
        "cache_size_after_prefill": None,
        "catastrophe_onsets": catastrophe_onsets or {},
        "signals": [_make_signal() for _ in range(n_tokens)],
    }


def _write_config_file(
    tmpdir: Path,
    press: str,
    ratio: float,
    results: list[dict],
    n_prompts: int = 50,
) -> Path:
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


def _build_full_sweep(tmpdir: Path, n_prompts: int = 50) -> None:
    """Create a complete sweep with all 16 configs."""
    ratios = [0.0, 0.25, 0.5, 0.625, 0.75, 0.875]
    methods = ["streaming_llm", "snapkv", "observed_attention"]

    for ratio in ratios:
        if ratio == 0.0:
            results = [
                _make_result(prompt_id=f"p{i}", press="none", ratio=0.0) for i in range(n_prompts)
            ]
            _write_config_file(tmpdir, "none", 0.0, results, n_prompts)
        else:
            for method in methods:
                cats = ["looping"] if ratio >= 0.75 else []
                onsets = {"looping": 10} if cats else {}
                results = [
                    _make_result(
                        prompt_id=f"p{i}",
                        press=method,
                        ratio=ratio,
                        catastrophes=cats if i % 3 == 0 else [],
                        catastrophe_onsets=onsets if i % 3 == 0 else {},
                    )
                    for i in range(n_prompts)
                ]
                _write_config_file(tmpdir, method, ratio, results, n_prompts)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVerifySweep:
    def test_full_sweep_passes(self, tmp_path: Path) -> None:
        """A complete, well-formed sweep should pass all checks."""
        _build_full_sweep(tmp_path)
        report = verify_sweep(tmp_path, num_prompts=50)
        assert isinstance(report, VerificationReport)
        assert report.passed, [
            f"{c.name}: {c.status} ({c.actual})" for c in report.checks if c.status != "PASS"
        ]

    def test_missing_files_fails(self, tmp_path: Path) -> None:
        """Only 1 config file when 16 expected → FAIL."""
        results = [_make_result(prompt_id=f"p{i}") for i in range(50)]
        _write_config_file(tmp_path, "none", 0.0, results)
        report = verify_sweep(tmp_path, num_prompts=50)
        file_check = next(c for c in report.checks if c.name == "file_count")
        assert file_check.status == "FAIL"

    def test_empty_dir_fails(self, tmp_path: Path) -> None:
        """No files at all → FAIL."""
        report = verify_sweep(tmp_path, num_prompts=50)
        assert not report.passed
        assert report.n_fail >= 1

    def test_wrong_prompt_count_fails(self, tmp_path: Path) -> None:
        """Config with 10 prompts when 50 expected → FAIL."""
        results = [_make_result(prompt_id=f"p{i}") for i in range(10)]
        _write_config_file(tmp_path, "none", 0.0, results)
        report = verify_sweep(tmp_path, num_prompts=50)
        prompt_checks = [c for c in report.checks if "prompt_count" in c.name]
        assert any(c.status == "FAIL" for c in prompt_checks)

    def test_missing_rep_count_fails(self, tmp_path: Path) -> None:
        """Signals without rep_count (stale data) → FAIL."""
        sig = _make_signal()
        del sig["rep_count"]
        result = _make_result()
        result["signals"] = [sig for _ in range(result["num_tokens_generated"])]
        _write_config_file(tmp_path, "none", 0.0, [result] * 50)
        report = verify_sweep(tmp_path, num_prompts=50)
        rep_check = next(c for c in report.checks if c.name == "rep_count_present")
        assert rep_check.status == "FAIL"

    def test_missing_catastrophe_onsets_fails(self, tmp_path: Path) -> None:
        """Results without catastrophe_onsets field → FAIL."""
        result = _make_result()
        del result["catastrophe_onsets"]
        _write_config_file(tmp_path, "none", 0.0, [result] * 50)
        report = verify_sweep(tmp_path, num_prompts=50)
        onset_check = next(c for c in report.checks if c.name == "catastrophe_onsets_present")
        assert onset_check.status == "FAIL"

    def test_signal_count_mismatch_fails(self, tmp_path: Path) -> None:
        """Fewer signals than num_tokens_generated → FAIL."""
        result = _make_result(n_tokens=20)
        result["signals"] = result["signals"][:10]  # only 10 signals for 20 tokens
        _write_config_file(tmp_path, "none", 0.0, [result] * 50)
        report = verify_sweep(tmp_path, num_prompts=50)
        sig_check = next(c for c in report.checks if c.name == "signal_completeness")
        assert sig_check.status == "FAIL"

    def test_to_dict(self, tmp_path: Path) -> None:
        """Report should serialize to dict."""
        _build_full_sweep(tmp_path)
        report = verify_sweep(tmp_path, num_prompts=50)
        d = report.to_dict()
        assert "passed" in d
        assert "checks" in d
        assert isinstance(d["checks"], list)

    def test_baseline_catastrophe_warns(self, tmp_path: Path) -> None:
        """Catastrophes in baseline should trigger a warning."""
        _build_full_sweep(tmp_path)
        # Overwrite baseline with catastrophic results
        results = [
            _make_result(
                prompt_id=f"p{i}",
                press="none",
                ratio=0.0,
                catastrophes=["looping"],
                catastrophe_onsets={"looping": 5},
            )
            for i in range(50)
        ]
        _write_config_file(tmp_path, "none", 0.0, results)
        report = verify_sweep(tmp_path, num_prompts=50)
        baseline_check = next((c for c in report.checks if c.name == "baseline_sanity"), None)
        assert baseline_check is not None
        assert baseline_check.status == "WARN"
