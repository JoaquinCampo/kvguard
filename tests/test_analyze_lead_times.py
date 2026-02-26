"""Tests for analyze_lead_times.py helper functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from analyze_lead_times import format_report  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    *,
    detected: bool = True,
    lead_time: int | None = 10,
    pre_onset: bool | None = True,
    catastrophe_types: list[str] | None = None,
    press: str = "streaming_llm",
    onset: int = 50,
) -> dict:
    """Create a minimal lead-time entry dict."""
    return {
        "trace_idx": 0,
        "prompt_id": "p0",
        "press": press,
        "compression_ratio": 0.875,
        "catastrophe_types": catastrophe_types or ["looping"],
        "onset": onset,
        "n_tokens": 100,
        "detected": detected,
        "lead_time": lead_time,
        "first_detection_token": onset - lead_time if detected and lead_time is not None else None,
        "pre_onset": pre_onset,
    }


# ---------------------------------------------------------------------------
# Tests: format_report
# ---------------------------------------------------------------------------


class TestFormatReport:
    def test_basic_structure(self) -> None:
        """Report contains expected section headers."""
        data = {
            "thresh_0.50": [
                _make_entry(detected=True, lead_time=10, pre_onset=True),
                _make_entry(detected=False, lead_time=None, pre_onset=None),
            ],
        }
        report = format_report(data, [0.50])
        assert "# S4: Prediction Lead Time Analysis" in report
        assert "## 2. Detection Rate by Threshold" in report
        assert "## 3. Lead Time Distribution" in report
        assert "## 4. Lead Time by Failure Mode" in report
        assert "## 7. Key Findings" in report

    def test_detection_rate(self) -> None:
        """Detection rate is computed correctly."""
        data = {
            "thresh_0.50": [
                _make_entry(detected=True, lead_time=10),
                _make_entry(detected=True, lead_time=5),
                _make_entry(detected=False, lead_time=None, pre_onset=None),
            ],
        }
        report = format_report(data, [0.50])
        # 2 out of 3 detected = 66.7%
        assert "66.7%" in report

    def test_lead_time_stats(self) -> None:
        """Lead time statistics are computed correctly."""
        data = {
            "thresh_0.50": [
                _make_entry(detected=True, lead_time=10),
                _make_entry(detected=True, lead_time=20),
                _make_entry(detected=True, lead_time=30),
            ],
        }
        report = format_report(data, [0.50])
        # Mean of [10, 20, 30] = 20.0
        assert "20.0" in report
        # Median = 20.0
        assert "20.0" in report

    def test_multiple_thresholds(self) -> None:
        """Multiple thresholds produce multiple rows."""
        data = {
            "thresh_0.30": [_make_entry(detected=True, lead_time=15)],
            "thresh_0.50": [_make_entry(detected=True, lead_time=10)],
            "thresh_0.70": [_make_entry(detected=False, lead_time=None, pre_onset=None)],
        }
        report = format_report(data, [0.30, 0.50, 0.70])
        assert "| 0.30 |" in report
        assert "| 0.50 |" in report
        # 0.70 has no detected traces so no lead time row, but detection rate row exists
        assert "| 0.70 |" in report

    def test_empty_data(self) -> None:
        """Empty data produces report without crashing."""
        report = format_report({}, [0.50])
        assert "# S4: Prediction Lead Time Analysis" in report
        # No data rows but headers are present
        assert "## 2. Detection Rate by Threshold" in report

    def test_per_mode_grouping(self) -> None:
        """Failure modes are grouped correctly."""
        data = {
            "thresh_0.50": [
                _make_entry(
                    detected=True,
                    lead_time=10,
                    catastrophe_types=["looping"],
                ),
                _make_entry(
                    detected=True,
                    lead_time=100,
                    catastrophe_types=["non_termination"],
                ),
            ],
        }
        report = format_report(data, [0.50])
        assert "| looping |" in report
        assert "| non_termination |" in report

    def test_per_compressor_grouping(self) -> None:
        """Compressors are grouped correctly in section 5."""
        data = {
            "thresh_0.50": [
                _make_entry(detected=True, lead_time=10, press="streaming_llm"),
                _make_entry(detected=True, lead_time=5, press="snapkv"),
            ],
        }
        report = format_report(data, [0.50])
        assert "## 5. Lead Time by Compressor" in report
        assert "snapkv" in report
        assert "streaming_llm" in report

    def test_horizon_validation(self) -> None:
        """Horizon validation section checks multiple H values."""
        data = {
            "thresh_0.50": [
                _make_entry(detected=True, lead_time=40),
                _make_entry(detected=True, lead_time=100),
                _make_entry(detected=True, lead_time=10),
            ],
        }
        report = format_report(data, [0.50])
        assert "## 6. Horizon Validation" in report
        # Should check H=8, 16, 32, 48, 64, 128
        assert "| 32 |" in report
        assert "| 64 |" in report

    def test_negative_lead_time(self) -> None:
        """Negative lead time (detected after onset) is handled."""
        data = {
            "thresh_0.50": [
                _make_entry(detected=True, lead_time=-5, pre_onset=False),
                _make_entry(detected=True, lead_time=10, pre_onset=True),
            ],
        }
        report = format_report(data, [0.50])
        # Pre-onset count should be 1
        assert "50.0%" in report  # 1 out of 2 pre-onset

    def test_all_undetected(self) -> None:
        """All traces undetected produces report without crash."""
        data = {
            "thresh_0.50": [
                _make_entry(detected=False, lead_time=None, pre_onset=None),
                _make_entry(detected=False, lead_time=None, pre_onset=None),
            ],
        }
        report = format_report(data, [0.50])
        assert "0.0%" in report  # 0% detection rate

    def test_single_trace(self) -> None:
        """Single trace produces valid report."""
        data = {
            "thresh_0.50": [
                _make_entry(detected=True, lead_time=25, pre_onset=True),
            ],
        }
        report = format_report(data, [0.50])
        assert "100.0%" in report  # 1/1 detected
        assert "25.0" in report  # mean lead time
