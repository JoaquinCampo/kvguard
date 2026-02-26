"""Tests for generate_figures.py helper functions.

These test the data-independent parts of figure generation:
style configuration and data structure expectations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import generate_figures  # noqa: E402

# ---------------------------------------------------------------------------
# Tests: Module constants and configuration
# ---------------------------------------------------------------------------


class TestModuleConfig:
    def test_output_dir_defined(self) -> None:
        assert generate_figures.OUTPUT_DIR == Path("paper/figures/generated")

    def test_compressors_defined(self) -> None:
        assert "streaming_llm" in generate_figures.COMPRESSORS
        assert "snapkv" in generate_figures.COMPRESSORS

    def test_ratios_defined(self) -> None:
        assert len(generate_figures.RATIOS) == 5
        assert generate_figures.RATIOS[0] == 0.250
        assert generate_figures.RATIOS[-1] == 0.875

    def test_ratio_labels_match_ratios(self) -> None:
        assert len(generate_figures.RATIO_LABELS) == len(generate_figures.RATIOS)

    def test_colors_defined(self) -> None:
        assert generate_figures.C_STREAM.startswith("#")
        assert generate_figures.C_SNAP.startswith("#")
        assert generate_figures.C_BASELINE.startswith("#")
        assert generate_figures.C_CORRECT.startswith("#")
        assert generate_figures.C_LOOPING.startswith("#")
        assert generate_figures.C_NONTERM.startswith("#")
        assert generate_figures.C_WRONG.startswith("#")

    def test_matplotlib_agg_backend(self) -> None:
        import matplotlib

        assert matplotlib.get_backend().lower() == "agg"


class TestFigureFunctions:
    """Test that all figure functions exist and are callable."""

    def test_figure_phase_transition_exists(self) -> None:
        assert callable(generate_figures.figure_phase_transition)

    def test_figure_failure_signatures_exists(self) -> None:
        assert callable(generate_figures.figure_failure_signatures)

    def test_figure_lead_time_exists(self) -> None:
        assert callable(generate_figures.figure_lead_time)

    def test_figure_ablation_exists(self) -> None:
        assert callable(generate_figures.figure_ablation)

    def test_figure_entropy_trajectory_exists(self) -> None:
        assert callable(generate_figures.figure_entropy_trajectory)

    def test_figure_phase_transition_looping_exists(self) -> None:
        assert callable(generate_figures.figure_phase_transition_looping)

    def test_all_figures_in_main(self) -> None:
        """main() should call all figure functions."""
        import inspect

        source = inspect.getsource(generate_figures.main)
        assert "figure_phase_transition()" in source
        assert "figure_failure_signatures()" in source
        assert "figure_lead_time()" in source
        assert "figure_ablation()" in source
        assert "figure_entropy_trajectory()" in source
        assert "figure_phase_transition_looping()" in source


class TestDataLoaders:
    """Test that data loader functions exist and have correct signatures."""

    def test_load_phase_transition_data_exists(self) -> None:
        assert callable(generate_figures.load_phase_transition_data)

    def test_load_failure_mode_data_exists(self) -> None:
        assert callable(generate_figures.load_failure_mode_data)

    def test_load_ablation_data_exists(self) -> None:
        assert callable(generate_figures.load_ablation_data)

    def test_load_lead_time_data_exists(self) -> None:
        assert callable(generate_figures.load_lead_time_data)

    def test_load_traces_for_example_exists(self) -> None:
        assert callable(generate_figures.load_traces_for_example)
