"""
Unit tests for fl_plots.py

Tests the pure-Python helper functions and ResultsPlotter utility that do not
require an active display or GPU.  Matplotlib is used in non-interactive mode
(Agg backend) so tests run headlessly in CI.
"""

import os
import tempfile
import pytest

# Force headless matplotlib before importing anything that uses it
import matplotlib
matplotlib.use("Agg")

from fl_plots import (
    _bar_figsize,
    _get_markers,
    _get_palette,
    _line_figsize,
    ResultsPlotter,
)
from fl_evaluator import EvalAccumulator, EvalResult


# ─── _get_palette ─────────────────────────────────────────────────────────────

class TestGetPalette:
    def test_returns_correct_length(self):
        for n in (1, 3, 5, 8, 12, 20):
            p = _get_palette(n)
            assert len(p) == n, f"_get_palette({n}) returned {len(p)} colours"

    def test_colours_are_hex_strings(self):
        import re
        hex_re = re.compile(r"^#[0-9a-fA-F]{6}$")
        for n in (3, 7, 15):
            for colour in _get_palette(n):
                assert hex_re.match(colour), f"Not a valid hex colour: {colour!r}"

    def test_small_palette_uses_base_values(self):
        """For n ≤ 5, the base palette should be returned exactly."""
        from fl_plots import _BASE_PALETTE
        for n in range(1, 6):
            assert _get_palette(n) == _BASE_PALETTE[:n]

    def test_large_palette_still_valid_hex(self):
        import re
        hex_re = re.compile(r"^#[0-9a-fA-F]{6}$")
        for colour in _get_palette(25):
            assert hex_re.match(colour), f"Not a valid hex colour: {colour!r}"

    def test_each_element_is_string(self):
        assert all(isinstance(c, str) for c in _get_palette(10))


# ─── _get_markers ─────────────────────────────────────────────────────────────

class TestGetMarkers:
    def test_returns_correct_length(self):
        for n in (1, 5, 10, 15):
            m = _get_markers(n)
            assert len(m) == n

    def test_cycles_beyond_base(self):
        from fl_plots import _MARKERS
        m = _get_markers(len(_MARKERS) + 1)
        assert m[len(_MARKERS)] == _MARKERS[0]   # wraps back to first

    def test_all_strings(self):
        assert all(isinstance(s, str) for s in _get_markers(8))


# ─── _line_figsize / _bar_figsize ─────────────────────────────────────────────

class TestFigsize:
    def test_line_figsize_returns_tuple(self):
        w, h = _line_figsize(5)
        assert isinstance(w, float)
        assert isinstance(h, float)

    def test_line_figsize_height_constant(self):
        """Height should be 5.0 regardless of series count."""
        for n in (1, 5, 20):
            _, h = _line_figsize(n)
            assert h == 5.0

    def test_line_figsize_grows_with_n(self):
        w1, _ = _line_figsize(1)
        w10, _ = _line_figsize(10)
        assert w10 >= w1

    def test_line_figsize_clamps_max(self):
        """Width should not exceed 14."""
        w, _ = _line_figsize(1000)
        assert w <= 14.0

    def test_bar_figsize_grows_with_labels(self):
        w1, _ = _bar_figsize(2)
        w2, _ = _bar_figsize(10)
        assert w2 > w1

    def test_bar_figsize_clamps_max(self):
        w, _ = _bar_figsize(1000)
        assert w <= 20.0

    def test_bar_figsize_returns_tuple(self):
        result = _bar_figsize(5)
        assert len(result) == 2


# ─── ResultsPlotter ───────────────────────────────────────────────────────────

def _make_eval_result(label="Test", round_num=0, accuracy=0.6, privacy=0.5):
    return EvalResult(
        run_label=label,
        round_num=round_num,
        accuracy=accuracy,
        loss=1.0 - accuracy,
        perplexity=2.0,
        num_eval_samples=50,
        elapsed_seconds=1.0,
        privacy_budget_spent=privacy,
    )


def _make_accumulator():
    acc = EvalAccumulator()
    for i in range(3):
        acc.add_eval_result(_make_eval_result("FedAvg+DP", round_num=i, accuracy=0.5 + i * 0.05))
        acc.add_eval_result(_make_eval_result("No-DP", round_num=i, accuracy=0.6 + i * 0.05, privacy=0.0))
    acc.add_client_metrics("hospital_00", {"train_loss": 0.8, "round": 0, "hospital": "General"})
    acc.add_client_metrics("hospital_00", {"train_loss": 0.6, "round": 1, "hospital": "General"})
    acc.add_privacy_snapshot(0, 1.0, 7.0, 8.0)
    acc.add_privacy_snapshot(1, 2.0, 6.0, 8.0)
    return acc


class TestResultsPlotter:
    def test_save_all_creates_files(self):
        acc = _make_accumulator()
        with tempfile.TemporaryDirectory() as tmp:
            plotter = ResultsPlotter(output_dir=tmp, fmt="png", dpi=72)
            paths = plotter.save_all(acc)
            assert len(paths) > 0
            for p in paths:
                assert os.path.exists(p), f"Plot file not created: {p}"

    def test_save_all_returns_png_paths(self):
        acc = _make_accumulator()
        with tempfile.TemporaryDirectory() as tmp:
            plotter = ResultsPlotter(output_dir=tmp, fmt="png", dpi=72)
            paths = plotter.save_all(acc)
            for p in paths:
                assert p.endswith(".png"), f"Expected .png, got {p}"

    def test_save_all_svg_format(self):
        acc = _make_accumulator()
        with tempfile.TemporaryDirectory() as tmp:
            plotter = ResultsPlotter(output_dir=tmp, fmt="svg", dpi=72)
            paths = plotter.save_all(acc)
            for p in paths:
                assert p.endswith(".svg")

    def test_save_all_empty_accumulator(self):
        """Empty accumulator should not crash — just produce no files."""
        acc = EvalAccumulator()
        with tempfile.TemporaryDirectory() as tmp:
            plotter = ResultsPlotter(output_dir=tmp, fmt="png", dpi=72)
            paths = plotter.save_all(acc)
            assert paths == []

    def test_creates_output_dir_if_missing(self):
        acc = _make_accumulator()
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "a", "b", "c")
            plotter = ResultsPlotter(output_dir=nested, fmt="png", dpi=72)
            assert os.path.isdir(nested)

    def test_plot_loss_vs_rounds_single_series(self):
        acc = EvalAccumulator()
        for rn in range(4):
            acc.add_eval_result(_make_eval_result("Solo", round_num=rn))
        with tempfile.TemporaryDirectory() as tmp:
            plotter = ResultsPlotter(output_dir=tmp, fmt="png", dpi=72)
            path = plotter.plot_loss_vs_rounds(acc.get_all_eval())
            assert os.path.exists(path)

    def test_plot_with_many_series_does_not_crash(self):
        """10 series should not exhaust the palette or crash."""
        acc = EvalAccumulator()
        for i in range(10):
            for rn in range(3):
                acc.add_eval_result(_make_eval_result(f"Run_{i}", round_num=rn))
        with tempfile.TemporaryDirectory() as tmp:
            plotter = ResultsPlotter(output_dir=tmp, fmt="png", dpi=72)
            paths = plotter.save_all(acc)
            assert any("accuracy" in p for p in paths)

    def test_per_hospital_loss_plot(self):
        acc = EvalAccumulator()
        for i in range(5):    # 5 hospitals
            for rn in range(3):
                acc.add_client_metrics(
                    f"hospital_{i:02d}",
                    {"train_loss": 0.9 - rn * 0.1, "round": rn, "hospital": f"H{i}"},
                )
        with tempfile.TemporaryDirectory() as tmp:
            plotter = ResultsPlotter(output_dir=tmp, fmt="png", dpi=72)
            path = plotter.plot_per_hospital_loss(acc.get_client_metrics())
            assert os.path.exists(path)

    def test_privacy_budget_timeline_plot(self):
        acc = EvalAccumulator()
        for rn in range(4):
            acc.add_privacy_snapshot(rn, rn * 2.0, 8.0 - rn * 2.0, 8.0)
        with tempfile.TemporaryDirectory() as tmp:
            plotter = ResultsPlotter(output_dir=tmp, fmt="png", dpi=72)
            path = plotter.plot_privacy_budget_timeline(acc.get_privacy_metrics())
            assert os.path.exists(path)
