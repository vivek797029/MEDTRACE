"""
MedTrace Federated Learning — Results Visualisation
=====================================================
Generates publication-quality comparison plots from ``EvalAccumulator`` data.

Plots produced
--------------
1. ``loss_vs_rounds.{fmt}``
   Line chart of average training loss per FL round for each system variant.
   Baseline (no-DP) vs FedAvg+DP side by side.

2. ``accuracy_vs_rounds.{fmt}``
   Line chart of MCQ evaluation accuracy per round for each system variant.

3. ``privacy_vs_performance.{fmt}``
   Scatter / trajectory plot: x = ε budget consumed, y = accuracy.
   Each point is one evaluated round; lines connect the trajectory.
   Ideal system sits top-left (high accuracy, low privacy spend).

4. ``per_hospital_loss.{fmt}``
   One line per hospital showing how local training loss evolves round-by-round.
   Reveals convergence speed and non-IID divergence between specialties.

5. ``privacy_budget_timeline.{fmt}``
   Stacked area chart: budget spent (red) / remaining (green) vs round.
   Visual confirmation that ε is never exceeded.

6. ``summary_comparison.{fmt}``
   Bar chart: final accuracy and final loss for each system variant.

All plots share a consistent palette, font sizes, and grid style so they
can be dropped into a paper or report without further editing.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from fl_evaluator import EvalAccumulator, EvalResult

logger = logging.getLogger(__name__)

# ─── Colour Palette ───────────────────────────────────────────────────────────
# Base palette for ≤ 5 series.  Beyond this, _get_palette() generates colours
# from matplotlib colormaps so the system works with any number of hospitals
# or experiment conditions.
_BASE_PALETTE = [
    "#2196F3",   # blue
    "#F44336",   # red
    "#4CAF50",   # green
    "#FF9800",   # orange
    "#9C27B0",   # purple
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "8"]


def _get_palette(n: int) -> List[str]:
    """
    Return a list of ``n`` hex colour strings.

    For n ≤ 5 the curated ``_BASE_PALETTE`` is used (best readability).
    For 6–10 the matplotlib ``tab10`` colormap is used.
    For > 10 ``tab20`` is used, cycling if necessary.
    """
    if n <= len(_BASE_PALETTE):
        return _BASE_PALETTE[:n]

    try:
        import matplotlib.pyplot as plt
        cmap_name = "tab10" if n <= 10 else "tab20"
        cmap = plt.get_cmap(cmap_name)
        return [
            "#{:02x}{:02x}{:02x}".format(
                int(cmap(i % cmap.N)[0] * 255),
                int(cmap(i % cmap.N)[1] * 255),
                int(cmap(i % cmap.N)[2] * 255),
            )
            for i in range(n)
        ]
    except Exception:
        # Fallback: cycle base palette
        return [_BASE_PALETTE[i % len(_BASE_PALETTE)] for i in range(n)]


def _get_markers(n: int) -> List[str]:
    """Return n marker strings, cycling through _MARKERS."""
    return [_MARKERS[i % len(_MARKERS)] for i in range(n)]


def _line_figsize(n_series: int) -> Tuple[float, float]:
    """Sensible figure width for line charts: wider when there are many series."""
    w = max(8.0, min(14.0, 6.0 + n_series * 0.5))
    return (w, 5.0)


def _bar_figsize(n_labels: int) -> Tuple[float, float]:
    """Sensible figure width for bar charts: grows with the number of groups."""
    w = max(6.0, min(20.0, n_labels * 2.5))
    return (w, 5.0)


# ─── Plotter ──────────────────────────────────────────────────────────────────

class ResultsPlotter:
    """
    Generates and saves all comparison plots from an ``EvalAccumulator``.

    Usage::

        plotter = ResultsPlotter(output_dir="outputs/evaluation/plots",
                                 fmt="png", dpi=150)
        paths = plotter.save_all(accumulator)
        # → list of absolute paths to saved plot files
    """

    def __init__(
        self,
        output_dir: str = "outputs/evaluation/plots",
        fmt: str = "png",
        dpi: int = 150,
    ):
        self.output_dir = output_dir
        self.fmt = fmt
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)

    # ── Convenience ───────────────────────────────────────────────────────────

    def save_all(self, acc: "EvalAccumulator") -> List[str]:
        """Generate every plot and return list of saved file paths."""
        saved: List[str] = []

        eval_data = acc.get_all_eval()
        client_data = acc.get_client_metrics()
        privacy_data = acc.get_privacy_metrics()

        if eval_data:
            saved.append(self.plot_loss_vs_rounds(eval_data))
            saved.append(self.plot_accuracy_vs_rounds(eval_data))
            saved.append(self.plot_privacy_vs_performance(eval_data))
            saved.append(self.plot_summary_comparison(eval_data))

        if client_data:
            saved.append(self.plot_per_hospital_loss(client_data))

        if privacy_data:
            saved.append(self.plot_privacy_budget_timeline(privacy_data))

        logger.info("Saved %d plots to %s", len(saved), self.output_dir)
        return [p for p in saved if p]

    # ── Plot 1: Loss vs Rounds ─────────────────────────────────────────────────

    def plot_loss_vs_rounds(
        self,
        eval_data: Dict[str, List["EvalResult"]],
        title: str = "Training Loss vs FL Rounds",
    ) -> str:
        import matplotlib.pyplot as plt

        n = len(eval_data)
        palette = _get_palette(n)
        markers = _get_markers(n)
        fig, ax = plt.subplots(figsize=_line_figsize(n))
        self._apply_style(ax)

        for i, (label, results) in enumerate(eval_data.items()):
            if not results:
                continue
            rounds = [r.round_num + 1 for r in results]
            losses = [r.loss for r in results]
            ax.plot(rounds, losses, color=palette[i], marker=markers[i],
                    linewidth=2, markersize=5, label=label)

        ax.set_xlabel("FL Round", fontsize=12)
        ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, framealpha=0.9)
        ax.set_ylim(bottom=0)
        fig.tight_layout()

        path = self._save(fig, "loss_vs_rounds")
        logger.info("Saved: %s", path)
        return path

    # ── Plot 2: Accuracy vs Rounds ─────────────────────────────────────────────

    def plot_accuracy_vs_rounds(
        self,
        eval_data: Dict[str, List["EvalResult"]],
        title: str = "MCQ Accuracy vs FL Rounds",
    ) -> str:
        import matplotlib.pyplot as plt

        n = len(eval_data)
        palette = _get_palette(n)
        markers = _get_markers(n)
        fig, ax = plt.subplots(figsize=_line_figsize(n))
        self._apply_style(ax)

        for i, (label, results) in enumerate(eval_data.items()):
            if not results:
                continue
            rounds = [r.round_num + 1 for r in results]
            accs   = [r.accuracy * 100 for r in results]   # as %
            ax.plot(rounds, accs, color=palette[i], marker=markers[i],
                    linewidth=2, markersize=5, label=label)

        ax.set_xlabel("FL Round", fontsize=12)
        ax.set_ylabel("MCQ Accuracy (%)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, framealpha=0.9)
        ax.set_ylim(0, 100)
        # Annotate final values (skip when many series to avoid clutter)
        if n <= 8:
            for i, (label, results) in enumerate(eval_data.items()):
                if results:
                    r = results[-1]
                    ax.annotate(
                        f"{r.accuracy * 100:.1f}%",
                        xy=(r.round_num + 1, r.accuracy * 100),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=9, color=palette[i],
                    )
        fig.tight_layout()

        path = self._save(fig, "accuracy_vs_rounds")
        logger.info("Saved: %s", path)
        return path

    # ── Plot 3: Privacy vs Performance ────────────────────────────────────────

    def plot_privacy_vs_performance(
        self,
        eval_data: Dict[str, List["EvalResult"]],
        title: str = "Privacy–Performance Trade-off",
    ) -> str:
        """
        X-axis: cumulative ε budget spent (privacy cost).
        Y-axis: MCQ accuracy (%).

        Each marker = one evaluated round.  Lines connect the trajectory.
        Ideal system: top-left corner (high accuracy, low privacy budget).
        """
        import matplotlib.pyplot as plt

        n = len(eval_data)
        palette = _get_palette(n)
        markers = _get_markers(n)
        fig, ax = plt.subplots(figsize=_line_figsize(n))
        self._apply_style(ax)

        has_data = False
        for i, (label, results) in enumerate(eval_data.items()):
            if not results:
                continue
            budgets = [r.privacy_budget_spent for r in results]
            accs    = [r.accuracy * 100 for r in results]
            color   = palette[i]
            marker  = markers[i]

            # Plot trajectory line
            ax.plot(budgets, accs, color=color, linewidth=1.5,
                    linestyle="--", alpha=0.6)
            # Plot individual round markers
            ax.scatter(budgets, accs, c=color, marker=marker,
                       s=60, label=label, zorder=3)

            # Annotate first and last points (skip when many series)
            if results and n <= 8:
                for idx in [0, len(results) - 1]:
                    r = results[idx]
                    ax.annotate(
                        f"R{r.round_num + 1}",
                        xy=(r.privacy_budget_spent, r.accuracy * 100),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=8, color=color,
                    )
            has_data = True

        if not has_data:
            plt.close(fig)
            return ""

        ax.set_xlabel("Privacy Budget Spent (ε)", fontsize=12)
        ax.set_ylabel("MCQ Accuracy (%)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, framealpha=0.9)

        # Add annotation arrow pointing toward the ideal corner
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.annotate(
            "← more private\nhigher accuracy ↑",
            xy=(xlim[0] + (xlim[1]-xlim[0])*0.05,
                ylim[0] + (ylim[1]-ylim[0])*0.85),
            fontsize=8, color="grey", style="italic",
        )

        fig.tight_layout()
        path = self._save(fig, "privacy_vs_performance")
        logger.info("Saved: %s", path)
        return path

    # ── Plot 4: Per-Hospital Loss ──────────────────────────────────────────────

    def plot_per_hospital_loss(
        self,
        client_data: Dict[str, List[dict]],
        title: str = "Per-Hospital Training Loss",
    ) -> str:
        import matplotlib.pyplot as plt

        n = len(client_data)
        palette = _get_palette(n)
        markers = _get_markers(n)
        fig, ax = plt.subplots(figsize=_line_figsize(n))
        self._apply_style(ax)

        for i, (hospital_id, metrics) in enumerate(client_data.items()):
            if not metrics:
                continue
            rounds = [m.get("round", j) + 1 for j, m in enumerate(metrics)]
            losses = [m.get("train_loss", 0.0) for m in metrics]
            name   = metrics[0].get("hospital", hospital_id)
            ax.plot(rounds, losses, color=palette[i], marker=markers[i],
                    linewidth=2, markersize=5, label=name)

        ax.set_xlabel("FL Round", fontsize=12)
        ax.set_ylabel("Local Training Loss", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, framealpha=0.9)
        ax.set_ylim(bottom=0)
        fig.tight_layout()

        path = self._save(fig, "per_hospital_loss")
        logger.info("Saved: %s", path)
        return path

    # ── Plot 5: Privacy Budget Timeline ───────────────────────────────────────

    def plot_privacy_budget_timeline(
        self,
        privacy_data: List[dict],
        title: str = "Privacy Budget Consumption",
    ) -> str:
        """
        Stacked area chart: ε spent (red) stacked on ε remaining (green).
        Confirms the total never exceeds the configured ε budget.
        """
        import matplotlib.pyplot as plt

        if not privacy_data:
            return ""

        rounds    = [d["round"] + 1 for d in privacy_data]
        spent     = [d["budget_spent"] for d in privacy_data]
        remaining = [d["budget_remaining"] for d in privacy_data]
        total_eps = privacy_data[0].get("total_epsilon", max(spent) + max(remaining))

        fig, ax = plt.subplots(figsize=(8, 4))
        self._apply_style(ax)

        ax.stackplot(
            rounds,
            spent, remaining,
            labels=["ε spent", "ε remaining"],
            colors=["#EF5350", "#66BB6A"],
            alpha=0.75,
        )
        ax.axhline(y=total_eps, color="#424242", linewidth=1.5,
                   linestyle="--", label=f"Total budget (ε={total_eps:.1f})")

        ax.set_xlabel("FL Round", fontsize=12)
        ax.set_ylabel("Privacy Budget (ε)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, framealpha=0.9)
        ax.set_ylim(0, total_eps * 1.1)
        fig.tight_layout()

        path = self._save(fig, "privacy_budget_timeline")
        logger.info("Saved: %s", path)
        return path

    # ── Plot 6: Summary Bar Chart ─────────────────────────────────────────────

    def plot_summary_comparison(
        self,
        eval_data: Dict[str, List["EvalResult"]],
        title: str = "Final Performance Comparison",
    ) -> str:
        """
        Grouped bar chart: final accuracy and normalised loss for each variant.
        Gives an at-a-glance comparison of all systems.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        labels  = [k for k, v in eval_data.items() if v]
        accs    = [eval_data[k][-1].accuracy * 100 for k in labels]
        losses  = [eval_data[k][-1].loss for k in labels]

        if not labels:
            return ""

        x  = np.arange(len(labels))
        w  = 0.35

        fig, ax1 = plt.subplots(figsize=_bar_figsize(len(labels)))
        self._apply_style(ax1)
        ax2 = ax1.twinx()

        bars1 = ax1.bar(x - w/2, accs,  width=w, label="Accuracy (%)",
                        color="#2196F3", alpha=0.85, zorder=3)
        bars2 = ax2.bar(x + w/2, losses, width=w, label="Final Loss",
                        color="#F44336", alpha=0.85, zorder=3)

        ax1.set_ylabel("MCQ Accuracy (%)", fontsize=12, color="#2196F3")
        ax2.set_ylabel("Cross-Entropy Loss", fontsize=12, color="#F44336")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
        ax1.set_ylim(0, 100)
        ax2.set_ylim(0, max(losses) * 1.25 if losses else 5)
        ax1.set_title(title, fontsize=14, fontweight="bold")

        # Value labels on bars
        for bar, val in zip(bars1, accs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
        for bar, val in zip(bars2, losses):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, framealpha=0.9)

        fig.tight_layout()
        path = self._save(fig, "summary_comparison")
        logger.info("Saved: %s", path)
        return path

    # ── Private helpers ───────────────────────────────────────────────────────

    def _apply_style(self, ax) -> None:
        """Apply a clean, consistent style to an Axes object."""
        ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)

    def _save(self, fig, name: str) -> str:
        """Save figure to output_dir/{name}.{fmt} and close it."""
        import matplotlib.pyplot as plt
        path = os.path.join(self.output_dir, f"{name}.{self.fmt}")
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path
