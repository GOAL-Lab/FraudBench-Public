"""Shared legend placement for matplotlib axes and figures.

Bar charts keep legends above the plotting area (not on top of bars), in a
single horizontal row unless the caller overrides ``ncol``.
"""
from __future__ import annotations

from typing import Any, Sequence


def above_axes_legend_kw(*, ncol: int, **extra: Any) -> dict[str, Any]:
    """Keyword args for ``ax.legend`` / ``fig.legend``: centered above the axes/figure.

    Uses ``loc="lower center"`` with ``bbox_to_anchor`` at the top centre so the
    legend sits wholly above the data region.
    """
    kw: dict[str, Any] = {
        "loc": "lower center",
        "bbox_to_anchor": (0.5, 1.0),
        "ncol": ncol,
        "framealpha": 0.9,
    }
    kw.update(extra)
    return kw


def count_legend_entries(series: Sequence[Any]) -> int:
    """Count :class:`~fraudbench.analysis.charts.bars.BarSeries` with a non-empty label."""
    return sum(1 for s in series if getattr(s, "label", ""))
