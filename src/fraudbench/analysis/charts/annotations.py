"""Axis annotation helpers shared by the bar primitives."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt


def add_group_labels_below(
    ax,
    centers: Sequence[float],
    labels: Sequence[str],
    *,
    y: float = -0.22,
    fontsize: int = 14,
    fontweight: str = "bold",
) -> None:
    """Render bold group labels below the x-axis at ``centers``.

    Uses an axis-fraction transform on the y-coordinate so ``y`` is the
    vertical offset below the axis (in axis units).
    """
    for center, label in zip(centers, labels):
        ax.text(
            center, y, label,
            ha="center", va="top",
            fontsize=fontsize, fontweight=fontweight,
            transform=ax.get_xaxis_transform(),
        )


def add_group_separators(
    ax,
    centers: Sequence[float],
    *,
    color: str = "gray",
    linestyle: str = "--",
    alpha: float = 0.3,
    linewidth: float = 1.0,
) -> None:
    """Draw dashed vertical lines between each consecutive pair of ``centers``."""
    for i in range(len(centers) - 1):
        sep_x = (centers[i] + centers[i + 1]) / 2
        ax.axvline(x=sep_x, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth)


def annotate_n_above_bars(
    ax,
    x_positions: Sequence[float],
    bar_heights: Sequence[float],
    n_values: Sequence[int],
    *,
    pad: float = 0.02,
    prefix: str = "n=",
    fontsize: int = 12,
    fontweight: str = "bold",
    color: str = "#444444",
    rotation: float = 0,
) -> None:
    """Write ``n=...`` labels above each bar position.

    ``pad`` is a fraction of the active y-axis span.  Using an axes-relative
    offset keeps the visual gap consistent across plots with different
    proportion ranges, unlike a fixed data-coordinate offset.

    ``rotation`` rotates the text counterclockwise (matplotlib convention);
    pass ``rotation=90`` for vertical labels that read bottom-to-top, which
    is useful when bars are narrow and horizontal ``n=...`` labels would
    overlap their neighbours.
    """
    y_min, y_max = ax.get_ylim()
    y_offset = (y_max - y_min) * pad
    for x, height, n in zip(x_positions, bar_heights, n_values):
        if rotation:
            ha, va = "center", "bottom"
        else:
            ha, va = "center", "bottom"
        ax.annotate(
            f"{prefix}{int(n)}",
            xy=(x, height + y_offset),
            ha=ha, va=va,
            fontsize=fontsize, fontweight=fontweight,
            color=color,
            rotation=rotation,
        )


def build_hatch_legend(
    color_entries: Sequence[tuple[str, str]],
    hatch_entries: Sequence[tuple[str, str]],
    *,
    edge_color: str = "black",
    linewidth: float = 0.5,
) -> tuple[list, list[str]]:
    """Build the standard 2-part (color + hatch) legend used by compare-settings plots.

    Parameters
    ----------
    color_entries
        Sequence of ``(label, color)`` tuples, one per metric series (no hatch).
    hatch_entries
        Sequence of ``(label, hatch)`` tuples, one per factor level (e.g. setting).

    Returns
    -------
    (handles, labels) ready to be passed to ``ax.legend(handles, labels, ...)``.
    """
    handles, labels = [], []
    for label, color in color_entries:
        handles.append(plt.Rectangle(
            (0, 0), 1, 1, facecolor=color, edgecolor=edge_color, linewidth=linewidth,
        ))
        labels.append(label)
    for label, hatch in hatch_entries:
        handles.append(plt.Rectangle(
            (0, 0), 1, 1, facecolor="white", hatch=hatch,
            edgecolor=edge_color, linewidth=linewidth,
        ))
        labels.append(label)
    return handles, labels


