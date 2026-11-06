"""Distribution-style chart primitives (raincloud, categorical bar)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .style import clean_axes


def _smart_wrap_label(label: str, max_line_len: int = 20) -> str:
    """Wrap a long categorical label onto two lines at a word boundary."""
    label = str(label)
    if len(label) <= max_line_len:
        return label
    words = label.split(" ")
    if len(words) == 1:
        return label

    line1: list[str] = []
    line1_len = 0
    for word in words:
        if line1_len + len(word) + (1 if line1 else 0) > max_line_len and line1:
            break
        line1.append(word)
        line1_len += len(word) + (1 if len(line1) > 1 else 0)
    line2_words = words[len(line1):]
    if not line2_words:
        return label
    return " ".join(line1) + "\n" + " ".join(line2_words)


def raincloud(
    ax,
    values: pd.Series | np.ndarray,
    *,
    x_label: str,
    title: str,
    stats_text: Optional[str] = None,
    color: str = "#4C72B0",
    group_label: str = "All Respondents",
    stats_fontsize: int = 10,
    bw: float = 0.2,
    width_viol: float = 0.7,
    alpha: float = 0.65,
    move: float = 0.2,
    point_size: int = 3,
) -> None:
    """Render a horizontal raincloud plot (box + scatter + half-violin).

    Requires the ``ptitprince`` package.
    """
    import ptitprince as pt

    if isinstance(values, np.ndarray):
        values = pd.Series(values)

    plot_df = pd.DataFrame({x_label: values.values, "group": group_label})
    pt.RainCloud(
        x="group", y=x_label, data=plot_df,
        palette=[color], bw=bw, width_viol=width_viol,
        ax=ax, orient="h", alpha=alpha, dodge=True, pointplot=False,
        move=move, point_size=point_size,
    )
    if stats_text:
        ax.text(
            0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=stats_fontsize,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
        )
    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel("")
    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_yticklabels([])
    clean_axes(ax, hide_left=True, ygrid=False, xgrid=True)


def categorical_bar(
    ax,
    categories: Sequence[str],
    counts: Sequence[int],
    *,
    total: Optional[int] = None,
    color: str = "#4C72B0",
    alpha: float = 0.85,
    edgecolor: str = "white",
    ylabel: str = "Count",
    xlabel: Optional[str] = None,
    title: Optional[str] = None,
    show_pct_labels: bool = True,
    use_proportions: bool = False,
    show_count_in_pct_labels: bool = False,
    pct_fontsize: int = 9,
    xtick_rotation: float = 45,
    xtick_ha: str = "right",
    xtick_fontsize: int = 9,
    wrap_labels: bool = True,
    wrap_max_line_len: int = 20,
    ylim_headroom: float = 0.15,
    apply_clean_axes: bool = True,
) -> np.ndarray:
    """Vertical categorical bar chart with percentage labels.

    Set ``use_proportions`` to plot each category's share of ``total`` on a
    0--1 y-axis. ``show_count_in_pct_labels`` appends the raw category count to
    each percentage annotation.

    Returns the array of x positions used.
    """
    if total is None:
        total = int(sum(counts)) or 1

    x = np.arange(len(categories))
    values = [count / total for count in counts] if use_proportions else list(counts)
    bars = ax.bar(x, values, color=color, edgecolor=edgecolor, alpha=alpha)

    if show_pct_labels:
        for count, bar in zip(counts, bars):
            pct = count / total * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.01 if use_proportions else 0.5),
                (
                    f"{pct:.1f}% ({count})"
                    if show_count_in_pct_labels
                    else f"{pct:.1f}%"
                ),
                ha="center",
                va="bottom",
                fontsize=pct_fontsize,
                clip_on=False,
            )

    if xlabel:
        ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    if title:
        ax.set_title(title, fontweight="bold")

    labels = [_smart_wrap_label(c, wrap_max_line_len) if wrap_labels else str(c) for c in categories]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=xtick_rotation, ha=xtick_ha, fontsize=xtick_fontsize)

    if use_proportions:
        ax.set_ylim(0, 1)
    else:
        max_count = max(counts) if len(counts) else 1
        ax.set_ylim(0, max_count * (1 + ylim_headroom))

    if apply_clean_axes:
        clean_axes(ax)
    return x
