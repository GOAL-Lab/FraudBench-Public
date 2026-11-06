"""Histogram of per-trajectory agentic tool-call counts.

Uses the denormalised ``agentic_num_tool_calls`` field by default (as requested
for paper tables). Optionally, it can be recomputed by walking assistant
``tool_calls`` inside ``agentic_trajectory``.

Backs ``analysis/tables/table_agentic_tool_calls_histogram.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from ..style import COLORS, save_figure, setup_plot_style

from .agentic_trajectory_common import count_tool_calls_in_trajectory, load_agentic_trajectories


def run_agentic_tool_calls_histogram(
    input_path: str,
    output: str,
    *,
    bins: Optional[int] = None,
    annotated_column: bool = True,
    discrete: bool = True,
    show: bool = False,
) -> pd.Series:
    """Plot and save a histogram of tool-call counts per trajectory row."""
    df = load_agentic_trajectories(input_path)

    if annotated_column:
        if "agentic_num_tool_calls" not in df.columns:
            raise ValueError(
                "annotated_column=True requires 'agentic_num_tool_calls' in the input."
            )
        counts = df["agentic_num_tool_calls"].fillna(0).astype(int)
        source_note = "agentic_num_tool_calls column"
    else:
        counts = df["agentic_trajectory"].apply(count_tool_calls_in_trajectory)
        source_note = "recomputed from agentic_trajectory"

    setup_plot_style()

    fig, ax = plt.subplots(figsize=(8, 5))
    series = counts.astype(float)
    if discrete:
        sns.histplot(
            series,
            discrete=True,
            bins=bins,
            kde=False,
            color=COLORS["blue"],
            ax=ax,
        )
    else:
        sns.histplot(
            series,
            bins=bins if bins is not None else "auto",
            kde=False,
            color=COLORS["blue"],
            ax=ax,
        )
    ax.set_xlabel("Tool calls per trajectory")
    ax.set_ylabel("Number of trajectories")
    ax.set_title("Distribution of agentic tool-call counts")

    # Tool calls are integer counts; force whole-number x-axis ticks.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Annotate raw counts above each bar (skip empty bins for readability).
    for container in ax.containers:
        labels = [
            f"{int(bar.get_height())}" if bar.get_height() > 0 else ""
            for bar in container
        ]
        ax.bar_label(container, labels=labels, padding=2, fontsize=9)

    # Add headroom so the topmost annotation isn't clipped.
    y_top = ax.get_ylim()[1]
    ax.set_ylim(top=y_top * 1.08)

    print("\n" + "=" * 60)
    print("TOOL CALLS PER TRAJECTORY")
    print("=" * 60)
    print(f"Source: {source_note}")
    print(f"n trajectories: {len(counts)}")
    print(f"mean: {counts.mean():.2f}, median: {counts.median():.1f}")
    print(f"min: {counts.min()}, max: {counts.max()}")

    if annotated_column:
        recomputed = df["agentic_trajectory"].apply(count_tool_calls_in_trajectory)
        ann = df["agentic_num_tool_calls"]
        mask = ann.notna()
        if mask.any():
            mism = int((recomputed[mask].values != ann[mask].astype(int).values).sum())
            if mism:
                print(
                    f"Note: {mism} rows where stored agentic_num_tool_calls "
                    "differs from recomputed trajectory count (histogram uses stored)."
                )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, bbox_inches="tight")
    print(f"\nSaved figure to {out_path}")
    plt.close(fig)

    if show:
        plt.show()

    return counts
