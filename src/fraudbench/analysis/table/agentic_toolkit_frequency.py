"""Bar chart of toolkit-level agentic tool-call frequencies.

Each assistant ``tool_calls`` entry contributes one toward its toolkit prefix
(e.g. ``GmailSendEmail`` → ``Gmail``), resolved via registered FunctionToolkit
names.

Backs ``analysis/tables/table_agentic_toolkit_frequency.py``.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..style import COLORS, save_figure, setup_plot_style

from .agentic_trajectory_common import (
    function_name_to_toolkit,
    iter_tool_functions_in_trajectory,
    load_agentic_trajectories,
)


def run_agentic_toolkit_frequency(
    input_path: str,
    output: str,
    *,
    top_n: Optional[int] = None,
    show: bool = False,
) -> pd.DataFrame:
    """Aggregate toolkit call counts and save a horizontal bar chart."""
    df = load_agentic_trajectories(input_path)

    toolkit_counts: Counter[str] = Counter()
    for traj in df["agentic_trajectory"]:
        for fn in iter_tool_functions_in_trajectory(traj):
            toolkit_counts[function_name_to_toolkit(fn)] += 1

    rows = [{"toolkit": k, "n_calls": v} for k, v in toolkit_counts.items()]
    tally_full = pd.DataFrame(rows).sort_values("n_calls", ascending=False).reset_index(drop=True)
    tally = tally_full.head(top_n).copy() if top_n is not None else tally_full

    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.35 * len(tally) + 1.5)))
    sns.barplot(
        data=tally,
        y="toolkit",
        x="n_calls",
        color=COLORS["dark_teal"],
        ax=ax,
    )
    ax.set_xlabel("Number of tool calls")
    ax.set_ylabel("Toolkit")
    ax.set_title("Agentic tool usage by toolkit")

    # Annotate raw counts to the right of each horizontal bar.
    for container in ax.containers:
        labels = [
            f"{int(bar.get_width())}" if bar.get_width() > 0 else ""
            for bar in container
        ]
        ax.bar_label(container, labels=labels, padding=3, fontsize=9)

    # Add headroom on the right so labels aren't clipped.
    x_right = ax.get_xlim()[1]
    ax.set_xlim(right=x_right * 1.08)

    print("\n" + "=" * 60)
    print("TOOLKIT CALL FREQUENCIES")
    print("=" * 60)
    total_dataset = int(tally_full["n_calls"].sum())
    total_plotted = int(tally["n_calls"].sum())
    print(f"Total tool calls (dataset): {total_dataset}")
    if top_n is not None:
        print(f"Calls in plotted top-{top_n} toolkits: {total_plotted}")
    print(tally.to_string(index=False))

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, bbox_inches="tight")
    print(f"\nSaved figure to {out_path}")
    plt.close(fig)

    if show:
        plt.show()

    return tally
