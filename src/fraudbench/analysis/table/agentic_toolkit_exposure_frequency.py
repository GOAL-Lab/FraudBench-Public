"""Bar chart of toolkit exposure frequencies (tools *available* per trajectory).

Counts how often each toolkit appears in the JSONL ``agentic_toolkits``
column, which represents which toolkits were exposed/available for that
trajectory (not how often they were called).

Backs ``analysis/tables/table_agentic_toolkit_exposure_frequency.py``.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..style import COLORS, save_figure, setup_plot_style

from .agentic_trajectory_common import load_agentic_trajectories


def _iter_exposed_toolkits(value: object) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x is not None]
    return [str(value)]


def run_agentic_toolkit_exposure_frequency(
    input_path: str,
    output: str,
    *,
    top_n: Optional[int] = None,
    show: bool = False,
) -> pd.DataFrame:
    """Aggregate toolkit exposure counts and save a horizontal bar chart."""
    df = load_agentic_trajectories(input_path)

    counts: Counter[str] = Counter()
    for toolkits in df["agentic_toolkits"]:
        for tk in set(_iter_exposed_toolkits(toolkits)):
            counts[tk] += 1

    rows = [{"toolkit": k, "n_trajectories_exposed": v} for k, v in counts.items()]
    tally_full = (
        pd.DataFrame(rows)
        .sort_values("n_trajectories_exposed", ascending=False)
        .reset_index(drop=True)
    )
    tally = tally_full.head(top_n).copy() if top_n is not None else tally_full

    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.35 * len(tally) + 1.5)))
    sns.barplot(
        data=tally,
        y="toolkit",
        x="n_trajectories_exposed",
        color=COLORS["dark_purple"],
        ax=ax,
    )
    ax.set_xlabel("Number of trajectories exposed to toolkit")
    ax.set_ylabel("Toolkit")
    ax.set_title("Toolkit exposure frequency (tools available per trajectory)")

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
    print("TOOLKIT EXPOSURE FREQUENCIES")
    print("=" * 60)
    print(f"n trajectories: {len(df)}")
    print(tally.to_string(index=False))

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, bbox_inches="tight")
    print(f"\nSaved figure to {out_path}")
    plt.close(fig)

    if show:
        plt.show()

    return tally

