"""Histogram of case publication years for \\textsc{FraudBench} scenarios.

Each \\textsc{FraudBench} scenario (one row per ``summary_id``) is derived from
an underlying legal case. Multiple scenarios may share the same ``case_id``,
so we count *scenarios* (not cases) per publication year of the source case.

The source case's publication year is derived from the top-level ISO ``date``
field in ``data/final/dataset_public.jsonl``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraudbench.utils import flatten_public_dataset

from ..charts.style import clean_axes
from ..style import COLORS, save_figure, setup_plot_style


DEFAULT_INPUT = "data/final/dataset_public.jsonl"
DEFAULT_OUTPUT_DIR = Path("paper/figures")


def load_scenario_years(input_path: str = DEFAULT_INPUT) -> pd.DataFrame:
    """Parse top-level ``date`` and return one row per scenario with its year."""
    scenarios = pd.read_json(input_path, lines=True)
    scenarios = flatten_public_dataset(scenarios)
    scenarios = scenarios.drop_duplicates(subset="summary_id").reset_index(drop=True)
    print(f"Loaded {len(scenarios)} scenarios from {input_path}")

    if "date" not in scenarios.columns:
        raise ValueError(
            "Input must contain the top-level 'date' field from "
            "data/final/dataset_public.jsonl."
        )

    scenarios["case_date"] = pd.to_datetime(scenarios["date"], errors="coerce")
    scenarios["case_year"] = scenarios["case_date"].dt.year
    n_missing = int(scenarios["case_year"].isna().sum())
    if n_missing:
        print(f"Warning: {n_missing} scenarios have no valid date and will be dropped")
        scenarios = scenarios.dropna(subset=["case_year"])
    scenarios["case_year"] = scenarios["case_year"].astype(int)
    return scenarios[["summary_id", "case_date", "case_year"]]


def compute_year_counts(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame indexed by year (gap-filled with zero counts)."""
    counts = scenarios["case_year"].value_counts().sort_index()
    full_index = pd.RangeIndex(int(counts.index.min()), int(counts.index.max()) + 1)
    counts = counts.reindex(full_index, fill_value=0)
    counts.index.name = "case_year"
    out = pd.DataFrame({"count": counts.astype(int).values}, index=counts.index)
    total = int(out["count"].sum())
    out["proportion"] = out["count"] / total if total else 0.0
    return out


def plot_year_histogram(
    counts: pd.DataFrame,
    output_path: Optional[Path] = None,
    *,
    show_pct_labels: bool = False,
):
    """Render a vertical bar histogram of scenario counts per case year."""
    setup_plot_style()

    years = counts.index.to_numpy()
    values = counts["count"].to_numpy()
    total = int(values.sum())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        years, values,
        color=COLORS["blue"], edgecolor="white", alpha=0.9, width=0.8,
    )

    for bar, n in zip(bars, values):
        if n == 0:
            continue
        text = f"{int(n)}"
        if show_pct_labels and total:
            text = f"{int(n)}\n({100 * n / total:.1f}\\%)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            text, ha="center", va="bottom", fontsize=9,
        )

    ax.set_xlabel("Case Publication Year", fontweight="bold")
    ax.set_ylabel(f"Number of Scenarios (n={total})", fontweight="bold")
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylim(0, max(values) * 1.15 if max(values) else 1)
    clean_axes(ax)

    plt.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(plt.gcf(), output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    return fig, ax


def run(
    input_path: str = DEFAULT_INPUT,
    output_dir: Optional[Path] = None,
    show: bool = False,
) -> pd.DataFrame:
    """Generate the case-year histogram from dates embedded in scenario data."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenario_years(input_path)
    counts = compute_year_counts(scenarios)

    print("\n" + "=" * 60)
    print("CASE YEAR DISTRIBUTION (by scenario)")
    print("=" * 60)
    print(counts.to_string(formatters={"proportion": lambda p: f"{p:.3f}"}))
    print(f"\nTotal scenarios: {int(counts['count'].sum())}")
    print(f"Year range: {int(counts.index.min())} – {int(counts.index.max())}")

    plot_year_histogram(counts, output_dir / "case_year_histogram.pdf")

    if show:
        plt.show()

    return counts
