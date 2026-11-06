"""Histogram of case publication years for FraudBench scenarios.

Counts are reported by ``summary_id`` (one row per scenario), where the year
is derived from the top-level ``date`` field in
``data/final/dataset_public.jsonl``.

See :mod:`fraudbench.analysis.plot.case_year_histogram` for the underlying
implementation. This script is a thin CLI wrapper.

Usage:
    uv run analysis/plots/plot_case_year_histogram.py
"""
from pathlib import Path

from jsonargparse import ArgumentParser

from fraudbench.analysis.plot.case_year_histogram import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    run,
)


def main() -> None:
    """CLI entry point: parse args and dispatch to ``case_year_histogram.run``."""
    parser = ArgumentParser(
        description=(
            "Plot a histogram of case publication years, weighted by scenario "
            "count (one count per unique summary_id)."
        )
    )
    parser.add_argument(
        "--input", "-i", type=str, default=DEFAULT_INPUT,
        help=f"Scenario-level data file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save figures (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display the figure in a window",
    )
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_dir=Path(args.output_dir),
        show=args.show,
    )


if __name__ == "__main__":
    main()
