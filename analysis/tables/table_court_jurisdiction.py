"""LaTeX table of scenario counts by court jurisdiction and level.

Jurisdiction and level labels are read directly from their top-level fields in
``data/final/dataset_public.jsonl``.

See :mod:`fraudbench.analysis.table.court_jurisdiction` for the implementation.
This script is a thin CLI wrapper.

Usage:
    uv run analysis/tables/table_court_jurisdiction.py
    uv run analysis/tables/table_court_jurisdiction.py --no-save
"""
from pathlib import Path

from jsonargparse import ArgumentParser

from fraudbench.analysis.table.court_jurisdiction import (
    DEFAULT_SCENARIOS,
    run,
)


SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT = _REPO_ROOT / "paper" / "tables" / "court_jurisdiction.tex"


def main() -> None:
    """Parse CLI arguments and generate the court distribution table."""
    parser = ArgumentParser(
        description=(
            "Generate a LaTeX table of FraudBench scenario counts by federal "
            "or state jurisdiction and trial, appeals, or supreme court level."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=DEFAULT_SCENARIOS,
        help=f"Scenario-level dataset (default: {DEFAULT_SCENARIOS})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output LaTeX file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save to file; print the LaTeX table only",
    )
    args = parser.parse_args()
    run(
        input_path=args.input,
        output=args.output,
        no_save=args.no_save,
    )


if __name__ == "__main__":
    main()
