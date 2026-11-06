"""LaTeX table of task frequencies and gross-output ranks by NAICS sector.

See :mod:`fraudbench.analysis.table.naics_sector_distribution` for the
underlying implementation. This script is a thin CLI wrapper.

Usage:
    uv run analysis/tables/table_naics_sector_distribution.py
    uv run analysis/tables/table_naics_sector_distribution.py --no-save
    uv run analysis/tables/table_naics_sector_distribution.py --year 2024
"""
from pathlib import Path

from jsonargparse import ArgumentParser

from fraudbench.analysis.table.naics_sector_distribution import (
    DEFAULT_GROSS_OUTPUT,
    DEFAULT_SCENARIOS,
    DEFAULT_YEAR,
    run,
)


SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT = _REPO_ROOT / "paper" / "tables" / "naics_sector_distribution.tex"


def main() -> None:
    """Parse CLI arguments and generate the NAICS-sector distribution table."""
    parser = ArgumentParser(
        description=(
            "Generate a LaTeX table of scenario counts, dataset shares, gross-output "
            "shares, and scenario and gross-output ranks by NAICS sector."
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
        "--gross-output",
        type=str,
        default=DEFAULT_GROSS_OUTPUT,
        help=f"BEA gross-output CSV (default: {DEFAULT_GROSS_OUTPUT})",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help=f"Gross-output year to rank (default: {DEFAULT_YEAR})",
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
        gross_output_path=args.gross_output,
        year=args.year,
        output=args.output,
        no_save=args.no_save,
    )


if __name__ == "__main__":
    main()
