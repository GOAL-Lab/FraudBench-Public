"""Standalone LaTeX table for Human baseline evaluation results.

See :mod:`fraudbench.analysis.table.human_baseline` for the implementation.

Usage:
    uv run analysis/tables/table_human_baseline.py
    uv run analysis/tables/table_human_baseline.py --no-save
"""
from pathlib import Path

from jsonargparse import ArgumentParser

from fraudbench.analysis.table.human_baseline import run


SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT = _REPO_ROOT / "paper" / "tables" / "per_model" / "per_model_human.tex"


def main() -> None:
    """Parse CLI arguments and generate the Human baseline table."""
    parser = ArgumentParser(description="Generate the standalone Human baseline LaTeX table.")
    parser.add_argument(
        "--input", "-i", type=str,
        default="data/final/obs_data.parquet",
        help="Input data file (.parquet, .csv, or .jsonl).",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output LaTeX file path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Print the table instead of writing it.",
    )
    parser.add_argument(
        "--no-percent", action="store_true",
        help="Display rates as proportions in [0, 1] instead of percentages.",
    )
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        no_save=args.no_save,
        no_percent=args.no_percent,
    )


if __name__ == "__main__":
    main()
