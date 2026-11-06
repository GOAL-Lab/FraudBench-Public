"""LaTeX summary table of evaluation scores by model and setting.

See :mod:`fraudbench.analysis.table.summary` for the underlying implementation.
This script is a thin CLI wrapper.

Two tables are written: the main table pooled across all 25
experimental conditions, and a control-condition table (rows with
neither an instructional nor a contextual condition applied).

Usage:
    uv run analysis/tables/table_summary.py --input data/final/obs_data.parquet
    uv run analysis/tables/table_summary.py --no-save
"""
from pathlib import Path

from jsonargparse import ArgumentParser

from fraudbench.analysis.table.summary import run


SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT = _REPO_ROOT / "paper" / "tables" / "summary.tex"
DEFAULT_CONTROL_OUTPUT = _REPO_ROOT / "paper" / "tables" / "summary_control.tex"


def main():
    """CLI entry point: parse args and dispatch to ``summary.run``."""
    parser = ArgumentParser(description="Generate LaTeX summary table of evaluation scores")
    parser.add_argument("--input", "-i", type=str, default="data/final/obs_data.parquet",
                        help="Input data file (.parquet, .csv, or .jsonl)")
    parser.add_argument("--output", "-o", type=str, default=str(DEFAULT_OUTPUT),
                        help=f"Output LaTeX file path for the main table "
                             f"(default: {DEFAULT_OUTPUT})")
    parser.add_argument("--control-output", type=str, default=str(DEFAULT_CONTROL_OUTPUT),
                        help="Output LaTeX file path for the control-condition "
                             "table, which reports the same statistics for the "
                             "rows with neither an instructional nor a "
                             "contextual condition applied "
                             f"(default: {DEFAULT_CONTROL_OUTPUT}). Pass an "
                             "empty string to skip writing it.")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save to file, only print to stdout")
    parser.add_argument("--no-percent", action="store_true",
                        help="Display as proportions (0-1) instead of percentages (0-100)")
    args = parser.parse_args()

    run(
        input_path=args.input,
        output=args.output,
        control_output=args.control_output,
        no_save=args.no_save,
        no_percent=args.no_percent,
    )


if __name__ == "__main__":
    main()
