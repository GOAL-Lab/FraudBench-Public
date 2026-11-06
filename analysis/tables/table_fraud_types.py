"""LaTeX table of scenario counts by fraud type (misrepresentation vs concealment).

See :mod:`fraudbench.analysis.table.fraud_types` for the underlying implementation.
This script is a thin CLI wrapper.

Usage:
    uv run analysis/tables/table_fraud_types.py
    uv run analysis/tables/table_fraud_types.py --input data/final/dataset_public.jsonl
    uv run analysis/tables/table_fraud_types.py --no-save
"""
from pathlib import Path

from jsonargparse import ArgumentParser

from fraudbench.analysis.table.fraud_types import run


SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT = _REPO_ROOT / "paper" / "tables" / "fraud_types.tex"


def main() -> None:
    """CLI entry point: parse args and dispatch to ``fraud_types.run``."""
    parser = ArgumentParser(
        description=(
            "Generate a LaTeX table of scenario counts by fraud type "
            "(Fraudulent Misrepresentation vs Fraudulent Concealment)."
        )
    )
    parser.add_argument(
        "--input", "-i", type=str, default="data/final/dataset_public.jsonl",
        help="Input scenarios file (.jsonl, .parquet, or .csv)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Output LaTeX file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save to file, only print to stdout",
    )
    args = parser.parse_args()

    run(
        input_path=args.input,
        output=args.output,
        no_save=args.no_save,
    )


if __name__ == "__main__":
    main()
