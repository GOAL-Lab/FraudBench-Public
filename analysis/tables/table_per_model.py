"""Per-model appendix tables: condition-level results for every metric/setting.

See :mod:`fraudbench.analysis.table.per_model` for the underlying
implementation. This script is a thin CLI wrapper.

Usage:
    # Empirical proportions with Wilson 95% binomial CIs.
    uv run analysis/tables/table_per_model.py --input data/final/obs_data.parquet

    uv run analysis/tables/table_per_model.py --no-save                # debug-print first table
"""
from pathlib import Path

from jsonargparse import ArgumentParser

from fraudbench.analysis.table.per_model import run


SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "paper" / "tables" / "per_model"


def main() -> None:
    """CLI entry point: parse args and dispatch to ``per_model.run``."""
    parser = ArgumentParser(
        description="Generate per-model condition-level LaTeX tables for the appendix."
    )
    parser.add_argument(
        "--input", "-i", type=str,
        default="data/final/obs_data.parquet",
        help="Input data file (.parquet, .csv, or .jsonl).",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for per-model tables (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not write files; print the first table to stdout for debugging.",
    )
    parser.add_argument(
        "--no-master", dest="write_master",
        action="store_false", default=True,
        help="Skip writing the master per_model_all.tex file.",
    )
    parser.add_argument(
        "--no-percent", action="store_true",
        help="Display rates as proportions in [0, 1] instead of percentages.",
    )
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_dir=args.output_dir,
        no_save=args.no_save,
        no_percent=args.no_percent,
        write_master=args.write_master,
    )


if __name__ == "__main__":
    main()
