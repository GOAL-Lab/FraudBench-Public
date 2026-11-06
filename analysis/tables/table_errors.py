"""LaTeX tables of sampling and judge errors per model and per judge.

See :mod:`fraudbench.analysis.table.errors` for the underlying implementation.
This script is a thin CLI wrapper that emits four LaTeX tables:

* ``errors_main.tex``                   -- per-model errors on main runs
* ``errors_reasoning_ablation.tex``     -- per-model errors on reasoning-ablation runs
* ``errors_temperature_ablation.tex``   -- per-model errors on temperature-ablation runs
* ``errors_by_judge.tex``               -- judge-level breakdown of judge errors

Usage:
    uv run analysis/tables/table_errors.py
    uv run analysis/tables/table_errors.py --no-save
    uv run analysis/tables/table_errors.py \\
        --input data/final/obs_data_per_score.parquet \\
        --input-excluded data/final/obs_data_per_score_excluded.parquet \\
        --output-dir paper/tables/errors
"""
from pathlib import Path

from jsonargparse import ArgumentParser

from fraudbench.analysis.table.errors import run


SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "paper" / "tables" / "errors"


def main() -> None:
    """CLI entry point: parse args and dispatch to ``errors.run``."""
    parser = ArgumentParser(
        description="Generate LaTeX tables of sampling and judge errors."
    )
    parser.add_argument(
        "--input", "-i", type=str,
        default="data/final/obs_data_per_score.parquet",
        help="Kept per-score observations (.parquet/.csv/.jsonl).",
    )
    parser.add_argument(
        "--input-excluded", "-x", type=str,
        default="data/final/obs_data_per_score_excluded.parquet",
        help="Excluded observations: rows with has_sampling_error == True.",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for the four .tex files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not write files; print each table to stdout instead.",
    )
    args = parser.parse_args()

    run(
        input_path=args.input,
        excluded_input_path=args.input_excluded,
        output_dir=args.output_dir,
        no_save=args.no_save,
    )


if __name__ == "__main__":
    main()
