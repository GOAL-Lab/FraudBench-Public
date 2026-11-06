"""Generate the regression coefficient and random-effect LaTeX tables.

Thin CLI wrapper around :mod:`fraudbench.analysis.table.regression_coefs`.
``--run`` selects the fitted regression and resolves the default
``inference_data.nc`` path under ``analysis/regressions/out/``:

    --run m1  ->  out/M1/models/fraudbench_m1/   (no filename suffix)
    --run m2  ->  out/M2/models/fraudbench_m2/   (suffix _m2)
    --run m3  ->  out/M3/models/fraudbench_m3/   (suffix _m3)

Usage:

    uv run analysis/tables/table_regression_coefs.py --run m1
    uv run analysis/tables/table_regression_coefs.py --run m3 --tables judge re_judge_model
    uv run analysis/tables/table_regression_coefs.py --run m2 --no-save
"""
from pathlib import Path
from typing import List, Optional

from jsonargparse import ArgumentParser

from fraudbench.analysis.table.regression_coefs import (
    VALID_RUNS,
    VALID_TABLES,
    default_inference_data_path,
    run,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "paper" / "tables" / "regressions"


def main() -> None:
    parser = ArgumentParser(description="Render regression coefficient / random-effect tables from a FraudBench posterior.")
    parser.add_argument("--run", type=str, default="m1", choices=list(VALID_RUNS),
                        help="Which fitted regression to render (m1, m2 or m3). Default: m1.")
    parser.add_argument("--inference-data", type=Optional[str], default=None,
                        help="Path to inference_data.nc. Defaults to the canonical run layout for --run.")
    parser.add_argument("--processed-data", type=Optional[str], default=None,
                        help="Override path to processed_data.parquet (default: <run>/processed_data.parquet).")
    parser.add_argument("--features", type=Optional[str], default=None,
                        help="Override path to features.pkl (default: <inference-data dir>/features.pkl).")
    parser.add_argument("--output-dir", "-o", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help=f"Output directory for the .tex files (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--tables", type=Optional[List[str]], default=None,
                        help=f"Subset of tables to build (default: all tables applicable to the run). Keys: {', '.join(VALID_TABLES)}.")
    parser.add_argument("--no-save", action="store_true", help="Print LaTeX to stdout instead of writing files.")
    args = parser.parse_args()

    inference_data = (
        default_inference_data_path(args.run, _REPO_ROOT) if args.inference_data is None else Path(args.inference_data)
    )
    run(
        inference_data_path=str(inference_data),
        run=args.run,
        processed_data_path=args.processed_data,
        features_path=args.features,
        output_dir=args.output_dir,
        tables=args.tables,
        no_save=args.no_save,
    )


if __name__ == "__main__":
    main()
