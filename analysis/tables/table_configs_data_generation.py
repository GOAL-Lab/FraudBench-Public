"""LaTeX table of model configurations used in the data-generation pipeline.

Covers the QA pipeline (binary / open / agentic-vignette generation) and the
trajectory-simulation roles (agent / simulator / critiquer). Implementation
lives in :mod:`fraudbench.analysis.table.data_generation_configs`.

Usage:
    uv run analysis/tables/table_configs_data_generation.py
    uv run analysis/tables/table_configs_data_generation.py --output paper/tables/configs_data_generation.tex
    uv run analysis/tables/table_configs_data_generation.py --no-save
"""
from pathlib import Path

from jsonargparse import ArgumentParser

from fraudbench.analysis.table.data_generation_configs import (
    run_data_generation_configs,
)


SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT = _REPO_ROOT / "paper" / "tables" / "configs_data_generation.tex"


def main() -> None:
    """CLI entry point: parse args and dispatch to ``data_generation_configs.run_data_generation_configs``."""
    parser = ArgumentParser(
        description=(
            "Generate a LaTeX tabularx snippet of model configurations used "
            "during \\textsc{FraudBench} data generation (QA pipeline + "
            "agentic trajectory simulation roles)."
        )
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output .tex file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print to stdout instead of saving to file",
    )
    args = parser.parse_args()

    run_data_generation_configs(output=args.output, no_save=args.no_save)


if __name__ == "__main__":
    main()
