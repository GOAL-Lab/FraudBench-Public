"""LaTeX table of evaluated model configuration parameters.

See :mod:`fraudbench.analysis.table.configs` for the underlying implementation.
This script is a thin CLI wrapper.

Usage:
    uv run analysis/tables/table_configs_model.py
    uv run analysis/tables/table_configs_model.py --output paper/tables/configs_model.tex
    uv run analysis/tables/table_configs_model.py --no-save
"""
from pathlib import Path

from jsonargparse import ArgumentParser

from fraudbench.analysis.table.configs import run_model_configs


SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT = _REPO_ROOT / "paper" / "tables" / "configs_model.tex"


def main():
    """CLI entry point: parse args and dispatch to ``configs.run_model_configs``."""
    parser = ArgumentParser(
        description="Generate LaTeX tabularx snippet of evaluated model configuration parameters"
    )
    parser.add_argument(
        "--output", "-o",
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

    run_model_configs(output=args.output, no_save=args.no_save)


if __name__ == "__main__":
    main()
