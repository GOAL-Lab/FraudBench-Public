"""LaTeX table of LLM-as-a-judge (grader) model configuration parameters.

See :mod:`fraudbench.analysis.table.configs` for the underlying implementation.
This script is a thin CLI wrapper.

Usage:
    uv run analysis/tables/table_configs_grader.py
    uv run analysis/tables/table_configs_grader.py --output paper/tables/configs_grader.tex
    uv run analysis/tables/table_configs_grader.py --no-save
"""
from pathlib import Path

from jsonargparse import ArgumentParser

from fraudbench.analysis.table.configs import run_grader_configs


SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT = _REPO_ROOT / "paper" / "tables" / "configs_grader.tex"


def main():
    """CLI entry point: parse args and dispatch to ``configs.run_grader_configs``."""
    parser = ArgumentParser(
        description="Generate LaTeX tabularx snippet of LLM-as-a-judge configuration parameters"
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

    run_grader_configs(output=args.output, no_save=args.no_save)


if __name__ == "__main__":
    main()
