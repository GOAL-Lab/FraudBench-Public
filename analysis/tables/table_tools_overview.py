"""LaTeX table summarising the virtual toolkits available to the agent.

Generates a ``tabularx`` snippet for the appendix listing each registered
toolkit in
``src/fraudbench/data_generation/toolemu/tools/core_virtual_tools.py``,
along with its method names and a short description.

See :mod:`fraudbench.analysis.table.tools_overview` for the underlying
implementation.  This script is a thin CLI wrapper.

Usage::

    uv run analysis/tables/table_tools_overview.py
    uv run analysis/tables/table_tools_overview.py --output paper/tables/tools_overview.tex
    uv run analysis/tables/table_tools_overview.py --use-human-description
    uv run analysis/tables/table_tools_overview.py --no-save
"""
from pathlib import Path

from jsonargparse import ArgumentParser

from fraudbench.analysis.table.tools_overview import run_tools_overview


SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT = _REPO_ROOT / "paper" / "tables" / "tools_overview.tex"


def main():
    """CLI entry point: parse args and dispatch to ``tools_overview.run_tools_overview``."""
    parser = ArgumentParser(
        description=(
            "Generate a LaTeX tabularx snippet summarising the virtual toolkits "
            "available during agentic simulation (toolkit name, methods, and "
            "short description)."
        )
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
    parser.add_argument(
        "--use-human-description",
        action="store_true",
        help=(
            "Use each toolkit's terse `description_for_human` instead of the "
            "more informative `description_for_model` (the default)."
        ),
    )
    args = parser.parse_args()

    run_tools_overview(
        output=args.output,
        no_save=args.no_save,
        use_human_description=args.use_human_description,
    )


if __name__ == "__main__":
    main()
