"""Frequency chart of toolkit-level agentic tool calls.

Maps each stored function name (e.g. ``GmailSendEmail``) to its toolkit via the
tool registry. Implementation lives in
:mod:`fraudbench.analysis.table.agentic_toolkit_frequency`.

Usage::
    uv run analysis/tables/table_agentic_toolkit_frequency.py
    uv run analysis/tables/table_agentic_toolkit_frequency.py \\
        --input data/final/dataset_public.jsonl \\
        --output paper/figures/agentic_toolkit_frequency.pdf
    uv run analysis/tables/table_agentic_toolkit_frequency.py --top-n 12
"""
from pathlib import Path

from jsonargparse import ArgumentParser

from fraudbench.analysis.table.agentic_toolkit_frequency import (
    run_agentic_toolkit_frequency,
)


SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = SCRIPT_DIR.parent.parent
_DEFAULT_INPUT = _REPO_ROOT / "data" / "final" / "dataset_public.jsonl"
_DEFAULT_OUTPUT = _REPO_ROOT / "paper" / "figures" / "agentic_toolkit_frequency.pdf"


def main() -> None:
    """CLI entry point: parse args and dispatch to ``agentic_toolkit_frequency.run_agentic_toolkit_frequency``."""
    parser = ArgumentParser(
        description=(
            "Plot toolkit-level frequencies of agentic tool calls "
            "(e.g. Gmail vs GmailSendEmail)."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=str(_DEFAULT_INPUT),
        help=f"Input scenarios JSONL (default: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(_DEFAULT_OUTPUT),
        help=f"Output figure path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Plot only the top N toolkits by call count (default: all).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure window after saving.",
    )
    args = parser.parse_args()

    run_agentic_toolkit_frequency(
        input_path=args.input,
        output=args.output,
        top_n=args.top_n,
        show=args.show,
    )


if __name__ == "__main__":
    main()
