"""Histogram of agentic tool-call counts per trajectory.

Reads ``agentic_trajectory`` / ``metadata.agentic_num_tool_calls`` from
``data/final/dataset_public.jsonl``. Implementation lives in
:mod:`fraudbench.analysis.table.agentic_tool_calls_histogram`.

Usage::
    uv run analysis/tables/table_agentic_tool_calls_histogram.py
    uv run analysis/tables/table_agentic_tool_calls_histogram.py \\
        --input data/final/dataset_public.jsonl \\
        --output paper/figures/agentic_tool_calls_histogram.pdf
    uv run analysis/tables/table_agentic_tool_calls_histogram.py --recompute-from-trajectory
"""
from pathlib import Path

from jsonargparse import ArgumentParser

from fraudbench.analysis.table.agentic_tool_calls_histogram import (
    run_agentic_tool_calls_histogram,
)


SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = SCRIPT_DIR.parent.parent
_DEFAULT_INPUT = _REPO_ROOT / "data" / "final" / "dataset_public.jsonl"
_DEFAULT_OUTPUT = _REPO_ROOT / "paper" / "figures" / "agentic_tool_calls_histogram.pdf"


def main() -> None:
    """CLI entry point: dispatch to ``agentic_tool_calls_histogram.run_agentic_tool_calls_histogram``."""
    parser = ArgumentParser(
        description=(
            "Plot a histogram of the number of agentic tool calls per "
            "trajectory from data/final/dataset_public.jsonl."
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
        "--bins",
        type=int,
        default=None,
        help="Histogram bins (default: seaborn discrete bins or 'auto' when --no-discrete)",
    )
    parser.add_argument(
        "--recompute-from-trajectory",
        action="store_true",
        help=(
            "Recompute tool call counts from agentic_trajectory tool_calls "
            "instead of using stored agentic_num_tool_calls."
        ),
    )
    parser.add_argument(
        "--no-discrete",
        action="store_true",
        help="Treat counts as continuous (wider bins) instead of discrete integers.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure window after saving.",
    )
    args = parser.parse_args()

    run_agentic_tool_calls_histogram(
        input_path=args.input,
        output=args.output,
        bins=args.bins,
        annotated_column=not args.recompute_from_trajectory,
        discrete=not args.no_discrete,
        show=args.show,
    )


if __name__ == "__main__":
    main()
