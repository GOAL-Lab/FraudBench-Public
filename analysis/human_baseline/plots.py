"""Demographic plots for human baseline survey respondents.

See :mod:`fraudbench.analysis.plot.human_demographics` for the underlying
implementation. This script is a thin CLI wrapper.

Usage:
    uv run analysis/human_baseline/plots.py
    uv run analysis/human_baseline/plots.py --input data/final/human_baseline/human_baseline_public.jsonl
    uv run analysis/human_baseline/plots.py --show
"""
from jsonargparse import ArgumentParser

from fraudbench.analysis.plot.human_demographics import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    run,
)


def main():
    """CLI entry point: parse args and dispatch to ``human_demographics.run``."""
    parser = ArgumentParser(description="Plot demographic statistics for human baseline data")
    parser.add_argument("--input", "-i", type=str, default=DEFAULT_INPUT,
                        help=f"Path to processed JSONL file (default: {DEFAULT_INPUT})")
    parser.add_argument("--show", action="store_true",
                        help="Display plots in window (default: just save to files)")
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_dir=DEFAULT_OUTPUT_DIR,
        show=args.show,
    )


if __name__ == "__main__":
    main()
