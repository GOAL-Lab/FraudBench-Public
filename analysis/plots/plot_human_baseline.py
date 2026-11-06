"""Human baseline vs AI comparison plots.

See :mod:`fraudbench.analysis.plot.human_baseline` for the underlying
implementation. This script is a thin CLI wrapper.

Usage:
    uv run analysis/plots/plot_human_baseline.py
"""
import argparse

from fraudbench.analysis.plot.human_baseline import run
from fraudbench.analysis.common import DEFAULT_INPUT_PATH


def main():
    """CLI entry point: parse args and dispatch to ``human_baseline.run``."""
    parser = argparse.ArgumentParser(description="Plot human baseline vs AI evaluation results")
    parser.add_argument("--input", "-i", type=str, default=DEFAULT_INPUT_PATH,
                        help=f"Pre-processed obs-data file (default: {DEFAULT_INPUT_PATH}). "
                             "Run `uv run analysis/process.py` first to (re)generate it from raw eval logs.")
    parser.add_argument("--show", action="store_true",
                        help="Display plots in window (default: just save to files)")
    args = parser.parse_args()

    run(
        input_path=args.input,
        show=args.show,
    )


if __name__ == "__main__":
    main()
