"""Experimental-condition plots: overall, by-model, and cross-setting comparisons.

See :mod:`fraudbench.analysis.plot.experimental_conditions` for the underlying
implementation. This script is a thin CLI wrapper.

Usage:
    # Empirical proportions with Wilson 95% binomial CIs.
    uv run analysis/plots/plot_experimental_conditions.py
    uv run analysis/plots/plot_experimental_conditions.py --by-model
    uv run analysis/plots/plot_experimental_conditions.py --compare-settings
"""
import argparse

from fraudbench.analysis.plot.experimental_conditions import run
from fraudbench.analysis.common import DEFAULT_INPUT_PATH, VALID_SETTINGS_WITH_ALL


def main():
    """CLI entry point: parse args and dispatch to ``experimental_conditions.run``."""
    parser = argparse.ArgumentParser(description="Plot overall metric rates by experimental condition")
    parser.add_argument("--setting", "-s", type=str, default="all",
                        choices=VALID_SETTINGS_WITH_ALL,
                        help="Experimental setting to plot (default: all)")
    parser.add_argument("--input", "-i", type=str, default=DEFAULT_INPUT_PATH,
                        help=f"Pre-processed obs-data file (default: {DEFAULT_INPUT_PATH}). "
                             "Run `uv run analysis/process.py` first to (re)generate it from raw eval logs.")
    parser.add_argument("--show", action="store_true",
                        help="Display plots in window (default: just save to files)")
    parser.add_argument("--by-model", action="store_true",
                        help="Also generate the by-model 2x1 chart")
    parser.add_argument("--compare-settings", action="store_true",
                        help="Plot all settings grouped together for each condition")
    args = parser.parse_args()

    run(
        setting=args.setting,
        input_path=args.input,
        by_model=args.by_model,
        compare_settings=args.compare_settings,
        show=args.show,
    )


if __name__ == "__main__":
    main()
