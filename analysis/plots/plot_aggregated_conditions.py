"""Aggregated-condition plots (fraud + refusal) for a given setting.

See :mod:`fraudbench.analysis.plot.aggregated_conditions` for the underlying
implementation. This script is a thin CLI wrapper.

Usage:
    # Empirical proportions with Wilson 95% binomial CIs.
    uv run analysis/plots/plot_aggregated_conditions.py
    uv run analysis/plots/plot_aggregated_conditions.py --compare-settings
    uv run analysis/plots/plot_aggregated_conditions.py --compare-settings-main-text
"""
import argparse

from fraudbench.analysis.plot.aggregated_conditions import run
from fraudbench.analysis.common import DEFAULT_INPUT_PATH, VALID_SETTINGS_WITH_ALL


def main():
    """CLI entry point: parse args and dispatch to ``aggregated_conditions.run``."""
    parser = argparse.ArgumentParser(description="Plot aggregated fraud/refusal rates by condition")
    parser.add_argument("--setting", "-s", type=str, default="all",
                        choices=VALID_SETTINGS_WITH_ALL,
                        help="Experimental setting to plot (default: all)")
    parser.add_argument("--input", "-i", type=str, default=DEFAULT_INPUT_PATH,
                        help=f"Pre-processed obs-data file (default: {DEFAULT_INPUT_PATH}). "
                             "Run `uv run analysis/process.py` first to (re)generate it from raw eval logs.")
    parser.add_argument("--show", action="store_true",
                        help="Display plots in window (default: just save to files)")
    compare_group = parser.add_mutually_exclusive_group()
    compare_group.add_argument("--compare-settings", action="store_true",
                        help="Plot all settings (closed, open, agentic) together with hatching patterns")
    compare_group.add_argument(
        "--compare-settings-main-text",
        action="store_true",
        help="Plot the fraud-and-refusal-only all-settings comparison for the main text",
    )
    args = parser.parse_args()

    run(
        setting=args.setting,
        input_path=args.input,
        compare_settings=args.compare_settings,
        compare_settings_main_text=args.compare_settings_main_text,
        show=args.show,
    )


if __name__ == "__main__":
    main()
