"""Plot conditional-rate plots over (fraud, refusal, awareness) variables.

By default generates four families, one per direction of the pairwise
conditional probabilities between fraud / refusal and the two awareness
variables (legal awareness, fraud awareness):

    awareness_given_fraud      P(awareness | fraud)
    fraud_given_awareness      P(fraud | awareness)
    refusal_given_awareness    P(refusal | awareness)
    awareness_given_refusal    P(awareness | refusal)

See :mod:`fraudbench.analysis.plot.proportion_aware` for the underlying
implementation. This script is a thin CLI wrapper.

Error bars are Wilson 95% binomial CIs on the empirical conditional rate.

Usage:
    uv run analysis/plots/plot_proportion_aware.py
    uv run analysis/plots/plot_proportion_aware.py --by-model
    uv run analysis/plots/plot_proportion_aware.py --compare-settings
    uv run analysis/plots/plot_proportion_aware.py --family awareness_given_fraud
    uv run analysis/plots/plot_proportion_aware.py \\
        --family fraud_given_awareness --family refusal_given_awareness
"""
import argparse

from fraudbench.analysis.plot.proportion_aware import FAMILIES_BY_NAME, run
from fraudbench.analysis.common import DEFAULT_INPUT_PATH, VALID_SETTINGS_WITH_ALL


def main():
    """CLI entry point: parse args and dispatch to ``proportion_aware.run``."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot conditional rates: P(awareness | fraud), P(fraud | awareness), "
            "P(refusal | awareness), and P(awareness | refusal). "
            "Each family produces a by-condition plot by default; pass "
            "--by-model and/or --compare-settings for the additional variants."
        ),
    )
    parser.add_argument(
        "--setting", "-s", type=str, default="all",
        choices=VALID_SETTINGS_WITH_ALL,
        help="Experimental setting to plot (default: all)",
    )
    parser.add_argument(
        "--input", "-i", type=str, default=DEFAULT_INPUT_PATH,
        help=(
            f"Pre-processed obs-data file (default: {DEFAULT_INPUT_PATH}). "
            "Run `uv run analysis/process.py` first to (re)generate it from "
            "raw eval logs."
        ),
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display plots in window (default: just save to files)",
    )
    parser.add_argument(
        "--by-model", action="store_true",
        help="Also generate the by-model chart for each family",
    )
    parser.add_argument(
        "--compare-settings", action="store_true",
        help="Plot all settings grouped together for each condition (per family)",
    )
    parser.add_argument(
        "--family", action="append", choices=list(FAMILIES_BY_NAME),
        help=(
            "Family to plot. Repeat to select multiple families. "
            "Default: plot all four families."
        ),
    )
    args = parser.parse_args()

    run(
        setting=args.setting,
        input_path=args.input,
        by_model=args.by_model,
        compare_settings=args.compare_settings,
        show=args.show,
        families=args.family,
    )


if __name__ == "__main__":
    main()
