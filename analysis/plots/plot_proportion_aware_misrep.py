"""Plot P(F | A_L) and P(A_L | F) split by fraud type.

Mirrors ``plot_proportion_aware.py`` but restricts to the two
conditional-rate directions involving (fraud, legal_awareness) and
segments each direction into bars for fraudulent misrepresentation
(``is_concealment == False``) vs. fraudulent concealment
(``is_concealment == True``):

    fraud_given_legal_awareness_misrep      P(F | A_L)
    legal_awareness_given_fraud_misrep      P(A_L | F)

Three plot variants per family (mirroring ``plot_proportion_aware.py``):

* ``{prefix}_overall_{setting}.pdf``       -- two bars, data pooled over
  conditions / models / settings (default).
* ``{prefix}_by_condition_{setting}.pdf``  -- nested grouped bars over
  (instructional x contextual) conditions, with one bar per fraud-type
  subset within each cell (default).
* ``{prefix}_by_model_{setting}.pdf``      -- one bar group per model,
  produced with ``--by-model``.
* ``{prefix}_compare_settings.pdf``        -- hatched bars across
  closed/open/agentic, produced with ``--compare-settings``.

See :mod:`fraudbench.analysis.plot.proportion_aware_misrep` for the
underlying implementation. This script is a thin CLI wrapper.

Error bars are Wilson 95% binomial CIs on the empirical conditional rate.

Usage:
    uv run analysis/plots/plot_proportion_aware_misrep.py
    uv run analysis/plots/plot_proportion_aware_misrep.py --by-model
    uv run analysis/plots/plot_proportion_aware_misrep.py --compare-settings
    uv run analysis/plots/plot_proportion_aware_misrep.py \\
        --family fraud_given_legal_awareness_misrep
"""
import argparse

from fraudbench.analysis.common import DEFAULT_INPUT_PATH, VALID_SETTINGS_WITH_ALL
from fraudbench.analysis.plot.case_attributes import DEFAULT_SCENARIO_PATH
from fraudbench.analysis.plot.proportion_aware_misrep import (
    FAMILIES_BY_NAME,
    run,
)


def main():
    """CLI entry point: parse args and dispatch to ``proportion_aware_misrep.run``."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot conditional rates P(F | A_L) and P(A_L | F) split by "
            "fraudulent misrepresentation vs. fraudulent concealment. "
            "Each family produces an overall + by-condition plot by "
            "default; pass --by-model and/or --compare-settings for the "
            "additional variants."
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
        "--scenarios", type=str, default=DEFAULT_SCENARIO_PATH,
        help=(
            f"Scenario-level dataset (default: {DEFAULT_SCENARIO_PATH}). "
            "Used to look up `metadata.is_concealment` per `summary_id`."
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
            "Family to plot. Repeat to select multiple. "
            "Default: plot both families."
        ),
    )
    args = parser.parse_args()

    run(
        setting=args.setting,
        input_path=args.input,
        scenarios_path=args.scenarios,
        by_model=args.by_model,
        compare_settings=args.compare_settings,
        show=args.show,
        families=args.family,
    )


if __name__ == "__main__":
    main()
