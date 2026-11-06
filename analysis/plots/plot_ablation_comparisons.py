"""Per-model main-vs-ablation comparison plots.

Generates one figure per ablation study (reasoning / temperature) showing
fraud, refusal, options, legal awareness, and fraud awareness rates per
model, with main-runs and ablation rates side-by-side. Only the models
that participated in each ablation (per
:data:`fraudbench.utils.globals.REASONING_ABLATION_MODELS` /
:data:`TEMPERATURE_ABLATION_MODELS`) are included.

See :mod:`fraudbench.analysis.plot.ablation_comparisons` for the underlying
implementation.

Usage:
    # Empirical proportions with Wilson 95% binomial CIs.
    uv run analysis/plots/plot_ablation_comparisons.py
    uv run analysis/plots/plot_ablation_comparisons.py --ablation reasoning
    uv run analysis/plots/plot_ablation_comparisons.py --ablation temperature
    uv run analysis/plots/plot_ablation_comparisons.py --setting closed
"""
import argparse

from fraudbench.analysis.common import DEFAULT_INPUT_PATH, VALID_SETTINGS_WITH_ALL
from fraudbench.analysis.plot.ablation_comparisons import (
    VALID_ABLATION_TYPES,
    run,
)


def main():
    """CLI entry point: parse args and dispatch to ``ablation_comparisons.run``."""
    parser = argparse.ArgumentParser(
        description="Plot fraud, refusal, options, legal awareness, and fraud "
                    "awareness rates per model for main vs reasoning/temperature "
                    "ablations.",
    )
    parser.add_argument("--ablation", "-a", type=str, default="all",
                        choices=list(VALID_ABLATION_TYPES) + ["all"],
                        help="Which ablation comparison to plot (default: all)")
    parser.add_argument("--setting", "-s", type=str, default="all",
                        choices=VALID_SETTINGS_WITH_ALL,
                        help="Experimental setting to restrict to (default: all)")
    parser.add_argument("--input", "-i", type=str, default=DEFAULT_INPUT_PATH,
                        help=f"Pre-processed obs-data file (default: {DEFAULT_INPUT_PATH}). "
                             "Run `uv run analysis/process.py` first to (re)generate it from raw eval logs.")
    parser.add_argument("--show", action="store_true",
                        help="Display plots in window (default: just save to files)")
    args = parser.parse_args()

    run(
        ablation=args.ablation,
        setting=args.setting,
        input_path=args.input,
        show=args.show,
    )


if __name__ == "__main__":
    main()
