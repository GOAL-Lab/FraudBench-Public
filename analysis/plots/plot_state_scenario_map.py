"""Map FraudBench scenario counts across U.S. states and DC.

State labels are read directly from the top-level ``state`` field in
``data/final/dataset_public.jsonl``.

See :mod:`fraudbench.analysis.plot.state_scenario_map` for the implementation.
This script is a thin CLI wrapper.

Usage:
    uv run analysis/plots/plot_state_scenario_map.py
"""
from pathlib import Path

from jsonargparse import ArgumentParser

from fraudbench.analysis.plot.state_scenario_map import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SCENARIOS,
    run,
)


def main() -> None:
    """Parse CLI arguments and generate the state choropleth."""
    parser = ArgumentParser(
        description=(
            "Generate a U.S. choropleth of FraudBench scenario counts by state."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=DEFAULT_SCENARIOS,
        help=f"Scenario-level dataset (default: {DEFAULT_SCENARIOS})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save figures (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure in a window",
    )
    args = parser.parse_args()
    run(
        input_path=args.input,
        output_dir=Path(args.output_dir),
        show=args.show,
    )


if __name__ == "__main__":
    main()
