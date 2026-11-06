"""LaTeX tables of ablation study configuration parameters.

Generates two ``tabularx`` snippets (for ``\\input{}``) showing which model
parameters changed relative to the base configuration:

* ``configs_ablations_temperature.tex`` -- temperature ablation models and
  the temperature / max-tokens change applied.
* ``configs_ablations_reasoning.tex``   -- reasoning ablation models and the
  reasoning-effort / reasoning-tokens / max-tokens change applied.

See :mod:`fraudbench.analysis.table.ablations` for the implementation.

Usage::

    uv run analysis/tables/table_configs_ablations.py
    uv run analysis/tables/table_configs_ablations.py --ablation temperature
    uv run analysis/tables/table_configs_ablations.py --ablation reasoning
    uv run analysis/tables/table_configs_ablations.py --no-save
    uv run analysis/tables/table_configs_ablations.py --logs-dir logs/ablations
"""
from pathlib import Path

from jsonargparse import ArgumentParser

from fraudbench.analysis.table.ablations import (
    DEFAULT_LOGS_DIR,
    run_reasoning_ablation,
    run_temperature_ablation,
)


SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_TEMP_OUTPUT = _REPO_ROOT / "paper" / "tables" / "configs_ablations_temperature.tex"
DEFAULT_REASONING_OUTPUT = _REPO_ROOT / "paper" / "tables" / "configs_ablations_reasoning.tex"


def main():
    """CLI entry point: parse args and dispatch to ``ablations.run_*_ablation``."""
    parser = ArgumentParser(
        description=(
            "Generate LaTeX tabularx snippets summarising ablation study "
            "configuration changes relative to the base model configs."
        )
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=DEFAULT_LOGS_DIR,
        help=f"Root directory containing ablation log subdirectories "
             f"(default: {DEFAULT_LOGS_DIR})",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="all",
        choices=["all", "temperature", "reasoning"],
        help="Which ablation table to generate (default: all)",
    )
    parser.add_argument(
        "--output-temperature",
        type=str,
        default=str(DEFAULT_TEMP_OUTPUT),
        help=f"Output path for the temperature ablation table "
             f"(default: {DEFAULT_TEMP_OUTPUT})",
    )
    parser.add_argument(
        "--output-reasoning",
        type=str,
        default=str(DEFAULT_REASONING_OUTPUT),
        help=f"Output path for the reasoning ablation table "
             f"(default: {DEFAULT_REASONING_OUTPUT})",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print to stdout instead of saving to file",
    )
    args = parser.parse_args()

    if args.ablation in ("all", "temperature"):
        run_temperature_ablation(
            logs_dir=args.logs_dir,
            output=args.output_temperature,
            no_save=args.no_save,
        )

    if args.ablation in ("all", "reasoning"):
        run_reasoning_ablation(
            logs_dir=args.logs_dir,
            output=args.output_reasoning,
            no_save=args.no_save,
        )


if __name__ == "__main__":
    main()
