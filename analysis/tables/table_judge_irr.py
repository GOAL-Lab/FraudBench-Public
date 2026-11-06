"""LaTeX table of LLM-judge accuracy and inter-rater reliability.

See :mod:`fraudbench.analysis.table.judge_irr` for the underlying
implementation. This script is a thin CLI wrapper that emits a single
LaTeX table with two panels:

* **(a)** Per-judge accuracy vs. the ensemble majority vote.
* **(b)** Cohen's $\\kappa$ between the LLM majority vote and human consensus.
* **(c)** Fleiss' $\\kappa$ among all LLM judges.
* **(d)** Fleiss' $\\kappa$ among human annotators.

Human-baseline responses are excluded from every panel.

Usage:
    uv run analysis/tables/table_judge_irr.py
    uv run analysis/tables/table_judge_irr.py --no-save
    uv run analysis/tables/table_judge_irr.py \\
        --input data/final/obs_data_per_score.parquet \\
        --output paper/tables/judge_irr.tex
"""
from pathlib import Path

from jsonargparse import ArgumentParser

from fraudbench.analysis.table.judge_irr import (
    DEFAULT_HUMAN_ANNOTATIONS_PATH,
    DEFAULT_TIE_BREAK_SEED,
    run,
)


SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT = _REPO_ROOT / "paper" / "tables" / "judge_irr.tex"


def main() -> None:
    """CLI entry point: parse args and dispatch to ``judge_irr.run``."""
    parser = ArgumentParser(
        description=(
            "Generate a LaTeX table of LLM-judge accuracy vs. the ensemble "
            "majority vote, LLM/human Cohen's kappa, and Fleiss' kappa."
        )
    )
    parser.add_argument(
        "--input", "-i", type=str,
        default="data/final/obs_data_per_score.parquet",
        help="Input per-score data file (.parquet, .csv, or .jsonl).",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output LaTeX file path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--human-annotations",
        type=str,
        default=str(DEFAULT_HUMAN_ANNOTATIONS_PATH),
        help=(
            "JSONL file of human annotations, one rating per line with an "
            "'annotator_id' field; an item may be rated by several annotators."
        ),
    )
    parser.add_argument(
        "--validated-samples",
        type=str,
        default=None,
        help=(
            "Optional CSV with the complete validation-sample ID list "
            "(column 'Sample ID'). Defaults to every human-annotated item."
        ),
    )
    parser.add_argument(
        "--obs-data",
        type=str,
        default=str(_REPO_ROOT / "data" / "final" / "obs_data.parquet"),
        help="Sample-level obs_data used to calculate population weights.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TIE_BREAK_SEED,
        help="Seed for reproducible random resolution of human-vote ties.",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save to file, only print to stdout.",
    )
    parser.add_argument(
        "--no-percent", action="store_true",
        help="Display panel (a) accuracies as proportions in [0, 1] "
             "instead of percentages.",
    )
    args = parser.parse_args()

    run(
        input_path=args.input,
        output=args.output,
        human_annotations_path=args.human_annotations,
        validated_samples_path=args.validated_samples,
        obs_data_path=args.obs_data,
        seed=args.seed,
        no_save=args.no_save,
        no_percent=args.no_percent,
    )


if __name__ == "__main__":
    main()
