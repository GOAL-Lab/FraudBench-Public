"""LaTeX tables of token usage by model and evaluation setting.

See :mod:`fraudbench.analysis.table.tokens` for the underlying
implementation. This script is a thin CLI wrapper that produces four
``.tex`` files in ``paper/tables/``:

* ``tokens_model_total.tex``    -- total in/out/reasoning tokens emitted
                                   by each evaluated model.
* ``tokens_model_average.tex``  -- mean per-sample in/out/reasoning
                                   tokens (rounded to the nearest whole
                                   token).
* ``tokens_scorer_total.tex``   -- total in/out/reasoning tokens
                                   consumed by the LLM-as-a-judge scorer
                                   ensemble while grading each model's
                                   outputs.
* ``tokens_scorer_average.tex`` -- mean per-sample scorer-side tokens,
                                   rounded as above.

Coverage spans the main results and both the reasoning and temperature
ablation runs. All human-baseline responses are folded into the ``Open``
block (see each table's caption for details). Human respondents remain
excluded from model-token tables because they carry no model-side token
counts, but their LLM-as-a-judge grading calls are included in scorer-token
tables. Generation and LLM-as-a-judge error rows are excluded.

Usage::

    uv run analysis/tables/table_tokens.py
    uv run analysis/tables/table_tokens.py --no-save           # print to stdout
    uv run analysis/tables/table_tokens.py --output-dir tmp/   # custom dir
"""
from pathlib import Path

from jsonargparse import ArgumentParser

from fraudbench.analysis.table.tokens import run


SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "paper" / "tables"


def main() -> None:
    """CLI entry point: parse args and dispatch to ``tokens.run``."""
    parser = ArgumentParser(
        description=(
            "Generate four LaTeX tables of total / average input, output, "
            "and reasoning tokens by evaluated model and setting -- two "
            "for the model side and two for the LLM-as-a-judge scorer "
            "ensemble."
        )
    )
    parser.add_argument(
        "--input", "-i", type=str,
        default="data/final/obs_data.parquet",
        help="Input data file (.parquet, .csv, or .jsonl).",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for token tables (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Print each table to stdout instead of writing files.",
    )
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_dir=args.output_dir,
        no_save=args.no_save,
    )


if __name__ == "__main__":
    main()
