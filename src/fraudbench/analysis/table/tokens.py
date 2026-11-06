"""LaTeX tables of token usage by model and evaluation setting.

Backs ``analysis/tables/table_tokens.py``.

Renders four LaTeX tables aggregating token consumption from
``data/final/obs_data.parquet``:

* ``tokens_model_total.tex``    -- total in/out/reasoning tokens emitted by
                                   each evaluated model.
* ``tokens_model_average.tex``  -- the same, but averaged per sample
                                   (rounded to the nearest whole token).
* ``tokens_scorer_total.tex``   -- total in/out/reasoning tokens consumed
                                   by the LLM-as-a-judge scorer ensemble
                                   while grading each model's outputs
                                   (summed across all scorer models).
* ``tokens_scorer_average.tex`` -- the same, but averaged per sample.

The model-token tables have rows for every evaluated AI model; the
LLM-as-a-judge tables additionally include a ``Human`` row for jury calls
used to grade human-baseline responses. Each has a final ``Total`` row
(across its included response sources). Columns are grouped by setting
(Binary / Open / Agentic) plus an Overall block; within each block we report
``In`` / ``Out`` / ``Reas.`` columns.

Coverage is the union of the main results and the
``reasoning_ablation`` / ``temperature_ablation`` runs, so a single number
captures everything we paid an inference provider for. All responses to the
human-baseline subset are folded into the ``Open`` block, mirroring the
human baseline's free-response format. Actual human respondents carry no
model-side token counts and are excluded from model-token tables, but their
LLM-as-a-judge calls are retained in scorer-token tables. Rows with either
generation or judge errors are excluded from all token aggregates.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..common import (
    ABLATION_RUN_CATEGORIES,
    AI_SETTINGS,
    HUMAN_BASELINE_SETTING,
    HUMAN_MODEL_ID,
    MAIN_RUN_CATEGORY,
    ensure_model_id_column,
    get_model_display_name,
    get_model_family,
)
from ..registry import (
    MODEL_ORDER,
    REASONING_MODELS,
    REASONING_SUBSCRIPT_CAPTION,
    REASONING_SYMBOL,
)


# ---------------------------------------------------------------------------
# Shared layout constants
# ---------------------------------------------------------------------------

#: Settings to render, in display order. The two ablation studies were only
#: run for these settings; ``human_baseline`` rows are folded into ``open``.
SETTING_ORDER: list[str] = ["closed", "open", "agentic"]

SETTING_DISPLAY: dict[str, str] = {
    "closed":  "Binary",
    "open":    "Open",
    "agentic": "Agentic",
}

#: Suffix label for the "across-settings" aggregate block.
OVERALL_DISPLAY: str = "Overall"

#: Token-type sub-columns within each setting block.
TOKEN_TYPES: list[tuple[str, str]] = [
    ("in",        "In"),
    ("out",       "Out"),
    ("reasoning", "Reasoning"),
]


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------

_NEEDED_COLUMNS: list[str] = [
    "model",
    "setting",
    "run_category",
    "has_sampling_error",
    "has_judge_error",
    "tokens_in",
    "tokens_out",
    "tokens_reasoning",
    "scorer_tokens",
]


def load_data(input_path: str) -> pd.DataFrame:
    """Load the obs-data file (parquet/csv/jsonl), with only the columns we need."""
    path = Path(input_path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(input_path, columns=_NEEDED_COLUMNS)
    elif path.suffix == ".csv":
        df = pd.read_csv(input_path)
    elif path.suffix == ".jsonl":
        df = pd.read_json(input_path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    print(f"Loaded {len(df)} rows from {input_path}")
    return ensure_model_id_column(df)


def _parse_scorer_tokens(raw: object) -> tuple[int, int, int]:
    """Sum ``input`` / ``output`` / ``reasoning`` across all scorer models.

    The ``scorer_tokens`` parquet column stores a JSON object keyed by
    grader model id (e.g. ``claude-haiku-4-5-20251001``,
    ``gemini-2.5-flash-lite``, ``gpt-5-nano-2025-08-07``); each value is
    itself a ``{input, output, reasoning}`` dict. We collapse that nested
    structure into a single (in, out, reas) triple per row by summing
    over all scorers, since the scorer ensemble runs as one unit when
    grading a sample.

    Empty / missing payloads collapse to ``(0, 0, 0)``.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return (0, 0, 0)
    if not isinstance(raw, str) or not raw.strip():
        return (0, 0, 0)
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return (0, 0, 0)
    if not isinstance(parsed, dict):
        return (0, 0, 0)
    in_t = 0
    out_t = 0
    reas_t = 0
    for entry in parsed.values():
        if not isinstance(entry, dict):
            continue
        in_t += int(entry.get("input", 0) or 0)
        out_t += int(entry.get("output", 0) or 0)
        reas_t += int(entry.get("reasoning", 0) or 0)
    return (in_t, out_t, reas_t)


def _add_scorer_token_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Materialize ``scorer_tokens_{in,out,reasoning}`` from the JSON column.

    Parsing 300k JSON strings via ``apply`` is the slowest step in this
    module by far, but only happens once per script invocation. We
    short-circuit when the parquet vintage already pre-flattened the
    payload (older pipelines wrote the totals as bare numeric columns).
    """
    needed = ["scorer_tokens_in", "scorer_tokens_out", "scorer_tokens_reasoning"]
    if all(col in df.columns for col in needed):
        return df

    parsed = df["scorer_tokens"].apply(_parse_scorer_tokens)
    df = df.copy()
    df["scorer_tokens_in"] = parsed.map(lambda t: t[0]).astype("int64")
    df["scorer_tokens_out"] = parsed.map(lambda t: t[1]).astype("int64")
    df["scorer_tokens_reasoning"] = parsed.map(lambda t: t[2]).astype("int64")
    return df


def prepare(df: pd.DataFrame, *, include_human: bool) -> pd.DataFrame:
    """Filter, relabel, and decode the rows we want to aggregate.

    Steps:

    1. Exclude rows flagged with generation or judge errors
       (``has_sampling_error`` / legacy ``has_error`` and
       ``has_judge_error`` respectively).
    2. Drop actual human respondents (``model_id == 'human'``) for model-side
       aggregates only; retain them for scorer-side aggregates so jury calls
       grading human responses are counted.
    3. Keep ``run_category in {'main', 'reasoning_ablation', 'temperature_ablation'}``
       so the totals reflect every paid inference run.
    4. Re-label ``setting == 'human_baseline'`` to ``'open'`` so AI re-runs
       on the human-baseline scenario subset are folded into the Open
       block (per spec; surfaced in each table caption).
    5. Restrict to the AI evaluation settings (closed / open / agentic).
    6. Materialize per-row scorer-token columns from the JSON payload.
    """
    df = df.copy()

    generation_error_column = (
        "has_sampling_error" if "has_sampling_error" in df.columns
        else "has_error" if "has_error" in df.columns
        else None
    )
    for error_column in (generation_error_column, "has_judge_error"):
        if error_column is not None and error_column in df.columns:
            df = df[df[error_column] == False]  # noqa: E712

    if not include_human:
        df = df[df["model_id"] != HUMAN_MODEL_ID]

    if "run_category" in df.columns:
        keep_run = (MAIN_RUN_CATEGORY,) + ABLATION_RUN_CATEGORIES
        rc = df["run_category"]
        df = df[rc.isna() | rc.isin(keep_run)]

    df.loc[df["setting"] == HUMAN_BASELINE_SETTING, "setting"] = "open"
    df = df[df["setting"].isin(AI_SETTINGS)]

    df = _add_scorer_token_columns(df)
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TokenSpec:
    """One renderable token-source family (model-side or scorer-side).

    ``column_prefix`` selects the parquet columns: e.g. ``"tokens"`` →
    ``tokens_in`` / ``tokens_out`` / ``tokens_reasoning``;
    ``"scorer_tokens"`` → ``scorer_tokens_in`` / ... .
    """

    key: str
    label: str           # short label used in captions / log lines
    column_prefix: str   # parquet column prefix (without trailing _<type>)


MODEL_SPEC: TokenSpec = TokenSpec(
    key="model",
    label="evaluated model",
    column_prefix="tokens",
)

SCORER_SPEC: TokenSpec = TokenSpec(
    key="scorer",
    label="scorer ensemble",
    column_prefix="scorer_tokens",
)


def _token_columns(spec: TokenSpec) -> list[str]:
    return [f"{spec.column_prefix}_{ttype}" for ttype, _ in TOKEN_TYPES]


def aggregate(
    df: pd.DataFrame,
    spec: TokenSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Return the four cross-cuts the renderer needs.

    Returns ``(per_cell_sum, per_cell_count, overall_sum_by_model,
    n_by_model)``:

    * ``per_cell_sum``       -- ``{(model_id, setting): {ttype: int}}``
    * ``per_cell_count``     -- ``{(model_id, setting): int}`` (sample size)
    * ``overall_sum_by_model`` -- ``{model_id: {ttype: int}}`` (sum over settings)
    * ``n_by_model``         -- ``{model_id: int}`` (sum over settings)

    Per-cell counts use the row count of ``df`` (every kept row contributes
    one sample to the denominator); the parquet pipeline guarantees
    non-null token columns for every retained row.
    """
    cols = _token_columns(spec)
    grouped = (
        df.groupby(["model_id", "setting"], observed=True)
        .agg(
            **{c: (c, "sum") for c in cols},
            n=("model_id", "size"),
        )
        .reset_index()
    )

    per_cell_sum: dict[tuple[str, str], dict[str, int]] = {}
    per_cell_count: dict[tuple[str, str], int] = {}
    for _, row in grouped.iterrows():
        key = (row["model_id"], row["setting"])
        per_cell_sum[key] = {
            ttype: int(row[f"{spec.column_prefix}_{ttype}"])
            for ttype, _ in TOKEN_TYPES
        }
        per_cell_count[key] = int(row["n"])

    overall_sum_by_model: dict[str, dict[str, int]] = {}
    n_by_model: dict[str, int] = {}
    for model_id, sub in grouped.groupby("model_id"):
        overall_sum_by_model[str(model_id)] = {
            ttype: int(sub[f"{spec.column_prefix}_{ttype}"].sum())
            for ttype, _ in TOKEN_TYPES
        }
        n_by_model[str(model_id)] = int(sub["n"].sum())

    return per_cell_sum, per_cell_count, overall_sum_by_model, n_by_model


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------

def _ordered_models(models_present: Iterable[str]) -> list[str]:
    """Return present model ids in canonical ``MODEL_ORDER`` first, then leftovers."""
    present = set(models_present)
    ordered = [m for m in MODEL_ORDER if m in present]
    ordered.extend([m for m in present if m not in ordered])
    return ordered


def _format_count(value: float, mode: str) -> str:
    """Format one numeric cell.

    * ``mode='total'``   -- integer with thousands separators (LaTeX-safe
                            commas).
    * ``mode='average'`` -- rounded to the nearest whole token, with
                            thousands separators.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "---"
    if mode == "total":
        return f"{int(round(value)):,}"
    if mode == "average":
        return f"{int(round(value)):,}"
    raise ValueError(f"Unknown format mode: {mode!r}")


def _model_display_for(model_id: str) -> str:
    name = get_model_display_name(model_id)
    if model_id in REASONING_MODELS:
        return f"{name} {REASONING_SYMBOL}"
    return name


def _is_last_in_family(model_id: str, ordered_models: list[str]) -> bool:
    family = get_model_family(model_id)
    idx = ordered_models.index(model_id)
    if idx >= len(ordered_models) - 1:
        return False
    return get_model_family(ordered_models[idx + 1]) != family


def _render_table(
    *,
    per_cell_sum: dict[tuple[str, str], dict[str, int]],
    per_cell_count: dict[tuple[str, str], int],
    overall_sum_by_model: dict[str, dict[str, int]],
    n_by_model: dict[str, int],
    ordered_models: list[str],
    mode: str,
    caption: str,
    label: str,
) -> str:
    """Render one of the four token tables.

    Layout (all four tables share this skeleton):

        Model | Binary [In | Out | Reas.] | Open ... | Agentic ... | Overall ...

    The trailing ``Overall`` block is the per-model sum / mean across the
    three settings. The final ``Total`` row is the per-setting (and
    overall) aggregate over all models.

    Wrapped in ``sidewaystable*`` (rotated 90 deg via the ``rotating``
    package) because the four blocks of three numeric columns each --
    plus the seven-digit token totals -- overflow a portrait page.
    """
    n_settings = len(SETTING_ORDER)
    n_blocks = n_settings + 1  # +1 for the Overall block
    n_cols_per_block = len(TOKEN_TYPES)
    n_data_cols = n_blocks * n_cols_per_block

    col_spec = "l" + "r" * n_data_cols
    T1 = "    "
    T2 = "        "

    lines: list[str] = []
    # ``sidewaystable*`` does not accept a placement specifier; the
    # ``rotating`` package always typesets it on its own page.
    lines.append("\\begin{sidewaystable*}")
    lines.append(f"{T1}\\centering")
    lines.append(f"{T1}\\footnotesize")
    lines.append(f"{T1}\\setlength{{\\tabcolsep}}{{3pt}}")
    lines.append(f"{T1}\\begin{{tabular}}{{{col_spec}}}")
    lines.append(f"{T2}\\toprule")

    header1: list[str] = [""]
    block_displays = [SETTING_DISPLAY[s] for s in SETTING_ORDER] + [OVERALL_DISPLAY]
    for display in block_displays:
        header1.append(
            f"\\multicolumn{{{n_cols_per_block}}}{{c}}{{\\textbf{{{display}}}}}"
        )
    lines.append(f"{T2}" + " & ".join(header1) + " \\\\")

    cmid_parts: list[str] = []
    cursor = 2  # first numeric column index (1-indexed) after the model column
    for _ in block_displays:
        start = cursor
        end = cursor + n_cols_per_block - 1
        cmid_parts.append(f"\\cmidrule(lr){{{start}-{end}}}")
        cursor = end + 1
    lines.append(f"{T2}" + " ".join(cmid_parts))

    # The In / Out / Reas. headers are wrapped in ``\multicolumn{1}{c}{}``
    # so they sit centered over their right-aligned numeric columns, which
    # otherwise force the labels to hug the right edge.
    header2: list[str] = ["\\textbf{Model}"]
    for _ in block_displays:
        for _, display in TOKEN_TYPES:
            header2.append(f"\\multicolumn{{1}}{{c}}{{\\textbf{{{display}}}}}")
    lines.append(f"{T2}" + " & ".join(header2) + " \\\\")
    lines.append(f"{T2}\\midrule")

    setting_totals: dict[str, dict[str, int]] = {
        setting: {ttype: 0 for ttype, _ in TOKEN_TYPES}
        for setting in SETTING_ORDER
    }
    setting_n: dict[str, int] = {setting: 0 for setting in SETTING_ORDER}
    overall_totals: dict[str, int] = {ttype: 0 for ttype, _ in TOKEN_TYPES}
    overall_n: int = 0

    for model_id in ordered_models:
        row_parts: list[str] = [_model_display_for(model_id)]

        for setting in SETTING_ORDER:
            sums = per_cell_sum.get((model_id, setting))
            n = per_cell_count.get((model_id, setting), 0)
            for ttype, _ in TOKEN_TYPES:
                if sums is None or n == 0:
                    row_parts.append("---")
                else:
                    s = sums.get(ttype, 0)
                    if mode == "total":
                        val = s
                    else:  # average
                        val = s / n if n > 0 else float("nan")
                    row_parts.append(_format_count(val, mode))
                    setting_totals[setting][ttype] += s
            setting_n[setting] += n

        overall_sums = overall_sum_by_model.get(model_id, {ttype: 0 for ttype, _ in TOKEN_TYPES})
        n_overall = n_by_model.get(model_id, 0)
        for ttype, _ in TOKEN_TYPES:
            s = overall_sums.get(ttype, 0)
            if n_overall == 0:
                row_parts.append("---")
            else:
                val = s if mode == "total" else (s / n_overall)
                row_parts.append(_format_count(val, mode))
            overall_totals[ttype] += s
        overall_n += n_overall

        lines.append(f"{T2}" + " & ".join(row_parts) + " \\\\")

        if _is_last_in_family(model_id, ordered_models):
            lines.append(f"{T2}\\midrule")

    lines.append(f"{T2}\\midrule")
    total_parts: list[str] = ["\\textbf{Total}"]
    for setting in SETTING_ORDER:
        n = setting_n[setting]
        for ttype, _ in TOKEN_TYPES:
            s = setting_totals[setting][ttype]
            if n == 0:
                total_parts.append("---")
            else:
                val = s if mode == "total" else (s / n)
                total_parts.append(f"\\textbf{{{_format_count(val, mode)}}}")
    for ttype, _ in TOKEN_TYPES:
        s = overall_totals[ttype]
        if overall_n == 0:
            total_parts.append("---")
        else:
            val = s if mode == "total" else (s / overall_n)
            total_parts.append(f"\\textbf{{{_format_count(val, mode)}}}")
    lines.append(f"{T2}" + " & ".join(total_parts) + " \\\\")

    lines.append(f"{T2}\\bottomrule")
    lines.append(f"{T1}\\end{{tabular}}")
    lines.append(f"{T1}\\caption{{{caption}}}")
    lines.append(f"{T1}\\label{{{label}}}")
    lines.append("\\end{sidewaystable*}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Captions / labels
# ---------------------------------------------------------------------------

#: Disclaimer surfaced in every caption: documents run coverage, the
#: human-baseline-as-Open relabel, and the error exclusion policy.
_COVERAGE_NOTE = (
    "Coverage spans all results, including main results, reasoning and "
    "temperature ablations, and LLM responses to the human baseline items "
    "(reported under ``Open''). Rows with generation or LLM-as-a-judge "
    "errors are excluded. "
)

_REASONING_NOTE = (
    f"{REASONING_SUBSCRIPT_CAPTION} (Reflects main run configuration only)."
)


def _caption(spec: TokenSpec, mode: str) -> str:
    if spec.key == "model":
        subject_phrase = (
            "input, output, and reasoning tokens by "
            "evaluated model"
        )
    else:
        subject_phrase = (
            "input, output, and reasoning tokens consumed by the three-member "
            "LLM-as-a-judge jury while grading evaluated LLM and human "
            "responses"
        )

    if mode == "total":
        head = f"Total counts of {subject_phrase}, broken out by task and overall."
    else:
        head = (
            f"Mean counts of {subject_phrase} per generation, "
            "rounded to the nearest whole token, broken out by task "
            "and overall."
        )

    return f"{head} {_COVERAGE_NOTE} {_REASONING_NOTE}"


def _label(spec: TokenSpec, mode: str) -> str:
    return f"tab:tokens_{spec.key}_{mode}"


def _output_filename(spec: TokenSpec, mode: str) -> str:
    return f"tokens_{spec.key}_{mode}.tex"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "paper" / "tables"


def run(
    input_path: str = "data/final/obs_data.parquet",
    output_dir: Optional[str] = None,
    *,
    no_save: bool = False,
) -> dict[tuple[str, str], Path]:
    """Generate all four token tables (model + scorer; total + average).

    Args:
        input_path: Observation-level data (parquet/csv/jsonl).
        output_dir: Directory for the rendered ``.tex`` files. Defaults to
            ``paper/tables/`` at the repo root.
        no_save: If True, print each table to stdout instead of writing a
            file. The dict returned in this case maps to the *intended*
            output paths even though no files were written.

    Returns:
        Mapping from ``(spec_key, mode)`` to the resolved output path.
    """
    out_dir = Path(output_dir) if output_dir else _default_output_dir()

    raw_df = load_data(input_path)
    print("Preparing data (filtering, decoding scorer_tokens JSON)...")
    prepared_by_spec = {
        MODEL_SPEC.key: prepare(raw_df, include_human=False),
        SCORER_SPEC.key: prepare(raw_df, include_human=True),
    }

    written: dict[tuple[str, str], Path] = {}
    for spec in (MODEL_SPEC, SCORER_SPEC):
        df = prepared_by_spec[spec.key]
        ordered_models = _ordered_models(df["model_id"].unique())
        print(
            f"Aggregating {spec.label} tokens across {len(ordered_models)} "
            f"response sources and {len(df)} samples."
        )
        per_cell_sum, per_cell_count, overall_sum, n_by_model = aggregate(df, spec)

        for mode in ("total", "average"):
            latex = _render_table(
                per_cell_sum=per_cell_sum,
                per_cell_count=per_cell_count,
                overall_sum_by_model=overall_sum,
                n_by_model=n_by_model,
                ordered_models=ordered_models,
                mode=mode,
                caption=_caption(spec, mode),
                label=_label(spec, mode),
            )

            out_path = out_dir / _output_filename(spec, mode)
            written[(spec.key, mode)] = out_path

            if no_save:
                print("\n" + "=" * 70)
                print(f"{spec.label.upper()} -- {mode.upper()} (would write to {out_path})")
                print("=" * 70)
                print(latex)
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path.write_text(latex, encoding="utf-8")
                print(f"  Wrote {out_path}")

    _print_summary(prepared_by_spec)
    return written


def _print_summary(prepared_by_spec: dict[str, pd.DataFrame]) -> None:
    """Echo a quick text summary so the script's stdout is informative."""
    print("\n" + "=" * 60)
    print("TOKEN USAGE SUMMARY (after coverage and error filtering)")
    print("=" * 60)
    for spec in (MODEL_SPEC, SCORER_SPEC):
        df = prepared_by_spec[spec.key]
        cols = _token_columns(spec)
        sums = {col: int(df[col].sum()) for col in cols}
        total = sum(sums.values())
        print(
            f"{spec.label.title()} ({len(df):,} samples): in={sums[cols[0]]:,}  "
            f"out={sums[cols[1]]:,}  reas={sums[cols[2]]:,}  "
            f"(total {total:,})"
        )
