"""Per-model appendix tables: condition-level results for every metric/setting.

For each AI model in :data:`fraudbench.analysis.registry.MODEL_ORDER`, this
module renders a wide LaTeX table covering the full 5x5 ``instruction x
context`` design across the three evaluation settings (Binary, Open,
Agentic). Each setting block reports ``N``, ``Fraud``, ``Refusal``, ``Legal``,
``Fraud Aw.``, and ``Options``; ``N`` is shown as ``N_eff/N_expected`` so
reviewers can see how many samples informed each cell relative to the design
size. The full-design tables are wide enough that they're wrapped in a
``sidewaystable*`` (rotated 90 deg via the ``rotating`` package) so they fit
on a landscape-oriented appendix page.

All scoring uses the final binary labels supplied in the data; nothing is
recomputed from free text. Where a metric column is missing (e.g. the current
``obs_data.parquet`` was generated before ``score_options`` existed) the
relevant cells are rendered blank so the column slot is a placeholder.

Each metric value carries a Wilson 95% binomial CI on the empirical
proportion, rendered as a smaller bracketed second line beneath the rate --
a descriptive choice that always contains the cell value. Inferential
output (regression coefficients, random-effect forests) lives in the
dedicated ``regression_coefs_*`` / ``regression_re_*`` tables.

Outputs:
    paper/tables/per_model/per_model_<slug>.tex -- one LaTeX ``table*`` per model.
    paper/tables/per_model_all.tex              -- master file that ``\\input``s every
                                                   per-model ``.tex`` in MODEL_ORDER.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..registry import (
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
    reasoning_label,
    REASONING_SUBSCRIPT_CAPTION,
    REASONING_SYMBOL,
)
from ..common import (
    CONTEXTUAL_CONDITIONS,
    INSTRUCTIONAL_CONDITIONS,
    add_condition_columns,
    filter_main_runs,
    get_model_display_name,
    load_data_from_source,
)
from ..style import latex_cell_color, latex_color_definitions
from ..uncertainty import wilson_interval


# ---------------------------------------------------------------------------
# Design constants
# ---------------------------------------------------------------------------

#: Instruction conditions in display order. ``"control"`` is the absence of
#: any instructional flag (no ``assist_user``/``high_agency``/etc.).
INSTRUCTION_ORDER: list[str] = [
    "control",
    "maximize_profit",
    "assist_user",
    "high_agency",
    "no_illegal_behavior",
]

INSTRUCTION_LABELS: dict[str, str] = {
    "control":             "Control",
    "maximize_profit":     "Maximize Profit",
    "assist_user":         "Maximize User Assistance",
    "high_agency":         "High Agency",
    "no_illegal_behavior": "No Illegal Behavior",
}

#: Context conditions in display order. ``"control"`` is the absence of any
#: contextual flag.
CONTEXT_ORDER: list[str] = [
    "control",
    "peer_pressure_low",
    "peer_pressure_high",
    "negative_incentive_low",
    "negative_incentive_high",
]

CONTEXT_LABELS: dict[str, str] = {
    "control":                 "Control",
    "peer_pressure_low":       "Peer Pressure (Low)",
    "peer_pressure_high":      "Peer Pressure (High)",
    "negative_incentive_low":  "Neg. Incentive (Low)",
    "negative_incentive_high": "Neg. Incentive (High)",
}

#: Default expected denominator per (setting, instruction, context) cell. The
#: real per-cell denominator is computed empirically (see
#: :func:`_expected_per_cell`) but this is the documented design size and is
#: used as a fallback / sanity-check.
DEFAULT_N_EXPECTED: int = 159

#: AI evaluation settings to render, in display order.
SETTING_ORDER: list[str] = ["closed", "open", "agentic"]

SETTING_DISPLAY: dict[str, str] = {
    "closed":  "Binary",
    "open":    "Open",
    "agentic": "Agentic",
}

#: Fixed widths for the condition-label columns. ``p{...}`` columns wrap
#: their contents automatically, keeping long condition names from widening
#: the table.
INSTRUCTION_COLUMN_WIDTH = "1.5cm"
CONTEXT_COLUMN_WIDTH = "2.25cm"

#: Metric specs in display order. Each entry maps an internal metric key to
#: the canonical column name (after :func:`_normalize_columns`), the
#: short-form column header used in the LaTeX table, and the LaTeX heatmap
#: color name used to shade its cells (white at 0% -> deep tint at 100%).
@dataclass(frozen=True)
class MetricSpec:
    """One metric column: internal key, parquet column name, header, color."""

    key: str
    column: str
    header: str
    color: str           # ``xcolor`` color name registered in the table preamble
    palette_key: str     # palette key in :data:`fraudbench.analysis.style.COLORS`


METRICS: list[MetricSpec] = [
    MetricSpec("fraud",           "score_fraud",           "Fraud",
               color="fraudcolor",        palette_key="bright_red"),
    MetricSpec("refusal",         "score_refusal",         "Refusal",
               color="refusalcolor",      palette_key="blue"),
    MetricSpec("legal_awareness", "score_legal_awareness_overall", "Legal Aw.",
               color="legalcolor",        palette_key="dark_purple"),
    MetricSpec("fraud_awareness", "score_fraud_awareness_overall", "Fraud Aw.",
               color="fraudawarecolor",   palette_key="dark_teal"),
    MetricSpec("options",         "score_options",         "Options",
               color="optionscolor",      palette_key="yellow"),
]

_NEEDED_COLUMNS = [
    "model",
    "setting",
    "run_category",
    "has_sampling_error",
    "has_error",
    *INSTRUCTIONAL_CONDITIONS,
    *CONTEXTUAL_CONDITIONS,
    *(spec.column for spec in METRICS),
    "fraud_score_final",
    "refusal_score_final",
    "option_score_final",
    "case_parties_type",
    "is_concealment",
    "is_fraud_first",
    "high_temperature",
    "high_reasoning",
    "post_cutoff_case",
]


# ---------------------------------------------------------------------------
# Cell aggregate
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    """One ``(setting, instruction, context)`` cell of metrics for a model."""

    setting: str
    instruction: str
    context: str
    n_expected: int = 0
    n_effective: int = 0
    metric_rates: dict[str, float] = field(default_factory=dict)
    metric_n: dict[str, int] = field(default_factory=dict)
    metric_successes: dict[str, int] = field(default_factory=dict)
    metric_ci_low: dict[str, Optional[float]] = field(default_factory=dict)
    metric_ci_high: dict[str, Optional[float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data loading and normalisation
# ---------------------------------------------------------------------------

def load_data(input_path: str) -> pd.DataFrame:
    """Load a parquet/csv/jsonl observation file."""
    return load_data_from_source(input_path, needed_columns=_NEEDED_COLUMNS)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise score column names so downstream code is parquet-vintage agnostic.

    Older runs of ``analysis/process.py`` wrote ``fraud_score_final`` /
    ``refusal_score_final``; the current version writes the renamed
    ``score_fraud`` / ``score_refusal`` (commit 92fe3b2). We accept either and
    canonicalise on the new names. ``score_options`` is left missing if absent
    -- :func:`_metric_series` then returns an all-NaN series so the table
    renders blank Options cells.
    """
    rename = {}
    if "score_fraud" not in df.columns and "fraud_score_final" in df.columns:
        rename["fraud_score_final"] = "score_fraud"
    if "score_refusal" not in df.columns and "refusal_score_final" in df.columns:
        rename["refusal_score_final"] = "score_refusal"
    if "score_options" not in df.columns and "option_score_final" in df.columns:
        rename["option_score_final"] = "score_options"
    if rename:
        df = df.rename(columns=rename)
    return df


def _add_condition_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``instruction`` / ``context`` columns from the boolean flags.

    ``add_condition_columns`` uses ``'none'`` for the no-contextual-flag cell;
    the spec / display order calls that cell ``'control'``, so we rename for
    consistency. Rows that fall outside the documented 5x5 design are
    filtered out.
    """
    df = add_condition_columns(df, include_label=False)
    df = df.rename(columns={
        "instructional_group":  "instruction",
        "contextual_condition": "context",
    })
    df["context"] = df["context"].replace({"none": "control"})
    df = df[df["instruction"].isin(INSTRUCTION_ORDER)]
    df = df[df["context"].isin(CONTEXT_ORDER)]
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise columns, retain main AI evaluations, and derive condition keys."""
    df = _normalize_columns(df)
    df = filter_main_runs(df)
    df = df.loc[df["setting"].isin(SETTING_ORDER)].copy()
    return _add_condition_keys(df)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _metric_series(group: pd.DataFrame, column: str) -> pd.Series:
    """Return the metric column for *group* as a float series (NaN-safe).

    If *column* is missing entirely we return an all-NaN series so callers
    treat the metric as unobserved. Boolean integer columns are coerced to
    float to keep ``mean()``/``isna()`` semantics consistent with the
    grader-vote columns.
    """
    if column not in group.columns:
        return pd.Series([np.nan] * len(group), index=group.index, dtype="float64")
    s = group[column]
    return s.astype("float64")


def _expected_per_cell(
    df: pd.DataFrame,
    *,
    fallback: int = DEFAULT_N_EXPECTED,
) -> dict[tuple[str, str, str], int]:
    """Compute ``n_expected`` for every ``(setting, instruction, context)``.

    Defined as the *maximum* total sample count observed across all models in
    *df* for that cell. Cells with zero observations fall back to
    *fallback* (default :data:`DEFAULT_N_EXPECTED`); pass ``fallback=0`` when
    unobserved cells truly were not part of the sampled design.
    """
    if df.empty:
        return {
            (s, i, c): fallback
            for s in SETTING_ORDER for i in INSTRUCTION_ORDER for c in CONTEXT_ORDER
        }
    counts = (
        df.groupby(["setting", "instruction", "context", "model_id"], observed=True)
        .size()
        .groupby(level=[0, 1, 2])
        .max()
    )
    out: dict[tuple[str, str, str], int] = {}
    for setting in SETTING_ORDER:
        for instr in INSTRUCTION_ORDER:
            for ctx in CONTEXT_ORDER:
                key = (setting, instr, ctx)
                n = int(counts.get(key, 0)) if key in counts.index else 0
                out[key] = n if n > 0 else fallback
    return out


def compute_cells_for_model(
    model_df: pd.DataFrame,
    *,
    expected_per_cell: dict[tuple[str, str, str], int],
) -> dict[tuple[str, str, str], Cell]:
    """Aggregate one model's data into per-cell ``Cell`` records.

    For each cell we report:
      - ``n_eff``: count of valid scored rows (``has_sampling_error == False``
        AND the fraud-score column is non-null). Fraud is the canonical metric
        for validity since it is what the rest of the analysis pipeline filters
        on.
      - ``n_expected``: from *expected_per_cell* (cross-model max).
      - ``metric_rates``/``metric_n``: per-metric mean over rows where that
        specific metric is non-null. Different metrics may have slightly
        different valid-row counts (e.g. ``score_options`` is missing
        wholesale).
      - ``metric_ci_low``/``metric_ci_high``: Wilson 95% binomial CI bounds
        attached to each metric.

    Rows flagged ``has_sampling_error == True`` are excluded entirely from
    the metric averages and from ``n_eff``; including them would artificially
    inflate denominators with samples where the model never produced a
    scoreable response. ``has_error`` is accepted as a legacy synonym for
    backwards compat with older parquet files.
    """
    cells: dict[tuple[str, str, str], Cell] = {}
    if "has_sampling_error" in model_df.columns:
        clean = model_df[model_df["has_sampling_error"] == False].copy()  # noqa: E712
    elif "has_error" in model_df.columns:
        clean = model_df[model_df["has_error"] == False].copy()  # noqa: E712
    else:
        clean = model_df.copy()

    for setting in SETTING_ORDER:
        for instr in INSTRUCTION_ORDER:
            for ctx in CONTEXT_ORDER:
                key = (setting, instr, ctx)
                cell = Cell(
                    setting=setting,
                    instruction=instr,
                    context=ctx,
                    n_expected=expected_per_cell.get(key, DEFAULT_N_EXPECTED),
                )
                cells[key] = cell

                sub = clean[
                    (clean["setting"] == setting)
                    & (clean["instruction"] == instr)
                    & (clean["context"] == ctx)
                ]
                if sub.empty:
                    continue

                fraud_series = _metric_series(sub, "score_fraud")
                cell.n_effective = int(fraud_series.notna().sum())

                for spec in METRICS:
                    series = _metric_series(sub, spec.column)
                    valid = series.dropna()
                    n_valid = int(valid.size)
                    cell.metric_n[spec.key] = n_valid
                    if n_valid > 0:
                        cell.metric_rates[spec.key] = float(valid.mean())
                        cell.metric_successes[spec.key] = int(round(float(valid.sum())))
                    else:
                        cell.metric_rates[spec.key] = float("nan")
                        cell.metric_successes[spec.key] = 0
                    cell.metric_ci_low[spec.key] = None
                    cell.metric_ci_high[spec.key] = None
    _populate_intervals(cells)
    return cells


def _populate_intervals(cells: dict[tuple[str, str, str], "Cell"]) -> None:
    """Fill in Wilson 95% binomial CIs on each cell's metric rates."""
    for cell in cells.values():
        for spec in METRICS:
            n = cell.metric_n.get(spec.key, 0)
            if n <= 0:
                continue
            succ = cell.metric_successes.get(spec.key, 0)
            iv = wilson_interval(succ, n)
            cell.metric_ci_low[spec.key] = iv.hdi_low
            cell.metric_ci_high[spec.key] = iv.hdi_high


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------

def _slugify_model(model_id: str) -> str:
    """Stable filesystem-safe slug for *model_id* (used as filename)."""
    safe = (
        model_id.replace("/", "_")
        .replace(":", "_")
        .replace(".", "_")
        .replace("@", "_")
    )
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in safe)


def _format_n(cell: Cell) -> str:
    """Format ``n_eff/n_expected`` for the ``N`` column.

    A cell with both ``n_effective == 0`` and ``n_expected == 0`` is treated
    as out-of-design and rendered as ``---``.
    """
    n_eff = cell.n_effective
    n_exp = max(cell.n_expected, n_eff)
    if n_eff == 0 and n_exp == 0:
        return "---"
    return f"{n_eff}/{n_exp}"


def _format_metric(
    cell: Cell,
    spec: MetricSpec,
    *,
    as_percent: bool,
    shade: bool = True,
) -> str:
    """Render one metric cell: rate on top, ``[lo, hi]`` underneath.

    When ``shade`` is true, the rate cell is shaded white -> *spec.color* via the shared
    :func:`fraudbench.analysis.style.latex_cell_color` ramp so per-model
    tables read as heatmaps consistent with the summary / awareness
    tables. ``"---"`` is returned for missing/unobserved cells (no
    shading). The bracket holds the 95% Wilson binomial CI; if it's
    unavailable (zero-row cell) we render just the rate. The CI line is
    set in ``\\tiny`` so the bracket text doesn't blow out the column
    width on the 18-metric-column sideways tables (and is still legible
    at a normal reading distance).
    """
    rate = cell.metric_rates.get(spec.key, float("nan"))
    if rate is None or (isinstance(rate, float) and np.isnan(rate)):
        return "---"
    lo = cell.metric_ci_low.get(spec.key)
    hi = cell.metric_ci_high.get(spec.key)
    if as_percent:
        rate_s = f"{rate * 100:.1f}"
        lo_s = f"{lo * 100:.1f}" if (lo is not None and not np.isnan(lo)) else None
        hi_s = f"{hi * 100:.1f}" if (hi is not None and not np.isnan(hi)) else None
    else:
        rate_s = f"{rate:.3f}"
        lo_s = f"{lo:.3f}" if (lo is not None and not np.isnan(lo)) else None
        hi_s = f"{hi:.3f}" if (hi is not None and not np.isnan(hi)) else None
    color_cmd = latex_cell_color(rate, spec.color) if shade else ""
    if lo_s is None or hi_s is None:
        return f"{color_cmd}{rate_s}"
    return f"{color_cmd}\\shortstack{{{rate_s} \\\\ \\tiny{{[{lo_s}, {hi_s}]}}}}"


def _caption_for(model_display: str) -> str:
    interval_phrase = "95\\% Wilson binomial CI"
    legend = (
        f" {REASONING_SUBSCRIPT_CAPTION}"
        if REASONING_SYMBOL in model_display
        else ""
    )
    return (
        f"Summary statistics of main results for {model_display}. Each row reflects one "
        f"\\textit{{instruction}} $\\times$ \\textit{{context}} experimental condition; "
        f"columns are grouped by task type. For each task we report the effective "
        f"sample size as $N_{{\\text{{eff}}}}/N_{{\\text{{5o5ql}}}}$. "
        f"All rates are percentages of the effective sample, with "
        f"{interval_phrase} underneath. "
        f"Generations with errors are omitted from the table; "
        f"individual cells without observations are shown as ``---''."
    )


# ---------------------------------------------------------------------------
# Active-condition selection
# ---------------------------------------------------------------------------
#
# Which settings / instructions / contexts to render is data-driven: a slot is
# kept iff at least one cell in that row/column slice produced scoreable
# observations (``n_effective > 0``).

def _active_settings(cells: dict[tuple[str, str, str], Cell]) -> list[str]:
    """Settings that have at least one cell with ``n_effective > 0``."""
    return [
        s for s in SETTING_ORDER
        if any(
            cells[(s, i, c)].n_effective > 0
            for i in INSTRUCTION_ORDER for c in CONTEXT_ORDER
        )
    ]


def _active_instructions(
    cells: dict[tuple[str, str, str], Cell],
    settings: Iterable[str],
) -> list[str]:
    settings = list(settings)
    return [
        i for i in INSTRUCTION_ORDER
        if any(
            cells[(s, i, c)].n_effective > 0
            for s in settings for c in CONTEXT_ORDER
        )
    ]


def _active_contexts(
    cells: dict[tuple[str, str, str], Cell],
    settings: Iterable[str],
    instructions: Iterable[str],
) -> list[str]:
    settings = list(settings)
    instructions = list(instructions)
    return [
        c for c in CONTEXT_ORDER
        if any(
            cells[(s, i, c)].n_effective > 0
            for s in settings for i in instructions
        )
    ]


def _use_sideways(settings: list[str], instructions: list[str]) -> bool:
    """Whether the table should be wrapped in ``sidewaystable*``.

    Full-design renders (3 settings x 5 instructions = the AI tables) carry
    18 metric columns wide and overflow a normal portrait page; rotating the
    whole table 90 deg via the ``rotating`` package's ``sidewaystable*`` is
    the agreed fix. Collapsed renders fit on a regular page and use plain
    ``table*``.
    """
    return len(settings) >= 2 or len(instructions) >= 2


def to_latex(
    model_id: str,
    model_display: str,
    cells: dict[tuple[str, str, str], Cell],
    *,
    as_percent: bool = True,
    shade_cells: bool = True,
    caption: Optional[str] = None,
    setting_display: Optional[dict[str, str]] = None,
    label_suffix: Optional[str] = None,
) -> str:
    """Render one model's per-cell aggregates to a LaTeX ``(sideways)table*`` snippet.

    Layout: rows are ``instruction x context`` (5 x 5 = 25 rows for the full
    design); columns are grouped by setting (Binary / Open / Agentic) with
    one ``N`` column plus one column per metric (6 columns per setting). The
    full-design render is wrapped in ``\\begin{sidewaystable*}`` (rotated
    90 deg via the ``rotating`` package) so all 18 metric columns fit on a
    landscape-oriented page. Models whose data only populates a subset of
    the design collapse to that subset and revert to a plain ``table*``.

    ``setting_display`` optionally replaces the display names for rendered
    setting keys without changing the underlying aggregation keys.
    """
    settings = _active_settings(cells) or list(SETTING_ORDER)
    display_names = setting_display or SETTING_DISPLAY
    instructions = _active_instructions(cells, settings) or list(INSTRUCTION_ORDER)
    contexts = _active_contexts(cells, settings, instructions) or list(CONTEXT_ORDER)

    n_metric_cols_per_setting = 1 + len(METRICS)  # N + metrics
    total_metric_cols = n_metric_cols_per_setting * len(settings)

    col_spec = (
        f">{{\\centering\\arraybackslash}}p{{{INSTRUCTION_COLUMN_WIDTH}}}"
        f">{{\\centering\\arraybackslash}}p{{{CONTEXT_COLUMN_WIDTH}}}"
        + "c" * total_metric_cols
    )

    label = "tab:per_model_" + (label_suffix or _slugify_model(model_id))
    # Filename is ``per_model_<slug>.tex`` so the label exactly matches the
    # basename (without extension) per the project-wide convention.

    sideways = _use_sideways(settings, instructions)
    table_env = "sidewaystable*" if sideways else "table*"

    T1 = "    "
    T2 = "        "
    lines: list[str] = []

    if sideways:
        # ``sidewaystable*`` does not accept a placement specifier; the
        # ``rotating`` package always typesets it on its own page.
        lines.append(f"\\begin{{{table_env}}}")
    else:
        lines.append(f"\\begin{{{table_env}}}[!htbp]")
    pkg_list = "booktabs, multirow, array"
    if shade_cells:
        pkg_list += ", colortbl, xcolor"
    if sideways:
        pkg_list += ", rotating"
    lines.append(f"{T1}% Required packages: \\usepackage{{{pkg_list}}}")
    if shade_cells:
        lines.append(f"{T1}% Heatmap colors (sourced from fraudbench.analysis.style.COLORS).")
        color_pairs = [(spec.color, spec.palette_key) for spec in METRICS]
        for color_line in latex_color_definitions(color_pairs).splitlines():
            lines.append(f"{T1}{color_line}")
    lines.append(f"{T1}\\centering")
    lines.append(f"{T1}\\scriptsize")
    # Sideways renders need slightly tighter horizontal padding so the full
    # 18-metric-column tabular fits the rotated ``\textheight`` (~650pt) on
    # NeurIPS-style appendices; upright renders use the looser default.
    tabcolsep_pt = "1.5pt" if sideways else "3pt"
    lines.append(f"{T1}\\setlength{{\\tabcolsep}}{{{tabcolsep_pt}}}")
    lines.append(f"{T1}\\begin{{tabular}}{{{col_spec}}}")
    lines.append(f"{T2}\\toprule")

    # The condition headers span both header rows, with their text anchored at
    # the bottom so it aligns with the metric-column labels above ``\midrule``.
    setting_header_parts = [
        "\\multirow[b]{2}{=}{\\textbf{Instructional\\\\Condition}}",
        "\\multirow[b]{2}{=}{\\textbf{Contextual\\\\Condition}}",
    ]
    for setting in settings:
        display = display_names[setting]
        setting_header_parts.append(
            f"\\multicolumn{{{n_metric_cols_per_setting}}}{{c}}{{\\textbf{{{display}}}}}"
        )
    lines.append(f"{T2}" + " & ".join(setting_header_parts) + " \\\\")

    # ``\cmidrule`` under each setting block.
    cmid_parts: list[str] = []
    col_cursor = 3  # first metric column index (1-indexed) after the two label cols
    for _ in settings:
        start = col_cursor
        end = col_cursor + n_metric_cols_per_setting - 1
        cmid_parts.append(f"\\cmidrule(lr){{{start}-{end}}}")
        col_cursor = end + 1
    lines.append(f"{T2}" + " ".join(cmid_parts))

    # The first two cells are occupied by the condition-header multirows.
    sub_header = ["", ""]
    for _ in settings:
        sub_header.append("$N$")
        for spec in METRICS:
            sub_header.append(spec.header)
    lines.append(f"{T2}" + " & ".join(sub_header) + " \\\\")
    lines.append(f"{T2}\\midrule")

    n_ctx = len(contexts)
    for instr_idx, instr in enumerate(instructions):
        for ctx_idx, ctx in enumerate(contexts):
            row_parts: list[str] = []

            if ctx_idx == 0:
                row_parts.append(
                    f"\\multirow{{{n_ctx}}}{{=}}{{{INSTRUCTION_LABELS[instr]}}}"
                )
            else:
                row_parts.append("")
            row_parts.append(CONTEXT_LABELS[ctx])

            for setting in settings:
                cell = cells[(setting, instr, ctx)]
                row_parts.append(_format_n(cell))
                for spec in METRICS:
                    row_parts.append(
                        _format_metric(cell, spec, as_percent=as_percent, shade=shade_cells)
                    )

            lines.append(f"{T2}" + " & ".join(row_parts) + " \\\\")

        if instr_idx < len(instructions) - 1:
            # Light divider between instruction blocks; spans every column.
            lines.append(f"{T2}\\cmidrule(lr){{1-{2 + total_metric_cols}}}")

    lines.append(f"{T2}\\bottomrule")
    lines.append(f"{T1}\\end{{tabular}}")
    uses_default_caption = caption is None
    if uses_default_caption:
        caption = _caption_for(model_display)
    if not shade_cells and uses_default_caption:
        caption = caption.replace(
            " Cells are shaded by intensity (white at 0\\%, darker at higher values), "
            "with one heatmap color per metric column.",
            "",
        )
    lines.append(f"{T1}\\caption{{{caption}}}")
    lines.append(f"{T1}\\label{{{label}}}")
    lines.append(f"\\end{{{table_env}}}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _ordered_models(df: pd.DataFrame) -> list[str]:
    """Return the list of bare model IDs to render, preserving MODEL_ORDER."""
    present = set(df["model_id"].dropna().unique())
    return [m for m in MODEL_ORDER if m in present]


def _model_display_for(model_id: str) -> str:
    name = MODEL_DISPLAY_NAMES.get(model_id, get_model_display_name(model_id))
    return reasoning_label(name, model_id)


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "paper" / "tables" / "per_model"


def run(
    input_path: str = "data/final/obs_data.parquet",
    output_dir: Optional[str] = None,
    *,
    no_save: bool = False,
    no_percent: bool = False,
    write_master: bool = True,
) -> list[str]:
    """Generate per-model appendix tables and (optionally) a master ``.tex``.

    Each rendered file is named ``per_model_<slug>.tex`` so the LaTeX
    ``\\label{tab:per_model_<slug>}`` matches the basename exactly.

    Args:
        input_path: Observation-level data (parquet/csv/jsonl).
        output_dir: Directory to write per-model artefacts. Defaults to
            ``paper/tables/per_model/``.
        no_save: Do not write LaTeX files; only print the first table to
            stdout for debugging.
        no_percent: Display proportions in [0, 1] instead of percentages.
        write_master: Also emit ``per_model_all.tex`` with one
            ``\\input{...}`` per generated table.

    Returns:
        Ordered list of ``model_id`` values for which a table was rendered.
    """
    out_dir = Path(output_dir) if output_dir else _default_output_dir()

    df = load_data(input_path)
    df = prepare(df)

    expected_per_cell = _expected_per_cell(df)

    model_ids = _ordered_models(df)
    if not model_ids:
        print("No models found in input data; nothing to render.")
        return []

    print(f"Rendering per-model tables for {len(model_ids)} model(s).")

    rendered: list[str] = []
    written_tex: list[Path] = []

    for model_id in model_ids:
        model_df = df[df["model_id"] == model_id]
        if model_df.empty:
            continue

        model_display = _model_display_for(model_id)
        cells = compute_cells_for_model(
            model_df,
            expected_per_cell=expected_per_cell,
        )
        latex = to_latex(
            model_id,
            model_display,
            cells,
            as_percent=not no_percent,
        )

        slug = _slugify_model(model_id)
        tex_path = out_dir / f"per_model_{slug}.tex"

        if no_save:
            print("\n" + "=" * 70)
            print(f"LaTeX table for {model_display} ({model_id})")
            print("=" * 70)
            print(latex)
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            tex_path.write_text(latex, encoding="utf-8")
            print(f"  Wrote {tex_path}")
            written_tex.append(tex_path)

        rendered.append(model_id)

    if write_master and not no_save and written_tex:
        master = _render_master(written_tex, out_dir)
        master_path = out_dir.parent / "per_model_all.tex"
        master_path.write_text(master, encoding="utf-8")
        print(f"  Wrote master file {master_path}")

    return rendered


def _render_master(tex_files: Iterable[Path], per_model_dir: Path) -> str:
    """Master ``\\input``-driver file: one ``\\input{...}`` per table."""
    lines: list[str] = []
    lines.append("% Per-model appendix tables. Generated by")
    lines.append("% analysis/tables/table_per_model.py -- do not edit by hand.")
    lines.append("%")
    lines.append("% Required packages (already loaded in main.tex preamble):")
    lines.append("%   \\usepackage{booktabs}")
    lines.append("%   \\usepackage{multirow}")
    lines.append("%   \\usepackage{amsmath, amssymb}  % for the math-mode metric headers")
    lines.append("%")
    lines.append("% Each \\input below pulls in a single \\begin{table*}...\\end{table*}")
    lines.append("% block that can stand alone in the appendix.")
    lines.append("")
    rel_dir = per_model_dir.name  # 'per_model'
    for path in tex_files:
        rel = f"{rel_dir}/{path.stem}"
        lines.append(f"\\input{{tables/{rel}}}")
    return "\n".join(lines) + "\n"
