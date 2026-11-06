"""LaTeX summary table of evaluation scores, backing ``table_summary.py``.

Creates a table with multi-level headers (Setting × Metric), heatmap-colored
cells, and family-based midrules.

Each cell shows the **empirical proportion** for a (model, setting) pair,
heatmap-shaded by intensity.

The same machinery also produces a control-only companion table
(``control_only=True``): identical layout and metrics, but restricted to
rows in the doubly-unmanipulated cell -- no instructional condition
(``instructional_group == "control"``) and no contextual condition
(``contextual_condition == "none"``).
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..registry import (
    MODEL_ORDER,
    REASONING_SUBSCRIPT_CAPTION,
    reasoning_label,
    SETTINGS,
)
from ..common import (
    CONTEXTUAL_CONDITIONS,
    INSTRUCTIONAL_CONDITIONS,
    add_condition_columns,
    filter_main_ai_runs,
    get_model_display_name,
    get_model_family,
    load_data_from_source,
)
from ..style import latex_cell_color, latex_color_definitions


# Metric column labels: abbreviated math-mode names matching the paper.
#   R_F   = fraud rate
#   R_R   = refusal rate
#   R_{A_L} = legal awareness rate
#   R_{A_F} = fraud awareness rate
METRICS = {
    "score_fraud":           r"$\mathcal{R}_F$",
    "score_refusal":         r"$\mathcal{R}_R$",
    "score_legal_awareness_overall": r"$\mathcal{R}_{A_L}$",
    "score_fraud_awareness_overall": r"$\mathcal{R}_{A_F}$",
}

_NEEDED_COLUMNS = [
    "model",
    "setting",
    "run_category",
    "has_sampling_error",
    "has_error",
    *INSTRUCTIONAL_CONDITIONS,
    *CONTEXTUAL_CONDITIONS,
    *METRICS,
    "case_parties_type",
    "is_concealment",
    "is_fraud_first",
    "high_temperature",
    "high_reasoning",
    "post_cutoff_case",
]


# Heatmap colors per metric column (LaTeX color name -> palette key in
# fraudbench.analysis.style.COLORS). Order matches METRICS.
_METRIC_LATEX_COLORS = [
    ("fraudcolor",      "bright_red"),
    ("refusalcolor",    "blue"),
    ("legalcolor",      "dark_purple"),
    ("fraudawarecolor", "dark_teal"),
]


SETTING_DISPLAY_NAMES_LATEX = {
    "Closed": "Binary",
    "Open": "Open",
    "Agentic": "Agentic",
}

# Extra intercolumn whitespace separates setting blocks without the visual
# weight of vertical rules. It replaces the ordinary column padding at each
# block boundary, so it remains effective in both header rows and body cells.
_SETTING_BLOCK_GAP = r"@{\hspace{10pt}}"


def load_data(input_path: str) -> pd.DataFrame:
    """Load data from parquet/csv/jsonl."""
    return load_data_from_source(input_path, needed_columns=_NEEDED_COLUMNS)


def filter_control_condition(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Keep only rows in the doubly-unmanipulated control cell.

    That is, rows with no instructional condition
    (``instructional_group == "control"``) *and* no contextual condition
    (``contextual_condition == "none"``). Expects the two condition
    columns to already be present (see
    :func:`fraudbench.analysis.common.add_condition_columns`).
    """
    mask = (
        df["instructional_group"].eq("control")
        & df["contextual_condition"].eq("none")
    )
    filtered = df[mask]
    if verbose:
        print(
            f"Filtered to the control condition (no instructional, no "
            f"contextual): {len(filtered)} rows"
        )
    return filtered


def _prep(df: pd.DataFrame, *, control_only: bool = False) -> pd.DataFrame:
    df = filter_main_ai_runs(df)

    err_col = "has_sampling_error" if "has_sampling_error" in df.columns else (
        "has_error" if "has_error" in df.columns else None
    )
    if err_col is not None:
        n_before = len(df)
        df = df[df[err_col] == False]  # noqa: E712
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"Dropped {n_dropped} samples with errors")
    for metric_col in METRICS.keys():
        if metric_col in df.columns:
            df[metric_col] = df[metric_col].apply(lambda x: np.nan if x is None else x)
    # Add the instructional_group / contextual_condition columns used by
    # the control-condition filter. Safe to call repeatedly: if these
    # columns are already present add_condition_columns is a no-op rename.
    if "instructional_group" not in df.columns or "contextual_condition" not in df.columns:
        df = add_condition_columns(df, include_label=False)
    if control_only:
        df = filter_control_condition(df)
    return df


def compute_scores(
    df: pd.DataFrame,
    *,
    control_only: bool = False,
) -> pd.DataFrame:
    """Per-(model_id, setting) metric scores.

    Returns one row per ``(model_id, setting)`` with three columns per
    metric: ``<metric_col>`` (rate in [0, 1]), ``<metric_col>_n``
    (effective sample size), and ``<metric_col>_successes`` (sum of
    1-coded outcomes).

    Pass ``control_only=True`` to restrict every rate to the control
    condition (no instructional and no contextual manipulation).
    """
    df = _prep(df, control_only=control_only)

    results = []
    for (model_id, setting), group in df.groupby(["model_id", "setting"]):
        row = {"model_id": model_id, "setting": setting, "n_samples": len(group)}
        for metric_col in METRICS.keys():
            if metric_col in group.columns:
                valid = group[metric_col].dropna()
                if len(valid) > 0:
                    successes = float(valid.sum())
                    n = int(len(valid))
                    row[metric_col] = successes / n
                    row[f"{metric_col}_n"] = n
                    row[f"{metric_col}_successes"] = int(round(successes))
                else:
                    row[metric_col] = np.nan
                    row[f"{metric_col}_n"] = 0
                    row[f"{metric_col}_successes"] = 0
            else:
                row[metric_col] = np.nan
                row[f"{metric_col}_n"] = 0
                row[f"{metric_col}_successes"] = 0
        results.append(row)

    return pd.DataFrame(results)


def create_summary_table(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Build a MultiIndex-column table: rows=models, columns=(Setting, Metric).

    Cell values are 3-tuples ``(rate, successes, n)``; ``rate`` is NaN for
    empty cells.
    """
    models_present = scores_df["model_id"].unique()
    ordered_models = [m for m in MODEL_ORDER if m in models_present]
    ordered_models.extend([m for m in models_present if m not in ordered_models])

    rows = []
    for model in ordered_models:
        model_data = scores_df[scores_df["model_id"] == model]
        row = {"Model": get_model_display_name(model)}
        for setting in SETTINGS:
            setting_data = model_data[model_data["setting"] == setting]
            for metric_col, metric_name in METRICS.items():
                col_name = f"{setting}_{metric_name}"
                if len(setting_data) > 0 and metric_col in setting_data.columns:
                    rate = setting_data[metric_col].values[0]
                    n = int(setting_data.get(f"{metric_col}_n", pd.Series([0])).values[0])
                    succ = int(setting_data.get(f"{metric_col}_successes", pd.Series([0])).values[0])
                    rate_clean = rate if pd.notna(rate) else np.nan
                    cell = (rate_clean, succ, n)
                else:
                    cell = (np.nan, 0, 0)
                row[col_name] = cell
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")

    new_columns = []
    for col in df.columns:
        parts = col.split("_", 1)
        new_columns.append((parts[0].capitalize(), parts[1]))
    df.columns = pd.MultiIndex.from_tuples(new_columns, names=["Setting", "Metric"])
    return df


def compute_overall_scores(
    df: pd.DataFrame,
    *,
    include_overall: bool = False,
    control_only: bool = False,
) -> dict:
    """Overall (pooled across models) rates keyed by ``(Setting, Metric)`` tuples.

    When ``include_overall=True``, an extra group of entries keyed by
    ``("Overall", metric_name)`` is appended -- the same rates computed
    across **all** settings combined (per-task aggregation pooled across
    tasks).

    Pass ``control_only=True`` to restrict every rate to the control
    condition (no instructional and no contextual manipulation).
    """
    df = _prep(df, control_only=control_only)

    overall: dict = {}
    setting_iter: list[Optional[str]] = list(SETTINGS)
    if include_overall:
        setting_iter.append(None)
    for setting in setting_iter:
        if setting is None:
            setting_df = df
            setting_label = "Overall"
        else:
            setting_df = df[df["setting"] == setting]
            setting_label = setting.capitalize()
        for metric_col, metric_name in METRICS.items():
            if metric_col not in setting_df.columns:
                continue
            valid = setting_df[metric_col].dropna()
            n = int(len(valid))
            if n == 0:
                continue
            rate = float(valid.sum()) / n
            overall[(setting_label, metric_name)] = rate
    return overall


def _cell_color(val: float, metric_idx: int) -> str:
    """Heatmap shade for one metric cell.

    Delegates to :func:`fraudbench.analysis.style.latex_cell_color` so the
    0\\% white -> 100\\% deep-tint ramp matches every other percentage
    table in the paper (per-model, awareness, party / fraud type, etc.).
    """
    if pd.isna(val):
        return ""
    color = _METRIC_LATEX_COLORS[metric_idx % len(_METRIC_LATEX_COLORS)][0]
    return latex_cell_color(val, color)


def _format_value(val: float, metric_idx: int, as_percent: bool = True) -> str:
    if pd.isna(val):
        return "---"
    color_cmd = _cell_color(val, metric_idx)
    formatted = f"{val * 100:.1f}" if as_percent else f"{val:.3f}"
    return f"{color_cmd}{formatted}" if color_cmd else formatted


def _is_last_in_family(model: str, ordered_models: list) -> bool:
    family = get_model_family(model)
    idx = ordered_models.index(model) if model in ordered_models else -1
    if 0 <= idx < len(ordered_models) - 1:
        return get_model_family(ordered_models[idx + 1]) != family
    return False


def to_latex(
    summary_df: pd.DataFrame,
    overall_scores: Optional[dict] = None,
    as_percent: bool = True,
    *,
    label: str = "tab:summary",
    control_only: bool = False,
) -> str:
    """Build the LaTeX source for the summary table.

    Each cell is the empirical proportion for a (model, setting) pair,
    aggregated across all 25 experimental conditions (instruction
    $\\times$ context) unless ``control_only=True``.

    ``control_only=True`` only changes the caption, to state that the
    rates cover the control condition rather than all 25 experimental
    conditions; the caller is responsible for passing scores computed
    with the matching filter.
    """
    ordered_model_ids = [m for m in MODEL_ORDER if get_model_display_name(m) in summary_df.index]

    settings_present = []
    for col in summary_df.columns:
        if col[0] not in settings_present:
            settings_present.append(col[0])

    metric_names = list(METRICS.values())
    n_metrics = len(metric_names)
    n_cols = len(settings_present) * n_metrics

    T1 = "    "
    T2 = "        "
    lines = []

    lines.append("\\begin{table*}[!htbp]")
    lines.append(f"{T1}% Required packages: \\usepackage{{booktabs, colortbl, xcolor}}")
    lines.append(f"{T1}% Heatmap colors (sourced from fraudbench.analysis.style.COLORS).")
    color_defs = latex_color_definitions(_METRIC_LATEX_COLORS)
    for color_line in color_defs.splitlines():
        lines.append(f"{T1}{color_line}")
    lines.append(f"{T1}\\centering")
    lines.append(f"{T1}\\footnotesize")
    lines.append(f"{T1}\\setlength{{\\tabcolsep}}{{3pt}}")

    setting_column_specs = ["c" * n_metrics for _ in settings_present]
    col_spec = "l" + _SETTING_BLOCK_GAP.join(setting_column_specs)
    lines.append(f"{T1}\\begin{{tabular}}{{{col_spec}}}")
    lines.append(f"{T2}\\toprule")

    header1_parts = [""]
    for setting in settings_present:
        display_setting = SETTING_DISPLAY_NAMES_LATEX.get(setting, setting)
        header1_parts.append(
            f"\\multicolumn{{{n_metrics}}}{{c}}{{\\textbf{{{display_setting}}}}}")
    lines.append(f"{T2}" + " & ".join(header1_parts) + " \\\\")

    header2_parts = ["Model"]
    for _ in settings_present:
        header2_parts.extend(metric_names)
    lines.append(f"{T2}" + " & ".join(header2_parts) + " \\\\")
    lines.append(f"{T2}\\midrule")

    for model_id in ordered_model_ids:
        display_name = reasoning_label(get_model_display_name(model_id), model_id)

        row_parts = [display_name]
        for setting in settings_present:
            for metric_idx, metric_name in enumerate(metric_names):
                col = (setting, metric_name)
                if col not in summary_df.columns:
                    row_parts.append("---")
                    continue
                cell = summary_df.loc[get_model_display_name(model_id), col]
                rate, _succ, _n = cell
                row_parts.append(_format_value(
                    rate, metric_idx, as_percent=as_percent,
                ))
        lines.append(f"{T2}" + " & ".join(row_parts) + " \\\\")

        if _is_last_in_family(model_id, ordered_model_ids) and model_id != ordered_model_ids[-1]:
            lines.append(f"{T2}\\midrule")

    if overall_scores:
        lines.append(f"{T2}\\midrule")
        row_parts = ["\\textbf{Overall}"]
        for setting in settings_present:
            for metric_idx, metric_name in enumerate(metric_names):
                key = (setting, metric_name)
                val = overall_scores.get(key)
                if val is None:
                    row_parts.append("---")
                    continue
                row_parts.append(_format_value(
                    val, metric_idx, as_percent=as_percent,
                ))
        lines.append(f"{T2}" + " & ".join(row_parts) + " \\\\")

    lines.append(f"{T2}\\bottomrule")
    lines.append(f"{T1}\\end{{tabular}}")

    scope_clause = (
        " This table presents results from the control condition only, i.e., no pressure is applied through either "
        "instructional or contextual conditions."
        if control_only
        else " Results are aggregated across all 25 experimental conditions."
    )
    lines.append(
        f"{T1}\\caption{{Aggregate rates (percentages) of fraud ($\\mathcal{{R}}_F$), "
        f"refusal ($\\mathcal{{R}}_R$), legal awareness "
        f"($\\mathcal{{R}}_{{A_L}}$), and fraud awareness "
        f"($\\mathcal{{R}}_{{A_F}}$) by model and task.{scope_clause} "
        f"Only main results are included (ablations excluded). "
        f"Cells are shaded by intensity (darker = higher value). {REASONING_SUBSCRIPT_CAPTION}}}"
    )
    lines.append(f"{T1}\\label{{{label}}}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


def _print_overall_stats(overall_scores: dict) -> None:
    """Print the pooled per-setting rates to stdout."""
    for setting in list(SETTINGS) + [None]:
        label = "OVERALL (all tasks)" if setting is None else setting.upper()
        print(f"\n{label}:")
        setting_label = "Overall" if setting is None else setting.capitalize()
        for metric_col, metric_name in METRICS.items():
            key = (setting_label, metric_name)
            if key in overall_scores:
                print(f"  {metric_name}: {overall_scores[key] * 100:.1f}%")


def run(
    input_path: str = "data/final/obs_data.parquet",
    output: Optional[str] = None,
    control_output: Optional[str] = None,
    no_save: bool = False,
    no_percent: bool = False,
) -> pd.DataFrame:
    """Generate the summary table, print stats and optionally save LaTeX.

    Two LaTeX tables are produced:

    * The main table (saved to ``output``) -- one column block per
      evaluation task, pooled across all 25 experimental conditions.
    * A control-only table (saved to ``control_output``) with
      the same layout as the first, but computed from the control
      condition alone -- rows with neither an instructional condition
      nor a contextual condition applied.

    Args:
        input_path: Observation-level data (parquet/csv/jsonl).
        output: Output ``.tex`` path for the upright table. Defaults to
            ``paper/tables/summary.tex``.
        control_output: Output ``.tex`` path for the control-condition
            table. Defaults to ``paper/tables/summary_control.tex``.
            Pass an empty string to skip building it altogether.
        no_save: Skip writing; print the LaTeX outputs to stdout instead.
        no_percent: Display proportions in [0, 1] instead of percentages.
    """
    tables_dir = Path(__file__).resolve().parents[4] / "paper" / "tables"
    if output is None:
        output = str(tables_dir / "summary.tex")
    if control_output is None:
        control_output = str(tables_dir / "summary_control.tex")

    df = load_data(input_path)

    print("\nComputing scores...")
    scores_df = compute_scores(df)

    print("Creating summary table...")
    summary_df = create_summary_table(scores_df)

    overall_scores = compute_overall_scores(df, include_overall=True)

    # Skip the control pass entirely when its output would be discarded.
    build_control = bool(control_output) or no_save
    control_summary_df = None
    control_overall_scores = None
    if build_control:
        print("\nComputing control-condition scores...")
        control_scores_df = compute_scores(df, control_only=True)
        control_summary_df = create_summary_table(control_scores_df)
        control_overall_scores = compute_overall_scores(
            df, include_overall=True, control_only=True,
        )

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    _print_overall_stats(overall_scores)

    if build_control:
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS (CONTROL CONDITION ONLY)")
        print("=" * 60)
        _print_overall_stats(control_overall_scores)

    latex = to_latex(
        summary_df, overall_scores,
        as_percent=not no_percent,
        label="tab:summary",
    )
    latex_control = to_latex(
        control_summary_df, control_overall_scores,
        as_percent=not no_percent,
        label="tab:summary_control",
        control_only=True,
    ) if build_control else None

    if not no_save:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(latex)
        print(f"\nSaved LaTeX table to: {output}")
        if control_output and latex_control is not None:
            Path(control_output).parent.mkdir(parents=True, exist_ok=True)
            with open(control_output, "w") as f:
                f.write(latex_control)
            print(f"Saved control-condition LaTeX table to: {control_output}")
    else:
        print("\n" + "=" * 60)
        print("LATEX TABLE")
        print("=" * 60)
        print(latex)
        if latex_control is not None:
            print("\n" + "=" * 60)
            print("LATEX TABLE (CONTROL CONDITION ONLY)")
            print("=" * 60)
            print(latex_control)

    print("\n" + "=" * 60)
    print("RAW SCORES (proportion)")
    print("=" * 60)
    print(summary_df.to_string())

    if control_summary_df is not None:
        print("\n" + "=" * 60)
        print("RAW SCORES (proportion, control condition only)")
        print("=" * 60)
        print(control_summary_df.to_string())

    return summary_df
