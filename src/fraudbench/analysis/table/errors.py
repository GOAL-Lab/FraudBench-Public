"""LaTeX tables of generation and judge errors per model and judge.

Four tables are generated, all backed by
``data/final/obs_data_per_score.parquet`` (kept per-scorer rows) and
``data/final/obs_data_per_score_excluded.parquet`` (sample-level rows dropped
because the evaluated model failed to produce a usable response, i.e. the
legacy ``has_sampling_error == True`` flag).

* ``errors_main.tex`` -- per-model error counts on main runs
  (``run_category == 'main'``).
* ``errors_reasoning_ablation.tex`` -- per-model error counts on the
  reasoning-effort/budget ablation rows.
* ``errors_temperature_ablation.tex`` -- per-model error counts on the
  temperature ablation rows.
* ``errors_by_judge.tex`` -- judge-level breakdown of judge errors,
  segmented by run category and setting.

Each per-model table has six body columns (``Binary``, ``Open``,
``Agentic`` $\\times$ ``Generation``, ``Judge``); rows are models in
:data:`fraudbench.analysis.registry.MODEL_ORDER` with family-based midrules.
A trailing ``Total Errors`` row sums the column above it, and a
``Total Samples`` row gives the per-setting denominator (sampled rows are
counted across both parquet files, so the denominator is the design size --
the union of kept and excluded rows). Models evaluated with reasoning
enabled under the main protocol are tagged with :data:`REASONING_SYMBOL`
(subscript \\textsc{r}); the reasoning-ablation table tags every model,
because reasoning was enabled for all systems in that study.

The judge-level table groups columns by run category and setting, with one
row per judge (grader) model; cells are the count of rows in
``obs_data_per_score.parquet`` where that judge produced an unparseable score on
*any* of the fraud / refusal / options dimensions.

Generation vs judge error semantics
-----------------------------------

* **Generation error** -- the *evaluated* model failed (provider error,
  content filter, max-tokens cutoff, ``RuntimeError``, ...); the row is
  partitioned into ``obs_data_per_score_excluded.parquet``. Counted at the row
  level.
* **Judge error** -- at least one judge model returned an unparseable
  response on at least one scoring dimension; the row stays in
  ``obs_data_per_score.parquet`` (with ``has_judge_error == True`` and the
  affected ``score_X`` set to ``pd.NA``). Counted at the row level for the
  per-model tables, and at the (row, judge) level for the by-judge table.

See ``analysis/process.py`` (the legacy ``_compute_has_sampling_error`` and
``_compute_has_judge_error``) for the upstream definitions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..common import (
    ensure_model_id_column,
    filter_ai_settings,
    get_model_display_name,
    get_model_family,
    load_data_from_source,
)
from ..judges import (
    GRADER_RAN_COLUMN,
    METRICS,
    OBS_COLUMN,
    SAMPLE_ID_COLUMN,
    SCORER_COLUMN,
    collapse_per_score_rows,
    votes_columns,
)
from ..registry import (
    GRADER_REASONING_MODELS,
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
    REASONING_MODELS,
    reasoning_label,
    REASONING_SUBSCRIPT_CAPTION,
    REASONING_SYMBOL,
)


# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------

#: AI evaluation settings rendered as columns, in display order. The
#: human-baseline rows live under ``setting == 'human_baseline'`` and are
#: intentionally excluded -- only AI runs can produce generation/judge errors
#: that this table is interested in.
SETTING_ORDER: list[str] = ["closed", "open", "agentic"]

SETTING_DISPLAY: dict[str, str] = {
    "closed":  "Binary",
    "open":    "Open",
    "agentic": "Agentic",
}

#: Per-model tables are rendered once per run category, in this order.
PER_MODEL_RUN_CATEGORIES: list[tuple[str, str, str]] = [
    ("main",                  "main",                  "Main"),
    ("reasoning_ablation",    "reasoning_ablation",    "Reasoning Ablation"),
    ("temperature_ablation",  "temperature_ablation",  "Temperature Ablation"),
]

#: Run-category column groups for the by-judge table.
BY_JUDGE_RUN_CATEGORIES: list[tuple[str, str]] = [
    ("main",                 "Main"),
    ("reasoning_ablation",   "Reasoning Ablation"),
    ("temperature_ablation", "Temperature Ablation"),
]

#: Error-type sub-columns shown under each setting in the per-model tables.
@dataclass(frozen=True)
class ErrorKind:
    """One sub-column under a setting block (generation vs. judge errors)."""

    key: str
    header: str


ERROR_KINDS: list[ErrorKind] = [
    ErrorKind("generation", "Generation"),
    ErrorKind("judge",    "Judge"),
]

_NEEDED_COLUMNS = [
    "model",
    "setting",
    "run_category",
    "has_judge_error",
    "has_sampling_error",
    SAMPLE_ID_COLUMN,
    SCORER_COLUMN,
    GRADER_RAN_COLUMN,
    *votes_columns(METRICS),
    *(OBS_COLUMN[metric] for metric in METRICS),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_any(path: str) -> pd.DataFrame:
    return load_data_from_source(path, needed_columns=_NEEDED_COLUMNS)


def load_data(input_path: str, excluded_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the kept/excluded observation parquets and tag each row.

    Returns ``(kept, excluded)``: both frames carry a normalised ``model_id``
    column, the original ``setting`` column, and a ``run_category`` column
    coerced to string (``''`` -> ``'main'`` for the kept frame, where older
    pipeline versions wrote an empty value for main runs).
    """
    kept = collapse_per_score_rows(
        ensure_model_id_column(_read_any(input_path)),
        METRICS,
    )
    excluded = collapse_per_score_rows(
        ensure_model_id_column(_read_any(excluded_path)),
        METRICS,
    )

    for df, label in ((kept, input_path), (excluded, excluded_path)):
        if "run_category" not in df.columns:
            df["run_category"] = "main"
        else:
            df["run_category"] = df["run_category"].fillna("").astype(str)
            df.loc[df["run_category"] == "", "run_category"] = "main"
        if "setting" not in df.columns:
            raise ValueError(f"`setting` column missing from {label}")
        if "has_judge_error" not in df.columns:
            df["has_judge_error"] = False
        if "has_sampling_error" not in df.columns:
            df["has_sampling_error"] = False

    print(f"Loaded {len(kept)} kept rows from {input_path}")
    print(f"Loaded {len(excluded)} excluded rows from {excluded_path}")
    return kept, excluded


def _filter_ai_settings(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to AI settings (drop human-baseline rows). Thin wrapper over
    :func:`fraudbench.analysis.common.filter_ai_settings` that suppresses the
    per-call print so the per-model error tables stay quiet."""
    return filter_ai_settings(df, verbose=False)


# ---------------------------------------------------------------------------
# Aggregation -- per-model tables
# ---------------------------------------------------------------------------

@dataclass
class PerModelCell:
    """Counts for one (model, run_category, setting) bucket."""

    n_samples: int = 0
    n_generation_errors: int = 0
    n_judge_errors: int = 0


def _empty_cells_for_models(
    model_ids: Iterable[str],
) -> dict[str, dict[str, PerModelCell]]:
    """Pre-seed ``{model_id: {setting: PerModelCell()}}`` for stable iteration."""
    return {
        m: {s: PerModelCell() for s in SETTING_ORDER}
        for m in model_ids
    }


def compute_per_model_counts(
    kept: pd.DataFrame,
    excluded: pd.DataFrame,
    run_category: str,
) -> dict[str, dict[str, PerModelCell]]:
    """Aggregate generation/judge errors per ``(model_id, setting)`` for *run_category*.

    ``n_samples`` is the union of kept and excluded rows for the cell -- i.e.
    the actual design size that was attempted. ``n_generation_errors`` is
    sourced from the excluded frame, ``n_judge_errors`` from the kept frame
    (generation errors happen *before* judges run, so the latter is always
    over the kept-rows subset).
    """
    kept_rc = kept[kept["run_category"] == run_category]
    exc_rc = excluded[excluded["run_category"] == run_category]
    kept_rc = _filter_ai_settings(kept_rc)
    exc_rc = _filter_ai_settings(exc_rc)

    model_ids = sorted(set(kept_rc["model_id"]).union(exc_rc["model_id"]))
    cells = _empty_cells_for_models(model_ids)

    if not kept_rc.empty:
        kept_n = (
            kept_rc.groupby(["model_id", "setting"], observed=True)
            .size()
            .to_dict()
        )
        for (m, s), n in kept_n.items():
            cells[m][s].n_samples += int(n)

        kept_je = (
            kept_rc.groupby(["model_id", "setting"], observed=True)["has_judge_error"]
            .sum()
            .to_dict()
        )
        for (m, s), n in kept_je.items():
            cells[m][s].n_judge_errors += int(n)

    if not exc_rc.empty:
        exc_n = (
            exc_rc.groupby(["model_id", "setting"], observed=True)
            .size()
            .to_dict()
        )
        for (m, s), n in exc_n.items():
            cells[m][s].n_samples += int(n)
            cells[m][s].n_generation_errors += int(n)

    return cells


def _ordered_models(model_ids: Iterable[str]) -> list[str]:
    """Sort by :data:`MODEL_ORDER`; unknown ids appended in alpha order."""
    present = set(model_ids)
    ordered = [m for m in MODEL_ORDER if m in present]
    extras = sorted(m for m in present if m not in MODEL_ORDER and m != "human")
    return ordered + extras


def _model_display(model_id: str, *, all_reasoning_subscript: bool = False) -> str:
    name = MODEL_DISPLAY_NAMES.get(model_id, get_model_display_name(model_id))
    if all_reasoning_subscript:
        return name + " " + REASONING_SYMBOL
    return reasoning_label(name, model_id)


def _is_last_in_family(model: str, ordered: list[str]) -> bool:
    family = get_model_family(model)
    idx = ordered.index(model)
    if idx < len(ordered) - 1:
        return get_model_family(ordered[idx + 1]) != family
    return False


# ---------------------------------------------------------------------------
# Aggregation -- by-judge table
# ---------------------------------------------------------------------------

def _parse_scores(x) -> dict:
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        if not x or x == "{}":
            return {}
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return {}
    return {}


def _judge_models_in(df: pd.DataFrame) -> list[str]:
    """Discover the set of judge (grader) model ids referenced by *df*.

    Walks the three score columns to collect every judge id seen, then
    sorts the result by :data:`MODEL_ORDER` so the column order matches
    the rest of the analysis tables. Unknown judges (not present in
    ``models.jsonl``) are appended in alphabetical order.
    """
    seen: set[str] = set()
    cols = [c for c in ("score_fraud_all", "score_refusal_all", "score_options_all") if c in df.columns]
    if not cols:
        return []
    for col in cols:
        for raw in df[col]:
            d = _parse_scores(raw)
            seen.update(d.keys())
        if len(seen) >= 8:
            # Judge sets are tiny and consistent across rows; bail once
            # we've seen enough distinct ids to populate the table.
            break
    ordered = [m for m in MODEL_ORDER if m in seen]
    extras = sorted(j for j in seen if j not in MODEL_ORDER)
    return ordered + extras


@dataclass
class JudgeCell:
    """Per-judge counts for one (run_category, setting) bucket."""

    n_errors: int = 0
    n_samples: int = 0


def compute_by_judge_counts(
    kept: pd.DataFrame,
) -> tuple[
    list[str],
    dict[tuple[str, str, str], dict[str, JudgeCell]],
]:
    """Per ``(judge, run_category, setting)`` error/sample counts.

    Returns ``(judge_ids, table)`` where ``table[(run_category, setting,
    'totals')]`` carries the per-column total samples and
    ``table[(run_category, setting, judge_id)]`` is the per-judge ``JudgeCell``.

    A row counts towards judge ``j``'s errors iff ``j`` appears in any of
    that row's ``score_fraud_all`` / ``score_refusal_all`` /
    ``score_options_all`` dicts AND its value is ``None`` (unparseable) on
    at least one of those dimensions. The denominator (``n_samples``) is
    the count of rows where judge ``j`` was actually run (typically equal
    to the row count itself, but tracked per judge for accuracy).
    """
    df = _filter_ai_settings(kept)
    judges = _judge_models_in(df)
    if not judges:
        return [], {}

    out: dict[tuple[str, str, str], dict[str, JudgeCell]] = {}
    for rc, _ in BY_JUDGE_RUN_CATEGORIES:
        for s in SETTING_ORDER:
            out[(rc, s, "_cells")] = {j: JudgeCell() for j in judges}

    rc_arr = df["run_category"].to_numpy()
    s_arr = df["setting"].to_numpy()
    fs = df["score_fraud_all"].to_numpy() if "score_fraud_all" in df.columns else np.empty(len(df))
    rs = df["score_refusal_all"].to_numpy() if "score_refusal_all" in df.columns else np.empty(len(df))
    os_ = df["score_options_all"].to_numpy() if "score_options_all" in df.columns else np.empty(len(df))

    for i in range(len(df)):
        rc = rc_arr[i]
        s = s_arr[i]
        bucket_key = (rc, s, "_cells")
        if bucket_key not in out:
            continue
        fd = _parse_scores(fs[i])
        rd = _parse_scores(rs[i])
        od = _parse_scores(os_[i])
        bucket = out[bucket_key]
        for j in judges:
            ran = (j in fd) or (j in rd) or (j in od)
            if not ran:
                continue
            cell = bucket[j]
            cell.n_samples += 1
            errored = (
                (j in fd and fd[j] is None)
                or (j in rd and rd[j] is None)
                or (j in od and od[j] is None)
            )
            if errored:
                cell.n_errors += 1

    return judges, out


# ---------------------------------------------------------------------------
# LaTeX rendering -- per-model table
# ---------------------------------------------------------------------------

T1 = "    "
T2 = "        "


def _format_count(n: int) -> str:
    return f"{n:,}" if n else "0"


def to_latex_per_model(
    cells: dict[str, dict[str, PerModelCell]],
    *,
    caption: str,
    label: str,
    all_reasoning_subscript: bool = False,
) -> str:
    """Render the per-model error counts to a LaTeX ``table*``."""
    n_settings = len(SETTING_ORDER)
    n_per_setting = len(ERROR_KINDS)
    n_data_cols = n_settings * n_per_setting

    col_spec = "l" + ("c" * n_data_cols)

    ordered = _ordered_models(cells.keys())

    lines: list[str] = []
    lines.append("\\begin{table*}[!htbp]")
    lines.append(f"{T1}\\centering")
    lines.append(f"{T1}\\small")
    lines.append(f"{T1}\\setlength{{\\tabcolsep}}{{4pt}}")
    lines.append(f"{T1}\\begin{{tabular}}{{{col_spec}}}")
    lines.append(f"{T2}\\toprule")

    setting_header = [""]
    for setting in SETTING_ORDER:
        display = SETTING_DISPLAY[setting]
        setting_header.append(
            f"\\multicolumn{{{n_per_setting}}}{{c}}{{\\textbf{{{display}}}}}"
        )
    lines.append(f"{T2}" + " & ".join(setting_header) + " \\\\")

    cmid_parts: list[str] = []
    col_cursor = 2
    for _ in SETTING_ORDER:
        start = col_cursor
        end = col_cursor + n_per_setting - 1
        cmid_parts.append(f"\\cmidrule(lr){{{start}-{end}}}")
        col_cursor = end + 1
    lines.append(f"{T2}" + " ".join(cmid_parts))

    sub_header = ["\\textbf{Model}"]
    for _ in SETTING_ORDER:
        for kind in ERROR_KINDS:
            sub_header.append(kind.header)
    lines.append(f"{T2}" + " & ".join(sub_header) + " \\\\")
    lines.append(f"{T2}\\midrule")

    for i, model_id in enumerate(ordered):
        row_parts = [_model_display(
            model_id, all_reasoning_subscript=all_reasoning_subscript,
        )]
        for setting in SETTING_ORDER:
            cell = cells[model_id][setting]
            row_parts.append(_format_count(cell.n_generation_errors))
            row_parts.append(_format_count(cell.n_judge_errors))
        lines.append(f"{T2}" + " & ".join(row_parts) + " \\\\")

        if _is_last_in_family(model_id, ordered) and i < len(ordered) - 1:
            lines.append(f"{T2}\\midrule")

    lines.append(f"{T2}\\midrule")

    total_errors = ["\\textbf{Total Errors}"]
    total_samples_per_setting: dict[str, int] = {s: 0 for s in SETTING_ORDER}
    for setting in SETTING_ORDER:
        generation = sum(cells[m][setting].n_generation_errors for m in ordered)
        judge = sum(cells[m][setting].n_judge_errors for m in ordered)
        total_errors.append(_format_count(generation))
        total_errors.append(_format_count(judge))
        total_samples_per_setting[setting] = sum(
            cells[m][setting].n_samples for m in ordered
        )
    lines.append(f"{T2}" + " & ".join(total_errors) + " \\\\")

    total_samples = ["\\textbf{Total Samples}"]
    for setting in SETTING_ORDER:
        total_samples.append(
            f"\\multicolumn{{{n_per_setting}}}{{c}}{{{_format_count(total_samples_per_setting[setting])}}}"
        )
    lines.append(f"{T2}" + " & ".join(total_samples) + " \\\\")

    lines.append(f"{T2}\\bottomrule")
    lines.append(f"{T1}\\end{{tabular}}")
    lines.append(f"{T1}\\caption{{{caption}}}")
    lines.append(f"{T1}\\label{{{label}}}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LaTeX rendering -- by-judge table
# ---------------------------------------------------------------------------

def _judge_display(judge_id: str) -> str:
    """Render a judge id with the canonical alias and the reasoning marker
    when the grader was configured with reasoning on (see
    :data:`fraudbench.analysis.registry.GRADER_REASONING_MODELS`)."""
    name = MODEL_DISPLAY_NAMES.get(judge_id, get_model_display_name(judge_id))
    if judge_id in GRADER_REASONING_MODELS:
        return name + " " + REASONING_SYMBOL
    return name


def to_latex_by_judge(
    judges: list[str],
    table: dict[tuple[str, str, str], dict[str, JudgeCell]],
    *,
    caption: str,
    label: str,
) -> str:
    """Render the by-judge errors table.

    Top header: run categories (``\\multicolumn`` blocks).
    Sub-header: settings under each run category.
    One body row per judge, then a ``Total Errors`` row and a
    ``Total Samples`` row.
    """
    n_settings = len(SETTING_ORDER)
    n_data_cols = len(BY_JUDGE_RUN_CATEGORIES) * n_settings

    col_spec = "l" + ("c" * n_data_cols)

    lines: list[str] = []
    lines.append("\\begin{table*}[!htbp]")
    lines.append(f"{T1}\\centering")
    lines.append(f"{T1}\\small")
    lines.append(f"{T1}\\setlength{{\\tabcolsep}}{{4pt}}")
    lines.append(f"{T1}\\begin{{tabular}}{{{col_spec}}}")
    lines.append(f"{T2}\\toprule")

    rc_header = [""]
    for _, rc_display in BY_JUDGE_RUN_CATEGORIES:
        rc_header.append(
            f"\\multicolumn{{{n_settings}}}{{c}}{{\\textbf{{{rc_display}}}}}"
        )
    lines.append(f"{T2}" + " & ".join(rc_header) + " \\\\")

    cmid_parts: list[str] = []
    col_cursor = 2
    for _ in BY_JUDGE_RUN_CATEGORIES:
        start = col_cursor
        end = col_cursor + n_settings - 1
        cmid_parts.append(f"\\cmidrule(lr){{{start}-{end}}}")
        col_cursor = end + 1
    lines.append(f"{T2}" + " ".join(cmid_parts))

    sub_header = ["\\textbf{Judge}"]
    for _ in BY_JUDGE_RUN_CATEGORIES:
        for setting in SETTING_ORDER:
            sub_header.append(SETTING_DISPLAY[setting])
    lines.append(f"{T2}" + " & ".join(sub_header) + " \\\\")
    lines.append(f"{T2}\\midrule")

    for j in judges:
        row_parts = [_judge_display(j)]
        for rc, _ in BY_JUDGE_RUN_CATEGORIES:
            for setting in SETTING_ORDER:
                cell = table[(rc, setting, "_cells")][j]
                row_parts.append(_format_count(cell.n_errors))
        lines.append(f"{T2}" + " & ".join(row_parts) + " \\\\")

    lines.append(f"{T2}\\midrule")

    total_errors = ["\\textbf{Total Errors}"]
    for rc, _ in BY_JUDGE_RUN_CATEGORIES:
        for setting in SETTING_ORDER:
            cells_for_col = table[(rc, setting, "_cells")]
            total_errors.append(
                _format_count(sum(c.n_errors for c in cells_for_col.values()))
            )
    lines.append(f"{T2}" + " & ".join(total_errors) + " \\\\")

    total_samples = ["\\textbf{Total Samples}"]
    for rc, _ in BY_JUDGE_RUN_CATEGORIES:
        for setting in SETTING_ORDER:
            cells_for_col = table[(rc, setting, "_cells")]
            n = max((c.n_samples for c in cells_for_col.values()), default=0)
            total_samples.append(_format_count(n))
    lines.append(f"{T2}" + " & ".join(total_samples) + " \\\\")

    lines.append(f"{T2}\\bottomrule")
    lines.append(f"{T1}\\end{{tabular}}")
    lines.append(f"{T1}\\caption{{{caption}}}")
    lines.append(f"{T1}\\label{{{label}}}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------

def _per_model_caption(rc_display: str, *, run_category: str) -> str:
    base = (
        f"Generations and judge errors per evaluated model on {rc_display.lower()} runs, segmented by task and error type. "
    )
    if run_category == "reasoning_ablation":
        legend = (
            f" {REASONING_SYMBOL} is shown for every model because reasoning "
            f"was enabled for all systems in this reasoning ablation."
        )
    else:
        legend = f" {REASONING_SUBSCRIPT_CAPTION}"
    return base + legend


_BY_JUDGE_CAPTION = (
    "Judge model errors broken out by judge model, segmented by run category and task. "
    f"{REASONING_SUBSCRIPT_CAPTION}"
)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _save_or_print(latex: str, output: Path, no_save: bool, label: str) -> None:
    if no_save:
        print(f"\n{'=' * 60}\n{label.upper()}\n{'=' * 60}\n")
        print(latex)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(latex, encoding="utf-8")
        print(f"Saved {label} to: {output}")


def _print_summary(
    label: str,
    cells: dict[str, dict[str, PerModelCell]],
) -> None:
    n_models = len(cells)
    totals_by_setting = {s: PerModelCell() for s in SETTING_ORDER}
    for model_cells in cells.values():
        for s, c in model_cells.items():
            totals_by_setting[s].n_samples += c.n_samples
            totals_by_setting[s].n_generation_errors += c.n_generation_errors
            totals_by_setting[s].n_judge_errors += c.n_judge_errors
    print(f"\n[{label}] {n_models} model(s)")
    for s in SETTING_ORDER:
        t = totals_by_setting[s]
        print(
            f"  {SETTING_DISPLAY[s]:<14} "
            f"samples={t.n_samples:>7,}  "
            f"generation_err={t.n_generation_errors:>5,}  "
            f"judge_err={t.n_judge_errors:>5,}"
        )


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "paper" / "tables" / "errors"


def run(
    input_path: str = "data/final/obs_data_per_score.parquet",
    excluded_input_path: str = "data/final/obs_data_per_score_excluded.parquet",
    output_dir: Optional[str] = None,
    *,
    no_save: bool = False,
) -> None:
    """Generate all four error tables.

    Args:
        input_path: Path to the kept observations
            (``has_sampling_error == False``).
        excluded_input_path: Path to the excluded observations
            (``has_sampling_error == True``).
        output_dir: Directory to write ``.tex`` files to. Defaults to
            ``paper/tables/errors/``.
        no_save: If True, print each table to stdout instead of writing.
    """
    out_dir = Path(output_dir) if output_dir else _default_output_dir()

    kept, excluded = load_data(input_path, excluded_input_path)

    for rc_key, rc_slug, rc_display in PER_MODEL_RUN_CATEGORIES:
        cells = compute_per_model_counts(kept, excluded, rc_key)
        if not cells:
            print(f"No rows for run_category='{rc_key}' -- skipping.")
            continue
        _print_summary(rc_display, cells)
        latex = to_latex_per_model(
            cells,
            caption=_per_model_caption(rc_display, run_category=rc_key),
            label=f"tab:errors_{rc_slug}",
            all_reasoning_subscript=(rc_key == "reasoning_ablation"),
        )
        out_path = out_dir / f"errors_{rc_slug}.tex"
        _save_or_print(latex, out_path, no_save, f"errors table ({rc_display})")

    judges, by_judge = compute_by_judge_counts(kept)
    if not judges:
        print("No judge models found in score columns -- skipping by-judge table.")
        return

    print(f"\n[By Judge] {len(judges)} judge(s): {', '.join(judges)}")
    latex = to_latex_by_judge(
        judges,
        by_judge,
        caption=_BY_JUDGE_CAPTION,
        label="tab:errors_by_judge",
    )
    out_path = out_dir / "errors_by_judge.tex"
    _save_or_print(latex, out_path, no_save, "errors-by-judge table")
