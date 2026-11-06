"""LaTeX tables of source-associated LLM-as-a-judge behaviour.

Backs ``analysis/tables/table_judge_source_preference.py``.

Every kept response in ``data/final/obs_data_per_score.parquet`` is graded by
the same three-member LLM jury, and each of those three judges is also one of
the generating models. This module asks a purely descriptive question: does a
judge label responses differently when it -- or a sibling from its own family
-- produced them?

Each (completion, judge) vote is assigned to one of three mutually exclusive
source categories, keyed off the generating model of the completion:

* **Exact self** -- the judge is grading its own output.
* **Same family, excl. self** -- another model from the judge's family in
  ``data/final/models.jsonl``. Families are exactly the registry's, so e.g.
  GPT-OSS is a family of its own rather than part of GPT.
* **Different family** -- everything else.

Three tables are emitted:

* ``judge_source_preference.tex`` -- per judge and source category, the raw
  positive-label rate plus the same judge's vote relative to the mean vote of
  the other two judges on the *same* completion. Because judges differ in
  baseline strictness, the last two columns net out each judge's
  different-family differential.
* ``judge_source_dissent.tex`` -- the leave-one-judge-out view: how often each
  judge was the lone negative or the lone positive vote, by source category.
* ``judge_source_preference_by_task.tex`` -- the two excess contrasts split by
  task, since fraud and refusal rates differ sharply across tasks.

Statistics
----------
For completion $i$ and judge $j$ with vote $Y_{ij} \\in \\{0, 1\\}$, and $k$,
$\\ell$ the other two judges, the co-judge differential is

.. math::

    D_{ij} = Y_{ij} - \\tfrac{1}{2}\\,(Y_{ik} + Y_{i\\ell}),

averaged within each source category to give $\\bar{D}_{j,\\mathrm{self}}$,
$\\bar{D}_{j,\\mathrm{family}}$ and $\\bar{D}_{j,\\mathrm{other}}$. The two
reported contrasts subtract the judge's general tendency to vote positive:

.. math::

    \\mathrm{ExactSelfExcess}_j =
        \\bar{D}_{j,\\mathrm{self}} - \\bar{D}_{j,\\mathrm{other}},
    \\qquad
    \\mathrm{FamilyExcess}_j =
        \\bar{D}_{j,\\mathrm{family}} - \\bar{D}_{j,\\mathrm{other}}.

Raw rates use every parseable vote. The paired statistics use only
completions where all three judges returned a parseable vote on that metric,
so every comparison rests on the same completion and a full jury. Overall
rows macro-average across judges rather than pooling votes, so neither a
judge nor a source category with more observations can dominate.

Optional 95% intervals come from a case-clustered bootstrap: curated cases
(not votes) are resampled with replacement, each carrying all of its
responses and judge votes, which respects the repeated structure of the data
better than binomial intervals would.

Everything here is descriptive. There is no adjustment for human reference
labels or for response-level predictors, so these tables are a companion to
the regressions, not a substitute for them.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from ..common import (
    AI_SETTINGS,
    drop_errors,
    filter_ai_settings,
    filter_main_ai_runs,
    get_model_display_name,
    get_model_family,
    load_data_from_source,
)
from ..judges import (
    METRIC_DISPLAY,
    OBS_COLUMN,
    GRADER_RAN_COLUMN,
    SAMPLE_ID_COLUMN,
    SCORER_COLUMN,
    collapse_per_score_rows,
    discover_judges,
    expand_votes,
    judge_display_name,
    require_individual_vote_columns,
    votes_columns,
)
from ..registry import REASONING_SUBSCRIPT_CAPTION, SETTING_DISPLAY_NAMES
from ..uncertainty import DEFAULT_HDI_PROB


METRICS: tuple[str, ...] = ("fraud", "refusal")
"""Graded dimensions reported here.

``options`` is deliberately excluded: it records whether a response presented
multiple courses of action, which carries no self-preference reading.
"""

TABLES: tuple[str, ...] = ("main", "dissent", "by_task")
"""Table keys accepted by :func:`run`, in output order."""

RUN_CATEGORY_CHOICES: tuple[str, ...] = ("main", "all")
"""Accepted ``run_category`` policies: main runs only, or every run category."""

#: Task keys used as columns of the by-task table, in display order.
SETTING_ORDER: tuple[str, ...] = AI_SETTINGS

#: Mutually exclusive source categories, in display order. Their positions are
#: the ``source`` axis of every statistics array.
SOURCE_KEYS: tuple[str, ...] = ("self", "family", "other")
SOURCE_SELF, SOURCE_FAMILY, SOURCE_OTHER = range(len(SOURCE_KEYS))

SOURCE_DISPLAY: dict[str, str] = {
    "self": "Exact self",
    "family": "Same family, excl. self",
    "other": "Different family",
}

#: Per-case sufficient statistics accumulated by :func:`aggregate_by_case`.
#: Every reported number is a ratio of sums over these, which is what lets the
#: bootstrap resample cases by summing rows of one small array.
QUANTITY_KEYS: tuple[str, ...] = (
    "vote_sum",       # positive votes cast by the judge
    "vote_n",         # parseable votes cast by the judge
    "diff_sum",       # sum of D_ij over full-jury completions
    "jury_n",         # full-jury completions
    "lone_positive",  # judge voted positive, both co-judges negative
    "lone_negative",  # judge voted negative, both co-judges positive
)
_QUANTITY_INDEX: dict[str, int] = {key: i for i, key in enumerate(QUANTITY_KEYS)}

#: Key of the pooled "all tasks" scope in :attr:`MetricResult.scopes`.
OVERALL_SCOPE: str = "all"
OVERALL_SCOPE_DISPLAY: str = "Overall"

DEFAULT_BOOTSTRAP_RESAMPLES: int = 2000
DEFAULT_SEED: int = 0

_NEEDED_COLUMNS: tuple[str, ...] = (
    "model",
    "setting",
    "summary_id",
    "run_category",
    "has_sampling_error",
    "has_error",
    *votes_columns(METRICS),
    SAMPLE_ID_COLUMN,
    SCORER_COLUMN,
    GRADER_RAN_COLUMN,
    *(OBS_COLUMN[metric] for metric in METRICS),
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(input_path: str) -> pd.DataFrame:
    """Load observations and validate either supported individual-vote schema."""
    df = load_data_from_source(input_path, needed_columns=list(_NEEDED_COLUMNS))
    require_individual_vote_columns(df, METRICS, source=input_path)
    missing = [c for c in ("model", "setting", "summary_id") if c not in df.columns]
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {missing}.")
    return df


def prepare_data(
    df: pd.DataFrame,
    run_category: str = "main",
    *,
    settings: Sequence[str] = SETTING_ORDER,
) -> pd.DataFrame:
    """Restrict to the AI evaluation runs in scope and add the ``case_id`` column.

    The human baseline is excluded throughout: its responses have no
    generating model, so they cannot belong to any source category.
    """
    if run_category not in RUN_CATEGORY_CHOICES:
        raise ValueError(
            f"run_category must be one of {RUN_CATEGORY_CHOICES}, got {run_category!r}."
        )
    df = filter_main_ai_runs(df) if run_category == "main" else filter_ai_settings(df)
    df = drop_errors(df)
    df = df[df["setting"].isin(list(settings))].copy()
    df = collapse_per_score_rows(df, METRICS)

    if df["model_id"].isna().any() or df["model_id"].eq("").any():
        raise ValueError("Some rows have no model id; cannot assign a source category.")
    df["case_id"] = case_ids(df)
    return df


def case_ids(df: pd.DataFrame) -> pd.Series:
    """Curated-case id of every row, parsed from ``summary_id``.

    ``summary_id`` is ``{case_id}-{summary_index}`` (see ``data/README.md``),
    so several vignettes can share one underlying lawsuit. The bootstrap
    clusters on the case rather than the vignette so those correlated
    vignettes are resampled together.
    """
    parsed = df["summary_id"].astype("string").str.split("-").str[0]
    if parsed.isna().any() or parsed.eq("").any():
        raise ValueError(
            "Could not parse a case id from every summary_id; expected the "
            "'{case_id}-{summary_index}' form documented in data/README.md."
        )
    return parsed


# ---------------------------------------------------------------------------
# Per-vote matrices
# ---------------------------------------------------------------------------

def vote_matrix(
    df: pd.DataFrame, metric: str, judges: Sequence[str]
) -> np.ndarray:
    """``(n_rows, n_judges)`` float matrix of votes, ``NaN`` where unparseable."""
    votes = expand_votes(df, metric, judges)
    return np.column_stack(
        [votes[judge].to_numpy(dtype="float64", na_value=np.nan) for judge in judges]
    )


def source_matrix(model_ids: pd.Series, judges: Sequence[str]) -> np.ndarray:
    """``(n_rows, n_judges)`` source-category index for every vote."""
    codes, unique_models = pd.factorize(model_ids)
    families = [get_model_family(model) for model in unique_models]
    out = np.empty((len(model_ids), len(judges)), dtype=np.int64)
    for position, judge in enumerate(judges):
        judge_family = get_model_family(judge)
        per_model = np.array(
            [
                SOURCE_SELF
                if model == judge
                else SOURCE_FAMILY
                if family == judge_family
                else SOURCE_OTHER
                for model, family in zip(unique_models, families)
            ],
            dtype=np.int64,
        )
        out[:, position] = per_model[codes]
    return out


def aggregate_by_case(
    df: pd.DataFrame,
    metric: str,
    judges: Sequence[str],
    *,
    settings: Sequence[str] = SETTING_ORDER,
) -> np.ndarray:
    """Sufficient statistics per case, shape ``(case, judge, source, setting, quantity)``.

    Keeping the case axis separate is what makes the clustered bootstrap
    cheap: a resample is a weighted sum over the first axis.
    """
    votes = vote_matrix(df, metric, judges)
    sources = source_matrix(df["model_id"], judges)
    case_codes, unique_cases = pd.factorize(df["case_id"], sort=True)
    setting_codes = pd.Categorical(df["setting"], categories=list(settings)).codes.astype(
        np.int64
    )
    if (setting_codes < 0).any():
        unexpected = sorted(set(df.loc[setting_codes < 0, "setting"]))
        raise ValueError(f"Rows with settings outside {tuple(settings)}: {unexpected}.")

    n_judges = len(judges)
    cast = ~np.isnan(votes)
    full_jury = cast.all(axis=1)
    # Only meaningful under ``full_jury``, where no vote is NaN.
    positive_votes = np.nansum(votes, axis=1)
    co_judge_mean = (positive_votes[:, None] - votes) / (n_judges - 1)
    differential = votes - co_judge_mean
    lone_positive = full_jury[:, None] & (votes == 1) & (positive_votes[:, None] == 1)
    lone_negative = (
        full_jury[:, None] & (votes == 0) & (positive_votes[:, None] == n_judges - 1)
    )

    shape = (len(unique_cases), n_judges, len(SOURCE_KEYS), len(settings))
    totals = np.zeros((*shape, len(QUANTITY_KEYS)))
    for position in range(n_judges):
        index = (
            (case_codes * n_judges + position) * len(SOURCE_KEYS) + sources[:, position]
        ) * len(settings) + setting_codes
        contributions = {
            "vote_sum": np.where(cast[:, position], votes[:, position], 0.0),
            "vote_n": cast[:, position].astype(float),
            "diff_sum": np.where(full_jury, differential[:, position], 0.0),
            "jury_n": full_jury.astype(float),
            "lone_positive": lone_positive[:, position].astype(float),
            "lone_negative": lone_negative[:, position].astype(float),
        }
        for quantity, weights in contributions.items():
            totals[..., _QUANTITY_INDEX[quantity]] += np.bincount(
                index, weights=weights, minlength=int(np.prod(shape))
            ).reshape(shape)
    return totals


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise ratio, ``NaN`` wherever the denominator is empty."""
    out = np.full(np.broadcast(numerator, denominator).shape, np.nan)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def _nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    """``np.nanmean`` that quietly yields ``NaN`` for all-``NaN`` slices."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=axis)


def _nanpercentile(values: np.ndarray, q: float, axis: int) -> np.ndarray:
    """``np.nanpercentile`` that quietly yields ``NaN`` for all-``NaN`` slices."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(values, q, axis=axis)


@dataclass(frozen=True)
class JudgeSourceStatistics:
    """Descriptive statistics for one metric and one scope (all tasks, or one).

    Every field is an array whose trailing axes are ``(judge, source)``,
    ``(judge,)``, ``(source,)`` or scalar. Bootstrap replicates carry an extra
    leading axis, so the same code computes point estimates and replicates.
    Rates and differentials are proportions; the renderers convert to percent
    and percentage points.
    """

    rate: np.ndarray
    n_votes: np.ndarray
    differential: np.ndarray
    n_jury: np.ndarray
    lone_positive: np.ndarray
    lone_negative: np.ndarray
    net_dissent: np.ndarray
    exact_self_excess: np.ndarray
    family_excess: np.ndarray
    macro_rate: np.ndarray
    macro_n_votes: np.ndarray
    macro_differential: np.ndarray
    macro_n_jury: np.ndarray
    macro_lone_positive: np.ndarray
    macro_lone_negative: np.ndarray
    macro_net_dissent: np.ndarray
    macro_exact_self_excess: np.ndarray
    macro_family_excess: np.ndarray


def summarise(totals: np.ndarray) -> JudgeSourceStatistics:
    """Turn summed sufficient statistics into the reported statistics.

    *totals* has shape ``(..., judge, source, quantity)``; any leading axes
    are preserved. Macro-averages are unweighted means over judges, computed
    here rather than derived later so the bootstrap gives them their own
    interval.
    """
    vote_n = totals[..., _QUANTITY_INDEX["vote_n"]]
    jury_n = totals[..., _QUANTITY_INDEX["jury_n"]]
    rate = _safe_divide(totals[..., _QUANTITY_INDEX["vote_sum"]], vote_n)
    differential = _safe_divide(totals[..., _QUANTITY_INDEX["diff_sum"]], jury_n)
    lone_positive = _safe_divide(totals[..., _QUANTITY_INDEX["lone_positive"]], jury_n)
    lone_negative = _safe_divide(totals[..., _QUANTITY_INDEX["lone_negative"]], jury_n)
    net_dissent = lone_negative - lone_positive
    exact_self_excess = differential[..., SOURCE_SELF] - differential[..., SOURCE_OTHER]
    family_excess = differential[..., SOURCE_FAMILY] - differential[..., SOURCE_OTHER]
    return JudgeSourceStatistics(
        rate=rate,
        n_votes=vote_n,
        differential=differential,
        n_jury=jury_n,
        lone_positive=lone_positive,
        lone_negative=lone_negative,
        net_dissent=net_dissent,
        exact_self_excess=exact_self_excess,
        family_excess=family_excess,
        macro_rate=_nanmean(rate, axis=-2),
        macro_n_votes=vote_n.sum(axis=-2),
        macro_differential=_nanmean(differential, axis=-2),
        macro_n_jury=jury_n.sum(axis=-2),
        macro_lone_positive=_nanmean(lone_positive, axis=-2),
        macro_lone_negative=_nanmean(lone_negative, axis=-2),
        macro_net_dissent=_nanmean(net_dissent, axis=-2),
        macro_exact_self_excess=_nanmean(exact_self_excess, axis=-1),
        macro_family_excess=_nanmean(family_excess, axis=-1),
    )


def reduce_scope(totals: np.ndarray, setting_index: Optional[int]) -> np.ndarray:
    """Collapse the setting axis onto one task, or sum over all of them."""
    if setting_index is None:
        return totals.sum(axis=-2)
    return totals[..., setting_index, :]


@dataclass(frozen=True)
class BootstrapConfig:
    """Settings for the case-clustered percentile bootstrap."""

    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    seed: int = DEFAULT_SEED
    prob: float = DEFAULT_HDI_PROB


def _map_arrays(
    statistics: JudgeSourceStatistics,
    fn: Callable[[np.ndarray], np.ndarray],
) -> JudgeSourceStatistics:
    """Apply *fn* to every array field of *statistics*."""
    return JudgeSourceStatistics(
        **{field.name: fn(getattr(statistics, field.name)) for field in fields(statistics)}
    )


def bootstrap_bounds(
    case_totals: np.ndarray,
    config: BootstrapConfig,
) -> tuple[JudgeSourceStatistics, JudgeSourceStatistics]:
    """Percentile-bootstrap bounds, resampling whole cases with replacement.

    *case_totals* has shape ``(case, judge, source, quantity)``. Drawing
    multinomial case counts and contracting them against the per-case totals
    is equivalent to resampling cases and re-aggregating, but costs one small
    matrix product per bootstrap batch.

    The seed is applied per call, so every metric and scope sees the same
    case resamples (common random numbers) and their intervals stay
    comparable with one another.
    """
    n_cases = case_totals.shape[0]
    rng = np.random.default_rng(config.seed)
    counts = rng.multinomial(
        n_cases, np.full(n_cases, 1.0 / n_cases), size=config.resamples
    )
    replicate_totals = (counts @ case_totals.reshape(n_cases, -1)).reshape(
        config.resamples, *case_totals.shape[1:]
    )
    replicates = summarise(replicate_totals)
    tail = 100.0 * (1.0 - config.prob) / 2.0
    return (
        _map_arrays(replicates, lambda a: _nanpercentile(a, tail, axis=0)),
        _map_arrays(replicates, lambda a: _nanpercentile(a, 100.0 - tail, axis=0)),
    )


@dataclass(frozen=True)
class ScopeResult:
    """Point estimates for one (metric, scope), plus optional bootstrap bounds."""

    statistics: JudgeSourceStatistics
    lower: Optional[JudgeSourceStatistics] = None
    upper: Optional[JudgeSourceStatistics] = None

    def value(self, name: str, *index: int) -> float:
        """Point estimate of one element of the *name* statistic."""
        return float(getattr(self.statistics, name)[index])

    def interval(self, name: str, *index: int) -> Optional[tuple[float, float]]:
        """Bootstrap interval for one element of *name*, or ``None`` if unavailable."""
        if self.lower is None or self.upper is None:
            return None
        return (
            float(getattr(self.lower, name)[index]),
            float(getattr(self.upper, name)[index]),
        )


@dataclass(frozen=True)
class MetricResult:
    """Every scope's statistics for one graded metric."""

    metric: str
    judges: tuple[str, ...]
    scopes: dict[str, ScopeResult]
    n_responses: int
    n_full_jury: int

    @property
    def overall(self) -> ScopeResult:
        return self.scopes[OVERALL_SCOPE]


def compute_metric_result(
    df: pd.DataFrame,
    metric: str,
    judges: Sequence[str],
    *,
    settings: Sequence[str] = SETTING_ORDER,
    bootstrap: Optional[BootstrapConfig] = None,
) -> MetricResult:
    """Compute the pooled and per-task statistics for one metric."""
    case_totals = aggregate_by_case(df, metric, judges, settings=settings)
    scopes: dict[str, ScopeResult] = {}
    for scope, setting_index in [
        (OVERALL_SCOPE, None),
        *((setting, i) for i, setting in enumerate(settings)),
    ]:
        reduced = reduce_scope(case_totals, setting_index)
        statistics = summarise(reduced.sum(axis=0))
        lower = upper = None
        if bootstrap is not None:
            lower, upper = bootstrap_bounds(reduced, bootstrap)
        scopes[scope] = ScopeResult(statistics, lower, upper)

    # Every judge shares the same set of full-jury completions, so any judge's
    # row counts them once.
    overall = scopes[OVERALL_SCOPE].statistics
    return MetricResult(
        metric=metric,
        judges=tuple(judges),
        scopes=scopes,
        n_responses=len(df),
        n_full_jury=int(np.nansum(overall.n_jury[0])),
    )


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------

T1 = "    "
T2 = "        "
NA_CELL = "---"
MACRO_ROW_LABEL = "\\textit{Macro-average}"


def _panel_label(position: int) -> str:
    return f"({chr(ord('a') + position)})"


def _panel_header(title: str, n_columns: int) -> str:
    return f"{T2}\\multicolumn{{{n_columns}}}{{l}}{{\\textit{{{title}}}}} \\\\"


def _stack(lines: Sequence[str]) -> str:
    """Right-aligned stacked cell; a single line is emitted unwrapped."""
    if len(lines) == 1:
        return lines[0]
    return "\\shortstack[r]{" + " \\\\ ".join(lines) + "}"


def _format_count(value: float) -> str:
    return f"{int(round(value)):,}"


def _format_percent(value: float, *, digits: int = 1) -> str:
    if not math.isfinite(value):
        return NA_CELL
    return f"{100.0 * value:.{digits}f}\\%"


def _interval_lines(
    interval: Optional[tuple[float, float]],
    *,
    signed: bool = False,
) -> list[str]:
    """Bracketed interval line for a stacked cell, rescaled from a proportion."""
    if interval is None or not all(math.isfinite(bound) for bound in interval):
        return []
    low, high = (100.0 * bound for bound in interval)
    spec = "+.1f" if signed else ".1f"
    return [f"\\scriptsize{{[{low:{spec}}, {high:{spec}}]}}"]


def _format_share_cell(
    share: float,
    interval: Optional[tuple[float, float]] = None,
) -> str:
    """Percentage share, plus any interval underneath."""
    if not math.isfinite(share):
        return NA_CELL
    return _stack([_format_percent(share), *_interval_lines(interval)])


def _format_rate_cell(
    rate: float,
    n_obs: float,
    interval: Optional[tuple[float, float]] = None,
) -> str:
    """Positive-label rate over its supporting vote count, plus any interval."""
    if not math.isfinite(rate):
        return NA_CELL
    return _stack(
        [
            _format_percent(rate),
            *_interval_lines(interval),
            f"\\scriptsize{{(n={_format_count(n_obs)})}}",
        ]
    )


def _format_points_cell(
    value: float,
    interval: Optional[tuple[float, float]] = None,
    *,
    bold: bool = False,
) -> str:
    """Signed percentage-point value (from a proportion), plus any interval."""
    if not math.isfinite(value):
        return NA_CELL
    text = f"{100.0 * value:+.1f}"
    if bold:
        text = f"\\textbf{{{text}}}"
    return _stack([text, *_interval_lines(interval, signed=True)])


def _table_open(col_spec: str, *, tabcolsep: str = "5pt") -> list[str]:
    return [
        "\\begin{table}[!htbp]",
        f"{T1}% Required packages: \\usepackage{{booktabs}}",
        f"{T1}\\centering",
        f"{T1}\\small",
        f"{T1}\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
        f"{T1}\\renewcommand{{\\arraystretch}}{{1.2}}",
        f"{T1}\\begin{{tabular}}{{{col_spec}}}",
        f"{T2}\\toprule",
    ]


def _table_close(caption: str, label: str) -> list[str]:
    return [
        f"{T2}\\bottomrule",
        f"{T1}\\end{{tabular}}",
        f"{T1}\\caption{{{caption}}}",
        f"{T1}\\label{{{label}}}",
        "\\end{table}",
    ]


def to_latex_main(
    results: dict[str, MetricResult],
    *,
    judges: Sequence[str],
    metrics: Sequence[str],
    caption: str,
    label: str,
    scope: str = OVERALL_SCOPE,
) -> str:
    """Render the raw-rate and co-judge-differential panels for each metric."""
    n_columns = 1 + len(SOURCE_KEYS) + 2
    lines = _table_open("l" + "r" * (n_columns - 1))

    source_span = f"\\multicolumn{{{len(SOURCE_KEYS)}}}{{c}}{{\\textbf{{Response source}}}}"
    excess_span = "\\multicolumn{2}{c}{\\textbf{Excess over different family}}"
    lines.append(f"{T2}" + " & ".join(["", source_span, excess_span]) + " \\\\")
    lines.append(
        f"{T2}\\cmidrule(lr){{2-{1 + len(SOURCE_KEYS)}}} "
        f"\\cmidrule(lr){{{2 + len(SOURCE_KEYS)}-{n_columns}}}"
    )
    lines.append(
        f"{T2}"
        + " & ".join(
            [
                "\\textbf{Judge}",
                "\\textbf{Exact self}",
                "\\shortstack{\\textbf{Same family,}\\\\\\textbf{excl. self}}",
                "\\textbf{Different family}",
                "\\textbf{Exact-self}",
                "\\textbf{Family}",
            ]
        )
        + " \\\\"
    )
    lines.append(f"{T2}\\midrule")

    panel = 0
    for metric in metrics:
        result = results[metric]
        scope_result = result.scopes[scope]
        statistics = scope_result.statistics
        display = METRIC_DISPLAY[metric]

        if panel:
            lines.append(f"{T2}\\midrule")
        lines.append(
            _panel_header(
                f"{_panel_label(panel)} {display} --- raw positive-label rate",
                n_columns,
            )
        )
        for position, judge in enumerate(judges):
            row = [judge_display_name(judge)]
            row += [
                _format_rate_cell(
                    statistics.rate[position, source],
                    statistics.n_votes[position, source],
                    scope_result.interval("rate", position, source),
                )
                for source in range(len(SOURCE_KEYS))
            ]
            row += [NA_CELL, NA_CELL]
            lines.append(f"{T2}" + " & ".join(row) + " \\\\")
        lines.append(f"{T2}\\addlinespace")
        macro_row = [MACRO_ROW_LABEL]
        macro_row += [
            _format_rate_cell(
                statistics.macro_rate[source],
                statistics.macro_n_votes[source],
                scope_result.interval("macro_rate", source),
            )
            for source in range(len(SOURCE_KEYS))
        ]
        macro_row += [NA_CELL, NA_CELL]
        lines.append(f"{T2}" + " & ".join(macro_row) + " \\\\")
        panel += 1

        lines.append(f"{T2}\\midrule")
        lines.append(
            _panel_header(
                f"{_panel_label(panel)} {display} --- difference from the other "
                "two judges (pp)",
                n_columns,
            )
        )
        for position, judge in enumerate(judges):
            row = [judge_display_name(judge)]
            row += [
                _format_points_cell(
                    statistics.differential[position, source],
                    scope_result.interval("differential", position, source),
                )
                for source in range(len(SOURCE_KEYS))
            ]
            row += [
                _format_points_cell(
                    statistics.exact_self_excess[position],
                    scope_result.interval("exact_self_excess", position),
                    bold=True,
                ),
                _format_points_cell(
                    statistics.family_excess[position],
                    scope_result.interval("family_excess", position),
                    bold=True,
                ),
            ]
            lines.append(f"{T2}" + " & ".join(row) + " \\\\")
        lines.append(f"{T2}\\addlinespace")
        macro_row = [MACRO_ROW_LABEL]
        macro_row += [
            _format_points_cell(
                statistics.macro_differential[source],
                scope_result.interval("macro_differential", source),
            )
            for source in range(len(SOURCE_KEYS))
        ]
        macro_row += [
            _format_points_cell(
                statistics.macro_exact_self_excess[()],
                scope_result.interval("macro_exact_self_excess"),
                bold=True,
            ),
            _format_points_cell(
                statistics.macro_family_excess[()],
                scope_result.interval("macro_family_excess"),
                bold=True,
            ),
        ]
        lines.append(f"{T2}" + " & ".join(macro_row) + " \\\\")
        panel += 1

    lines += _table_close(caption, label)
    return "\n".join(lines)


def _dissent_rows(
    scope_result: ScopeResult,
    *,
    row_label: str,
    position: Optional[int],
) -> list[str]:
    """One source row per category, for a single judge or for the macro-average.

    ``position=None`` selects the macro-averaged ``macro_*`` statistics, whose
    arrays are indexed by source alone.
    """
    statistics = scope_result.statistics
    prefix = "macro_" if position is None else ""
    rows: list[str] = []
    for source, source_key in enumerate(SOURCE_KEYS):
        index = (source,) if position is None else (position, source)

        def statistic(name: str) -> tuple[float, Optional[tuple[float, float]]]:
            full_name = f"{prefix}{name}"
            return (
                float(getattr(statistics, full_name)[index]),
                scope_result.interval(full_name, *index),
            )

        cells = [row_label if source == 0 else "", SOURCE_DISPLAY[source_key]]
        cells += [_format_share_cell(*statistic(name))
                  for name in ("lone_negative", "lone_positive")]
        cells.append(_format_points_cell(*statistic("net_dissent"), bold=True))
        cells.append(_format_count(float(getattr(statistics, f"{prefix}n_jury")[index])))
        rows.append(f"{T2}" + " & ".join(cells) + " \\\\")
    return rows


def to_latex_dissent(
    results: dict[str, MetricResult],
    *,
    judges: Sequence[str],
    metrics: Sequence[str],
    caption: str,
    label: str,
    scope: str = OVERALL_SCOPE,
) -> str:
    """Render the leave-one-judge-out dissent panels for each metric."""
    n_columns = 6
    lines = _table_open("ll" + "r" * (n_columns - 2), tabcolsep="5pt")
    lines.append(
        f"{T2}"
        + " & ".join(
            [
                "\\textbf{Judge}",
                "\\textbf{Response source}",
                "\\shortstack{\\textbf{Lone}\\\\\\textbf{negative}}",
                "\\shortstack{\\textbf{Lone}\\\\\\textbf{positive}}",
                "\\textbf{Net (pp)}",
                "\\textbf{N}",
            ]
        )
        + " \\\\"
    )
    lines.append(f"{T2}\\midrule")

    for panel, metric in enumerate(metrics):
        scope_result = results[metric].scopes[scope]
        if panel:
            lines.append(f"{T2}\\midrule")
        lines.append(
            _panel_header(f"{_panel_label(panel)} {METRIC_DISPLAY[metric]}", n_columns)
        )

        blocks = [
            (judge_display_name(judge), position)
            for position, judge in enumerate(judges)
        ]
        blocks.append((MACRO_ROW_LABEL, None))
        for block, (row_label, position) in enumerate(blocks):
            if block:
                lines.append(f"{T2}\\addlinespace")
            lines += _dissent_rows(
                scope_result, row_label=row_label, position=position
            )

    lines += _table_close(caption, label)
    return "\n".join(lines)


def to_latex_by_task(
    results: dict[str, MetricResult],
    *,
    judges: Sequence[str],
    metrics: Sequence[str],
    settings: Sequence[str],
    caption: str,
    label: str,
) -> str:
    """Render the two excess contrasts split by task, one panel per contrast."""
    scopes = [*settings, OVERALL_SCOPE]
    n_columns = 1 + len(scopes)
    lines = _table_open("l" + "r" * len(scopes), tabcolsep="8pt")
    header = ["\\textbf{Judge}"]
    header += [
        f"\\textbf{{{SETTING_DISPLAY_NAMES.get(scope, scope)}}}" for scope in settings
    ]
    header.append(f"\\textbf{{{OVERALL_SCOPE_DISPLAY}}}")
    lines.append(f"{T2}" + " & ".join(header) + " \\\\")
    lines.append(f"{T2}\\midrule")

    contrasts = (
        ("exact_self_excess", "exact-self excess"),
        ("family_excess", "family excess"),
    )
    panel = 0
    for metric in metrics:
        result = results[metric]
        for name, contrast_display in contrasts:
            if panel:
                lines.append(f"{T2}\\midrule")
            lines.append(
                _panel_header(
                    f"{_panel_label(panel)} {METRIC_DISPLAY[metric]} --- "
                    f"{contrast_display} (pp)",
                    n_columns,
                )
            )
            for position, judge in enumerate(judges):
                row = [judge_display_name(judge)]
                for scope in scopes:
                    scope_result = result.scopes[scope]
                    row.append(
                        _format_points_cell(
                            scope_result.value(name, position),
                            scope_result.interval(name, position),
                        )
                    )
                lines.append(f"{T2}" + " & ".join(row) + " \\\\")
            lines.append(f"{T2}\\addlinespace")
            macro_row = [MACRO_ROW_LABEL]
            for scope in scopes:
                scope_result = result.scopes[scope]
                macro_row.append(
                    _format_points_cell(
                        scope_result.value(f"macro_{name}"),
                        scope_result.interval(f"macro_{name}"),
                        bold=True,
                    )
                )
            lines.append(f"{T2}" + " & ".join(macro_row) + " \\\\")
            panel += 1

    lines += _table_close(caption, label)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------

def _scope_clause(run_category: str) -> str:
    return (
        "main evaluation runs"
        if run_category == "main"
        else "all run categories (main plus the reasoning and temperature ablations)"
    )


def _interval_clause(bootstrap: Optional[BootstrapConfig]) -> str:
    if bootstrap is None:
        return ""
    return (
        f"Bracketed values are {int(round(100 * bootstrap.prob))}\\% percentile "
        f"intervals from a case-clustered bootstrap "
        f"({bootstrap.resamples:,} resamples of the curated cases, each carrying "
        "all of its responses and judge votes). "
    )


def _shared_caption_tail(
    results: dict[str, MetricResult],
    *,
    run_category: str,
    bootstrap: Optional[BootstrapConfig],
) -> str:
    full_jury = ", ".join(
        f"{METRIC_DISPLAY[metric]} {result.n_full_jury:,}"
        for metric, result in results.items()
    )
    return (
        f"{REASONING_SUBSCRIPT_CAPTION}"
    )


def _main_caption(results, *, run_category, bootstrap) -> str:
    return (
        "Summary statistics of judge scores by evaluated model/model family. (a) and (c) report the percentage of individual judge "
        "votes assigning a positive label, grouped by whether the evaluated "
        "response was generated by the exact judge model, by another model "
        "from the judge's family, or by a model from another family. "
        "(b) and (d) each judge's positive-vote rate relative to "
        "the mean vote of the other two judges on the same completions. The "
        "final two columns subtract the corresponding differential on "
        "different-family completions, accounting descriptively for each "
        "judge's general tendency to assign positive labels. For Fraud, a "
        "negative value is evidence of leniency towards its own outputs. "
        + _shared_caption_tail(results, run_category=run_category, bootstrap=bootstrap)
    )


def _dissent_caption(results, *, run_category, bootstrap) -> str:
    return (
        "Judge dissent by evaluated model. Lone negative/positive shows the share of "
        "items on which the judge voted negative/positive while both other LLM judges "
        "voted in the other direction. Net is the difference between lone negative and lone positive. "
        "For fraud, a positive value means the "
        "judge dissented towards \\emph{not} labeling the response "
        "fraudulent more often than the reverse. "
        + _shared_caption_tail(results, run_category=run_category, bootstrap=bootstrap)
    )


def _by_task_caption(results, *, run_category, bootstrap) -> str:
    return (
        "Summary statistics of judge scores by evaluated model/model family, segmented by task type."
        + _shared_caption_tail(results, run_category=run_category, bootstrap=bootstrap)
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_summary(
    result: MetricResult,
    *,
    judges: Sequence[str],
    scope: str = OVERALL_SCOPE,
) -> None:
    statistics = result.scopes[scope].statistics
    print("\n" + "=" * 96)
    print(
        f"{METRIC_DISPLAY[result.metric].upper()} -- judge positive-label rates "
        f"by response source ({result.n_responses:,} responses, "
        f"{result.n_full_jury:,} with a full jury)"
    )
    print("=" * 96)
    header = f"{'judge':<26}" + "".join(
        f"{SOURCE_DISPLAY[key]:>26}" for key in SOURCE_KEYS
    )
    print(header + f"{'self excess':>14}{'family excess':>15}")
    for position, judge in enumerate(judges):
        cells = "".join(
            f"{statistics.rate[position, source]:>14.1%} "
            f"(n={int(statistics.n_votes[position, source]):>8,})"
            for source in range(len(SOURCE_KEYS))
        )
        print(
            f"{get_model_display_name(judge)[:25]:<26}{cells}"
            f"{100 * statistics.exact_self_excess[position]:>+14.1f}"
            f"{100 * statistics.family_excess[position]:>+15.1f}"
        )
    print(
        f"{'macro-average':<26}"
        + "".join(
            f"{statistics.macro_rate[source]:>14.1%} "
            f"(n={int(statistics.macro_n_votes[source]):>8,})"
            for source in range(len(SOURCE_KEYS))
        )
        + f"{100 * statistics.macro_exact_self_excess:>+14.1f}"
        + f"{100 * statistics.macro_family_excess:>+15.1f}"
    )
    print("\nco-judge differential (pp, vs. the other two judges on the same completion)")
    for position, judge in enumerate(judges):
        cells = "".join(
            f"{100 * statistics.differential[position, source]:>+26.1f}"
            for source in range(len(SOURCE_KEYS))
        )
        print(f"{get_model_display_name(judge)[:25]:<26}{cells}")


def _save_or_print(latex: str, output: Path, *, no_save: bool, label: str) -> None:
    if no_save:
        print(f"\n{'=' * 96}\n{label.upper()}\n{'=' * 96}\n")
        print(latex)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(latex + "\n", encoding="utf-8")
        print(f"Saved {label} to: {output}")


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "paper" / "tables"


def run(
    input_path: str = "data/final/obs_data_per_score.parquet",
    output_dir: Optional[str] = None,
    *,
    metrics: Sequence[str] = METRICS,
    tables: Sequence[str] = TABLES,
    run_category: str = "main",
    bootstrap: bool = False,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_SEED,
    interval_prob: float = DEFAULT_HDI_PROB,
    no_save: bool = False,
) -> dict[str, MetricResult]:
    """Generate the source-preference LaTeX tables.

    Args:
        input_path: Path to the kept observations
            (``data/final/obs_data_per_score.parquet``). Must carry the
            per-scorer observation columns written by
            ``analysis/process.py --per-score``. The older per-sample vote
            dictionary schema is also accepted.
        output_dir: Directory to write the ``.tex`` files into. Defaults to
            ``paper/tables/``.
        metrics: Graded dimensions to report, a subset of :data:`METRICS`.
        tables: Which of :data:`TABLES` to emit.
        run_category: ``main`` (default) for the main runs only, or ``all`` to
            pool the reasoning / temperature ablations in as well. The
            ablations change the number and composition of responses per
            generating model, so the paper tables use ``main``.
        bootstrap: If True, add case-clustered bootstrap intervals.
        bootstrap_resamples: Number of bootstrap resamples.
        seed: Seed for the bootstrap resampling.
        interval_prob: Probability mass of the bootstrap interval.
        no_save: If True, print each table to stdout instead of writing it.

    Returns:
        The per-metric results, keyed by metric.
    """
    unknown_metrics = [m for m in metrics if m not in METRICS]
    if unknown_metrics:
        raise ValueError(
            f"Unknown metric(s) {unknown_metrics}; expected a subset of {METRICS}."
        )
    unknown_tables = [t for t in tables if t not in TABLES]
    if unknown_tables:
        raise ValueError(
            f"Unknown table(s) {unknown_tables}; expected a subset of {TABLES}."
        )

    out_dir = Path(output_dir) if output_dir else _default_output_dir()
    bootstrap_config = (
        BootstrapConfig(resamples=bootstrap_resamples, seed=seed, prob=interval_prob)
        if bootstrap
        else None
    )

    df = prepare_data(load_data(input_path), run_category=run_category)
    if df.empty:
        print("No rows left after filtering -- skipping tables.")
        return {}

    judges = discover_judges(df, metrics)
    if not judges:
        print("No judge models found in the vote columns -- skipping tables.")
        return {}
    print(f"Found {len(judges)} judge model(s): {', '.join(judges)}")
    settings = [s for s in SETTING_ORDER if (df["setting"] == s).any()]

    results = {
        metric: compute_metric_result(
            df, metric, judges, settings=settings, bootstrap=bootstrap_config
        )
        for metric in metrics
    }
    for metric in metrics:
        _print_summary(results[metric], judges=judges)

    renderers = {
        "main": (
            "judge_source_preference.tex",
            "judge positive-label rates and source-matched voting differentials",
            lambda: to_latex_main(
                results,
                judges=judges,
                metrics=metrics,
                caption=_main_caption(
                    results, run_category=run_category, bootstrap=bootstrap_config
                ),
                label="tab:judge_source_preference",
            ),
        ),
        "dissent": (
            "judge_source_dissent.tex",
            "leave-one-judge-out dissent by response source",
            lambda: to_latex_dissent(
                results,
                judges=judges,
                metrics=metrics,
                caption=_dissent_caption(
                    results, run_category=run_category, bootstrap=bootstrap_config
                ),
                label="tab:judge_source_dissent",
            ),
        ),
        "by_task": (
            "judge_source_preference_by_task.tex",
            "source-associated judge behaviour by task",
            lambda: to_latex_by_task(
                results,
                judges=judges,
                metrics=metrics,
                settings=settings,
                caption=_by_task_caption(
                    results, run_category=run_category, bootstrap=bootstrap_config
                ),
                label="tab:judge_source_preference_by_task",
            ),
        ),
    }
    for table in TABLES:
        if table not in tables:
            continue
        filename, label, render = renderers[table]
        _save_or_print(render(), out_dir / filename, no_save=no_save, label=label)

    return results
