"""Conditional-rate plots P(F | A_L) and P(A_L | F) split by fraud type.

Backs ``plot_proportion_aware_misrep.py``. Mirrors the family / spec
structure of :mod:`fraudbench.analysis.plot.proportion_aware`, except
each family fixes a single ``(condition_col, metric_col)`` pair on the
(fraud, legal_awareness) pair and the two bar series within a family
are the two fraud-type subsets:

* Fraudulent Misrepresentation (``is_concealment == False``)
* Fraudulent Concealment       (``is_concealment == True``)

Two families:

* ``fraud_given_legal_awareness_misrep`` -- P(F | A_L)
* ``legal_awareness_given_fraud_misrep`` -- P(A_L | F)

Each family has the same three plot variants as
:mod:`fraudbench.analysis.plot.proportion_aware`:

* ``{prefix}_by_condition_{setting}.pdf``  -- nested grouped bars over
  (instructional x contextual) experimental conditions, with one bar per
  fraud-type subset within each cell.
* ``{prefix}_by_model_{setting}.pdf``      -- one bar group per model
  (with ``--by-model``).
* ``{prefix}_compare_settings.pdf``        -- hatched bars across
  closed / open / agentic (with ``--compare-settings``).

Each bar carries a 95% Wilson binomial CI on the empirical conditional
rate (denominator: rows in the cell where ``condition_col == 1`` AND
the spec's fraud-type predicate holds).

``is_concealment`` is sourced via the same path as
:mod:`fraudbench.analysis.plot.case_attributes`: a left-join from
``data/final/dataset_public.jsonl`` on ``summary_id``.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..registry import MODEL_ORDER, SETTING_DISPLAY_NAMES

from ..style import COLORS, save_figure, setup_plot_style
from ..charts import (
    BarSeries,
    HatchedBarSeries,
    annotate_n_above_bars,
    bars_with_hatch_overlay,
    build_hatch_legend,
    grouped_bars,
    nested_grouped_bars,
)
from ..common import (
    CONTEXTUAL_CONDITIONS,
    CONTEXTUAL_LABELS,
    CONTEXTUAL_ORDER,
    DEFAULT_INPUT_PATH,
    INSTRUCTIONAL_CONDITIONS,
    INSTRUCTIONAL_LABELS,
    SETTING_HATCHES,
    SETTING_ORDER,
    VALID_SETTINGS_WITH_ALL,
    compute_conditional_rate,
    get_model_display_name,
    load_and_prepare_data,
    setting_file_suffix,
)
from ..uncertainty import wilson_halfwidths
from .case_attributes import DEFAULT_SCENARIO_PATH, merge_scenario_attributes


# ---------------------------------------------------------------------------
# Family + fraud-type spec definitions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FraudTypeSpec:
    """One bar series: P(``metric_col`` = 1 | ``condition_col`` = 1) on the
    rows where ``is_concealment == is_concealment_value``."""
    is_concealment_value: bool
    label: str
    color: str


@dataclass(frozen=True)
class MisrepConditionalFamily:
    """Family fixing the (condition_col, metric_col) pair across fraud types.

    The two specs in a family always use the same conditional rate
    direction; what differs is the per-subset fraud-type filter applied
    to the data before the rate is computed.
    """
    name: str
    title: str
    file_prefix: str
    condition_col: str
    metric_col: str
    specs: tuple[FraudTypeSpec, ...]


# Bar colors: orange / purple from the ColorBrewer PuOr ramp already in
# the paper palette (cf. CONTEXTUAL_COLORS). Both are color-blind-safe
# and are not "owned" by any of the metric / model-family axes, so they
# don't collide with the legends used in adjacent figures.
_MISREP_COLOR = COLORS["orange_dark"]
_CONCEALMENT_COLOR = COLORS["purple_dark"]


_MISREP_SPECS: tuple[FraudTypeSpec, ...] = (
    FraudTypeSpec(
        is_concealment_value=False,
        label="Fraudulent Misrepresentation",
        color=_MISREP_COLOR,
    ),
    FraudTypeSpec(
        is_concealment_value=True,
        label="Fraudulent Concealment",
        color=_CONCEALMENT_COLOR,
    ),
)


FAMILY_FRAUD_GIVEN_LEGAL = MisrepConditionalFamily(
    name="fraud_given_legal_awareness_misrep",
    title="P(Fraud | Legal Awareness) by Fraud Type",
    file_prefix="fraud_given_legal_awareness_misrep",
    condition_col="score_legal_awareness_overall",
    metric_col="score_fraud",
    specs=_MISREP_SPECS,
)

FAMILY_LEGAL_GIVEN_FRAUD = MisrepConditionalFamily(
    name="legal_awareness_given_fraud_misrep",
    title="P(Legal Awareness | Fraud) by Fraud Type",
    file_prefix="legal_awareness_given_fraud_misrep",
    condition_col="score_fraud",
    metric_col="score_legal_awareness_overall",
    specs=_MISREP_SPECS,
)


ALL_FAMILIES: tuple[MisrepConditionalFamily, ...] = (
    FAMILY_FRAUD_GIVEN_LEGAL,
    FAMILY_LEGAL_GIVEN_FRAUD,
)
FAMILIES_BY_NAME: dict[str, MisrepConditionalFamily] = {
    f.name: f for f in ALL_FAMILIES
}

_NEEDED_COLUMNS = [
    "model",
    "summary_id",
    "setting",
    "has_sampling_error",
    *INSTRUCTIONAL_CONDITIONS,
    *CONTEXTUAL_CONDITIONS,
    *dict.fromkeys(
        column
        for family in ALL_FAMILIES
        for column in (family.condition_col, family.metric_col)
    ),
]


MISREP_COLUMN = "is_concealment"


# ---------------------------------------------------------------------------
# Empirical rate + Wilson half-width
# ---------------------------------------------------------------------------
def _conditional_rate_and_halfwidth(
    subset: pd.DataFrame,
    condition_col: str,
    metric_col: str,
) -> tuple[float, float]:
    """Empirical P(metric=1 | condition=1) + matching Wilson half-width.

    Returns ``(0.0, 0.0)`` when the subset is empty or the conditioning
    variable is unobserved (zero rows where ``condition_col == 1``).
    Mirrors the helper in :mod:`fraudbench.analysis.plot.proportion_aware`
    so the two modules report identical Wilson intervals on identical
    subsets.
    """
    if (
        metric_col not in subset.columns
        or condition_col not in subset.columns
        or len(subset) == 0
    ):
        return 0.0, 0.0
    stats = compute_conditional_rate(subset, condition_col, metric_col)
    rate = float(stats["rate"].values[0])
    n = int(stats["n_fraud"].values[0])
    if n <= 0:
        return rate, 0.0
    successes = int(round(rate * n))
    hw = wilson_halfwidths(np.asarray([successes]), np.asarray([n]))[0]
    return rate, (0.0 if np.isnan(hw) else float(hw))


def _conditioning_n(
    subset: pd.DataFrame,
    condition_col: str,
    metric_col: str,
) -> int:
    """Conditioning denominator (rows where condition_col == 1)."""
    if (
        metric_col not in subset.columns
        or condition_col not in subset.columns
        or len(subset) == 0
    ):
        return 0
    stats = compute_conditional_rate(subset, condition_col, metric_col)
    return int(stats["n_fraud"].values[0])


def _columns_present(df: pd.DataFrame, family: MisrepConditionalFamily) -> bool:
    return (
        family.condition_col in df.columns
        and family.metric_col in df.columns
    )


def _spec_subset(df: pd.DataFrame, spec: FraudTypeSpec) -> pd.DataFrame:
    """Slice ``df`` to rows whose ``is_concealment`` matches the spec."""
    return df[df[MISREP_COLUMN] == spec.is_concealment_value]


# ---------------------------------------------------------------------------
# Series builders
# ---------------------------------------------------------------------------
def _build_cond_series(
    df: pd.DataFrame,
    family: MisrepConditionalFamily,
    instructional_present: list[str],
    contextual_present: list[str],
) -> tuple[list[BarSeries], list[FraudTypeSpec], list[list[int]]]:
    """Series of conditional rates per fraud-type spec, flattened (instr x ctx).

    Returns ``(series, plotted_specs, n_per_spec)`` where ``n_per_spec[i]``
    is the conditioning denominator per (instr, ctx) cell for the i-th
    spec, in the same row-major order as ``series[i].values``.
    """
    series: list[BarSeries] = []
    plotted_specs: list[FraudTypeSpec] = []
    n_per_spec: list[list[int]] = []
    for spec in family.specs:
        spec_df = _spec_subset(df, spec)
        values: list[float] = []
        errors: list[float] = []
        ns: list[int] = []
        for instr in instructional_present:
            for ctx in contextual_present:
                mask = (
                    (spec_df["instructional_group"] == instr)
                    & (spec_df["contextual_condition"] == ctx)
                )
                subset = spec_df[mask]
                rate, hw = _conditional_rate_and_halfwidth(
                    subset, family.condition_col, family.metric_col,
                )
                values.append(rate)
                errors.append(hw)
                ns.append(_conditioning_n(
                    subset, family.condition_col, family.metric_col,
                ))
        series.append(BarSeries(
            values=values,
            errors=errors,
            color=spec.color,
            label=spec.label,
        ))
        plotted_specs.append(spec)
        n_per_spec.append(ns)
    return series, plotted_specs, n_per_spec


def _build_cond_hatched_series(
    df: pd.DataFrame,
    family: MisrepConditionalFamily,
    condition_order: list[tuple[str, str]],
    settings_in_data: list[str],
) -> tuple[
    list[HatchedBarSeries],
    list[FraudTypeSpec],
    list[dict[str, list[int]]],
]:
    """HatchedBarSeries per fraud-type spec, over (instr, ctx) pairs x setting.

    Returns ``(series, plotted_specs, n_by_hatch_per_spec)`` where the
    third element parallels ``series``: each entry is
    ``{setting: [n, ...]}``, one n per (condition, setting) cell.
    """
    series: list[HatchedBarSeries] = []
    plotted_specs: list[FraudTypeSpec] = []
    n_by_hatch_per_spec: list[dict[str, list[int]]] = []
    for spec in family.specs:
        spec_df = _spec_subset(df, spec)
        by_hatch: dict = {}
        ns_by_hatch: dict[str, list[int]] = {}
        for setting in settings_in_data:
            rates: list[float] = []
            ci95s: list[float] = []
            ns: list[int] = []
            for instr, ctx in condition_order:
                mask = (
                    (spec_df["instructional_group"] == instr)
                    & (spec_df["contextual_condition"] == ctx)
                    & (spec_df["setting"] == setting)
                )
                subset = spec_df[mask]
                rate, hw = _conditional_rate_and_halfwidth(
                    subset, family.condition_col, family.metric_col,
                )
                rates.append(rate)
                ci95s.append(hw)
                ns.append(_conditioning_n(
                    subset, family.condition_col, family.metric_col,
                ))
            by_hatch[setting] = {"rates": rates, "ci95s": ci95s}
            ns_by_hatch[setting] = ns
        series.append(HatchedBarSeries(
            label=spec.label, color=spec.color, by_hatch=by_hatch,
        ))
        plotted_specs.append(spec)
        n_by_hatch_per_spec.append(ns_by_hatch)
    return series, plotted_specs, n_by_hatch_per_spec


# ---------------------------------------------------------------------------
# Plot composers
# ---------------------------------------------------------------------------
def plot_family_by_condition(
    df: pd.DataFrame,
    family: MisrepConditionalFamily,
    output_path: Optional[Path] = None,
    setting_name: str = "",  # noqa: ARG001
):
    """Grouped bars of P(A | B) split by fraud type across (instr x ctx)."""
    setup_plot_style()

    if not _columns_present(df, family):
        print(f"Warning: missing columns for family {family.name!r}, skipping")
        return None, None

    instructional_order = ["control"] + INSTRUCTIONAL_CONDITIONS
    instructional_present = [
        i for i in instructional_order if i in df["instructional_group"].unique()
    ]
    contextual_present = [
        c for c in CONTEXTUAL_ORDER if c in df["contextual_condition"].unique()
    ]
    if not instructional_present or not contextual_present:
        print("No conditions found in data")
        return None, None

    series, _plotted_specs, n_per_spec = _build_cond_series(
        df, family, instructional_present, contextual_present,
    )
    if not series:
        print(f"No specs to plot for family {family.name!r}")
        return None, None

    fig, ax = plt.subplots(figsize=(16, 7))
    result = nested_grouped_bars(
        ax,
        outer_labels=[INSTRUCTIONAL_LABELS[i] for i in instructional_present],
        inner_labels=[CONTEXTUAL_LABELS[c] for c in contextual_present],
        series=series,
        bar_width=0.12,
        group_spacing=0.4,
        inner_gap=0.05,
        outer_label_y=-0.82,
        inner_xtick_rotation=30,
        inner_xtick_ha="right",
        inner_xtick_fontsize=14,
        ylabel="Rate (Proportion)",
        ylim_cap=None,
        ylim_headroom=0.30,
    )
    ax.tick_params(axis="x", pad=12)

    # Per-bar n labels: the conditioning column is shared across the
    # two specs, but the conditioning denominator differs because the
    # per-spec data subsets differ. Annotate every bar individually.
    bar_max_per_spec = [
        [v + (e or 0.0) for v, e in zip(s.values, s.errors or [0.0] * len(s.values))]
        for s in series
    ]
    for i, _spec in enumerate(family.specs):
        annotate_n_above_bars(
            ax,
            x_positions=result["positions_by_series"][i],
            bar_heights=bar_max_per_spec[i],
            n_values=n_per_spec[i],
            rotation=90,
            pad=0.045,
            fontsize=9,
        )

    plt.subplots_adjust(bottom=0.52, top=0.92)

    if output_path:
        save_figure(plt.gcf(), output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig, ax


def plot_family_all_settings(
    df: pd.DataFrame,
    family: MisrepConditionalFamily,
    output_path: Optional[Path] = None,
    *,
    max_conditions_per_row: int = 5,
):
    """Compare all settings via hatching in instructional-condition facets.

    Each row contains contextual conditions under an instructional-condition
    title, so contextual labels are not flattened into combined labels.
    Each bar carries a vertical ``n=...``
    annotation reporting the (condition, setting, fraud-type)
    conditioning denominator.
    """
    setup_plot_style()

    if not _columns_present(df, family):
        print(f"Warning: missing columns for family {family.name!r}, skipping")
        return None, None

    settings_in_data = [s for s in SETTING_ORDER if s in df["setting"].unique()]
    if not settings_in_data:
        print("No valid settings found in data")
        return None, None

    instructional_order = ["control"] + INSTRUCTIONAL_CONDITIONS
    condition_order: list[tuple[str, str]] = []
    condition_labels: list[str] = []
    for instr in instructional_order:
        for ctx in CONTEXTUAL_ORDER:
            mask = (
                (df["instructional_group"] == instr)
                & (df["contextual_condition"] == ctx)
            )
            if mask.sum() > 0:
                condition_order.append((instr, ctx))
                condition_labels.append(CONTEXTUAL_LABELS[ctx])

    if not condition_order:
        print("No conditions found in data")
        return None, None

    series, _plotted_specs, n_by_hatch_per_spec = _build_cond_hatched_series(
        df, family, condition_order, settings_in_data,
    )
    if not series:
        print(f"No specs to plot for family {family.name!r}")
        return None, None

    n_conditions = len(condition_order)
    n_rows = max(1, int(np.ceil(n_conditions / max_conditions_per_row)))
    chunk_size = int(np.ceil(n_conditions / n_rows))

    global_max = 0.0
    for s in series:
        for hatch_level in settings_in_data:
            d = s.by_hatch.get(hatch_level, {})
            for r, e in zip(d.get("rates", []), d.get("ci95s", [])):
                global_max = max(global_max, float(r) + float(e or 0))
    global_top = min(1.0, global_max * 1.15) + 0.30

    width = max(16, chunk_size * 2.0)
    height_per_row = 4.5
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=1,
        figsize=(width, height_per_row * n_rows + 0.8),
        squeeze=False,
    )
    axes = axes[:, 0]

    for row_idx, ax in enumerate(axes):
        start = row_idx * chunk_size
        end = min(start + chunk_size, n_conditions)
        row_labels = condition_labels[start:end]
        row_condition_order = condition_order[start:end]
        row_instructions = list(dict.fromkeys(instr for instr, _ctx in row_condition_order))

        row_series: list[HatchedBarSeries] = []
        row_ns_per_spec: list[dict[str, list[int]]] = []
        for s, ns_by_hatch in zip(series, n_by_hatch_per_spec):
            sliced_by_hatch = {
                h: {
                    "rates": s.by_hatch[h]["rates"][start:end],
                    "ci95s": s.by_hatch[h]["ci95s"][start:end],
                }
                for h in settings_in_data
                if h in s.by_hatch
            }
            row_series.append(HatchedBarSeries(
                label=s.label, color=s.color, by_hatch=sliced_by_hatch,
            ))
            row_ns_per_spec.append({
                h: ns_by_hatch[h][start:end] for h in settings_in_data
            })

        bars_with_hatch_overlay(
            ax,
            x_labels=row_labels,
            series=row_series,
            hatch_order=settings_in_data,
            hatch_patterns=SETTING_HATCHES,
            hatch_display_labels=SETTING_DISPLAY_NAMES,
            bar_width=0.12,
            ylabel="Rate (Proportion)",
            xtick_rotation=30,
            xtick_ha="right",
            ylim_cap=None,
            ylim_headroom=0.30,
            show_legend=False,
        )
        ax.set_ylim(0, global_top)
        ax.set_title(
            " / ".join(INSTRUCTIONAL_LABELS[instr] for instr in row_instructions),
            pad=10,
        )

        # Vertical n labels above every bar (per-spec, per-hatch).
        n_metrics = len(row_series)
        n_hatch = len(settings_in_data)
        total_bars = n_metrics * n_hatch
        bar_width = 0.12
        x_arr = np.arange(end - start)

        for spec_idx in range(n_metrics):
            for h_idx, hatch_level in enumerate(settings_in_data):
                ns = row_ns_per_spec[spec_idx].get(
                    hatch_level, [0] * (end - start),
                )
                bar_idx = spec_idx * n_hatch + h_idx
                offset = (bar_idx - total_bars / 2 + 0.5) * bar_width
                positions = [x_arr[c] + offset for c in range(end - start)]
                d = row_series[spec_idx].by_hatch.get(hatch_level, {})
                rates = d.get("rates", [0] * (end - start))
                ci95s = d.get("ci95s", [0] * (end - start))
                heights = [
                    float(r) + float(e or 0) for r, e in zip(rates, ci95s)
                ]
                annotate_n_above_bars(
                    ax,
                    x_positions=positions,
                    bar_heights=heights,
                    n_values=ns,
                    rotation=90,
                    pad=0.045,
                    fontsize=9,
                )

    color_entries = [(s.label, s.color) for s in series]
    hatch_entries = [
        (SETTING_DISPLAY_NAMES[h], SETTING_HATCHES[h]) for h in settings_in_data
    ]
    handles, labels = build_hatch_legend(color_entries, hatch_entries)
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(labels),
        frameon=False,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))

    if output_path:
        save_figure(plt.gcf(), output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig, axes


def plot_family_by_model(
    df: pd.DataFrame,
    family: MisrepConditionalFamily,
    output_path: Optional[Path] = None,
    setting_name: str = "",  # noqa: ARG001
):
    """Per-model conditional rates split by fraud type, with conditioning n.

    The conditioning column is shared across the two specs but each
    spec's data subset (misrep vs concealment) is disjoint, so the two
    bars within a model group typically have different denominators;
    each bar is annotated individually.
    """
    setup_plot_style()

    if not _columns_present(df, family):
        print(f"Warning: missing columns for family {family.name!r}, skipping")
        return None, None

    models_in_data = df["model_id"].unique()
    ordered_models = [m for m in MODEL_ORDER if m in models_in_data]
    for m in models_in_data:
        if m not in ordered_models:
            ordered_models.append(m)
    if not ordered_models:
        print("No models found in data")
        return None, None

    metric_series: list[BarSeries] = []
    n_per_spec: list[list[int]] = []
    bar_max_per_spec: list[list[float]] = []

    for spec in family.specs:
        spec_df = _spec_subset(df, spec)
        rates: list[float] = []
        errors: list[float] = []
        ns: list[int] = []
        max_heights: list[float] = []
        for model in ordered_models:
            subset = spec_df[spec_df["model_id"] == model]
            r, hw = _conditional_rate_and_halfwidth(
                subset, family.condition_col, family.metric_col,
            )
            rates.append(r)
            errors.append(hw)
            ns.append(_conditioning_n(
                subset, family.condition_col, family.metric_col,
            ))
            max_heights.append(r + hw)
        metric_series.append(BarSeries(
            values=rates,
            errors=errors,
            color=spec.color,
            label=spec.label,
        ))
        n_per_spec.append(ns)
        bar_max_per_spec.append(max_heights)

    if not metric_series:
        print(f"No specs to plot for family {family.name!r}")
        return None, None

    fig, ax = plt.subplots(figsize=(max(14, len(ordered_models) * 0.8), 6.5))
    result = grouped_bars(
        ax,
        x_labels=[get_model_display_name(m) for m in ordered_models],
        series=metric_series,
        bar_width=0.35,
        xtick_rotation=30,
        xtick_ha="right",
        ylabel="Rate (Proportion)",
        xlabel="Model",
        ylim_cap=None,
        ylim_headroom=0.30,
    )

    for spec_idx in range(len(metric_series)):
        annotate_n_above_bars(
            ax,
            x_positions=result["positions"][spec_idx],
            bar_heights=bar_max_per_spec[spec_idx],
            n_values=n_per_spec[spec_idx],
            fontsize=9,
            rotation=90,
            pad=0.045,
        )

    plt.tight_layout()

    if output_path:
        save_figure(plt.gcf(), output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig, ax


def plot_family_overall(
    df: pd.DataFrame,
    family: MisrepConditionalFamily,
    output_path: Optional[Path] = None,
):
    """Single-figure summary: per-fraud-type bars grouped by evaluation task.

    The x-axis carries one group per scope -- ``Overall`` (pooled across
    every task) followed by the per-task buckets (``Binary`` / ``Open``
    / ``Agentic``) present in ``df`` -- and each group holds one bar per
    fraud-type spec (misrepresentation vs. concealment). Error bars are
    Wilson 95% binomial CIs; the conditioning denominator for each
    (spec, scope) cell is annotated vertically above its bar.

    The ``Overall`` bars match what the legacy single-axis figure showed
    pre-update; the per-task bars surface task-level variation that was
    previously only visible in :func:`plot_family_all_settings`.
    """
    setup_plot_style()

    if not _columns_present(df, family):
        print(f"Warning: missing columns for family {family.name!r}, skipping")
        return None, None

    settings_in_data = [s for s in SETTING_ORDER if s in df["setting"].unique()]

    # First "group" is the data-pooled overall rate (``setting=None`` is
    # a sentinel meaning "no filter"); the rest are per-setting subsets
    # in the canonical SETTING_ORDER.
    group_keys: list[Optional[str]] = [None] + settings_in_data
    group_labels: list[str] = (
        ["Overall"] + [SETTING_DISPLAY_NAMES[s] for s in settings_in_data]
    )

    series: list[BarSeries] = []
    n_per_spec: list[list[int]] = []
    for spec in family.specs:
        spec_df = _spec_subset(df, spec)
        rates: list[float] = []
        errors: list[float] = []
        ns: list[int] = []
        for group_key in group_keys:
            subset = spec_df if group_key is None else spec_df[spec_df["setting"] == group_key]
            rate, hw = _conditional_rate_and_halfwidth(
                subset, family.condition_col, family.metric_col,
            )
            rates.append(rate)
            errors.append(hw)
            ns.append(_conditioning_n(
                subset, family.condition_col, family.metric_col,
            ))
        series.append(BarSeries(
            values=rates,
            errors=errors,
            color=spec.color,
            label=spec.label,
        ))
        n_per_spec.append(ns)

    n_groups = len(group_labels)
    fig_width = max(8.0, 2.0 * n_groups + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))

    result = grouped_bars(
        ax,
        x_labels=group_labels,
        series=series,
        bar_width=0.35,
        xtick_rotation=0,
        xtick_ha="center",
        xtick_fontsize=12,
        ylabel="Rate (Proportion)",
        xlabel="Evaluation Task",
        ylim_cap=None,
        ylim_headroom=0.30,
    )

    # Per-bar conditioning denominator: the two specs have disjoint data
    # subsets (misrep vs concealment), so n differs across both spec and
    # group dimensions -- annotate every bar individually.
    for spec_idx in range(len(series)):
        s = series[spec_idx]
        bar_heights = [
            v + (e or 0.0)
            for v, e in zip(s.values, s.errors or [0.0] * len(s.values))
        ]
        annotate_n_above_bars(
            ax,
            x_positions=result["positions"][spec_idx],
            bar_heights=bar_heights,
            n_values=n_per_spec[spec_idx],
            rotation=90,
            pad=0.045,
            fontsize=9,
        )

    # Title intentionally omitted: the figure caption in the paper
    # already names the conditional rate; an axis title would collide
    # with the auto-placed legend that lives above the axes.
    plt.tight_layout()

    if output_path:
        save_figure(plt.gcf(), output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig, ax


# ---------------------------------------------------------------------------
# Console summaries
# ---------------------------------------------------------------------------
def _print_overview(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"\nTotal samples: {len(df)}")
    print(f"Settings: {sorted(df['setting'].unique().tolist())}")
    print(f"Models: {df['model_id'].nunique()}")
    if MISREP_COLUMN in df.columns:
        print(f"\n--- Sample counts by {MISREP_COLUMN} ---")
        for spec in _MISREP_SPECS:
            n = int((df[MISREP_COLUMN] == spec.is_concealment_value).sum())
            print(f"  {spec.label}: {n}")


def print_family_stats(
    df: pd.DataFrame, family: MisrepConditionalFamily,
) -> None:
    """Print the empirical conditional rates for one family, per fraud type."""
    print(f"\n--- {family.title} ---")
    if not _columns_present(df, family):
        print(f"{family.name}: n/a (missing column)")
        return
    for spec in family.specs:
        spec_df = _spec_subset(df, spec)
        cases = spec_df[spec_df[family.condition_col] == 1]
        valid = cases[family.metric_col].dropna()
        n = len(valid)
        rate = float(valid.sum() / n) if n > 0 else 0.0
        print(f"{spec.label}: {rate:.3f} (n_condition={n})")


# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------
def _prepare_data(
    setting: str,
    input_path: str,
    scenarios_path: str,
) -> pd.DataFrame:
    """Load obs_data, merge ``is_concealment`` from the scenarios file."""
    df = load_and_prepare_data(
        setting=setting,
        input_path=input_path,
        include_condition_label=False,
        valid_settings=VALID_SETTINGS_WITH_ALL,
        needed_columns=_NEEDED_COLUMNS,
    )
    df = merge_scenario_attributes(df, scenarios_path=scenarios_path)
    df = df.dropna(subset=[MISREP_COLUMN]).copy()
    df[MISREP_COLUMN] = df[MISREP_COLUMN].astype(bool)
    return df


def run(
    setting: str = "all",
    input_path: str = DEFAULT_INPUT_PATH,
    scenarios_path: str = DEFAULT_SCENARIO_PATH,
    by_model: bool = False,
    compare_settings: bool = False,
    show: bool = False,
    output_dir: Optional[Path] = None,
    *,
    families: Optional[Sequence[str]] = None,
):
    """Generate misrep-vs-concealment conditional-rate plots.

    Parameters
    ----------
    families
        Sequence of family names from :data:`FAMILIES_BY_NAME`.
        ``None`` (the default) plots both families.
    """
    if not families:
        selected = list(ALL_FAMILIES)
    else:
        unknown = [f for f in families if f not in FAMILIES_BY_NAME]
        if unknown:
            raise ValueError(
                f"Unknown family names: {unknown}. "
                f"Choose from {list(FAMILIES_BY_NAME)}."
            )
        selected = [FAMILIES_BY_NAME[f] for f in families]

    if output_dir is None:
        output_dir = Path("./paper/figures")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setting_to_load = "all" if compare_settings else setting

    df = _prepare_data(
        setting=setting_to_load,
        input_path=input_path,
        scenarios_path=scenarios_path,
    )

    _print_overview(df)
    for family in selected:
        print_family_stats(df, family)

    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    for family in selected:
        file_setting = setting_file_suffix(setting)
        if compare_settings:
            print(
                f"\n[{family.title}] All settings grouped by experimental "
                "condition..."
            )
            fig, _axes = plot_family_all_settings(
                df, family,
                output_dir / f"{family.file_prefix}_compare_settings.pdf",
            )
            if fig is not None and not show:
                plt.close(fig)
        else:
            print(
                f"\n[{family.title}] Overall (setting={setting})..."
            )
            fig, _ax = plot_family_overall(
                df, family,
                output_dir / f"{family.file_prefix}_overall_{file_setting}.pdf",
            )
            if fig is not None and not show:
                plt.close(fig)
            print(
                f"[{family.title}] By experimental condition (setting={setting})..."
            )
            fig, _ax = plot_family_by_condition(
                df, family,
                output_dir / f"{family.file_prefix}_by_condition_{file_setting}.pdf",
            )
            if fig is not None and not show:
                plt.close(fig)
            if by_model:
                print(f"[{family.title}] By model (setting={setting})...")
                fig, _ax = plot_family_by_model(
                    df, family,
                    output_dir / f"{family.file_prefix}_by_model_{file_setting}.pdf",
                )
                if fig is not None and not show:
                    plt.close(fig)

    print(f"\nDone! Plots saved to: {output_dir}")

    if show:
        plt.show()
