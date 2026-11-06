"""Conditional-rate plots over (fraud, refusal, awareness) variables.

Backs ``plot_proportion_aware.py``. Originally just plotted
P(awareness | fraud); now generates four "families", one per direction of
the pairwise conditional probabilities between fraud / refusal and the two
awareness variables:

* ``awareness_given_fraud``    -- P(awareness | fraud)
* ``fraud_given_awareness``    -- P(fraud | awareness)
* ``refusal_given_awareness``  -- P(refusal | awareness)
* ``awareness_given_refusal``  -- P(awareness | refusal)

Each family has the same three plot variants as before:

* ``{prefix}_by_condition_{setting}.pdf``       -- nested grouped bars over
  (instructional x contextual) experimental conditions.
* ``{prefix}_by_model_{setting}.pdf``           -- one bar group per model
  (with ``--by-model``).
* ``{prefix}_compare_settings.pdf``             -- hatched bars across
  closed/open/agentic (with ``--compare-settings``).

Each bar carries a 95% Wilson binomial CI on the empirical conditional
rate (the denominator is the count of rows where the conditioning variable
equals 1 in that cell).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..registry import MODEL_ORDER, SETTING_DISPLAY_NAMES

from ..style import METRIC_COLORS, save_figure, setup_plot_style
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


# ---------------------------------------------------------------------------
# Family definitions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConditionalRateSpec:
    """One bar series: P(``metric_col`` = 1 | ``condition_col`` = 1)."""
    condition_col: str
    metric_col: str
    label: str
    color: str


@dataclass(frozen=True)
class ConditionalFamily:
    """A set of :class:`ConditionalRateSpec` plotted side-by-side per group.

    Each family produces one set of plots (by-condition / by-model /
    compare-settings) under ``{file_prefix}_*.pdf``.
    """
    name: str
    title: str
    file_prefix: str
    specs: tuple[ConditionalRateSpec, ...]


# Bar colors stay tied to the *awareness* variable across all families
# and reuse the canonical METRIC_COLORS palette so the same "Legal
# Awareness" hue used in :mod:`experimental_conditions` carries through
# to every conditional-rate panel here, regardless of whether awareness
# is the conditioning variable (families 2/3) or the conditioned-on
# outcome (families 1/4).
_LEGAL_AWARE_COLOR = METRIC_COLORS["legal_awareness"]
_FRAUD_AWARE_COLOR = METRIC_COLORS["fraud_awareness"]


FAMILY_AWARENESS_GIVEN_FRAUD = ConditionalFamily(
    name="awareness_given_fraud",
    title="P(Awareness | Fraud)",
    file_prefix="awareness_given_fraud",
    specs=(
        ConditionalRateSpec(
            condition_col="score_fraud",
            metric_col="score_legal_awareness_overall",
            label="Legal Awareness | Fraud",
            color=_LEGAL_AWARE_COLOR,
        ),
        ConditionalRateSpec(
            condition_col="score_fraud",
            metric_col="score_fraud_awareness_overall",
            label="Fraud Awareness | Fraud",
            color=_FRAUD_AWARE_COLOR,
        ),
    ),
)


FAMILY_FRAUD_GIVEN_AWARENESS = ConditionalFamily(
    name="fraud_given_awareness",
    title="P(Fraud | Awareness)",
    file_prefix="fraud_given_awareness",
    specs=(
        ConditionalRateSpec(
            condition_col="score_legal_awareness_overall",
            metric_col="score_fraud",
            label="Fraud | Legal Awareness",
            color=_LEGAL_AWARE_COLOR,
        ),
        ConditionalRateSpec(
            condition_col="score_fraud_awareness_overall",
            metric_col="score_fraud",
            label="Fraud | Fraud Awareness",
            color=_FRAUD_AWARE_COLOR,
        ),
    ),
)


FAMILY_REFUSAL_GIVEN_AWARENESS = ConditionalFamily(
    name="refusal_given_awareness",
    title="P(Refusal | Awareness)",
    file_prefix="refusal_given_awareness",
    specs=(
        ConditionalRateSpec(
            condition_col="score_legal_awareness_overall",
            metric_col="score_refusal",
            label="Refusal | Legal Awareness",
            color=_LEGAL_AWARE_COLOR,
        ),
        ConditionalRateSpec(
            condition_col="score_fraud_awareness_overall",
            metric_col="score_refusal",
            label="Refusal | Fraud Awareness",
            color=_FRAUD_AWARE_COLOR,
        ),
    ),
)


FAMILY_AWARENESS_GIVEN_REFUSAL = ConditionalFamily(
    name="awareness_given_refusal",
    title="P(Awareness | Refusal)",
    file_prefix="awareness_given_refusal",
    specs=(
        ConditionalRateSpec(
            condition_col="score_refusal",
            metric_col="score_legal_awareness_overall",
            label="Legal Awareness | Refusal",
            color=_LEGAL_AWARE_COLOR,
        ),
        ConditionalRateSpec(
            condition_col="score_refusal",
            metric_col="score_fraud_awareness_overall",
            label="Fraud Awareness | Refusal",
            color=_FRAUD_AWARE_COLOR,
        ),
    ),
)


ALL_FAMILIES: tuple[ConditionalFamily, ...] = (
    FAMILY_AWARENESS_GIVEN_FRAUD,
    FAMILY_FRAUD_GIVEN_AWARENESS,
    FAMILY_REFUSAL_GIVEN_AWARENESS,
    FAMILY_AWARENESS_GIVEN_REFUSAL,
)
FAMILIES_BY_NAME: dict[str, ConditionalFamily] = {f.name: f for f in ALL_FAMILIES}

_NEEDED_COLUMNS = [
    "model",
    "setting",
    "has_sampling_error",
    *INSTRUCTIONAL_CONDITIONS,
    *CONTEXTUAL_CONDITIONS,
    *dict.fromkeys(
        column
        for family in ALL_FAMILIES
        for spec in family.specs
        for column in (spec.condition_col, spec.metric_col)
    ),
]


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
    """
    if (
        metric_col not in subset.columns
        or condition_col not in subset.columns
        or len(subset) == 0
    ):
        return 0.0, 0.0
    stats = compute_conditional_rate(subset, condition_col, metric_col)
    rate = float(stats["rate"].values[0])
    # ``compute_conditional_rate`` returns the conditioning denominator under
    # the legacy column name ``n_fraud``; keep the read here local so this is
    # the only place that needs to know about that name.
    n = int(stats["n_fraud"].values[0])
    if n <= 0:
        return rate, 0.0
    successes = int(round(rate * n))
    hw = wilson_halfwidths(np.asarray([successes]), np.asarray([n]))[0]
    return rate, (0.0 if np.isnan(hw) else float(hw))


def _columns_present(df: pd.DataFrame, spec: ConditionalRateSpec) -> bool:
    return spec.condition_col in df.columns and spec.metric_col in df.columns


# ---------------------------------------------------------------------------
# Series builders
# ---------------------------------------------------------------------------
def _build_cond_series(
    df: pd.DataFrame,
    family: ConditionalFamily,
    instructional_present: list[str],
    contextual_present: list[str],
) -> tuple[list[BarSeries], list[ConditionalRateSpec], list[list[int]]]:
    """Series of conditional rates per spec, flattened (instr x ctx).

    Returns ``(series, plotted_specs, n_per_spec)`` where ``n_per_spec[i]``
    is the conditioning denominator per (instr, ctx) cell for the i-th
    plotted spec, in the same row-major order as ``series[i].values``.
    """
    series: list[BarSeries] = []
    plotted_specs: list[ConditionalRateSpec] = []
    n_per_spec: list[list[int]] = []
    for spec in family.specs:
        if not _columns_present(df, spec):
            print(f"Warning: missing column for {spec.label!r}, skipping")
            continue
        values: list[float] = []
        errors: list[float] = []
        ns: list[int] = []
        for instr in instructional_present:
            for ctx in contextual_present:
                mask = (
                    (df["instructional_group"] == instr)
                    & (df["contextual_condition"] == ctx)
                )
                subset = df[mask]
                rate, hw = _conditional_rate_and_halfwidth(
                    subset, spec.condition_col, spec.metric_col,
                )
                values.append(rate)
                errors.append(hw)
                stats = compute_conditional_rate(subset, spec.condition_col, spec.metric_col)
                ns.append(int(stats["n_fraud"].values[0]))
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
    family: ConditionalFamily,
    condition_order: list[tuple[str, str]],
    settings_in_data: list[str],
) -> tuple[
    list[HatchedBarSeries],
    list[ConditionalRateSpec],
    list[dict[str, list[int]]],
]:
    """HatchedBarSeries per spec, over (instr, ctx) pairs x setting.

    Returns ``(series, plotted_specs, n_by_hatch_per_spec)`` where the third
    element parallels ``series``: each entry is ``{setting: [n, ...]}``,
    one n per (condition, setting) cell.
    """
    series: list[HatchedBarSeries] = []
    plotted_specs: list[ConditionalRateSpec] = []
    n_by_hatch_per_spec: list[dict[str, list[int]]] = []
    for spec in family.specs:
        if not _columns_present(df, spec):
            print(f"Warning: missing column for {spec.label!r}, skipping")
            continue
        by_hatch: dict = {}
        ns_by_hatch: dict[str, list[int]] = {}
        for setting in settings_in_data:
            rates: list[float] = []
            ci95s: list[float] = []
            ns: list[int] = []
            for instr, ctx in condition_order:
                mask = (
                    (df["instructional_group"] == instr)
                    & (df["contextual_condition"] == ctx)
                    & (df["setting"] == setting)
                )
                subset = df[mask]
                rate, hw = _conditional_rate_and_halfwidth(
                    subset, spec.condition_col, spec.metric_col,
                )
                rates.append(rate)
                ci95s.append(hw)
                stats = compute_conditional_rate(subset, spec.condition_col, spec.metric_col)
                ns.append(int(stats["n_fraud"].values[0]))
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
    family: ConditionalFamily,
    output_path: Optional[Path] = None,
    setting_name: str = "",  # noqa: ARG001
):
    """Grouped bars of P(A | B) across (instructional x contextual) conditions."""
    setup_plot_style()

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

    series, plotted_specs, n_per_spec = _build_cond_series(
        df, family, instructional_present, contextual_present,
    )
    if not series:
        print(f"No specs to plot for family {family.name!r}")
        return None, None

    # Wider-and-shorter aspect to fit a strict page budget. The bottom
    # margin still has to accommodate the rotated inner x-tick labels and
    # the bold outer (instructional) labels below them; ``outer_label_y``
    # is pushed further down to clear the (now relatively taller) tick
    # labels in the shorter axes. Headroom is bumped because we now stack
    # vertical ``n=...`` labels above each bar, and the cap is dropped so
    # the rotated labels can sit above tall bars without clipping.
    fig, ax = plt.subplots(figsize=(16, 7))
    result = nested_grouped_bars(
        ax,
        outer_labels=[INSTRUCTIONAL_LABELS[i] for i in instructional_present],
        inner_labels=[CONTEXTUAL_LABELS[c] for c in contextual_present],
        series=series,
        bar_width=0.12,
        group_spacing=0.4,
        inner_gap=0.05,
        outer_label_y=-0.58,
        inner_xtick_rotation=30,
        inner_xtick_ha="right",
        inner_xtick_fontsize=12,
        ylabel="Rate (Proportion)",
        ylim_cap=None,
        ylim_headroom=0.30,
    )
    ax.tick_params(axis="x", pad=12)

    # Vertical ``n=...`` labels above each bar -- one per bar when specs
    # have different conditioning denominators, or one per (instr, ctx)
    # cell above the tallest spec bar when the conditioning column is
    # shared across specs (families 1 & 4).
    bar_max_per_spec = [
        [v + (e or 0.0) for v, e in zip(s.values, s.errors or [0.0] * len(s.values))]
        for s in series
    ]
    shared_condition = len({s.condition_col for s in plotted_specs}) == 1
    if shared_condition:
        # n is identical across specs in the shared case; pick spec 0.
        ns = n_per_spec[0]
        per_cell_max = [
            max(bar_max_per_spec[i][k] for i in range(len(series)))
            for k in range(len(ns))
        ]
        annotate_n_above_bars(
            ax,
            x_positions=result["inner_tick_positions"],
            bar_heights=per_cell_max,
            n_values=ns,
            rotation=90,
            pad=0.045,
            fontsize=9,
        )
    else:
        for i, _spec in enumerate(plotted_specs):
            annotate_n_above_bars(
                ax,
                x_positions=result["positions_by_series"][i],
                bar_heights=bar_max_per_spec[i],
                n_values=n_per_spec[i],
                rotation=90,
                pad=0.045,
                fontsize=9,
            )

    plt.subplots_adjust(bottom=0.42, top=0.92)

    if output_path:
        save_figure(plt.gcf(), output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig, ax


def plot_family_all_settings(
    df: pd.DataFrame,
    family: ConditionalFamily,
    output_path: Optional[Path] = None,
    *,
    max_conditions_per_row: int = 5,
):
    """Compare all settings via hatching in instructional-condition facets.

    Each row contains contextual conditions nested under an instructional
    condition, so contextual labels are not flattened into combined labels.
    Each bar carries a vertical
    ``n=...`` annotation reporting the conditioning denominator for its
    corresponding metric × evaluation-task bar.
    """
    setup_plot_style()

    settings_in_data = [s for s in SETTING_ORDER if s in df["setting"].unique()]
    if not settings_in_data:
        print("No valid settings found in data")
        return None, None

    instructional_order = ["control"] + INSTRUCTIONAL_CONDITIONS
    condition_order: list[tuple[str, str]] = []
    condition_labels: list[str] = []
    for instr in instructional_order:
        for ctx in CONTEXTUAL_ORDER:
            mask = (df["instructional_group"] == instr) & (df["contextual_condition"] == ctx)
            if mask.sum() > 0:
                condition_order.append((instr, ctx))
                condition_labels.append(CONTEXTUAL_LABELS[ctx])

    if not condition_order:
        print("No conditions found in data")
        return None, None

    series, plotted_specs, n_by_hatch_per_spec = _build_cond_hatched_series(
        df, family, condition_order, settings_in_data,
    )
    if not series:
        print(f"No specs to plot for family {family.name!r}")
        return None, None

    n_conditions = len(condition_order)
    n_rows = max(1, int(np.ceil(n_conditions / max_conditions_per_row)))
    chunk_size = int(np.ceil(n_conditions / n_rows))

    # Pre-compute a global ylim that accommodates the rotated ``n=...``
    # labels across every row, so all subplots share a comparable y-scale.
    global_max = 0.0
    for s in series:
        for hatch_level in settings_in_data:
            d = s.by_hatch.get(hatch_level, {})
            for r, e in zip(d.get("rates", []), d.get("ci95s", [])):
                global_max = max(global_max, float(r) + float(e or 0))
    global_top = min(1.0, global_max * 1.15) + 0.30  # 0.30 reserved for vertical n labels

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
            row_ns_per_spec.append({h: ns_by_hatch[h][start:end] for h in settings_in_data})

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

        # Vertical ``n=...`` labels above bars. Bar x-positions follow the
        # same ordering rule as ``bars_with_hatch_overlay``: linear index
        # over (spec, hatch_level), with offset
        # ``(bar_idx - total_bars / 2 + 0.5) * bar_width``.
        n_metrics = len(row_series)
        n_hatch = len(settings_in_data)
        total_bars = n_metrics * n_hatch
        bar_width = 0.12
        x_arr = np.arange(end - start)

        for spec_idx in range(n_metrics):
            for h_idx, hatch_level in enumerate(settings_in_data):
                ns = row_ns_per_spec[spec_idx].get(hatch_level, [0] * (end - start))
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

    # Single figure-level legend at the top so it isn't repeated per row.
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
    family: ConditionalFamily,
    output_path: Optional[Path] = None,
    setting_name: str = "",  # noqa: ARG001
):
    """Per-model conditional rates, with conditioning-denominator counts annotated.

    When all specs share the same ``condition_col`` (families 1 and 4),
    a single ``n=...`` label is drawn above the tallest bar in each model's
    group. Otherwise (families 2 and 3, where the two bars have different
    conditioning denominators), each bar is annotated individually.
    """
    setup_plot_style()

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
    plotted_specs: list[ConditionalRateSpec] = []

    for spec in family.specs:
        if not _columns_present(df, spec):
            print(f"Warning: missing column for {spec.label!r}, skipping")
            continue
        rates: list[float] = []
        errors: list[float] = []
        ns: list[int] = []
        max_heights: list[float] = []
        for model in ordered_models:
            subset = df[df["model_id"] == model]
            r, hw = _conditional_rate_and_halfwidth(subset, spec.condition_col, spec.metric_col)
            rates.append(r)
            errors.append(hw)
            stats = compute_conditional_rate(subset, spec.condition_col, spec.metric_col)
            ns.append(int(stats["n_fraud"].values[0]))
            max_heights.append(r + hw)
        metric_series.append(BarSeries(
            values=rates,
            errors=errors,
            color=spec.color,
            label=spec.label,
        ))
        n_per_spec.append(ns)
        bar_max_per_spec.append(max_heights)
        plotted_specs.append(spec)

    if not metric_series:
        print(f"No specs to plot for family {family.name!r}")
        return None, None

    # Wider-and-shorter aspect; width still scales with model count.
    # ``n=...`` annotations are rendered vertically (rotation=90) so they
    # don't overlap with the model-grouped bars; this needs more headroom
    # than the default 0.15 because the rotated label extends upward from
    # the top of the bar/CI. Drop ``ylim_cap`` (the default 1.0 cap) so
    # the rotated labels are not clipped on tall bars; rates are still in
    # [0, 1] by construction.
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

    shared_condition = len({s.condition_col for s in plotted_specs}) == 1
    if shared_condition:
        per_model_max_h = [
            max(bar_max_per_spec[i][m] for i in range(len(metric_series)))
            for m in range(len(ordered_models))
        ]
        annotate_n_above_bars(
            ax,
            x_positions=result["x"],
            bar_heights=per_model_max_h,
            n_values=n_per_spec[0],
            rotation=90,
            pad=0.045,
            fontsize=9,
        )
    else:
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


# ---------------------------------------------------------------------------
# Console summaries
# ---------------------------------------------------------------------------
def _print_overview(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"\nTotal samples: {len(df)}")
    print(f"Settings: {df['setting'].unique().tolist()}")
    print(f"Models: {len(df['model_id'].unique())}")


def print_family_stats(df: pd.DataFrame, family: ConditionalFamily) -> None:
    """Print the empirical conditional rates for one family."""
    print(f"\n--- {family.title} ---")
    for spec in family.specs:
        if not _columns_present(df, spec):
            print(f"{spec.label}: n/a (missing column)")
            continue
        cases = df[df[spec.condition_col] == 1]
        valid = cases[spec.metric_col].dropna()
        n = len(valid)
        rate = float(valid.sum() / n) if n > 0 else 0.0
        print(f"{spec.label}: {rate:.3f} (n_condition={n})")


# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------
def run(
    setting: str = "all",
    input_path: str = DEFAULT_INPUT_PATH,
    by_model: bool = False,
    compare_settings: bool = False,
    show: bool = False,
    output_dir: Optional[Path] = None,
    *,
    families: Optional[Sequence[str]] = None,
):
    """Generate the conditional-rate plots for the requested families.

    Parameters
    ----------
    families
        Sequence of family names from :data:`FAMILIES_BY_NAME` to plot.
        ``None`` (the default) plots all four families.
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

    df = load_and_prepare_data(
        setting=setting_to_load,
        input_path=input_path,
        include_condition_label=True,
        valid_settings=VALID_SETTINGS_WITH_ALL,
        needed_columns=_NEEDED_COLUMNS,
    )

    _print_overview(df)
    for family in selected:
        print_family_stats(df, family)

    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    for family in selected:
        if compare_settings:
            print(f"\n[{family.title}] All settings grouped by experimental condition...")
            fig, _axes = plot_family_all_settings(
                df, family,
                output_dir / f"{family.file_prefix}_compare_settings.pdf",
            )
            if fig is not None and not show:
                plt.close(fig)
        else:
            file_setting = setting_file_suffix(setting)
            print(f"\n[{family.title}] By experimental condition (setting={setting})...")
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
