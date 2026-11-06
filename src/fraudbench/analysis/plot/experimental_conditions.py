"""Experimental-condition plots, backing ``plot_experimental_conditions.py``.

Supports three variants:

1. Overall rates: 2x1 subplot with fraud/refusal on top and awareness metrics on bottom.
2. By-model rates: 2x1 subplot with metrics clustered by model.
3. Compare settings: 2x1 subplot with settings overlaid via hatching patterns.

Every bar carries a 95% Wilson binomial CI on the empirical proportion
as a symmetric error bar. See the matching tables under
``paper/tables/regressions/`` for the inferential output of the M1 fit.
"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from ..registry import MODEL_ORDER, SETTING_DISPLAY_NAMES

from ..style import METRIC_COLORS, save_figure, setup_plot_style
from ..charts import (
    BarSeries,
    HatchedBarSeries,
    above_axes_legend_kw,
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
    METRIC_CONFIGS_FULL,
    SETTING_HATCHES,
    SETTING_ORDER,
    VALID_SETTINGS_WITH_ALL,
    get_model_display_name,
    load_and_prepare_data,
    setting_file_suffix,
)
from ..uncertainty import rate_and_halfwidth as _rate_and_halfwidth


METRIC_CONFIGS = {
    **METRIC_CONFIGS_FULL,
    "options": {
        "column": "score_options",
        "display_name": "Options",
        "y_label": "Options Rate (Proportion)",
        "file_suffix": "options",
        "color": METRIC_COLORS["options"],
    },
}

_NEEDED_COLUMNS = [
    "model",
    "setting",
    "has_sampling_error",
    *INSTRUCTIONAL_CONDITIONS,
    *CONTEXTUAL_CONDITIONS,
    *(config["column"] for config in METRIC_CONFIGS.values()),
]


def _build_nested_series(
    df: pd.DataFrame,
    metrics: list[str],
    instructional_present: list[str],
    contextual_present: list[str],
) -> list[BarSeries]:
    """One :class:`BarSeries` per metric, flattened (instr-first, ctx-second)."""
    series: list[BarSeries] = []
    for metric in metrics:
        config = METRIC_CONFIGS[metric]
        col = config["column"]
        values: list[float] = []
        errors: list[float] = []
        for instr in instructional_present:
            for ctx in contextual_present:
                mask = (df["instructional_group"] == instr) & (df["contextual_condition"] == ctx)
                rate, hw = _rate_and_halfwidth(df[mask], col)
                values.append(rate)
                errors.append(hw)
        series.append(BarSeries(
            values=values,
            errors=errors,
            color=config["color"], label=config["display_name"],
        ))
    return series


def plot_overall_by_condition(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    setting_name: str = "",  # noqa: ARG001
):
    """2x1 subplot: behavioural metrics on top, awareness metrics on bottom."""
    setup_plot_style()

    instructional_order = ["control"] + INSTRUCTIONAL_CONDITIONS
    instructional_present = [i for i in instructional_order if i in df["instructional_group"].unique()]
    contextual_present = [c for c in CONTEXTUAL_ORDER if c in df["contextual_condition"].unique()]

    if not instructional_present or not contextual_present:
        print("No conditions found in data")
        return None, None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 26))

    def _draw(ax, metrics, *, show_inner_labels: bool, show_outer_labels: bool):
        nested_grouped_bars(
            ax,
            outer_labels=[INSTRUCTIONAL_LABELS[i] for i in instructional_present],
            inner_labels=[CONTEXTUAL_LABELS[c] for c in contextual_present],
            series=_build_nested_series(
                df, metrics, instructional_present, contextual_present,
            ),
            bar_width=0.12,
            group_spacing=0.4,
            inner_gap=0.05,
            outer_label_y=-0.92,
            inner_xtick_rotation=30,
            inner_xtick_ha="right",
            inner_xtick_fontsize=14,
            ylabel="Rate (Proportion)",
            show_inner_xticks=show_inner_labels,
            show_outer_labels=show_outer_labels,
            show_legend=False,
        )

    _draw(
        ax1,
        ["fraud", "refusal", "options"],
        show_inner_labels=True,
        show_outer_labels=False,
    )
    _draw(
        ax2,
        ["legal_awareness", "fraud_awareness"],
        show_inner_labels=True,
        show_outer_labels=True,
    )
    behaviour_handles, behaviour_labels = ax1.get_legend_handles_labels()
    awareness_handles, awareness_labels = ax2.get_legend_handles_labels()
    fig.legend(
        behaviour_handles, behaviour_labels,
        **above_axes_legend_kw(ncol=len(behaviour_labels), bbox_to_anchor=(0.5, 0.99)),
    )
    fig.legend(
        awareness_handles, awareness_labels,
        **above_axes_legend_kw(ncol=len(awareness_labels), bbox_to_anchor=(0.5, 0.955)),
    )
    plt.subplots_adjust(bottom=0.66, top=0.93, hspace=0.95)

    if output_path:
        save_figure(plt.gcf(), output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig, (ax1, ax2)


def _build_hatched_series(
    df: pd.DataFrame,
    metrics: list[str],
    condition_order: list[tuple[str, str]],
    settings_in_data: list[str],
) -> list[HatchedBarSeries]:
    """One :class:`HatchedBarSeries` per metric, indexed by (instr, ctx) pairs x setting."""
    series: list[HatchedBarSeries] = []
    for metric in metrics:
        config = METRIC_CONFIGS[metric]
        col = config["column"]
        if col not in df.columns:
            print(f"Warning: Column {col} not found, skipping {metric}")
            continue
        by_hatch: dict = {}
        for setting in settings_in_data:
            rates: list[float] = []
            ci95s: list[float] = []
            for instr, ctx in condition_order:
                mask = ((df["instructional_group"] == instr)
                        & (df["contextual_condition"] == ctx)
                        & (df["setting"] == setting))
                rate, hw = _rate_and_halfwidth(df[mask], col)
                rates.append(rate)
                ci95s.append(hw)
            by_hatch[setting] = {"rates": rates, "ci95s": ci95s}
        series.append(HatchedBarSeries(
            label=config["display_name"], color=config["color"], by_hatch=by_hatch,
        ))
    return series


def plot_overall_by_condition_all_settings(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
):
    """Faceted condition comparison with settings distinguished by hatching."""
    setup_plot_style()

    settings_in_data = [s for s in SETTING_ORDER if s in df["setting"].unique()]
    if not settings_in_data:
        print("No valid settings found in data")
        return None, None

    instructional_order = ["control"] + INSTRUCTIONAL_CONDITIONS
    instructional_present = [
        instr for instr in instructional_order
        if (df["instructional_group"] == instr).any()
    ]
    if not instructional_present:
        print("No conditions found in data")
        return None, None

    fig, axes = plt.subplots(
        len(instructional_present), 2,
        figsize=(18, max(16, len(instructional_present) * 5)),
        squeeze=False,
    )
    behaviour_metrics = ["fraud", "refusal", "options"]
    awareness_metrics = ["legal_awareness", "fraud_awareness"]

    for row, instr in enumerate(instructional_present):
        contexts = [
            ctx for ctx in CONTEXTUAL_ORDER
            if ((df["instructional_group"] == instr) & (df["contextual_condition"] == ctx)).any()
        ]
        condition_order = [(instr, ctx) for ctx in contexts]
        contextual_labels = [CONTEXTUAL_LABELS[ctx] for ctx in contexts]
        behavioural_series = _build_hatched_series(
            df, behaviour_metrics, condition_order, settings_in_data,
        )
        awareness_series = _build_hatched_series(
            df, awareness_metrics, condition_order, settings_in_data,
        )

        behaviour_ax, awareness_ax = axes[row]
        bars_with_hatch_overlay(
            behaviour_ax,
            x_labels=contextual_labels,
            series=behavioural_series,
            hatch_order=settings_in_data,
            hatch_patterns=SETTING_HATCHES,
            hatch_display_labels=SETTING_DISPLAY_NAMES,
            bar_width=0.10,
            ylabel="Rate (Proportion)",
            xtick_rotation=30,
            xtick_ha="right",
            xtick_fontsize=12,
            ylim_headroom=0.35,
            show_legend=False,
        )
        bars_with_hatch_overlay(
            awareness_ax,
            x_labels=contextual_labels,
            series=awareness_series,
            hatch_order=settings_in_data,
            hatch_patterns=SETTING_HATCHES,
            hatch_display_labels=SETTING_DISPLAY_NAMES,
            bar_width=0.10,
            ylabel="Rate (Proportion)",
            xtick_rotation=30,
            xtick_ha="right",
            xtick_fontsize=12,
            ylim_headroom=0.35,
            show_legend=False,
        )
    behaviour_handles, behaviour_labels = build_hatch_legend(
        [(METRIC_CONFIGS[m]["display_name"], METRIC_CONFIGS[m]["color"]) for m in behaviour_metrics],
        [(SETTING_DISPLAY_NAMES[s], SETTING_HATCHES[s]) for s in settings_in_data],
    )
    awareness_handles, awareness_labels = build_hatch_legend(
        [(METRIC_CONFIGS[m]["display_name"], METRIC_CONFIGS[m]["color"]) for m in awareness_metrics],
        [(SETTING_DISPLAY_NAMES[s], SETTING_HATCHES[s]) for s in settings_in_data],
    )
    fig.legend(
        behaviour_handles, behaviour_labels,
        **above_axes_legend_kw(ncol=len(behaviour_labels), bbox_to_anchor=(0.5, 0.99)),
    )
    fig.legend(
        awareness_handles, awareness_labels,
        **above_axes_legend_kw(ncol=len(awareness_labels), bbox_to_anchor=(0.5, 0.95)),
    )
    fig.subplots_adjust(top=0.91, bottom=0.08, hspace=0.76, wspace=0.28)
    for row, instr in enumerate(instructional_present):
        behaviour_ax, awareness_ax = axes[row]
        centre_x = (behaviour_ax.get_position().x0 + awareness_ax.get_position().x1) / 2
        fig.text(
            centre_x,
            behaviour_ax.get_position().y1 + 0.008,
            INSTRUCTIONAL_LABELS[instr],
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

    if output_path:
        save_figure(plt.gcf(), output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig, axes


def plot_by_model(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    setting_name: str = "",  # noqa: ARG001
):
    """2x1 subplot of metric rates by model (behavioural top, awareness bottom)."""
    setup_plot_style()

    models_in_data = df["model_id"].unique()
    ordered_models = [m for m in MODEL_ORDER if m in models_in_data]
    for m in models_in_data:
        if m not in ordered_models:
            ordered_models.append(m)

    if not ordered_models:
        print("No models found in data")
        return None, None

    def _model_series(metrics):
        series = []
        for metric in metrics:
            config = METRIC_CONFIGS[metric]
            col = config["column"]
            if col not in df.columns:
                print(f"Warning: Column {col} not found, skipping {metric}")
                continue
            rates: list[float] = []
            errors: list[float] = []
            for model in ordered_models:
                rate, hw = _rate_and_halfwidth(df[df["model_id"] == model], col)
                rates.append(rate)
                errors.append(hw)
            series.append(BarSeries(
                values=rates,
                errors=errors,
                color=config["color"], label=config["display_name"],
            ))
        return series

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, len(ordered_models) * 0.8), 10))
    model_labels = [get_model_display_name(m) for m in ordered_models]

    grouped_bars(
        ax1,
        x_labels=model_labels,
        series=_model_series(["fraud", "refusal", "options"]),
        bar_width=0.25,
        xtick_rotation=30,
        ylabel="Rate (Proportion)",
    )
    grouped_bars(
        ax2,
        x_labels=model_labels,
        series=_model_series(["legal_awareness", "fraud_awareness"]),
        bar_width=0.35,
        xtick_rotation=30,
        ylabel="Rate (Proportion)",
        xlabel="Model",
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if output_path:
        save_figure(plt.gcf(), output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig, (ax1, ax2)


def print_summary_stats(df: pd.DataFrame):
    """Print aggregated summary statistics across all four metrics."""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    print(f"\nTotal samples: {len(df)}")
    print(f"Settings: {df['setting'].unique().tolist()}")
    print(f"Models: {len(df['model_id'].unique())}")

    print("\n--- Overall Rates (Aggregated) ---")
    for _, config in METRIC_CONFIGS.items():
        col = config["column"]
        if col in df.columns:
            valid = df[col].dropna()
            rate = valid.sum() / len(valid) if len(valid) > 0 else 0
            print(f"{config['display_name']}: {rate:.3f} (n={len(valid)})")

    print("\n--- Samples per Model ---")
    models_in_data = set(df["model_id"].unique())
    ordered_models = [m for m in MODEL_ORDER if m in models_in_data]
    for m in models_in_data:
        if m not in ordered_models:
            ordered_models.append(m)
    for model in ordered_models:
        n = len(df[df["model_id"] == model])
        print(f"  {get_model_display_name(model)}: {n}")


def run(
    setting: str = "all",
    input_path: str = DEFAULT_INPUT_PATH,
    by_model: bool = False,
    compare_settings: bool = False,
    show: bool = False,
    output_dir: Optional[Path] = None,
):
    """Generate the experimental-condition plots. Mirrors the legacy CLI.

    Error bars are 95% Wilson binomial CIs on the empirical proportions.
    """
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

    print_summary_stats(df)

    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    setting_label = setting.capitalize() if setting != "all" else "All Settings"

    if compare_settings:
        print("\nPlot: All settings grouped by experimental condition...")
        fig, _axes = plot_overall_by_condition_all_settings(
            df, output_dir / "overall_by_condition_compare_settings.pdf",
        )
        if fig is not None and not show:
            plt.close(fig)
    else:
        file_setting = setting_file_suffix(setting)
        print("\nPlot 1: Overall rates by experimental condition...")
        fig, _axes = plot_overall_by_condition(
            df, output_dir / f"overall_by_condition_{file_setting}.pdf", setting_name=setting_label,
        )
        if fig is not None and not show:
            plt.close(fig)
        if by_model:
            print("\nPlot 2: Rates by model (2x1 subplot)...")
            fig, _axes = plot_by_model(
                df, output_dir / f"by_model_{file_setting}.pdf", setting_name=setting_label,
            )
            if fig is not None and not show:
                plt.close(fig)

    print(f"\nDone! Plots saved to: {output_dir}")

    if show:
        plt.show()
