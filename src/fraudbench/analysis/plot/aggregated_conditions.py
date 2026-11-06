"""Aggregated-condition plots (fraud, refusal, and options), backing ``plot_aggregated_conditions.py``.

Error bars are 95% Wilson binomial CIs on the empirical proportions
(descriptive, always contain the empirical mean, robust near 0 / 100%).
The matching inferential output (regression coefficients on the same
conditions) lives in ``regression_coefs_scenario.tex`` from the M1 fit.
"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from ..registry import SETTING_DISPLAY_NAMES

from ..style import METRIC_COLORS, save_figure, setup_plot_style
from ..charts import (
    BarSeries,
    HatchedBarSeries,
    above_axes_legend_kw,
    bars_with_hatch_overlay,
    build_hatch_legend,
    grouped_bars,
)
from ..common import (
    CONTEXTUAL_LABELS,
    CONTEXTUAL_ORDER,
    DEFAULT_INPUT_PATH,
    INSTRUCTIONAL_CONDITIONS,
    INSTRUCTIONAL_LABELS,
    METRIC_CONFIGS_FULL,
    SETTING_HATCHES,
    SETTING_ORDER,
    VALID_SETTINGS_WITH_ALL,
    load_and_prepare_data,
    setting_file_suffix,
)
from ..uncertainty import rate_and_halfwidth


# Behavioural metrics aggregated in this script.
METRIC_CONFIGS = {
    "fraud": METRIC_CONFIGS_FULL["fraud"],
    "refusal": METRIC_CONFIGS_FULL["refusal"],
    "options": {
        "column": "score_options",
        "display_name": "Options",
        "y_label": "Options Rate (Proportion)",
        "file_suffix": "options",
        "color": METRIC_COLORS["options"],
    },
}

# The compact main-text version of the compare-settings figure excludes the
# auxiliary options score while retaining the two behavioural outcomes.
MAIN_TEXT_METRIC_CONFIGS = {
    metric: METRIC_CONFIGS[metric] for metric in ("fraud", "refusal")
}

# Columns required from the parquet/cache for this analysis.
_NEEDED_COLUMNS = [
    "setting", "model", "has_sampling_error",
    "score_fraud", "score_refusal", "score_options",
    "assist_user", "no_illegal_behavior", "high_agency", "maximize_profit",
    "peer_pressure_low", "peer_pressure_high",
    "negative_incentive_low", "negative_incentive_high",
]


def _series_for_group(
    df: pd.DataFrame,
    group_col: str,
    groups: list[str],
) -> list[BarSeries]:
    """Build one :class:`BarSeries` per metric, indexed by category values.

    Each series carries the matching 95% Wilson half-widths in
    :attr:`BarSeries.errors`.
    """
    series: list[BarSeries] = []
    for _, config in METRIC_CONFIGS.items():
        col = config["column"]
        if col not in df.columns:
            continue
        rates: list[float] = []
        errors: list[float] = []
        for group in groups:
            subset = df[df[group_col] == group]
            rate, hw = rate_and_halfwidth(subset, col)
            rates.append(rate)
            errors.append(hw)
        series.append(BarSeries(
            values=rates,
            errors=errors,
            color=config["color"], label=config["display_name"],
        ))
    return series


def plot_aggregated_conditions(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    setting_name: str = "",  # noqa: ARG001
):
    """1x2 subplot of behavioural metrics by instructional and contextual condition."""
    setup_plot_style()

    instructional_order = ["control"] + INSTRUCTIONAL_CONDITIONS
    instructional_present = [i for i in instructional_order if i in df["instructional_group"].unique()]
    contextual_present = [c for c in CONTEXTUAL_ORDER if c in df["contextual_condition"].unique()]

    if not instructional_present:
        print("No instructional conditions found in data")
        return None, None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Single shared behavioural-metric legend centered above both panels.
    s_inst = _series_for_group(df, "instructional_group", instructional_present)
    s_ctx = _series_for_group(df, "contextual_condition", contextual_present)

    grouped_bars(
        ax1,
        x_labels=[INSTRUCTIONAL_LABELS[i] for i in instructional_present],
        series=s_inst,
        bar_width=0.25,
        bar_alpha=0.85,
        ylabel="Rate (Proportion)",
        xlabel="Instructional Condition",
        ylim_headroom=0.25,
        show_legend=False,
    )

    grouped_bars(
        ax2,
        x_labels=[CONTEXTUAL_LABELS[c] for c in contextual_present],
        series=s_ctx,
        bar_width=0.25,
        bar_alpha=0.85,
        ylabel="Rate (Proportion)",
        xlabel="Contextual Condition",
        ylim_headroom=0.25,
        show_legend=False,
    )

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, **above_axes_legend_kw(ncol=len(labels)))

    fig.tight_layout()
    fig.align_xlabels([ax1, ax2])
    plt.subplots_adjust(bottom=0.30, top=0.88)

    if output_path:
        save_figure(plt.gcf(), output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig, (ax1, ax2)


def _hatched_series_for_group(
    df: pd.DataFrame,
    group_col: str,
    groups: list[str],
    settings_in_data: list[str],
    *,
    metric_configs: dict = METRIC_CONFIGS,
) -> list[HatchedBarSeries]:
    """Build one :class:`HatchedBarSeries` per metric x setting.

    ``by_hatch[setting]`` carries ``rates`` and ``ci95s`` (Wilson
    half-widths) so the underlying primitive can render error bars.
    """
    series: list[HatchedBarSeries] = []
    for _, config in metric_configs.items():
        col = config["column"]
        if col not in df.columns:
            continue
        by_hatch: dict = {}
        for setting in settings_in_data:
            rates: list[float] = []
            errors: list[float] = []
            for group in groups:
                mask = (df[group_col] == group) & (df["setting"] == setting)
                rate, hw = rate_and_halfwidth(df[mask], col)
                rates.append(rate)
                errors.append(hw)
            by_hatch[setting] = {"rates": rates, "ci95s": errors}
        series.append(HatchedBarSeries(
            label=config["display_name"], color=config["color"], by_hatch=by_hatch,
        ))
    return series


def plot_aggregated_conditions_compare_settings(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    *,
    metric_configs: dict = METRIC_CONFIGS,
    figure_title: Optional[str] = None,
):
    """Overlay all AI settings with hatching for each requested metric."""
    setup_plot_style()

    settings_in_data = [s for s in SETTING_ORDER if s in df["setting"].unique()]
    if not settings_in_data:
        print("No valid settings found in data")
        return None, None

    instructional_order = ["control"] + INSTRUCTIONAL_CONDITIONS
    instructional_present = [i for i in instructional_order if i in df["instructional_group"].unique()]
    contextual_present = [c for c in CONTEXTUAL_ORDER if c in df["contextual_condition"].unique()]

    if not instructional_present:
        print("No instructional conditions found in data")
        return None, None

    # Wider-and-shorter aspect to fit a strict page budget. Headroom is
    # trimmed because the hatched bars cap at ~0.30 and the figure-level
    # legend sits above the axes (not over them).
    fig_height = 5.5 if figure_title else 4.5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, fig_height))

    hatched_inst = _hatched_series_for_group(
        df, "instructional_group", instructional_present, settings_in_data,
        metric_configs=metric_configs,
    )
    hatched_ctx = _hatched_series_for_group(
        df, "contextual_condition", contextual_present, settings_in_data,
        metric_configs=metric_configs,
    )

    bars_with_hatch_overlay(
        ax1,
        x_labels=[INSTRUCTIONAL_LABELS[i] for i in instructional_present],
        series=hatched_inst,
        hatch_order=settings_in_data,
        hatch_patterns=SETTING_HATCHES,
        hatch_display_labels=SETTING_DISPLAY_NAMES,
        bar_width=0.10,
        ylabel="Rate (Proportion)",
        xlabel="Instructional Condition",
        ylim_headroom=0.20,
        show_legend=False,
    )

    bars_with_hatch_overlay(
        ax2,
        x_labels=[CONTEXTUAL_LABELS[c] for c in contextual_present],
        series=hatched_ctx,
        hatch_order=settings_in_data,
        hatch_patterns=SETTING_HATCHES,
        hatch_display_labels=SETTING_DISPLAY_NAMES,
        bar_width=0.10,
        ylabel="Rate (Proportion)",
        xlabel="Contextual Condition",
        ylim_headroom=0.20,
        show_legend=False,
    )

    if figure_title:
        for ax in (ax1, ax2):
            ax.xaxis.label.set_fontsize(18)
            ax.yaxis.label.set_fontsize(18)
            ax.tick_params(axis="both", which="major", labelsize=16)

    color_entries = [(s.label, s.color) for s in hatched_inst]
    hatch_entries = [
        (SETTING_DISPLAY_NAMES[h], SETTING_HATCHES[h]) for h in settings_in_data
    ]
    handles, labels = build_hatch_legend(color_entries, hatch_entries)

    fig.tight_layout()
    fig.align_xlabels([ax1, ax2])
    plt.subplots_adjust(bottom=0.24, top=0.70 if figure_title else 0.88)

    legend_y = 0.74 if figure_title else 1.0
    legend_kw = above_axes_legend_kw(
        ncol=len(labels), bbox_to_anchor=(0.5, legend_y)
    )
    if figure_title:
        legend_kw["fontsize"] = 16
    fig.legend(
        handles,
        labels,
        **legend_kw,
    )
    if figure_title:
        fig.suptitle(figure_title, fontsize=22, fontweight="bold", y=0.98)

    if output_path:
        save_figure(plt.gcf(), output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig, (ax1, ax2)


def plot_aggregated_conditions_compare_settings_main_text(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
):
    """Plot the compact fraud-and-refusal-only compare-settings figure."""
    return plot_aggregated_conditions_compare_settings(
        df,
        output_path,
        metric_configs=MAIN_TEXT_METRIC_CONFIGS,
        figure_title=(
            "Fraud rates increase when (1) tasks are more agentic,\n"
            "(2) models are asked to act agentically, and "
            "(3) contextual pressures increase"
        ),
    )


def print_summary_stats(df: pd.DataFrame):
    """Print aggregated summary statistics for fraud and refusal."""
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

    print("\n--- Rates by Instructional Condition ---")
    for instr in ["control"] + INSTRUCTIONAL_CONDITIONS:
        subset = df[df["instructional_group"] == instr]
        if len(subset) == 0:
            continue
        print(f"\n{INSTRUCTIONAL_LABELS[instr]}:")
        for _, config in METRIC_CONFIGS.items():
            col = config["column"]
            if col in df.columns:
                valid = subset[col].dropna()
                rate = valid.sum() / len(valid) if len(valid) > 0 else 0
                print(f"  {config['display_name']}: {rate:.3f} (n={len(valid)})")

    print("\n--- Rates by Contextual Condition ---")
    for ctx in CONTEXTUAL_ORDER:
        subset = df[df["contextual_condition"] == ctx]
        if len(subset) == 0:
            continue
        label = CONTEXTUAL_LABELS[ctx].replace("\n", " ")
        print(f"\n{label}:")
        for _, config in METRIC_CONFIGS.items():
            col = config["column"]
            if col in df.columns:
                valid = subset[col].dropna()
                rate = valid.sum() / len(valid) if len(valid) > 0 else 0
                print(f"  {config['display_name']}: {rate:.3f} (n={len(valid)})")


def run(
    setting: str = "all",
    input_path: str = DEFAULT_INPUT_PATH,
    compare_settings: bool = False,
    compare_settings_main_text: bool = False,
    show: bool = False,
    output_dir: Optional[Path] = None,
):
    """Generate the aggregated-condition plots. Mirrors the legacy CLI.

    Error bars are 95% Wilson binomial CIs on the empirical proportions.
    """
    if output_dir is None:
        output_dir = Path("./paper/figures")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setting_to_load = (
        "all"
        if (compare_settings or compare_settings_main_text)
        else setting
    )

    df = load_and_prepare_data(
        setting=setting_to_load,
        input_path=input_path,
        metric_columns=[cfg["column"] for cfg in METRIC_CONFIGS.values()],
        valid_settings=VALID_SETTINGS_WITH_ALL,
        needed_columns=_NEEDED_COLUMNS,
    )

    print_summary_stats(df)

    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    setting_label = setting.capitalize() if setting != "all" else "All Settings"

    if compare_settings_main_text:
        print("\nPlot: Main-text all-settings comparison by condition type...")
        fig, _axes = plot_aggregated_conditions_compare_settings_main_text(
            df, output_dir / "main_text_aggregated_conditions_compare_settings.pdf",
        )
        if fig is not None and not show:
            plt.close(fig)
    elif compare_settings:
        print("\nPlot: All settings compared by condition type...")
        fig, _axes = plot_aggregated_conditions_compare_settings(
            df, output_dir / "aggregated_conditions_compare_settings.pdf",
        )
        if fig is not None and not show:
            plt.close(fig)
    else:
        print(f"\nPlot: Aggregated rates by condition type ({setting_label})...")
        fig, _axes = plot_aggregated_conditions(
            df, output_dir / f"aggregated_conditions_{setting_file_suffix(setting)}.pdf",
            setting_name=setting_label,
        )
        if fig is not None and not show:
            plt.close(fig)

    print(f"\nDone! Plots saved to: {output_dir}")

    if show:
        plt.show()
