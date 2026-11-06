"""Main vs ablation comparison plots, broken out by model and setting.

For each ablation study (``reasoning`` / ``temperature``) we render one
standalone figure per metric (fraud, refusal, options, legal awareness,
fraud awareness). Within each metric figure the bars are nested:

* outer groups: model (one per ablation model)
* inner categories: AI setting (Binary / Open / Agentic / Overall)
* two series per inner category: ``Main`` (solid) and the ablation (hatched)

The "Overall" inner column aggregates that model's matched cells across
all settings; the per-setting columns split the same data by setting. All
rates are computed on the matched-cell main subset (same
``model_id × summary_id × condition × setting`` cells the ablation actually
ran on), so every comparison is strictly within-cell.

Only the models that participated in the ablation (per
:data:`fraudbench.utils.globals.REASONING_ABLATION_MODELS` /
:data:`TEMPERATURE_ABLATION_MODELS`) are included.

Error bars are 95% Wilson binomial CIs on the empirical proportions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from fraudbench.utils.globals import (
    REASONING_ABLATION_MODELS,
    TEMPERATURE_ABLATION_MODELS,
)

from ..charts import BarSeries, above_axes_legend_kw, nested_grouped_bars
from ..common import (
    AI_SETTINGS,
    CONTEXTUAL_CONDITIONS,
    DEFAULT_INPUT_PATH,
    INSTRUCTIONAL_CONDITIONS,
    METRIC_CONFIGS_FULL,
    VALID_SETTINGS_WITH_ALL,
    add_condition_columns,
    coerce_none_to_nan,
    compute_metric_rate,
    drop_errors,
    filter_setting,
    get_model_display_name,
    load_data_from_source,
    setting_file_suffix,
)
from ..registry import MODEL_ORDER, normalize_model_id
from ..style import METRIC_COLORS, save_figure, setup_plot_style
from ..uncertainty import rate_and_halfwidth as _rate_and_halfwidth


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: Five-metric configuration. Mirrors :mod:`case_attributes.py` (extends the
#: shared four-metric registry with ``options`` so we can plot
#: ``score_options`` next to the rest).
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

#: Order in which metrics are rendered (one standalone figure per metric).
METRIC_ORDER: list[str] = [
    "fraud", "refusal", "options",
    "legal_awareness", "fraud_awareness",
]


#: Run-category value (in ``obs_data.parquet``) per ablation type.
ABLATION_RUN_CATEGORIES: dict[str, str] = {
    "reasoning":   "reasoning_ablation",
    "temperature": "temperature_ablation",
}


#: Display label for each ablation type (used in legends and titles).
ABLATION_DISPLAY_NAMES: dict[str, str] = {
    "reasoning":   "Reasoning Ablation",
    "temperature": "Temperature Ablation",
}


def _bare(model: str) -> str:
    return normalize_model_id(model)


#: Bare-id model lists per ablation type, taken from
#: :mod:`fraudbench.utils.globals` and normalised so they line up with
#: ``model_id`` in ``obs_data.parquet``.
ABLATION_MODELS: dict[str, list[str]] = {
    "reasoning":   [_bare(m) for m in REASONING_ABLATION_MODELS],
    "temperature": [_bare(m) for m in TEMPERATURE_ABLATION_MODELS],
}


VALID_ABLATION_TYPES: tuple[str, ...] = tuple(ABLATION_RUN_CATEGORIES.keys())


#: Per-evaluation-cell key used to align main runs with ablation runs.
#:
#: Each ablation run a model performed corresponds to a specific cell of the
#: 5x5 (instructional × contextual) experimental design on a specific
#: scenario in a specific evaluation setting. The ablations were not run on
#: the full sample: e.g. the temperature ablation evaluated only ~53
#: scenarios and the reasoning ablation covered different
#: (summary_id, condition, setting) cells across its four models. To make
#: the main vs ablation comparison strictly within-cell, we restrict the
#: main runs to exactly the (model, summary_id, condition, setting)
#: cells that the ablation actually ran on.
ABLATION_KEY_COLS: list[str] = [
    "model_id",
    "summary_id",
    "instructional_group",
    "contextual_condition",
    "setting",
]

# Keep the parquet projection close to the consumer so additions to the
# analysis cannot silently pull the multi-gigabyte completion/COT columns
# back into memory.
_NEEDED_COLUMNS: list[str] = [
    "model",
    "summary_id",
    "setting",
    "run_category",
    "has_sampling_error",
    *INSTRUCTIONAL_CONDITIONS,
    *CONTEXTUAL_CONDITIONS,
    *(config["column"] for config in METRIC_CONFIGS.values()),
]


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------


def _load_obs(input_path: str, setting: str) -> pd.DataFrame:
    """Load obs_data, restrict to AI settings, drop error rows, coerce NaNs.

    Crucially this does *not* filter on ``run_category``: callers want both
    ``main`` and ``<ablation>_ablation`` rows. We also derive the
    ``instructional_group`` / ``contextual_condition`` columns up front so
    callers can use them as merge keys when matching main against ablation.
    """
    df = load_data_from_source(input_path, needed_columns=_NEEDED_COLUMNS)

    if "setting" in df.columns:
        df = df[df["setting"].isin(list(AI_SETTINGS))].copy()
    df = filter_setting(df, setting=setting, ai_settings=AI_SETTINGS)
    df = drop_errors(df)
    df = coerce_none_to_nan(df, [cfg["column"] for cfg in METRIC_CONFIGS.values()])
    df = add_condition_columns(df)
    return df


def _restrict_main_to_ablation_cells(
    main_df: pd.DataFrame,
    abl_df: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only main rows whose ablation-key tuple appears in the ablation set.

    The ablations were run on a strict subset of the
    (model_id × summary_id × instructional × contextual × setting) cells
    used by the main runs. To compare like-with-like we filter the main
    rows to the same cells. Replicates are kept (the rate aggregator
    will average across replicates as usual).
    """
    if abl_df.empty:
        return main_df.iloc[0:0].copy()
    keys = abl_df[ABLATION_KEY_COLS].drop_duplicates()
    n_before = len(main_df)
    out = main_df.merge(keys, on=ABLATION_KEY_COLS, how="inner")
    print(
        f"Restricted main rows to ablation cells: {n_before} -> {len(out)} "
        f"({len(keys)} unique cells)"
    )
    return out


def _split_main_vs_ablation(
    df: pd.DataFrame,
    ablation_type: str,
    models: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return ``(main_df_restricted, ablation_df, models_present)``.

    ``main_df_restricted`` contains only the main-run rows whose
    (model, summary_id, condition, setting) cells also appear in the
    ablation runs (see :data:`ABLATION_KEY_COLS`).

    ``models_present`` keeps only the models that have rows in *both* the
    main and ablation slices, ordered per :data:`MODEL_ORDER`.
    """
    if "run_category" not in df.columns:
        raise ValueError(
            "_split_main_vs_ablation: 'run_category' column missing from data."
        )
    run_category = ABLATION_RUN_CATEGORIES[ablation_type]
    bare_ids = set(models)

    in_models = df["model_id"].isin(bare_ids)
    main_df = df[in_models & (df["run_category"] == "main")].copy()
    abl_df = df[in_models & (df["run_category"] == run_category)].copy()

    print(
        f"Pre-restriction: {len(main_df)} main rows, {len(abl_df)} {run_category} rows"
    )
    main_df = _restrict_main_to_ablation_cells(main_df, abl_df)

    main_models = set(main_df["model_id"].unique())
    abl_models = set(abl_df["model_id"].unique())
    missing_main = bare_ids - main_models
    missing_abl = bare_ids - abl_models
    if missing_main:
        print(f"Warning: {len(missing_main)} ablation model(s) have no matching main-runs cells: "
              f"{sorted(missing_main)}")
    if missing_abl:
        print(f"Warning: {len(missing_abl)} ablation model(s) have no {run_category} data: "
              f"{sorted(missing_abl)}")

    common = main_models & abl_models
    ordered = [m for m in MODEL_ORDER if m in common]
    for m in sorted(common):
        if m not in ordered:
            ordered.append(m)

    print(f"Found {len(ordered)} models with both main and {run_category} data: {ordered}")
    return main_df, abl_df, ordered


def _rate(df: pd.DataFrame, metric_col: str) -> float:
    """Compute a single rate for *df* on *metric_col*, NaN-safe."""
    if metric_col not in df.columns or len(df) == 0:
        return 0.0
    stats = compute_metric_rate(df, metric_col)
    return float(stats["rate"].values[0])


def _wrap_model_label(label: str) -> str:
    """Insert a single newline so long model labels fit under narrow groups.

    Prefers to break before a parenthesised qualifier (e.g.
    ``"Grok 4.1 Fast (Reasoning)"`` → ``"Grok 4.1 Fast\\n(Reasoning)"``);
    otherwise breaks at the space closest to the label's midpoint. Short
    labels (no spaces, or already short) are returned unchanged.
    """
    paren_idx = label.find(" (")
    if paren_idx > 0:
        return label[:paren_idx] + "\n" + label[paren_idx + 1:]
    spaces = [i for i, c in enumerate(label) if c == " "]
    if not spaces or len(label) <= 12:
        return label
    mid = len(label) / 2
    best = min(spaces, key=lambda i: abs(i - mid))
    return label[:best] + "\n" + label[best + 1:]


def _rates_per_model(
    df: pd.DataFrame,
    metric_col: str,
    models: Sequence[str],
) -> list[float]:
    """Compute the metric rate for each model, in ``models`` order."""
    return [_rate(df[df["model_id"] == m], metric_col) for m in models]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


_HATCH_MAIN: str = ""
_HATCH_ABLATION: str = "///"

# Keep both axes' tick labels and axis/group labels visually consistent in
# the dense model × setting ablation figures.  These deliberately override
# the broader paper defaults, whose y-axis typography is too large here.
_TICK_LABEL_FONTSIZE: int = 14
_AXIS_LABEL_FONTSIZE: int = 16
_LEGEND_FONTSIZE: int = 14


SETTING_DISPLAY: dict[str, str] = {
    "closed":  "Binary",
    "open":    "Open",
    "agentic": "Agentic",
}
"""Inner-axis labels for AI settings inside the per-metric panels."""

OVERALL_LABEL: str = "Overall"


def _draw_metric_panel(
    ax,
    *,
    metric: str,
    main_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    models: Sequence[str],
    settings: Sequence[str],
    ablation_label: str,
) -> None:
    """One metric panel: outer = model, inner = setting (+ Overall), 2 bars per inner.

    Values are flattened row-major (model-first, setting-second) per the
    contract of :func:`nested_grouped_bars`.
    """
    cfg = METRIC_CONFIGS[metric]
    col = cfg["column"]
    color = cfg["color"]

    main_values: list[float] = []
    main_errors: list[float] = []
    abl_values: list[float] = []
    abl_errors: list[float] = []
    for m in models:
        m_main = main_df[main_df["model_id"] == m]
        m_abl = ablation_df[ablation_df["model_id"] == m]
        for s in settings:
            r, e = _rate_and_halfwidth(m_main[m_main["setting"] == s], col)
            main_values.append(r); main_errors.append(e)
            r, e = _rate_and_halfwidth(m_abl[m_abl["setting"] == s], col)
            abl_values.append(r); abl_errors.append(e)
        r, e = _rate_and_halfwidth(m_main, col)
        main_values.append(r); main_errors.append(e)
        r, e = _rate_and_halfwidth(m_abl, col)
        abl_values.append(r); abl_errors.append(e)

    inner_labels = [SETTING_DISPLAY.get(s, s) for s in settings] + [OVERALL_LABEL]
    outer_labels = [_wrap_model_label(get_model_display_name(m)) for m in models]

    series = [
        BarSeries(values=main_values, errors=main_errors, color=color, label="Main"),
        BarSeries(values=abl_values, errors=abl_errors, color=color,
                  label=ablation_label, hatch=_HATCH_ABLATION),
    ]

    nested_grouped_bars(
        ax,
        outer_labels=outer_labels,
        inner_labels=inner_labels,
        series=series,
        bar_width=0.18,
        group_spacing=0.55,
        inner_gap=0.04,
        outer_label_y=-0.34,
        outer_label_fontsize=_AXIS_LABEL_FONTSIZE,
        inner_xtick_rotation=45,
        inner_xtick_ha="right",
        inner_xtick_fontsize=_TICK_LABEL_FONTSIZE,
        ylabel=cfg["y_label"],
        show_legend=False,
    )
    ax.tick_params(axis="y", labelsize=_TICK_LABEL_FONTSIZE)
    ax.yaxis.label.set_size(_AXIS_LABEL_FONTSIZE)


def plot_main_vs_ablation(
    df: pd.DataFrame,
    *,
    ablation_type: str,
    output_dir: Optional[Path] = None,
    setting: str = "all",
) -> dict[str, tuple]:
    """Render one standalone figure per metric (nested model × setting bars).

    For each metric in :data:`METRIC_ORDER` a separate figure is created
    that groups bars by model (outer), then by setting + Overall (inner),
    with two bars per inner category (Main solid, ablation hatched).

    Args:
        df: ``obs_data`` frame with main and ablation rows.
        ablation_type: ``"reasoning"`` or ``"temperature"``.
        output_dir: If provided, each per-metric figure is saved as
            ``main_vs_{ablation_type}_ablation_{file_suffix}_{setting_file_suffix(setting)}.pdf``
            inside this directory (``closed`` maps to ``binary`` in the
            filename suffix).
        setting: AI setting label used purely for filename disambiguation;
            data filtering should already have been applied upstream by
            :func:`_load_obs`.

    Returns:
        ``{metric: (fig, ax)}`` for the figures that were produced
        (empty dict if no overlapping models were available).
    """
    if ablation_type not in VALID_ABLATION_TYPES:
        raise ValueError(
            f"Unknown ablation_type {ablation_type!r}. "
            f"Must be one of {list(VALID_ABLATION_TYPES)}."
        )
    setup_plot_style()

    models = ABLATION_MODELS[ablation_type]
    main_df, abl_df, models_present = _split_main_vs_ablation(df, ablation_type, models)

    if not models_present:
        print(f"No overlapping models for ablation {ablation_type!r}; nothing to plot.")
        return {}

    ablation_label = ABLATION_DISPLAY_NAMES[ablation_type]
    settings_present = [s for s in AI_SETTINGS if s in abl_df["setting"].unique()]

    # Per-figure width grows with the number of models because each outer
    # group hosts ``n_settings + 1`` inner categories times two bars. The
    # wrapped two-line model labels need vertical room below the axes,
    # which ``tight_layout`` handles per-figure.
    n_models = len(models_present)
    fig_width = max(8.0, 4.0 + 1.6 * n_models)
    fig_height = 7.0 if n_models >= 6 else 6.0

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    figures: dict[str, tuple] = {}
    for metric in METRIC_ORDER:
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        _draw_metric_panel(
            ax,
            metric=metric,
            main_df=main_df,
            ablation_df=abl_df,
            models=models_present,
            settings=settings_present,
            ablation_label=ablation_label,
        )

        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor="#999999",
                          edgecolor="black", linewidth=0.5),
            plt.Rectangle((0, 0), 1, 1, facecolor="#999999",
                          edgecolor="black", linewidth=0.5, hatch=_HATCH_ABLATION),
        ]
        fig.legend(
            handles, ["Main", ablation_label],
            **above_axes_legend_kw(ncol=2),
            fontsize=_LEGEND_FONTSIZE,
        )

        fig.tight_layout()

        if output_dir is not None:
            file_suffix = METRIC_CONFIGS[metric]["file_suffix"]
            output_path = (
                output_dir
                / f"main_vs_{ablation_type}_ablation_{file_suffix}_{setting_file_suffix(setting)}.pdf"
            )
            save_figure(fig, output_path, bbox_inches="tight")
            print(f"Saved plot to {output_path}")

        figures[metric] = (fig, ax)

    return figures


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def _print_summary(
    main_df: pd.DataFrame,
    abl_df: pd.DataFrame,
    models: Sequence[str],
    ablation_type: str,
) -> None:
    """Print per-model rate comparisons for each metric."""
    print("\n" + "=" * 70)
    print(f"DATA SUMMARY: main vs {ablation_type}_ablation (matched cells)")
    print("=" * 70)
    print(f"Main rows (restricted to ablation cells): {len(main_df)}")
    print(f"Ablation rows:                            {len(abl_df)}")
    print(f"Models compared ({len(models)}): {[get_model_display_name(m) for m in models]}")

    print("\n--- Per-model evaluation-cell coverage ---")
    for m in models:
        main_subset = main_df[main_df["model_id"] == m]
        abl_subset = abl_df[abl_df["model_id"] == m]
        n_cells_main = main_subset.drop_duplicates(subset=ABLATION_KEY_COLS).shape[0]
        n_cells_abl = abl_subset.drop_duplicates(subset=ABLATION_KEY_COLS).shape[0]
        n_summaries = abl_subset["summary_id"].nunique()
        print(f"  {get_model_display_name(m):<32s}  "
              f"main_rows={len(main_subset):>5d}  abl_rows={len(abl_subset):>5d}  "
              f"cells={n_cells_abl:>4d}  summaries={n_summaries:>3d}  "
              f"main_cells_matched={n_cells_main:>4d}")

    for metric in METRIC_ORDER:
        cfg = METRIC_CONFIGS[metric]
        col = cfg["column"]
        if col not in main_df.columns and col not in abl_df.columns:
            continue
        print(f"\n--- {cfg['display_name']} ({col}) ---")
        for m in models:
            main_subset = main_df[main_df["model_id"] == m]
            abl_subset = abl_df[abl_df["model_id"] == m]
            main_rate = main_subset[col].dropna().mean() if col in main_subset.columns else float("nan")
            abl_rate = abl_subset[col].dropna().mean() if col in abl_subset.columns else float("nan")
            label = get_model_display_name(m)
            print(f"  {label:<32s}  main={main_rate:.3f}  "
                  f"abl={abl_rate:.3f}  Δ={abl_rate - main_rate:+.3f}")


# ---------------------------------------------------------------------------
# Run entry points
# ---------------------------------------------------------------------------


def run_one(
    ablation: str,
    setting: str = "all",
    input_path: str = DEFAULT_INPUT_PATH,
    output_dir: Optional[Path] = None,
    show: bool = False,
    *,
    prepared_df: Optional[pd.DataFrame] = None,
    close_figures: Optional[bool] = None,
) -> None:
    """Generate the main-vs-ablation plot for a single ablation type."""
    if ablation not in VALID_ABLATION_TYPES:
        raise ValueError(
            f"Unknown ablation {ablation!r}. Must be one of {list(VALID_ABLATION_TYPES)}."
        )
    if setting not in VALID_SETTINGS_WITH_ALL:
        raise ValueError(
            f"Invalid setting {setting!r}. Must be one of {VALID_SETTINGS_WITH_ALL}."
        )

    if output_dir is None:
        output_dir = Path("./paper/figures")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = (
        prepared_df
        if prepared_df is not None
        else _load_obs(input_path, setting=setting)
    )

    main_df, abl_df, models_present = _split_main_vs_ablation(
        df, ablation, ABLATION_MODELS[ablation],
    )
    _print_summary(main_df, abl_df, models_present, ablation)

    print("\n" + "=" * 70)
    print(f"GENERATING PLOTS: main vs {ablation}_ablation (one figure per metric)")
    print("=" * 70)

    figures = plot_main_vs_ablation(
        df,
        ablation_type=ablation,
        output_dir=output_dir,
        setting=setting,
    )

    print(f"\nDone! Plots saved to: {output_dir}")
    if close_figures is None:
        close_figures = not show
    if close_figures:
        for fig, _ax in figures.values():
            plt.close(fig)
    if show:
        plt.show()


def run(
    ablation: str = "all",
    setting: str = "all",
    input_path: str = DEFAULT_INPUT_PATH,
    output_dir: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Generate one ablation comparison figure per metric (for one or both ablations).

    Args:
        ablation: One of ``"reasoning"``, ``"temperature"``, or ``"all"``.
        setting: AI setting to restrict to (``"closed"`` / ``"open"`` /
            ``"agentic"`` / ``"all"``).
        input_path: Path to ``obs_data.parquet`` (or sibling parquet).
        output_dir: Directory to write the per-metric PNGs to.
        show: If True, also display the figures interactively.
    """
    if ablation == "all":
        df = _load_obs(input_path, setting=setting)
        for abl in VALID_ABLATION_TYPES:
            run_one(
                ablation=abl,
                setting=setting,
                input_path=input_path,
                output_dir=output_dir,
                show=False,
                prepared_df=df,
                close_figures=not show,
            )
        if show:
            plt.show()
    else:
        run_one(
            ablation=ablation,
            setting=setting,
            input_path=input_path,
            output_dir=output_dir,
            show=show,
            close_figures=not show,
        )
