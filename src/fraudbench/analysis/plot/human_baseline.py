"""Human baseline vs AI plots, backing ``plot_human_baseline.py``.

Fraud risk metric: proportion of samples where fraud=True (lower = better).
Every figure reports the metric on two denominators: all graded responses,
and the *non-refusing* subset only (``score_refusal == 0``). Refusals are a
large share of AI responses and a negligible share of human ones, so the
refusal-excluded bars separate "how often does the respondent commit fraud"
from "how often does the respondent engage at all".

Each bar carries a 95% Wilson binomial CI on the empirical proportion as
a symmetric error bar. Inferential output (the human-vs-AI baseline
contrast and demographic effects) lives in
:file:`paper/tables/regressions/regression_coefs_human.tex`.
"""
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..style import (
    save_figure,
    AI_HUMAN_COLORS,
    HUMAN_BASELINE_CONDITION_COLORS,
    setup_plot_style,
)
from ..charts import (
    BarSeries,
    HatchedBarSeries,
    above_axes_legend_kw,
    bars_with_hatch_overlay,
    build_hatch_legend,
    grouped_bars,
)
from ..common import (
    DEFAULT_INPUT_PATH,
    drop_errors,
    filter_human_baseline,
    get_model_display_name,
    load_data_from_source,
)
from ..uncertainty import wilson_halfwidths


# Pulled from the dedicated AI-vs-Human palette in style.py so a "Human"
# bar is visually distinct from the "Fraud" bar in every other figure
# (which uses ``METRIC_COLORS["fraud"]``).
HUMAN_COLOR = AI_HUMAN_COLORS["human"]
AI_COLOR = AI_HUMAN_COLORS["ai"]

CONDITION_LABELS = {
    "control": "Control",
    "peer_pressure_low": "Peer Pressure\n(Low)",
    "peer_pressure_high": "Peer Pressure\n(High)",
    "negative_incentive_low": "Negative Incentive\n(Low)",
    "negative_incentive_high": "Negative Incentive\n(High)",
}

CONDITION_ORDER = [
    "control", "peer_pressure_low", "peer_pressure_high",
    "negative_incentive_low", "negative_incentive_high",
]

# Pseudo-model ids for the two aggregate x-categories. They share the
# ``model_id`` column with the real models so a single frame can drive
# both per-model figures.
HUMAN_LABEL = "Human"
AI_AVERAGE_LABEL = "AI (Average)"

# The two denominators every bar can be computed on. ``non_refusal``
# drops rows with ``score_refusal == 1`` (and rows where refusal could
# not be graded) from both numerator and denominator.
SUBSET_ALL = "all"
SUBSET_NON_REFUSAL = "non_refusal"
SUBSET_ORDER = [SUBSET_ALL, SUBSET_NON_REFUSAL]
SUBSET_LABELS = {
    SUBSET_ALL: "All Responses",
    SUBSET_NON_REFUSAL: "Excluding Refusals",
}
# Empty pattern for the primary subset matches ``SETTING_HATCHES``: the
# reference bars stay solid and only the derived subset is hatched.
SUBSET_HATCHES = {SUBSET_ALL: "", SUBSET_NON_REFUSAL: "///"}
_SUBSET_COLUMNS = {
    SUBSET_ALL: ("fraud_risk", "successes", "n_samples"),
    SUBSET_NON_REFUSAL: (
        "fraud_risk_non_refusal", "successes_non_refusal", "n_non_refusal",
    ),
}

NON_REFUSAL_FRAUD_COLUMN = "score_fraud_non_refusal"

_CONDITION_COLUMNS = CONDITION_ORDER[1:]
_NEEDED_COLUMNS = [
    "model",
    "setting",
    "has_sampling_error",
    "has_error",
    "score_fraud",
    "score_refusal",
    *_CONDITION_COLUMNS,
]


def add_non_refusal_fraud_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``score_fraud_non_refusal``: ``score_fraud`` masked to non-refusals.

    Rows that are refusals *and* rows whose refusal status could not be
    graded become ``NaN``, so ``count``/``sum``/``mean`` aggregations
    exclude them from the refusal-adjusted denominator automatically.
    """
    if "score_refusal" not in df.columns:
        raise ValueError(
            "Input data is missing the 'score_refusal' column, which the "
            "refusal-excluded bars need. Regenerate the obs-data file with "
            "`uv run analysis/process.py`."
        )
    df = df.copy()
    refusal = pd.to_numeric(df["score_refusal"], errors="coerce")
    df[NON_REFUSAL_FRAUD_COLUMN] = df["score_fraud"].where(refusal == 0)
    return df


def load_and_prepare_data(
    input_path: str = DEFAULT_INPUT_PATH,
) -> pd.DataFrame:
    """Load data and filter to ``setting='human_baseline'`` with ``condition`` column."""
    df = load_data_from_source(input_path, needed_columns=_NEEDED_COLUMNS)

    df = filter_human_baseline(df)

    df = drop_errors(df)

    df = df.copy()
    df["condition"] = np.select(
        [
            df[column].fillna(False).astype(bool).to_numpy()
            if column in df.columns
            else np.zeros(len(df), dtype=bool)
            for column in _CONDITION_COLUMNS
        ],
        _CONDITION_COLUMNS,
        default="control",
    )
    df["score_fraud"] = df["score_fraud"].where(df["score_fraud"].notna(), np.nan)
    df["is_human"] = df["model"] == "human"
    df = df[df["condition"].isin(CONDITION_ORDER)]
    df = df.dropna(subset=["score_fraud"])

    return add_non_refusal_fraud_column(df)


def compute_fraud_risk(df: pd.DataFrame, group_cols) -> pd.DataFrame:
    """Compute fraud risk (mean of ``score_fraud``) per group.

    Returns the rate, sample count and success count on both denominators
    (all responses, and non-refusals only) so downstream callers can
    compute Wilson CIs without re-aggregating. Pass an empty ``group_cols``
    to aggregate the whole frame into a single row.
    """
    if NON_REFUSAL_FRAUD_COLUMN not in df.columns:
        df = add_non_refusal_fraud_column(df)

    aggregations = {
        "fraud_risk": ("score_fraud", "mean"),
        "n_samples": ("score_fraud", "count"),
        "successes": ("score_fraud", "sum"),
        "fraud_risk_non_refusal": (NON_REFUSAL_FRAUD_COLUMN, "mean"),
        "n_non_refusal": (NON_REFUSAL_FRAUD_COLUMN, "count"),
        "successes_non_refusal": (NON_REFUSAL_FRAUD_COLUMN, "sum"),
    }

    group_cols = list(group_cols)
    if not group_cols:
        return pd.DataFrame([
            {name: df[column].agg(func) for name, (column, func) in aggregations.items()}
        ])
    return df.groupby(group_cols).agg(**aggregations).reset_index()


def _rate_stats(
    risk: pd.DataFrame,
    key_cols: Sequence[str],
    keys: Sequence,
    *,
    subset: str = SUBSET_ALL,
) -> dict:
    """Look up ``keys`` in a :func:`compute_fraud_risk` frame for one denominator.

    Returns ``{"rates": [...], "ci95s": [...]}`` in ``keys`` order, ready for
    either bar primitive. Keys absent from ``risk`` (and groups with an empty
    denominator) render as a zero-height bar with no error bar.
    """
    rate_column, success_column, n_column = _SUBSET_COLUMNS[subset]
    records = risk.set_index(list(key_cols)).to_dict("index")

    rates: list[float] = []
    successes: list[float] = []
    ns: list[int] = []
    for key in keys:
        row = records.get(key)
        if row is None:
            rates.append(0.0)
            successes.append(0.0)
            ns.append(0)
            continue
        rate = row[rate_column]
        success = row[success_column]
        rates.append(0.0 if pd.isna(rate) else float(rate))
        successes.append(0.0 if pd.isna(success) else float(success))
        ns.append(int(row[n_column]))

    halfwidths = wilson_halfwidths(np.asarray(successes), np.asarray(ns))
    return {
        "rates": rates,
        "ci95s": np.where(np.isnan(halfwidths), 0.0, halfwidths).tolist(),
    }


def _risk_by_model(
    df: pd.DataFrame,
    extra_group_cols: Sequence[str] = (),
) -> tuple[pd.DataFrame, list[str]]:
    """Fraud risk keyed by ``model_id``, with ``Human`` and ``AI (Average)`` rows.

    ``AI (Average)`` pools every AI response (so it is response-weighted, not
    a mean of per-model rates), matching the aggregate bar in
    :func:`plot_human_vs_ai_average`. Returns the stacked frame plus the
    x-axis order: human, the AI aggregate, then each model id alphabetically.
    """
    extra_group_cols = list(extra_group_cols)
    ai_data = df[~df["is_human"]]

    human_risk = compute_fraud_risk(df[df["is_human"]], extra_group_cols)
    human_risk["model_id"] = HUMAN_LABEL

    ai_risk = compute_fraud_risk(ai_data, extra_group_cols)
    ai_risk["model_id"] = AI_AVERAGE_LABEL

    per_model_risk = compute_fraud_risk(ai_data, ["model_id", *extra_group_cols])

    order = [HUMAN_LABEL, AI_AVERAGE_LABEL, *sorted(ai_data["model_id"].unique())]
    risk = pd.concat([human_risk, ai_risk, per_model_risk], ignore_index=True)
    return risk, order


def _x_label(model_id: str) -> str:
    """Display name for an x-category, passing the two aggregates through."""
    if model_id in (HUMAN_LABEL, AI_AVERAGE_LABEL):
        return model_id
    return get_model_display_name(model_id)


def plot_human_vs_ai_average(df: pd.DataFrame, output_path: Optional[Path] = None):
    """Grouped bars: Human vs AI-average fraud risk per condition.

    Each source contributes two bars per condition: all responses (solid)
    and non-refusals only (hatched).
    """
    setup_plot_style()

    human_risk = compute_fraud_risk(df[df["is_human"]], ["condition"])
    ai_risk = compute_fraud_risk(df[~df["is_human"]], ["condition"])

    def by_subset(risk: pd.DataFrame) -> dict:
        return {
            subset: _rate_stats(risk, ["condition"], CONDITION_ORDER, subset=subset)
            for subset in SUBSET_ORDER
        }

    # Wider-and-shorter aspect for paper wrapfigure (0.52\textwidth, 17 lines).
    # Lower headroom keeps the legend snug and trims dead space above the bars.
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_with_hatch_overlay(
        ax,
        x_labels=[CONDITION_LABELS[c] for c in CONDITION_ORDER],
        series=[
            HatchedBarSeries(label="Human", color=HUMAN_COLOR,
                             by_hatch=by_subset(human_risk)),
            HatchedBarSeries(label=AI_AVERAGE_LABEL, color=AI_COLOR,
                             by_hatch=by_subset(ai_risk)),
        ],
        hatch_order=SUBSET_ORDER,
        hatch_patterns=SUBSET_HATCHES,
        hatch_display_labels=SUBSET_LABELS,
        bar_width=0.2,
        bar_alpha=0.9,
        xtick_rotation=30,
        xtick_ha="right",
        ylabel="Fraud Risk (Proportion)",
        xlabel="Experimental Condition",
        ylim_headroom=0.10,
    )
    plt.tight_layout()

    if output_path:
        save_figure(plt.gcf(), output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig, ax


def plot_fraud_vs_non_refusal_by_model(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
):
    """Grouped bars: per-respondent fraud risk on both denominators.

    One group per x-category (Human, the AI average, then each model),
    pooled over experimental conditions, with a solid all-responses bar and
    a hatched non-refusals-only bar. Bar color encodes human vs AI.
    """
    setup_plot_style()

    risk, models = _risk_by_model(df)
    bar_colors = [HUMAN_COLOR if m == HUMAN_LABEL else AI_COLOR for m in models]
    stats = {
        subset: _rate_stats(risk, ["model_id"], models, subset=subset)
        for subset in SUBSET_ORDER
    }

    fig, ax = plt.subplots(figsize=(16, 8))
    grouped_bars(
        ax,
        x_labels=[_x_label(m) for m in models],
        series=[
            BarSeries(
                values=stats[subset]["rates"],
                errors=stats[subset]["ci95s"],
                color=bar_colors,
                hatch=SUBSET_HATCHES[subset] or None,
            )
            for subset in SUBSET_ORDER
        ],
        bar_width=0.4,
        bar_alpha=0.85,
        show_legend=False,
        xtick_rotation=30,
        xtick_ha="right",
        ylabel="Fraud Risk (Proportion)",
        xlabel="Model",
    )
    # ``grouped_bars`` only outlines hatched series; outline the solid ones
    # too so this figure matches the edged bars in ``bars_with_hatch_overlay``.
    for patch in ax.patches:
        patch.set_edgecolor("black")
        patch.set_linewidth(0.5)
    handles, labels = build_hatch_legend(
        [("Human", HUMAN_COLOR), ("AI", AI_COLOR)],
        [(SUBSET_LABELS[s], SUBSET_HATCHES[s]) for s in SUBSET_ORDER],
    )
    ax.legend(handles, labels, **above_axes_legend_kw(ncol=4))
    # Separate the two aggregate categories from the individual models.
    for boundary in (0.5, 1.5):
        ax.axvline(x=boundary, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if output_path:
        save_figure(plt.gcf(), output_path)
        print(f"Saved plot to {output_path}")

    return fig, ax


def plot_human_vs_each_model(df: pd.DataFrame, output_path: Optional[Path] = None):
    """Grouped bars: per-model fraud risk colored by condition.

    X-categories are Human, the AI average, then each model.
    """
    setup_plot_style()

    risk, models = _risk_by_model(df, ["condition"])

    series = []
    for cond in CONDITION_ORDER:
        stats = _rate_stats(
            risk, ["model_id", "condition"], [(m, cond) for m in models],
        )
        series.append(BarSeries(
            values=stats["rates"],
            errors=stats["ci95s"],
            color=HUMAN_BASELINE_CONDITION_COLORS[cond],
            label=CONDITION_LABELS[cond].replace("\n", " "),
        ))

    fig, ax = plt.subplots(figsize=(16, 8))
    grouped_bars(
        ax,
        x_labels=[_x_label(m) for m in models],
        series=series,
        bar_width=0.15,
        bar_alpha=0.85,
        xtick_rotation=30,
        xtick_ha="right",
        ylabel="Fraud Risk (Proportion)",
        xlabel="Model",
    )
    # Separate the two aggregate categories from the individual models.
    for boundary in (0.5, 1.5):
        ax.axvline(x=boundary, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if output_path:
        save_figure(plt.gcf(), output_path)
        print(f"Saved plot to {output_path}")

    return fig, ax


def print_summary_stats(df: pd.DataFrame):
    """Print human-baseline summary statistics."""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    print(f"\nTotal samples: {len(df)}")
    print(f"Human samples: {df['is_human'].sum()}")
    print(f"AI samples: {(~df['is_human']).sum()}")

    print("\n--- Samples per Condition ---")
    for cond in CONDITION_ORDER:
        human_n = len(df[(df["condition"] == cond) & df["is_human"]])
        ai_n = len(df[(df["condition"] == cond) & ~df["is_human"]])
        print(f"{cond}: Human={human_n}, AI={ai_n}")

    ai_models = df[~df["is_human"]]["model_id"].unique()
    print(f"\n--- AI Models ({len(ai_models)}) ---")
    for m in sorted(ai_models):
        n = len(df[df["model_id"] == m])
        print(f"  {m}: {n} samples")

    print("\n--- Fraud Risk by Condition (all responses) ---")
    for cond in CONDITION_ORDER:
        human_risk = df[(df["condition"] == cond) & df["is_human"]]["score_fraud"].mean()
        ai_risk = df[(df["condition"] == cond) & ~df["is_human"]]["score_fraud"].mean()
        print(f"{cond}: Human={human_risk:.3f}, AI={ai_risk:.3f}")

    print("\n--- Fraud Risk by Condition (excluding refusals) ---")
    for cond in CONDITION_ORDER:
        subset = df[df["condition"] == cond]
        human_risk = subset[subset["is_human"]][NON_REFUSAL_FRAUD_COLUMN].mean()
        ai_risk = subset[~subset["is_human"]][NON_REFUSAL_FRAUD_COLUMN].mean()
        print(f"{cond}: Human={human_risk:.3f}, AI={ai_risk:.3f}")


def run(
    input_path: str = DEFAULT_INPUT_PATH,
    show: bool = False,
    output_dir: Optional[Path] = None,
):
    """Load human-baseline data and generate all three plots."""
    if output_dir is None:
        output_dir = Path("./paper/figures")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data(input_path=input_path)

    print_summary_stats(df)

    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    print("\nPlot 1: Human vs AI Average...")
    fig, _ax = plot_human_vs_ai_average(df, output_dir / "human_vs_ai_average.pdf")
    if not show:
        plt.close(fig)

    print("\nPlot 2: Fraud Risk With and Without Refusals, by Model...")
    fig, _ax = plot_fraud_vs_non_refusal_by_model(
        df, output_dir / "fraud_vs_non_refusal_by_model.pdf",
    )
    if not show:
        plt.close(fig)

    print("\nPlot 3: Human vs Each Model...")
    fig, _ax = plot_human_vs_each_model(df, output_dir / "human_vs_each_model.pdf")
    if not show:
        plt.close(fig)

    print(f"\nDone! Plots saved to: {output_dir}")

    if show:
        plt.show()
