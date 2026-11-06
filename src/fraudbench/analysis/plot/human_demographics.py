"""Demographic plots for human-baseline survey respondents.

Backs ``analysis/human_baseline/plots.py``. Generates:

* Bar charts for Age bucket, Sex, Ethnicity, AI Usage (the public release)
* Raincloud plots for Age, Duration, and Response Length, plus bar charts for
  Employment Status and Work Experience, when run on the private full export
  (those columns are absent from the public file and are skipped with a warning)
"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from ..style import COLORS, save_figure, setup_plot_style
from ..charts import categorical_bar, raincloud


DEFAULT_INPUT = "./data/final/human_baseline/human_baseline_public.jsonl"
DEFAULT_OUTPUT_DIR = Path("./paper/figures")


DEMOGRAPHIC_COLUMNS = {
    "age_bucket": "Age",
    "Prolific_Age": "Age",
    "Prolific_Sex": "Sex",
    "Prolific_Ethnicity simplified": "Ethnicity",
    "Prolific_Employment status": "Employment Status",
    "Work Experience": "Work Experience",
}

INVALID_VALUES = {"CONSENT_REVOKED", "DATA_EXPIRED"}

WORK_EXPERIENCE_ORDER = [
    "Less than 1 year", "1 – 2 years", "3 – 5 years", "6 – 10 years",
    "11 – 15 years", "16 – 20 years", "16+ years", "More than 20 years",
]

EMPLOYMENT_STATUS_ORDER = [
    "Full-Time", "Part-Time", "Self-employed", "Unemployed (and job seeking)",
    "Not in paid work (e.g. homemaker', 'retired or disabled)",
    "Due to start a new job within the next month", "Other",
]

AGE_BUCKET_ORDER = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+", "missing"]

AI_USE_ORDER = [
    "Almost constantly", "Several times a day", "About once a day",
    "Several times a week", "Less often",
]

AXIS_LABEL_FONT_SIZE = 20
Y_TICK_FONT_SIZE = 20
NUMERIC_X_TICK_FONT_SIZE = 18
CATEGORICAL_X_TICK_FONT_SIZE = 12
STATS_ANNOTATION_FONT_SIZE = 12
CATEGORICAL_ANNOTATION_FONT_SIZE = 12


def load_and_prepare_data(input_path: str) -> pd.DataFrame:
    """Load processed survey data and return one row per unique respondent."""
    print(f"Loading data from {input_path}...")
    df = pd.read_json(input_path, lines=True)
    print(f"Loaded {len(df)} rows")

    demographic_cols = ["respondent_id", "Duration (in seconds)", "AI_use"] + list(DEMOGRAPHIC_COLUMNS.keys())
    existing_cols = [col for col in demographic_cols if col in df.columns]

    df_respondents = df[existing_cols].drop_duplicates(subset=["respondent_id"])
    print(f"Found {len(df_respondents)} unique respondents")

    initial_count = len(df_respondents)
    for col in DEMOGRAPHIC_COLUMNS.keys():
        if col in df_respondents.columns:
            mask = df_respondents[col].isin(INVALID_VALUES)
            n_invalid = mask.sum()
            if n_invalid > 0:
                print(f"  Filtering {n_invalid} respondents with invalid values in {col}")
                df_respondents = df_respondents[~mask]

    print(f"After filtering invalid values: {len(df_respondents)} respondents "
          f"(removed {initial_count - len(df_respondents)})")
    return df_respondents


def _increase_axis_font_sizes(
    ax,
    *,
    x_tick_fontsize: int,
    y_tick_fontsize: int = Y_TICK_FONT_SIZE,
    axis_label_fontsize: int = AXIS_LABEL_FONT_SIZE,
) -> None:
    """Apply the requested two-point increase to this figure's axis text."""
    ax.xaxis.label.set_fontsize(axis_label_fontsize)
    ax.yaxis.label.set_fontsize(axis_label_fontsize)
    ax.tick_params(axis="x", labelsize=x_tick_fontsize)
    ax.tick_params(axis="y", labelsize=y_tick_fontsize)


def _numeric_stats_text(values: pd.Series, unit: str, *, integer: bool = False,
                         count_noun: Optional[str] = None) -> str:
    """Format a standard stats text block for a numeric distribution."""
    int_fmt = ".0f"
    lead_fmt = ".1f"
    n_label = f"n = {len(values)}" + (f" {count_noun}" if count_noun else "")
    q25, q75 = values.quantile(0.25), values.quantile(0.75)
    q_fmt = int_fmt if integer else lead_fmt
    return (
        f"{n_label}\nMean: {values.mean():.1f} {unit}\n"
        f"Median: {values.median():{int_fmt if integer else lead_fmt}} {unit}\n"
        f"IQR: [{q25:{q_fmt}}, {q75:{q_fmt}}] {unit}\n"
        f"Range: [{values.min():{q_fmt}}, {values.max():{q_fmt}}] {unit}"
    )


def _raincloud_figure(values: pd.Series, *, x_label: str, title: str, stats_text: str,
                      figsize=(12, 6), output_path: Optional[Path] = None):
    """Compose a single-axis raincloud figure via the shared primitive."""
    fig, ax = plt.subplots(figsize=figsize)
    raincloud(
        ax,
        values,
        x_label=x_label,
        title=title,
        stats_text=stats_text,
        stats_fontsize=STATS_ANNOTATION_FONT_SIZE,
        color=COLORS["blue"],
    )
    _increase_axis_font_sizes(ax, x_tick_fontsize=NUMERIC_X_TICK_FONT_SIZE)
    plt.tight_layout()
    if output_path:
        save_figure(plt.gcf(), output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    return fig, ax


def plot_age_bucket_bar(df: pd.DataFrame, output_path: Optional[Path] = None):
    """Bar chart of the released age buckets (``age_bucket`` column)."""
    return _categorical_bar_figure(
        df, "age_bucket", "Age", order=AGE_BUCKET_ORDER, figsize=(10, 6),
        wrap_labels=False, output_path=output_path,
    )


def plot_age_raincloud(df: pd.DataFrame, output_path: Optional[Path] = None):
    """Raincloud plot of respondent ages (exact ages; private export only)."""
    setup_plot_style()
    if "Prolific_Age" not in df.columns:
        print("Warning: Prolific_Age column not found")
        return None, None

    ages = pd.to_numeric(df["Prolific_Age"], errors="coerce").dropna()
    return _raincloud_figure(
        ages,
        x_label="Age (years)",
        title=f"Age Distribution of Survey Respondents (n={len(ages)})",
        stats_text=_numeric_stats_text(ages, "years", integer=True),
        output_path=output_path,
    )


def _categorical_bar_figure(df: pd.DataFrame, column: str, display_name: str,
                            *, order: Optional[list] = None,
                            figsize=(12, 6), title: Optional[str] = None,
                            wrap_labels: bool = True,
                            output_path: Optional[Path] = None):
    """Shared wrapper: count a categorical column and render with the primitive."""
    setup_plot_style()
    if column not in df.columns:
        print(f"Warning: {column} column not found")
        return None, None

    values = df[column].dropna()
    value_counts = values.value_counts()
    if order:
        ordered_categories = [c for c in order if c in value_counts.index]
        for c in value_counts.index:
            if c not in ordered_categories:
                ordered_categories.append(c)
        value_counts = value_counts.reindex(ordered_categories)

    fig, ax = plt.subplots(figsize=figsize)
    categorical_bar(
        ax,
        categories=list(value_counts.index),
        counts=list(value_counts.values),
        total=len(values),
        color=COLORS["blue"],
        xlabel=display_name if title is None else None,
        ylabel="Percentage",
        title=title if title else f"{display_name} Distribution (n={len(values)})",
        use_proportions=True,
        show_count_in_pct_labels=True,
        pct_fontsize=CATEGORICAL_ANNOTATION_FONT_SIZE,
        xtick_fontsize=CATEGORICAL_X_TICK_FONT_SIZE,
        wrap_labels=wrap_labels,
    )
    _increase_axis_font_sizes(
        ax,
        x_tick_fontsize=CATEGORICAL_X_TICK_FONT_SIZE,
        y_tick_fontsize=CATEGORICAL_X_TICK_FONT_SIZE,
        axis_label_fontsize=CATEGORICAL_X_TICK_FONT_SIZE,
    )
    plt.tight_layout()
    if output_path:
        save_figure(plt.gcf(), output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    return fig, ax


def plot_categorical_bar(df: pd.DataFrame, column: str, display_name: str,
                         order: Optional[list] = None, output_path: Optional[Path] = None):
    """Bar chart for a categorical demographic column."""
    return _categorical_bar_figure(df, column, display_name, order=order,
                                    figsize=(12, 6), output_path=output_path)


def plot_duration_raincloud(df: pd.DataFrame, output_path: Optional[Path] = None):
    """Raincloud plot of survey duration (in minutes)."""
    setup_plot_style()
    if "Duration (in seconds)" not in df.columns:
        print("Warning: Duration (in seconds) column not found")
        return None, None

    duration = pd.to_numeric(df["Duration (in seconds)"], errors="coerce").dropna() / 60
    return _raincloud_figure(
        duration,
        x_label="Duration (minutes)",
        title=f"Survey Completion Time Distribution (n={len(duration)})",
        stats_text=_numeric_stats_text(duration, "min"),
        output_path=output_path,
    )


def plot_ai_use_bar(df: pd.DataFrame, output_path: Optional[Path] = None):
    """Bar chart of AI usage frequency."""
    return _categorical_bar_figure(
        df, "AI_use", "AI Usage Frequency",
        order=AI_USE_ORDER, figsize=(10, 6),
        title=(f"AI Usage Frequency Distribution "
               f"(n={df['AI_use'].dropna().shape[0] if 'AI_use' in df.columns else 0})"),
        wrap_labels=False,
        output_path=output_path,
    )


def plot_response_length_raincloud(df_full: pd.DataFrame, output_path: Optional[Path] = None):
    """Raincloud plot of response length (words). Uses the *full* dataframe."""
    setup_plot_style()
    if "response" not in df_full.columns:
        print("Warning: response column not found")
        return None, None

    responses = df_full["response"].dropna()
    word_counts = responses.apply(lambda x: len(str(x).split()))
    return _raincloud_figure(
        word_counts,
        x_label="Response Length (words)",
        title=f"Response Length Distribution (n={len(word_counts)} responses)",
        stats_text=_numeric_stats_text(word_counts, "words", integer=True, count_noun="responses"),
        output_path=output_path,
    )


def print_summary(df: pd.DataFrame):
    """Print demographic summary statistics."""
    print("\n" + "=" * 60)
    print("DEMOGRAPHIC SUMMARY")
    print("=" * 60)
    print(f"\nTotal respondents: {len(df)}")

    if "Duration (in seconds)" in df.columns:
        duration = pd.to_numeric(df["Duration (in seconds)"], errors="coerce").dropna()
        print("\nDuration (minutes):")
        print(f"  Mean: {duration.mean() / 60:.1f}, Median: {duration.median() / 60:.1f}")
        print(f"  Min: {duration.min() / 60:.1f}, Max: {duration.max() / 60:.1f}")
        print(f"  IQR: [{duration.quantile(0.25) / 60:.1f}, {duration.quantile(0.75) / 60:.1f}]")

    for col, display_name in DEMOGRAPHIC_COLUMNS.items():
        if col not in df.columns:
            continue
        values = df[col].dropna()
        print(f"\n{display_name} (n={len(values)}):")
        if col == "Prolific_Age":
            ages = pd.to_numeric(values, errors="coerce").dropna()
            print(f"  Mean: {ages.mean():.1f}, Median: {ages.median():.1f}")
            print(f"  Min: {ages.min():.0f}, Max: {ages.max():.0f}")
        else:
            value_counts = values.value_counts()
            for cat, count in value_counts.items():
                pct = count / len(values) * 100
                print(f"  {cat}: {count} ({pct:.1f}%)")


def run(
    input_path: str = DEFAULT_INPUT,
    output_dir: Optional[Path] = None,
    show: bool = False,
):
    """Generate all demographic plots (PNG only)."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data(input_path)
    df_full = pd.read_json(input_path, lines=True)

    print_summary(df)

    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    print("\nPlot 1: Age distribution...")
    if "Prolific_Age" in df.columns:
        plot_age_raincloud(df, output_dir / "age_distribution.pdf")
    else:
        plot_age_bucket_bar(df, output_dir / "age_distribution.pdf")

    print("\nPlot 2: Sex distribution...")
    plot_categorical_bar(df, "Prolific_Sex", "Sex",
                         output_path=output_dir / "sex_distribution.pdf")

    print("\nPlot 3: Ethnicity distribution...")
    plot_categorical_bar(df, "Prolific_Ethnicity simplified", "Ethnicity",
                         output_path=output_dir / "ethnicity_distribution.pdf")

    print("\nPlot 4: Employment status distribution...")
    plot_categorical_bar(df, "Prolific_Employment status", "Employment Status",
                         order=EMPLOYMENT_STATUS_ORDER,
                         output_path=output_dir / "employment_status_distribution.pdf")

    print("\nPlot 5: Work experience distribution...")
    plot_categorical_bar(df, "Work Experience", "Work Experience",
                         order=WORK_EXPERIENCE_ORDER,
                         output_path=output_dir / "work_experience_distribution.pdf")

    print("\nPlot 6: Survey duration (raincloud)...")
    plot_duration_raincloud(df, output_dir / "duration_distribution.pdf")

    print("\nPlot 7: AI usage frequency...")
    plot_ai_use_bar(df, output_dir / "ai_use_distribution.pdf")

    print("\nPlot 8: Response length (raincloud)...")
    plot_response_length_raincloud(df_full, output_dir / "response_length_distribution.pdf")

    print(f"\nDone! Plots saved to: {output_dir}")

    if show:
        plt.show()
