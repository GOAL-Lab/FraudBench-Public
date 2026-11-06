"""Sample-level metric-correlation heatmap.

Backs the left panel of ``main_text_correlations_and_conditional.pdf``
(see ``analysis/plots/plot_main_text_awareness_and_fraud_type.py``): a
triangular Pearson-correlation heatmap over the four sample-level binary
metrics (fraud, refusal, legal awareness, fraud awareness) with
significance markers.
"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..style import save_figure, setup_plot_style
from ..charts import correlation_heatmap
from ..common import METRIC_CONFIGS_FULL


METRIC_CONFIGS = METRIC_CONFIGS_FULL
METRIC_ORDER = ["fraud", "refusal", "legal_awareness", "fraud_awareness"]


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Return a correlation matrix for all configured metrics (display names)."""
    metric_cols = [METRIC_CONFIGS[m]["column"] for m in METRIC_ORDER]
    metric_names = [METRIC_CONFIGS[m]["display_name"] for m in METRIC_ORDER]
    subset = df[metric_cols].copy()
    subset.columns = metric_names
    return subset.corr()


def compute_pairwise_correlation(df: pd.DataFrame, metric1: str, metric2: str):
    """Return ``(pearson_r, p_value, n_valid)`` for the pair after NaN-filtering."""
    col1 = METRIC_CONFIGS[metric1]["column"]
    col2 = METRIC_CONFIGS[metric2]["column"]

    valid_mask = df[col1].notna() & df[col2].notna()
    x = df.loc[valid_mask, col1].values
    y = df.loc[valid_mask, col2].values

    if len(x) < 2:
        return 0, 1, 0

    corr, pval = stats.pearsonr(x, y)
    return corr, pval, len(x)


def compute_correlation_pvalues(df: pd.DataFrame) -> np.ndarray:
    """Return the matrix of pairwise Pearson p-values (zeros on the diagonal)."""
    n_metrics = len(METRIC_ORDER)
    pval_matrix = np.zeros((n_metrics, n_metrics))
    for i, m1 in enumerate(METRIC_ORDER):
        for j, m2 in enumerate(METRIC_ORDER):
            if i != j:
                _, pval, _ = compute_pairwise_correlation(df, m1, m2)
                pval_matrix[i, j] = pval
    return pval_matrix


def plot_full_correlation_heatmap(df: pd.DataFrame, output_path: Optional[Path] = None,
                                   setting_name: str = "",  # noqa: ARG001
                                   *, ax=None, **heatmap_kwargs):
    """Triangular correlation heatmap with significance markers.

    Pass ``ax`` to draw into an existing axes (e.g. as a panel of a larger
    figure); extra keyword arguments are forwarded to ``correlation_heatmap``.
    """
    if ax is None:
        setup_plot_style()

    corr_matrix = compute_correlations(df)
    pval_matrix = compute_correlation_pvalues(df)

    kwargs = {
        "annotate_with_sig": True,
        "figsize": (10, 8),
        "annot_fontsize": 13,
        "gray_diagonal": True,
        **heatmap_kwargs,
    }
    fig, ax = correlation_heatmap(corr_matrix, pval_matrix=pval_matrix, ax=ax, **kwargs)

    if output_path:
        save_figure(plt.gcf(), output_path, bbox_inches="tight")
        print(f"Saved full heatmap to {output_path}")

    return fig, ax
