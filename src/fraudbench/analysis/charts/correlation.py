"""Correlation chart primitives: annotated correlation heatmap."""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def sig_marker(pval: float) -> str:
    """Return a significance marker (``*``, ``**``, ``***``, or empty)."""
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return ""


def correlation_heatmap(
    corr_matrix,
    *,
    pval_matrix: Optional[np.ndarray] = None,
    annotate_with_sig: bool = False,
    figsize: tuple = (8, 7),
    triangular: bool = True,
    center: float = 0,
    vmin: float = -1,
    vmax: float = 1,
    fmt: str = ".3f",
    annot_fontsize: int = 14,
    cbar_label: str = "Correlation",
    caption: Optional[str] = None,
    xtick_rotation: float = 30,
    xtick_ha: str = "right",
    gray_diagonal: bool = False,
    diagonal_color: str = "#d9d9d9",
    ax=None,
    cbar_kws: Optional[dict] = None,
    cbar_ax=None,
):
    """Draw a (optionally triangular) correlation heatmap.

    If ``pval_matrix`` is provided and ``annotate_with_sig`` is True, annotations
    are rendered as ``r.rr{sig}`` where ``sig`` is ``*``, ``**``, ``***``, or "".
    If ``gray_diagonal`` is True, the trivial self-correlation cells are drawn
    as plain gray squares without annotations.

    If ``ax`` is given, the heatmap is drawn into it (``figsize`` is ignored and
    no ``tight_layout`` is applied). ``cbar_ax`` draws the colorbar into that
    axes instead of stealing space from ``ax``. Returns ``(fig, ax)``.
    """
    sns.set_style("white")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        owns_figure = True
    else:
        fig = ax.figure
        owns_figure = False
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    n = corr_matrix.shape[0]
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1) if triangular else np.zeros((n, n), dtype=bool)
    if gray_diagonal:
        mask = mask | np.eye(n, dtype=bool)

    annot_arg: object
    fmt_arg = fmt
    if annotate_with_sig and pval_matrix is not None:
        annot = []
        for i in range(n):
            row = []
            for j in range(n):
                r = corr_matrix.iloc[i, j] if hasattr(corr_matrix, "iloc") else corr_matrix[i, j]
                if i == j:
                    row.append(f"{r:.2f}")
                else:
                    row.append(f"{r:.2f}{sig_marker(pval_matrix[i, j])}")
            annot.append(row)
        annot_arg = np.array(annot)
        fmt_arg = ""
    else:
        annot_arg = True

    sns.heatmap(
        corr_matrix, mask=mask, annot=annot_arg, fmt=fmt_arg, cmap=cmap,
        center=center, vmin=vmin, vmax=vmax, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": cbar_label, **(cbar_kws or {})},
        cbar_ax=cbar_ax,
        annot_kws={"size": annot_fontsize, "fontweight": "bold"},
        ax=ax,
    )

    if gray_diagonal:
        for i in range(n):
            ax.add_patch(plt.Rectangle(
                (i, i), 1, 1, facecolor=diagonal_color, edgecolor="white", linewidth=0.5,
            ))

    ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation, ha=xtick_ha)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    if caption:
        fig.text(0.5, -0.02, caption, ha="center", fontsize=12, style="italic")

    if owns_figure:
        plt.tight_layout()
    return fig, ax
