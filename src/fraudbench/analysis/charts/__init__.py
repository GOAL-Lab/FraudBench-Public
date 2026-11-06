"""Reusable, domain-agnostic chart primitives.

These primitives are organised by chart *type*, not by analysis context. A
domain composer in :mod:`fraudbench.analysis.plot` pre-computes stats (rates,
CIs, ns) and passes them as :class:`BarSeries` / :class:`HatchedBarSeries`
objects or plain numpy arrays to the primitives here.

Primitives never create figures; they draw onto an ``ax`` passed by the caller.
"""
from .annotations import (
    add_group_labels_below,
    add_group_separators,
    annotate_n_above_bars,
    build_hatch_legend,
)
from .legend import above_axes_legend_kw
from .bars import (
    BarSeries,
    HatchedBarSeries,
    bars_with_hatch_overlay,
    grouped_bars,
    nested_grouped_bars,
)
from .correlation import correlation_heatmap, sig_marker
from .distributions import categorical_bar, raincloud
from .style import (
    clean_axes,
    merge_error_kw,
    set_headroom_ylim,
)

__all__ = [
    # style
    "clean_axes",
    "merge_error_kw",
    "set_headroom_ylim",
    # legend
    "above_axes_legend_kw",
    # annotations
    "add_group_labels_below",
    "add_group_separators",
    "annotate_n_above_bars",
    "build_hatch_legend",
    # bars
    "BarSeries",
    "HatchedBarSeries",
    "bars_with_hatch_overlay",
    "grouped_bars",
    "nested_grouped_bars",
    # correlation
    "correlation_heatmap",
    "sig_marker",
    # distributions
    "categorical_bar",
    "raincloud",
]
