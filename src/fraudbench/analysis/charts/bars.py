"""Reusable bar-chart primitives (domain-agnostic).

Every primitive here takes a matplotlib ``ax`` plus numeric arrays and labels and
draws bars onto it. Primitives do NOT create figures, NOT read DataFrames, and
NOT know about fraudbench domain concepts (metrics, conditions, settings).

Composers in ``fraudbench.analysis.plot.*`` pre-compute stats and pass the
resulting numeric arrays via :class:`BarSeries` objects.

Three primitives cover every bar chart in the codebase:

* :func:`grouped_bars` — legend above axes (horizontal), not over bars.
* :func:`nested_grouped_bars` — same.
* :func:`bars_with_hatch_overlay` — color + hatch entries in one horizontal legend
  above the axes unless ``show_legend=False``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .annotations import (
    add_group_labels_below,
    add_group_separators,
    build_hatch_legend,
)
from .legend import above_axes_legend_kw, count_legend_entries
from .style import clean_axes, merge_error_kw, set_headroom_ylim


@dataclass
class BarSeries:
    """One bar series within a grouped-bar chart.

    ``values`` and ``errors`` must have the same length as the number of
    x-categories for the target primitive (flat ``grouped_bars``) or
    ``n_outer * n_inner`` for :func:`nested_grouped_bars`.

    ``color`` may be a single color string (one color for the whole series) OR a
    sequence of per-bar colors. When per-bar colors are used, the series usually
    should not contribute to the legend (pass ``show_legend=False`` or an empty
    ``label``).
    """

    values: Sequence[float]
    errors: Optional[Sequence[float]] = None
    color: str | Sequence[str] = "#4C72B0"
    label: str = ""
    hatch: Optional[str] = None


@dataclass
class HatchedBarSeries:
    """A metric series repeated across multiple hatch levels.

    ``by_hatch[hatch]`` is a dict with ``rates`` and optional ``ci95s`` arrays,
    both of length equal to the number of x-categories.
    """

    label: str
    color: str
    by_hatch: dict
    # e.g. by_hatch = {"closed": {"rates": [...], "ci95s": [...]}, ...}


def grouped_bars(
    ax,
    x_labels: Sequence[str],
    series: Sequence[BarSeries],
    *,
    bar_width: float = 0.35,
    bar_alpha: float = 0.85,
    error_kw: Optional[dict] = None,
    legend_loc: Optional[str] = "lower center",
    legend_kw: Optional[dict] = None,
    show_legend: bool = True,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    xlabelpad: Optional[float] = None,
    xtick_rotation: float = 30,
    xtick_ha: str = "right",
    xtick_fontsize: Optional[int] = None,
    ylim_cap: Optional[float] = None,
    ylim_headroom: float = 0.15,
    apply_clean_axes: bool = True,
) -> dict:
    """One axis, N metric series side-by-side over ``len(x_labels)`` x-categories.

    Returns a dict with ``x``, ``positions`` (per-series bar positions), and
    ``max_value`` (including errors) so composers can do extra annotation.
    """
    n = len(series)
    x = np.arange(len(x_labels))
    ekw = merge_error_kw(error_kw)

    series_positions: list[np.ndarray] = []
    all_vals_with_err: list[float] = []

    for i, s in enumerate(series):
        offset = (i - n / 2 + 0.5) * bar_width
        pos = x + offset
        series_positions.append(pos)
        yerr = list(s.errors) if s.errors is not None else None

        kwargs = dict(
            label=s.label, color=s.color,
            yerr=yerr, capsize=2, alpha=bar_alpha, error_kw=ekw,
        )
        if s.hatch:
            kwargs["hatch"] = s.hatch
            kwargs["edgecolor"] = "black"
            kwargs["linewidth"] = 0.5
        ax.bar(pos, list(s.values), bar_width, **kwargs)

        if s.errors is not None:
            all_vals_with_err.extend(float(v) + float(e or 0) for v, e in zip(s.values, s.errors))
        else:
            all_vals_with_err.extend(float(v) for v in s.values)

    ax.set_xticks(x)
    if xtick_fontsize is not None:
        ax.set_xticklabels(x_labels, rotation=xtick_rotation, ha=xtick_ha, fontsize=xtick_fontsize)
    else:
        ax.set_xticklabels(x_labels, rotation=xtick_rotation, ha=xtick_ha)
    if ylabel:
        ax.set_ylabel(ylabel, fontweight="bold")
    if xlabel:
        if xlabelpad is not None:
            ax.set_xlabel(xlabel, fontweight="bold", labelpad=xlabelpad)
        else:
            ax.set_xlabel(xlabel, fontweight="bold")

    if show_legend and any(s.label for s in series):
        n_lab = count_legend_entries(series)
        lkw = above_axes_legend_kw(ncol=n_lab)
        lkw["loc"] = legend_loc
        if legend_kw:
            lkw.update(legend_kw)
        if "ncol" not in lkw:
            lkw["ncol"] = n_lab
        ax.legend(**lkw)

    max_value = set_headroom_ylim(
        ax, all_vals_with_err, headroom=ylim_headroom, cap=ylim_cap,
    )
    if apply_clean_axes:
        clean_axes(ax)

    return {"x": x, "positions": series_positions, "max_value": max_value}


def nested_grouped_bars(
    ax,
    outer_labels: Sequence[str],
    inner_labels: Sequence[str],
    series: Sequence[BarSeries],
    *,
    bar_width: float = 0.12,
    group_spacing: float = 0.3,
    inner_gap: float = 0.0,
    bar_alpha: float = 0.85,
    error_kw: Optional[dict] = None,
    show_inner_xticks: bool = True,
    show_outer_labels: bool = True,
    show_group_separators: bool = True,
    outer_label_y: float = -0.22,
    outer_label_fontsize: int = 14,
    inner_xtick_rotation: float = 0,
    inner_xtick_ha: str = "center",
    inner_xtick_fontsize: Optional[int] = None,
    ylabel: Optional[str] = None,
    legend_loc: Optional[str] = "lower center",
    legend_kw: Optional[dict] = None,
    show_legend: bool = True,
    ylim_cap: Optional[float] = None,
    ylim_headroom: float = 0.15,
    bar_width_frac: float = 0.9,
    apply_clean_axes: bool = True,
) -> dict:
    """Outer groups × inner categories × N metric series, with below-axis group labels.

    Each element of ``series`` has values/errors flattened row-major
    (outer-first, inner-second), length ``len(outer_labels) * len(inner_labels)``.

    Returns ``{"group_centers": [...], "inner_tick_positions": [...], "max_value": float,
               "positions_by_series": [...]}``.
    """
    n_metrics = len(series)
    n_outer = len(outer_labels)
    n_inner = len(inner_labels)
    ekw = merge_error_kw(error_kw)

    group_centers: list[float] = []
    inner_tick_positions: list[float] = []

    positions_by_series: list[list[float]] = [[] for _ in series]
    current_x = 0.0

    for outer_idx in range(n_outer):
        group_start = current_x
        for inner_idx in range(n_inner):
            inner_center = current_x + (n_metrics - 1) * bar_width / 2
            inner_tick_positions.append(inner_center)
            for m_idx in range(n_metrics):
                positions_by_series[m_idx].append(current_x + m_idx * bar_width)
            current_x += n_metrics * bar_width + inner_gap
        group_end = current_x - inner_gap - bar_width
        group_centers.append((group_start + group_end) / 2)
        current_x += group_spacing

    all_vals_with_err: list[float] = []
    for m_idx, s in enumerate(series):
        values = list(s.values)
        errors = list(s.errors) if s.errors is not None else None
        kwargs = dict(
            label=s.label, color=s.color,
            yerr=errors, capsize=2, alpha=bar_alpha, error_kw=ekw,
        )
        if s.hatch:
            kwargs["hatch"] = s.hatch
            kwargs["edgecolor"] = "black"
            kwargs["linewidth"] = 0.5
        ax.bar(positions_by_series[m_idx], values, bar_width * bar_width_frac, **kwargs)
        if errors is not None:
            all_vals_with_err.extend(float(v) + float(e or 0) for v, e in zip(values, errors))
        else:
            all_vals_with_err.extend(float(v) for v in values)

    if show_inner_xticks:
        ax.set_xticks(inner_tick_positions)
        kwargs = dict(rotation=inner_xtick_rotation, ha=inner_xtick_ha)
        if inner_xtick_fontsize is not None:
            kwargs["fontsize"] = inner_xtick_fontsize
        ax.set_xticklabels(list(inner_labels) * n_outer, **kwargs)
    else:
        ax.set_xticks([])

    if show_outer_labels:
        add_group_labels_below(
            ax, group_centers, outer_labels,
            y=outer_label_y, fontsize=outer_label_fontsize,
        )

    if show_group_separators:
        add_group_separators(ax, group_centers)

    if ylabel:
        ax.set_ylabel(ylabel, fontweight="bold")

    if show_legend and any(s.label for s in series):
        n_lab = count_legend_entries(series)
        lkw = above_axes_legend_kw(ncol=n_lab)
        lkw["loc"] = legend_loc
        if legend_kw:
            lkw.update(legend_kw)
        if "ncol" not in lkw:
            lkw["ncol"] = n_lab
        ax.legend(**lkw)

    max_value = set_headroom_ylim(
        ax, all_vals_with_err, headroom=ylim_headroom, cap=ylim_cap,
    )
    if apply_clean_axes:
        clean_axes(ax)

    return {
        "group_centers": group_centers,
        "inner_tick_positions": inner_tick_positions,
        "positions_by_series": positions_by_series,
        "max_value": max_value,
    }


def bars_with_hatch_overlay(
    ax,
    x_labels: Sequence[str],
    series: Sequence[HatchedBarSeries],
    hatch_order: Sequence[str],
    hatch_patterns: dict,
    hatch_display_labels: dict,
    *,
    bar_width: float = 0.12,
    bar_alpha: float = 0.85,
    error_kw: Optional[dict] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    xtick_rotation: float = 30,
    xtick_ha: str = "right",
    xtick_fontsize: Optional[int] = None,
    group_labels: Optional[Sequence[str]] = None,
    group_sizes: Optional[Sequence[int]] = None,
    group_spacing: float = 0.3,
    group_label_y: float = -0.22,
    group_label_fontsize: int = 14,
    legend_loc: Optional[str] = "lower center",
    legend_ncol: Optional[int] = None,
    show_legend: bool = True,
    legend_kw: Optional[dict] = None,
    ylim_cap: Optional[float] = None,
    ylim_headroom: float = 0.15,
    apply_clean_axes: bool = True,
) -> dict:
    """Categories on x-axis × (metric × hatch level) bars side-by-side.

    Every metric in ``series`` is repeated once per ``hatch`` value, using
    ``hatch_patterns[hatch]`` as the hatch pattern. The legend splits into
    a color-per-metric half and a hatch-per-level half.

    Optionally, consecutive categories can be organized into labelled outer
    groups. ``group_sizes`` must partition ``x_labels`` and have one entry for
    every ``group_labels`` entry.
    """
    n_categories = len(x_labels)
    n_metrics = len(series)
    n_hatch = len(hatch_order)
    total_bars = n_metrics * n_hatch
    ekw = merge_error_kw(error_kw)

    if (group_labels is None) != (group_sizes is None):
        raise ValueError("group_labels and group_sizes must be provided together")
    if group_sizes is not None:
        if len(group_labels) != len(group_sizes):
            raise ValueError("group_labels and group_sizes must have the same length")
        if any(size <= 0 for size in group_sizes) or sum(group_sizes) != n_categories:
            raise ValueError("group_sizes must be positive and sum to len(x_labels)")
        x = np.arange(n_categories, dtype=float)
        group_centers: list[float] = []
        start = 0
        offset = 0.0
        for size in group_sizes:
            indices = slice(start, start + size)
            x[indices] += offset
            group_centers.append(float((x[start] + x[start + size - 1]) / 2))
            start += size
            offset += group_spacing
    else:
        x = np.arange(n_categories)
        group_centers = []
    all_vals_with_err: list[float] = []

    bar_idx = 0
    for s in series:
        for hatch_level in hatch_order:
            d = s.by_hatch.get(hatch_level)
            if d is None:
                rates = [0] * n_categories
                ci95s = [0] * n_categories
            else:
                rates = list(d.get("rates", [0] * n_categories))
                ci95s = list(d.get("ci95s", [0] * n_categories))
            offset = (bar_idx - total_bars / 2 + 0.5) * bar_width
            ax.bar(
                x + offset, rates, bar_width,
                color=s.color, hatch=hatch_patterns[hatch_level],
                edgecolor="black", linewidth=0.5,
                yerr=ci95s, capsize=2, alpha=bar_alpha, error_kw=ekw,
            )
            all_vals_with_err.extend(float(v) + float(e or 0) for v, e in zip(rates, ci95s))
            bar_idx += 1

    color_entries = [(s.label, s.color) for s in series]
    hatch_entries = [(hatch_display_labels[h], hatch_patterns[h]) for h in hatch_order]
    handles, labels = build_hatch_legend(color_entries, hatch_entries)

    ax.set_xticks(x)
    tick_kwargs = dict(rotation=xtick_rotation, ha=xtick_ha)
    if xtick_fontsize is not None:
        tick_kwargs["fontsize"] = xtick_fontsize
    ax.set_xticklabels(list(x_labels), **tick_kwargs)
    if group_labels is not None:
        add_group_labels_below(
            ax, group_centers, group_labels,
            y=group_label_y, fontsize=group_label_fontsize,
        )
        add_group_separators(ax, group_centers)

    if ylabel:
        ax.set_ylabel(ylabel, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontweight="bold")

    ncol = len(labels) if legend_ncol is None else legend_ncol
    if show_legend and legend_loc is not None:
        lkw = above_axes_legend_kw(ncol=ncol)
        lkw["loc"] = legend_loc
        if legend_kw:
            lkw.update(legend_kw)
        ax.legend(handles, labels, **lkw)

    max_value = set_headroom_ylim(
        ax, all_vals_with_err, headroom=ylim_headroom, cap=ylim_cap,
    )
    if apply_clean_axes:
        clean_axes(ax)

    return {"x": x, "max_value": max_value}
