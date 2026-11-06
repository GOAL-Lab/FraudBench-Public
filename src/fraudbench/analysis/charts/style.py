"""Shared styling helpers for chart primitives.

These helpers capture the handful of styling idioms that are duplicated across
every bar/scatter/heatmap plot in the codebase. Domain composers should never
set spine/grid/ylim themselves: call :func:`clean_axes` and :func:`set_headroom_ylim`
instead.
"""
from __future__ import annotations

from typing import Iterable, Optional


_DEFAULT_ERROR_KW: dict = {"elinewidth": 1, "capthick": 0.8, "alpha": 0.7}
"""Matplotlib ``error_kw`` used by the bar primitives."""


def clean_axes(
    ax,
    *,
    hide_top: bool = True,
    hide_right: bool = True,
    hide_left: bool = False,
    ygrid: bool = True,
    xgrid: bool = False,
    grid_linestyle: str = "--",
    grid_alpha: float = 0.7,
) -> None:
    """Apply the standard "clean" axes styling.

    Hides the top/right spines, enables a dashed y-axis grid underneath bars,
    and sets ``axisbelow`` so bars render on top of the grid.
    """
    if hide_top:
        ax.spines["top"].set_visible(False)
    if hide_right:
        ax.spines["right"].set_visible(False)
    if hide_left:
        ax.spines["left"].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, linestyle=grid_linestyle, alpha=grid_alpha)
    if xgrid:
        ax.xaxis.grid(True, linestyle=grid_linestyle, alpha=grid_alpha)
    ax.set_axisbelow(True)


def set_headroom_ylim(
    ax,
    values: Iterable[float],
    *,
    errors: Optional[Iterable[float]] = None,
    headroom: float = 0.15,
    cap: Optional[float] = None,
    floor: float = 0.0,
    fallback: float = 1.0,
) -> float:
    """Set ``ax`` y-axis limits to ``[floor, max(values + errors) * (1 + headroom)]``.

    ``errors`` are added elementwise when provided. ``cap`` optionally clamps
    the upper bound (e.g. ``cap=1.0`` for proportion plots). Returns the
    computed upper bound.
    """
    vals = [v for v in values if v is not None]
    if errors is not None:
        errs = list(errors)
        if len(errs) == len(vals):
            vals = [v + (e or 0) for v, e in zip(vals, errs)]

    if not vals:
        upper = fallback
    else:
        top = max(vals)
        if top <= 0:
            upper = fallback
        else:
            upper = top * (1 + headroom)
    if cap is not None:
        upper = min(upper, cap)
    ax.set_ylim(floor, upper)
    return upper


def merge_error_kw(override: Optional[dict] = None) -> dict:
    """Return ``_DEFAULT_ERROR_KW`` merged with ``override``."""
    if not override:
        return dict(_DEFAULT_ERROR_KW)
    merged = dict(_DEFAULT_ERROR_KW)
    merged.update(override)
    return merged
