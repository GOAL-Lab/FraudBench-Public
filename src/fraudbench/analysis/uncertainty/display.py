"""Formatting helpers for posterior-derived means and intervals.

Used by the new coefficient tables to render LaTeX cells like
``-0.34 [-0.45, -0.22]`` and by retrofitted rate tables to render
percentages like ``42.3 [38.1, 46.9]``.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from .intervals import Interval


def _as_interval(
    arg: Interval | tuple[float, float, float] | Sequence[float],
) -> Interval:
    """Coerce a tuple ``(mean, lo, hi)`` to an :class:`Interval`."""
    if isinstance(arg, Interval):
        return arg
    seq = list(arg)
    if len(seq) != 3:
        raise ValueError(f"Expected (mean, lo, hi); got {arg!r}")
    mean, lo, hi = seq
    return Interval(mean=float(mean), hdi_low=float(lo), hdi_high=float(hi))


def format_interval(
    interval: Interval | tuple[float, float, float],
    *,
    digits: int = 2,
    bracket_open: str = "[",
    bracket_close: str = "]",
    sep: str = ", ",
    na: str = "---",
) -> str:
    """Render an interval as ``mean [lo, hi]`` on its native scale.

    Returns ``na`` if any of the three values is non-finite (e.g. when
    the underlying coefficient is absent from this posterior).
    """
    iv = _as_interval(interval)
    if not (math.isfinite(iv.mean) and math.isfinite(iv.hdi_low) and math.isfinite(iv.hdi_high)):
        return na
    fmt = f"{{:.{digits}f}}"
    mean = fmt.format(iv.mean)
    lo = fmt.format(iv.hdi_low)
    hi = fmt.format(iv.hdi_high)
    return f"{mean} {bracket_open}{lo}{sep}{hi}{bracket_close}"


def format_percent_interval(
    interval: Interval | tuple[float, float, float],
    *,
    digits: int = 1,
    bracket_open: str = "[",
    bracket_close: str = "]",
    sep: str = ", ",
    na: str = "---",
) -> str:
    """Render a probability interval (in [0, 1]) as ``42.3 [38.1, 46.9]``.

    Multiplies every value by 100 before formatting; otherwise behaves
    like :func:`format_interval`.
    """
    iv = _as_interval(interval)
    if not (math.isfinite(iv.mean) and math.isfinite(iv.hdi_low) and math.isfinite(iv.hdi_high)):
        return na
    fmt = f"{{:.{digits}f}}"
    mean = fmt.format(iv.mean * 100)
    lo = fmt.format(iv.hdi_low * 100)
    hi = fmt.format(iv.hdi_high * 100)
    return f"{mean} {bracket_open}{lo}{sep}{hi}{bracket_close}"


def latex_significance_marker(
    interval: Interval | tuple[float, float, float],
    *,
    null: float = 0.0,
) -> str:
    """Return a TeX marker for HDIs that exclude *null* on the relevant side.

    A coefficient whose 95% HDI excludes zero is conventionally marked
    with a star in regression tables. This helper returns ``"^{*}"`` for
    HDIs strictly above or below *null* and the empty string otherwise.
    """
    iv = _as_interval(interval)
    if not (math.isfinite(iv.hdi_low) and math.isfinite(iv.hdi_high)):
        return ""
    if iv.hdi_low > null or iv.hdi_high < null:
        return "^{*}"
    return ""
