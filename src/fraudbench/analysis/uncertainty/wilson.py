"""Wilson binomial confidence intervals for the descriptive rate tables and figures.

Every bar height / table cell in the paper is the **empirical proportion**,
and every error bar / bracketed interval is a 95% **Wilson** score interval
on that proportion. Wilson always contains the empirical mean and has
correct coverage near 0 / 1, which makes it the standard descriptive choice
for binomial proportions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .intervals import DEFAULT_HDI_PROB, Interval


def _normal_quantile(p: float) -> float:
    """Inverse standard-normal CDF for ``p`` in (0, 1) (Acklam's approximation)."""
    if not (0.0 < p < 1.0):
        raise ValueError(f"_normal_quantile expects p in (0, 1); got {p}")
    # Acklam (2003) rational approximation; good to ~1e-9 over (0,1).
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
         3.754408661907416]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = float(np.sqrt(-2.0 * np.log(p)))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = float(np.sqrt(-2.0 * np.log(1.0 - p)))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def wilson_interval(
    successes: int,
    n: int,
    *,
    hdi_prob: float = DEFAULT_HDI_PROB,
) -> Interval:
    """Wilson score interval at confidence ``hdi_prob`` (default 95%).

    ``mean`` is the raw proportion ``successes / n`` (the number shown in
    tables and as bar heights); ``hdi_low`` / ``hdi_high`` are the Wilson
    bounds. Returns an all-NaN interval when ``n <= 0``.
    """
    if n <= 0:
        return Interval(float("nan"), float("nan"), float("nan"), hdi_prob)
    z = _normal_quantile((1.0 + hdi_prob) / 2.0)
    p = successes / n
    denom = 1.0 + (z ** 2) / n
    centre = p + (z ** 2) / (2.0 * n)
    margin = z * float(np.sqrt(p * (1.0 - p) / n + (z ** 2) / (4.0 * n ** 2)))
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return Interval(mean=p, hdi_low=lo, hdi_high=hi, hdi_prob=hdi_prob)


def wilson_halfwidths(
    successes: np.ndarray,
    n: np.ndarray,
    *,
    hdi_prob: float = DEFAULT_HDI_PROB,
) -> np.ndarray:
    """Vectorised Wilson half-widths for use as matplotlib ``yerr``.

    Returns ``np.nan`` where ``n <= 0``. The half-width is the larger of the
    two asymmetric Wilson offsets from the empirical proportion, so a
    symmetric error bar always visually spans the interval even near 0 / 1.
    """
    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    out = np.full_like(n, np.nan, dtype=float)
    valid = n > 0
    if not valid.any():
        return out
    for idx in np.ndindex(n.shape):
        if not valid[idx]:
            continue
        iv = wilson_interval(int(round(successes[idx])), int(n[idx]), hdi_prob=hdi_prob)
        p = iv.mean
        out[idx] = max(p - iv.hdi_low, iv.hdi_high - p)
    return out


def rate_and_halfwidth(
    subset_df: pd.DataFrame,
    metric_column: str,
    *,
    hdi_prob: float = DEFAULT_HDI_PROB,
) -> tuple[float, float]:
    """Empirical rate of a binary metric column plus its Wilson half-width.

    Rows with a NaN metric value are ignored. Returns ``(0.0, 0.0)`` when the
    column is missing or no row has a value, so plot code can draw an empty
    bar without special-casing.
    """
    if metric_column not in subset_df.columns:
        return 0.0, 0.0
    valid = subset_df[metric_column].dropna()
    n = int(len(valid))
    if n == 0:
        return 0.0, 0.0
    successes = int(round(float(valid.sum())))
    rate = float(valid.sum()) / n
    iv = wilson_interval(successes, n, hdi_prob=hdi_prob)
    if not np.isfinite(iv.hdi_low) or not np.isfinite(iv.hdi_high):
        return rate, 0.0
    return rate, float(max(iv.mean - iv.hdi_low, iv.hdi_high - iv.mean))
