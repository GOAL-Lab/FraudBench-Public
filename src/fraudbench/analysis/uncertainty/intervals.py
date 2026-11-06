"""Interval primitives shared by the other submodules.

:class:`Interval` is the common container for a point estimate plus a 95%
interval, whether that interval is a posterior HDI (regression tables) or a
Wilson binomial CI (:mod:`.wilson`, rate tables and figures). The default
probability mass is 95% to match the paper text.

Posterior summaries use the highest-density interval (HDI) rather than
equal-tailed quantiles because HDIs are robust to asymmetric posteriors of
bounded quantities like probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import numpy as np


DEFAULT_HDI_PROB: float = 0.95
"""Default HDI mass used everywhere we don't pass an explicit probability."""


@dataclass(frozen=True)
class Interval:
    """A point estimate plus a 95% interval (posterior HDI or Wilson CI)."""

    mean: float
    hdi_low: float
    hdi_high: float
    hdi_prob: float = DEFAULT_HDI_PROB

    @property
    def half_width(self) -> float:
        """Average half-width of the interval; useful as a symmetric error-bar value."""
        return (self.hdi_high - self.hdi_low) / 2.0

    @property
    def err_low(self) -> float:
        """Asymmetric error-bar offset below the mean (always non-negative)."""
        return max(self.mean - self.hdi_low, 0.0)

    @property
    def err_high(self) -> float:
        """Asymmetric error-bar offset above the mean (always non-negative)."""
        return max(self.hdi_high - self.mean, 0.0)


def hdi(
    samples: np.ndarray,
    *,
    hdi_prob: float = DEFAULT_HDI_PROB,
) -> tuple[float, float]:
    """Return the ``hdi_prob`` HDI of a flat sample array.

    Wraps :func:`arviz.hdi` so the rest of the package never imports arviz
    directly (and we have one place to swap in a different estimator).

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples. Any shape; flattened internally.
    hdi_prob : float
        HDI mass in (0, 1); defaults to :data:`DEFAULT_HDI_PROB`.
    """
    flat = np.asarray(samples).reshape(-1)
    if flat.size == 0:
        return float("nan"), float("nan")
    interval = az.hdi(flat, hdi_prob=hdi_prob)
    lo, hi = float(interval[0]), float(interval[1])
    return lo, hi


def mean_with_hdi(
    samples: np.ndarray,
    *,
    hdi_prob: float = DEFAULT_HDI_PROB,
) -> Interval:
    """Posterior-mean + HDI summary of *samples* as an :class:`Interval`."""
    flat = np.asarray(samples).reshape(-1)
    if flat.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), hdi_prob)
    lo, hi = hdi(flat, hdi_prob=hdi_prob)
    return Interval(
        mean=float(np.mean(flat)),
        hdi_low=lo,
        hdi_high=hi,
        hdi_prob=hdi_prob,
    )
