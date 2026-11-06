"""Interval helpers for FraudBench tables and figures.

Two kinds of 95% interval appear in the paper, and this package is the
single home for both:

* **Wilson binomial CIs** on empirical proportions -- every rate table and
  error bar (:mod:`.wilson`).
* **Posterior HDIs** of the M1 / M2 / M3 regression coefficients -- the
  regression tables and the posterior-predictive human-vs-LLM gap
  (:mod:`.loaders`, :mod:`.extractors`, :mod:`.intervals`, :mod:`.display`).

Submodules
----------
* :mod:`.intervals` -- :class:`Interval` plus HDI / mean-with-HDI primitives.
* :mod:`.wilson` -- Wilson interval, vectorised half-widths and the
  ``rate_and_halfwidth`` helper used by the plot composers.
* :mod:`.loaders` -- cached loader for ``inference_data.nc`` (posterior
  group only), ``processed_data.parquet`` and ``features.pkl``.
* :mod:`.extractors` -- typed accessors (``DataFrame``-returning) for the
  coefficient blocks of the three regressions.
* :mod:`.display` -- ``mean [lo, hi]`` string formatting helpers.

The default probability mass is 95% to match the paper text.
"""
from __future__ import annotations

from .intervals import DEFAULT_HDI_PROB, Interval, hdi, mean_with_hdi
from .loaders import (
    PosteriorBundle,
    load_bundle,
    load_features,
    load_posterior,
    load_processed_data,
    stack_draws,
)
from .display import format_interval, format_percent_interval, latex_significance_marker
from .extractors import (
    get_family_re,
    get_human_coefs,
    get_interaction_matrices,
    get_judge_coefs,
    get_judge_model_re,
    get_model_coefs,
    get_model_re,
    get_scenario_coefs,
    get_token_coefs,
)
from .wilson import rate_and_halfwidth, wilson_halfwidths, wilson_interval


__all__ = [
    "DEFAULT_HDI_PROB",
    "Interval",
    "PosteriorBundle",
    "format_interval",
    "format_percent_interval",
    "get_family_re",
    "get_human_coefs",
    "get_interaction_matrices",
    "get_judge_coefs",
    "get_judge_model_re",
    "get_model_coefs",
    "get_model_re",
    "get_scenario_coefs",
    "get_token_coefs",
    "hdi",
    "latex_significance_marker",
    "load_bundle",
    "load_features",
    "load_posterior",
    "load_processed_data",
    "mean_with_hdi",
    "rate_and_halfwidth",
    "stack_draws",
    "wilson_halfwidths",
    "wilson_interval",
]
