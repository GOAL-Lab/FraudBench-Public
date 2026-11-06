"""Taylor linearization for stratified simple-random samples of items."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._validation import _ordered_groups, _validated_population_sizes


@dataclass(frozen=True, slots=True)
class _LinearizationResult:
    """Private result from the shared stratified variance calculation."""

    standard_error: float
    design_degrees_of_freedom: int
    stratum_sample_sizes: Mapping[Hashable, int]
    stratum_weight_shares: Mapping[Hashable, float]
    stratum_variance_contributions: Mapping[Hashable, float]
    finite_population_correction_applied: bool


def _stratified_linearized_se(
    linearized: NDArray[np.float64],
    sample_weight: NDArray[np.float64],
    strata: NDArray[np.object_],
    population_sizes: int | Mapping[Hashable, int] | None,
) -> _LinearizationResult:
    """Calculate a Taylor-linearized SE under stratified SRS(WOR).

    Args:
        linearized: Item-level, mean-zero linearized values.
        sample_weight: Positive item weights, constant within each stratum.
        strata: Validated stratum identifiers.
        population_sizes: Optional finite target-population stratum sizes.

    Returns:
        Standard error and design diagnostics.

    Raises:
        ValueError: If the supplied design is unsupported or not estimable.
    """
    groups = _ordered_groups(strata)
    populations = _validated_population_sizes(population_sizes, groups)
    total_weight = sample_weight.sum(dtype=np.float64)

    sample_sizes: dict[Hashable, int] = {}
    weight_shares: dict[Hashable, float] = {}
    contributions: dict[Hashable, float] = {}
    design_df = 0

    if populations is not None:
        population_total = sum(populations.values())
    else:
        population_total = 0

    for stratum, index in groups:
        n_h = len(index)
        stratum_weights = sample_weight[index]
        relative_weights = stratum_weights / stratum_weights[0]
        if not np.allclose(
            relative_weights,
            np.ones_like(relative_weights),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                "Linearization standard errors support stratified simple random "
                "sampling with constant item weights within each stratum. Request "
                "`se_method=None` to calculate only the point estimate for arbitrary "
                "item weights."
            )

        share = float(stratum_weights.sum(dtype=np.float64) / total_weight)
        sample_sizes[stratum] = int(n_h)
        weight_shares[stratum] = share
        design_df += n_h - 1

        if populations is not None:
            population_share = populations[stratum] / population_total
            if not np.isclose(
                share,
                population_share,
                rtol=1e-8,
                atol=1e-10,
            ):
                raise ValueError(
                    "Weight-implied and population-size-implied shares disagree "
                    f"for stratum {stratum!r}: {share} versus {population_share}. "
                    "Use calibrated stratum weights proportional to N_h / n_h."
                )

        if n_h == 1:
            if populations is None or populations[stratum] > 1:
                raise ValueError(
                    f"Stratum {stratum!r} has one sampled item, so its within-stratum "
                    "variance is not estimable. Only a certainty stratum with "
                    "n_h=N_h=1 is supported."
                )
            contribution = 0.0
        else:
            variance = float(np.var(linearized[index], ddof=1))
            sampling_fraction = (
                0.0 if populations is None else n_h / populations[stratum]
            )
            contribution = float(
                share * share
                * (1.0 - sampling_fraction)
                * variance
                / n_h
            )
        contributions[stratum] = contribution

    total_variance = float(sum(contributions.values()))
    tolerance = (
        10.0
        * np.finfo(np.float64).eps
        * max(1.0, sum(abs(value) for value in contributions.values()))
    )
    if total_variance < -tolerance:
        raise RuntimeError("Internal numerical error: negative linearized variance.")
    total_variance = max(total_variance, 0.0)

    if design_df == 0:
        all_certainty_singletons = (
            populations is not None
            and all(
                len(index) == populations[stratum] == 1
                for stratum, index in groups
            )
        )
        if not all_certainty_singletons:
            raise ValueError("The sampling design has no estimable design degrees of freedom.")

    return _LinearizationResult(
        standard_error=float(np.sqrt(total_variance)),
        design_degrees_of_freedom=int(design_df),
        stratum_sample_sizes=sample_sizes,
        stratum_weight_shares=weight_shares,
        stratum_variance_contributions=contributions,
        finite_population_correction_applied=populations is not None,
    )
