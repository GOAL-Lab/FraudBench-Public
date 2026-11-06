"""Nominal item-weighted Cohen's and Fleiss/Scott kappa estimators."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import t

from ._linearization import _stratified_linearized_se
from ._validation import (
    _as_1d,
    _as_2d_ratings,
    _common_stratum_diagnostics,
    _is_missing,
    _validate_indexes,
    _validate_labels,
    _validate_options,
    _validate_strata,
    _validate_weights,
)


@dataclass(frozen=True, slots=True)
class KappaResult:
    """Result of an item-weighted nominal kappa calculation."""

    estimate: float
    standard_error: float | None
    confidence_interval: tuple[float, float] | None
    observed_agreement: float
    expected_agreement: float
    n_items: int
    labels: tuple[Hashable, ...]
    weight_sum: float
    n_strata: int
    design_degrees_of_freedom: int | None
    method: str
    se_method: str | None
    components: Mapping[str, Any]


def _checked_probability(value: float, *, name: str) -> float:
    """Correct only roundoff-sized excursions outside the probability range."""
    tolerance = 10.0 * np.finfo(np.float64).eps
    if -tolerance <= value < 0.0:
        return 0.0
    if 1.0 < value <= 1.0 + tolerance:
        return 1.0
    if not 0.0 <= value <= 1.0:
        raise RuntimeError(f"Internal numerical error: {name} is outside [0, 1].")
    return float(value)


def _estimate_from_agreements(
    observed_agreement: float,
    expected_agreement: float,
) -> tuple[float, float, float]:
    """Calculate kappa after checking for degenerate or unstable marginals."""
    observed = _checked_probability(observed_agreement, name="observed agreement")
    expected = _checked_probability(expected_agreement, name="expected agreement")
    denominator = 1.0 - expected
    threshold = 10.0 * np.finfo(np.float64).eps
    if denominator <= threshold:
        raise ValueError(
            "The marginal category distribution is degenerate: expected agreement "
            "is numerically one, so kappa is undefined."
        )
    if denominator < 1e-8:
        warnings.warn(
            "Expected agreement is extremely close to one; the kappa estimate and "
            "standard error may be numerically unstable.",
            RuntimeWarning,
            stacklevel=3,
        )
    return (
        float((observed - expected) / denominator),
        observed,
        expected,
    )


def _kappa_component_derivatives(
    observed_agreement: float,
    expected_agreement: float,
) -> tuple[float, float]:
    """Return derivatives of kappa with respect to observed and expected agreement."""
    denominator = 1.0 - expected_agreement
    return (
        float(1.0 / denominator),
        float(
            (observed_agreement - 1.0)
            / (denominator * denominator)
        ),
    )


def _cohen_component_gradients(
    observed_agreement: float,
    marginal_a: NDArray[np.float64],
    marginal_b: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Return the gradient for Cohen's component-mean parameterization."""
    expected_agreement = float(np.dot(marginal_a, marginal_b))
    d_observed, d_expected = _kappa_component_derivatives(
        observed_agreement,
        expected_agreement,
    )
    return d_observed, d_expected * marginal_b, d_expected * marginal_a


def _fleiss_component_gradients(
    observed_agreement: float,
    category_proportions: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """Return the gradient for Fleiss/Scott's component-mean parameterization."""
    expected_agreement = float(
        np.dot(category_proportions, category_proportions)
    )
    d_observed, d_expected = _kappa_component_derivatives(
        observed_agreement,
        expected_agreement,
    )
    return d_observed, 2.0 * d_expected * category_proportions


def _cohen_linearized_values(
    agreement_indicator: NDArray[np.float64],
    a_code: NDArray[np.intp],
    b_code: NDArray[np.intp],
    observed_agreement: float,
    marginal_a: NDArray[np.float64],
    marginal_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate item-level Cohen linearized values without one-hot matrices."""
    d_observed, gradient_a, gradient_b = _cohen_component_gradients(
        observed_agreement,
        marginal_a,
        marginal_b,
    )
    center_a = float(np.dot(gradient_a, marginal_a))
    center_b = float(np.dot(gradient_b, marginal_b))
    return (
        d_observed * (agreement_indicator - observed_agreement)
        + gradient_a[a_code]
        - center_a
        + gradient_b[b_code]
        - center_b
    ).astype(np.float64, copy=False)


def _fleiss_linearized_values(
    item_agreement: NDArray[np.float64],
    item_category_proportions: NDArray[np.float64],
    observed_agreement: float,
    category_proportions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate item-level Fleiss/Scott linearized values."""
    d_observed, gradient_category = _fleiss_component_gradients(
        observed_agreement,
        category_proportions,
    )
    return (
        d_observed * (item_agreement - observed_agreement)
        + (item_category_proportions - category_proportions[None, :])
        @ gradient_category
    ).astype(np.float64, copy=False)


def _readonly_array(array: NDArray[Any]) -> NDArray[Any]:
    """Return an independent read-only diagnostic array."""
    result = np.array(array, copy=True)
    result.flags.writeable = False
    return result


def _readonly_mapping(mapping: Mapping[Hashable, Any]) -> Mapping[Hashable, Any]:
    """Return an independent read-only shallow mapping."""
    return MappingProxyType(dict(mapping))


def _build_result(
    *,
    estimate: float,
    observed_agreement: float,
    expected_agreement: float,
    n_items: int,
    labels: tuple[Hashable, ...],
    weight_sum: float,
    weights: NDArray[np.float64],
    strata: NDArray[np.object_],
    population_sizes: int | Mapping[Hashable, int] | None,
    confidence_level: float,
    se_method: Literal["linearization"] | None,
    linearized: NDArray[np.float64],
    method: str,
    metric_components: dict[str, Any],
) -> KappaResult:
    """Combine inference and immutable diagnostics into the public result."""
    if se_method == "linearization":
        linearization = _stratified_linearized_se(
            linearized,
            weights,
            strata,
            population_sizes,
        )
        standard_error: float | None = float(linearization.standard_error)
        design_df: int | None = int(linearization.design_degrees_of_freedom)
        if design_df == 0:
            confidence_interval: tuple[float, float] | None = (estimate, estimate)
        else:
            alpha = 1.0 - float(confidence_level)
            critical = float(t.ppf(1.0 - alpha / 2.0, df=design_df))
            confidence_interval = (
                float(estimate - critical * standard_error),
                float(estimate + critical * standard_error),
            )
        sample_sizes = linearization.stratum_sample_sizes
        weight_shares = linearization.stratum_weight_shares
        metric_components["stratum_variance_contributions"] = _readonly_mapping(
            linearization.stratum_variance_contributions
        )
        fpc_applied = linearization.finite_population_correction_applied
        result_se_method: str | None = "stratified Taylor linearization"
    else:
        standard_error = None
        confidence_interval = None
        design_df = None
        sample_sizes, weight_shares = _common_stratum_diagnostics(weights, strata)
        fpc_applied = False
        result_se_method = None

    metric_components["stratum_sample_sizes"] = _readonly_mapping(sample_sizes)
    metric_components["stratum_weight_shares"] = _readonly_mapping(weight_shares)
    metric_components["finite_population_correction_applied"] = bool(fpc_applied)

    return KappaResult(
        estimate=float(estimate),
        standard_error=standard_error,
        confidence_interval=confidence_interval,
        observed_agreement=float(observed_agreement),
        expected_agreement=float(expected_agreement),
        n_items=int(n_items),
        labels=labels,
        weight_sum=float(weight_sum),
        n_strata=int(len(sample_sizes)),
        design_degrees_of_freedom=design_df,
        method=method,
        se_method=result_se_method,
        components=MappingProxyType(metric_components.copy()),
    )


def item_weighted_cohen_kappa(
    annotations_a: ArrayLike,
    annotations_b: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    labels: Sequence[Hashable] | None = None,
    strata: ArrayLike | None = None,
    population_sizes: int | Mapping[Hashable, int] | None = None,
    confidence_level: float = 0.95,
    se_method: Literal["linearization"] | None = "linearization",
    nan_policy: Literal["raise", "omit"] = "raise",
) -> KappaResult:
    """Calculate nominal Cohen's kappa using item-level target weights.

    ``sample_weight`` contains item-level sampling or standardization weights.
    These weights determine how strongly each item contributes to the
    target-population estimate. They are not linear, quadratic, or other
    category-disagreement weights.

    The linearized standard error assumes independent samples across supplied
    strata, SRSWOR within each stratum, and weights that are constant within
    strata. Optional finite-population corrections use the supplied target
    stratum sizes. Clustered, multistage, replicate-weight, and
    unequal-probability within-stratum designs are not supported. The overall
    estimate weights agreement components before calculating kappa; it is not
    an average of stratum-specific kappas. The unclipped Wald interval may
    extend outside ``[-1, 1]``.

    Args:
        annotations_a: One nominal annotation per sampled item from annotator A.
        annotations_b: One nominal annotation per sampled item from annotator B.
        sample_weight: Optional strictly positive finite item weights.
        labels: Optional explicit category labels.
        strata: Optional sampling-stratum identifier per item.
        population_sizes: Optional target stratum sizes for finite-population
            corrections.
        confidence_level: Confidence level for the two-sided Wald interval.
        se_method: ``"linearization"`` for design-based inference or ``None``.
        nan_policy: ``"raise"`` or complete-case ``"omit"``.

    Returns:
        An immutable :class:`KappaResult` with estimates and diagnostics.

    Raises:
        ValueError: If inputs are invalid, kappa is undefined, or the requested
            standard error is unsupported by the supplied design.
    """
    _validate_options(
        confidence_level=confidence_level,
        se_method=se_method,
        nan_policy=nan_policy,
        population_sizes=population_sizes,
    )
    a, a_index = _as_1d(annotations_a, name="annotations_a")
    b, b_index = _as_1d(annotations_b, name="annotations_b")
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Cohen annotation inputs must not be empty.")
    if len(a) != len(b):
        raise ValueError(
            "annotations_a and annotations_b must have equal length; "
            f"received {len(a)} and {len(b)}."
        )

    n_initial = len(a)
    weights, weight_index = _validate_weights(sample_weight, n_items=n_initial)
    stratum_values, strata_index, _ = _validate_strata(
        strata,
        n_items=n_initial,
    )
    _validate_indexes(
        (
            ("annotations_a", a_index),
            ("annotations_b", b_index),
            ("sample_weight", weight_index),
            ("strata", strata_index),
        )
    )

    missing_a = np.fromiter((_is_missing(value) for value in a), dtype=bool)
    missing_b = np.fromiter((_is_missing(value) for value in b), dtype=bool)
    missing = missing_a | missing_b
    if nan_policy == "raise" and np.any(missing):
        raise ValueError(
            "annotations_a and annotations_b contain missing values; use "
            "nan_policy='omit' for complete-case omission."
        )
    if nan_policy == "omit":
        keep = ~missing
        a = a[keep]
        b = b[keep]
        weights = weights[keep]
        stratum_values = stratum_values[keep]
    if len(a) == 0:
        raise ValueError("No Cohen items remain after complete-case omission.")
    if se_method == "linearization" and len(a) < 2:
        raise ValueError("At least two analyzed items are required for a standard error.")

    observed_values = [*a.tolist(), *b.tolist()]
    result_labels, label_to_code = _validate_labels(observed_values, labels)
    a_code = np.fromiter(
        (label_to_code[value] for value in a),
        dtype=np.intp,
        count=len(a),
    )
    b_code = np.fromiter(
        (label_to_code[value] for value in b),
        dtype=np.intp,
        count=len(b),
    )
    n_categories = len(result_labels)
    weight_sum = weights.sum(dtype=np.float64)
    normalized_weight = weights / weight_sum

    marginal_a = np.bincount(
        a_code,
        weights=normalized_weight,
        minlength=n_categories,
    ).astype(np.float64, copy=False)
    marginal_b = np.bincount(
        b_code,
        weights=normalized_weight,
        minlength=n_categories,
    ).astype(np.float64, copy=False)
    agreement_indicator = (a_code == b_code).astype(np.float64)
    observed_agreement = float(np.dot(normalized_weight, agreement_indicator))
    expected_agreement = float(np.dot(marginal_a, marginal_b))
    (
        estimate,
        observed_agreement,
        expected_agreement,
    ) = _estimate_from_agreements(
        observed_agreement,
        expected_agreement,
    )

    flat_pair_code = a_code * n_categories + b_code
    joint = np.bincount(
        flat_pair_code,
        weights=normalized_weight,
        minlength=n_categories * n_categories,
    ).reshape(n_categories, n_categories)
    joint = joint.astype(np.float64, copy=False)

    linearized = _cohen_linearized_values(
        agreement_indicator,
        a_code,
        b_code,
        observed_agreement,
        marginal_a,
        marginal_b,
    )

    return _build_result(
        estimate=estimate,
        observed_agreement=observed_agreement,
        expected_agreement=expected_agreement,
        n_items=len(a),
        labels=result_labels,
        weight_sum=weight_sum,
        weights=weights,
        strata=stratum_values,
        population_sizes=population_sizes,
        confidence_level=confidence_level,
        se_method=se_method,
        linearized=linearized,
        method="item-weighted nominal Cohen's kappa",
        metric_components={
            "joint_proportions": _readonly_array(joint),
            "marginal_a": _readonly_array(marginal_a),
            "marginal_b": _readonly_array(marginal_b),
            "n_items_omitted": int(n_initial - len(a)),
        },
    )


def item_weighted_fleiss_kappa(
    ratings: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    labels: Sequence[Hashable] | None = None,
    strata: ArrayLike | None = None,
    population_sizes: int | Mapping[Hashable, int] | None = None,
    confidence_level: float = 0.95,
    se_method: Literal["linearization"] | None = "linearization",
    nan_policy: Literal["raise", "omit"] = "raise",
) -> KappaResult:
    """Calculate nominal item-weighted Fleiss/Scott kappa.

    ``sample_weight`` contains item-level sampling or standardization weights,
    not category-disagreement weights. Items, rather than individual ratings,
    are the sampling units; every item contributes according to its weight even
    when valid rater counts differ.

    With ``nan_policy="omit"``, the estimand conditions on the observed rater
    assignments and retained rating cells. Target-population interpretation
    therefore requires rater assignment and rating missingness to be ignorable
    for the intended agreement outcome. The reported standard error reflects
    item sampling only, not uncertainty from assigning or sampling raters.

    The linearized standard error assumes independent samples across supplied
    strata, SRSWOR within each stratum, and weights that are constant within
    strata. Optional FPCs use target stratum sizes. Clustered, multistage,
    replicate-weight, and unequal-probability within-stratum designs are not
    supported. The overall estimate weights components before calculating
    kappa; it is not an average of stratum-specific kappas. The unclipped Wald
    interval may extend outside ``[-1, 1]``.

    Args:
        ratings: Raw item-by-rater nominal rating matrix.
        sample_weight: Optional strictly positive finite item weights.
        labels: Optional explicit category labels.
        strata: Optional sampling-stratum identifier per item.
        population_sizes: Optional target stratum sizes for finite-population
            corrections.
        confidence_level: Confidence level for the two-sided Wald interval.
        se_method: ``"linearization"`` for design-based inference or ``None``.
        nan_policy: ``"raise"`` or cell-wise ``"omit"``.

    Returns:
        An immutable :class:`KappaResult` with estimates and diagnostics.

    Raises:
        ValueError: If inputs are invalid, kappa is undefined, or the requested
            standard error is unsupported by the supplied design.
    """
    _validate_options(
        confidence_level=confidence_level,
        se_method=se_method,
        nan_policy=nan_policy,
        population_sizes=population_sizes,
    )
    rating_array, ratings_index = _as_2d_ratings(ratings)
    n_items, n_raters = rating_array.shape
    weights, weight_index = _validate_weights(sample_weight, n_items=n_items)
    stratum_values, strata_index, _ = _validate_strata(strata, n_items=n_items)
    _validate_indexes(
        (
            ("ratings", ratings_index),
            ("sample_weight", weight_index),
            ("strata", strata_index),
        )
    )
    if se_method == "linearization" and n_items < 2:
        raise ValueError("At least two analyzed items are required for a standard error.")

    flat = rating_array.ravel(order="C")
    missing = np.fromiter((_is_missing(value) for value in flat), dtype=bool)
    n_missing = int(missing.sum())
    if nan_policy == "raise" and n_missing:
        raise ValueError(
            "ratings contains missing values; use nan_policy='omit' to ignore "
            "missing rating cells."
        )
    valid = ~missing
    observed_values = flat[valid].tolist()
    result_labels, label_to_code = _validate_labels(observed_values, labels)
    n_categories = len(result_labels)

    row_index = np.repeat(np.arange(n_items, dtype=np.intp), n_raters)
    valid_rows = row_index[valid]
    valid_values = flat[valid]
    category_code = np.fromiter(
        (label_to_code[value] for value in valid_values),
        dtype=np.intp,
        count=len(valid_values),
    )
    flat_count_code = valid_rows * n_categories + category_code
    counts = np.bincount(
        flat_count_code,
        minlength=n_items * n_categories,
    ).reshape(n_items, n_categories)
    counts = counts.astype(np.int64, copy=False)
    rater_count = counts.sum(axis=1, dtype=np.int64)
    invalid_rows = np.flatnonzero(rater_count < 2)
    if invalid_rows.size:
        raise ValueError(
            "Every Fleiss item must retain at least two valid ratings; "
            f"invalid item positions: {invalid_rows.tolist()}."
        )

    pair_denominator = (
        rater_count.astype(np.float64) * (rater_count - 1)
    )
    item_agreement = (
        (counts * (counts - 1)).sum(axis=1, dtype=np.int64)
        / pair_denominator
    )
    item_category_proportions = counts / rater_count[:, None]

    weight_sum = weights.sum(dtype=np.float64)
    normalized_weight = weights / weight_sum
    observed_agreement = float(np.dot(normalized_weight, item_agreement))
    category_proportions = np.sum(
        normalized_weight[:, None] * item_category_proportions,
        axis=0,
        dtype=np.float64,
    )
    expected_agreement = float(
        np.dot(category_proportions, category_proportions)
    )
    (
        estimate,
        observed_agreement,
        expected_agreement,
    ) = _estimate_from_agreements(
        observed_agreement,
        expected_agreement,
    )

    linearized = _fleiss_linearized_values(
        item_agreement,
        item_category_proportions,
        observed_agreement,
        category_proportions,
    )

    return _build_result(
        estimate=estimate,
        observed_agreement=observed_agreement,
        expected_agreement=expected_agreement,
        n_items=n_items,
        labels=result_labels,
        weight_sum=weight_sum,
        weights=weights,
        strata=stratum_values,
        population_sizes=population_sizes,
        confidence_level=confidence_level,
        se_method=se_method,
        linearized=linearized,
        method="item-weighted nominal Fleiss/Scott kappa",
        metric_components={
            "category_proportions": _readonly_array(category_proportions),
            "minimum_raters_per_item": int(rater_count.min()),
            "maximum_raters_per_item": int(rater_count.max()),
            "mean_raters_per_item": float(rater_count.mean()),
            "n_missing_ratings_omitted": n_missing,
        },
    )
