"""Design-based standard-error tests shared by both kappa estimators."""

from __future__ import annotations

import numpy as np
import pytest

from fraudbench.analysis.agreement import (
    item_weighted_cohen_kappa,
    item_weighted_fleiss_kappa,
)
from fraudbench.analysis.agreement.kappa import (
    _cohen_component_gradients,
    _cohen_linearized_values,
    _fleiss_component_gradients,
    _fleiss_linearized_values,
)
from fraudbench.analysis.agreement._linearization import (
    _stratified_linearized_se,
)


def _central_difference(function, values: np.ndarray, position: int) -> float:
    step = 1e-6
    lower = values.copy()
    upper = values.copy()
    lower[position] -= step
    upper[position] += step
    return float((function(upper) - function(lower)) / (2.0 * step))


def test_cohen_analytic_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(20260726)
    for _ in range(5):
        marginal_a = rng.dirichlet(np.ones(3))
        marginal_b = rng.dirichlet(np.ones(3))
        observed = float(rng.uniform(0.2, 0.8))
        values = np.concatenate(([observed], marginal_a, marginal_b))

        def coefficient(component: np.ndarray) -> float:
            expected = float(np.dot(component[1:4], component[4:7]))
            return float((component[0] - expected) / (1.0 - expected))

        d_observed, gradient_a, gradient_b = _cohen_component_gradients(
            observed,
            marginal_a,
            marginal_b,
        )
        analytic = np.concatenate(([d_observed], gradient_a, gradient_b))
        numeric = np.array(
            [_central_difference(coefficient, values, index) for index in range(7)]
        )
        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_fleiss_analytic_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(20260727)
    for _ in range(5):
        proportions = rng.dirichlet(np.ones(4))
        observed = float(rng.uniform(0.2, 0.8))
        values = np.concatenate(([observed], proportions))

        def coefficient(component: np.ndarray) -> float:
            expected = float(np.dot(component[1:], component[1:]))
            return float((component[0] - expected) / (1.0 - expected))

        d_observed, gradient = _fleiss_component_gradients(
            observed,
            proportions,
        )
        analytic = np.concatenate(([d_observed], gradient))
        numeric = np.array(
            [_central_difference(coefficient, values, index) for index in range(5)]
        )
        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_item_linearized_values_have_zero_weighted_mean() -> None:
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    normalized = weights / weights.sum()

    a_code = np.array([0, 0, 1, 1], dtype=np.intp)
    b_code = np.array([0, 1, 1, 0], dtype=np.intp)
    agreement = (a_code == b_code).astype(np.float64)
    marginal_a = np.bincount(a_code, weights=normalized, minlength=2)
    marginal_b = np.bincount(b_code, weights=normalized, minlength=2)
    observed = float(np.dot(normalized, agreement))
    cohen_values = _cohen_linearized_values(
        agreement,
        a_code,
        b_code,
        observed,
        marginal_a,
        marginal_b,
    )
    assert np.dot(normalized, cohen_values) == pytest.approx(0.0, abs=1e-14)

    item_category = np.array(
        [[2 / 3, 1 / 3], [0.0, 1.0], [1 / 3, 2 / 3], [1.0, 0.0]]
    )
    item_agreement = np.array([1 / 3, 1.0, 1 / 3, 1.0])
    category = np.sum(normalized[:, None] * item_category, axis=0)
    observed = float(np.dot(normalized, item_agreement))
    fleiss_values = _fleiss_linearized_values(
        item_agreement,
        item_category,
        observed,
        category,
    )
    assert np.dot(normalized, fleiss_values) == pytest.approx(0.0, abs=1e-14)


def test_unequal_within_stratum_weights_require_point_only_mode() -> None:
    a = [0, 0, 1, 1]
    b = [0, 1, 1, 0]
    weights = [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(ValueError, match="constant item weights"):
        item_weighted_cohen_kappa(a, b, sample_weight=weights)
    result = item_weighted_cohen_kappa(
        a, b, sample_weight=weights, se_method=None
    )
    assert result.standard_error is None
    assert result.confidence_interval is None
    assert result.design_degrees_of_freedom is None
    assert result.se_method is None
    assert "stratum_variance_contributions" not in result.components


def test_strata_change_se_but_not_point_estimate() -> None:
    a = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    b = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    unstratified = item_weighted_cohen_kappa(a, b)
    stratified = item_weighted_cohen_kappa(
        a, b, strata=np.array(["a"] * 4 + ["b"] * 4)
    )
    assert stratified.estimate == pytest.approx(unstratified.estimate)
    assert stratified.standard_error != pytest.approx(
        unstratified.standard_error
    )


def test_census_fpc_gives_zero_se_and_point_interval() -> None:
    ratings = np.array(
        [[0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 0]]
    )
    result = item_weighted_fleiss_kappa(
        ratings,
        population_sizes=4,
    )
    assert result.standard_error == 0.0
    assert result.confidence_interval == (result.estimate, result.estimate)
    assert result.components["finite_population_correction_applied"] is True
    assert result.components["stratum_variance_contributions"]["__all__"] == 0.0


def test_fpc_cannot_increase_standard_error() -> None:
    a = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    b = np.array([0, 1, 1, 0, 1, 1, 0, 0])
    strata = np.array(["x"] * 4 + ["y"] * 4)
    weights = np.array([2.5] * 4 + [5.0] * 4)
    without = item_weighted_cohen_kappa(
        a, b, sample_weight=weights, strata=strata
    )
    with_fpc = item_weighted_cohen_kappa(
        a,
        b,
        sample_weight=weights,
        strata=strata,
        population_sizes={"x": 10, "y": 20},
    )
    assert with_fpc.standard_error <= without.standard_error


def test_stratum_specific_fpcs_use_the_matching_population_sizes() -> None:
    linearized = np.array([-2.0, 0.0, 2.0, -3.0, -1.0, 1.0, 3.0])
    strata = np.array(["high"] * 3 + ["low"] * 4, dtype=object)
    population_sizes = {"high": 4, "low": 20}
    weights = np.array([4 / 3] * 3 + [5.0] * 4)

    result = _stratified_linearized_se(
        linearized,
        weights,
        strata,
        population_sizes,
    )

    total_population = sum(population_sizes.values())
    expected_high = (
        (population_sizes["high"] / total_population) ** 2
        * (1.0 - 3 / population_sizes["high"])
        * np.var(linearized[:3], ddof=1)
        / 3
    )
    expected_low = (
        (population_sizes["low"] / total_population) ** 2
        * (1.0 - 4 / population_sizes["low"])
        * np.var(linearized[3:], ddof=1)
        / 4
    )
    assert result.stratum_variance_contributions["high"] == pytest.approx(
        expected_high
    )
    assert result.stratum_variance_contributions["low"] == pytest.approx(
        expected_low
    )
    assert result.standard_error == pytest.approx(
        np.sqrt(expected_high + expected_low)
    )


@pytest.mark.parametrize(
    ("population_sizes", "match"),
    [
        (None, "one sampled item"),
        ({"singleton": 2, "regular": 3}, "one sampled item"),
    ],
)
def test_noncertainty_singleton_stratum_raises(population_sizes, match) -> None:
    kwargs = {} if population_sizes is None else {"population_sizes": population_sizes}
    weights = (
        [1.0, 1.0, 1.0, 1.0]
        if population_sizes is None
        else [2.0, 1.0, 1.0, 1.0]
    )
    with pytest.raises(ValueError, match=match):
        item_weighted_cohen_kappa(
            [0, 0, 1, 1],
            [0, 1, 1, 0],
            strata=["singleton", "regular", "regular", "regular"],
            sample_weight=weights,
            **kwargs,
        )


def test_certainty_singleton_contributes_zero() -> None:
    result = item_weighted_cohen_kappa(
        [0, 0, 1],
        [0, 1, 1],
        strata=["certainty", "sampled", "sampled"],
        sample_weight=[1.0, 2.0, 2.0],
        population_sizes={"certainty": 1, "sampled": 4},
    )
    assert result.components["stratum_variance_contributions"]["certainty"] == 0.0
    assert result.design_degrees_of_freedom == 1


def test_all_certainty_singletons_are_valid_zero_df_design() -> None:
    result = item_weighted_cohen_kappa(
        [0, 1],
        [0, 1],
        strata=["a", "b"],
        population_sizes={"a": 1, "b": 1},
    )
    assert result.standard_error == 0.0
    assert result.design_degrees_of_freedom == 0
    assert result.confidence_interval == (1.0, 1.0)


def test_population_mapping_requires_exact_coverage_and_valid_sizes() -> None:
    common = {
        "annotations_a": [0, 0, 1, 1],
        "annotations_b": [0, 1, 1, 0],
        "strata": ["a", "a", "b", "b"],
    }
    with pytest.raises(ValueError, match="exactly cover"):
        item_weighted_cohen_kappa(
            **common, population_sizes={"a": 2, "b": 2, "extra": 1}
        )
    with pytest.raises(ValueError, match="positive integer"):
        item_weighted_cohen_kappa(
            **common, population_sizes={"a": True, "b": 2}
        )
    with pytest.raises(ValueError, match="cannot be smaller"):
        item_weighted_cohen_kappa(
            **common, population_sizes={"a": 1, "b": 2}
        )


def test_population_and_weight_shares_must_agree() -> None:
    with pytest.raises(ValueError, match="shares disagree"):
        item_weighted_cohen_kappa(
            [0, 0, 1, 1],
            [0, 1, 1, 0],
            strata=["a", "a", "b", "b"],
            sample_weight=[1.0, 1.0, 1.0, 1.0],
            population_sizes={"a": 2, "b": 6},
        )


def test_weighted_linearized_values_have_zero_mean_indirectly() -> None:
    # Global replication of every item leaves both point estimate and the
    # component-centered influence calculation unchanged.
    ratings = np.array(
        [[0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 0]]
    )
    baseline = item_weighted_fleiss_kappa(ratings)
    duplicated = item_weighted_fleiss_kappa(np.repeat(ratings, 2, axis=0))
    assert duplicated.estimate == pytest.approx(baseline.estimate)
