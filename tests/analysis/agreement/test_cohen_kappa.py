"""Point-estimate and result-contract tests for item-weighted Cohen kappa."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

from fraudbench.analysis.agreement import (
    KappaResult,
    item_weighted_cohen_kappa,
)


@pytest.mark.parametrize(
    ("a", "b", "weights"),
    [
        ([0, 0, 1, 1], [0, 1, 1, 1], [1.0, 1.0, 1.0, 1.0]),
        (["a", "a", "b", "c"], ["a", "b", "b", "b"], [0.2, 1.7, 3.1, 0.5]),
        ([0, 0, 1, 1], [0, 2, 1, 2], [1.0, 2.0, 4.0, 3.0]),
    ],
)
def test_matches_sklearn_point_estimate(a, b, weights) -> None:
    result = item_weighted_cohen_kappa(
        a,
        b,
        sample_weight=weights,
        se_method=None,
    )
    expected = cohen_kappa_score(a, b, sample_weight=weights)
    assert result.estimate == pytest.approx(expected)


def test_integer_item_weights_equal_physical_replication() -> None:
    a = np.array(["red", "red", "blue", "green"], dtype=object)
    b = np.array(["red", "blue", "blue", "red"], dtype=object)
    weights = np.array([1, 3, 2, 4])

    weighted = item_weighted_cohen_kappa(
        a, b, sample_weight=weights, se_method=None
    )
    repeated = item_weighted_cohen_kappa(
        np.repeat(a, weights),
        np.repeat(b, weights),
        se_method=None,
    )
    assert weighted.estimate == pytest.approx(repeated.estimate)
    assert weighted.observed_agreement == pytest.approx(repeated.observed_agreement)
    assert weighted.expected_agreement == pytest.approx(repeated.expected_agreement)
    np.testing.assert_allclose(
        weighted.components["joint_proportions"],
        repeated.components["joint_proportions"],
    )


def test_specified_stratification_fixture() -> None:
    a = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    b = np.array([0, 0, 1, 0, 1, 1, 1, 0])
    weights = np.array([0.5] * 4 + [1.5] * 4)
    strata = np.array(["male"] * 4 + ["female"] * 4)

    result = item_weighted_cohen_kappa(
        a, b, sample_weight=weights, strata=strata
    )
    assert result.observed_agreement == pytest.approx(0.375)
    assert result.expected_agreement == pytest.approx(0.5)
    assert result.estimate == pytest.approx(-0.25)
    assert result.standard_error == pytest.approx(0.3528621809573817)
    assert result.design_degrees_of_freedom == 6


def test_swap_relabel_and_weight_scale_invariance() -> None:
    a = np.array(["one", "one", "two", "two", "one", "two"])
    b = np.array(["one", "two", "two", "one", "one", "two"])
    strata = np.array(["x"] * 3 + ["y"] * 3)
    weights = np.array([0.25] * 3 + [0.75] * 3)
    baseline = item_weighted_cohen_kappa(
        a, b, sample_weight=weights, strata=strata
    )
    swapped = item_weighted_cohen_kappa(
        b, a, sample_weight=weights, strata=strata
    )
    relabeled = item_weighted_cohen_kappa(
        np.where(a == "one", 99, -2),
        np.where(b == "one", 99, -2),
        sample_weight=weights,
        strata=strata,
    )
    scaled = item_weighted_cohen_kappa(
        a,
        b,
        sample_weight=weights * 1e100,
        strata=strata,
    )
    for other in (swapped, relabeled, scaled):
        assert other.estimate == pytest.approx(baseline.estimate)
        assert other.observed_agreement == pytest.approx(
            baseline.observed_agreement
        )
        assert other.expected_agreement == pytest.approx(
            baseline.expected_agreement
        )
        assert other.standard_error == pytest.approx(baseline.standard_error)


def test_complete_case_omission_and_diagnostics() -> None:
    result = item_weighted_cohen_kappa(
        [0, None, 1, 1],
        [0, 1, np.nan, 0],
        sample_weight=[1.0, 2.0, 3.0, 4.0],
        nan_policy="omit",
        se_method=None,
    )
    assert result.n_items == 2
    assert result.weight_sum == 5.0
    assert result.components["n_items_omitted"] == 2


def test_tuple_labels_and_mixed_labels_preserve_first_occurrence() -> None:
    tuple_a = [("a", 1), ("b", 2), ("a", 1), ("b", 2)]
    tuple_b = [("a", 1), ("a", 1), ("b", 2), ("b", 2)]
    result = item_weighted_cohen_kappa(tuple_a, tuple_b, se_method=None)
    assert result.labels == (("a", 1), ("b", 2))

    mixed = item_weighted_cohen_kappa(
        ["x", 2, "x", 2],
        [2, 2, "x", "x"],
        se_method=None,
    )
    assert mixed.labels == ("x", 2)


def test_pandas_inputs_require_identical_indexes() -> None:
    index = pd.Index([4, 2, 8], dtype="int64")
    a = pd.Series([0, 1, 0], index=index)
    b = pd.DataFrame({"rating": [0, 0, 1]}, index=index)
    weights = pd.DataFrame({"weight": [1.0, 1.0, 1.0]}, index=index)
    result = item_weighted_cohen_kappa(
        a, b, sample_weight=weights, se_method=None
    )
    assert result.n_items == 3

    with pytest.raises(ValueError, match="Pandas indexes"):
        item_weighted_cohen_kappa(
            a,
            b.iloc[::-1],
            sample_weight=weights,
            se_method=None,
        )


def test_result_and_nested_diagnostics_are_immutable() -> None:
    result = item_weighted_cohen_kappa(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        se_method=None,
    )
    assert isinstance(result, KappaResult)
    with pytest.raises(FrozenInstanceError):
        result.estimate = 0.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.components["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        result.components["stratum_sample_sizes"]["__all__"] = 0  # type: ignore[index]
    with pytest.raises(ValueError):
        result.components["marginal_a"][0] = 0.0


def test_perfect_agreement_has_one_estimate_and_zero_se() -> None:
    result = item_weighted_cohen_kappa([0, 0, 1, 1], [0, 0, 1, 1])
    assert result.estimate == pytest.approx(1.0)
    assert result.standard_error == pytest.approx(0.0)
