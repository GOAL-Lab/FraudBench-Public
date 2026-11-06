"""Point-estimate and edge-case tests for item-weighted Fleiss kappa."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

from fraudbench.analysis.agreement import item_weighted_fleiss_kappa


def test_matches_statsmodels_for_complete_equal_rater_data() -> None:
    ratings = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [1, 1, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
        ]
    )
    table, _ = aggregate_raters(ratings)
    expected = fleiss_kappa(table)
    result = item_weighted_fleiss_kappa(ratings, se_method=None)
    assert result.estimate == pytest.approx(expected)


def test_specified_stratification_fixture() -> None:
    ratings = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [1, 1, 1],
            [1, 1, 0],
            [0, 1, 1],
            [0, 1, 0],
            [1, 0, 1],
            [1, 0, 0],
        ]
    )
    weights = np.array([0.5] * 4 + [1.5] * 4)
    strata = np.array(["male"] * 4 + ["female"] * 4)
    result = item_weighted_fleiss_kappa(
        ratings, sample_weight=weights, strata=strata
    )
    assert result.observed_agreement == pytest.approx(5.0 / 12.0)
    assert result.expected_agreement == pytest.approx(0.5)
    assert result.estimate == pytest.approx(-1.0 / 6.0)
    assert result.standard_error == pytest.approx(0.09622504486493763)
    assert result.design_degrees_of_freedom == 6


def test_integer_item_weights_equal_replication() -> None:
    ratings = np.array(
        [[0, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 0]]
    )
    weights = np.array([1, 3, 2, 4])
    weighted = item_weighted_fleiss_kappa(
        ratings, sample_weight=weights, se_method=None
    )
    repeated = item_weighted_fleiss_kappa(
        np.repeat(ratings, weights, axis=0),
        se_method=None,
    )
    assert weighted.estimate == pytest.approx(repeated.estimate)
    assert weighted.observed_agreement == pytest.approx(repeated.observed_agreement)
    assert weighted.expected_agreement == pytest.approx(repeated.expected_agreement)
    np.testing.assert_allclose(
        weighted.components["category_proportions"],
        repeated.components["category_proportions"],
    )


def test_missing_cells_are_omitted_without_reweighting_by_rater_count() -> None:
    ratings = [
        ["yes", "yes", None, "no"],
        ["no", "no", "yes", "no"],
        ["yes", pd.NA, "yes", "yes"],
    ]
    result = item_weighted_fleiss_kappa(
        ratings,
        sample_weight=[2.0, 1.0, 3.0],
        nan_policy="omit",
        se_method=None,
    )
    assert result.n_items == 3
    assert result.components["n_missing_ratings_omitted"] == 2
    assert result.components["minimum_raters_per_item"] == 3
    assert result.components["maximum_raters_per_item"] == 4
    assert result.components["mean_raters_per_item"] == pytest.approx(10 / 3)


def test_tuple_categories_are_scalar_cells() -> None:
    a = ("label", 1)
    b = ("label", 2)
    ratings = [[a, a, b], [b, b, b], [a, b, a]]
    result = item_weighted_fleiss_kappa(ratings, se_method=None)
    assert result.labels == (a, b)


def test_pandas_frame_aligns_only_by_identical_row_index() -> None:
    index = pd.Index(["i3", "i1", "i2"])
    ratings = pd.DataFrame(
        {"r1": [0, 1, 0], "r2": [0, 1, 1], "r3": [1, 1, 0]},
        index=index,
    )
    weights = pd.Series([1.0, 2.0, 3.0], index=index)
    strata = pd.DataFrame({"s": ["a", "a", "a"]}, index=index)
    result = item_weighted_fleiss_kappa(
        ratings,
        sample_weight=weights,
        strata=strata,
        se_method=None,
    )
    assert result.n_items == 3

    with pytest.raises(ValueError, match="Pandas indexes"):
        item_weighted_fleiss_kappa(
            ratings,
            sample_weight=weights.iloc[::-1],
            strata=strata,
            se_method=None,
        )


def test_perfect_agreement_over_two_categories() -> None:
    result = item_weighted_fleiss_kappa(
        [[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]]
    )
    assert result.estimate == pytest.approx(1.0)
    assert result.standard_error == pytest.approx(0.0)


def test_weight_scale_invariance() -> None:
    ratings = np.array(
        [[0, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 0], [1, 0, 1], [1, 1, 0]]
    )
    strata = np.array(["a"] * 3 + ["b"] * 3)
    weights = np.array([0.5] * 3 + [1.5] * 3)
    baseline = item_weighted_fleiss_kappa(
        ratings, sample_weight=weights, strata=strata
    )
    for scale in (1e-100, 1e100):
        scaled = item_weighted_fleiss_kappa(
            ratings,
            sample_weight=weights * scale,
            strata=strata,
        )
        assert scaled.estimate == pytest.approx(baseline.estimate)
        assert scaled.standard_error == pytest.approx(baseline.standard_error)
        np.testing.assert_allclose(
            scaled.components["category_proportions"],
            baseline.components["category_proportions"],
        )
