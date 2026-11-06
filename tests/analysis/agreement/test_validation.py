"""Public validation behavior for both item-weighted kappa estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudbench.analysis.agreement import (
    item_weighted_cohen_kappa,
    item_weighted_fleiss_kappa,
)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"nan_policy": "drop"}, "nan_policy"),
        ({"se_method": "bootstrap"}, "se_method"),
        ({"confidence_level": 0.0}, "confidence_level"),
        ({"confidence_level": 1.0}, "confidence_level"),
        ({"confidence_level": np.inf}, "confidence_level"),
        ({"confidence_level": True}, "confidence_level"),
        ({"se_method": None, "population_sizes": 10}, "population_sizes"),
    ],
)
def test_shared_invalid_options(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        item_weighted_cohen_kappa([0, 1], [0, 1], **kwargs)


@pytest.mark.parametrize(
    "weights",
    [
        [1.0],
        [1.0, 0.0],
        [1.0, -1.0],
        [1.0, np.inf],
        [1.0, np.nan],
        [True, False],
        ["one", "two"],
    ],
)
def test_invalid_weights_raise(weights) -> None:
    with pytest.raises(ValueError, match="sample_weight"):
        item_weighted_cohen_kappa(
            [0, 1],
            [0, 1],
            sample_weight=weights,
            se_method=None,
        )


def test_weight_sum_overflow_raises() -> None:
    with pytest.raises(ValueError, match="sum"):
        item_weighted_cohen_kappa(
            [0, 1],
            [0, 1],
            sample_weight=[np.finfo(np.float64).max] * 2,
            se_method=None,
        )


@pytest.mark.parametrize(
    "missing",
    [
        None,
        np.nan,
        np.float64("nan"),
        pd.NA,
        pd.NaT,
        np.datetime64("NaT"),
        np.timedelta64("NaT"),
    ],
)
def test_supported_missing_sentinels_raise(missing) -> None:
    with pytest.raises(ValueError, match="missing"):
        item_weighted_cohen_kappa([0, missing], [0, 1], se_method=None)


def test_missing_like_strings_are_valid_categories() -> None:
    result = item_weighted_cohen_kappa(
        ["nan", "NaT", "None", ""],
        ["nan", "None", "None", ""],
        se_method=None,
    )
    assert result.labels == ("nan", "NaT", "None", "")


def test_explicit_label_validation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        item_weighted_cohen_kappa([0, 1], [0, 1], labels=[], se_method=None)
    with pytest.raises(ValueError, match="duplicate"):
        item_weighted_cohen_kappa(
            [0, 1], [0, 1], labels=[0, False, 1], se_method=None
        )
    with pytest.raises(ValueError, match="absent"):
        item_weighted_cohen_kappa(
            [0, 1], [0, 2], labels=[0, 1], se_method=None
        )
    with pytest.raises(ValueError, match="unhashable"):
        item_weighted_cohen_kappa(
            [0, 1], [0, 1], labels=[0, ["bad"]], se_method=None
        )


def test_nested_missing_tuple_identifier_rejected() -> None:
    with pytest.raises(ValueError, match="missing"):
        item_weighted_cohen_kappa(
            [("ok", 1), ("bad", None)],
            [("ok", 1), ("ok", 1)],
            se_method=None,
        )
    with pytest.raises(ValueError, match="strata"):
        item_weighted_cohen_kappa(
            [0, 1],
            [0, 1],
            strata=[("ok", 1), ("bad", np.nan)],
            se_method=None,
        )


def test_shapes_and_empty_inputs_raise() -> None:
    with pytest.raises(ValueError, match="empty"):
        item_weighted_cohen_kappa([], [], se_method=None)
    with pytest.raises(ValueError, match="equal length"):
        item_weighted_cohen_kappa([0], [0, 1], se_method=None)
    with pytest.raises(ValueError, match="one-dimensional"):
        item_weighted_cohen_kappa(np.zeros((2, 2)), [0, 1], se_method=None)
    with pytest.raises(ValueError, match="one-column"):
        item_weighted_cohen_kappa(
            pd.DataFrame({"a": [0, 1], "b": [1, 0]}),
            [0, 1],
            se_method=None,
        )
    with pytest.raises(ValueError, match="two-dimensional"):
        item_weighted_fleiss_kappa([0, 1], se_method=None)
    with pytest.raises(ValueError, match="two rater"):
        item_weighted_fleiss_kappa([[0], [1]], se_method=None)


def test_undefined_kappa_raises() -> None:
    with pytest.raises(ValueError, match="undefined"):
        item_weighted_cohen_kappa([1, 1], [1, 1], se_method=None)
    with pytest.raises(ValueError, match="undefined"):
        item_weighted_fleiss_kappa([[1, 1], [1, 1]], se_method=None)


def test_fleiss_item_with_too_few_valid_ratings_raises() -> None:
    with pytest.raises(ValueError, match="at least two"):
        item_weighted_fleiss_kappa(
            [[0, None, None], [0, 1, 1]],
            nan_policy="omit",
            se_method=None,
        )


def test_all_cohen_items_omitted_raises() -> None:
    with pytest.raises(ValueError, match="No Cohen items"):
        item_weighted_cohen_kappa(
            [None, np.nan],
            [0, None],
            nan_policy="omit",
            se_method=None,
        )


def test_weight_and_strata_are_validated_before_cohen_omission() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        item_weighted_cohen_kappa(
            [None, 0],
            [0, 0],
            sample_weight=[0.0, 1.0],
            nan_policy="omit",
            se_method=None,
        )
    with pytest.raises(ValueError, match="strata"):
        item_weighted_cohen_kappa(
            [None, 0],
            [0, 0],
            strata=[None, "ok"],
            nan_policy="omit",
            se_method=None,
        )


def test_positional_pandas_index_objects_do_not_trigger_alignment() -> None:
    result = item_weighted_cohen_kappa(
        pd.Series([0, 1], index=["a", "b"]),
        pd.Series([0, 1], index=["a", "b"]),
        sample_weight=pd.Index([1.0, 2.0]),
        strata=pd.Index(["all", "all"]),
        se_method=None,
    )
    assert result.n_items == 2


def test_equivalent_range_and_integer_indexes_are_accepted() -> None:
    result = item_weighted_cohen_kappa(
        pd.Series([0, 1, 0], index=pd.RangeIndex(3)),
        pd.Series([0, 1, 1], index=pd.Index([0, 1, 2], dtype="int64")),
        se_method=None,
    )
    assert result.n_items == 3


@pytest.mark.parametrize("argument", ["sample_weight", "strata"])
@pytest.mark.parametrize("columns", [0, 2])
def test_weight_and_strata_dataframes_require_one_column(
    argument: str,
    columns: int,
) -> None:
    frame = pd.DataFrame(index=range(2))
    for position in range(columns):
        frame[f"c{position}"] = [1, 1]
    kwargs = {argument: frame, "se_method": None}
    with pytest.raises(ValueError, match="one-column"):
        item_weighted_cohen_kappa([0, 1], [0, 1], **kwargs)


def test_exact_component_keys_and_native_scalar_contract() -> None:
    cohen = item_weighted_cohen_kappa(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
    )
    assert set(cohen.components) == {
        "joint_proportions",
        "marginal_a",
        "marginal_b",
        "n_items_omitted",
        "stratum_sample_sizes",
        "stratum_weight_shares",
        "stratum_variance_contributions",
        "finite_population_correction_applied",
    }
    fleiss = item_weighted_fleiss_kappa(
        [[0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 0]],
        se_method=None,
    )
    assert set(fleiss.components) == {
        "category_proportions",
        "minimum_raters_per_item",
        "maximum_raters_per_item",
        "mean_raters_per_item",
        "n_missing_ratings_omitted",
        "stratum_sample_sizes",
        "stratum_weight_shares",
        "finite_population_correction_applied",
    }
    for result in (cohen, fleiss):
        assert type(result.estimate) is float
        assert type(result.observed_agreement) is float
        assert type(result.expected_agreement) is float
        assert type(result.weight_sum) is float
        assert type(result.n_items) is int
        assert type(result.n_strata) is int
    assert type(cohen.standard_error) is float
    assert type(cohen.confidence_interval[0]) is float
    assert type(cohen.design_degrees_of_freedom) is int
    assert fleiss.standard_error is None
    assert fleiss.confidence_interval is None
    assert fleiss.design_degrees_of_freedom is None


def test_diagnostic_arrays_do_not_share_writable_input_memory() -> None:
    weights = np.ones(4)
    result = item_weighted_cohen_kappa(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
        sample_weight=weights,
        se_method=None,
    )
    before = result.components["marginal_a"].copy()
    weights[:] = 100.0
    np.testing.assert_array_equal(result.components["marginal_a"], before)
    assert result.components["marginal_a"].flags.writeable is False
