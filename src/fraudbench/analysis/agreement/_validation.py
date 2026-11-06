"""Shared input validation for item-weighted agreement statistics."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def _is_missing(value: object) -> bool:
    """Return whether a scalar identifier is a supported missing sentinel."""
    if isinstance(value, tuple):
        return any(_is_missing(element) for element in value)
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if np.ndim(result) == 0 else False


def _object_array(values: Sequence[object]) -> NDArray[np.object_]:
    """Build a one-dimensional object array without expanding tuple values."""
    result = np.empty(len(values), dtype=object)
    result[:] = list(values)
    return result


def _as_1d(
    value: ArrayLike,
    *,
    name: str,
) -> tuple[NDArray[np.object_], pd.Index | None]:
    """Convert a one-dimensional argument while retaining a pandas row index."""
    row_index: pd.Index | None = None
    if isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            raise ValueError(
                f"{name} must be one-dimensional or a one-column DataFrame; "
                f"received {value.shape[1]} columns."
            )
        row_index = value.index
        values = value.iloc[:, 0].tolist()
    elif isinstance(value, pd.Series):
        row_index = value.index
        values = value.tolist()
    elif isinstance(value, pd.Index):
        values = value.tolist()
    elif isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
        values = value.tolist()
    else:
        try:
            values = list(value)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError(f"{name} must be one-dimensional.") from error

    return _object_array(values), row_index


def _as_2d_ratings(
    ratings: ArrayLike,
) -> tuple[NDArray[np.object_], pd.Index | None]:
    """Convert a rectangular item-by-rater input to a two-dimensional array."""
    row_index: pd.Index | None = None
    if isinstance(ratings, pd.DataFrame):
        row_index = ratings.index
        array = ratings.to_numpy(dtype=object, copy=True)
    elif isinstance(ratings, np.ndarray):
        array = np.asarray(ratings, dtype=object)
    else:
        try:
            rows = list(ratings)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ValueError("ratings must be a rectangular two-dimensional array.") from error
        if not rows:
            raise ValueError("ratings must contain at least one item.")
        converted_rows: list[list[object]] = []
        for row in rows:
            if isinstance(row, (str, bytes)):
                raise ValueError("ratings must be two-dimensional.")
            try:
                converted_rows.append(list(row))  # type: ignore[arg-type]
            except TypeError as error:
                raise ValueError("ratings must be two-dimensional.") from error
        widths = {len(row) for row in converted_rows}
        if len(widths) != 1:
            raise ValueError("ratings must be a rectangular two-dimensional array.")
        width = widths.pop()
        array = np.empty((len(converted_rows), width), dtype=object)
        for position, row in enumerate(converted_rows):
            array[position, :] = row

    if array.ndim != 2:
        raise ValueError("ratings must be two-dimensional.")
    if array.shape[0] == 0:
        raise ValueError("ratings must contain at least one item.")
    if array.shape[1] < 2:
        raise ValueError("ratings must contain at least two rater columns.")
    return array.copy(), row_index


def _indexes_equal(left: pd.Index, right: pd.Index) -> bool:
    """Compare pandas indexes without permitting dtype-changing alignment."""
    if isinstance(left, pd.MultiIndex) != isinstance(right, pd.MultiIndex):
        return False
    if isinstance(left, pd.MultiIndex):
        assert isinstance(right, pd.MultiIndex)
        if left.nlevels != right.nlevels:
            return False
        if any(a != b for a, b in zip(left.dtypes, right.dtypes, strict=True)):
            return False
    elif left.dtype != right.dtype:
        return False
    return bool(left.equals(right))


def _validate_indexes(
    indexed_objects: Sequence[tuple[str, pd.Index | None]],
) -> None:
    """Require every supplied pandas row index to be exactly identical."""
    present = [(name, index) for name, index in indexed_objects if index is not None]
    if len(present) < 2:
        return
    reference_name, reference = present[0]
    assert reference is not None
    for name, index in present[1:]:
        assert index is not None
        if not _indexes_equal(reference, index):
            raise ValueError(
                f"Pandas indexes for {reference_name} and {name} must be identical "
                "in values, dtype, and order; inputs are never aligned automatically."
            )


def _validate_options(
    *,
    confidence_level: float,
    se_method: Literal["linearization"] | None,
    nan_policy: Literal["raise", "omit"],
    population_sizes: int | Mapping[Hashable, int] | None,
) -> None:
    """Validate public scalar options shared by both estimators."""
    if not isinstance(nan_policy, str) or nan_policy not in {"raise", "omit"}:
        raise ValueError("nan_policy must be either 'raise' or 'omit'.")
    if se_method is not None and (
        not isinstance(se_method, str) or se_method != "linearization"
    ):
        raise ValueError("se_method must be 'linearization' or None.")
    if isinstance(confidence_level, (bool, np.bool_)) or not isinstance(
        confidence_level, Real
    ):
        raise ValueError("confidence_level must be a finite real number in (0, 1).")
    confidence = float(confidence_level)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be a finite real number in (0, 1).")
    if se_method is None and population_sizes is not None:
        raise ValueError("population_sizes must be None when se_method=None.")


def _validate_weights(
    sample_weight: ArrayLike | None,
    *,
    n_items: int,
) -> tuple[NDArray[np.float64], pd.Index | None]:
    """Validate positive finite item weights and return a private float64 copy."""
    if sample_weight is None:
        return np.ones(n_items, dtype=np.float64), None

    raw, index = _as_1d(sample_weight, name="sample_weight")
    if len(raw) != n_items:
        raise ValueError(
            f"sample_weight must have length {n_items}; received {len(raw)}."
        )
    for value in raw:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("sample_weight must contain numeric values, not booleans.")
        if not isinstance(value, Real):
            raise ValueError(
                f"sample_weight must contain numeric values; received {value!r}."
            )
    try:
        weights = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("sample_weight must contain numeric values.") from error
    if not np.all(np.isfinite(weights)):
        raise ValueError("sample_weight must contain only finite values.")
    if not np.all(weights > 0.0):
        raise ValueError("sample_weight values must be strictly positive.")
    with np.errstate(over="ignore", invalid="ignore"):
        weight_sum = weights.sum(dtype=np.float64)
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("The float64 sum of sample_weight must be finite and positive.")
    return weights.copy(), index


def _validate_hashable(
    value: object,
    *,
    argument: str,
) -> Hashable:
    """Validate a nonmissing hashable category or stratum identifier."""
    if _is_missing(value):
        raise ValueError(f"{argument} contains a missing value.")
    try:
        hash(value)
    except TypeError as error:
        raise ValueError(f"{argument} contains an unhashable value: {value!r}.") from error
    return value  # type: ignore[return-value]


def _validate_strata(
    strata: ArrayLike | None,
    *,
    n_items: int,
) -> tuple[NDArray[np.object_], pd.Index | None, bool]:
    """Validate stratum identifiers and preserve first-occurrence semantics."""
    if strata is None:
        result = np.empty(n_items, dtype=object)
        result[:] = "__all__"
        return result, None, True

    raw, index = _as_1d(strata, name="strata")
    if len(raw) != n_items:
        raise ValueError(f"strata must have length {n_items}; received {len(raw)}.")
    for value in raw:
        _validate_hashable(value, argument="strata")
    return raw.copy(), index, False


def _validate_labels(
    observed: Sequence[object],
    labels: Sequence[Hashable] | None,
) -> tuple[tuple[Hashable, ...], dict[Hashable, int]]:
    """Validate or infer labels using Python equality and hashing semantics."""
    if labels is None:
        ordered: list[Hashable] = []
        mapping: dict[Hashable, int] = {}
        for value in observed:
            valid = _validate_hashable(value, argument="annotations")
            if valid not in mapping:
                mapping[valid] = len(ordered)
                ordered.append(valid)
        return tuple(ordered), mapping

    try:
        supplied = list(labels)
    except TypeError as error:
        raise ValueError("labels must be a nonempty sequence.") from error
    if not supplied:
        raise ValueError("labels must contain at least one category.")
    ordered = []
    mapping = {}
    for value in supplied:
        valid = _validate_hashable(value, argument="labels")
        if valid in mapping:
            raise ValueError(f"labels contains a duplicate category: {value!r}.")
        mapping[valid] = len(ordered)
        ordered.append(valid)

    for value in observed:
        valid = _validate_hashable(value, argument="annotations")
        if valid not in mapping:
            raise ValueError(
                f"Observed category {value!r} is absent from explicit labels."
            )
    return tuple(ordered), mapping


def _ordered_groups(
    strata: NDArray[np.object_],
) -> tuple[tuple[Hashable, NDArray[np.intp]], ...]:
    """Return stratum indexes in stable first-occurrence order."""
    order: list[Hashable] = []
    positions: dict[Hashable, list[int]] = {}
    for position, raw in enumerate(strata):
        value = raw  # already validated as hashable
        if value not in positions:
            order.append(value)
            positions[value] = []
        positions[value].append(position)
    return tuple(
        (value, np.asarray(positions[value], dtype=np.intp)) for value in order
    )


def _validate_population_size(value: object, *, stratum: Hashable) -> int:
    """Validate one finite-population stratum count."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(
            f"population size for stratum {stratum!r} must be a positive integer."
        )
    size = int(value)
    if size <= 0:
        raise ValueError(
            f"population size for stratum {stratum!r} must be a positive integer."
        )
    return size


def _validated_population_sizes(
    population_sizes: int | Mapping[Hashable, int] | None,
    groups: Sequence[tuple[Hashable, NDArray[np.intp]]],
) -> dict[Hashable, int] | None:
    """Validate finite-population counts with exact observed-stratum coverage."""
    if population_sizes is None:
        return None

    observed = [stratum for stratum, _ in groups]
    if isinstance(population_sizes, Mapping):
        supplied: dict[Hashable, object] = {}
        for key, value in population_sizes.items():
            valid_key = _validate_hashable(key, argument="population_sizes keys")
            supplied[valid_key] = value
        missing = [key for key in observed if key not in supplied]
        extra = [key for key in supplied if key not in set(observed)]
        if missing or extra:
            raise ValueError(
                "population_sizes mapping must exactly cover analyzed strata; "
                f"missing={missing!r}, extra={extra!r}."
            )
        result = {
            key: _validate_population_size(supplied[key], stratum=key)
            for key in observed
        }
    else:
        if len(observed) != 1:
            raise ValueError(
                "population_sizes must be a mapping when multiple strata are observed."
            )
        result = {
            observed[0]: _validate_population_size(
                population_sizes, stratum=observed[0]
            )
        }

    for stratum, index in groups:
        if result[stratum] < len(index):
            raise ValueError(
                f"population size N_h={result[stratum]} for stratum {stratum!r} "
                f"cannot be smaller than sample size n_h={len(index)}."
            )
    return result


def _common_stratum_diagnostics(
    weights: NDArray[np.float64],
    strata: NDArray[np.object_],
) -> tuple[dict[Hashable, int], dict[Hashable, float]]:
    """Calculate stable sample-size and target-share diagnostics."""
    groups = _ordered_groups(strata)
    weight_sum = weights.sum(dtype=np.float64)
    sample_sizes = {stratum: int(len(index)) for stratum, index in groups}
    weight_shares = {
        stratum: float(weights[index].sum(dtype=np.float64) / weight_sum)
        for stratum, index in groups
    }
    return sample_sizes, weight_shares
