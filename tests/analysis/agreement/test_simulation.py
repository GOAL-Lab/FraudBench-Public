"""Slow finite-population validation of Taylor-linearized standard errors."""

from __future__ import annotations

import numpy as np
import pytest

from fraudbench.analysis.agreement import (
    item_weighted_cohen_kappa,
    item_weighted_fleiss_kappa,
)


def _three_category_populations(
    population_sizes: dict[str, int],
    category_patterns: dict[str, tuple[int, ...]],
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, np.ndarray],
]:
    """Construct deterministic populations with unbalanced category marginals."""
    cohen: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fleiss: dict[str, np.ndarray] = {}
    for position, (stratum, size) in enumerate(population_sizes.items()):
        pattern = np.asarray(category_patterns[stratum], dtype=np.int64)
        base = np.resize(pattern, size)
        item = np.arange(size)

        other = base.copy()
        other_mask = (item + position) % 5 == 0
        other[other_mask] = (base[other_mask] + 1 + position % 2) % 3

        third = base.copy()
        third_mask = (item + 2 * position) % 7 == 0
        third[third_mask] = (base[third_mask] + 2) % 3

        cohen[stratum] = (base, other)
        fleiss[stratum] = np.column_stack((base, other, third))
    return cohen, fleiss


def _simulate_design(
    *,
    population_sizes: dict[str, int],
    sample_sizes: dict[str, int],
    category_patterns: dict[str, tuple[int, ...]],
    seed: int,
    repetitions: int,
) -> tuple[float, float]:
    """Return mean-SE/Monte-Carlo-SD ratios for Cohen and Fleiss kappa."""
    rng = np.random.default_rng(seed)
    cohen_populations, fleiss_populations = _three_category_populations(
        population_sizes,
        category_patterns,
    )
    cohen_estimates = np.empty(repetitions)
    cohen_ses = np.empty(repetitions)
    fleiss_estimates = np.empty(repetitions)
    fleiss_ses = np.empty(repetitions)

    for repetition in range(repetitions):
        sampled_a: list[np.ndarray] = []
        sampled_b: list[np.ndarray] = []
        sampled_ratings: list[np.ndarray] = []
        sampled_weights: list[np.ndarray] = []
        sampled_strata: list[np.ndarray] = []
        for stratum, population_size in population_sizes.items():
            sample_size = sample_sizes[stratum]
            selected = rng.choice(population_size, size=sample_size, replace=False)
            pop_a, pop_b = cohen_populations[stratum]
            sampled_a.append(pop_a[selected])
            sampled_b.append(pop_b[selected])
            sampled_ratings.append(fleiss_populations[stratum][selected])
            sampled_weights.append(
                np.full(sample_size, population_size / sample_size)
            )
            sampled_strata.append(np.full(sample_size, stratum, dtype=object))

        weights = np.concatenate(sampled_weights)
        strata = np.concatenate(sampled_strata)
        cohen = item_weighted_cohen_kappa(
            np.concatenate(sampled_a),
            np.concatenate(sampled_b),
            sample_weight=weights,
            strata=strata,
            population_sizes=population_sizes,
        )
        fleiss = item_weighted_fleiss_kappa(
            np.concatenate(sampled_ratings),
            sample_weight=weights,
            strata=strata,
            population_sizes=population_sizes,
        )
        cohen_estimates[repetition] = cohen.estimate
        cohen_ses[repetition] = cohen.standard_error
        fleiss_estimates[repetition] = fleiss.estimate
        fleiss_ses[repetition] = fleiss.standard_error

    cohen_ratio = float(cohen_ses.mean() / cohen_estimates.std(ddof=1))
    fleiss_ratio = float(fleiss_ses.mean() / fleiss_estimates.std(ddof=1))
    return cohen_ratio, fleiss_ratio


@pytest.mark.slow
def test_finite_population_standard_errors_track_monte_carlo_variation() -> None:
    rng = np.random.default_rng(20260726)
    population_sizes = {"first": 200, "second": 300}
    sample_sizes = {"first": 120, "second": 180}

    cohen_populations: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fleiss_populations: dict[str, np.ndarray] = {}
    for position, (stratum, size) in enumerate(population_sizes.items()):
        base = np.arange(size) % 2
        agreement = (np.arange(size) + position) % 5 != 0
        other = np.where(agreement, base, 1 - base)
        cohen_populations[stratum] = (base, other)

        third = np.where((np.arange(size) + position) % 4 == 0, 1 - base, base)
        fleiss_populations[stratum] = np.column_stack((base, other, third))

    cohen_estimates: list[float] = []
    cohen_ses: list[float] = []
    fleiss_estimates: list[float] = []
    fleiss_ses: list[float] = []

    for _ in range(3_000):
        sampled_a: list[np.ndarray] = []
        sampled_b: list[np.ndarray] = []
        sampled_ratings: list[np.ndarray] = []
        sampled_weights: list[np.ndarray] = []
        sampled_strata: list[np.ndarray] = []
        for stratum, population_size in population_sizes.items():
            sample_size = sample_sizes[stratum]
            selected = rng.choice(population_size, size=sample_size, replace=False)
            pop_a, pop_b = cohen_populations[stratum]
            sampled_a.append(pop_a[selected])
            sampled_b.append(pop_b[selected])
            sampled_ratings.append(fleiss_populations[stratum][selected])
            sampled_weights.append(
                np.full(sample_size, population_size / sample_size)
            )
            sampled_strata.append(np.full(sample_size, stratum, dtype=object))

        weights = np.concatenate(sampled_weights)
        strata = np.concatenate(sampled_strata)
        cohen = item_weighted_cohen_kappa(
            np.concatenate(sampled_a),
            np.concatenate(sampled_b),
            sample_weight=weights,
            strata=strata,
            population_sizes=population_sizes,
        )
        fleiss = item_weighted_fleiss_kappa(
            np.concatenate(sampled_ratings),
            sample_weight=weights,
            strata=strata,
            population_sizes=population_sizes,
        )
        cohen_estimates.append(cohen.estimate)
        cohen_ses.append(cohen.standard_error)
        fleiss_estimates.append(fleiss.estimate)
        fleiss_ses.append(fleiss.standard_error)

    for estimates, standard_errors in (
        (cohen_estimates, cohen_ses),
        (fleiss_estimates, fleiss_ses),
    ):
        monte_carlo_sd = np.std(estimates, ddof=1)
        ratio = np.mean(standard_errors) / monte_carlo_sd
        assert 0.85 <= ratio <= 1.15


@pytest.mark.slow
@pytest.mark.parametrize(
    (
        "population_sizes",
        "sample_sizes",
        "category_patterns",
        "seed",
    ),
    [
        pytest.param(
            {"high_fraction": 200, "low_fraction": 300},
            {"high_fraction": 160, "low_fraction": 60},
            {
                "high_fraction": (0, 0, 0, 0, 0, 1, 1, 1, 2, 2),
                "low_fraction": (0, 0, 0, 1, 1, 1, 1, 2, 2, 2),
            },
            20260727,
            id="sampling-fractions-0.8-and-0.2",
        ),
        pytest.param(
            {"high": 240, "medium": 300, "low": 400},
            {"high": 200, "medium": 150, "low": 40},
            {
                "high": (0, 0, 0, 0, 0, 0, 1, 1, 1, 2),
                "medium": (0, 0, 0, 1, 1, 1, 1, 1, 2, 2),
                "low": (0, 0, 0, 0, 1, 1, 2, 2, 2, 2),
            },
            20260728,
            id="sampling-fractions-0.83-0.5-and-0.1",
        ),
    ],
)
def test_unbalanced_multicategory_populations_and_unequal_fpcs(
    population_sizes: dict[str, int],
    sample_sizes: dict[str, int],
    category_patterns: dict[str, tuple[int, ...]],
    seed: int,
) -> None:
    cohen_ratio, fleiss_ratio = _simulate_design(
        population_sizes=population_sizes,
        sample_sizes=sample_sizes,
        category_patterns=category_patterns,
        seed=seed,
        repetitions=2_000,
    )
    assert 0.85 <= cohen_ratio <= 1.15
    assert 0.85 <= fleiss_ratio <= 1.15
