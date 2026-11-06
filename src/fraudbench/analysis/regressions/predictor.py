"""NumPy reconstruction of the M1 linear predictor from posterior draws.

:func:`m1_eta` mirrors :func:`fraudbench.analysis.regressions.models.m1_predictor`
term by term so that the fitted posterior can be evaluated on arbitrary rows
(observed or counterfactual) without re-running NumPyro. It is used by the
posterior-predictive human-vs-LLM comparison
(``analysis/regressions/posterior_human_ai_gap.py``).

``draws`` maps posterior site names to arrays with a leading draw axis
``(D, ...)``; ``features`` maps the feature names of ``process.py`` to NumPy
arrays with a leading row axis ``(N, ...)`` (plus the ``num_*`` scalars). A
site absent from ``draws`` contributes zero, which is how M2 (no
``beta_reasoning``) and the per-cell helpers (random effects integrated or
pinned to zero) reuse the same code.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np


_ROW_FEATURES: Sequence[str] = (
    "task_idx", "instruction_idx", "context_idx", "scenario_idx", "case_idx",
    "party_idx", "participant_idx", "age_idx", "race_idx", "gender_idx", "ai_use_idx",
    "model_idx", "is_ai", "is_human", "is_concealment", "is_fraud_first",
    "reasoning_on", "high_temperature", "high_reasoning", "post_cutoff_case",
)


def _scalar(draws: Mapping[str, np.ndarray], name: str, n_draws: int) -> np.ndarray:
    arr = draws.get(name)
    if arr is None:
        return np.zeros(n_draws, dtype=np.float64)
    return np.asarray(arr, dtype=np.float64).reshape(n_draws)


def _gather(draws: Mapping[str, np.ndarray], name: str, idx: np.ndarray, n_draws: int) -> np.ndarray:
    """``(D, N)`` with ``out[d, i] = draws[name][d, idx[i]]`` (zeros if absent)."""
    arr = draws.get(name)
    if arr is None:
        return np.zeros((n_draws, idx.shape[0]), dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)[:, idx]


def _gather2(
    draws: Mapping[str, np.ndarray], name: str, idx_a: np.ndarray, idx_b: np.ndarray, n_draws: int,
) -> np.ndarray:
    arr = draws.get(name)
    if arr is None:
        return np.zeros((n_draws, idx_a.shape[0]), dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)[:, idx_a, idx_b]


def _rows(features: Mapping[str, np.ndarray], rows: Optional[np.ndarray]) -> dict:
    out = {}
    for name in _ROW_FEATURES:
        if name not in features:
            continue
        arr = np.asarray(features[name])
        out[name] = arr if rows is None else arr[rows]
    return out


def n_draws_of(draws: Mapping[str, np.ndarray]) -> int:
    """Leading draw-axis length shared by every array in ``draws``."""
    for v in draws.values():
        return int(np.asarray(v).shape[0])
    raise ValueError("draws is empty")


def m1_eta(
    draws: Mapping[str, np.ndarray],
    features: Mapping[str, np.ndarray],
    rows: Optional[np.ndarray] = None,
    *,
    include_reasoning: bool = True,
    include_human: bool = True,
) -> np.ndarray:
    """Return ``eta`` of shape ``(D, N)`` for the selected rows.

    Args:
        draws: posterior arrays ``(D, ...)`` keyed by site name.
        features: feature arrays keyed as in ``process.build_fraudbench_indices``.
        rows: optional integer index array selecting rows of ``features``.
        include_reasoning: add ``beta_reasoning * reasoning_on`` (M1 / M3).
        include_human: add the human block on rows with ``is_human == 1``.
    """
    D = n_draws_of(draws)
    f = _rows(features, rows)
    task, instr, ctx = f["task_idx"], f["instruction_idx"], f["context_idx"]
    is_ai = f["is_ai"].astype(np.float64)[None, :]

    eta = _scalar(draws, "alpha", D)[:, None] + _gather(draws, "task_effect", task, D)

    # Scenario block
    eta = (
        eta
        + _gather(draws, "case_re", f["case_idx"], D)
        + _gather(draws, "scenario_re", f["scenario_idx"], D)
        + _gather(draws, "party_type_effect", f["party_idx"], D)
        + _scalar(draws, "beta_concealment", D)[:, None] * f["is_concealment"][None, :]
        + _scalar(draws, "beta_order", D)[:, None] * f["is_fraud_first"][None, :]
    )

    # Design block (every row)
    eta = (
        eta
        + _gather(draws, "instruction_effect", instr, D)
        + _gather(draws, "context_effect", ctx, D)
        + _gather2(draws, "instruction_context_effect", instr, ctx, D)
        + _gather2(draws, "task_context_effect", task, ctx, D)
    )

    # LLM block
    runtime = (
        _gather(draws, "model_re", f["model_idx"], D)
        + _scalar(draws, "beta_high_temperature", D)[:, None] * f["high_temperature"][None, :]
        + _scalar(draws, "beta_high_reasoning", D)[:, None] * f["high_reasoning"][None, :]
        + _scalar(draws, "beta_post_cutoff", D)[:, None] * f["post_cutoff_case"][None, :]
    )
    if include_reasoning:
        runtime = runtime + _scalar(draws, "beta_reasoning", D)[:, None] * f["reasoning_on"][None, :]
    eta = eta + is_ai * runtime

    # Human block
    if include_human and "is_human" in f:
        is_human = f["is_human"].astype(np.float64)[None, :]
        human = (
            _scalar(draws, "beta_human", D)[:, None]
            + _gather(draws, "human_context_effect", ctx, D)
            + _gather(draws, "participant_re", f["participant_idx"], D)
            + _gather(draws, "age_effect", f["age_idx"], D)
            + _gather(draws, "race_effect", f["race_idx"], D)
            + _gather(draws, "gender_effect", f["gender_idx"], D)
            + _gather(draws, "ai_use_effect", f["ai_use_idx"], D)
        )
        eta = eta + is_human * human

    return eta


def expit(x: np.ndarray) -> np.ndarray:
    """Numerically stable inverse logit."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


__all__ = ["m1_eta", "expit", "n_draws_of"]
