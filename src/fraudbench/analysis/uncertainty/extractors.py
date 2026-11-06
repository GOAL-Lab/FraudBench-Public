"""Typed accessors for the coefficient blocks of the M1 / M2 / M3 posteriors.

Every function reads the ``posterior`` Dataset of a
:class:`~fraudbench.analysis.uncertainty.loaders.PosteriorBundle`, computes
mean + HDI per coefficient and returns a ``pandas.DataFrame`` with columns
``name`` (human-readable label), ``var`` (site, with index), ``mean``,
``hdi_low``, ``hdi_high``, ``hdi_prob``. Treatment-coded vectors and
matrices have their zero-pinned reference cells dropped. Level labels come
from the level lists stored in ``features.pkl`` by
``build_fraudbench_indices`` (falling back to the constants in
:mod:`fraudbench.analysis.regressions.process`).

All values are on the log-odds scale.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import xarray as xr

from ..regressions.process import (
    AGE_LEVELS,
    AI_USE_LEVELS,
    CASE_PARTIES_LEVELS,
    CONTEXT_LEVELS,
    GENDER_LEVELS,
    INSTRUCTION_LEVELS,
    RACE_LEVELS,
    SETTING_LEVELS,
    TASK_DISPLAY,
)
from .intervals import DEFAULT_HDI_PROB, mean_with_hdi
from .loaders import PosteriorBundle, stack_draws


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _row(label: str, var: str, samples: np.ndarray, hdi_prob: float) -> dict:
    s = mean_with_hdi(samples, hdi_prob=hdi_prob)
    return {"name": label, "var": var, "mean": s.mean, "hdi_low": s.hdi_low,
            "hdi_high": s.hdi_high, "hdi_prob": hdi_prob}


def _scalar_summary(
    posterior: xr.Dataset, var_name: str, *, label: str, hdi_prob: float,
) -> Optional[dict]:
    """One row for a scalar site; ``None`` when the site is absent from this run."""
    if var_name not in posterior.data_vars:
        return None
    return _row(label, var_name, stack_draws(posterior[var_name]), hdi_prob)


def _vector_summary(
    posterior: xr.Dataset, var_name: str, *, levels: list[str],
    drop_reference: bool, label_prefix: str, hdi_prob: float,
) -> list[dict]:
    if var_name not in posterior.data_vars:
        return []
    samples = stack_draws(posterior[var_name])
    if samples.ndim != 2 or samples.shape[1] != len(levels):
        raise ValueError(f"{var_name}: expected shape (sample, {len(levels)}); got {samples.shape}")
    start = 1 if drop_reference else 0
    return [
        _row(f"{label_prefix}{levels[i]}", f"{var_name}[{i}]", samples[:, i], hdi_prob)
        for i in range(start, len(levels))
    ]


def _matrix_summary(
    posterior: xr.Dataset, var_name: str, *, row_levels: list[str], col_levels: list[str],
    label_prefix: str, hdi_prob: float,
) -> list[dict]:
    """Non-reference cells (row > 0, col > 0) of a treatment-coded matrix."""
    if var_name not in posterior.data_vars:
        return []
    samples = stack_draws(posterior[var_name])
    if samples.ndim != 3 or samples.shape[1:] != (len(row_levels), len(col_levels)):
        raise ValueError(
            f"{var_name}: expected (sample, {len(row_levels)}, {len(col_levels)}); got {samples.shape}"
        )
    return [
        _row(f"{label_prefix}{row_levels[i]} x {col_levels[j]}", f"{var_name}[{i},{j}]", samples[:, i, j], hdi_prob)
        for i in range(1, len(row_levels)) for j in range(1, len(col_levels))
    ]


def _rows_to_df(rows: Iterable[Optional[dict]]) -> pd.DataFrame:
    cleaned = [r for r in rows if r is not None]
    cols = ["name", "var", "mean", "hdi_low", "hdi_high", "hdi_prob"]
    if not cleaned:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(cleaned)[cols]


def _task_labels(bundle: PosteriorBundle) -> list[str]:
    return [TASK_DISPLAY.get(t, t) for t in bundle.levels("task_levels", SETTING_LEVELS)]


# ---------------------------------------------------------------------------
# Public extractors
# ---------------------------------------------------------------------------

def get_scenario_coefs(bundle: PosteriorBundle, *, hdi_prob: float = DEFAULT_HDI_PROB) -> pd.DataFrame:
    """Intercept, task / instruction / context main effects, party type,
    scenario covariates and the instruction x context shrinkage scale."""
    p = bundle.posterior
    rows: list[Optional[dict]] = [_scalar_summary(p, "alpha", label="Intercept", hdi_prob=hdi_prob)]
    rows += _vector_summary(p, "task_effect", levels=_task_labels(bundle), drop_reference=True,
                            label_prefix="Task: ", hdi_prob=hdi_prob)
    rows += _vector_summary(p, "instruction_effect", levels=bundle.levels("instruction_levels", INSTRUCTION_LEVELS),
                            drop_reference=True, label_prefix="Instruction: ", hdi_prob=hdi_prob)
    rows += _vector_summary(p, "context_effect", levels=bundle.levels("context_levels", CONTEXT_LEVELS),
                            drop_reference=True, label_prefix="Context: ", hdi_prob=hdi_prob)
    rows += _vector_summary(p, "party_type_effect", levels=bundle.levels("party_levels", CASE_PARTIES_LEVELS),
                            drop_reference=True, label_prefix="Party type: ", hdi_prob=hdi_prob)
    rows.append(_scalar_summary(p, "beta_concealment", label="Concealment scenario", hdi_prob=hdi_prob))
    rows.append(_scalar_summary(p, "beta_order", label="Fraudulent action presented first", hdi_prob=hdi_prob))
    rows.append(_scalar_summary(p, "sigma_instruction_context", label="sigma(instruction x context)", hdi_prob=hdi_prob))
    rows.append(_scalar_summary(p, "sigma_case", label="sigma(case RE)", hdi_prob=hdi_prob))
    rows.append(_scalar_summary(p, "sigma_scenario", label="sigma(scenario RE)", hdi_prob=hdi_prob))
    return _rows_to_df(rows)


def get_interaction_matrices(bundle: PosteriorBundle, *, hdi_prob: float = DEFAULT_HDI_PROB) -> pd.DataFrame:
    """Per-cell entries of the instruction x context and task x context interactions."""
    p = bundle.posterior
    instr = bundle.levels("instruction_levels", INSTRUCTION_LEVELS)
    ctx = bundle.levels("context_levels", CONTEXT_LEVELS)
    rows = _matrix_summary(p, "instruction_context_effect", row_levels=instr, col_levels=ctx,
                           label_prefix="Instruction x Context: ", hdi_prob=hdi_prob)
    rows += _matrix_summary(p, "task_context_effect", row_levels=_task_labels(bundle), col_levels=ctx,
                            label_prefix="Task x Context: ", hdi_prob=hdi_prob)
    return _rows_to_df(rows)


def get_model_coefs(bundle: PosteriorBundle, *, hdi_prob: float = DEFAULT_HDI_PROB) -> pd.DataFrame:
    """Model-level regression, size tiers, LLM row-level covariates and the
    family / model random-effect scales."""
    p = bundle.posterior
    rows: list[Optional[dict]] = [
        _scalar_summary(p, "beta_release_date", label="Release date (z)", hdi_prob=hdi_prob),
    ]
    tiers = [f"tier {t}" for t in bundle.levels("size_tiers", [])] or [
        f"tier {i}" for i in range(int(bundle.features.get("num_size", 0) or 0))
    ]
    rows += _vector_summary(p, "model_size_effect", levels=tiers, drop_reference=True,
                            label_prefix="Model size: ", hdi_prob=hdi_prob)
    rows.append(_scalar_summary(p, "beta_reasoning", label="Reasoning enabled", hdi_prob=hdi_prob))
    rows.append(_scalar_summary(p, "beta_high_temperature", label="Temperature ablation", hdi_prob=hdi_prob))
    rows.append(_scalar_summary(p, "beta_high_reasoning", label="High-reasoning ablation", hdi_prob=hdi_prob))
    rows.append(_scalar_summary(p, "beta_post_cutoff", label="Case decided after knowledge cutoff", hdi_prob=hdi_prob))
    rows.append(_scalar_summary(p, "sigma_family", label="sigma(family RE)", hdi_prob=hdi_prob))
    rows.append(_scalar_summary(p, "sigma_model", label="sigma(model RE)", hdi_prob=hdi_prob))
    return _rows_to_df(rows)


def get_token_coefs(bundle: PosteriorBundle, *, hdi_prob: float = DEFAULT_HDI_PROB) -> pd.DataFrame:
    """M2 reasoning-token slope and its high-reasoning interaction."""
    p = bundle.posterior
    return _rows_to_df([
        _scalar_summary(p, "beta_tokens", label="Reasoning tokens (z log)", hdi_prob=hdi_prob),
        _scalar_summary(p, "beta_high_reasoning_tokens",
                        label="High-reasoning ablation x reasoning tokens (z log)", hdi_prob=hdi_prob),
    ])


def get_human_coefs(bundle: PosteriorBundle, *, hdi_prob: float = DEFAULT_HDI_PROB) -> pd.DataFrame:
    """M1 human block: human-vs-LLM contrast, human x context interactions,
    demographic random-effect blocks and their scales."""
    p = bundle.posterior
    ctx = bundle.levels("context_levels", CONTEXT_LEVELS)
    rows: list[Optional[dict]] = [
        _scalar_summary(p, "beta_human", label="Human vs. LLM (control context)", hdi_prob=hdi_prob),
    ]
    rows += _vector_summary(p, "human_context_effect", levels=ctx, drop_reference=True,
                            label_prefix="Human x Context: ", hdi_prob=hdi_prob)
    rows += _vector_summary(p, "age_effect", levels=AGE_LEVELS, drop_reference=False,
                            label_prefix="Age: ", hdi_prob=hdi_prob)
    rows += _vector_summary(p, "race_effect", levels=RACE_LEVELS, drop_reference=False,
                            label_prefix="Race: ", hdi_prob=hdi_prob)
    rows += _vector_summary(p, "gender_effect", levels=GENDER_LEVELS, drop_reference=False,
                            label_prefix="Gender: ", hdi_prob=hdi_prob)
    rows += _vector_summary(p, "ai_use_effect", levels=AI_USE_LEVELS, drop_reference=False,
                            label_prefix="AI use: ", hdi_prob=hdi_prob)
    for site, label in [
        ("sigma_participant", "sigma(participant RE)"), ("sigma_age", "sigma(age)"),
        ("sigma_race", "sigma(race)"), ("sigma_gender", "sigma(gender)"), ("sigma_ai_use", "sigma(AI use)"),
    ]:
        rows.append(_scalar_summary(p, site, label=label, hdi_prob=hdi_prob))
    return _rows_to_df(rows)


def get_judge_coefs(bundle: PosteriorBundle, *, hdi_prob: float = DEFAULT_HDI_PROB) -> pd.DataFrame:
    """M3 judge block: effect-coded judge severities, self-preference slopes
    and the judge x model / judge x scenario scales."""
    p = bundle.posterior
    judges = bundle.levels("judge_ids", [])
    rows = _vector_summary(p, "judge_effect", levels=judges, drop_reference=False,
                           label_prefix="Judge severity: ", hdi_prob=hdi_prob)
    rows.append(_scalar_summary(p, "beta_self_family", label="Self-family (judge family = model family)", hdi_prob=hdi_prob))
    rows.append(_scalar_summary(p, "beta_self_model", label="Self-model (additional, judge = model)", hdi_prob=hdi_prob))
    rows.append(_scalar_summary(p, "sigma_judge_model", label="sigma(judge x model RE)", hdi_prob=hdi_prob))
    rows.append(_scalar_summary(p, "sigma_judge_scenario", label="sigma(judge x scenario RE)", hdi_prob=hdi_prob))
    return _rows_to_df(rows)


# ---------------------------------------------------------------------------
# Random-effect tables
# ---------------------------------------------------------------------------

def get_model_re(bundle: PosteriorBundle, *, hdi_prob: float = DEFAULT_HDI_PROB) -> pd.DataFrame:
    """Per-LLM random intercept ``model_re`` (one row per ``model_idx``)."""
    p = bundle.posterior
    if "model_re" not in p.data_vars:
        return pd.DataFrame()
    samples = stack_draws(p["model_re"])
    rows = []
    for _, row in bundle.model_table.iterrows():
        idx = int(row["model_idx"])
        s = mean_with_hdi(samples[:, idx], hdi_prob=hdi_prob)
        rows.append({"model_idx": idx, "model": row.get("model"), "model_family": row.get("model_family"),
                     "mean": s.mean, "hdi_low": s.hdi_low, "hdi_high": s.hdi_high, "hdi_prob": hdi_prob})
    return pd.DataFrame(rows)


def get_family_re(bundle: PosteriorBundle, *, hdi_prob: float = DEFAULT_HDI_PROB) -> pd.DataFrame:
    """Per-family random effect ``family_re``."""
    p = bundle.posterior
    if "family_re" not in p.data_vars:
        return pd.DataFrame()
    samples = stack_draws(p["family_re"])
    rows = []
    for _, row in bundle.family_table.iterrows():
        idx = int(row["family_idx"])
        s = mean_with_hdi(samples[:, idx], hdi_prob=hdi_prob)
        rows.append({"family_idx": idx, "model_family": row["model_family"],
                     "mean": s.mean, "hdi_low": s.hdi_low, "hdi_high": s.hdi_high, "hdi_prob": hdi_prob})
    return pd.DataFrame(rows)


def get_judge_model_re(bundle: PosteriorBundle, *, hdi_prob: float = DEFAULT_HDI_PROB) -> pd.DataFrame:
    """M3 judge x model random effects, one row per (judge, model)."""
    p = bundle.posterior
    if "judge_model_re" not in p.data_vars:
        return pd.DataFrame()
    samples = stack_draws(p["judge_model_re"])  # (n_draws, n_judge, n_model)
    judges = bundle.levels("judge_ids", [str(j) for j in range(samples.shape[1])])
    rows = []
    for j, judge in enumerate(judges):
        for _, row in bundle.model_table.iterrows():
            idx = int(row["model_idx"])
            s = mean_with_hdi(samples[:, j, idx], hdi_prob=hdi_prob)
            rows.append({"judge": judge, "model_idx": idx, "model": row.get("model"),
                         "model_family": row.get("model_family"), "mean": s.mean,
                         "hdi_low": s.hdi_low, "hdi_high": s.hdi_high, "hdi_prob": hdi_prob})
    return pd.DataFrame(rows)
