"""End-to-end checks for the M1 / M2 / M3 regression pipeline.

A small synthetic evaluation frame is built from the real metadata files
(scenarios, models, human-baseline respondents) so the ``merge_*``
processors, the index builders and the three model builders run exactly as
they do on ``obs_data.parquet``. The NumPy predictor is checked against the
NumPyro model's log-likelihood term by term.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pandas as pd
import pytest
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.util import log_likelihood

from hibayes.analysis import AnalysisState

from fraudbench.analysis.regressions import merge_external as me
from fraudbench.analysis.regressions import process as pr
from fraudbench.analysis.regressions.models import (
    PRIORS, fraudbench_m1, fraudbench_m2, fraudbench_m3,
)
from fraudbench.analysis.regressions.predictor import expit, m1_eta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic import make_frame, make_per_score  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
SCENARIOS = REPO / "data/final/dataset_public.jsonl"
MODELS = REPO / "data/final/models.jsonl"
HUMANS = REPO / "data/final/human_baseline/human_baseline_public.jsonl"

pytestmark = pytest.mark.skipif(
    not (SCENARIOS.is_file() and MODELS.is_file() and HUMANS.is_file()),
    reason="metadata files missing",
)


@pytest.fixture(scope="module")
def synthetic_frame() -> pd.DataFrame:
    return make_frame(REPO)


@pytest.fixture(scope="module")
def per_score_path(synthetic_frame, tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("per_score") / "per_score.parquet"
    make_per_score(synthetic_frame).to_parquet(path)
    return path


def _run(state: AnalysisState, *processors) -> AnalysisState:
    for p in processors:
        state = p(state)
    return state


def _merged(df: pd.DataFrame) -> AnalysisState:
    return _run(
        AnalysisState(data=df.copy()),
        me.merge_scenario_metadata(), me.merge_model_metadata(), me.merge_human_baseline_metadata(),
    )


@pytest.fixture(scope="module")
def m1_state(synthetic_frame) -> AnalysisState:
    return _run(
        _merged(synthetic_frame),
        pr.filter_outcome_observed(outcome_column="score_fraud"), pr.build_fraudbench_indices(),
        pr.select_outcome(outcome_column="score_fraud"),
    )


@pytest.fixture(scope="module")
def m2_state(synthetic_frame) -> AnalysisState:
    return _run(
        _merged(synthetic_frame),
        pr.filter_ai_only(), pr.filter_reasoning_tokens_positive(),
        pr.filter_outcome_observed(outcome_column="score_fraud"), pr.build_fraudbench_indices(),
        pr.select_outcome(outcome_column="score_fraud"),
    )


@pytest.fixture(scope="module")
def m3_state(synthetic_frame, per_score_path) -> AnalysisState:
    return _run(
        _merged(synthetic_frame),
        pr.filter_ai_only(), pr.expand_to_judge_votes(path=str(per_score_path)),
        pr.build_fraudbench_indices(), pr.build_judge_indices(),
        pr.select_outcome(outcome_column="fraud_vote"),
    )


def _np_features(features: dict) -> dict:
    return {k: (np.asarray(v) if hasattr(v, "shape") else v) for k, v in features.items()}


# --------------------------------------------------------------------------
# Processors
# --------------------------------------------------------------------------

def test_m1_features(m1_state, synthetic_frame):
    f, df = m1_state.features, m1_state.processed_data
    assert len(df) == len(synthetic_frame) - 3
    assert f["num_model"] == 20 and f["num_family"] == 9
    # Paper tiers (0: <10B, 1: <100B, 2: 100-500B, 3: 500B-1T, 4: >1T): no evaluated model is in tier 1.
    assert f["num_size"] == 4 and tuple(f["size_tiers"]) == (0, 2, 3, 4)
    assert f["num_participant"] == df.loc[df.is_human == 1, "human_respondent_id"].nunique() + 1
    # Human rows: human_baseline task, control instruction, participant slot > 0.
    hum = df[df.is_human == 1]
    assert (hum.task_idx == 3).all() and (hum.instruction_idx == 0).all() and (hum.participant_idx > 0).all()
    assert (df.loc[df.is_ai == 1, "participant_idx"] == 0).all()
    # Model-level arrays align with the sorted model ids.
    assert list(f["model_ids"]) == sorted(df.loc[df.is_ai == 1, "model"].unique())
    assert np.asarray(f["family_idx_by_model"]).shape == (20,)
    assert abs(float(np.asarray(f["z_release_date_by_model"]).mean())) < 1e-6
    # Reasoning flags: every high-reasoning-ablation row is flagged, and only those.
    abl = df.run_category == "reasoning_ablation"
    assert (df.loc[abl, "high_reasoning"] == 1).all() and (df.loc[~abl, "high_reasoning"] == 0).all()
    assert (df.loc[abl & df.model_max_reasoning_effort.notna(), "reasoning_on"] == 1).all()
    assert (df.loc[df.run_category == "temperature_ablation", "high_temperature"] == 1).all()
    assert set(np.unique(np.asarray(f["obs"]))) <= {0, 1}


def test_m2_filters_and_standardization(m2_state):
    f, df = m2_state.features, m2_state.processed_data
    assert (df.model != "human").all() and (df.tokens_reasoning > 0).all()
    z = np.asarray(f["z_log_reasoning_tokens"])
    assert abs(z.mean()) < 1e-5 and abs(z.std() - 1) < 1e-4
    assert f["num_model"] < 20  # non-reasoning models fall out
    assert f["num_model"] == len(f["model_ids"])


def test_m3_judge_indices(m3_state):
    f, df = m3_state.features, m3_state.processed_data
    assert f["num_judge"] == 3
    assert (df.groupby("sample_id").size() <= 3).all()
    self_model = df[df.is_self_model == 1]
    assert (self_model.model == self_model.judge_model.map(lambda j: j)).all()
    # Every self-model pair is also a self-family pair; some family pairs are not self-model.
    assert (df.loc[df.is_self_model == 1, "is_self_family"] == 1).all()
    assert df.is_self_family.sum() > df.is_self_model.sum()
    assert set(np.unique(np.asarray(f["obs"]))) <= {0, 1}


# --------------------------------------------------------------------------
# Models and the NumPy predictor
# --------------------------------------------------------------------------

def _prior_draws(model, features, n: int = 4) -> dict:
    feats = {k: (v if k != "obs" else None) for k, v in features.items()}
    return Predictive(model, num_samples=n)(jax.random.PRNGKey(0), feats)


@pytest.mark.parametrize("builder,state_name", [
    (fraudbench_m1, "m1_state"), (fraudbench_m2, "m2_state"), (fraudbench_m3, "m3_state"),
])
def test_models_run_and_expose_sites(builder, state_name, request):
    state = request.getfixturevalue(state_name)
    model = builder()
    draws = _prior_draws(model, state.features)
    assert draws["obs"].shape == (4, int(np.asarray(state.features["obs"]).shape[0]))
    expected = {"alpha", "task_effect", "instruction_context_effect", "task_context_effect",
                "model_re", "family_re", "model_size_effect", "beta_release_date",
                "beta_high_temperature", "beta_high_reasoning", "beta_post_cutoff"}
    if builder is fraudbench_m1:
        expected |= {"beta_reasoning", "beta_human", "human_context_effect", "participant_re", "age_effect"}
    if builder is fraudbench_m2:
        expected |= {"beta_tokens", "beta_high_reasoning_tokens"}
        assert "beta_reasoning" not in draws and "beta_human" not in draws
    if builder is fraudbench_m3:
        expected |= {"beta_reasoning", "judge_effect", "beta_self_family", "beta_self_model",
                     "judge_model_re", "judge_scenario_re", "sigma_judge_model", "sigma_judge_scenario"}
        assert "beta_human" not in draws
        assert np.allclose(np.asarray(draws["judge_effect"]).sum(axis=1), 0.0, atol=1e-6)
    assert expected <= set(draws)
    # Treatment coding: reference level / row / column pinned to zero.
    assert np.all(np.asarray(draws["task_effect"])[:, 0] == 0)
    assert np.all(np.asarray(draws["task_context_effect"])[:, 0, :] == 0)
    assert np.all(np.asarray(draws["task_context_effect"])[:, :, 0] == 0)


@pytest.mark.parametrize("builder,state_name,kwargs", [
    (fraudbench_m1, "m1_state", dict(include_reasoning=True, include_human=True)),
    (fraudbench_m2, "m2_state", dict(include_reasoning=False, include_human=False)),
])
def test_numpy_predictor_matches_numpyro(builder, state_name, kwargs, request):
    state = request.getfixturevalue(state_name)
    model = builder()
    draws = _prior_draws(model, state.features, n=3)
    params = {k: v for k, v in draws.items() if k != "obs"}
    ll = np.asarray(log_likelihood(model, params, state.features, batch_ndims=1)["obs"])

    np_draws = {k: np.asarray(v) for k, v in params.items()}
    feats = _np_features(state.features)
    eta = m1_eta(np_draws, feats, **kwargs)
    if builder is fraudbench_m2:
        z = feats["z_log_reasoning_tokens"][None, :]
        eta = eta + np_draws["beta_tokens"][:, None] * z \
            + np_draws["beta_high_reasoning_tokens"][:, None] * feats["high_reasoning"][None, :] * z
    obs = feats["obs"][None, :]
    p = expit(eta)
    expected = obs * np.log(p) + (1 - obs) * np.log1p(-p)
    assert np.allclose(ll, expected, atol=1e-4)


def test_prior_scales_match_paper():
    assert PRIORS["alpha"] == 1.5
    assert PRIORS["task_effect"] == PRIORS["beta_human"] == 0.7
    for k in ["instruction_effect", "context_effect", "beta_reasoning", "beta_high_temperature",
              "beta_high_reasoning", "beta_post_cutoff", "beta_release_date"]:
        assert PRIORS[k] == 0.5
    for k in ["party_type_effect", "beta_concealment", "beta_order", "model_size_effect",
              "task_context_effect", "human_context_effect", "beta_tokens"]:
        assert PRIORS[k] == 0.4
    assert PRIORS["beta_high_reasoning_tokens"] == 0.2
    for k in ["sigma_case", "sigma_scenario", "sigma_family", "sigma_model", "sigma_participant"]:
        assert PRIORS[k] == 0.5
    assert PRIORS["sigma_instruction_context"] == 0.25
    for k in ["sigma_age", "sigma_race", "sigma_gender", "sigma_ai_use"]:
        assert PRIORS[k] == 0.3
    for k in ["judge_effect", "beta_self_family", "beta_self_model"]:
        assert PRIORS[k] == 0.5
    assert PRIORS["sigma_judge_model"] == PRIORS["sigma_judge_scenario"] == 0.25


def test_m1_nuts_smoke(m1_state):
    mcmc = MCMC(NUTS(fraudbench_m1()), num_warmup=15, num_samples=10, num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), m1_state.features)
    samples = mcmc.get_samples()
    assert samples["alpha"].shape == (10,)
    assert np.isfinite(np.asarray(samples["model_re"])).all()
