"""FraudBench regression models M1, M2 and M3 (HiBayes / NumPyro).

The three builders implement the Bayesian logistic regressions specified in
the paper appendix (``paper/main.tex``, *Full Results > Regressions*):

* :func:`fraudbench_m1` -- **M1**, the main regression on the fraud outcome,
  fit on every generation (LLM and human alike).
* :func:`fraudbench_m2` -- **M2**, the reasoning-tokens variant. Fit on LLM
  generations with a strictly positive reasoning-token count; the M1
  predictor with the reasoning-enabled main effect removed, plus a
  standardized log-token slope and its high-reasoning interaction.
* :func:`fraudbench_m3` -- **M3**, the self-preference variant. Unit of
  observation is the (generation, judge) pair; the M1 predictor (LLM rows
  only) is augmented with the judge block of eq. ``eq:m3_eta``.

Every term is on the log-odds scale. Site names follow the paper's notation
(``task_effect`` = :math:`\\mu^{\\text{task}}`, ``beta_human`` =
:math:`\\beta_{\\text{Human}}`, ``scenario_re`` = :math:`u^{\\text{scen}}`,
...). Treatment-coded effects pin index 0 (the reference level) to zero;
random effects are sampled non-centered. Prior scales are collected in
:data:`PRIORS` and match the *Priors* paragraphs of the appendix.

Feature names consumed here are produced by
:mod:`fraudbench.analysis.regressions.process`; see that module's docstring
for the index codings. :mod:`fraudbench.analysis.regressions.predictor`
re-implements the M1 linear predictor in NumPy for posterior
reconstruction and is tested against :func:`m1_predictor`.
"""

from __future__ import annotations

from typing import List, Optional

import arviz as az
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from hibayes.model import Model, check_features, model


# --------------------------------------------------------------------------
# Prior scales (standard deviations), one entry per parameter family.
# --------------------------------------------------------------------------

PRIORS: dict = {
    # Intercept
    "alpha": 1.5,
    # N(0, 0.7^2)
    "task_effect": 0.7,
    "beta_human": 0.7,
    # N(0, 0.5^2)
    "instruction_effect": 0.5,
    "context_effect": 0.5,
    "beta_reasoning": 0.5,
    "beta_high_temperature": 0.5,
    "beta_high_reasoning": 0.5,
    "beta_post_cutoff": 0.5,
    "beta_release_date": 0.5,
    # N(0, 0.4^2)
    "party_type_effect": 0.4,
    "beta_concealment": 0.4,
    "beta_order": 0.4,
    "model_size_effect": 0.4,
    "task_context_effect": 0.4,
    "human_context_effect": 0.4,
    # Half-N(0.5)
    "sigma_case": 0.5,
    "sigma_scenario": 0.5,
    "sigma_family": 0.5,
    "sigma_model": 0.5,
    "sigma_participant": 0.5,
    # Half-N(0.25)
    "sigma_instruction_context": 0.25,
    # Half-N(0.3)
    "sigma_age": 0.3,
    "sigma_race": 0.3,
    "sigma_gender": 0.3,
    "sigma_ai_use": 0.3,
    # M2 token block
    "beta_tokens": 0.4,
    "beta_high_reasoning_tokens": 0.2,
    # M3 judge block
    "judge_effect": 0.5,
    "beta_self_family": 0.5,
    "beta_self_model": 0.5,
    "sigma_judge_model": 0.25,
    "sigma_judge_scenario": 0.25,
}


def _resolve_priors(overrides: Optional[dict]) -> dict:
    out = dict(PRIORS)
    if overrides:
        unknown = set(overrides) - set(PRIORS)
        if unknown:
            raise ValueError(f"Unknown prior override(s): {sorted(unknown)}")
        out.update(overrides)
    return out


# --------------------------------------------------------------------------
# Feature requirements.
# --------------------------------------------------------------------------

_M1_REQUIRED: List[str] = [
    "obs",
    "task_idx", "num_task",
    "scenario_idx", "num_scenario",
    "case_idx", "num_case",
    "party_idx", "num_party",
    "is_concealment", "is_fraud_first",
    "instruction_idx", "num_instruction",
    "context_idx", "num_context",
    "is_ai", "is_human",
    "model_idx", "num_model",
    "num_family", "num_size",
    "family_idx_by_model", "size_idx_by_model", "z_release_date_by_model",
    "reasoning_on", "high_temperature", "high_reasoning", "post_cutoff_case",
]
_HUMAN_REQUIRED: List[str] = [
    "participant_idx", "num_participant",
    "age_idx", "num_age",
    "race_idx", "num_race",
    "gender_idx", "num_gender",
    "ai_use_idx", "num_ai_use",
]
_M2_REQUIRED: List[str] = ["z_log_reasoning_tokens"]
_M3_REQUIRED: List[str] = [
    "judge_idx", "num_judge", "is_self_family", "is_self_model",
]


# --------------------------------------------------------------------------
# Coefficient-block helpers.
# --------------------------------------------------------------------------

def _treatment_coded_vec(name: str, n: int, scale: float) -> jnp.ndarray:
    """Length-``n`` fixed-effect vector with index 0 (reference) pinned at 0."""
    if n <= 1:
        full = jnp.zeros((max(n, 1),))
    else:
        raw = numpyro.sample(f"{name}_raw", dist.Normal(0.0, scale).expand([n - 1]))
        full = jnp.concatenate([jnp.zeros((1,)), raw])
    numpyro.deterministic(name, full)
    return full


def _treatment_coded_matrix(name: str, n1: int, n2: int, scale: float) -> jnp.ndarray:
    """``(n1, n2)`` fixed-effect matrix with row 0 and column 0 pinned at 0."""
    if n1 <= 1 or n2 <= 1:
        full = jnp.zeros((n1, n2))
    else:
        raw = numpyro.sample(
            f"{name}_raw",
            dist.Normal(0.0, scale).expand([n1 - 1, n2 - 1]).to_event(2),
        )
        full = jnp.zeros((n1, n2)).at[1:, 1:].set(raw)
    numpyro.deterministic(name, full)
    return full


def _shrunk_interaction_matrix(
    name: str, n1: int, n2: int, sigma_name: str, sigma_scale: float,
) -> jnp.ndarray:
    """``(n1, n2)`` matrix with row 0 / column 0 pinned at 0 whose free cells
    are ``sigma`` times a standard normal, ``sigma ~ Half-N(sigma_scale)``."""
    if n1 <= 1 or n2 <= 1:
        full = jnp.zeros((n1, n2))
    else:
        sigma = numpyro.sample(sigma_name, dist.HalfNormal(sigma_scale))
        raw = numpyro.sample(
            f"{name}_raw",
            dist.Normal(0.0, 1.0).expand([n1 - 1, n2 - 1]).to_event(2),
        )
        full = jnp.zeros((n1, n2)).at[1:, 1:].set(raw * sigma)
    numpyro.deterministic(name, full)
    return full


def _random_effect_vec(name: str, n: int, sigma_name: str, sigma_scale: float) -> jnp.ndarray:
    """Exchangeable ``u ~ N(0, sigma^2)`` of length ``n``, non-centered."""
    sigma = numpyro.sample(sigma_name, dist.HalfNormal(sigma_scale))
    z = numpyro.sample(f"{name}_raw", dist.Normal(0.0, 1.0).expand([n]))
    full = sigma * z
    numpyro.deterministic(name, full)
    return full


def _random_effect_matrix(
    name: str, n1: int, n2: int, sigma_name: str, sigma_scale: float,
) -> jnp.ndarray:
    """Exchangeable ``(n1, n2)`` random-effect matrix, non-centered."""
    sigma = numpyro.sample(sigma_name, dist.HalfNormal(sigma_scale))
    z = numpyro.sample(
        f"{name}_raw", dist.Normal(0.0, 1.0).expand([n1, n2]).to_event(2),
    )
    full = sigma * z
    numpyro.deterministic(name, full)
    return full


def _scalar(name: str, scale: float) -> jnp.ndarray:
    return numpyro.sample(name, dist.Normal(0.0, scale))


# --------------------------------------------------------------------------
# M1 linear predictor (shared by M2 and M3).
# --------------------------------------------------------------------------

def _model_random_intercepts(features, priors: dict) -> jnp.ndarray:
    """Per-LLM random intercept with the model-level regression (eq. ``eq:m1_model``):

        u_model[m] ~ N(u_family[f(m)] + mu_size[tier(m)] + beta_release * z_release[m], sigma_model^2)
    """
    num_model = int(features["num_model"])
    family_re = _random_effect_vec(
        "family_re", int(features["num_family"]), "sigma_family", priors["sigma_family"],
    )
    size_effect = _treatment_coded_vec(
        "model_size_effect", int(features["num_size"]), priors["model_size_effect"],
    )
    beta_release = _scalar("beta_release_date", priors["beta_release_date"])
    model_mu = (
        family_re[features["family_idx_by_model"]]
        + size_effect[features["size_idx_by_model"]]
        + beta_release * features["z_release_date_by_model"]
    )
    numpyro.deterministic("model_mu", model_mu)
    sigma_model = numpyro.sample("sigma_model", dist.HalfNormal(priors["sigma_model"]))
    z_model = numpyro.sample("model_re_raw", dist.Normal(0.0, 1.0).expand([num_model]))
    model_re = model_mu + sigma_model * z_model
    numpyro.deterministic("model_re", model_re)
    return model_re


def m1_predictor(
    features,
    priors: dict,
    *,
    include_reasoning: bool = True,
    include_human: bool = True,
) -> jnp.ndarray:
    """Row-level linear predictor :math:`\\eta_i` of M1 (eq. ``eq:m1_eta``).

    Args:
        features: HiBayes feature dict (see ``process.py``).
        priors: resolved prior scales.
        include_reasoning: include the reasoning-enabled main effect
            :math:`\\beta_{\\text{Reason}}`. M2 sets this to ``False`` because
            the indicator is identically one on its rows.
        include_human: include the human block. ``False`` for the LLM-only
            fits M2 and M3, where the block would never fire.
    """
    f = features
    task_idx = f["task_idx"]
    context_idx = f["context_idx"]
    instruction_idx = f["instruction_idx"]
    is_ai = f["is_ai"].astype(jnp.float32)

    eta = _scalar("alpha", priors["alpha"])

    # Task fixed effect ------------------------------------------------------
    task_effect = _treatment_coded_vec("task_effect", int(f["num_task"]), priors["task_effect"])
    eta = eta + task_effect[task_idx]

    # Scenario block (eq:m1_scen) -------------------------------------------
    case_re = _random_effect_vec("case_re", int(f["num_case"]), "sigma_case", priors["sigma_case"])
    scenario_re = _random_effect_vec(
        "scenario_re", int(f["num_scenario"]), "sigma_scenario", priors["sigma_scenario"],
    )
    party_effect = _treatment_coded_vec("party_type_effect", int(f["num_party"]), priors["party_type_effect"])
    eta = (
        eta
        + case_re[f["case_idx"]]
        + scenario_re[f["scenario_idx"]]
        + party_effect[f["party_idx"]]
        + _scalar("beta_concealment", priors["beta_concealment"]) * f["is_concealment"]
        + _scalar("beta_order", priors["beta_order"]) * f["is_fraud_first"]
    )

    # Design block (eq:m1_design), applied to every row ----------------------
    n_instr, n_ctx, n_task = int(f["num_instruction"]), int(f["num_context"]), int(f["num_task"])
    instruction_effect = _treatment_coded_vec("instruction_effect", n_instr, priors["instruction_effect"])
    context_effect = _treatment_coded_vec("context_effect", n_ctx, priors["context_effect"])
    instr_ctx = _shrunk_interaction_matrix(
        "instruction_context_effect", n_instr, n_ctx,
        "sigma_instruction_context", priors["sigma_instruction_context"],
    )
    task_ctx = _treatment_coded_matrix("task_context_effect", n_task, n_ctx, priors["task_context_effect"])
    eta = (
        eta
        + instruction_effect[instruction_idx]
        + context_effect[context_idx]
        + instr_ctx[instruction_idx, context_idx]
        + task_ctx[task_idx, context_idx]
    )

    # LLM block (eq:m1_runtime), LLM rows only ------------------------------
    model_re = _model_random_intercepts(f, priors)
    runtime = (
        model_re[f["model_idx"]]
        + _scalar("beta_high_temperature", priors["beta_high_temperature"]) * f["high_temperature"]
        + _scalar("beta_high_reasoning", priors["beta_high_reasoning"]) * f["high_reasoning"]
        + _scalar("beta_post_cutoff", priors["beta_post_cutoff"]) * f["post_cutoff_case"]
    )
    if include_reasoning:
        runtime = runtime + _scalar("beta_reasoning", priors["beta_reasoning"]) * f["reasoning_on"]
    eta = eta + is_ai * runtime

    # Human block (eq:m1_human), human rows only ----------------------------
    if include_human:
        is_human = f["is_human"].astype(jnp.float32)
        beta_human = _scalar("beta_human", priors["beta_human"])
        human_ctx = _treatment_coded_vec("human_context_effect", n_ctx, priors["human_context_effect"])
        participant_re = _random_effect_vec(
            "participant_re", int(f["num_participant"]), "sigma_participant", priors["sigma_participant"],
        )
        age = _random_effect_vec("age_effect", int(f["num_age"]), "sigma_age", priors["sigma_age"])
        race = _random_effect_vec("race_effect", int(f["num_race"]), "sigma_race", priors["sigma_race"])
        gender = _random_effect_vec("gender_effect", int(f["num_gender"]), "sigma_gender", priors["sigma_gender"])
        ai_use = _random_effect_vec("ai_use_effect", int(f["num_ai_use"]), "sigma_ai_use", priors["sigma_ai_use"])
        human = (
            beta_human
            + human_ctx[context_idx]
            + participant_re[f["participant_idx"]]
            + age[f["age_idx"]]
            + race[f["race_idx"]]
            + gender[f["gender_idx"]]
            + ai_use[f["ai_use_idx"]]
        )
        eta = eta + is_human * human

    return eta


# --------------------------------------------------------------------------
# M3 judge block.
# --------------------------------------------------------------------------

def m3_judge_block(features, priors: dict) -> jnp.ndarray:
    """Pair-level judge block :math:`\\xi_{ij}` of M3 (eq. ``eq:m3_eta``)."""
    f = features
    num_judge = int(f["num_judge"])
    judge_idx = f["judge_idx"]

    # Judge severity, effect-coded (sums to zero): num_judge - 1 free entries.
    if num_judge <= 1:
        judge_effect = jnp.zeros((max(num_judge, 1),))
    else:
        raw = numpyro.sample(
            "judge_effect_raw", dist.Normal(0.0, priors["judge_effect"]).expand([num_judge - 1]),
        )
        judge_effect = jnp.concatenate([raw, -jnp.sum(raw, keepdims=True)])
    numpyro.deterministic("judge_effect", judge_effect)

    judge_model_re = _random_effect_matrix(
        "judge_model_re", num_judge, int(f["num_model"]),
        "sigma_judge_model", priors["sigma_judge_model"],
    )
    judge_scenario_re = _random_effect_matrix(
        "judge_scenario_re", num_judge, int(f["num_scenario"]),
        "sigma_judge_scenario", priors["sigma_judge_scenario"],
    )
    return (
        judge_effect[judge_idx]
        + _scalar("beta_self_family", priors["beta_self_family"]) * f["is_self_family"]
        + _scalar("beta_self_model", priors["beta_self_model"]) * f["is_self_model"]
        + judge_model_re[judge_idx, f["model_idx"]]
        + judge_scenario_re[judge_idx, f["scenario_idx"]]
    )


# --------------------------------------------------------------------------
# Builders registered with HiBayes.
# --------------------------------------------------------------------------

def _set_log_likelihood_storage(store: bool) -> None:
    """Tell arviz whether ``az.from_numpyro`` should materialise the pointwise
    log-likelihood group (``n_draws x n_obs`` floats) when HiBayes converts the
    MCMC run. M1 and M2 keep it so HiBayes's ``loo`` / ``waic`` checkers can
    run; M3 has three rows per generation and skips it by default so the fit
    does not run out of memory at conversion time."""
    az.rcParams["data.log_likelihood"] = bool(store)


@model
def fraudbench_m1(
    tag: str = "",
    priors: Optional[dict] = None,
    store_log_likelihood: bool = True,
) -> Model:
    """M1: main regression on the fraud outcome, full evaluation panel.

    ``tag`` is consumed by HiBayes to name the output directory. ``priors``
    may override entries of :data:`PRIORS` (scales only).
    """
    resolved = _resolve_priors(priors)
    _set_log_likelihood_storage(store_log_likelihood)

    def _m1(features) -> None:
        check_features(features, _M1_REQUIRED + _HUMAN_REQUIRED)
        eta = m1_predictor(features, resolved, include_reasoning=True, include_human=True)
        numpyro.sample("obs", dist.Bernoulli(logits=eta), obs=features["obs"])

    return _m1


@model
def fraudbench_m2(
    tag: str = "",
    priors: Optional[dict] = None,
    store_log_likelihood: bool = True,
) -> Model:
    """M2: reasoning-tokens variant (eq. ``eq:m2_eta``), LLM rows with a
    positive reasoning-token count only.

    Requires ``filter_ai_only`` -> ``filter_reasoning_tokens_positive`` ->
    ``build_fraudbench_indices`` to have run so that reasoning was enabled
    on every retained row; the reasoning-enabled main effect is dropped.
    """
    resolved = _resolve_priors(priors)
    _set_log_likelihood_storage(store_log_likelihood)

    def _m2(features) -> None:
        check_features(features, _M1_REQUIRED + _M2_REQUIRED)
        eta = m1_predictor(features, resolved, include_reasoning=False, include_human=False)
        z_tok = features["z_log_reasoning_tokens"]
        eta = (
            eta
            + _scalar("beta_tokens", resolved["beta_tokens"]) * z_tok
            + _scalar("beta_high_reasoning_tokens", resolved["beta_high_reasoning_tokens"])
            * features["high_reasoning"] * z_tok
        )
        numpyro.sample("obs", dist.Bernoulli(logits=eta), obs=features["obs"])

    return _m2


@model
def fraudbench_m3(
    tag: str = "",
    priors: Optional[dict] = None,
    store_log_likelihood: bool = False,
) -> Model:
    """M3: self-preference variant on (generation, judge) pairs (eq. ``eq:m3_eta``).

    Requires ``filter_ai_only`` -> ``expand_to_judge_votes`` ->
    ``build_fraudbench_indices`` -> ``build_judge_indices``.
    """
    resolved = _resolve_priors(priors)
    _set_log_likelihood_storage(store_log_likelihood)

    def _m3(features) -> None:
        check_features(features, _M1_REQUIRED + _M3_REQUIRED)
        eta = m1_predictor(features, resolved, include_reasoning=True, include_human=False)
        xi = m3_judge_block(features, resolved)
        numpyro.sample("obs", dist.Bernoulli(logits=eta + xi), obs=features["obs"])

    return _m3


__all__ = [
    "PRIORS",
    "fraudbench_m1",
    "fraudbench_m2",
    "fraudbench_m3",
    "m1_predictor",
    "m3_judge_block",
]
