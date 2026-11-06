"""Custom Hibayes checkers for fraudbench regressions.

The built-in ``posterior_predictive_plot`` checker generates the
posterior-predictive distribution AND draws a matplotlib figure that asks
the user (via the CLI) whether the predictive distribution looks
acceptable. That interactive prompt makes it unusable in non-interactive
batch pipelines.

This module provides a stripped-down, non-interactive variant that
mirrors the predictive-sampling logic of
``hibayes.check.checkers.posterior_predictive_plot`` but skips the
plotting / prompting entirely. The resulting samples are written to
``state.inference_data`` under the ``posterior_predictive`` group, which
Hibayes persists as part of the standard ``inference_data.nc`` artifact
on disk.
"""

from __future__ import annotations

from typing import Tuple

import arviz as az
import jax
from numpyro.infer import Predictive

from hibayes.analysis_state import ModelAnalysisState
from hibayes.check._check import Checker, CheckerResult, checker
from hibayes.ui import ModellingDisplay


def _posterior_samples_as_dict(idata: az.InferenceData) -> dict:
    """Convert ``idata.posterior`` into the dict format ``Predictive`` expects.

    Stacks the ``(chain, draw, ...)`` arrays into ``(chain * draw, ...)``,
    matching the helper inside hibayes's own ``posterior_predictive_plot``.
    """
    out: dict = {}
    for v in idata.posterior.data_vars:
        arr = idata.posterior[v].values
        out[v] = arr.reshape(-1, *arr.shape[2:])
    return out


@checker
def posterior_predictive(
    num_samples: int = 2000,
    predictive_kwargs: dict | None = None,
) -> Checker:
    """Generate posterior-predictive samples without plotting or prompting.

    Mirrors the predictive-sampling step performed inside Hibayes's
    ``posterior_predictive_plot`` checker, but does not create a figure
    and does not request CLI confirmation from the user. Useful for
    batch fits where the predictive samples are needed downstream
    (e.g. for posterior-predictive figures generated outside the fit
    pipeline) but no human-in-the-loop validation is desired.

    The samples are attached to ``state.inference_data`` under the
    ``posterior_predictive`` group via ``InferenceData.extend``. Hibayes
    saves ``state.inference_data`` to ``inference_data.nc`` in the per-
    model output directory at the end of the fit, so the predictive
    group is persisted automatically.

    Args:
        num_samples: Forwarded to ``numpyro.infer.Predictive``. When
            ``posterior_samples`` is provided, numpyro draws one
            predictive replicate per posterior sample, so this argument
            mainly serves as a sanity hint.
        predictive_kwargs: Extra keyword arguments forwarded to
            ``numpyro.infer.Predictive`` (e.g. ``return_sites``).

    Returns:
        A Hibayes-compatible checker function.
    """
    if predictive_kwargs is None:
        predictive_kwargs = {}

    def check(
        state: ModelAnalysisState,
        display: ModellingDisplay | None = None,
    ) -> Tuple[ModelAnalysisState, CheckerResult]:
        if state.inference_data.get("posterior_predictive") is not None:
            if display is not None:
                display.logger.info(
                    "posterior_predictive: skipping; group already present "
                    "on inference_data."
                )
            return state, "NA"

        rng_key = jax.random.PRNGKey(state.model_config.fit.seed + 1)
        predictive = Predictive(
            state.model,
            posterior_samples=_posterior_samples_as_dict(state.inference_data),
            num_samples=num_samples,
            # Vectorising Predictive is much heavier than the in-MCMC
            # vectorisation -- match posterior_predictive_plot and run
            # sequentially to avoid OOM on the larger fraudbench fits.
            parallel=False,
            **predictive_kwargs,
        )
        pp_samples = predictive(rng_key, state.prior_features)

        pp_ds = az.from_numpyro(posterior_predictive=pp_samples)
        state.inference_data.extend(pp_ds)

        if display is not None:
            n_vars = len(list(pp_ds.posterior_predictive.data_vars))
            display.logger.info(
                f"posterior_predictive: stored {n_vars} variable(s) under "
                "inference_data.posterior_predictive."
            )
        return state, "NA"

    return check
