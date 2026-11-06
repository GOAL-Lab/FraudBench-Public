"""Cached I/O helpers for HiBayes regression artefacts.

Three artefacts written by ``hibayes-model`` are bundled together so
downstream code never has to re-derive index-to-label mappings:

* ``inference_data.nc``        -- the posterior. Only the ``posterior``
                                  group is read (the ``log_likelihood`` /
                                  ``posterior_predictive`` groups are several
                                  GB and unused here).
* ``processed_data.parquet``   -- the per-row processed frame written by the
                                  regression's ``data_process`` step. Carries
                                  every index column (``model_idx``,
                                  ``task_idx``, ...) alongside the original
                                  ``model``, ``setting``, ``model_family``, ...
* ``features.pkl``             -- the feature dict published by
                                  ``build_fraudbench_indices``: per-model
                                  arrays, level counts and level lists.

Loads are memoised by absolute path because a table or plot script may
request the same bundle many times.
"""
from __future__ import annotations

import functools
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr


def _abspath(path: str | os.PathLike) -> str:
    return str(Path(path).expanduser().resolve())


@functools.lru_cache(maxsize=4)
def load_posterior(inference_data_path: str | os.PathLike) -> xr.Dataset:
    """Load the ``posterior`` group of a HiBayes ``inference_data.nc`` file."""
    abspath = _abspath(inference_data_path)
    if not Path(abspath).is_file():
        raise FileNotFoundError(
            f"inference_data.nc not found: {inference_data_path!r}. "
            "Expected the .nc file produced by hibayes-model."
        )
    with xr.open_dataset(abspath, group="posterior") as ds:
        return ds.load()


# Free-text columns carried through from ``obs_data.parquet``. No table or
# plot code reads them, and on the M3 frame (~895k rows) they alone take
# >10 GB in pandas, so they are never loaded here.
_PROCESSED_TEXT_COLUMNS = frozenset({
    "completion", "completion_extended", "user_chat", "system_prompt",
    "reasoning_summary", "reasoning_history", "cot_reasoning", "cot_summary",
    "log", "log_path", "scorer_tokens",
    "score_fraud_all", "score_refusal_all", "score_options_all",
    "score_human_baseline_second_round_instruction_following_all",
    "score_human_baseline_second_round_fraud_all",
})


@functools.lru_cache(maxsize=4)
def load_processed_data(processed_data_path: str | os.PathLike) -> pd.DataFrame:
    """Load the ``processed_data.parquet`` written by the regression pipeline.

    The free-text columns in :data:`_PROCESSED_TEXT_COLUMNS` are skipped so
    the frame fits comfortably in memory; everything else is read as-is.
    """
    abspath = _abspath(processed_data_path)
    if not Path(abspath).is_file():
        raise FileNotFoundError(
            f"processed_data.parquet not found: {processed_data_path!r}. "
            "This file is written by the data_process step of the regression pipeline."
        )
    import pyarrow.parquet as pq

    names = pq.read_schema(abspath).names
    columns = [c for c in names if c not in _PROCESSED_TEXT_COLUMNS]
    return pd.read_parquet(abspath, columns=columns)


@functools.lru_cache(maxsize=4)
def load_features(features_path: str | os.PathLike) -> dict:
    """Load ``features.pkl`` (JAX arrays converted to NumPy)."""
    abspath = _abspath(features_path)
    if not Path(abspath).is_file():
        raise FileNotFoundError(
            f"features.pkl not found: {features_path!r}. "
            "This file is written by hibayes-model under the model dir."
        )
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    with open(abspath, "rb") as f:
        raw = pickle.load(f)
    return {k: (np.asarray(v) if hasattr(v, "shape") else v) for k, v in raw.items()}


def stack_draws(da: xr.DataArray) -> np.ndarray:
    """Stack ``(chain, draw, ...)`` into ``(chain * draw, ...)``."""
    if "chain" not in da.dims or "draw" not in da.dims:
        raise ValueError(f"DataArray missing chain/draw dims; got {da.dims}")
    return np.asarray(da.stack(sample=("chain", "draw")).transpose("sample", ...).values)


@dataclass(frozen=True)
class PosteriorBundle:
    """The three artefacts of one HiBayes regression run."""

    posterior: xr.Dataset
    processed: pd.DataFrame
    features: dict

    @property
    def n_draws(self) -> int:
        return int(self.posterior.sizes["chain"]) * int(self.posterior.sizes["draw"])

    @property
    def num_model(self) -> int:
        return int(self.features["num_model"])

    def levels(self, name: str, default: list[str]) -> list[str]:
        """Level labels stored by ``build_fraudbench_indices`` (``task_levels``, ...)."""
        return list(self.features.get(name, default))

    @functools.cached_property
    def draws(self) -> dict[str, np.ndarray]:
        """Every posterior site as ``(n_draws, ...)`` NumPy arrays."""
        return {v: stack_draws(self.posterior[v]) for v in self.posterior.data_vars}

    @functools.cached_property
    def model_table(self) -> pd.DataFrame:
        """One row per ``model_idx`` (length ``num_model``) with the model id
        used at fit time, its family, size tier and reasoning flag."""
        ai = self.processed[self.processed["model"] != "human"]
        cols = ["model_idx", "model", "model_family", "model_size", "model_reasoning",
                "model_open_source", "model_geo"]
        present = [c for c in cols if c in ai.columns]
        table = (
            ai[present].drop_duplicates(subset=["model_idx"]).sort_values("model_idx").reset_index(drop=True)
        )
        if len(table) != self.num_model:
            raise RuntimeError(
                f"model_table has {len(table)} rows but num_model={self.num_model}; "
                "processed_data may be inconsistent with features.pkl."
            )
        return table

    @functools.cached_property
    def family_table(self) -> pd.DataFrame:
        """One row per family index: ``family_idx``, ``model_family``."""
        names = self.features.get("family_names")
        if names is not None:
            return pd.DataFrame({"family_idx": range(len(names)), "model_family": list(names)})
        family_idx_by_model = np.asarray(self.features["family_idx_by_model"]).astype(int)
        models = self.model_table.assign(family_idx=family_idx_by_model)
        return (
            models.groupby("family_idx", observed=True)["model_family"].first()
            .reset_index().sort_values("family_idx").reset_index(drop=True)
        )


def load_bundle(
    inference_data_path: str | os.PathLike,
    processed_data_path: Optional[str | os.PathLike] = None,
    features_path: Optional[str | os.PathLike] = None,
) -> PosteriorBundle:
    """Load all three regression artefacts as a :class:`PosteriorBundle`.

    ``processed_data_path`` defaults to ``<run>/processed_data.parquet`` two
    levels above the model directory and ``features_path`` to
    ``<model dir>/features.pkl``, matching the ``hibayes-model`` layout.
    """
    inference_data_path = Path(inference_data_path)
    if processed_data_path is None:
        processed_data_path = inference_data_path.parent.parent.parent / "processed_data.parquet"
    if features_path is None:
        features_path = inference_data_path.parent / "features.pkl"
    return PosteriorBundle(
        posterior=load_posterior(inference_data_path),
        processed=load_processed_data(processed_data_path),
        features=load_features(features_path),
    )
