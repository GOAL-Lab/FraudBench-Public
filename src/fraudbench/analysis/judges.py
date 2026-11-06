"""Per-judge vote helpers shared by the LLM-as-a-judge analysis modules.

Every kept response is graded by a three-member LLM-as-a-judge jury on three
dimensions (``fraud``, ``refusal``, ``options``). The canonical
``data/final/obs_data_per_score.parquet`` stores one row per
``(sample_id, scorer_model)`` with the individual votes in ``<metric>_obs``.
Older per-sample artefacts instead store one JSON cell per dimension --
``score_<metric>_all`` -- holding a ``{judge_model: bool | null}`` mapping.

This module centralises everything needed to read those cells: naming the
columns, parsing the JSON, discovering which judges voted, and expanding the
votes into one nullable-boolean column per judge. Table and plot modules that
reason about individual judges (rather than the majority vote) should go
through here instead of re-implementing the parsing.
"""
from __future__ import annotations

import json
from typing import Sequence

import numpy as np
import pandas as pd

from .common import get_model_display_name
from .registry import (
    GRADER_REASONING_MODELS,
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
    REASONING_SYMBOL,
)


METRICS: tuple[str, ...] = ("fraud", "refusal", "options")
"""Scoring dimensions graded by the jury."""

METRIC_DISPLAY: dict[str, str] = {
    "fraud": "Fraud",
    "refusal": "Refusal",
    "options": "Options",
}
"""Display names for the graded dimensions."""

VOTES_COLUMN: dict[str, str] = {metric: f"score_{metric}_all" for metric in METRICS}
"""Legacy per-sample vote-dictionary column for each metric."""

OBS_COLUMN: dict[str, str] = {
    "fraud": "fraud_obs",
    "refusal": "refusal_obs",
    "options": "option_obs",
}
"""Per-score observation column for each metric."""

SAMPLE_ID_COLUMN = "sample_id"
SCORER_COLUMN = "scorer_model"
GRADER_RAN_COLUMN = "grader_ran"


def votes_columns(metrics: Sequence[str] = METRICS) -> list[str]:
    """Vote column names for *metrics*, for parquet column pushdown."""
    return [VOTES_COLUMN[metric] for metric in metrics]


def require_votes_columns(
    df: pd.DataFrame,
    metrics: Sequence[str] = METRICS,
    *,
    source: str = "the input data",
) -> None:
    """Raise if any metric's per-judge vote column is missing from *df*."""
    missing = [VOTES_COLUMN[m] for m in metrics if VOTES_COLUMN[m] not in df.columns]
    if missing:
        raise ValueError(
            f"{source} is missing required columns: {missing}. Expected the "
            "per-judge vote dict columns written by analysis/process.py; run "
            "`uv run analysis/process.py` if the parquet is stale."
        )


def is_per_score_data(df: pd.DataFrame) -> bool:
    """Whether *df* uses the one-row-per-scorer observation schema."""
    return SCORER_COLUMN in df.columns


def require_individual_vote_columns(
    df: pd.DataFrame,
    metrics: Sequence[str] = METRICS,
    *,
    source: str = "the input data",
) -> None:
    """Validate individual votes in either supported observation schema."""
    if is_per_score_data(df):
        required = [SAMPLE_ID_COLUMN, SCORER_COLUMN, GRADER_RAN_COLUMN]
        required.extend(OBS_COLUMN[metric] for metric in metrics)
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(
                f"{source} is missing required per-score columns: {missing}. "
                "Expected sample_id, scorer_model, grader_ran, and the "
                "<metric>_obs columns written by analysis/process.py --per-score."
            )
        return
    require_votes_columns(df, metrics, source=source)


def collapse_per_score_rows(
    df: pd.DataFrame,
    metrics: Sequence[str] = METRICS,
) -> pd.DataFrame:
    """Collapse long-form scorer rows to one row per evaluated response.

    Sample-level columns are taken from the first row for each ``sample_id``.
    For every requested metric, scorer observations are pivoted into the
    legacy ``score_<metric>_all`` dictionary representation used internally by
    the existing analysis code. A scorer that ran but returned an unparseable
    value is retained with a ``None`` vote.

    Per-sample inputs are returned unchanged, which keeps callers compatible
    with older ``obs_data.parquet`` files.
    """
    if not is_per_score_data(df):
        return df

    require_individual_vote_columns(df, metrics)
    if df[SAMPLE_ID_COLUMN].isna().any() or df[SAMPLE_ID_COLUMN].eq("").any():
        raise ValueError("Per-score data contains rows without a sample_id.")
    if df[SCORER_COLUMN].isna().any() or df[SCORER_COLUMN].eq("").any():
        raise ValueError("Per-score data contains rows without a scorer_model.")

    ran = df[GRADER_RAN_COLUMN].fillna(False).astype(bool)
    scores = df.loc[
        ran,
        [SAMPLE_ID_COLUMN, SCORER_COLUMN, *(OBS_COLUMN[m] for m in metrics)],
    ]
    duplicate = scores.duplicated([SAMPLE_ID_COLUMN, SCORER_COLUMN], keep=False)
    if duplicate.any():
        examples = (
            scores.loc[duplicate, [SAMPLE_ID_COLUMN, SCORER_COLUMN]]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Per-score data has duplicate (sample_id, scorer_model) rows; "
            f"examples: {examples}."
        )

    scorer_specific = {
        SCORER_COLUMN,
        GRADER_RAN_COLUMN,
        *OBS_COLUMN.values(),
        *VOTES_COLUMN.values(),
    }
    sample_columns = [column for column in df.columns if column not in scorer_specific]
    base = (
        df[sample_columns]
        .drop_duplicates(SAMPLE_ID_COLUMN, keep="first")
        .set_index(SAMPLE_ID_COLUMN, drop=False)
    )

    for metric in metrics:
        wide = scores.pivot(
            index=SAMPLE_ID_COLUMN,
            columns=SCORER_COLUMN,
            values=OBS_COLUMN[metric],
        ).reindex(base.index)
        base[VOTES_COLUMN[metric]] = [
            {
                judge: None if pd.isna(value) else bool(value)
                for judge, value in row.items()
            }
            for row in wide.to_dict(orient="records")
        ]

    return base.reset_index(drop=True)


def parse_votes(value: object) -> dict:
    """Parse one ``score_<metric>_all`` cell into a ``{judge: bool | None}`` dict.

    Cells are JSON strings like ``{"<judge>": <bool|null>, ...}``. Returns
    ``{}`` for missing / unparseable cells so downstream masks treat missing
    judges uniformly.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, float) and np.isnan(value):
        return {}
    if isinstance(value, str):
        if not value or value == "{}":
            return {}
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def discover_judges(
    df: pd.DataFrame,
    metrics: Sequence[str] = METRICS,
) -> tuple[str, ...]:
    """Judge models referenced anywhere in the per-judge vote columns.

    Sorted by :data:`fraudbench.analysis.registry.MODEL_ORDER` so callers
    respect the project-wide model ordering; ids unknown to ``models.jsonl``
    are appended alphabetically. Only the distinct raw cell values are
    parsed, which keeps the scan cheap on the full observation frame.
    """
    if is_per_score_data(df):
        seen = set(df.loc[df[GRADER_RAN_COLUMN].fillna(False), SCORER_COLUMN].dropna())
        ordered = [model for model in MODEL_ORDER if model in seen]
        extras = sorted(judge for judge in seen if judge not in MODEL_ORDER)
        return tuple(ordered + extras)

    seen: set[str] = set()
    for metric in metrics:
        column = VOTES_COLUMN[metric]
        if column not in df.columns:
            continue
        for raw in df[column].drop_duplicates():
            seen.update(parse_votes(raw).keys())
    ordered = [model for model in MODEL_ORDER if model in seen]
    extras = sorted(judge for judge in seen if judge not in MODEL_ORDER)
    return tuple(ordered + extras)


def expand_votes(
    df: pd.DataFrame,
    metric: str,
    judges: Sequence[str],
    *,
    prefix: str = "",
) -> pd.DataFrame:
    """Expand one metric's vote dicts into a nullable-boolean column per judge.

    Returns a frame indexed like *df* with one ``boolean`` column per judge,
    named ``f"{prefix}{judge}"``. Judges missing from a cell, and votes the
    grader failed to parse (JSON ``null``), both surface as ``pd.NA``.
    """
    parsed = df[VOTES_COLUMN[metric]].map(parse_votes)
    out = pd.DataFrame(index=df.index)
    for judge in judges:
        raw = parsed.map(lambda votes, j=judge: votes.get(j, None))
        out[f"{prefix}{judge}"] = pd.array(
            [
                pd.NA if v is None or (isinstance(v, float) and np.isnan(v)) else bool(v)
                for v in raw
            ],
            dtype="boolean",
        )
    return out


def judge_display_name(judge_id: str) -> str:
    """Canonical display name for a judge, with the reasoning marker appended.

    The marker is added for graders configured to run with reasoning on; see
    :data:`fraudbench.analysis.registry.GRADER_REASONING_MODELS`, which is
    deliberately separate from the main-protocol ``REASONING_MODELS``.
    """
    name = MODEL_DISPLAY_NAMES.get(judge_id, get_model_display_name(judge_id))
    if judge_id in GRADER_REASONING_MODELS:
        return f"{name} {REASONING_SYMBOL}"
    return name
