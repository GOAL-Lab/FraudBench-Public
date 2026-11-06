"""Helpers for loading ``data/final/dataset_public.jsonl``.

The public dataset nests per-scenario internals under ``metadata``, ``seed``,
and ``human_baseline`` objects (see ``data/README.md`` for the full schema).
Most of our analysis / evaluation scripts find it more convenient to work
with a flat DataFrame where those nested keys live as top-level columns.

The :func:`flatten_public_dataset` function promotes the keys inside each
nested object to flat columns:

* ``metadata.<key>`` → ``<key>`` (e.g. ``metadata.case_id`` → ``case_id``,
  ``metadata.is_concealment`` → ``is_concealment``).
* ``seed.<key>`` → ``<key>`` (e.g. ``seed.plaintiff_name`` →
  ``plaintiff_name``, ``seed.misrep_actual_facts`` →
  ``misrep_actual_facts``).
* ``human_baseline.<key>`` → ``human_baseline_<key>`` (e.g.
  ``human_baseline.system_prompt`` → ``human_baseline_system_prompt``,
  ``human_baseline.vignette`` → ``human_baseline_vignette``,
  ``human_baseline.query_open`` → ``human_baseline_query_open``).

Top-level public columns (``summary_id``, ``case_name``, ``case_citation``,
``court``, ``date``, ``state``, ``jurisdiction``, ``level``, ``naics_sector``,
``system_prompt``, ``vignette``, ``query_binary``, ``query_open``,
``scoring_criteria_binary``, ``scoring_criteria_open``, ``agentic_trajectory``,
``agentic_toolkits``, ``experiment_*``) are passed through unchanged. The
function is a no-op for rows that do not contain a given nested object, which
makes it safe to call on intermediate JSONLs that have not yet been packaged
into the public schema.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


METADATA_FIELD = "metadata"
SEED_FIELD = "seed"
HUMAN_BASELINE_FIELD = "human_baseline"
HUMAN_BASELINE_PREFIX = "human_baseline_"


def _expand_nested(df: pd.DataFrame, field: str, prefix: str = "") -> pd.DataFrame:
    """Promote keys from ``df[field]`` (a column of dicts) into flat columns.

    Rows where ``df[field]`` is missing or not a dict contribute NaN. The
    nested column itself is dropped from the result. If ``df`` does not
    contain ``field``, the input is returned unchanged.
    """
    if field not in df.columns:
        return df
    nested = df[field].apply(lambda v: v if isinstance(v, dict) else {})
    expanded = pd.json_normalize(nested).add_prefix(prefix)
    expanded.index = df.index
    out = df.drop(columns=[field]).copy()
    for col in expanded.columns:
        if col in out.columns:
            out[col] = out[col].combine_first(expanded[col])
        else:
            out[col] = expanded[col]
    return out


def flatten_public_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with ``metadata`` / ``seed`` / ``human_baseline`` flattened."""
    out = _expand_nested(df, METADATA_FIELD)
    out = _expand_nested(out, SEED_FIELD)
    out = _expand_nested(out, HUMAN_BASELINE_FIELD, prefix=HUMAN_BASELINE_PREFIX)
    return out


def load_public_dataset(
    path: str,
    *,
    flatten: bool = True,
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load ``data/final/dataset_public.jsonl`` into a DataFrame.

    Args:
        path: Path to the public-dataset JSONL.
        flatten: If True (default), call :func:`flatten_public_dataset` on the
            loaded frame so that nested ``metadata`` / ``seed`` /
            ``human_baseline`` keys appear as flat columns.
        columns: Optional iterable of columns to subset to after flattening.
            Missing columns are silently ignored.
    """
    df = pd.read_json(path, lines=True)
    if flatten:
        df = flatten_public_dataset(df)
    if columns is not None:
        wanted = [c for c in columns if c in df.columns]
        df = df[wanted]
    return df
