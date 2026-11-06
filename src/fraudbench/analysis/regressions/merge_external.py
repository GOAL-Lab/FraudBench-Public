"""External-data merge processors for the ``fraudbench_*`` Hibayes models.

The processors in :mod:`process.py` (``build_fraudbench_indices`` etc.) expect
``state.processed_data`` to already contain every column listed in
``data/README.md``. As of the latest data refactor, the per-row evaluation
parquet (``data/final/obs_data.parquet``) intentionally only carries the
sample-level columns that come straight out of the Inspect-AI logs; the
scenario-, case-, model-, and respondent-level metadata live in separate
canonical files and are joined on demand by the processors below.

Run order in a typical regression YAML::

    data_process:
      path:
        - src/fraudbench/analysis/regressions/merge_external.py
        - src/fraudbench/analysis/regressions/process.py
      processors:
        - merge_scenario_metadata     # adds case_id, case_date,
                                      # case_parties_type, is_concealment,
                                      # is_fraud_first
        - merge_model_metadata        # adds model_family, model_size,
                                      # model_parameters, model_release_date,
                                      # model_knowledge_cutoff, model_open_source,
                                      # model_reasoning, model_geo
        - merge_human_baseline_metadata  # adds human_age, human_sex,
                                         # human_ethnicity_simplified, human_ai_use
        - build_fraudbench_indices
        - select_outcome:
            outcome_column: "score_fraud"

Each processor logs how many rows received each merged column, so a missed
join is visible in the run log. Paths are workspace-relative by default and
can be overridden in YAML; all processors also accept absolute paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from hibayes.analysis import AnalysisState
from hibayes.process import process
from hibayes.ui import ModellingDisplay

from fraudbench.analysis.registry import normalize_model_id


# --------------------------------------------------------------------------
# Default canonical paths (resolved relative to the repo root).
# --------------------------------------------------------------------------

DEFAULT_SCENARIO_PATH = "data/final/dataset_public.jsonl"
DEFAULT_MODELS_PATH = "data/final/models.jsonl"
DEFAULT_HUMAN_BASELINE_PATH = "data/final/human_baseline/human_baseline_public.jsonl"


def _resolve_repo_root() -> Path:
    """Walk up from this file to find the repo root (sibling ``data/`` dir)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data").is_dir():
            return parent
    return Path.cwd()


def _resolve_path(path: str) -> Path:
    """Resolve ``path`` to an existing file.

    Order of resolution:
      1. As-is (absolute or cwd-relative).
      2. Relative to the repo root located by walking up from this module.
    """
    p = Path(path)
    if p.is_file():
        return p
    repo_root = _resolve_repo_root()
    candidate = repo_root / path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Could not find {path!r}. Tried {p.resolve()} and {candidate}."
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSON-lines file into a list of dicts; raise on bad lines."""
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {lineno} of {path}: {e}"
                ) from e
    return rows


def _log_merge(
    display: Optional[ModellingDisplay],
    name: str,
    df: pd.DataFrame,
    new_cols: Iterable[str],
) -> None:
    """Emit a one-line summary of post-merge non-null counts per added column."""
    if display is None:
        return
    parts = []
    for c in new_cols:
        if c in df.columns:
            parts.append(f"{c}={int(df[c].notna().sum())}")
        else:
            parts.append(f"{c}=MISSING")
    display.logger.info(
        f"{name}: {len(df)} rows; non-null counts: " + ", ".join(parts)
    )


# ==========================================================================
# 1. Scenario metadata  (data/final/dataset_public.jsonl)
# ==========================================================================


def _derive_case_parties_type(plaintiff_is_org: Any, defendant_is_org: Any) -> Optional[str]:
    """Map ``(*_is_org)`` flags to the ``B2B / B2C / C2B / C2C`` label.

    The first letter is the **defendant** (alleged perpetrator), the second
    is the **plaintiff** (alleged victim). Mirrors the convention used in
    :mod:`fraudbench.analysis.table.party_types`.
    """
    if plaintiff_is_org is None or defendant_is_org is None:
        return None
    try:
        d = bool(defendant_is_org)
        p = bool(plaintiff_is_org)
    except (TypeError, ValueError):
        return None
    return f"{'B' if d else 'C'}2{'B' if p else 'C'}"


@process
def merge_scenario_metadata(path: str = DEFAULT_SCENARIO_PATH):
    """Join scenario-level metadata onto ``state.processed_data`` on ``summary_id``.

    Reads ``dataset_public.jsonl`` (one record per scenario) and adds:

    * ``case_id`` (int, from ``metadata.case_id``)
    * ``case_date`` (str, from the top-level ``date`` field)
    * ``case_parties_type`` (str, derived from ``seed.plaintiff_is_org`` and
      ``seed.defendant_is_org``; one of ``B2B`` / ``B2C`` / ``C2B`` / ``C2C``)
    * ``is_concealment`` (bool, from ``metadata.is_concealment``)
    * ``is_fraud_first`` (bool, from ``metadata.is_fraud_first``)

    Args:
        path: Path to the scenario JSONL file. Defaults to
            ``data/final/dataset_public.jsonl``.
    """

    def processor(
        state: AnalysisState,
        display: Optional[ModellingDisplay] = None,
    ) -> AnalysisState:
        df = state.processed_data.copy()
        if "summary_id" not in df.columns:
            raise ValueError(
                "merge_scenario_metadata: 'summary_id' is required on the "
                "incoming data but is missing."
            )

        records = _read_jsonl(_resolve_path(path))
        missing_date = [
            r.get("summary_id") for r in records if "date" not in r
        ]
        if missing_date:
            preview = ", ".join(str(summary_id) for summary_id in missing_date[:5])
            raise ValueError(
                f"merge_scenario_metadata: {path} has {len(missing_date)} row(s) "
                f"without the top-level 'date' field (for example: {preview})."
            )
        scenario_rows = []
        for r in records:
            meta = r.get("metadata") or {}
            seed = r.get("seed") or {}
            scenario_rows.append({
                "summary_id": r.get("summary_id"),
                "case_id": meta.get("case_id"),
                "case_date": r.get("date"),
                "is_concealment": meta.get("is_concealment"),
                "is_fraud_first": meta.get("is_fraud_first"),
                "case_parties_type": _derive_case_parties_type(
                    seed.get("plaintiff_is_org"),
                    seed.get("defendant_is_org"),
                ),
            })
        scenario_df = pd.DataFrame(scenario_rows)
        if scenario_df["summary_id"].duplicated().any():
            n_dup = int(scenario_df["summary_id"].duplicated().sum())
            raise ValueError(
                f"merge_scenario_metadata: {path} contains {n_dup} duplicate "
                "summary_id rows; expected one record per scenario."
            )

        new_cols = [
            "case_id", "case_date", "case_parties_type", "is_concealment",
            "is_fraud_first",
        ]
        df = df.drop(columns=[c for c in new_cols if c in df.columns], errors="ignore")
        df = df.merge(scenario_df, on="summary_id", how="left")

        if df["case_id"].notna().any():
            df["case_id"] = df["case_id"].astype("Int64")

        state.processed_data = df
        _log_merge(display, "merge_scenario_metadata", df, new_cols)
        return state

    return processor


# ==========================================================================
# 2. Model metadata  (data/final/models.jsonl)
# ==========================================================================


# Mapping from ``models.jsonl`` keys to the ``model_*`` column names expected
# by ``build_fraudbench_indices`` / ``models.py``.
_MODEL_FIELD_MAP: dict[str, str] = {
    "family": "model_family",
    "model_size": "model_size",
    "model_parameters": "model_parameters",
    "release_date": "model_release_date",
    "knowledge_cutoff": "model_knowledge_cutoff",
    "open_source": "model_open_source",
    "reasoning": "model_reasoning",
    "geo": "model_geo",
    "max_reasoning_effort": "model_max_reasoning_effort",
    "max_reasoning_tokens": "model_max_reasoning_tokens",
}


@process
def merge_model_metadata(path: str = DEFAULT_MODELS_PATH):
    """Join model-level metadata onto ``state.processed_data`` on a normalized
    model id (provider/route stripped via :func:`normalize_model_id`).

    Reads ``models.jsonl`` and adds the columns expected by the regression
    backbone:

    * ``model_family``
    * ``model_size`` (Int64; nullable)
    * ``model_parameters`` (Int64; parameter count in billions, nullable)
    * ``model_release_date`` (str ``MM-DD-YYYY``)
    * ``model_knowledge_cutoff`` (str ``MM-DD-YYYY``)
    * ``model_open_source`` (bool)
    * ``model_reasoning`` (bool; reasoning enabled in the model's main-results
      configuration -- used as the row-level fallback when the runtime data
      does not expose ``reasoning_effort`` / ``reasoning_budget`` columns)
    * ``model_geo`` (str)
    * ``model_max_reasoning_effort`` (str; e.g. ``"high"`` for OpenAI/Gemini
      providers that gate reasoning by an effort label, ``None`` otherwise)
    * ``model_max_reasoning_tokens`` (Int64; nullable; e.g. ``32768`` for
      Gemini Pro, used by providers that gate reasoning by a token budget)

    Human-baseline rows (``model == 'human'``) get ``NaN`` across all of
    these fields, matching ``data/README.md``.

    Args:
        path: Path to ``models.jsonl``. Defaults to ``data/final/models.jsonl``.
    """

    def processor(
        state: AnalysisState,
        display: Optional[ModellingDisplay] = None,
    ) -> AnalysisState:
        df = state.processed_data.copy()
        if "model" not in df.columns:
            raise ValueError(
                "merge_model_metadata: 'model' column is required on the data."
            )

        records = _read_jsonl(_resolve_path(path))
        rows = []
        for r in records:
            row: dict[str, Any] = {"_model_id": r.get("model")}
            for src, dst in _MODEL_FIELD_MAP.items():
                row[dst] = r.get(src)
            rows.append(row)
        models_df = pd.DataFrame(rows)
        if models_df["_model_id"].duplicated().any():
            n_dup = int(models_df["_model_id"].duplicated().sum())
            raise ValueError(
                f"merge_model_metadata: {path} contains {n_dup} duplicate "
                "model entries; expected one record per bare model id."
            )

        df["_model_id"] = df["model"].astype(str).map(normalize_model_id)
        new_cols = list(_MODEL_FIELD_MAP.values())
        df = df.drop(columns=[c for c in new_cols if c in df.columns], errors="ignore")
        df = df.merge(models_df, on="_model_id", how="left")
        df = df.drop(columns=["_model_id"])

        if "model_size" in df.columns:
            df["model_size"] = pd.to_numeric(df["model_size"], errors="coerce").astype("Int64")
        if "model_parameters" in df.columns:
            df["model_parameters"] = pd.to_numeric(df["model_parameters"], errors="coerce").astype("Int64")
        if "model_max_reasoning_tokens" in df.columns:
            df["model_max_reasoning_tokens"] = pd.to_numeric(
                df["model_max_reasoning_tokens"], errors="coerce",
            ).astype("Int64")
        for c in ("model_open_source", "model_reasoning"):
            if c in df.columns:
                df[c] = df[c].astype("boolean")

        state.processed_data = df
        _log_merge(display, "merge_model_metadata", df, new_cols)
        return state

    return processor


# ==========================================================================
# 3. Human-baseline respondent metadata
#    (data/final/human_baseline/human_baseline_public.jsonl)
# ==========================================================================


# Mapping from columns in the human-baseline JSONL to ``human_*`` column
# names expected by ``build_fraudbench_indices``.
_HUMAN_FIELD_MAP: dict[str, str] = {
    "age_bucket": "human_age",
    "Prolific_Sex": "human_sex",
    "Prolific_Ethnicity simplified": "human_ethnicity_simplified",
    "AI_use": "human_ai_use",
}


def _collapse_experiments(value: Any) -> str:
    """Collapse a row's ``experiments`` list to a single experiment string.

    Per ``data/README.md`` the human-baseline join uses ``experiment`` as a
    third key alongside ``(respondent_id, summary_id)``. The obs-data side
    stores experiments as a list-like column:

        ``[]``                       -> ``"control"``
        ``["peer_pressure_low"]``    -> ``"peer_pressure_low"``
        ``["a", "b"]`` (rare)        -> ``"a+b"`` (sorted, joined)
    """
    if value is None:
        return "control"
    try:
        items = list(value)
    except TypeError:
        return "control"
    items = [str(x) for x in items if x is not None and str(x) != ""]
    if not items:
        return "control"
    if len(items) == 1:
        return items[0]
    return "+".join(sorted(items))


@process
def merge_human_baseline_metadata(
    path: str = DEFAULT_HUMAN_BASELINE_PATH,
):
    """Join human-respondent demographics on
    ``(human_respondent_id, summary_id, experiment)``.

    Reads ``human_baseline_public.jsonl`` and adds:

    * ``human_age`` (str; the released age bucket ``18-24`` … ``65+`` or
      ``missing``; ``build_fraudbench_indices`` also accepts raw ages)
    * ``human_sex`` (str)
    * ``human_ethnicity_simplified`` (str)
    * ``human_ai_use`` (str)

    Only rows whose ``human_respondent_id`` matches a respondent in the
    JSONL receive non-null values; AI rows (which have an empty
    ``human_respondent_id``) keep ``NaN`` across all four columns. The
    third join key is computed from the ``experiments`` list via
    :func:`_collapse_experiments`.

    Args:
        path: Path to the human-baseline JSONL. Defaults to
            ``data/final/human_baseline/human_baseline_public.jsonl``.
    """

    def processor(
        state: AnalysisState,
        display: Optional[ModellingDisplay] = None,
    ) -> AnalysisState:
        df = state.processed_data.copy()
        for required in ("human_respondent_id", "summary_id", "experiments"):
            if required not in df.columns:
                raise ValueError(
                    f"merge_human_baseline_metadata: '{required}' is required on the "
                    "incoming data but is missing."
                )

        records = _read_jsonl(_resolve_path(path))
        wanted_src = list(_HUMAN_FIELD_MAP.keys())
        rows = []
        for r in records:
            row: dict[str, Any] = {
                "human_respondent_id": r.get("respondent_id"),
                "summary_id": r.get("summary_id"),
                "_experiment_key": r.get("experiment"),
            }
            for src in wanted_src:
                row[_HUMAN_FIELD_MAP[src]] = r.get(src)
            rows.append(row)
        hb_df = pd.DataFrame(rows)

        keys = ["human_respondent_id", "summary_id", "_experiment_key"]
        if hb_df.duplicated(subset=keys).any():
            # Same respondent answering the same vignette under the same
            # experimental condition multiple times is not expected; surface
            # duplicates as a hard error so we don't silently drop joins.
            n_dup = int(hb_df.duplicated(subset=keys).sum())
            raise ValueError(
                f"merge_human_baseline_metadata: {path} has {n_dup} duplicate "
                f"({'+'.join(keys)}) tuples."
            )

        df["_experiment_key"] = df["experiments"].apply(_collapse_experiments)
        df["human_respondent_id"] = df["human_respondent_id"].fillna("").astype(str)

        new_cols = list(_HUMAN_FIELD_MAP.values())
        df = df.drop(columns=[c for c in new_cols if c in df.columns], errors="ignore")
        df = df.merge(hb_df, on=keys, how="left")
        df = df.drop(columns=["_experiment_key"])

        state.processed_data = df
        _log_merge(display, "merge_human_baseline_metadata", df, new_cols)

        # If at least one human-baseline row exists but none picked up demo
        # fields, the join key almost certainly drifted (most commonly: the
        # ``respondent_id`` hashes in the JSONL and the parquet were produced
        # under different ``.hash_key`` files). Surface that loudly rather
        # than letting the model silently fit on an all-missing column.
        n_human_rows = int((df["model"] == "human").sum())
        n_matched = int(df.loc[df["model"] == "human", new_cols[0]].notna().sum())
        if n_human_rows > 0 and n_matched == 0 and display is not None:
            display.logger.warning(
                "merge_human_baseline_metadata: 0 of "
                f"{n_human_rows} human-baseline rows matched a respondent in "
                f"{path}. Check that human_respondent_id values in the parquet "
                "and respondent_id values in the JSONL were hashed with the "
                "same .hash_key (FRAUDBENCH_HASH_KEY)."
            )
        return state

    return processor


__all__ = [
    "merge_scenario_metadata",
    "merge_model_metadata",
    "merge_human_baseline_metadata",
]
