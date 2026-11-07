"""LaTeX table of LLM-judge accuracy and inter-rater reliability.

Backs ``analysis/tables/table_judge_irr.py``.

Renders a single LaTeX table summarising the three-judge ensemble used to
score every kept response in ``data/final/obs_data_per_score.parquet`` on
three metrics (``fraud``, ``refusal``, ``options``). The table has four panels:

* **(a) Per-judge accuracy vs. majority vote.** For each judge model and
  metric, the fraction of rows where that judge's vote matches the
  ensemble majority (``score_fraud`` / ``score_refusal`` /
  ``score_options``). Restricted to rows where both the judge produced
  a parseable vote and the majority is defined. Note these are
  self-inclusive: each judge is one of the three voters in its own
  majority.
* **(b) Majority/human Cohen's $\\kappa$.** The LLM majority vote is compared
  with post-stratified human consensus on the validation subset.
* **(c) LLM Fleiss' $\\kappa$.** Agreement among all LLM judges.
* **(d) Human Fleiss' $\\kappa$.** Post-stratified agreement among human
  annotators.

Human-baseline responses (``setting == "human_baseline"``) are excluded from
every panel: the table describes the LLM jury on the closed, open, and
agentic LLM settings only.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from ..agreement import item_weighted_cohen_kappa, item_weighted_fleiss_kappa
from ..common import load_data_from_source
from ..judges import (
    GRADER_RAN_COLUMN,
    METRIC_DISPLAY,
    METRICS,
    OBS_COLUMN,
    SAMPLE_ID_COLUMN,
    SCORER_COLUMN,
    collapse_per_score_rows,
    discover_judges,
    expand_votes,
    judge_display_name,
    require_individual_vote_columns,
    votes_columns,
)


MAJORITY_COL: dict[str, str] = {m: f"score_{m}" for m in METRICS}
HUMAN_SCORE_FIELD: dict[str, str] = {m: f"{m}_score" for m in METRICS}
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_HUMAN_ANNOTATIONS_PATH = (
    REPO_ROOT / "data" / "annotations" / "llm_judge" / "human_annotations.jsonl"
)
# ``None`` means the validation sample is every item with at least one human
# rating in the annotations file; pass a ``Sample ID`` CSV to override.
DEFAULT_VALIDATED_SAMPLES_PATH: Path | None = None
DEFAULT_OBS_DATA_PATH = REPO_ROOT / "data" / "final" / "obs_data.parquet"
DEFAULT_TIE_BREAK_SEED = 1487

HUMAN_BASELINE_SETTING = "human_baseline"
# Settings that form the human-validation target population. Human-baseline
# responses are excluded from this table entirely.
VALIDATION_SETTINGS: tuple[str, ...] = (
    "closed",
    "open",
    "agentic",
)
STRATUM_COLUMNS: tuple[str, ...] = (
    "setting",
    "grader_counts_fraud",
    "grader_counts_refusal",
)
ANNOTATION_ID_FIELD = "sample_id"
ANNOTATOR_FIELD = "annotator_id"
# Column of the optional ``--validated-samples`` CSV.
VALIDATED_SAMPLES_ID_COLUMN = "Sample ID"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_any(path: str) -> pd.DataFrame:
    return load_data_from_source(
        path,
        needed_columns=[
            SAMPLE_ID_COLUMN,
            *votes_columns(METRICS),
            SCORER_COLUMN,
            GRADER_RAN_COLUMN,
            *(OBS_COLUMN[metric] for metric in METRICS),
            *MAJORITY_COL.values(),
            "setting",
        ],
    )


def load_data(input_path: str) -> pd.DataFrame:
    """Load the kept observations parquet and validate the required columns."""
    df = _read_any(input_path)
    require_individual_vote_columns(df, METRICS, source=input_path)
    missing = [c for c in MAJORITY_COL.values() if c not in df.columns]
    if missing:
        raise ValueError(
            f"{input_path} is missing required columns: {missing}. Expected the "
            "majority-vote score_<metric> columns produced by analysis/process.py."
        )
    df = collapse_per_score_rows(df, METRICS)
    print(f"Loaded {len(df)} rows from {input_path}")
    return df


@dataclass(frozen=True)
class ValidationDesign:
    """Post-stratification data for the human-validation sample."""

    items: pd.DataFrame
    population_sizes: dict[tuple[str, int, int], int]
    sample_sizes: dict[tuple[str, int, int], int]


def _calibrated_weights(
    items: pd.DataFrame,
    population_sizes: dict[tuple[str, int, int], int],
) -> pd.Series:
    """Return ``N_h/n_h`` weights for the strata represented in ``items``."""
    sample_sizes: Counter[tuple[str, int, int]] = Counter(items["stratum"])
    unsampled_strata = sorted(set(population_sizes) - set(sample_sizes))
    if unsampled_strata:
        raise ValueError(
            "The analyzed items do not represent every non-empty population "
            f"stratum; unsampled strata: {unsampled_strata}."
        )
    weights = pd.Series(
        [
            population_sizes[stratum] / sample_sizes[stratum]
            for stratum in items["stratum"]
        ],
        index=items.index,
        dtype=float,
        name="item_weight",
    )
    population_total = int(sum(population_sizes.values()))
    if not np.isclose(weights.sum(), population_total):
        raise RuntimeError(
            "Post-stratification weights do not sum to the represented "
            "obs_data population size."
        )
    return weights


def load_validated_sample_ids(path: str | Path) -> list[str]:
    """Load a unique, non-empty validation-sample ID list."""
    path = Path(path)
    raw = pd.read_csv(path, dtype=str)
    if VALIDATED_SAMPLES_ID_COLUMN not in raw.columns:
        raise ValueError(
            f"{path} is missing required column {VALIDATED_SAMPLES_ID_COLUMN!r}."
        )
    ids = raw[VALIDATED_SAMPLES_ID_COLUMN]
    missing = ids.isna() | ids.fillna("").str.strip().eq("")
    if missing.any():
        raise ValueError(f"{path} contains {int(missing.sum())} blank sample ID(s).")
    ids = ids.str.strip()
    duplicated = ids[ids.duplicated(keep=False)]
    if not duplicated.empty:
        examples = duplicated.drop_duplicates().head(5).tolist()
        raise ValueError(
            f"{path} contains duplicate sample IDs; examples: {examples}."
        )
    if ids.empty:
        raise ValueError(f"{path} contains no validation sample IDs.")
    return ids.tolist()


def resolve_validation_sample_ids(
    human_long: pd.DataFrame,
    validated_samples_path: str | Path | None = None,
) -> list[str]:
    """Return the complete validation-sample ID list.

    With ``validated_samples_path`` the list is read from that CSV and must
    coincide with the set of human-annotated items. Without it, the
    validation sample is every item that received at least one human rating,
    in sorted order.
    """
    annotated_ids = set(human_long["sample_id"])
    if validated_samples_path is None:
        if not annotated_ids:
            raise ValueError("No human-annotated sample IDs were found.")
        return sorted(annotated_ids)

    validated_samples_path = Path(validated_samples_path)
    validated_ids = load_validated_sample_ids(validated_samples_path)
    unknown_ids = sorted(annotated_ids - set(validated_ids))
    if unknown_ids:
        raise ValueError(
            f"The human annotations file contains {len(unknown_ids)} sample ID(s) absent "
            f"from {validated_samples_path}; examples: {unknown_ids[:5]}."
        )
    missing_ids = [
        sample_id for sample_id in validated_ids if sample_id not in annotated_ids
    ]
    if missing_ids:
        raise ValueError(
            f"The human annotations file is missing {len(missing_ids)} validation "
            f"sample ID(s); examples: {missing_ids[:5]}."
        )
    return validated_ids


def _as_validation_sample_ids(
    validated_samples: Sequence[str] | str | Path,
) -> list[str]:
    """Accept either a ``Sample ID`` CSV path or an in-memory ID sequence."""
    if isinstance(validated_samples, (str, Path)):
        return load_validated_sample_ids(validated_samples)
    ids = [str(sample_id).strip() for sample_id in validated_samples]
    if not ids:
        raise ValueError("The validation sample ID list is empty.")
    if any(sample_id == "" for sample_id in ids):
        raise ValueError("The validation sample ID list contains blank IDs.")
    duplicates = sorted(
        sample_id for sample_id, count in Counter(ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(
            f"The validation sample ID list contains duplicates; examples: "
            f"{duplicates[:5]}."
        )
    return ids


def _validation_population_batches(path: str | Path) -> Iterator[pd.DataFrame]:
    """Yield small obs-data batches containing the sampling-design columns."""
    path = Path(path)
    required = [
        SAMPLE_ID_COLUMN,
        "setting",
        "run_category",
        "completion_extended",
        "grader_counts_fraud",
        "grader_counts_refusal",
    ]
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        missing = [column for column in required if column not in parquet.schema.names]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}.")
        for batch in parquet.iter_batches(batch_size=4096, columns=required):
            yield batch.to_pandas()
        return

    frame = load_data_from_source(str(path), needed_columns=required)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}.")
    yield frame[required]


def _eligible_validation_population(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the same population filters as the validation sampler."""
    mask = frame["run_category"].eq("main")
    mask &= frame["setting"].isin(VALIDATION_SETTINGS)
    is_agentic = frame["setting"].eq("agentic")
    has_extended_completion = (
        frame["completion_extended"].fillna("").astype(str).str.strip().ne("")
    )
    mask &= ~is_agentic | has_extended_completion

    population = frame.loc[
        mask,
        [
            SAMPLE_ID_COLUMN,
            "setting",
            "grader_counts_fraud",
            "grader_counts_refusal",
        ],
    ].copy()
    if population[SAMPLE_ID_COLUMN].isna().any():
        raise ValueError("Eligible obs_data population contains null sample IDs.")
    population[SAMPLE_ID_COLUMN] = population[SAMPLE_ID_COLUMN].astype(str).str.strip()
    if population[SAMPLE_ID_COLUMN].eq("").any():
        raise ValueError("Eligible obs_data population contains blank sample IDs.")

    for column in ("grader_counts_fraud", "grader_counts_refusal"):
        numeric = pd.to_numeric(population[column], errors="coerce")
        invalid = numeric.isna() | ~numeric.isin(range(4))
        if invalid.any():
            examples = (
                population.loc[invalid, SAMPLE_ID_COLUMN].astype(str).head(5).tolist()
            )
            raise ValueError(
                f"Eligible obs_data rows contain invalid {column} values; "
                f"sample IDs: {examples}."
            )
        population[column] = numeric.astype(int)
    return population


def build_validation_design(
    obs_data_path: str | Path,
    validated_samples: Sequence[str] | str | Path,
) -> ValidationDesign:
    """Build setting-by-jury-cell post-stratification weights.

    Every validation item receives ``N_h / n_h``, where ``N_h`` is its
    eligible obs-data population count and ``n_h`` is the count among all
    validation sample IDs for the same
    ``(setting, grader_counts_fraud, grader_counts_refusal)`` stratum.

    ``validated_samples`` is either a path to a ``Sample ID`` CSV or the ID
    list itself (see :func:`resolve_validation_sample_ids`).
    """
    validated_ids = _as_validation_sample_ids(validated_samples)
    validated_set = set(validated_ids)
    population_sizes: Counter[tuple[str, int, int]] = Counter()
    matched_frames: list[pd.DataFrame] = []
    seen_population_ids: set[str] = set()

    for batch in _validation_population_batches(obs_data_path):
        population = _eligible_validation_population(batch)
        duplicate_ids = seen_population_ids.intersection(population[SAMPLE_ID_COLUMN])
        if duplicate_ids:
            raise ValueError(
                "Eligible obs_data population contains duplicate sample IDs; "
                f"examples: {sorted(duplicate_ids)[:5]}."
            )
        seen_population_ids.update(population[SAMPLE_ID_COLUMN])
        strata = list(
            population[list(STRATUM_COLUMNS)].itertuples(index=False, name=None)
        )
        population_sizes.update(strata)
        selected = population[population[SAMPLE_ID_COLUMN].isin(validated_set)].copy()
        if not selected.empty:
            matched_frames.append(selected)

    if not matched_frames:
        raise ValueError("No validation sample IDs were found in the obs_data population.")
    matched = pd.concat(matched_frames, ignore_index=True)
    matched_ids = set(matched[SAMPLE_ID_COLUMN])
    missing_ids = [sample_id for sample_id in validated_ids if sample_id not in matched_ids]
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} validation sample ID(s) are absent from the eligible "
            f"obs_data population; examples: {missing_ids[:5]}."
        )
    if matched[SAMPLE_ID_COLUMN].duplicated().any():
        duplicates = (
            matched.loc[
                matched[SAMPLE_ID_COLUMN].duplicated(keep=False), SAMPLE_ID_COLUMN
            ]
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        raise ValueError(
            "Validation sample IDs match multiple eligible obs_data rows; "
            f"examples: {duplicates}."
        )

    items = matched.set_index(SAMPLE_ID_COLUMN).loc[validated_ids].reset_index()
    items["stratum"] = list(
        items[list(STRATUM_COLUMNS)].itertuples(index=False, name=None)
    )
    sample_sizes: Counter[tuple[str, int, int]] = Counter(items["stratum"])
    unsampled_strata = sorted(set(population_sizes) - set(sample_sizes))
    if unsampled_strata:
        raise ValueError(
            "The validation sample does not represent every non-empty population "
            f"stratum; unsampled strata: {unsampled_strata}."
        )
    items["item_weight"] = _calibrated_weights(items, dict(population_sizes))
    return ValidationDesign(
        items=items,
        population_sizes=dict(population_sizes),
        sample_sizes=dict(sample_sizes),
    )


# ---------------------------------------------------------------------------
# Per-judge expansion
# ---------------------------------------------------------------------------

def _expand_metric(
    df: pd.DataFrame, metric: str, judges: tuple[str, ...]
) -> pd.DataFrame:
    """Build a per-row DataFrame with one nullable bool col per judge.

    Returns ``[majority] + [judge_<j> for j in judges]`` indexed by
    ``df.index``. Nulls in the judge dicts (parse failures) propagate
    as ``pd.NA``.
    """
    out = expand_votes(df, metric, judges, prefix="judge_")
    out.insert(0, "majority", df[MAJORITY_COL[metric]])
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class AccuracyCell:
    """One per-judge cell: accuracy vs. majority for a single metric."""

    n: int = 0
    accuracy: float = float("nan")


@dataclass
class KappaCell:
    """One agreement statistic for a single metric."""

    n: int = 0
    kappa: float = float("nan")
    observed_agreement: float = float("nan")
    positive_prevalence: float | tuple[float, float] = float("nan")


def compute_accuracy(
    df: pd.DataFrame, judges: tuple[str, ...]
) -> dict[str, dict[str, AccuracyCell]]:
    """Per-judge accuracy vs. majority vote, for every (judge, metric)."""
    out: dict[str, dict[str, AccuracyCell]] = {
        j: {m: AccuracyCell() for m in METRICS} for j in judges
    }
    for metric in METRICS:
        expanded = _expand_metric(df, metric, judges)
        majority = expanded["majority"]
        for judge in judges:
            j_col = expanded[f"judge_{judge}"]
            mask = j_col.notna() & majority.notna()
            n = int(mask.sum())
            if n == 0:
                continue
            j_arr = j_col[mask].astype(bool).to_numpy()
            m_arr = majority[mask].astype(int).astype(bool).to_numpy()
            out[judge][metric] = AccuracyCell(
                n=n, accuracy=float(np.mean(j_arr == m_arr))
            )
    return out


def _parse_bool(value: object) -> bool | None:
    """Return a boolean for supported annotation values, else ``None``."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        if value in (0, 1):
            return bool(value)
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    return None


def _fleiss_statistics(votes: np.ndarray) -> tuple[float, float, float]:
    """Return Fleiss' kappa, observed agreement, and positive prevalence."""
    if votes.ndim != 2 or votes.shape[0] == 0 or votes.shape[1] < 2:
        return float("nan"), float("nan"), float("nan")
    result = item_weighted_fleiss_kappa(
        votes,
        labels=(False, True),
        se_method=None,
    )
    return (
        result.estimate,
        result.observed_agreement,
        float(result.components["category_proportions"][1]),
    )


def compute_llm_fleiss_kappa(
    df: pd.DataFrame, judges: tuple[str, ...]
) -> dict[str, KappaCell]:
    """Fleiss' kappa among all LLM judges for each metric."""
    out = {metric: KappaCell() for metric in METRICS}
    for metric in METRICS:
        expanded = _expand_metric(df, metric, judges)
        judge_cols = [f"judge_{judge}" for judge in judges]
        complete = expanded[judge_cols].dropna()
        if complete.empty:
            continue
        votes = complete.astype(bool).to_numpy()
        kappa, agreement, prevalence = _fleiss_statistics(votes)
        out[metric] = KappaCell(
            n=len(complete),
            kappa=kappa,
            observed_agreement=agreement,
            positive_prevalence=prevalence,
        )
    return out


def load_human_annotations(
    annotations_path: str | Path = DEFAULT_HUMAN_ANNOTATIONS_PATH,
    validated_samples_path: str | Path | None = DEFAULT_VALIDATED_SAMPLES_PATH,
) -> pd.DataFrame:
    """Load and strictly validate human ratings in long form.

    ``annotations_path`` is a JSONL file with one human rating per line.
    Required fields: ``sample_id``, ``annotator_id``, ``fraud_score``,
    ``refusal_score``, ``options_score``. Score values must parse as
    booleans (``true``/``false``/``1``/``0``). An item rated by several
    annotators receives several ratings, which the agreement statistics
    treat as independent human ratings of that item; each annotator may
    rate a given item at most once.

    Returns one row per rating with columns ``sample_id``, ``annotator_id``
    and ``<metric>_human`` for each metric.

    When ``validated_samples_path`` is given, the annotated IDs must
    coincide with its ID list.
    """
    annotations_path = Path(annotations_path)
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Human annotations file not found: {annotations_path}.")
    with annotations_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"{annotations_path} contains no human ratings.")
    raw = pd.DataFrame.from_records(records)

    score_fields = list(HUMAN_SCORE_FIELD.values())
    required = [ANNOTATION_ID_FIELD, ANNOTATOR_FIELD, *score_fields]
    missing = [field for field in required if field not in raw.columns]
    if missing:
        raise ValueError(f"{annotations_path} is missing required fields: {missing}.")

    frame = pd.DataFrame(index=raw.index)
    for field in (ANNOTATION_ID_FIELD, ANNOTATOR_FIELD):
        values = raw[field]
        blank = values.isna() | values.astype(str).str.strip().eq("")
        if blank.any():
            raise ValueError(
                f"{annotations_path} contains {int(blank.sum())} rating(s) with a "
                f"blank {field!r}."
            )
        frame[field] = values.astype(str).str.strip()
    for metric in METRICS:
        source = HUMAN_SCORE_FIELD[metric]
        parsed = raw[source].map(_parse_bool)
        invalid = parsed.isna()
        if invalid.any():
            examples = raw.loc[invalid, [ANNOTATION_ID_FIELD, source]].head(5)
            raise ValueError(
                f"{annotations_path} contains {int(invalid.sum())} missing or "
                f"invalid {source} value(s); examples: "
                f"{examples.to_dict(orient='records')}."
            )
        frame[f"{metric}_human"] = parsed.astype(bool)

    duplicate = frame.duplicated([ANNOTATOR_FIELD, ANNOTATION_ID_FIELD], keep=False)
    if duplicate.any():
        examples = (
            frame.loc[duplicate, [ANNOTATOR_FIELD, ANNOTATION_ID_FIELD]]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"{annotations_path} rates the same item more than once for one "
            f"annotator; examples: {examples}."
        )

    human_long = frame.reset_index(drop=True)
    if validated_samples_path is not None:
        resolve_validation_sample_ids(human_long, validated_samples_path)
    return human_long


def _ratings_matrix(human_long: pd.DataFrame, vote_col: str) -> pd.DataFrame:
    """Item-by-rating matrix with one column per rating slot.

    Fleiss' kappa depends only on how many ratings of each category an item
    received, not on which rater produced them, so ratings are laid out by
    position within the item rather than by annotator label. Items with
    fewer ratings than the widest item are padded with ``NaN``.
    """
    ordered = human_long.sort_values(["sample_id", "annotator_id"], kind="stable")
    slots = ordered.groupby("sample_id", sort=False).cumcount()
    matrix = ordered.assign(_slot=slots).pivot(
        index="sample_id", columns="_slot", values=vote_col
    )
    matrix.columns = [f"rating_{slot + 1}" for slot in matrix.columns]
    return matrix


def _human_consensus(
    human_long: pd.DataFrame,
    metric: str,
    rng: np.random.Generator,
) -> pd.Series:
    """Return one human label per item, resolving even splits with ``rng``."""
    vote_col = f"{metric}_human"
    counts = (
        human_long.sort_values(["sample_id", "annotator_id"])
        .groupby("sample_id", sort=True)[vote_col]
        .agg(n_votes="size", n_true="sum")
    )
    consensus = (2 * counts["n_true"] > counts["n_votes"]).astype(bool)
    ties = 2 * counts["n_true"] == counts["n_votes"]
    consensus.loc[ties] = rng.integers(0, 2, size=int(ties.sum())).astype(bool)
    consensus.name = "human_consensus"
    return consensus


def compute_human_agreement(
    df: pd.DataFrame,
    human_long: pd.DataFrame,
    validation_design: ValidationDesign,
    *,
    seed: int = DEFAULT_TIE_BREAK_SEED,
) -> tuple[dict[str, KappaCell], dict[str, KappaCell]]:
    """Compute post-stratified LLM/human Cohen and human Fleiss kappas."""
    if "sample_id" not in df:
        raise ValueError("obs_data must contain 'sample_id' to join human annotations.")
    if df["sample_id"].duplicated().any():
        raise ValueError("obs_data must have unique sample_id values for human validation.")

    cohen = {metric: KappaCell() for metric in METRICS}
    fleiss = {metric: KappaCell() for metric in METRICS}
    design = validation_design.items.set_index("sample_id")
    if design.index.duplicated().any():
        raise ValueError("Validation design contains duplicate sample IDs.")
    if set(human_long["sample_id"]) != set(design.index):
        raise ValueError(
            "Human annotations and validation design must contain identical sample IDs."
        )
    llm_by_sample = df.set_index("sample_id")
    rng = np.random.default_rng(seed)

    for metric in METRICS:
        vote_col = f"{metric}_human"
        consensus = _human_consensus(human_long, metric, rng)
        comparison = (
            design[["item_weight", "stratum"]]
            .join(consensus, how="left")
            .join(llm_by_sample[[MAJORITY_COL[metric]]], how="left")
        )
        raw_llm = comparison[MAJORITY_COL[metric]]
        parsed_llm = raw_llm.map(_parse_bool)
        invalid_llm = raw_llm.notna() & parsed_llm.isna()
        if invalid_llm.any():
            examples = raw_llm[invalid_llm].head(5).to_dict()
            raise ValueError(
                f"{MAJORITY_COL[metric]} contains invalid non-binary values: {examples}."
            )
        comparison["llm_majority"] = parsed_llm
        comparison = comparison.dropna(subset=["human_consensus", "llm_majority"])
        if not comparison.empty:
            comparison["analysis_weight"] = _calibrated_weights(
                comparison,
                validation_design.population_sizes,
            )
            result = item_weighted_cohen_kappa(
                comparison["llm_majority"].astype(bool),
                comparison["human_consensus"].astype(bool),
                sample_weight=comparison["analysis_weight"],
                labels=(False, True),
                strata=comparison["stratum"],
                se_method=None,
            )
            cohen[metric] = KappaCell(
                n=result.n_items,
                kappa=result.estimate,
                observed_agreement=result.observed_agreement,
                positive_prevalence=(
                    float(result.components["marginal_a"][1]),
                    float(result.components["marginal_b"][1]),
                ),
            )

        ratings = _ratings_matrix(human_long, vote_col)
        ratings = ratings.loc[ratings.notna().sum(axis=1) >= 2]
        if not ratings.empty:
            retained_design = design.loc[ratings.index].copy()
            weights = _calibrated_weights(
                retained_design,
                validation_design.population_sizes,
            )
            strata = retained_design["stratum"]
            result = item_weighted_fleiss_kappa(
                ratings,
                sample_weight=weights,
                labels=(False, True),
                strata=strata,
                nan_policy="omit",
                se_method=None,
            )
            fleiss[metric] = KappaCell(
                n=result.n_items,
                kappa=result.estimate,
                observed_agreement=result.observed_agreement,
                positive_prevalence=float(
                    result.components["category_proportions"][1]
                ),
            )
    return cohen, fleiss


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------

T1 = "    "
T2 = "        "


def _format_pct(p: float, *, no_percent: bool) -> str:
    """Format an accuracy proportion as a percent (or proportion)."""
    if not np.isfinite(p):
        return "---"
    if no_percent:
        return f"{p:.3f}"
    return f"{100.0 * p:.1f}\\%"


def _format_kappa(k: float) -> str:
    if not np.isfinite(k):
        return "---"
    return f"{k:.3f}"


def _format_prevalence(prevalence: float | tuple[float, float]) -> str:
    """Format a pooled positive-label rate or the two compared rates."""
    if isinstance(prevalence, tuple):
        if not all(np.isfinite(rate) for rate in prevalence):
            return "---"
        return "/".join(f"{100.0 * rate:.1f}\\%" for rate in prevalence)
    if not np.isfinite(prevalence):
        return "---"
    return f"{100.0 * prevalence:.1f}\\%"


def _stat_row(label: str, cells: dict[str, KappaCell], attribute: str) -> list[str]:
    """Build one LaTeX agreement-statistic row across metrics."""
    formatter = _format_prevalence if attribute == "positive_prevalence" else _format_pct
    values = []
    for metric in METRICS:
        value = getattr(cells[metric], attribute)
        values.append(
            formatter(value, no_percent=False)
            if attribute == "observed_agreement"
            else formatter(value)
        )
    return [label, *values]


def _n_range(values: Iterable[int]) -> str:
    vs = [int(v) for v in values if int(v) > 0]
    if not vs:
        return "0"
    lo, hi = min(vs), max(vs)
    if lo == hi:
        return f"{lo:,}"
    return f"{lo:,}--{hi:,}"


def to_latex(
    accuracy: dict[str, dict[str, AccuracyCell]],
    majority_human_kappa: dict[str, KappaCell],
    llm_fleiss_kappa: dict[str, KappaCell],
    human_fleiss_kappa: dict[str, KappaCell],
    *,
    judges: tuple[str, ...],
    caption: str,
    label: str,
    no_percent: bool = False,
) -> str:
    """Render the combined accuracy + agreement-statistics table.

    Single ``tabular`` with an ``l`` column for the row label and one
    ``c`` column per metric. The two panels are separated by an italic
    sub-header row spanning all columns plus a ``\\midrule``.
    """
    n_metrics = len(METRICS)
    col_spec = "l" + ("c" * n_metrics)

    lines: list[str] = []
    lines.append("\\begin{table}[!htbp]")
    lines.append(f"{T1}% Required packages: \\usepackage{{booktabs}}")
    lines.append(f"{T1}\\centering")
    lines.append(f"{T1}\\small")
    lines.append(f"{T1}\\setlength{{\\tabcolsep}}{{6pt}}")
    lines.append(f"{T1}\\begin{{tabular}}{{{col_spec}}}")
    lines.append(f"{T2}\\toprule")

    header = [""] + [
        f"\\textbf{{{METRIC_DISPLAY[m]}}}" for m in METRICS
    ]
    lines.append(f"{T2}" + " & ".join(header) + " \\\\")
    lines.append(f"{T2}\\midrule")

    panel_a_header = (
        f"\\multicolumn{{{n_metrics + 1}}}{{l}}"
        f"{{\\textit{{(a) Per-judge accuracy vs. LLM jury majority vote}}}}"
    )
    lines.append(f"{T2}{panel_a_header} \\\\")
    for judge in judges:
        row = [judge_display_name(judge)]
        for metric in METRICS:
            cell = accuracy[judge][metric]
            row.append(_format_pct(cell.accuracy, no_percent=no_percent))
        lines.append(f"{T2}" + " & ".join(row) + " \\\\")

    lines.append(f"{T2}\\midrule")

    panel_b_header = (
        f"\\multicolumn{{{n_metrics + 1}}}{{l}}"
        f"{{\\textit{{(b) LLM majority vote vs. human consensus Cohen's $\\kappa$}}}}"
    )
    lines.append(f"{T2}{panel_b_header} \\\\")
    lines.append(
        f"{T2}" + " & ".join(
            ["Cohen's $\\kappa$"]
            + [_format_kappa(majority_human_kappa[m].kappa) for m in METRICS]
        ) + " \\\\"
    )
    for label_text, attribute in (
        ("Observed agreement", "observed_agreement"),
        ("Positive prevalence (LLM/human)", "positive_prevalence"),
    ):
        lines.append(f"{T2}" + " & ".join(
            _stat_row(label_text, majority_human_kappa, attribute)
        ) + " \\\\")

    lines.append(f"{T2}\\midrule")
    panel_c_header = (
        f"\\multicolumn{{{n_metrics + 1}}}{{l}}"
        f"{{\\textit{{(c) Fleiss' $\\kappa$ among all LLM judges}}}}"
    )
    lines.append(f"{T2}{panel_c_header} \\\\")
    lines.append(
        f"{T2}" + " & ".join(
            ["Fleiss' $\\kappa$"]
            + [_format_kappa(llm_fleiss_kappa[m].kappa) for m in METRICS]
        ) + " \\\\"
    )
    for label_text, attribute in (
        ("Observed agreement", "observed_agreement"),
        ("Positive prevalence", "positive_prevalence"),
    ):
        lines.append(f"{T2}" + " & ".join(
            _stat_row(label_text, llm_fleiss_kappa, attribute)
        ) + " \\\\")

    lines.append(f"{T2}\\midrule")
    panel_d_header = (
        f"\\multicolumn{{{n_metrics + 1}}}{{l}}"
        f"{{\\textit{{(d) Fleiss' $\\kappa$ among human annotators}}}}"
    )
    lines.append(f"{T2}{panel_d_header} \\\\")
    lines.append(
        f"{T2}" + " & ".join(
            ["Fleiss' $\\kappa$"]
            + [_format_kappa(human_fleiss_kappa[m].kappa) for m in METRICS]
        ) + " \\\\"
    )
    for label_text, attribute in (
        ("Observed agreement", "observed_agreement"),
        ("Positive prevalence", "positive_prevalence"),
    ):
        lines.append(f"{T2}" + " & ".join(
            _stat_row(label_text, human_fleiss_kappa, attribute)
        ) + " \\\\")

    lines.append(f"{T2}\\bottomrule")
    lines.append(f"{T1}\\end{{tabular}}")
    lines.append(f"{T1}\\caption{{{caption}}}")
    lines.append(f"{T1}\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Caption
# ---------------------------------------------------------------------------

def _caption(
    accuracy: dict[str, dict[str, AccuracyCell]],
    majority_human_kappa: dict[str, KappaCell],
    llm_fleiss_kappa: dict[str, KappaCell],
    human_fleiss_kappa: dict[str, KappaCell],
    judges: tuple[str, ...],
    validation_n: int,
    seed: int,
) -> str:
    """Caption text including the supporting sample-size range."""
    majority_human_n = _n_range(
        majority_human_kappa[metric].n for metric in METRICS
    )
    human_fleiss_n = _n_range(human_fleiss_kappa[metric].n for metric in METRICS)
    human_fleiss_omitted = _n_range(
        validation_n - human_fleiss_kappa[metric].n for metric in METRICS
    )
    return (
        "Accuracy and reliability of the three-member LLM-as-a-judge jury used in evaluation, as compared to human annotations. "
        "We report various agreement metrics, as well as observed agreement (fraction of matching labels) and "
        "rate of positive prevalence (rate of positive labels for each score). "
        "Generations with judge errors are excluded from analysis."
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_summary(
    accuracy: dict[str, dict[str, AccuracyCell]],
    majority_human_kappa: dict[str, KappaCell],
    llm_fleiss_kappa: dict[str, KappaCell],
    human_fleiss_kappa: dict[str, KappaCell],
    judges: tuple[str, ...],
) -> None:
    print("\n" + "=" * 60)
    print("PER-JUDGE ACCURACY VS. MAJORITY VOTE")
    print("=" * 60)
    print(
        "{:<28s} {:>8s} {:>10s} {:>10s} {:>10s}".format(
            "judge", "metric", "n", "accuracy", ""
        )
    )
    for judge in judges:
        for metric in METRICS:
            c = accuracy[judge][metric]
            print(
                "{:<28s} {:>8s} {:>10d} {:>10.4f}".format(
                    judge_display_name(judge), metric, c.n, c.accuracy
                )
            )

    for title, cells in (
        ("LLM MAJORITY VS. HUMAN CONSENSUS COHEN'S KAPPA", majority_human_kappa),
        ("FLEISS' KAPPA AMONG ALL LLM JUDGES", llm_fleiss_kappa),
        ("FLEISS' KAPPA AMONG HUMAN ANNOTATORS", human_fleiss_kappa),
    ):
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)
        for metric in METRICS:
            cell = cells[metric]
            print(
                f"{metric:<10s} n={cell.n:>6d}  kappa={cell.kappa:>8.4f}  "
                f"agreement={cell.observed_agreement:>7.2%}  "
                f"positive={_format_prevalence(cell.positive_prevalence)}"
            )


def run(
    input_path: str = "data/final/obs_data_per_score.parquet",
    output: Optional[str] = None,
    *,
    human_annotations_path: str | Path = DEFAULT_HUMAN_ANNOTATIONS_PATH,
    validated_samples_path: str | Path | None = DEFAULT_VALIDATED_SAMPLES_PATH,
    obs_data_path: str | Path = DEFAULT_OBS_DATA_PATH,
    seed: int = DEFAULT_TIE_BREAK_SEED,
    no_save: bool = False,
    no_percent: bool = False,
) -> None:
    """Generate the judge accuracy + IRR LaTeX table.

    Human-baseline responses are dropped from the input before any panel is
    computed, and the validation population covers the closed, open, and
    agentic settings only.

    Args:
        input_path: Path to the kept observations
            (``data/final/obs_data_per_score.parquet``). Must contain the
            ``scorer_model`` and per-grader ``fraud_obs`` / ``refusal_obs`` /
            ``option_obs`` columns plus the corresponding majority-vote
            ``score_fraud`` / ``score_refusal`` / ``score_options`` columns.
            The older per-sample vote-dictionary schema is also accepted.
        output: Output ``.tex`` path. Defaults to
            ``paper/tables/judge_irr.tex``.
        human_annotations_path: JSONL file of human LLM-response ratings,
            one rating per line with an ``annotator_id`` field (an item may be
            rated by several annotators).
        validated_samples_path: Optional CSV containing the complete
            validation-sample ID list used to calculate stratum sample
            counts. Defaults to every human-annotated item.
        obs_data_path: Canonical sample-level obs-data file used to calculate
            eligible target-population stratum counts.
        seed: Reproducible seed used to resolve tied human consensus votes.
        no_save: If True, print the LaTeX to stdout instead of writing.
        no_percent: If True, render panel (a) accuracies as proportions
            in [0, 1] instead of percentages.
    """
    if output is None:
        output = str(
            Path(__file__).resolve().parents[4]
            / "paper" / "tables" / "judge_irr.tex"
        )

    df = load_data(input_path)
    if "setting" not in df.columns:
        raise ValueError(
            "The input data must contain 'setting' to exclude human-baseline rows."
        )
    n_before = len(df)
    df = df.loc[df["setting"].ne(HUMAN_BASELINE_SETTING)].copy()
    print(f"Excluded {n_before - len(df)} human-baseline rows; {len(df)} remain.")
    judges = discover_judges(df, METRICS)
    if not judges:
        print("No judge models found in score columns -- skipping table.")
        return
    print(f"Found {len(judges)} judge model(s): {', '.join(judges)}")

    accuracy = compute_accuracy(df, judges)
    llm_fleiss_kappa = compute_llm_fleiss_kappa(df, judges)
    human_long = load_human_annotations(
        human_annotations_path,
        validated_samples_path,
    )
    validated_ids = resolve_validation_sample_ids(human_long, validated_samples_path)
    print(
        f"Loaded {len(human_long)} human ratings from "
        f"{human_long['annotator_id'].nunique()} annotator(s) covering "
        f"{len(validated_ids)} validation items."
    )
    validation_design = build_validation_design(obs_data_path, validated_ids)
    print(
        f"Built validation weights for {len(validation_design.items)} items "
        f"across {len(validation_design.sample_sizes)} non-empty strata."
    )
    analyzed_ids = set(validation_design.items["sample_id"])
    human_long = human_long.loc[
        human_long["sample_id"].isin(analyzed_ids)
    ].copy()
    majority_human_kappa, human_fleiss_kappa = compute_human_agreement(
        df,
        human_long,
        validation_design,
        seed=seed,
    )

    _print_summary(
        accuracy, majority_human_kappa, llm_fleiss_kappa, human_fleiss_kappa, judges
    )

    latex = to_latex(
        accuracy,
        majority_human_kappa,
        llm_fleiss_kappa,
        human_fleiss_kappa,
        judges=judges,
        caption=_caption(
            accuracy,
            majority_human_kappa,
            llm_fleiss_kappa,
            human_fleiss_kappa,
            judges,
            len(validation_design.items),
            seed,
        ),
        label="tab:judge_irr",
        no_percent=no_percent,
    )

    if no_save:
        print("\n" + "=" * 60)
        print("LATEX TABLE")
        print("=" * 60)
        print(latex)
    else:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(latex + "\n", encoding="utf-8")
        print(f"\nSaved LaTeX table to: {out}")
