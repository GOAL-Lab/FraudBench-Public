"""
Top up an existing human-validation sample while preserving stratification.

The input is a CSV containing the ``sample_id`` values that have already been
annotated. The script infers their experimental setting from ``obs_data``,
counts them toward the existing per-cell minimums and group targets, and
samples only the additional rows needed to complete a target design.

New rows are drawn from the same pool ``stratified_sample.py`` uses, so the two
scripts share their eligibility filters. Previously annotated rows that no
longer clear the gradeability filter (an agentic row whose
``completion_extended`` is now blank, say) still count toward the targets but
are reported and left out of the population.

Usage:
    uv run analysis/validation/top_up_stratified_sample.py \
        annotated_sample_ids.csv
    uv run analysis/validation/top_up_stratified_sample.py \
        annotated_sample_ids.csv \
        --total_n 400 --seed 7
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from jsonargparse import CLI

import fraudbench.utils.globals as globals

VALIDATION_DIR = Path(__file__).resolve().parent
# stratified_sample.py is a sibling script rather than an installed module, so
# make it importable regardless of how this file is invoked.
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

import stratified_sample as base  # noqa: E402


def _load_annotated_ids(path: Path) -> list[str]:
    """Load and deduplicate sample IDs from a CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"annotated_ids_path does not exist: {path}")

    ids_df = pd.read_csv(path, dtype=str)
    matching_columns = [
        column
        for column in ids_df.columns
        if column.strip().lower().replace(" ", "_") == "sample_id"
    ]
    if not matching_columns:
        raise ValueError(
            f"{path.name} must contain a 'sample_id' or 'Sample ID' column."
        )
    if len(matching_columns) > 1:
        raise ValueError(
            f"{path.name} contains multiple sample-ID columns: "
            f"{matching_columns}."
        )

    id_column = matching_columns[0]
    ids = ids_df[id_column].dropna().str.strip()
    n_blank = int((ids == "").sum())
    ids = ids[ids != ""]
    if ids.empty:
        raise ValueError(f"{path.name} contains no non-empty sample IDs.")

    n_before = len(ids)
    unique_ids = ids.drop_duplicates().tolist()
    n_duplicates = n_before - len(unique_ids)
    if n_duplicates or n_blank:
        print(
            f"[warn] Ignored {n_duplicates} duplicate and {n_blank} blank "
            "sample-ID row(s)."
        )
    return unique_ids


def _prepare_pool(
    obs: pd.DataFrame, annotated_ids: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Validate annotated IDs, infer their setting, and build the sample pool."""
    required_columns = {
        "sample_id",
        "run_category",
        "setting",
        "model",
        "grader_counts_fraud",
        "grader_counts_refusal",
        "summary_id",
    }
    missing_columns = required_columns - set(obs.columns)
    if missing_columns:
        raise ValueError(
            f"obs_data is missing required columns: {sorted(missing_columns)}"
        )

    obs = obs.copy()
    if obs["sample_id"].isna().any():
        raise ValueError("obs_data contains null sample_id values.")
    obs["sample_id"] = obs["sample_id"].astype(str)
    duplicate_source_ids = obs.loc[
        obs["sample_id"].duplicated(keep=False), "sample_id"
    ].unique()
    if len(duplicate_source_ids):
        raise ValueError(
            "obs_data contains duplicate sample_id values; examples: "
            f"{duplicate_source_ids[:5].tolist()}"
        )

    annotated_id_set = set(annotated_ids)
    matched = obs[obs["sample_id"].isin(annotated_id_set)].copy()
    matched_id_set = set(matched["sample_id"])
    unknown_ids = [
        sample_id
        for sample_id in annotated_ids
        if sample_id not in matched_id_set
    ]
    if unknown_ids:
        raise ValueError(
            f"{len(unknown_ids)} annotated sample ID(s) were not found in "
            f"obs_data; examples: {unknown_ids[:5]}"
        )

    if matched["setting"].isna().any():
        null_setting_ids = matched.loc[matched["setting"].isna(), "sample_id"].tolist()
        raise ValueError(
            f"{len(null_setting_ids)} annotated ID(s) have a null setting in "
            f"obs_data; examples: {null_setting_ids[:5]}"
        )
    settings = matched["setting"].unique().tolist()
    if len(settings) != 1 or settings[0] not in base.SETTING_CHOICES:
        raise ValueError(
            "Annotated IDs must belong to exactly one supported setting; "
            f"found: {settings}"
        )
    setting = str(settings[0])

    if setting == "agentic" and "completion_extended" not in obs.columns:
        raise ValueError(
            "obs_data is missing 'completion_extended', which is required to "
            "identify gradeable agentic rows."
        )

    # Rows that belong to the setting, before the gradeability filter. Annotated
    # rows must clear this bar; failing it means they came from a different pool.
    candidates = obs.loc[base.setting_mask(obs, setting)].copy()
    candidate_ids = set(candidates["sample_id"])
    ineligible_ids = [
        sample_id for sample_id in annotated_ids if sample_id not in candidate_ids
    ]
    if ineligible_ids:
        raise ValueError(
            f"{len(ineligible_ids)} annotated ID(s) are outside the validation "
            f"pool for setting={setting!r}; examples: {ineligible_ids[:5]}"
        )

    # Ungradeable rows only matter if they were already annotated, in which case
    # they still count toward the targets.
    candidates = candidates.loc[
        base.gradeable_mask(candidates, setting)
        | candidates["sample_id"].isin(annotated_id_set)
    ].copy()

    for column in ("grader_counts_fraud", "grader_counts_refusal"):
        numeric = pd.to_numeric(candidates[column], errors="coerce")
        invalid = numeric.isna() | ~numeric.isin(base.LEVELS)
        if invalid.any():
            bad_ids = candidates.loc[invalid, "sample_id"].head(5).tolist()
            raise ValueError(
                f"Eligible rows contain invalid {column} values; sample IDs: "
                f"{bad_ids}"
            )
        candidates[column] = numeric.astype(int)

    annotated = candidates[candidates["sample_id"].isin(annotated_id_set)].copy()
    if len(annotated) != len(annotated_ids):
        raise AssertionError("Validated annotated IDs were lost while building the pool.")

    pool = candidates.loc[base.gradeable_mask(candidates, setting)].copy()
    if pool.empty:
        raise RuntimeError(f"No gradeable rows found for setting={setting!r}.")

    ungradeable_annotated = ~annotated["sample_id"].isin(set(pool["sample_id"]))
    if ungradeable_annotated.any():
        stale_ids = annotated.loc[ungradeable_annotated, "sample_id"].tolist()
        print(
            f"[warn] {len(stale_ids)} annotated row(s) fall outside the current "
            f"gradeable pool ({base.pool_filter_description(setting)}). They "
            "still count toward the targets but are not part of the population; "
            f"examples: {stale_ids[:5]}"
        )
    return pool, annotated, setting


def _group_total(counts: pd.DataFrame, max_value: int) -> int:
    """Sum a cell-count matrix over one max(fraud, refusal) group."""
    return int(
        sum(
            counts.loc[f, r]
            for f in base.LEVELS
            for r in base.LEVELS
            if max(f, r) == max_value
        )
    )


def _sample_ids(
    pool: pd.DataFrame,
    eligible_mask: pd.Series,
    take: int,
    excluded_ids: set[str],
    rng: np.random.Generator,
    description: str,
) -> list[str]:
    """Sample IDs matching a mask, excluding annotated and prior selections."""
    if take <= 0:
        return []
    eligible = pool.loc[
        eligible_mask & ~pool["sample_id"].isin(excluded_ids), "sample_id"
    ].to_numpy()
    if len(eligible) < take:
        print(
            f"[warn] Only {len(eligible)} unannotated row(s) remain for "
            f"{description}; taking all (needed {take})."
        )
        take = len(eligible)
    if take == 0:
        return []
    return [str(value) for value in rng.choice(eligible, size=take, replace=False)]


def _attach_scoring_criteria(
    sample: pd.DataFrame, setting: str, dataset_public_path: Path
) -> tuple[pd.DataFrame, int]:
    """Merge the setting-appropriate scoring criterion onto sampled rows."""
    dataset_public_path = Path(dataset_public_path)
    if not dataset_public_path.exists():
        raise FileNotFoundError(
            f"dataset_public_path does not exist: {dataset_public_path}"
        )

    criteria_column = base.SETTING_TO_CRITERIA[setting]
    scenarios = pd.read_json(dataset_public_path, lines=True)
    missing_columns = {"summary_id", criteria_column} - set(scenarios.columns)
    if missing_columns:
        raise ValueError(
            f"{dataset_public_path.name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )
    criteria = (
        scenarios[["summary_id", criteria_column]]
        .drop_duplicates(subset="summary_id")
        .rename(columns={criteria_column: "scoring_criteria"})
    )

    before = len(sample)
    sample = sample.merge(criteria, on="summary_id", how="left", validate="many_to_one")
    if len(sample) != before:
        raise AssertionError(
            f"Scoring-criteria merge changed row count: {before} -> {len(sample)}."
        )
    n_missing = int(sample["scoring_criteria"].isna().sum())
    if n_missing:
        missing_ids = sample.loc[
            sample["scoring_criteria"].isna(), "summary_id"
        ].unique()
        print(
            f"[warn] {n_missing} additional row(s) have no matching scoring "
            f"criteria; example summary IDs: {missing_ids[:5].tolist()}"
        )
    return sample, n_missing


def top_up_validation_sample(
    annotated_ids_path: Path,
    seed: int = globals.RANDOM_SEED,
    total_n: int = base.TOTAL_N,
    min_per_cell: int = base.MIN_PER_CELL,
    input_path: Path = base.DEFAULT_INPUT,
    dataset_public_path: Path = base.DEFAULT_DATASET_PUBLIC,
    output_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
) -> Path:
    """
    Select only the additional rows needed for a stratified validation sample.

    Args:
        annotated_ids_path: CSV containing a ``sample_id`` or ``Sample ID``
            column with all previously annotated IDs.
        seed: Random seed used for reproducible selection.
        total_n: Target size of the combined annotated and additional sample.
        min_per_cell: Minimum combined count in each fraud/refusal cell, capped
            by the number of available rows in that cell.
        input_path: Source ``obs_data.parquet`` path.
        dataset_public_path: Source of setting-specific scoring criteria.
        output_path: CSV for additional rows only. Defaults to
            ``analysis/validation/validation_sample_{setting}_top_up.csv``.
        log_path: JSON log recording the seed, inputs and per-cell counts.
            Defaults to ``{output_path}.log.json``.
    """
    if total_n < 0:
        raise ValueError(f"total_n must be >= 0, got {total_n}.")
    if min_per_cell < 0:
        raise ValueError(f"min_per_cell must be >= 0, got {min_per_cell}.")

    annotated_ids = _load_annotated_ids(annotated_ids_path)
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input_path does not exist: {input_path}")
    obs = pd.read_parquet(input_path)
    pool, annotated, setting = _prepare_pool(obs, annotated_ids)

    population_counts = base.population_matrix(pool)
    annotated_counts = base.counts_from_rows(annotated)
    minimum_targets = population_counts.clip(upper=min_per_cell).astype(int)
    minimum_total = int(minimum_targets.to_numpy().sum())
    if total_n < minimum_total:
        raise ValueError(
            f"total_n={total_n} is smaller than the {minimum_total} rows "
            "needed to satisfy the population-capped per-cell minimums."
        )

    group_weight_sum = sum(base.GROUP_PERCENTS.values())
    if group_weight_sum <= 0:
        raise ValueError("GROUP_PERCENTS must have at least one positive entry.")
    group_top_up_budget = total_n - minimum_total
    raw_group_top_up_targets = {
        max_value: group_top_up_budget
        * (base.GROUP_PERCENTS[max_value] / group_weight_sum)
        for max_value in sorted(base.GROUP_PERCENTS)
    }
    group_top_up_targets = base.largest_remainder(
        raw_group_top_up_targets, group_top_up_budget
    )
    group_final_targets = {
        max_value: _group_total(minimum_targets, max_value)
        + group_top_up_targets[max_value]
        for max_value in sorted(base.GROUP_PERCENTS)
    }

    print(f"Using seed={seed} (RANDOM_SEED_OFFSET={globals.RANDOM_SEED_OFFSET})")
    print(f"Inferred setting={setting!r} from {len(annotated)} annotated IDs.")
    print(f"Eligible population: {len(pool)} rows")
    print("\nPopulation counts:")
    print(population_counts.to_string())
    print("\nAlready annotated counts:")
    print(annotated_counts.to_string())

    rng = np.random.default_rng(seed)
    excluded_ids = set(annotated_ids)
    selected_ids: list[str] = []

    cell_shortfalls: dict[tuple[int, int], int] = {}
    group_shortfalls: dict[int, int] = {}

    phase1_counts = base.empty_counts()
    for fraud_count in base.LEVELS:
        for refusal_count in base.LEVELS:
            deficit = max(
                int(minimum_targets.loc[fraud_count, refusal_count])
                - int(annotated_counts.loc[fraud_count, refusal_count]),
                0,
            )
            cell_mask = pool["grader_counts_fraud"].eq(fraud_count) & pool[
                "grader_counts_refusal"
            ].eq(refusal_count)
            picked_ids = _sample_ids(
                pool,
                cell_mask,
                deficit,
                excluded_ids,
                rng,
                f"cell (fraud={fraud_count}, refusal={refusal_count})",
            )
            selected_ids.extend(picked_ids)
            excluded_ids.update(picked_ids)
            phase1_counts.loc[fraud_count, refusal_count] = len(picked_ids)
            if len(picked_ids) < deficit:
                cell_shortfalls[(fraud_count, refusal_count)] = deficit - len(
                    picked_ids
                )

    counts_after_minimums = annotated_counts + phase1_counts
    phase2_counts = base.empty_counts()
    for max_value in sorted(base.GROUP_PERCENTS):
        deficit = max(
            group_final_targets[max_value]
            - _group_total(counts_after_minimums, max_value),
            0,
        )
        group_mask = pool[
            ["grader_counts_fraud", "grader_counts_refusal"]
        ].max(axis=1).eq(max_value)
        picked_ids = _sample_ids(
            pool,
            group_mask,
            deficit,
            excluded_ids,
            rng,
            f"group max(fraud, refusal)={max_value}",
        )
        selected_ids.extend(picked_ids)
        excluded_ids.update(picked_ids)
        if len(picked_ids) < deficit:
            group_shortfalls[max_value] = deficit - len(picked_ids)
        if picked_ids:
            picked_rows = pool[pool["sample_id"].isin(picked_ids)]
            phase2_counts = phase2_counts.add(
                base.counts_from_rows(picked_rows), fill_value=0
            ).astype(int)

    additional_counts = phase1_counts + phase2_counts
    final_counts = annotated_counts + additional_counts
    # Targets are only enforced up to what the pool could actually supply; any
    # gap must be explained by a reported shortfall.
    for fraud_count in base.LEVELS:
        for refusal_count in base.LEVELS:
            achieved = int(final_counts.loc[fraud_count, refusal_count])
            achieved += cell_shortfalls.get((fraud_count, refusal_count), 0)
            if achieved < int(minimum_targets.loc[fraud_count, refusal_count]):
                raise AssertionError(
                    "Final sample does not satisfy the minimum for cell "
                    f"(fraud={fraud_count}, refusal={refusal_count})."
                )
    for max_value, target in group_final_targets.items():
        achieved = _group_total(final_counts, max_value)
        achieved += group_shortfalls.get(max_value, 0)
        if achieved < target:
            raise AssertionError(
                f"Final sample does not satisfy the target for group "
                f"max(fraud, refusal)={max_value}."
            )
    if set(selected_ids) & set(annotated_ids):
        raise AssertionError("Additional selections overlap annotated sample IDs.")
    if len(selected_ids) != len(set(selected_ids)):
        raise AssertionError("Additional selections contain duplicate sample IDs.")

    additional = pool.set_index("sample_id", drop=False).loc[selected_ids].copy()
    if len(additional):
        additional = additional.iloc[rng.permutation(len(additional))]
    additional = additional.reset_index(drop=True)
    additional, n_missing_criteria = _attach_scoring_criteria(
        additional, setting, Path(dataset_public_path)
    )

    if output_path is None:
        output_path = (
            VALIDATION_DIR / f"validation_sample_{setting}_top_up.csv"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    additional.to_csv(output_path, index=False, encoding="utf-8-sig")

    if log_path is None:
        log_path = output_path.with_suffix(output_path.suffix + ".log.json")
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "setting": setting,
        "seed": int(seed),
        "random_seed_offset": int(globals.RANDOM_SEED_OFFSET),
        "total_n": int(total_n),
        "min_per_cell": int(min_per_cell),
        "group_percents": {int(k): float(v) for k, v in base.GROUP_PERCENTS.items()},
        "annotated_ids_path": str(Path(annotated_ids_path).resolve()),
        "n_annotated": int(len(annotated)),
        "input_path": str(input_path.resolve()),
        "dataset_public_path": str(Path(dataset_public_path).resolve()),
        "criteria_column": base.SETTING_TO_CRITERIA[setting],
        "output_path": str(output_path.resolve()),
        "pool_size": int(len(pool)),
        "additional_rows": int(len(additional)),
        "rows_missing_criteria": int(n_missing_criteria),
        "cell_counts": {
            "population": base.counts_to_dict(population_counts),
            "minimum_targets": base.counts_to_dict(minimum_targets),
            "annotated": base.counts_to_dict(annotated_counts),
            "phase1": base.counts_to_dict(phase1_counts),
            "phase2": base.counts_to_dict(phase2_counts),
            "final": base.counts_to_dict(final_counts),
        },
        "group_totals": {
            int(max_value): {
                "target": int(group_final_targets[max_value]),
                "final": _group_total(final_counts, max_value),
                "shortfall": int(group_shortfalls.get(max_value, 0)),
                "pct_of_top_up_budget": float(
                    base.GROUP_PERCENTS[max_value] / group_weight_sum
                ),
            }
            for max_value in sorted(group_final_targets)
        },
        "cell_shortfalls": {
            f"{fraud_count},{refusal_count}": int(shortfall)
            for (fraud_count, refusal_count), shortfall in cell_shortfalls.items()
        },
    }
    log_path.write_text(json.dumps(log_record, indent=2) + "\n")

    print("\nAdditional rows used to satisfy per-cell minimums:")
    print(phase1_counts.to_string())
    print("\nAdditional rows used to satisfy group totals:")
    print(phase2_counts.to_string())
    print("\nFinal combined allocation (annotated + additional):")
    print(final_counts.to_string())
    print("\nGroup targets and final totals:")
    for max_value in sorted(group_final_targets):
        print(
            f"  max={max_value}: target={group_final_targets[max_value]:>4d}  "
            f"final={_group_total(final_counts, max_value):>4d}"
        )
    final_total = int(final_counts.to_numpy().sum())
    print(
        f"\nAnnotated: {len(annotated)}; additional: {len(additional)}; "
        f"combined: {final_total} (target: {total_n})"
    )
    if final_total > total_n:
        print(
            f"[note] Combined total exceeds the target by "
            f"{final_total - total_n} because existing annotations overfilled "
            "one or more stratification requirements."
        )
    if cell_shortfalls or group_shortfalls:
        print(
            f"[warn] {sum(cell_shortfalls.values()) + sum(group_shortfalls.values())}"
            " row(s) of the target design could not be filled; the pool ran out "
            "of unannotated rows. See 'cell_shortfalls'/'group_totals' in the log."
        )
    if additional.empty:
        print(
            f"[warn] No additional rows were needed: the {len(annotated)} "
            "annotated row(s) already satisfy every cell minimum and group "
            f"target for total_n={total_n}. Raise --total_n to sample more."
        )
    print(f"Wrote {len(additional)} additional rows to {output_path}")
    print(f"Wrote sampling log to {log_path}")
    return output_path


if __name__ == "__main__":
    CLI(top_up_validation_sample)
