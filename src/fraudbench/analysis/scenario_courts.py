"""Load court fields embedded directly in ``dataset_public.jsonl``.

The released scenario records carry top-level ``state``, ``jurisdiction``, and
``level`` fields, so analyses should not rejoin ``courts_metadata.json``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraudbench.utils import flatten_public_dataset


DEFAULT_SCENARIOS = "data/final/dataset_public.jsonl"
COURT_COLUMNS = ["state", "jurisdiction", "level"]


def load_scenario_courts(input_path: str = DEFAULT_SCENARIOS) -> pd.DataFrame:
    """Return one row per scenario with normalized, embedded court fields.

    JSONL inputs are flattened to match the other scenario-level analysis
    loaders. The top-level court fields remain unchanged by flattening.
    """
    input_suffix = Path(input_path).suffix
    if input_suffix in {".json", ".jsonl"}:
        scenarios = pd.read_json(input_path, lines=True)
        scenarios = flatten_public_dataset(scenarios)
    elif input_suffix == ".parquet":
        scenarios = pd.read_parquet(input_path)
    elif input_suffix == ".csv":
        scenarios = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported scenario file format: {input_suffix}")

    required_columns = {"summary_id", *COURT_COLUMNS}
    missing_columns = required_columns - set(scenarios.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Scenario data is missing required embedded court fields: {missing}"
        )

    scenarios = scenarios.drop_duplicates(subset="summary_id").reset_index(drop=True)
    for column in COURT_COLUMNS:
        scenarios[column] = scenarios[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    print(f"Loaded {len(scenarios)} scenarios from {input_path}")
    return scenarios[["summary_id", *COURT_COLUMNS]]
