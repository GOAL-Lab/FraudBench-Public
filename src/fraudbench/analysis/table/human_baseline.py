"""Open-QA Human-baseline table, separate from AI per-model appendices.

The Human baseline evaluates the Open QA condition under the Control instruction
across contextual conditions. This module owns its table-specific caption,
layout choices, and unshaded cell rendering while reusing the shared
aggregation and LaTeX helpers from :mod:`fraudbench.analysis.table.per_model`.
"""
from pathlib import Path
from typing import Optional

import pandas as pd

from ..common import (
    HUMAN_DISPLAY_NAME,
    HUMAN_MODEL_ID,
    available_parquet_columns,
    ensure_model_id_column,
    filter_human_baseline,
    filter_main_runs,
)
from .per_model import (
    _NEEDED_COLUMNS,
    _add_condition_keys,
    _expected_per_cell,
    _normalize_columns,
    compute_cells_for_model,
    load_data,
    to_latex,
)

HUMAN_BASELINE_SETTING_LABEL = "Open QA (Human Baseline Version)"


def _default_output_path() -> Path:
    """Return the stable human-table path used by the appendix."""
    return (
        Path(__file__).resolve().parents[4]
        / "paper"
        / "tables"
        / "per_model"
        / "per_model_human.tex"
    )


def _load_human_baseline_data(input_path: str) -> pd.DataFrame:
    """Load only Human-baseline rows from parquet inputs when possible."""
    path = Path(input_path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(
            input_path,
            columns=available_parquet_columns(path, _NEEDED_COLUMNS),
            filters=[("setting", "==", "human_baseline")],
        )
        print(f"Loaded {len(df)} Human-baseline rows from {input_path}")
        return ensure_model_id_column(df)
    return load_data(input_path)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Select Human Open-QA baseline rows and derive table condition keys."""
    df = _normalize_columns(df)
    df = filter_main_runs(df)
    df = filter_human_baseline(df, verbose=False)
    df = df[df["model_id"] == HUMAN_MODEL_ID].copy()
    # Reuse the shared per-model aggregation key while overriding the rendered
    # header with ``HUMAN_BASELINE_SETTING_LABEL`` below.
    df["setting"] = "closed"
    return _add_condition_keys(df)


def _caption(*, as_percent: bool) -> str:
    interval_phrase = "95\\% Wilson binomial CI"
    return (
        "Results for the human baseline (human respondents only). The baseline is run only on the human baseline version of the Open QA tasks, and only contextual conditions are tested. "
        f"For every metric, the table reports the {'percentage' if as_percent else 'proportion'} "
        "of effective samples "
        f"with a bracketed {interval_phrase} underneath. "
    )


def run(
    input_path: str = "data/final/obs_data.parquet",
    output_path: Optional[str] = None,
    *,
    no_save: bool = False,
    no_percent: bool = False,
) -> Optional[Path]:
    """Generate the standalone Human baseline LaTeX table."""
    df = prepare(_load_human_baseline_data(input_path))
    if df.empty:
        print("No Human baseline rows found in input data; nothing to render.")
        return None

    expected_per_cell = _expected_per_cell(df, fallback=0)
    cells = compute_cells_for_model(df, expected_per_cell=expected_per_cell)
    latex = to_latex(
        HUMAN_MODEL_ID,
        HUMAN_DISPLAY_NAME,
        cells,
        as_percent=not no_percent,
        shade_cells=False,
        caption=_caption(as_percent=not no_percent),
        setting_display={"closed": HUMAN_BASELINE_SETTING_LABEL},
    )

    if no_save:
        print(latex)
        return None

    path = Path(output_path) if output_path else _default_output_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex, encoding="utf-8")
    print(f"Wrote {path}")
    return path
