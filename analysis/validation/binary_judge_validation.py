"""
Validate the binary fraud-presence LLM annotations against human labels.

Reads the per-holding LLM annotations in
``data/annotations/fraud_presence/holdings_llm_annotations.jsonl`` and the
consolidated human annotations in
``data/annotations/fraud_presence/holdings_human_annotations.jsonl``.
For every holding that has both a human label and an LLM consensus label,
computes accuracy and Cohen's Kappa against:

    - each individual judge model's overall label (majority vote across
      that model's 3 runs, stored as ``overall_score_<model>``)
    - the combined cross-model overall label (majority vote across the
      9 model x run scores, stored as ``overall_score``)

Labels are 3-class: TRUE / FALSE / INCONCLUSIVE.

Usage:
    uv run analysis/validation/binary_judge_validation.py
    uv run analysis/validation/binary_judge_validation.py \
        --human_annotations_path \
            data/annotations/fraud_presence/holdings_human_annotations.jsonl \
        --llm_annotations_path \
            data/annotations/fraud_presence/holdings_llm_annotations.jsonl
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from jsonargparse import CLI
from sklearn.metrics import cohen_kappa_score


VALIDATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = VALIDATION_DIR.parents[1]
DEFAULT_ANNOTATIONS_DIR = REPO_ROOT / "data" / "annotations" / "fraud_presence"
DEFAULT_LLM_ANNOTATIONS = (
    DEFAULT_ANNOTATIONS_DIR / "holdings_llm_annotations.jsonl"
)
DEFAULT_HUMAN_ANNOTATIONS = (
    DEFAULT_ANNOTATIONS_DIR / "holdings_human_annotations.jsonl"
)

LABEL_VALUES: tuple[str, ...] = ("TRUE", "FALSE", "INCONCLUSIVE")
HUMAN_KEYS: tuple[str, ...] = ("k8", "k15", "k94")
LLM_MODELS: tuple[str, ...] = (
    "gpt-4o",
    "claude-sonnet-4-20250514",
    "gemini-2.5-flash",
)
COMBINED_JUDGE = "combined"
JUDGES: tuple[str, ...] = (*LLM_MODELS, COMBINED_JUDGE)


def _judge_col(judge: str) -> str:
    """Column name in the LLM JSONL holding the consensus label for ``judge``."""
    if judge == COMBINED_JUDGE:
        return "overall_score"
    return f"overall_score_{judge}"


def _normalize_label(value) -> Optional[str]:
    """Normalize a label to one of LABEL_VALUES, or None if missing/invalid."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    s = str(value).strip().upper()
    if s in LABEL_VALUES:
        return s
    return None


def _load_human_annotations(human_annotations_path: Path) -> pd.DataFrame:
    """Load the consolidated human annotations JSONL into a long-form df.

    One row per distinct holding annotated by one human annotator, across
    all West keys (``k8``/``k15``/``k94``). Rows with a missing or
    non-canonical ``is_fraud_human`` value are dropped.

    Returns a DataFrame with columns:
        holding_id, key, is_fraud_human (str in LABEL_VALUES)
    """
    human_annotations_path = Path(human_annotations_path)
    if not human_annotations_path.exists():
        raise FileNotFoundError(
            f"Missing human annotations JSONL: {human_annotations_path}"
        )
    df = pd.read_json(human_annotations_path, lines=True)
    for col in ("holding_id", "key", "is_fraud_human"):
        if col not in df.columns:
            raise ValueError(
                f"{human_annotations_path.name} is missing required "
                f"column {col!r}."
            )
    df = df[["holding_id", "key", "is_fraud_human"]].copy()
    df["is_fraud_human"] = df["is_fraud_human"].map(_normalize_label)
    n_total = len(df)
    df = df.dropna(subset=["is_fraud_human"]).reset_index(drop=True)
    n_dropped = n_total - len(df)
    df["key"] = df["key"].astype(str)

    if df["holding_id"].duplicated().any():
        dup_ids = df.loc[
            df["holding_id"].duplicated(keep=False), "holding_id"
        ].unique()
        raise ValueError(
            f"Duplicate holding_id in human annotations: {list(dup_ids)[:5]}"
        )

    print(
        f"Loaded {n_total} human-annotated rows from "
        f"{human_annotations_path.name}; dropped {n_dropped} rows with "
        f"missing/invalid is_fraud_human."
    )
    return df


def _load_llm_annotations(llm_annotations_path: Path) -> pd.DataFrame:
    """Load the per-holding LLM consensus annotations."""
    llm_annotations_path = Path(llm_annotations_path)
    df = pd.read_json(llm_annotations_path, lines=True)
    required = ["holding_id", "overall_score"] + [
        _judge_col(j) for j in LLM_MODELS
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"{llm_annotations_path.name} is missing required column "
                f"{col!r}."
            )
    for j in JUDGES:
        col = _judge_col(j)
        df[col] = df[col].map(_normalize_label)
    return df


def _evaluate(
    merged: pd.DataFrame, judges: tuple[str, ...] = JUDGES
) -> pd.DataFrame:
    """Compute accuracy and Cohen's Kappa per judge over labeled holdings.

    For each judge, restricts to rows with both a non-null human label and
    a non-null judge label, then reports counts and metrics.
    """
    rows: list[dict] = []
    for judge in judges:
        col = _judge_col(judge)
        mask = merged["is_fraud_human"].notna() & merged[col].notna()
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "judge": judge,
                    "n_compared": 0,
                    "accuracy": float("nan"),
                    "cohen_kappa": float("nan"),
                    "n_judge_missing": int(merged[col].isna().sum()),
                }
            )
            continue
        human = merged.loc[mask, "is_fraud_human"].astype(str).to_numpy()
        llm = merged.loc[mask, col].astype(str).to_numpy()
        accuracy = float(np.mean(human == llm))
        kappa = float(
            cohen_kappa_score(human, llm, labels=list(LABEL_VALUES))
        )
        rows.append(
            {
                "judge": judge,
                "n_compared": n,
                "accuracy": accuracy,
                "cohen_kappa": kappa,
                "n_judge_missing": int(merged[col].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def _per_key_evaluate(
    merged: pd.DataFrame, judges: tuple[str, ...] = JUDGES
) -> pd.DataFrame:
    """Re-run ``_evaluate`` separately for each human-annotation key."""
    out: list[pd.DataFrame] = []
    for key in HUMAN_KEYS:
        sub = merged[merged["key"] == key]
        if sub.empty:
            continue
        df = _evaluate(sub, judges)
        df.insert(0, "key", key)
        df.insert(1, "n_holdings", len(sub))
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _confusion_matrix(
    merged: pd.DataFrame, judge: str
) -> pd.DataFrame:
    """Return a TRUE/FALSE/INCONCLUSIVE confusion matrix for a single judge.

    Rows are the human label, columns are the judge label.
    """
    col = _judge_col(judge)
    sub = merged.dropna(subset=["is_fraud_human", col])
    cm = pd.crosstab(
        sub["is_fraud_human"],
        sub[col],
        rownames=["human"],
        colnames=[judge],
        dropna=False,
    )
    cm = cm.reindex(index=list(LABEL_VALUES), columns=list(LABEL_VALUES), fill_value=0)
    return cm


def _print_label_distribution(human: pd.DataFrame) -> None:
    print("\n=== Human label distribution ===")
    overall = human["is_fraud_human"].value_counts().reindex(LABEL_VALUES, fill_value=0)
    total = int(overall.sum())
    print(f"\nOverall (n={total}):")
    for label in LABEL_VALUES:
        n = int(overall[label])
        share = n / total if total else float("nan")
        print(f"  {label:<13s} {n:>5d}  ({share:.1%})")
    by_key = (
        human.groupby("key")["is_fraud_human"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=list(LABEL_VALUES), fill_value=0)
        .reindex(index=list(HUMAN_KEYS))
    )
    print("\nBy key:")
    print(by_key.to_string())


def _print_results_table(results: pd.DataFrame, title: str) -> None:
    print(f"\n=== {title} ===")
    if results.empty:
        print("(no rows)")
        return
    has_key = "key" in results.columns
    headers = (
        ["key", "n_holdings", "judge", "n_compared", "accuracy", "cohen_kappa"]
        if has_key
        else ["judge", "n_compared", "accuracy", "cohen_kappa"]
    )
    fmt_header = (
        "{:<6s} {:>11s} {:<28s} {:>10s} {:>10s} {:>14s}"
        if has_key
        else "{:<28s} {:>10s} {:>10s} {:>14s}"
    )
    fmt_row = (
        "{:<6s} {:>11d} {:<28s} {:>10d} {:>10.4f} {:>14.4f}"
        if has_key
        else "{:<28s} {:>10d} {:>10.4f} {:>14.4f}"
    )
    print("\n" + fmt_header.format(*headers))
    for _, row in results.iterrows():
        if has_key:
            print(
                fmt_row.format(
                    str(row["key"]),
                    int(row["n_holdings"]),
                    str(row["judge"]),
                    int(row["n_compared"]),
                    float(row["accuracy"]),
                    float(row["cohen_kappa"]),
                )
            )
        else:
            print(
                fmt_row.format(
                    str(row["judge"]),
                    int(row["n_compared"]),
                    float(row["accuracy"]),
                    float(row["cohen_kappa"]),
                )
            )


def _print_confusion_matrices(merged: pd.DataFrame) -> None:
    print("\n=== Per-judge confusion matrices (rows=human, cols=judge) ===")
    for judge in JUDGES:
        cm = _confusion_matrix(merged, judge)
        print(f"\n[{judge}]")
        print(cm.to_string())


def validate(
    human_annotations_path: Path = DEFAULT_HUMAN_ANNOTATIONS,
    llm_annotations_path: Path = DEFAULT_LLM_ANNOTATIONS,
) -> None:
    """Run the binary fraud-presence judge validation and print results.

    Args:
        human_annotations_path: Path to ``holdings_human_annotations.jsonl``.
            Must contain ``holding_id``, ``key``, and ``is_fraud_human``
            columns.
        llm_annotations_path: Path to ``holdings_llm_annotations.jsonl``.
            Must contain ``holding_id``, ``overall_score``, and
            ``overall_score_<model>`` columns for the configured models.
    """
    human_annotations_path = Path(human_annotations_path)
    llm_annotations_path = Path(llm_annotations_path)

    human = _load_human_annotations(human_annotations_path)
    n_holdings = human["holding_id"].nunique()
    print(
        f"Resolved {n_holdings} unique human-labeled holdings from "
        f"{human_annotations_path}."
    )

    llm = _load_llm_annotations(llm_annotations_path)
    print(
        f"Loaded {len(llm)} per-holding LLM consensus rows from "
        f"{llm_annotations_path}."
    )

    keep_cols = ["holding_id"] + [_judge_col(j) for j in JUDGES]
    merged = human.merge(
        llm[keep_cols], on="holding_id", how="left", validate="1:1"
    )
    n_missing_llm = int(merged[_judge_col(COMBINED_JUDGE)].isna().sum())
    print(
        f"Joined human and LLM tables on holding_id: {len(merged)} rows "
        f"({n_missing_llm} human holdings without an LLM combined label)."
    )

    _print_label_distribution(human)

    overall_results = _evaluate(merged)
    _print_results_table(
        overall_results, "Validation against LLM consensus (overall)"
    )

    per_key_results = _per_key_evaluate(merged)
    _print_results_table(
        per_key_results, "Validation against LLM consensus (per key)"
    )

    _print_confusion_matrices(merged)


if __name__ == "__main__":
    CLI(validate)
