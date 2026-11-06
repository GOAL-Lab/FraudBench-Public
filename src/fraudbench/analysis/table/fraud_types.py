"""LaTeX table of scenario counts by fraud type (misrepresentation vs concealment).

For each scenario (one row per ``summary_id``) we derive the fraud type from
``metadata.is_concealment`` in ``data/final/dataset_public.jsonl`` (True →
Fraudulent Concealment, False → Fraudulent Misrepresentation).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from fraudbench.utils import flatten_public_dataset

FRAUD_TYPE_ORDER = ["Fraudulent Misrepresentation", "Fraudulent Concealment"]

def load_scenarios(input_path: str) -> pd.DataFrame:
    """Load the scenario-level dataset, deduplicated to one row per summary_id."""
    path = Path(input_path)
    if path.suffix == ".jsonl":
        df = pd.read_json(input_path, lines=True)
        df = flatten_public_dataset(df)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif path.suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    df = df.drop_duplicates(subset="summary_id").reset_index(drop=True)
    print(f"Loaded {len(df)} scenarios from {input_path}")
    return df


def _derive_fraud_type(df: pd.DataFrame) -> pd.Series:
    """Map ``is_concealment`` boolean to a human-readable fraud-type label."""
    if "is_concealment" not in df.columns:
        raise ValueError(
            "Input must contain 'is_concealment' (from "
            "data/final/dataset_public.jsonl metadata.is_concealment)."
        )
    return df["is_concealment"].map(
        {True: "Fraudulent Concealment", False: "Fraudulent Misrepresentation"}
    )


def compute_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame indexed by fraud type with count and proportion columns."""
    fraud_type = _derive_fraud_type(df)
    counts = fraud_type.value_counts(dropna=False)
    ordered = [t for t in FRAUD_TYPE_ORDER if t in counts.index]
    extras = [t for t in counts.index if t not in ordered]
    counts = counts.reindex(ordered + extras)
    total = int(counts.sum())
    out = pd.DataFrame({
        "count": counts.astype(int).values,
        "proportion": counts.values / total if total else 0.0,
    }, index=counts.index)
    out.index.name = "fraud_type"
    out.loc["Total"] = [total, 1.0 if total else 0.0]
    return out


def to_latex(counts: pd.DataFrame) -> str:
    """Render the fraud-type frequency table as a LaTeX ``table`` environment."""
    T1 = "    "
    T2 = "        "
    lines: list[str] = []
    lines.append("\\begin{table}[!htbp]")
    lines.append(f"{T1}\\centering")
    lines.append(f"{T1}\\small")
    lines.append(f"{T1}\\setlength{{\\tabcolsep}}{{8pt}}")
    lines.append(f"{T1}\\begin{{tabular}}{{lrr}}")
    lines.append(f"{T2}\\toprule")
    lines.append(f"{T2}\\textbf{{Fraud Type}} & \\textbf{{Count}} & \\textbf{{Proportion}} \\\\")
    lines.append(f"{T2}\\midrule")
    for fraud_type in counts.index:
        if fraud_type == "Total":
            continue
        n = int(counts.loc[fraud_type, "count"])
        share = float(counts.loc[fraud_type, "proportion"])
        pct = 100.0 * share
        lines.append(f"{T2}{fraud_type} & {n} & {pct:.1f}\\% \\\\")
    lines.append(f"{T2}\\midrule")
    total_n = int(counts.loc["Total", "count"])
    lines.append(
        f"{T2}\\textbf{{Total}} & \\textbf{{{total_n}}} & "
        "\\textbf{100.0\\%} \\\\"
    )
    lines.append(f"{T2}\\bottomrule")
    lines.append(f"{T1}\\end{{tabular}}")
    lines.append(
        f"{T1}\\caption{{Distribution of \\textsc{{FraudBench}} scenarios "
        "by fraud type. Fraudulent "
        "misrepresentation involves a material false representation, while "
        "fraudulent concealment involves the non-disclosure of a material "
        "fact while under a duty to disclose.}"
    )
    lines.append(f"{T1}\\label{{tab:fraud_types}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def run(
    input_path: str = "data/final/dataset_public.jsonl",
    output: Optional[str] = None,
    no_save: bool = False,
) -> pd.DataFrame:
    """Generate the fraud-type frequency table and optionally save LaTeX."""
    if output is None:
        output = str(
            Path(__file__).resolve().parents[4] / "paper" / "tables" / "fraud_types.tex"
        )

    df = load_scenarios(input_path)
    counts = compute_counts(df)

    print("\n" + "=" * 60)
    print("FRAUD TYPE COUNTS")
    print("=" * 60)
    print(counts.to_string())

    latex = to_latex(counts)

    if not no_save:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(latex + "\n", encoding="utf-8")
        print(f"\nSaved LaTeX table to: {out}")
    else:
        print("\n" + "=" * 60)
        print("LATEX TABLE")
        print("=" * 60)
        print(latex)

    return counts
