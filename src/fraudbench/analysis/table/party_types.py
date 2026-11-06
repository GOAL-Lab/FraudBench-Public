"""LaTeX 2x2 contingency table of plaintiff/defendant party types per scenario.

For each scenario (one row per ``summary_id``) we classify the defendant and
plaintiff as either a business (organisation) or a consumer (individual) using
``seed.defendant_is_org`` / ``seed.plaintiff_is_org`` from
``data/final/dataset_public.jsonl``. The four cells of the 2x2 table
correspond to the standard B2B / B2C / C2B / C2C labels, where the first letter
is the defendant (alleged perpetrator) and the second is the plaintiff (victim).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from fraudbench.utils import flatten_public_dataset

PARTY_LABELS = {True: "Business", False: "Consumer"}

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


def compute_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return a 2x2 DataFrame of scenario counts.

    Rows are defendant type (Business / Consumer) and columns are plaintiff
    type (Business / Consumer). Margins (row/column totals) are included.
    """
    missing = df[["defendant_is_org", "plaintiff_is_org"]].isna().any(axis=1)
    if missing.any():
        n_missing = int(missing.sum())
        print(f"Dropping {n_missing} scenarios with missing party annotations")
        df = df.loc[~missing].copy()

    df["defendant_party"] = df["defendant_is_org"].map(PARTY_LABELS)
    df["plaintiff_party"] = df["plaintiff_is_org"].map(PARTY_LABELS)

    table = pd.crosstab(
        df["defendant_party"],
        df["plaintiff_party"],
        margins=True,
        margins_name="Total",
    )
    table = table.reindex(index=["Business", "Consumer", "Total"],
                          columns=["Business", "Consumer", "Total"])
    return table


def to_latex(counts: pd.DataFrame) -> str:
    """Render the 2x2 contingency table as a LaTeX ``table`` environment."""
    total = int(counts.loc["Total", "Total"])

    def cell(defendant: str, plaintiff: str) -> str:
        n = int(counts.loc[defendant, plaintiff])
        share = (n / total) if total else 0.0
        pct = 100.0 * share
        return f"{n} ({pct:.1f}\\%)"

    def margin(value: int) -> str:
        share = (value / total) if total else 0.0
        pct = 100.0 * share
        return f"\\textbf{{{value}}} ({pct:.1f}\\%)"

    T1 = "    "
    T2 = "        "
    lines: list[str] = []
    lines.append("\\begin{table}[!htbp]")
    lines.append(f"{T1}\\centering")
    lines.append(f"{T1}\\small")
    lines.append(f"{T1}\\setlength{{\\tabcolsep}}{{6pt}}")
    lines.append(f"{T1}\\begin{{tabular}}{{lccc}}")
    lines.append(f"{T2}\\toprule")
    lines.append(
        f"{T2} & \\multicolumn{{2}}{{c}}{{\\textbf{{Plaintiff}}}} & \\\\"
    )
    lines.append(f"{T2}\\cmidrule(lr){{2-3}}")
    lines.append(
        f"{T2}\\textbf{{Defendant}} & Business & Consumer & \\textbf{{Total}} \\\\"
    )
    lines.append(f"{T2}\\midrule")
    for defendant in ("Business", "Consumer"):
        row = [
            defendant,
            cell(defendant, "Business"),
            cell(defendant, "Consumer"),
            margin(int(counts.loc[defendant, "Total"])),
        ]
        lines.append(f"{T2}" + " & ".join(row) + " \\\\")
    lines.append(f"{T2}\\midrule")
    total_row = [
        "\\textbf{Total}",
        margin(int(counts.loc["Total", "Business"])),
        margin(int(counts.loc["Total", "Consumer"])),
        margin(total),
    ]
    lines.append(f"{T2}" + " & ".join(total_row) + " \\\\")
    lines.append(f"{T2}\\bottomrule")
    lines.append(f"{T1}\\end{{tabular}}")
    lines.append(
        f"{T1}\\caption{{Distribution of \\textsc{{FraudBench}} scenarios "
        "by defendant and plaintiff party types.}"
    )
    lines.append(f"{T1}\\label{{tab:party_types}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def run(
    input_path: str = "data/final/dataset_public.jsonl",
    output: Optional[str] = None,
    no_save: bool = False,
) -> pd.DataFrame:
    """Generate the party-type frequency table and optionally save LaTeX."""
    if output is None:
        output = str(
            Path(__file__).resolve().parents[4] / "paper" / "tables" / "party_types.tex"
        )

    df = load_scenarios(input_path)
    counts = compute_counts(df)

    print("\n" + "=" * 60)
    print("PARTY TYPE COUNTS (rows: defendant, columns: plaintiff)")
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
