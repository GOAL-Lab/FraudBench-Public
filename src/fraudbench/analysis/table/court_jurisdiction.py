"""LaTeX table of scenario counts by jurisdiction and court level.

The table reads top-level ``jurisdiction`` and ``level`` fields directly from
``data/final/dataset_public.jsonl``. ``Federal`` is treated as federal
jurisdiction; every other jurisdiction label is classified as state.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..scenario_courts import DEFAULT_SCENARIOS, load_scenario_courts


JURISDICTION_ORDER = ["Federal", "State"]
LEVEL_ORDER = ["Trial", "Appeals", "Supreme", "Unspecified"]
LEVEL_LABELS = {"trial": "Trial", "appeal": "Appeals", "supreme": "Supreme"}


def compute_counts(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate embedded jurisdiction and level fields by scenario."""
    scenarios = scenarios.copy()
    scenarios["jurisdiction_type"] = scenarios["jurisdiction"].eq("Federal").map(
        {True: "Federal", False: "State"}
    )
    scenarios["court_level"] = scenarios["level"].map(LEVEL_LABELS).fillna(
        "Unspecified"
    )

    unexpected_levels = sorted(
        set(scenarios["court_level"].unique()) - set(LEVEL_ORDER)
    )
    if unexpected_levels:
        raise ValueError(f"Unknown court levels: {', '.join(unexpected_levels)}")

    counts = pd.crosstab(
        scenarios["jurisdiction_type"],
        scenarios["court_level"],
        margins=True,
        margins_name="Total",
    )
    counts = counts.reindex(
        index=JURISDICTION_ORDER + ["Total"],
        columns=LEVEL_ORDER + ["Total"],
        fill_value=0,
    )
    counts.index.name = "jurisdiction"
    counts.columns.name = "level"
    return counts.astype(int)


def to_latex(counts: pd.DataFrame) -> str:
    """Render the jurisdiction-by-level counts as a LaTeX table."""
    include_unspecified = int(counts.loc["Total", "Unspecified"]) > 0
    levels = LEVEL_ORDER if include_unspecified else LEVEL_ORDER[:-1]
    column_spec = "l" + "r" * (len(levels) + 1)
    T1 = "    "
    T2 = "        "

    lines = [
        "\\begin{table}[!htbp]",
        f"{T1}\\centering",
        f"{T1}\\small",
        f"{T1}\\setlength{{\\tabcolsep}}{{7pt}}",
        f"{T1}\\begin{{tabular}}{{{column_spec}}}",
        f"{T2}\\toprule",
        f"{T2}\\textbf{{Jurisdiction}} & "
        + " & ".join(f"\\textbf{{{level}}}" for level in levels)
        + " & \\textbf{Total} \\\\",
        f"{T2}\\midrule",
    ]
    for jurisdiction in JURISDICTION_ORDER:
        values = [str(int(counts.loc[jurisdiction, level])) for level in levels]
        values.append(f"\\textbf{{{int(counts.loc[jurisdiction, 'Total'])}}}")
        lines.append(f"{T2}{jurisdiction} & " + " & ".join(values) + " \\\\")

    total_values = [
        f"\\textbf{{{int(counts.loc['Total', level])}}}" for level in levels
    ]
    total_values.append(f"\\textbf{{{int(counts.loc['Total', 'Total'])}}}")
    lines.extend(
        [
            f"{T2}\\midrule",
            f"{T2}\\textbf{{Total}} & " + " & ".join(total_values) + " \\\\",
            f"{T2}\\bottomrule",
            f"{T1}\\end{{tabular}}",
            f"{T1}\\caption{{Distribution of \\textsc{{FraudBench}} scenarios "
            "by court jurisdiction and level.}",
            f"{T1}\\label{{tab:court_jurisdiction}}",
        ]
    )
    if include_unspecified:
        lines.append(
            f"{T1}\\vspace{{2pt}}\\parbox{{0.9\\linewidth}}{{\\footnotesize "
            "Note: Unspecified denotes a court whose level is missing from the "
            "court metadata.}}"
        )
    lines.append("\\end{table}")
    return "\n".join(lines)


def run(
    input_path: str = DEFAULT_SCENARIOS,
    output: Optional[str] = None,
    no_save: bool = False,
) -> pd.DataFrame:
    """Generate the table from court fields embedded in scenario data."""
    if output is None:
        output = str(
            Path(__file__).resolve().parents[4]
            / "paper"
            / "tables"
            / "court_jurisdiction.tex"
        )

    scenarios = load_scenario_courts(input_path)
    counts = compute_counts(scenarios)

    print("\n" + "=" * 60)
    print("SCENARIO COUNTS BY COURT JURISDICTION AND LEVEL")
    print("=" * 60)
    print(counts.to_string())

    latex = to_latex(counts)
    if no_save:
        print("\n" + "=" * 60)
        print("LATEX TABLE")
        print("=" * 60)
        print(latex)
    else:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex + "\n", encoding="utf-8")
        print(f"\nSaved LaTeX table to: {output_path}")
    return counts
