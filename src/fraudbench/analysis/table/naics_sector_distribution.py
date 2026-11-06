"""LaTeX table comparing Scenario and industry-size ranks by NAICS sector.

Scenario counts come from the top-level ``naics_sector`` field in
``data/final/dataset_public.jsonl``. Industry ranks come from the Bureau of
Economic Analysis gross-output table. Because every industry's gross-output
share has the same denominator, ranking gross-output shares is equivalent to
ranking their gross-output levels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SCENARIOS = str(_REPO_ROOT / "data" / "final" / "dataset_public.jsonl")
DEFAULT_GROSS_OUTPUT = str(
    _REPO_ROOT
    / "data"
    / "annotations"
    / "industry_categories"
    / "gross_output_by_industry.csv"
)
DEFAULT_YEAR = 2024

NAICS_SECTORS = (
    "Agriculture, Forestry, Fishing and Hunting",
    "Mining, Quarrying, and Oil and Gas Extraction",
    "Utilities",
    "Construction",
    "Manufacturing",
    "Wholesale Trade",
    "Retail Trade",
    "Transportation and Warehousing",
    "Information",
    "Finance and Insurance",
    "Real Estate and Rental and Leasing",
    "Professional, Scientific, and Technical Services",
    "Management of Companies and Enterprises",
    "Administrative and Support and Waste Management and Remediation Services",
    "Educational Services",
    "Health Care and Social Assistance",
    "Arts, Entertainment, and Recreation",
    "Accommodation and Food Services",
    "Other Services (except Public Administration)",
    "Public Administration",
)

BEA_TO_NAICS_SECTOR = {
    "Agriculture, forestry, fishing, and hunting": (
        "Agriculture, Forestry, Fishing and Hunting"
    ),
    "Mining": "Mining, Quarrying, and Oil and Gas Extraction",
    "Utilities": "Utilities",
    "Construction": "Construction",
    "Manufacturing": "Manufacturing",
    "Wholesale trade": "Wholesale Trade",
    "Retail trade": "Retail Trade",
    "Transportation and warehousing": "Transportation and Warehousing",
    "Information": "Information",
    "Finance and insurance": "Finance and Insurance",
    "Real estate and rental and leasing": "Real Estate and Rental and Leasing",
    "Professional, scientific, and technical services": (
        "Professional, Scientific, and Technical Services"
    ),
    "Management of companies and enterprises": (
        "Management of Companies and Enterprises"
    ),
    "Administrative and waste management services": (
        "Administrative and Support and Waste Management and Remediation Services"
    ),
    "Educational services": "Educational Services",
    "Health care and social assistance": "Health Care and Social Assistance",
    "Arts, entertainment, and recreation": "Arts, Entertainment, and Recreation",
    "Accommodation and food services": "Accommodation and Food Services",
    "Other services, except government": (
        "Other Services (except Public Administration)"
    ),
    # The BEA table reports Government rather than NAICS Public Administration.
    "Government": "Public Administration",
}

GOVERNMENT_PROXY_SECTOR = "Public Administration"


def load_scenarios(input_path: str) -> pd.DataFrame:
    """Load scenario-level data and retain one row per ``summary_id``."""
    path = Path(input_path)
    if path.suffix == ".jsonl":
        scenarios = pd.read_json(path, lines=True)
    elif path.suffix == ".parquet":
        scenarios = pd.read_parquet(path)
    elif path.suffix == ".csv":
        scenarios = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported scenario file format: {path.suffix}")

    required = {"summary_id", "naics_sector"}
    missing_columns = sorted(required - set(scenarios.columns))
    if missing_columns:
        raise ValueError(
            "Scenario data is missing required columns: "
            + ", ".join(missing_columns)
        )

    scenarios = scenarios.drop_duplicates(subset="summary_id").reset_index(drop=True)
    if scenarios["naics_sector"].isna().any():
        n_missing = int(scenarios["naics_sector"].isna().sum())
        raise ValueError(f"Found {n_missing} scenarios without a NAICS sector")

    unexpected = sorted(set(scenarios["naics_sector"]) - set(NAICS_SECTORS))
    if unexpected:
        raise ValueError(f"Unknown NAICS sectors: {', '.join(unexpected)}")

    print(f"Loaded {len(scenarios)} scenarios from {input_path}")
    return scenarios


def load_gross_output_metrics(
    gross_output_path: str,
    year: int = DEFAULT_YEAR,
) -> pd.DataFrame:
    """Load BEA gross-output shares and descending ranks for ``year``."""
    path = Path(gross_output_path)
    gross_output = pd.read_csv(path, header=3)
    if len(gross_output.columns) < 2:
        raise ValueError(f"Malformed BEA gross-output table: {gross_output_path}")

    industry_column = gross_output.columns[1]
    year_column = str(year)
    if year_column not in gross_output.columns:
        available_years = sorted(
            column for column in gross_output.columns if str(column).isdigit()
        )
        raise ValueError(
            f"Gross output has no {year_column} column; available years: "
            + ", ".join(available_years)
        )

    gross_output[industry_column] = (
        gross_output[industry_column].astype("string").str.strip()
    )
    all_industries = gross_output.loc[
        gross_output[industry_column].eq("All industries"), year_column
    ]
    if len(all_industries) != 1:
        raise ValueError(
            "BEA gross-output table must contain exactly one 'All industries' row"
        )
    gross_output_total = float(pd.to_numeric(all_industries.iloc[0], errors="raise"))
    if gross_output_total <= 0:
        raise ValueError("BEA gross output for all industries must be positive")

    selected = gross_output.loc[
        gross_output[industry_column].isin(BEA_TO_NAICS_SECTOR),
        [industry_column, year_column],
    ].copy()
    selected["naics_sector"] = selected[industry_column].map(BEA_TO_NAICS_SECTOR)
    selected["gross_output"] = pd.to_numeric(selected[year_column], errors="raise")

    duplicates = sorted(
        selected.loc[
            selected["naics_sector"].duplicated(keep=False), "naics_sector"
        ].unique()
    )
    if duplicates:
        raise ValueError(
            "Duplicate industries in BEA gross-output table: " + ", ".join(duplicates)
        )

    missing_sectors = sorted(set(NAICS_SECTORS) - set(selected["naics_sector"]))
    if missing_sectors:
        raise ValueError(
            "BEA gross-output table is missing mapped sectors: "
            + ", ".join(missing_sectors)
        )

    selected["gross_output_rank"] = (
        selected["gross_output"].rank(method="min", ascending=False).astype(int)
    )
    selected["gross_output_share"] = selected["gross_output"] / gross_output_total
    metrics = selected.set_index("naics_sector")[
        ["gross_output_share", "gross_output_rank"]
    ]
    return metrics.reindex(NAICS_SECTORS)


def compute_distribution(
    scenarios: pd.DataFrame,
    gross_output_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Compute scenario counts and dataset and gross-output shares and ranks."""
    counts = (
        scenarios["naics_sector"]
        .value_counts()
        .reindex(NAICS_SECTORS, fill_value=0)
        .astype(int)
    )
    total = int(counts.sum())
    if total == 0:
        raise ValueError("Cannot compute a sector distribution from an empty dataset")

    required_metrics = {"gross_output_share", "gross_output_rank"}
    missing_metrics = sorted(required_metrics - set(gross_output_metrics.columns))
    if missing_metrics:
        raise ValueError(
            "Gross-output data is missing required columns: "
            + ", ".join(missing_metrics)
        )
    missing_sectors = sorted(
        set(NAICS_SECTORS) - set(gross_output_metrics.dropna().index)
    )
    if missing_sectors:
        raise ValueError(
            "Missing gross-output data for sectors: " + ", ".join(missing_sectors)
        )

    distribution = pd.DataFrame(
        {
            "scenario_count": counts,
            "dataset_share": counts / total,
            "gross_output_share": gross_output_metrics["gross_output_share"],
            "scenario_rank": counts.rank(method="min", ascending=False).astype(int),
            "gross_output_rank": gross_output_metrics["gross_output_rank"].astype(int),
        }
    )
    distribution.index.name = "naics_sector"
    distribution = (
        distribution.reset_index()
        .sort_values(
            ["scenario_count", "gross_output_rank", "naics_sector"],
            ascending=[False, True, True],
        )
        .set_index("naics_sector")
    )
    return distribution


def _escape_latex(value: str) -> str:
    """Escape LaTeX-special characters in a table cell."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def to_latex(distribution: pd.DataFrame, year: int = DEFAULT_YEAR) -> str:
    """Render the NAICS-sector distribution as a LaTeX ``table*``."""
    total = int(distribution["scenario_count"].sum())
    T1 = "    "
    T2 = "        "
    lines = [
        "\\begin{table*}[!htbp]",
        f"{T1}\\centering",
        f"{T1}\\small",
        f"{T1}\\renewcommand{{\\arraystretch}}{{1.15}}",
        f"{T1}\\setlength{{\\tabcolsep}}{{6pt}}",
        (
            f"{T1}\\begin{{tabularx}}{{\\textwidth}}"
            "{>{\\raggedright\\arraybackslash}Xrrrrr}"
        ),
        f"{T2}\\toprule",
        (
            f"{T2}\\textbf{{NAICS Sector}} & \\textbf{{Scenarios}} & "
            "\\textbf{Dataset Share} & "
            "\\shortstack{\\textbf{Gross Output}\\\\\\textbf{Share}} & "
            "\\shortstack{\\textbf{Scenario}\\\\\\textbf{Rank}} & "
            f"\\shortstack{{\\textbf{{{year} Gross Output}}"
            "\\\\\\textbf{rank}} \\\\"
        ),
        f"{T2}\\midrule",
    ]

    for sector, row in distribution.iterrows():
        sector_label = _escape_latex(str(sector))
        gross_output_rank = str(int(row["gross_output_rank"]))
        if sector == GOVERNMENT_PROXY_SECTOR:
            gross_output_rank += r"\textsuperscript{*}"
        lines.append(
            f"{T2}{sector_label} & {int(row['scenario_count'])} & "
            f"{100.0 * float(row['dataset_share']):.1f}\\% & "
            f"{100.0 * float(row['gross_output_share']):.1f}\\% & "
            f"{int(row['scenario_rank'])} & {gross_output_rank} \\\\"
        )

    lines.extend(
        [
            f"{T2}\\midrule",
            (
                f"{T2}\\textbf{{Total}} & \\textbf{{{total}}} & "
                "\\textbf{100.0\\%} & \\textbf{100.0\\%} & "
                "\\textemdash & \\textemdash \\\\"
            ),
            f"{T2}\\bottomrule",
            f"{T1}\\end{{tabularx}}",
            (
                f"{T1}\\caption{{Distribution of \\textsc{{FraudBench}} scenarios "
                f"across NAICS sectors, compared with {year} U.S. industry ranks by share of gross output.}}"
            ),
            f"{T1}\\label{{tab:naics_sector_distribution}}",
            (
                f"{T1}\\vspace{{2pt}}\\parbox{{\\textwidth}}{{\\footnotesize "
                "\\textsuperscript{*}The Bureau of Economic Analysis reports "
                "Government rather than the narrower NAICS Public "
                "Administration sector; its Government aggregate is used here "
                "as a proxy.}"
            ),
            "\\end{table*}",
        ]
    )
    return "\n".join(lines)


def run(
    input_path: str = DEFAULT_SCENARIOS,
    gross_output_path: str = DEFAULT_GROSS_OUTPUT,
    year: int = DEFAULT_YEAR,
    output: Optional[str] = None,
    no_save: bool = False,
) -> pd.DataFrame:
    """Generate the sector distribution table and optionally save its LaTeX."""
    if output is None:
        output = str(
            _REPO_ROOT / "paper" / "tables" / "naics_sector_distribution.tex"
        )

    scenarios = load_scenarios(input_path)
    gross_output_metrics = load_gross_output_metrics(gross_output_path, year=year)
    distribution = compute_distribution(scenarios, gross_output_metrics)

    print("\n" + "=" * 60)
    print(f"NAICS SECTOR DISTRIBUTION AND {year} GROSS-OUTPUT RANKS")
    print("=" * 60)
    print(distribution.to_string())

    latex = to_latex(distribution, year=year)
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

    return distribution
