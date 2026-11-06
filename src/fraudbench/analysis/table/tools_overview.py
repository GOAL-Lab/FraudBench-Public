"""LaTeX table generator for the agentic-simulation toolkit overview.

Backs ``analysis/tables/table_tools_overview.py``.

Reads the registered virtual toolkits defined in
:mod:`fraudbench.data_generation.toolemu.tools.core_virtual_tools` and emits an
``xltabular`` snippet listing each toolkit's display name, its method names,
and a short description.

The output is a self-contained ``xltabular`` (longtable+tabularx) block that
spans page boundaries automatically. It carries its own ``\\caption`` and
``\\label`` and is meant to be included verbatim (e.g. via ``\\include{}``).

Requires ``\\usepackage{xltabular}`` in the main preamble.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# Importing the package registers all @register_toolkit() decorated classes.
from fraudbench.data_generation.toolemu.tools import (  # noqa: F401  (side-effect)
    core_virtual_tools,
)
from fraudbench.data_generation.toolemu.tools.register import get_toolkit_dict


# ---------------------------------------------------------------------------
# Display tweaks
# ---------------------------------------------------------------------------

# Long camelCase toolkit names that need a soft-hyphen hint to break nicely
# in a narrow tabularx column. Add entries here if more wide names appear.
_NAME_HYPHENATION_HINTS: dict[str, str] = {
    "InventoryManagementSystem": r"InventoryManagement\-System",
}

# CamelCase boundary: a lowercase letter or digit immediately followed by an
# uppercase letter.  Used to insert soft-hyphen breakpoints inside long method
# identifiers so they can still wrap inside a narrow tabularx column.
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _format_toolkit_name(name: str) -> str:
    """Return the toolkit name with optional LaTeX soft-hyphen breaks."""
    return _NAME_HYPHENATION_HINTS.get(name, name)


def _strip_toolkit_prefix(method_name: str, toolkit_name: str) -> str:
    """Drop the toolkit name prefix from a method name.

    Example: ``GoogleCalendarReadEvents`` with toolkit ``GoogleCalendar``
    becomes ``ReadEvents``.  If the method does not start with the prefix,
    or stripping would yield an empty string, the original name is returned.
    """
    if method_name.startswith(toolkit_name):
        stripped = method_name[len(toolkit_name):]
        if stripped:
            return stripped
    return method_name


def _format_method_name(name: str) -> str:
    """Add LaTeX soft-hyphen breakpoints at camelCase boundaries.

    Inserts ``\\-`` between lowercase/digit -> uppercase transitions so that
    long camelCase identifiers can wrap inside a narrow tabularx column.
    The discretionary hyphen is invisible unless LaTeX breaks the word at
    that point (matching the style used for ``_NAME_HYPHENATION_HINTS``).
    """
    return _CAMEL_BOUNDARY_RE.sub(r"\\-", name)


def _clean_description(text: str) -> str:
    """Lightly normalise a toolkit description for table display.

    * Collapses internal whitespace.
    * Ensures the string ends with a single period.
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return cleaned
    if not cleaned.endswith("."):
        cleaned += "."
    return cleaned


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------

def build_tools_rows(
    use_human_description: bool = False,
) -> list[dict]:
    """Build ordered row data for the toolkit overview table.

    Args:
        use_human_description: if True, use ``description_for_human`` (terse,
            one short sentence each).  Otherwise, use ``description_for_model``
            (more informative, default), which mirrors what the agent sees.

    Returns a list of dicts, one per registered toolkit, with keys:

    * ``"toolkit"``     — display name (with optional LaTeX hyphen hints)
    * ``"methods"``     — comma-joined list of method names
    * ``"description"`` — short description string
    """
    rows: list[dict] = []
    desc_attr = "description_for_human" if use_human_description else "description_for_model"

    for toolkit_name, toolkit_cls in get_toolkit_dict().items():
        # Instantiate the toolkit to materialise its tool objects (the
        # canonical access pattern used elsewhere in the codebase).
        toolkit = toolkit_cls()
        method_names = [
            _format_method_name(_strip_toolkit_prefix(tool.name, toolkit_name))
            for tool in toolkit.tools
        ]
        description = getattr(toolkit_cls, desc_attr, "") or ""

        rows.append(
            {
                "toolkit": _format_toolkit_name(toolkit_name),
                "methods": ", ".join(method_names),
                "description": _clean_description(description),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------

DEFAULT_CAPTION = "Toolkits available during agentic simulation."
# Label matches the basename of the resulting ``.tex`` file (no path / extension).
DEFAULT_LABEL = "tab:tools_overview"


def to_latex_tools_overview(
    rows: list[dict],
    caption: str = DEFAULT_CAPTION,
    label: str = DEFAULT_LABEL,
) -> str:
    """Render row dicts to a self-contained ``xltabular`` LaTeX block.

    Uses ``xltabular`` (longtable + tabularx) so the table can span multiple
    pages while still wrapping the ``Methods`` and ``Description`` columns to
    fit ``\\linewidth``.  The whole environment is wrapped in
    ``\\begingroup\\small ... \\endgroup`` so the smaller font applies to the
    table without breaking ``longtable``'s alignment context (``\\caption``
    must be the first thing inside the body of an ``xltabular`` -- inserting
    a ``\\small`` token before it triggers the ``Misplaced \\noalign`` /
    ``Misplaced \\omit`` errors raised by ``\\caption``'s internal
    ``\\noalign``).  The caption/label are placed inside the environment
    using ``longtable``-style ``\\endfirsthead`` / ``\\endhead`` /
    ``\\endfoot`` blocks so the header repeats on each page.
    """
    T = "    "
    lines: list[str] = []

    header_row = (
        f"{T}\\textbf{{Toolkit}} & \\textbf{{Methods}} & \\textbf{{Description}} \\\\"
    )

    lines.append(r"\begingroup\small")
    lines.append(r"\begin{xltabular}{\linewidth}{lXX}")
    lines.append(f"{T}\\caption{{{caption}}}\\label{{{label}}} \\\\")
    lines.append(f"{T}\\toprule")
    lines.append(header_row)
    lines.append(f"{T}\\midrule")
    lines.append(f"{T}\\endfirsthead")
    lines.append(f"{T}\\multicolumn{{3}}{{l}}{{\\textit{{(continued from previous page)}}}} \\\\")
    lines.append(f"{T}\\toprule")
    lines.append(header_row)
    lines.append(f"{T}\\midrule")
    lines.append(f"{T}\\endhead")
    lines.append(f"{T}\\midrule")
    lines.append(f"{T}\\multicolumn{{3}}{{r}}{{\\textit{{(continued on next page)}}}} \\\\")
    lines.append(f"{T}\\endfoot")
    lines.append(f"{T}\\bottomrule")
    lines.append(f"{T}\\endlastfoot")

    last_idx = len(rows) - 1
    for i, row in enumerate(rows):
        parts = [row["toolkit"], row["methods"], row["description"]]
        lines.append(f"{T}" + " & ".join(parts) + " \\\\")
        if i < last_idx:
            lines.append(f"{T}\\addlinespace")

    lines.append(r"\end{xltabular}")
    lines.append(r"\endgroup")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _save_or_print(latex: str, output: str, no_save: bool, label: str) -> None:
    if no_save:
        print(f"\n{'=' * 60}\n{label.upper()}\n{'=' * 60}\n")
        print(latex)
    else:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(latex, encoding="utf-8")
        print(f"Saved {label} to: {out}")


def run_tools_overview(
    output: Optional[str] = None,
    no_save: bool = False,
    use_human_description: bool = False,
    caption: str = DEFAULT_CAPTION,
    label: str = DEFAULT_LABEL,
) -> list[dict]:
    """Generate the toolkit overview table and optionally save it.

    Args:
        output: destination ``.tex`` file path
            (default: ``paper/tables/tools_overview.tex`` relative to repo root).
        no_save: if True, print to stdout instead of writing a file.
        use_human_description: prefer ``description_for_human`` over
            ``description_for_model``.
        caption: caption text inserted via ``\\caption{...}``.
        label: label inserted via ``\\label{...}``.
    """
    if output is None:
        output = str(
            Path(__file__).resolve().parents[4]
            / "paper"
            / "tables"
            / "tools_overview.tex"
        )

    rows = build_tools_rows(use_human_description=use_human_description)
    latex = to_latex_tools_overview(rows, caption=caption, label=label)
    _save_or_print(latex, output, no_save, "tools overview table")
    print(f"\n{len(rows)} toolkits with "
          f"{sum(len(r['methods'].split(', ')) for r in rows)} methods total.")
    return rows
