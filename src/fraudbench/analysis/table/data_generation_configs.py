"""LaTeX table of model configurations used in agentic simulation / ToolEmu.

Backs ``analysis/tables/table_configs_data_generation.py``.

Documents only the ToolEmu agentic trajectory simulation roles (agent,
simulator, critiquer) and their LiteLLM defaults from
:data:`TRAJECTORY_LLM_DEFAULTS`.

The trajectory simulator default is duplicated here (see
``TRAJECTORY_SIMULATION_SCRIPT``) because importing that module would execute
CLI parsing at import time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import fraudbench.utils.globals as g
from ..registry import MODEL_DISPLAY_NAMES, normalize_model_id
from .configs import load_model_metadata

# Must stay aligned with ``data_generation/simulated_trajectories/simulate_trajectory.py``.
TRAJECTORY_SIMULATION_SCRIPT = "data_generation/simulated_trajectories/simulate_trajectory.py"
TRAJECTORY_LLM_DEFAULTS: dict[str, dict[str, Any]] = {
    "agent": {
        "temperature": 0.4,
        "reasoning_effort": "medium",
        "max_retries": 12,
        "request_timeout": 300,
    },
    "simulator": {
        "temperature": 0.4,
        "reasoning_effort": "medium",
        "max_retries": 12,
        "request_timeout": 300,
    },
    "critiquer": {
        "temperature": 0.4,
        "reasoning_effort": "medium",
        "max_retries": 12,
        "request_timeout": 300,
    },
}

TRAJECTORY_SIMULATOR_DEFAULT = "gemini/gemini-3.1-pro-preview"

NA = "---"
_DASHED_ENTRIES_NOTE = (
    r"Dashed entries (\texttt{---}) indicate that a configuration variable "
    r"was not available for that model."
)

DEFAULT_CAPTION = (
    "Configurations used for agentic simulation/ToolEmu. "
    "Non-agentic model configurations are documented in the evaluation "
    "configuration tables."
)
DEFAULT_LABEL = "tab:configs_data_generation"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _bare_model_id(litellm_id: str) -> str:
    """Return the bare model id (used for display-name and metadata lookup)."""
    return normalize_model_id(litellm_id)


def _display_name(litellm_id: str, *, meta_by_key: dict[str, dict]) -> str:
    """Pick the prettiest display name available for *litellm_id*."""
    bare = _bare_model_id(litellm_id)
    name = MODEL_DISPLAY_NAMES.get(bare)
    if name:
        return name
    meta = meta_by_key.get(bare, {})
    if meta.get("display_name"):
        return str(meta["display_name"])
    return bare


def _provider_for(litellm_id: str, *, meta_by_key: dict[str, dict]) -> str:
    """Best-effort provider label from ``models.jsonl`` (e.g. ``"OpenAI"``)."""
    meta = meta_by_key.get(_bare_model_id(litellm_id), {})
    providers = meta.get("inference_providers")
    if isinstance(providers, list) and providers:
        return " / ".join(str(p) for p in providers)
    if isinstance(providers, str) and providers:
        return providers
    return NA


def _fmt_temperature(val: Any) -> str:
    if val is None:
        return NA
    return f"{float(val):.1f}"


def _fmt_str(val: Any) -> str:
    if val is None:
        return NA
    return str(val)


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------

#: Column key, header pair. Order is the LaTeX render order.
COLUMNS: list[tuple[str, str]] = [
    ("role", "Pipeline Role"),
    ("provider", "Provider"),
    ("temperature", "Temperature"),
    ("reasoning_effort", "Reasoning Effort"),
]


def _trajectory_row(
    litellm_id: str,
    *,
    role_label: str,
    role_key: str,
    meta_by_key: dict[str, dict],
) -> dict[str, str]:
    """Row for a model used in trajectory simulation (TRAJECTORY_LLM_DEFAULTS)."""
    cfg = TRAJECTORY_LLM_DEFAULTS[role_key]
    return {
        "label": _display_name(litellm_id, meta_by_key=meta_by_key),
        "role": role_label,
        "provider": _provider_for(litellm_id, meta_by_key=meta_by_key),
        "temperature": _fmt_temperature(cfg.get("temperature")),
        "reasoning_effort": _fmt_str(cfg.get("reasoning_effort")),
    }


def build_data_generation_rows() -> list[dict[str, str]]:
    """Build ordered row dicts for the data-generation configuration table.

    The result has three ``\\midrule``-separated row groups:

    1. Trajectory: Agent (``MODELS_ADVANCED``).
    2. Trajectory: Simulator (single model, ``TRAJECTORY_SIMULATOR_DEFAULT``).
    3. Trajectory: Critiquer (``MODELS_ADVANCED``).

    Family-boundary rows are flagged via ``"is_last_in_group": True`` so the
    renderer can emit a ``\\midrule`` between groups.
    """
    meta_by_key = load_model_metadata()
    rows: list[dict[str, str]] = []

    advanced_pool = list(g.MODELS_ADVANCED)

    def _flag_last(group: list[dict[str, str]]) -> None:
        if not group:
            return
        for r in group[:-1]:
            r["is_last_in_group"] = False
        group[-1]["is_last_in_group"] = True

    agent_group = [
        _trajectory_row(m, role_label="Trajectory: Agent",
                        role_key="agent", meta_by_key=meta_by_key)
        for m in advanced_pool
    ]
    _flag_last(agent_group)
    rows.extend(agent_group)

    simulator_group = [
        _trajectory_row(TRAJECTORY_SIMULATOR_DEFAULT,
                        role_label="Trajectory: Simulator",
                        role_key="simulator", meta_by_key=meta_by_key),
    ]
    _flag_last(simulator_group)
    rows.extend(simulator_group)

    critiquer_group = [
        _trajectory_row(m, role_label="Trajectory: Critiquer",
                        role_key="critiquer", meta_by_key=meta_by_key)
        for m in advanced_pool
    ]
    _flag_last(critiquer_group)
    rows.extend(critiquer_group)

    # Last row should not draw a midrule after itself.
    if rows:
        rows[-1]["is_last_in_group"] = False
    return rows


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------

def _tabularx_colspec(n_data_cols: int) -> str:
    return (
        r">{\raggedright\arraybackslash}X"
        + r"*{"
        + str(n_data_cols)
        + r"}{>{\centering\arraybackslash}X}"
    )


def to_latex_data_generation_configs(
    rows: list[dict[str, str]],
    caption: str = DEFAULT_CAPTION,
    label: str = DEFAULT_LABEL,
) -> str:
    """Render row dicts to a styled ``tabularx`` LaTeX block."""
    col_keys = [k for k, _ in COLUMNS]
    col_headers = [h for _, h in COLUMNS]
    n_data = len(col_keys)

    T = "    "
    lines: list[str] = []
    lines.append("\\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabularx}}{{\\textwidth}}{{{_tabularx_colspec(n_data)}}}")
    lines.append(f"{T}\\toprule")

    header_parts = ["\\textbf{Model}"] + [f"\\textbf{{{h}}}" for h in col_headers]
    lines.append(f"{T}" + " & ".join(header_parts) + " \\\\")
    lines.append(f"{T}\\midrule")

    last_idx = len(rows) - 1
    for i, row in enumerate(rows):
        parts = [row["label"]] + [row[k] for k in col_keys]
        lines.append(f"{T}" + " & ".join(parts) + " \\\\")
        if row.get("is_last_in_group") and i < last_idx:
            lines.append(f"{T}\\midrule")

    lines.append(f"{T}\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append(r"\vspace{2pt}")
    lines.append(r"\begin{minipage}{\textwidth}")
    lines.append(r"\footnotesize")
    lines.append(r"\textit{Note.} " + _DASHED_ENTRIES_NOTE + r"\par")
    lines.append(r"\end{minipage}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry points
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


def run_data_generation_configs(
    output: Optional[str] = None,
    no_save: bool = False,
    caption: str = DEFAULT_CAPTION,
    label: str = DEFAULT_LABEL,
) -> list[dict[str, str]]:
    """Generate the data-generation configuration table and optionally save it.

    Args:
        output: Destination ``.tex`` file path
            (default: ``paper/tables/configs_data_generation.tex``).
        no_save: If True, print to stdout instead of writing a file.
        caption: caption text inserted via ``\\caption{...}``.
        label: label inserted via ``\\label{...}``.
    """
    if output is None:
        output = str(
            Path(__file__).resolve().parents[4]
            / "paper"
            / "tables"
            / "configs_data_generation.tex"
        )
    rows = build_data_generation_rows()
    latex = to_latex_data_generation_configs(rows, caption=caption, label=label)
    _save_or_print(latex, output, no_save, "data generation configs table")
    return rows
