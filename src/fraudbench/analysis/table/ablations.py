"""LaTeX table generators for ablation study configuration parameters.

**Data sources (in priority order):**

1. A pre-processed ``.parquet`` file (e.g. ``data/final/obs_data.parquet``)
   produced by ``analysis/process.py``.  Rows are filtered by
   ``run_category`` (``"temperature_ablation"`` / ``"reasoning_ablation"``) and
   aggregated to one row per model.

2. Raw Inspect AI ``.eval`` log files from
   ``logs/ablations/{reasoning,temperature}/`` — used as fallback when the
   parquet is not available or contains no rows for that run_category.

In both cases the ablation values are compared against the base configuration
from :mod:`fraudbench.utils.globals` to produce a ``tabularx`` LaTeX snippet.
"""
import json
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

from fraudbench.utils.globals import MODEL_CONFIGS
from ..registry import (
    MODEL_ORDER,
    REASONING_MODELS,
    reasoning_label,
    REASONING_SUBSCRIPT_CAPTION,
    REASONING_SYMBOL,
    normalize_model_id,
)

from ..common import (
    available_parquet_columns,
    ensure_model_id_column,
    filter_ai_settings,
    get_model_display_name,
    get_model_family,
)


def _row_label(model: str, *, all_reasoning_subscript: bool) -> str:
    """Display alias with optional :data:`REASONING_SYMBOL` suffix.

    *model* may be a bare or LiteLLM id; the bare id is used to look up the
    main-protocol reasoning state. Set ``all_reasoning_subscript=True`` for
    reasoning-ablation rows, where reasoning was enabled for every model.
    """
    bare = normalize_model_id(model)
    name = get_model_display_name(bare)
    if all_reasoning_subscript:
        return f"{name} {REASONING_SYMBOL}"
    return reasoning_label(name, bare)


def _bare_to_litellm(bare: str) -> Optional[str]:
    """Return the active LiteLLM id in :data:`MODEL_CONFIGS` for *bare*, if any."""
    for litellm_id in MODEL_CONFIGS:
        if normalize_model_id(litellm_id) == bare:
            return litellm_id
    return None


DEFAULT_LOGS_DIR = "logs/ablations"

_DEFAULT_PARQUET = str(
    Path(__file__).resolve().parents[4] / "data" / "final" / "obs_data.parquet"
)

NA = "---"
_DASHED_ENTRIES_NOTE = (
    r"Dashed entries (\texttt{---}) indicate that a configuration variable "
    r"was not available for that model."
)


# ---------------------------------------------------------------------------
# Parquet loading
# ---------------------------------------------------------------------------

_ABLATION_RUN_CATEGORY: dict[str, str] = {
    "temperature": "temperature_ablation",
    "reasoning":   "reasoning_ablation",
}

# Parquet column → key used in the ablation_cfg dict consumed by _ablation_value
_PARQUET_ABLATION_COL_MAP: dict[str, str] = {
    "temperature":      "temperature",
    "reasoning_effort": "reasoning_effort",  # also covers 'effort' via _ablation_value
    "reasoning_budget": "reasoning_tokens",  # parquet stores as reasoning_budget
    "max_tokens":       "max_tokens",
}


def load_ablation_configs_from_parquet(
    parquet_path: str,
    ablation_type: str,
) -> dict[str, dict]:
    """Return ``{model_id: ablation_cfg}`` from a pre-processed parquet.

    Filters by the ``run_category`` that corresponds to *ablation_type*
    (``"temperature"`` → ``"temperature_ablation"``,
    ``"reasoning"`` → ``"reasoning_ablation"``).

    The returned dicts use the same key names expected by
    :func:`_ablation_value` so they are a drop-in replacement for configs
    loaded from raw ``.eval`` log files.

    Returns an empty dict when the parquet is missing or has no matching rows.
    """
    run_category = _ABLATION_RUN_CATEGORY.get(ablation_type, "")
    if not run_category:
        return {}

    path = Path(parquet_path)
    if not path.exists():
        return {}

    needed = ["model", "run_category", "setting"] + list(_PARQUET_ABLATION_COL_MAP)
    df = pd.read_parquet(
        path,
        columns=available_parquet_columns(path, needed),
    )

    available = [c for c in needed if c in df.columns]
    df = df[available].copy()

    if "run_category" not in df.columns or "model" not in df.columns:
        return {}

    df = df[df["run_category"] == run_category]
    df = filter_ai_settings(df, verbose=False)
    df = df[df["model"].notna() & (df["model"] != "human")]
    df = ensure_model_id_column(df)

    if df.empty:
        return {}

    configs: dict[str, dict] = {}
    for model_id, group in df.groupby("model_id"):
        cfg: dict = {}
        for pq_col, cfg_key in _PARQUET_ABLATION_COL_MAP.items():
            if pq_col not in group.columns:
                continue
            series = group[pq_col].dropna()
            if series.empty:
                continue
            cfg[cfg_key] = series.mode().iloc[0]
        if cfg:
            configs[str(model_id)] = cfg

    print(
        f"Loaded ablation configs from parquet for {len(configs)} model(s) "
        f"(run_category='{run_category}'): {path}"
    )
    return configs


# ---------------------------------------------------------------------------
# Log reading (fallback)
# ---------------------------------------------------------------------------

def _read_eval_start(path: Path) -> dict:
    """Read ``_journal/start.json`` from an Inspect AI ``.eval`` ZIP file."""
    with zipfile.ZipFile(path) as zf:
        with zf.open("_journal/start.json") as f:
            return json.load(f)


def load_ablation_configs_from_logs(logs_dir: str, ablation_type: str) -> dict[str, dict]:
    """Return ``{model_id: model_generate_config}`` for all ``.eval`` files.

    A model may appear in multiple log files (different evaluation settings).
    All files for the same model should carry identical generate configs, so
    last-seen wins with no loss of information.
    """
    dir_ = Path(logs_dir) / ablation_type
    if not dir_.exists():
        print(f"Warning: directory not found: {dir_}")
        return {}

    model_configs: dict[str, dict] = {}
    for p in sorted(dir_.glob("*.eval")):
        try:
            start = _read_eval_start(p)
            ev = start.get("eval", {})
            model = ev.get("model", "")
            cfg = ev.get("model_generate_config", {})
            if model:
                # Key by bare id so the dict matches the parquet path and
                # the bare-id-keyed registry (MODEL_ORDER, etc.).
                model_configs[normalize_model_id(model)] = cfg
        except Exception as exc:
            print(f"Warning: skipping {p.name} ({exc})")

    print(f"Found {len(model_configs)} unique models in {ablation_type!r} ablation logs")
    return model_configs


def load_ablation_configs(
    ablation_type: str,
    logs_dir: str = DEFAULT_LOGS_DIR,
    parquet_path: Optional[str] = None,
) -> dict[str, dict]:
    """Load ablation configs, preferring the parquet over raw log files.

    Tries the parquet first; if it returns no results for *ablation_type*,
    falls back to scanning raw ``.eval`` files in *logs_dir*.
    """
    pq = parquet_path if parquet_path is not None else _DEFAULT_PARQUET
    configs = load_ablation_configs_from_parquet(pq, ablation_type)
    if configs:
        return configs
    print(
        f"No parquet data for '{ablation_type}' ablation — "
        f"falling back to log files in {logs_dir}"
    )
    return load_ablation_configs_from_logs(logs_dir, ablation_type)


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------

def _fmt(val) -> str:
    """Format a config value as a clean string for table display."""
    if val is None:
        return NA
    if isinstance(val, bool):
        return "No" if not val else "Yes"
    # Strip trailing .0 from whole-number floats (e.g. 50000.0 → "50000")
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return str(val)


def _base_value(model: str, key: str) -> str:
    """Return the formatted base config value for *model* and parameter *key*.

    *model* may be a bare model id (e.g. ``"claude-opus-4-5-20251101"``) or a
    full LiteLLM id (e.g. ``"anthropic/claude-opus-4-5-20251101"``). Bare ids
    are resolved against :data:`MODEL_CONFIGS` via the active LiteLLM id.

    The special key ``"_reasoning_effort"`` is resolved against both the
    ``reasoning_effort`` and ``effort`` key names used by different providers.
    """
    if model in MODEL_CONFIGS:
        base = MODEL_CONFIGS[model]
    else:
        litellm_id = _bare_to_litellm(model)
        if litellm_id is None:
            return NA
        base = MODEL_CONFIGS[litellm_id]

    if key == "_reasoning_effort":
        for k in ("reasoning_effort", "effort"):
            if k in base:
                return _fmt(base[k])
        return NA

    if key == "temperature" and key in base:
        val = base.get(key)
        return f"{float(val):.1f}" if val is not None else NA
    return _fmt(base.get(key)) if key in base else NA


def _ablation_value(ablation_cfg: dict, key: str) -> str:
    """Return the formatted ablation config value for parameter *key*."""
    if key == "_reasoning_effort":
        for k in ("reasoning_effort", "effort"):
            if k in ablation_cfg:
                return _fmt(ablation_cfg[k])
        return NA

    if key == "temperature" and key in ablation_cfg:
        val = ablation_cfg.get(key)
        return f"{float(val):.1f}" if val is not None else NA
    return _fmt(ablation_cfg.get(key)) if key in ablation_cfg else NA


# ---------------------------------------------------------------------------
# Ordering and grouping
# ---------------------------------------------------------------------------

def _ordered_models(model_ids: set[str]) -> list[str]:
    """Sort models by :data:`MODEL_ORDER`, unknown models appended at end."""
    ordered = [m for m in MODEL_ORDER if m in model_ids]
    ordered += sorted(m for m in model_ids if m not in ordered)
    return ordered


def _is_last_in_family(model: str, ordered: list[str]) -> bool:
    """True if *model* is the last member of its family in *ordered*."""
    family = get_model_family(model)
    idx = ordered.index(model)
    if idx < len(ordered) - 1:
        return get_model_family(ordered[idx + 1]) != family
    return False


# ---------------------------------------------------------------------------
# Column definitions
# Each entry: (internal key for _base_value/_ablation_value, display header)
# ---------------------------------------------------------------------------

#: Temperature ablation: only temperature is the intervention parameter;
#: max_tokens was also increased uniformly across all models (25,000 → 50,000).
TEMP_PARAM_COLS: list[tuple[str, str]] = [
    ("temperature",  "Temperature"),
    ("max_tokens",   "Max Tokens"),
]

#: Reasoning ablation: effort-based (GPT-5) or token-based (Gemini) changes;
#: max_tokens was also increased uniformly across all models (25,000 → 50,000).
REASONING_PARAM_COLS: list[tuple[str, str]] = [
    ("_reasoning_effort", "Reasoning Effort"),
    ("reasoning_tokens",  "Reasoning Tokens"),
    ("max_tokens",        "Max Tokens"),
]


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------

def build_ablation_rows(
    ablation_configs: dict[str, dict],
    param_cols: list[tuple[str, str]],
    *,
    all_reasoning_subscript: bool = False,
) -> list[dict]:
    """Build ordered row dicts for an ablation configuration table.

    Each row contains:

    * ``"label"``            – model display name (with optional R subscript)
    * ``"is_last_in_family"`` – True on the last row before a family boundary
    * ``"{key}"``            – formatted ablation value for each parameter

    Set *all_reasoning_subscript* to True for reasoning-ablation tables, where
    reasoning was enabled for every model regardless of its main-protocol
    setting (notably Gemini 2.5 Flash Lite, off in main, on here).
    """
    ordered = _ordered_models(set(ablation_configs.keys()))
    rows: list[dict] = []
    for model in ordered:
        ablation_cfg = ablation_configs[model]
        row: dict = {
            "label": _row_label(model, all_reasoning_subscript=all_reasoning_subscript),
            "is_last_in_family": _is_last_in_family(model, ordered),
        }
        for key, _ in param_cols:
            row[key] = _ablation_value(ablation_cfg, key)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------

def _tabularx_colspec(n_data: int) -> str:
    return (
        r">{\raggedright\arraybackslash}X"
        + r"*{"
        + str(n_data)
        + r"}{>{\centering\arraybackslash}X}"
    )


# Default captions/labels for each ablation configuration table.
TEMP_ABLATION_CAPTION = (
    "Model configuration parameters, temperature ablation results. "
    f"{REASONING_SUBSCRIPT_CAPTION}"
)
TEMP_ABLATION_LABEL = "tab:configs_ablations_temperature"
REASONING_ABLATION_CAPTION = (
    "Model configuration parameters, reasoning ablation results. "
    f"{REASONING_SYMBOL} is shown for every row because reasoning was enabled "
    "for all models in this reasoning ablation."
)
REASONING_ABLATION_LABEL = "tab:configs_ablations_reasoning"


def to_latex_ablation(
    rows: list[dict],
    param_cols: list[tuple[str, str]],
    caption: str,
    label: str,
) -> str:
    """Render ablation row dicts to a styled ``tabularx`` LaTeX block.

    The output begins with the standard styling preamble (``\\centering``,
    ``\\small``, ``\\caption``, ``\\label``) followed by the ``tabularx``
    environment. A ``\\midrule`` is inserted between model families.
    """
    n_data = len(param_cols)
    T = "    "
    lines: list[str] = []

    lines.append("\\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabularx}}{{\\textwidth}}{{{_tabularx_colspec(n_data)}}}")
    lines.append(f"{T}\\toprule")

    header_parts = ["\\textbf{Model}"]
    for _, header in param_cols:
        header_parts.append(f"\\textbf{{{header}}}")
    lines.append(f"{T}" + " & ".join(header_parts) + " \\\\")
    lines.append(f"{T}\\midrule")

    last_idx = len(rows) - 1
    for i, row in enumerate(rows):
        parts = [row["label"]]
        for key, _ in param_cols:
            parts.append(row[key])
        lines.append(f"{T}" + " & ".join(parts) + " \\\\")
        if row.get("is_last_in_family") and i < last_idx:
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


def _default_output(name: str) -> str:
    return str(Path(__file__).resolve().parents[4] / "paper" / "tables" / name)


def run_temperature_ablation(
    logs_dir: str = DEFAULT_LOGS_DIR,
    output: Optional[str] = None,
    no_save: bool = False,
    parquet_path: Optional[str] = None,
    caption: str = TEMP_ABLATION_CAPTION,
    label: str = TEMP_ABLATION_LABEL,
) -> list[dict]:
    """Generate the temperature ablation configuration table.

    Args:
        logs_dir: Directory containing ``ablations/{temperature}/`` log files.
            Used as fallback when the parquet has no matching rows.
        output: Destination ``.tex`` file path.
        no_save: If True, print to stdout instead of writing a file.
        parquet_path: Path to a pre-processed ``.parquet`` file.  Defaults to
            ``data/final/obs_data.parquet`` relative to the repo root.
        caption: caption text inserted via ``\\caption{...}``.
        label: label inserted via ``\\label{...}``.
    """
    if output is None:
        output = _default_output("configs_ablations_temperature.tex")
    configs = load_ablation_configs("temperature", logs_dir=logs_dir, parquet_path=parquet_path)
    rows = build_ablation_rows(configs, TEMP_PARAM_COLS)  # main-protocol R
    latex = to_latex_ablation(rows, TEMP_PARAM_COLS, caption=caption, label=label)
    _save_or_print(latex, output, no_save, "temperature ablation configs table")
    return rows


def run_reasoning_ablation(
    logs_dir: str = DEFAULT_LOGS_DIR,
    output: Optional[str] = None,
    no_save: bool = False,
    parquet_path: Optional[str] = None,
    caption: str = REASONING_ABLATION_CAPTION,
    label: str = REASONING_ABLATION_LABEL,
) -> list[dict]:
    """Generate the reasoning ablation configuration table.

    Args:
        logs_dir: Directory containing ``ablations/{reasoning}/`` log files.
            Used as fallback when the parquet has no matching rows.
        output: Destination ``.tex`` file path.
        no_save: If True, print to stdout instead of writing a file.
        parquet_path: Path to a pre-processed ``.parquet`` file.  Defaults to
            ``data/final/obs_data.parquet`` relative to the repo root.
        caption: caption text inserted via ``\\caption{...}``.
        label: label inserted via ``\\label{...}``.
    """
    if output is None:
        output = _default_output("configs_ablations_reasoning.tex")
    configs = load_ablation_configs("reasoning", logs_dir=logs_dir, parquet_path=parquet_path)
    rows = build_ablation_rows(
        configs, REASONING_PARAM_COLS, all_reasoning_subscript=True,
    )
    latex = to_latex_ablation(rows, REASONING_PARAM_COLS, caption=caption, label=label)
    _save_or_print(latex, output, no_save, "reasoning ablation configs table")
    return rows
