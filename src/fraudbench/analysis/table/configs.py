"""LaTeX table generators for model and grader configuration parameters.

Backs ``analysis/tables/table_model_configs.py`` and
``analysis/tables/table_grader_configs.py``.

**Data sources (in priority order):**

1. A pre-processed ``.parquet`` file (e.g. ``data/final/obs_data.parquet``)
   produced by ``analysis/process.py``.  Per-sample columns extracted
   from ModelEvents — ``temperature``, ``reasoning_effort``, ``reasoning_budget``
   (= reasoning tokens), ``max_connections``, and ``max_tokens`` — are
   aggregated to one row per (model, run_category) and used as the primary
   source.

2. :mod:`fraudbench.utils.globals` — used as fallback for any column absent
   from the parquet (``max_connections`` if not yet reprocessed, etc.) and for
   columns that are never per-sample (``max_connections``).
"""
from pathlib import Path
from typing import Optional

import json
import re

import pandas as pd

from fraudbench.utils.globals import (
    GRADER_MODEL_CONFIGS,
    MODEL_CONFIGS,
    MAX_TOKENS,
)
from ..registry import (
    GRADER_REASONING_MODELS,
    MODEL_FAMILIES,
    REASONING_SETTINGS,
    REASONING_SUBSCRIPT_CAPTION,
    REASONING_SYMBOL,
    normalize_model_id,
    reasoning_label,
)
from ..common import (
    available_parquet_columns,
    ensure_model_id_column,
    filter_ai_settings,
    get_model_display_name,
)


def _bare_to_litellm(bare: str) -> Optional[str]:
    """Find the active LiteLLM id for *bare* in :data:`MODEL_CONFIGS`."""
    for litellm_id in MODEL_CONFIGS:
        if normalize_model_id(litellm_id) == bare:
            return litellm_id
    return None


# Em-dash placeholder for parameters not applicable to a given model
NA = "---"

# Parquet columns that map to config table column keys.
# parquet col name  →  internal attr key used in MODEL_CONFIG_COLUMNS
_PARQUET_COL_MAP: dict[str, str] = {
    "temperature":        "temperature",
    "reasoning_effort":   "_reasoning_effort",
    "reasoning_budget":   "reasoning_tokens",  # parquet stores as reasoning_budget
    "max_connections":    "max_connections",
    "max_tokens":         "max_tokens",
}


# ---------------------------------------------------------------------------
# Parquet loading
# ---------------------------------------------------------------------------

def load_configs_from_parquet(
    parquet_path: str,
    run_category: str = "main",
) -> dict[str, dict]:
    """Read per-model config values from a pre-processed parquet file.

    Returns ``{model_id: {attr_key: value}}`` for every column listed in
    :data:`_PARQUET_COL_MAP` that is present in the parquet.  Missing columns
    are silently skipped; callers fall back to globals for anything absent.

    Only rows matching *run_category* are considered.  The modal (most
    common) non-null value per model is used so a single errant sample
    cannot corrupt the config.
    """
    path = Path(parquet_path)
    if not path.exists():
        return {}

    needed = ["model", "run_category", "setting"] + [c for c in _PARQUET_COL_MAP if c != "max_connections"]
    df = pd.read_parquet(
        path,
        columns=available_parquet_columns(path, needed),
    )

    # Filter columns that actually exist
    available = [c for c in needed if c in df.columns]
    df = df[available].copy()

    if "run_category" not in df.columns or "model" not in df.columns:
        return {}

    df = df[df["run_category"] == run_category]
    df = filter_ai_settings(df, verbose=False)
    df = df[df["model"].notna() & (df["model"] != "human")]
    df = ensure_model_id_column(df)

    configs: dict[str, dict] = {}
    for model_id, group in df.groupby("model_id"):
        cfg: dict = {}
        for pq_col, attr_key in _PARQUET_COL_MAP.items():
            if pq_col not in group.columns:
                continue
            series = group[pq_col].dropna()
            if series.empty:
                continue
            # Use modal value (most common) for robustness
            cfg[attr_key] = series.mode().iloc[0]
        if cfg:
            configs[str(model_id)] = cfg
    return configs


# ---------------------------------------------------------------------------
# Value formatting helpers
# ---------------------------------------------------------------------------

def _fmt(val, *, force_one_decimal: bool = False) -> str:
    """Format a single config value for display in a LaTeX table cell."""
    if val is None:
        return NA
    if isinstance(val, bool):
        return "No" if not val else "Yes"
    # Render lists as " / "-joined strings (used for ``inference_providers``).
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return NA
        return " / ".join(_fmt(v) for v in val)
    if isinstance(val, (int, float)) and force_one_decimal:
        return f"{float(val):.1f}"
    # Numeric: strip trailing .0 for cleanliness (except when force_one_decimal)
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return str(val)


def _get_reasoning_effort(cfg: dict) -> str:
    """Return the reasoning-effort setting.

    Handles both provider spellings: ``reasoning_effort`` (GPT-5) and
    ``effort`` (Claude Opus).
    """
    for key in ("reasoning_effort", "effort"):
        if key in cfg:
            return _fmt(cfg[key])
    return NA


def _get_col_value(cfg: dict, attr: str) -> str:
    """Retrieve and format one column value from a config dict.

    The special key ``"_reasoning_effort"`` is dispatched to
    :func:`_get_reasoning_effort` to handle provider-specific key names.
    Any key absent from *cfg* yields :data:`NA`.
    """
    if attr == "_reasoning_effort":
        return _get_reasoning_effort(cfg)
    if attr not in cfg:
        return NA
    if attr == "temperature":
        return _fmt(cfg[attr], force_one_decimal=True)
    return _fmt(cfg[attr])


# ---------------------------------------------------------------------------
# Column definitions
# Each entry is (internal key passed to _get_col_value, LaTeX header text).
# ---------------------------------------------------------------------------

MODEL_CONFIG_COLUMNS: list[tuple[str, str]] = [
    ("inference_providers", "Provider"),
    ("release_date", "Release Date"),
    ("knowledge_cutoff", "Knowledge Cutoff"),
    ("temperature",        "Temperature"),
    ("_reasoning_effort",  "Reasoning Effort"),
    ("reasoning_tokens",   "Reasoning Tokens"),
    ("max_tokens",         "Max Tokens"),
]

GRADER_CONFIG_COLUMNS: list[tuple[str, str]] = [
    ("inference_providers", "Provider"),
    ("release_date", "Release Date"),
    ("knowledge_cutoff", "Knowledge Cutoff"),
    ("temperature",        "Temperature"),
    ("_reasoning_effort",  "Reasoning Effort"),
    ("max_tokens",         "Max Tokens"),
]


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------

_MODELS_JSONL_DEFAULT = str(Path(__file__).resolve().parents[4] / "data" / "final" / "models.jsonl")


def _coerce_models_jsonl_line(line: str) -> str:
    # Some lines use unquoted date-like tokens (invalid JSON), e.g.
    # `"knowledge_cutoff": 05-31-2024`. Coerce those to strings.
    def _fix_field(field: str, s: str) -> str:
        pat = rf'("{field}"\s*:\s*)([0-9]{{2}}-[0-9]{{2}}-[0-9]{{4}})'
        return re.sub(pat, rf'\1"\2"', s)

    line = _fix_field("knowledge_cutoff", line)
    line = _fix_field("release_date", line)
    return line


def load_model_metadata(models_jsonl_path: str = _MODELS_JSONL_DEFAULT) -> dict[str, dict]:
    """Load model metadata keyed by normalized model id."""
    path = Path(models_jsonl_path)
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(_coerce_models_jsonl_line(raw))
        except Exception:
            continue
        key = str(obj.get("model", "")).strip()
        if not key:
            continue
        out[key] = obj
    return out


def _get_metadata_cols(model_id: str, meta_by_key: dict[str, dict]) -> dict[str, str]:
    """Look up metadata cells from ``models.jsonl`` keyed by bare model id.

    *model_id* may be either a bare or a LiteLLM-style id; it is normalized
    via :func:`fraudbench.analysis.registry.normalize_model_id` to match the
    bare ``model`` keys in ``models.jsonl``.
    """
    key = normalize_model_id(model_id)
    meta = meta_by_key.get(key, {})
    return {
        "inference_providers": _fmt(meta.get("inference_providers")),
        "release_date": _fmt(meta.get("release_date")),
        "knowledge_cutoff": _fmt(meta.get("knowledge_cutoff")),
    }


def _wrap_model_label(label: str) -> str:
    # Keep model names on one line where possible (wider Model column).
    return label


def _label_with_reasoning(bare_id: str) -> str:
    """Display alias for *bare_id*, with :data:`REASONING_SYMBOL` appended for
    models evaluated with reasoning enabled under the **main** protocol."""
    return reasoning_label(get_model_display_name(bare_id), bare_id)


def _grader_label_with_reasoning(bare_id: str) -> str:
    """Display alias for a grader, with :data:`REASONING_SYMBOL` appended for
    graders that ran with reasoning enabled (see
    :data:`fraudbench.analysis.registry.GRADER_REASONING_MODELS`)."""
    name = get_model_display_name(bare_id)
    if bare_id in GRADER_REASONING_MODELS:
        return f"{name} {REASONING_SYMBOL}"
    return name


def _build_group_label(
    family_name: str,
    group_models: list[str],
    all_family_models: list[str],
) -> str:
    """Construct the display label for a group of models within a family.

    * Single model in group → full display name.
    * All models in family share the same config → ``"Family (all variants)"``.
    * Subset of family → ``"Family (Alias1, Alias2, ...)"``.
    """
    if len(group_models) == 1:
        return get_model_display_name(group_models[0])
    if set(group_models) == set(all_family_models):
        return f"{family_name} (all variants)"
    display_names = [get_model_display_name(m) for m in group_models]
    return f"{family_name} ({', '.join(display_names)})"


def _merge_cfg(globals_cfg: dict, parquet_cfg: dict) -> dict:
    """Return a merged view of *globals_cfg* overridden by *parquet_cfg*.

    ``parquet_cfg`` keys use the internal attr-key names (same as
    :data:`_PARQUET_COL_MAP` values), so they map directly onto the column
    keys used in :data:`MODEL_CONFIG_COLUMNS`.  Keys present in the parquet
    take precedence; globals fills in everything else (``do_sample``, etc.).
    """
    merged = dict(globals_cfg)
    for attr_key, val in parquet_cfg.items():
        if attr_key == "_reasoning_effort":
            # Store under the canonical key so _get_reasoning_effort finds it
            merged.setdefault("reasoning_effort", val)
            merged["reasoning_effort"] = val
        else:
            merged[attr_key] = val
    return merged


def build_model_config_rows(parquet_configs: Optional[dict] = None) -> list[dict]:
    """Build ordered row data for the evaluated-model configuration table.

    Produces one row per model (no consolidation), ordered by :data:`MODEL_FAMILIES`.

    Each row dict contains:

    * ``"label"``             – display string for the Model column
    * ``"family"``            – family name (for grouping)
    * ``"is_last_in_family"`` – True for the final row of each family block
    * one entry per column key from :data:`MODEL_CONFIG_COLUMNS`

    Args:
        parquet_configs: mapping ``{model_id: {attr_key: value}}`` as returned
            by :func:`load_configs_from_parquet`.  When provided, per-model
            values override the globals defaults.
    """
    parquet_configs = parquet_configs or {}
    rows: list[dict] = []
    col_keys = [k for k, _ in MODEL_CONFIG_COLUMNS]

    meta_by_key = load_model_metadata()

    # ``MODEL_FAMILIES`` is keyed by bare model id; ``MODEL_CONFIGS`` is keyed
    # by LiteLLM id. We resolve each bare id to its active LiteLLM variant and
    # skip families with no live config (e.g. retired models).
    for family_name, family_models in MODEL_FAMILIES:
        resolved: list[tuple[str, str]] = []  # (bare_id, litellm_id)
        for bare_id in family_models:
            litellm_id = _bare_to_litellm(bare_id)
            if litellm_id is not None:
                resolved.append((bare_id, litellm_id))
        if not resolved:
            continue

        for bare_id, litellm_id in resolved:
            base = MODEL_CONFIGS[litellm_id]
            # ``parquet_configs`` is bare-id-keyed; ``MODEL_CONFIGS`` is
            # LiteLLM-keyed. Look up the parquet override by bare id.
            pq = parquet_configs.get(bare_id, {})
            cfg = _merge_cfg(base, pq)
            cfg.update(_get_metadata_cols(bare_id, meta_by_key))
            label = _wrap_model_label(_label_with_reasoning(bare_id))
            row: dict = {
                "label": label,
                "family": family_name,
                "is_last_in_family": False,
            }
            for key in col_keys:
                row[key] = _get_col_value(cfg, key)
            # Models whose reasoning was enabled only in some settings (see
            # ``REASONING_SETTINGS``): the parquet modal value comes from the
            # reasoning-enabled settings only, so say so in the cell.
            if bare_id in REASONING_SETTINGS:
                only = ", ".join(REASONING_SETTINGS[bare_id]) + " only"
                for key in ("_reasoning_effort", "reasoning_tokens"):
                    if row[key] != NA:
                        row[key] = f"{row[key]} ({only})"
            rows.append(row)

        # Flag the last row of this family for a \midrule separator.
        rows[-1]["is_last_in_family"] = True

    return rows


# Effective ``max_tokens`` applied to graders that do not set one explicitly.
# Inspect-AI's Anthropic provider caps the request at 32,000 output tokens
# for Claude 4.5 models (``inspect_ai/model/_providers/anthropic.py``,
# ``max_tokens_for_config``); the scorer ``ModelEvent`` configs in the
# evaluation logs record ``max_tokens: 32000`` for Claude Haiku 4.5 and no
# value for the OpenAI / Google graders (provider defaults).
_GRADER_EFFECTIVE_MAX_TOKENS: dict[str, int] = {
    "claude-haiku-4-5-20251001": 32000,
}


def build_grader_config_rows(parquet_configs: Optional[dict] = None) -> list[dict]:
    """Build ordered row data for the LLM-as-a-judge configuration table.

    Preserves the insertion order of :data:`GRADER_MODEL_CONFIGS`.

    The rows are built from :data:`GRADER_MODEL_CONFIGS` **only**. The
    per-sample config columns of ``obs_data.parquet`` describe the *evaluated*
    model's solver call, not the grader calls, so they must not be used as
    overrides here even though the three graders are also evaluated models
    (doing so previously reported the evaluated-model settings -- e.g.
    ``reasoning_effort=minimal`` / ``temperature=0.0`` -- for the graders).
    The values were verified against the scorer ``ModelEvent`` configs in the
    evaluation logs and against ``obs_data.parquet.scorer_tokens``.

    Args:
        parquet_configs: ignored; accepted for signature compatibility.
    """
    del parquet_configs  # intentionally unused, see docstring
    rows: list[dict] = []
    col_keys = [k for k, _ in GRADER_CONFIG_COLUMNS]

    meta_by_key = load_model_metadata()

    for model, cfg in GRADER_MODEL_CONFIGS.items():
        bare_id = normalize_model_id(model)
        merged = dict(cfg)
        if "max_tokens" not in merged and bare_id in _GRADER_EFFECTIVE_MAX_TOKENS:
            merged["max_tokens"] = _GRADER_EFFECTIVE_MAX_TOKENS[bare_id]
        merged.update(_get_metadata_cols(bare_id, meta_by_key))
        row: dict = {"label": _grader_label_with_reasoning(bare_id)}
        for key in col_keys:
            row[key] = _get_col_value(merged, key)
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------

def _tabularx_colspec(
    n_data_cols: int,
    *,
    model_col_width: Optional[str] = None,
) -> str:
    """Return a tabularx colspec with optional fixed-width Model column."""
    first_col = (
        r">{\raggedright\arraybackslash}p{" + model_col_width + "}"
        if model_col_width
        else r">{\raggedright\arraybackslash}X"
    )
    return (
        first_col
        + r"*{"
        + str(n_data_cols)
        + r"}{>{\centering\arraybackslash}X}"
    )


# Default captions/labels for each rendered configuration table.
# Labels match the basename of the resulting ``.tex`` file (no path / extension).
MODEL_CONFIGS_CAPTION = (
    f"Model configuration parameters, main results. {REASONING_SUBSCRIPT_CAPTION}"
)
MODEL_CONFIGS_LABEL = "tab:configs_model"
GRADER_CONFIGS_CAPTION = (
    f"Model configuration parameters, LLM-as-a-judge models. "
    f"{REASONING_SUBSCRIPT_CAPTION}"
)
GRADER_CONFIGS_LABEL = "tab:configs_grader"

_DASHED_ENTRIES_NOTE = (
    r"Dashed entries (\texttt{---}) indicate that a configuration variable "
    r"was not available for that model."
)
_QWEN_PROVIDER_NOTE = (
    "For Qwen 3 235B models, providers varied across the evaluation because "
    "Fireworks AI stopped offering that model during runs."
)
_GLM_PROVIDER_NOTE = (
    "For GLM-4.7 models, providers were switched during evaluation to mitigate "
    "latency issues."
)


def _render_latex(
    rows: list[dict],
    columns: list[tuple[str, str]],
    caption: str,
    label: str,
    *,
    sideways: bool = False,
    table_width: str = r"\textwidth",
    model_col_width: Optional[str] = None,
    notes: Optional[list[str]] = None,
) -> str:
    """Render row dicts to a styled ``tabularx`` LaTeX block.

    Renders a full table float containing a ``tabularx``.

    Args:
        sideways: if True, wraps in a ``sidewaystable`` (rotated 90° CCW,
            requires ``\\usepackage{rotating}``) instead of ``table``.
        table_width: width passed to ``tabularx`` / note minipage.
        model_col_width: optional fixed width for the first (Model) column.
        notes: optional note lines rendered below the table body.
    """
    col_keys = [k for k, _ in columns]
    col_headers = [h for _, h in columns]
    n_data = len(col_keys)

    env = "sidewaystable" if sideways else "table"
    T = "    "
    lines: list[str] = []

    lines.append(f"\\begin{{{env}}}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(
        f"\\begin{{tabularx}}{{{table_width}}}"
        f"{{{_tabularx_colspec(n_data, model_col_width=model_col_width)}}}"
    )
    lines.append(f"{T}\\toprule")

    header_parts = ["\\textbf{Model}"] + [f"\\textbf{{{h}}}" for h in col_headers]
    lines.append(f"{T}" + " & ".join(header_parts) + " \\\\")
    lines.append(f"{T}\\midrule")

    last_idx = len(rows) - 1
    for i, row in enumerate(rows):
        parts = [row["label"]] + [row[k] for k in col_keys]
        lines.append(f"{T}" + " & ".join(parts) + " \\\\")
        if row.get("is_last_in_family") and i < last_idx:
            lines.append(f"{T}\\midrule")

    lines.append(f"{T}\\bottomrule")
    lines.append("\\end{tabularx}")
    if notes:
        lines.append(r"\vspace{2pt}")
        lines.append(f"\\begin{{minipage}}{{{table_width}}}")
        lines.append(r"\footnotesize")
        for i, note in enumerate(notes):
            prefix = r"\textit{Note.} " if i == 0 else r"\textit{Additional note.} "
            lines.append(prefix + note + r"\par")
        lines.append(r"\end{minipage}")
    lines.append(f"\\end{{{env}}}")

    return "\n".join(lines)


def to_latex_model_configs(
    rows: list[dict],
    caption: str = MODEL_CONFIGS_CAPTION,
    label: str = MODEL_CONFIGS_LABEL,
) -> str:
    """Render the evaluated-model configs as a styled ``tabularx`` block.

    Wrapped in ``sidewaystable`` because the main config table has too many
    columns to fit upright.
    """
    return _render_latex(
        rows,
        MODEL_CONFIG_COLUMNS,
        caption=caption,
        label=label,
        sideways=True,
        model_col_width=r"0.36\textwidth",
        notes=[_DASHED_ENTRIES_NOTE, _QWEN_PROVIDER_NOTE, _GLM_PROVIDER_NOTE],
    )


_GRADER_SETTINGS_NOTE = (
    "Values are the grader configurations applied to every scorer call "
    "(\\texttt{GRADER\\_MODEL\\_CONFIGS}). Where no maximum was set, the "
    "provider default applied; for Claude Haiku 4.5 the Inspect-AI Anthropic "
    "provider capped output at 32{,}000 tokens. Claude Haiku 4.5 and Gemini 2.5 "
    "Flash Lite graded without extended thinking (zero reasoning tokens on "
    "every graded sample)."
)


def to_latex_grader_configs(
    rows: list[dict],
    caption: str = GRADER_CONFIGS_CAPTION,
    label: str = GRADER_CONFIGS_LABEL,
) -> str:
    """Render the grader model configs as a styled ``tabularx`` block."""
    return _render_latex(
        rows,
        GRADER_CONFIG_COLUMNS,
        caption=caption,
        label=label,
        notes=[_DASHED_ENTRIES_NOTE, _GRADER_SETTINGS_NOTE],
    )


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


_DEFAULT_PARQUET = str(
    Path(__file__).resolve().parents[4] / "data" / "final" / "obs_data.parquet"
)


def _load_parquet_configs(
    parquet_path: Optional[str],
    run_category: str,
) -> dict:
    """Try to load parquet configs; return empty dict on failure."""
    pq = parquet_path if parquet_path is not None else _DEFAULT_PARQUET
    configs = load_configs_from_parquet(pq, run_category=run_category)
    if configs:
        print(f"Loaded config overrides from parquet for {len(configs)} model(s) "
              f"(run_category='{run_category}'): {pq}")
    else:
        print(f"No parquet config data found at {pq} — using globals only.")
    return configs


def run_model_configs(
    output: Optional[str] = None,
    no_save: bool = False,
    parquet_path: Optional[str] = None,
    caption: str = MODEL_CONFIGS_CAPTION,
    label: str = MODEL_CONFIGS_LABEL,
) -> list[dict]:
    """Generate the evaluated-model configuration table and optionally save it.

    Args:
        output: Destination ``.tex`` file path (default: ``paper/tables/configs_model.tex``).
        no_save: If True, print to stdout instead of writing a file.
        parquet_path: Path to a pre-processed ``.parquet`` file.  Defaults to
            ``data/final/obs_data.parquet`` relative to the repo root.
            Pass an empty string to skip parquet loading entirely.
        caption: caption text inserted via ``\\caption{...}``.
        label: label inserted via ``\\label{...}``.
    """
    if output is None:
        output = str(
            Path(__file__).resolve().parents[4] / "paper" / "tables" / "configs_model.tex"
        )
    parquet_configs = _load_parquet_configs(parquet_path, run_category="main")
    rows = build_model_config_rows(parquet_configs=parquet_configs)
    latex = to_latex_model_configs(rows, caption=caption, label=label)
    _save_or_print(latex, output, no_save, "model configs table")
    return rows


def run_grader_configs(
    output: Optional[str] = None,
    no_save: bool = False,
    parquet_path: Optional[str] = None,
    caption: str = GRADER_CONFIGS_CAPTION,
    label: str = GRADER_CONFIGS_LABEL,
) -> list[dict]:
    """Generate the grader configuration table and optionally save it.

    Args:
        output: Destination ``.tex`` file path (default: ``paper/tables/configs_grader.tex``).
        no_save: If True, print to stdout instead of writing a file.
        parquet_path: Path to a pre-processed ``.parquet`` file.  Defaults to
            ``data/final/obs_data.parquet`` relative to the repo root.
            Pass an empty string to skip parquet loading entirely.
        caption: caption text inserted via ``\\caption{...}``.
        label: label inserted via ``\\label{...}``.
    """
    if output is None:
        output = str(
            Path(__file__).resolve().parents[4] / "paper" / "tables" / "configs_grader.tex"
        )
    # Grader rows are built from ``GRADER_MODEL_CONFIGS`` only; the parquet's
    # per-sample config columns describe the evaluated model, not the graders.
    del parquet_path
    rows = build_grader_config_rows()
    latex = to_latex_grader_configs(rows, caption=caption, label=label)
    _save_or_print(latex, output, no_save, "grader configs table")
    return rows
