"""LaTeX coefficient and random-effect tables for the M1 / M2 / M3 posteriors.

Tables per run (``--run`` on the CLI wrapper):

=====================  ====  ====  ====  ============================================
table                  M1    M2    M3    contents
=====================  ====  ====  ====  ============================================
scenario               x     x     x     intercept, task / instruction / context,
                                         party type, scenario covariates, RE scales
scenario_interactions  x     x     x     instruction x context and task x context cells
model                  x     x     x     model-level regression, size tiers, LLM
                                         covariates (+ token slopes for M2)
human                  x                 human-vs-LLM contrast, human x context,
                                         demographic blocks
judge                              x     judge severities, self-preference slopes,
                                         judge RE scales
re_family              x     x     x     per-family random effects
re_model               x     x     x     per-model random intercepts
re_judge_model                     x     judge x model random effects
=====================  ====  ====  ====  ============================================

Every table reports posterior means and 95% HDIs on the log-odds scale; a
``*`` marks coefficients whose HDI excludes zero. Output filenames carry the
run suffix (none for M1, ``_m2``, ``_m3``) so all table sets co-exist under
``paper/tables/regressions/``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ..registry import (
    MODEL_DISPLAY_NAMES,
    MODEL_FAMILIES,
    MODEL_ORDER,
    REASONING_MODELS,
    REASONING_SUBSCRIPT_CAPTION,
    REASONING_SYMBOL,
    normalize_model_id,
)
from ..uncertainty import (
    DEFAULT_HDI_PROB,
    PosteriorBundle,
    format_interval,
    get_family_re,
    get_human_coefs,
    get_interaction_matrices,
    get_judge_coefs,
    get_judge_model_re,
    get_model_coefs,
    get_model_re,
    get_scenario_coefs,
    get_token_coefs,
    latex_significance_marker,
    load_bundle,
)


# ---------------------------------------------------------------------------
# Run registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunInfo:
    """Filenames, labels and caption text for one fitted regression."""

    name: str          # CLI key: m1 / m2 / m3
    run_id: str        # analysis/regressions/out/<run_id>
    model_dir: str     # <run_id>/models/<model_dir>
    suffix: str        # appended to filenames and \\label{}s
    label: str         # M1 / M2 / M3
    description: str   # caption phrase
    unit: str          # what one observation is


RUNS: dict[str, RunInfo] = {
    "m1": RunInfo("m1", "M1", "fraudbench_m1", "", "M1",
                  "the main regression on the fraud outcome", "generations"),
    "m2": RunInfo("m2", "M2", "fraudbench_m2", "_m2", "M2",
                  "the reasoning-tokens variant (LLM generations with a positive reasoning-token count)",
                  "generations"),
    "m3": RunInfo("m3", "M3", "fraudbench_m3", "_m3", "M3",
                  "the self-preference variant (one observation per generation and judge)",
                  "(generation, judge) pairs"),
}
VALID_RUNS: tuple[str, ...] = tuple(RUNS)


def get_run_info(run: str) -> RunInfo:
    if run not in RUNS:
        raise ValueError(f"run must be one of {list(RUNS)}, got {run!r}")
    return RUNS[run]


def default_inference_data_path(run: str, repo_root: Path) -> Path:
    """``analysis/regressions/out/<run_id>/models/<model_dir>/inference_data.nc``."""
    info = get_run_info(run)
    return repo_root / "analysis" / "regressions" / "out" / info.run_id / "models" / info.model_dir / "inference_data.nc"


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def _value_cell(row: pd.Series, *, digits: int = 2) -> str:
    return format_interval((row["mean"], row["hdi_low"], row["hdi_high"]), digits=digits)


def _sig_cell(row: pd.Series) -> str:
    if str(row.get("var", "")).startswith("sigma"):
        return ""  # scales are positive by construction; the star is meaningless
    return "$^{*}$" if latex_significance_marker((row["mean"], row["hdi_low"], row["hdi_high"])) else ""


@dataclass(frozen=True)
class _Section:
    title: str
    df: pd.DataFrame


def _caption(block: str, info: RunInfo, bundle: PosteriorBundle, hdi_prob: float, note: str = "") -> str:
    return (
        f"{block} from {info.label}, {info.description}. "
        f"$N = {len(bundle.processed):,}$ {info.unit}. Coefficients are on the log-odds "
        f"scale, reported as posterior mean and {int(hdi_prob * 100)}\\% highest-density interval. " + note
    ).strip()


def _table(sections: Iterable[_Section], *, caption: str, label: str, header: str, digits: int = 2) -> str:
    T1, T2 = "    ", "        "
    lines = [
        "\\begin{table}[!htbp]", f"{T1}\\centering", f"{T1}\\small",
        f"{T1}\\caption{{{caption}}}", f"{T1}\\label{{{label}}}",
        f"{T1}\\begin{{tabular}}{{lrc}}", f"{T2}\\toprule",
        f"{T2}\\textbf{{Term}} & \\textbf{{{header}}} & \\\\", f"{T2}\\midrule",
    ]
    sections = [s for s in sections if not s.df.empty]
    for k, section in enumerate(sections):
        if section.title:
            lines.append(f"{T2}\\multicolumn{{3}}{{l}}{{\\textit{{{section.title}}}}} \\\\")
        for _, row in section.df.iterrows():
            lines.append(f"{T2}{_latex_escape(str(row['name']))} & {_value_cell(row, digits=digits)} & {_sig_cell(row)} \\\\")
        if k < len(sections) - 1:
            lines.append(f"{T2}\\midrule")
    lines += [f"{T2}\\bottomrule", f"{T1}\\end{{tabular}}", "\\end{table}"]
    return "\n".join(lines)


def _starts(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return df[df["var"].str.startswith(prefix)]


def _isin(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    return df[df["var"].isin(names)]


# ---------------------------------------------------------------------------
# Coefficient tables
# ---------------------------------------------------------------------------

def build_scenario_table(bundle: PosteriorBundle, *, run: str, hdi_prob: float = DEFAULT_HDI_PROB) -> str:
    info = get_run_info(run)
    df = get_scenario_coefs(bundle, hdi_prob=hdi_prob)
    sections = [
        _Section("Intercept", _isin(df, ["alpha"])),
        _Section("Task (vs. binary)", _starts(df, "task_effect[")),
        _Section("Instructional condition (vs. control)", _starts(df, "instruction_effect[")),
        _Section("Contextual condition (vs. control)", _starts(df, "context_effect[")),
        _Section("Case party type (vs. B2B)", _starts(df, "party_type_effect[")),
        _Section("Scenario covariates", _isin(df, ["beta_concealment", "beta_order"])),
        _Section("Random-effect and interaction scales",
                 _isin(df, ["sigma_case", "sigma_scenario", "sigma_instruction_context"])),
    ]
    caption = _caption(
        "Scenario and experimental design fixed effects", info, bundle, hdi_prob,
        note=(""),
    )
    return _table(sections, caption=caption, label=f"tab:reg_coefs_scenario{info.suffix}", header="Coefficient (log-odds)")


def build_scenario_interactions_table(bundle: PosteriorBundle, *, run: str, hdi_prob: float = DEFAULT_HDI_PROB) -> str:
    info = get_run_info(run)
    df = get_interaction_matrices(bundle, hdi_prob=hdi_prob)
    sections = [
        _Section("Instructional $\\times$ Contextual", _starts(df, "instruction_context_effect[")),
        _Section("Task $\\times$ Contextual", _starts(df, "task_context_effect[")),
    ]
    caption = _caption(
        "Design interactions (all non-reference cells)", info, bundle, hdi_prob,
        note=("Each cell is the additional log-odds contribution beyond the main effects in "
              f"\\Cref{{tab:reg_coefs_scenario{info.suffix}}}."),
    )
    return _table(sections, caption=caption, label=f"tab:reg_coefs_scenario_interactions{info.suffix}",
                  header="Coefficient (log-odds)")


def build_model_table(bundle: PosteriorBundle, *, run: str, hdi_prob: float = DEFAULT_HDI_PROB) -> str:
    info = get_run_info(run)
    df = get_model_coefs(bundle, hdi_prob=hdi_prob)
    sections = [
        _Section("Model-level regression", _isin(df, ["beta_release_date"])),
        _Section("Model size tier (vs. smallest tier present)", _starts(df, "model_size_effect[")),
        _Section("LLM covariates", _isin(df, ["beta_reasoning", "beta_high_temperature",
                                              "beta_high_reasoning", "beta_post_cutoff"])),
    ]
    note = "LLM covariates enter the linear predictor on LLM rows only."
    if run == "m2":
        sections.append(_Section("Reasoning-token covariates", get_token_coefs(bundle, hdi_prob=hdi_prob)))
        note += (" Reasoning tokens are $\\log(1 + n)$, standardized over the generations in this "
                 "regression. ")
    sections.append(_Section("Random-effect scales", _isin(df, ["sigma_family", "sigma_model"])))
    caption = _caption("Model-level coefficients", info, bundle, hdi_prob, note=note)
    return _table(sections, caption=caption, label=f"tab:reg_coefs_model{info.suffix}", header="Coefficient (log-odds)")


def build_human_table(bundle: PosteriorBundle, *, run: str, hdi_prob: float = DEFAULT_HDI_PROB) -> str:
    info = get_run_info(run)
    df = get_human_coefs(bundle, hdi_prob=hdi_prob)
    sections = [
        _Section("Human vs. LLM", _isin(df, ["beta_human"])),
        _Section("Human $\\times$ Contextual (vs. control)", _starts(df, "human_context_effect[")),
        _Section("Age (random-effect block)", _starts(df, "age_effect[")),
        _Section("Race (random-effect block)", _starts(df, "race_effect[")),
        _Section("Gender (random-effect block)", _starts(df, "gender_effect[")),
        _Section("AI use (random-effect block)", _starts(df, "ai_use_effect[")),
        _Section("Random-effect scales", _starts(df, "sigma_")),
    ]
    caption = _caption(
        "Human-baseline block coefficients", info, bundle, hdi_prob,
        note=(""),
    )
    return _table(sections, caption=caption, label=f"tab:reg_coefs_human{info.suffix}", header="Coefficient (log-odds)")


def build_judge_table(bundle: PosteriorBundle, *, run: str, hdi_prob: float = DEFAULT_HDI_PROB) -> str:
    info = get_run_info(run)
    df = get_judge_coefs(bundle, hdi_prob=hdi_prob)
    for judge in bundle.levels("judge_ids", []):
        df["name"] = df["name"].str.replace(judge, MODEL_DISPLAY_NAMES.get(normalize_model_id(judge), judge), regex=False)
    sections = [
        _Section("Judge severity (effect-coded, sums to zero)", _starts(df, "judge_effect[")),
        _Section("Self-preference", _isin(df, ["beta_self_family", "beta_self_model"])),
        _Section("Random-effect scales", _isin(df, ["sigma_judge_model", "sigma_judge_scenario"])),
    ]
    caption = _caption(
        "Judge-block coefficients", info, bundle, hdi_prob,
        note=("Because the outcome is the judge's fraud vote, negative coefficients represent leniency towards the judge's own family or "
              "model."),
    )
    return _table(sections, caption=caption, label=f"tab:reg_coefs_judge{info.suffix}", header="Coefficient (log-odds)")


# ---------------------------------------------------------------------------
# Random-effect tables
# ---------------------------------------------------------------------------

def _display_name(model: str) -> str:
    return _latex_escape(MODEL_DISPLAY_NAMES.get(normalize_model_id(model), model))


def _model_label(model: str) -> str:
    """TeX label: display name plus the reasoning subscript for reasoning models."""
    bare = normalize_model_id(model)
    return _display_name(model) + (" " + REASONING_SYMBOL if bare in REASONING_MODELS else "")


def _order_models(df: pd.DataFrame) -> pd.DataFrame:
    family_order = {f: i for i, (f, _) in enumerate(MODEL_FAMILIES)}
    model_order = {m: i for i, m in enumerate(MODEL_ORDER)}
    df = df.copy()
    df["_f"] = df["model_family"].map(lambda f: family_order.get(f, len(family_order)))
    df["_m"] = df["model"].astype(str).map(lambda m: model_order.get(normalize_model_id(m), len(model_order)))
    df["label"] = df["model"].astype(str).map(_model_label)
    return df.sort_values(["_f", "_m", "model"]).reset_index(drop=True)


def _forest(df: pd.DataFrame, *, caption: str, label: str, group_col: Optional[str] = None, size: str = "small") -> str:
    T1, T2 = "    ", "        "
    lines = [
        "\\begin{table}[!htbp]", f"{T1}\\centering", f"{T1}\\{size}",
        f"{T1}\\caption{{{caption}}}", f"{T1}\\label{{{label}}}",
        f"{T1}\\begin{{tabular}}{{lrc}}", f"{T2}\\toprule",
        f"{T2}\\textbf{{Term}} & \\textbf{{Random effect (log-odds)}} & \\\\", f"{T2}\\midrule",
    ]
    last = None
    for _, row in df.iterrows():
        if group_col is not None:
            if last is not None and row[group_col] != last:
                lines.append(f"{T2}\\midrule")
            last = row[group_col]
        lines.append(f"{T2}{row['label']} & {_value_cell(row)} & {_sig_cell(row)} \\\\")
    lines += [f"{T2}\\bottomrule", f"{T1}\\end{{tabular}}", "\\end{table}"]
    return "\n".join(lines)


def build_model_re_table(bundle: PosteriorBundle, *, run: str, hdi_prob: float = DEFAULT_HDI_PROB) -> str:
    info = get_run_info(run)
    df = get_model_re(bundle, hdi_prob=hdi_prob)
    if df.empty:
        return ""
    caption = (
        f"Per-model random intercepts $u^{{\\text{{model}}}}_m$ from {info.label}. {REASONING_SUBSCRIPT_CAPTION}"
    )
    return _forest(_order_models(df), caption=caption, label=f"tab:reg_re_model{info.suffix}", group_col="model_family")


def build_family_re_table(bundle: PosteriorBundle, *, run: str, hdi_prob: float = DEFAULT_HDI_PROB) -> str:
    info = get_run_info(run)
    df = get_family_re(bundle, hdi_prob=hdi_prob)
    if df.empty:
        return ""
    order = {f: i for i, (f, _) in enumerate(MODEL_FAMILIES)}
    df = df.assign(_o=df["model_family"].map(lambda f: order.get(f, len(order))),
                   label=df["model_family"].map(lambda f: _latex_escape(str(f))))
    df = df.sort_values(["_o", "model_family"]).reset_index(drop=True)
    caption = (
        f"Per-family random effects $u^{{\\text{{family}}}}_f$ from {info.label}."
    )
    return _forest(df, caption=caption, label=f"tab:reg_re_family{info.suffix}")


def build_judge_model_re_table(bundle: PosteriorBundle, *, run: str, hdi_prob: float = DEFAULT_HDI_PROB) -> str:
    info = get_run_info(run)
    df = get_judge_model_re(bundle, hdi_prob=hdi_prob)
    if df.empty:
        return ""
    parts = []
    for judge, sub in df.groupby("judge", sort=True):
        ordered = _order_models(sub)
        ordered["label"] = ordered["label"].map(lambda m, j=judge: f"{_display_name(j)} $\\times$ {m}")
        parts.append(ordered)
    out = pd.concat(parts, ignore_index=True)
    caption = (
        f"Judge$\\times$model random effects $u^{{\\text{{judge}}\\times\\text{{model}}}}_{{j,m}}$ from "
        f"{info.label}, grouped by judge. "
    )
    # 3 judges x 20 models = 60 rows: \small overflows a NeurIPS page.
    return _forest(out, caption=caption, label=f"tab:reg_re_judge_model{info.suffix}", group_col="judge", size="footnotesize")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_BUILDERS: dict[str, tuple] = {
    "scenario": (build_scenario_table, "regression_coefs_scenario"),
    "scenario_interactions": (build_scenario_interactions_table, "regression_coefs_scenario_interactions"),
    "model": (build_model_table, "regression_coefs_model"),
    "human": (build_human_table, "regression_coefs_human"),
    "judge": (build_judge_table, "regression_coefs_judge"),
    "re_family": (build_family_re_table, "regression_re_family"),
    "re_model": (build_model_re_table, "regression_re_model"),
    "re_judge_model": (build_judge_model_re_table, "regression_re_judge_model"),
}
VALID_TABLES: tuple[str, ...] = tuple(_BUILDERS)

_TABLES_FOR_RUN: dict[str, tuple[str, ...]] = {
    "m1": ("scenario", "scenario_interactions", "model", "human", "re_family", "re_model"),
    "m2": ("scenario", "scenario_interactions", "model", "re_family", "re_model"),
    "m3": ("scenario", "scenario_interactions", "model", "judge", "re_family", "re_model", "re_judge_model"),
}


def tables_for_run(run: str) -> tuple[str, ...]:
    return _TABLES_FOR_RUN[get_run_info(run).name]


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "paper" / "tables" / "regressions"


def run(
    inference_data_path: str,
    *,
    run: str = "m1",
    processed_data_path: Optional[str] = None,
    features_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    tables: Optional[Iterable[str]] = None,
    hdi_prob: float = DEFAULT_HDI_PROB,
    no_save: bool = False,
) -> dict[str, str]:
    """Build the tables of one run from its posterior; return ``{table: latex}``.

    ``tables`` defaults to every table applicable to ``run``; requesting a
    table that does not apply (e.g. ``human`` for M2) is an error.
    """
    info = get_run_info(run)
    requested = list(tables) if tables is not None else list(tables_for_run(run))
    bad = [t for t in requested if t not in _BUILDERS]
    if bad:
        raise ValueError(f"Unknown table(s) {bad}; valid: {list(_BUILDERS)}")
    inapplicable = [t for t in requested if t not in tables_for_run(run)]
    if inapplicable:
        raise ValueError(f"Table(s) {inapplicable} do not apply to run {run!r}")

    bundle = load_bundle(inference_data_path, processed_data_path=processed_data_path, features_path=features_path)
    out_dir = Path(output_dir) if output_dir else _default_output_dir()
    rendered: dict[str, str] = {}
    for name in requested:
        build, base = _BUILDERS[name]
        latex = build(bundle, run=run, hdi_prob=hdi_prob)
        rendered[name] = latex
        if no_save:
            print(f"\n{'=' * 60}\n{name} ({info.label})\n{'=' * 60}\n{latex}")
            continue
        path = out_dir / f"{base}{info.suffix}.tex"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(latex, encoding="utf-8")
        print(f"Saved {name} table to: {path}")
    return rendered
