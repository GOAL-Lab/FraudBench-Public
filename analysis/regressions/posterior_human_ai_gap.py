"""Posterior-predictive human-vs-LLM fraud-rate gap from the M1 posterior.

Implements the *Human-vs-LLM comparison* paragraph of the regression
appendix (paper eq. ``eq:m1_ppd``). On the common support of the two groups
-- the 100 scenarios of the ``human_baseline`` task, its five contextual
conditions and the ``control`` instructional condition -- every posterior
draw gives, for each scenario ``s`` and contextual condition ``e``,

    theta_LLM[s, e]   = mean over the evaluated LLMs m  of expit(eta[s, e, m])
    theta_Human[s, e] = mean over the 996 respondents p of expit(eta[s, e, p])

where ``eta`` is the M1 linear predictor evaluated with

* LLM side: task = human_baseline, instruction = control, the scenario's
  observed covariates and its posterior case / scenario random effects, the
  LLM's posterior random intercept, the reasoning flag as configured in the
  LLM's main run, temperature / high-reasoning ablation flags = 0, and the
  post-cutoff indicator at its observed value for the (model, scenario) pair;
* Human side: the same scenario and design terms plus the human block at the
  respondent's posterior participant and demographic effects.

Both sides are then averaged uniformly over the 100 scenarios (per-condition
gap) and over the five conditions (overall gap). The gap is reported on the
probability scale and as the log-odds ratio of the two averaged
probabilities, with posterior means and 95% HDIs.

The predictor is reconstructed with
:func:`fraudbench.analysis.regressions.predictor.m1_eta`, the NumPy mirror
of the fitted NumPyro model, on synthetic rows laid out exactly like the
fitted feature arrays.

Usage::

    uv run analysis/regressions/posterior_human_ai_gap.py            # canonical M1 run
    uv run analysis/regressions/posterior_human_ai_gap.py --chunk_size 20

Outputs (default ``paper/figures/regressions/`` and ``paper/tables/regressions/``):
``human_ai_gap.csv`` / ``.json`` / ``_draws.npz``, ``human_ai_gap.tex`` and the
figures ``human_ai_gap_delta.pdf`` and ``human_ai_gap_by_side.pdf``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import arviz as az
import jsonargparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraudbench.analysis.regressions.predictor import expit, m1_eta
from fraudbench.analysis.regressions.process import CONTEXT_LEVELS, SETTING_LEVELS
from fraudbench.analysis.uncertainty import PosteriorBundle, load_bundle
from fraudbench.analysis.style import save_figure

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INFERENCE_DATA = REPO_ROOT / "analysis/regressions/out/M1/models/fraudbench_m1/inference_data.nc"
DEFAULT_FIGURE_DIR = REPO_ROOT / "paper/figures/regressions"
DEFAULT_TABLE_DIR = REPO_ROOT / "paper/tables/regressions"

CONTEXT_LABELS = {
    "control": "Control",
    "peer_pressure_low": "Peer pressure (low)",
    "peer_pressure_high": "Peer pressure (high)",
    "negative_incentive_low": "Negative incentive (low)",
    "negative_incentive_high": "Negative incentive (high)",
}

logger = logging.getLogger("posterior_human_ai_gap")


# --------------------------------------------------------------------------
# Common-support design
# --------------------------------------------------------------------------

def _hb_task_idx(bundle: PosteriorBundle) -> int:
    return list(bundle.levels("task_levels", SETTING_LEVELS)).index("human_baseline")


def build_design(bundle: PosteriorBundle) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scenario, LLM and respondent tables of the common support.

    Returns ``(scenarios, models, respondents)`` where ``scenarios`` has one
    row per human-baseline scenario (``scenario_idx``, ``case_idx``,
    ``party_idx``, ``is_concealment``, ``is_fraud_first``), ``models`` one row
    per LLM (``model_idx``, ``reasoning_on`` as configured in the main run)
    and ``respondents`` one row per human respondent with their demographic
    indices.
    """
    df = bundle.processed
    hb = df[df["task_idx"] == _hb_task_idx(bundle)]
    if hb.empty:
        raise ValueError("No human_baseline rows in processed_data; cannot build the common support.")
    scenarios = (
        hb.groupby("scenario_idx", sort=True)[["case_idx", "party_idx", "is_concealment", "is_fraud_first"]]
        .first().reset_index()
    )
    models = pd.DataFrame({
        "model_idx": np.arange(bundle.num_model),
        "reasoning_on": np.asarray(bundle.features["reasoning_main_by_model"], dtype=float),
    })
    humans = df[df["is_human"] == 1]
    respondents = (
        humans.groupby("participant_idx", sort=True)[["age_idx", "race_idx", "gender_idx", "ai_use_idx"]]
        .first().reset_index()
    )
    if respondents.empty:
        raise ValueError("No human rows in processed_data; M1 must be fit on the full panel.")
    return scenarios, models, respondents


def post_cutoff_matrix(bundle: PosteriorBundle, scenarios: pd.DataFrame) -> np.ndarray:
    """``(num_model, n_scenario)`` observed post-cutoff indicator per (LLM, scenario).

    The indicator depends only on the case date and the model's cutoff, so
    any observed row of the pair gives its value; pairs never observed
    (none in the full panel) default to 0 with a warning.
    """
    ai = bundle.processed[bundle.processed["is_ai"] == 1]
    observed = ai.groupby(["model_idx", "scenario_idx"])["post_cutoff_case"].max()
    out = np.zeros((bundle.num_model, len(scenarios)), dtype=float)
    missing = 0
    for j, s in enumerate(scenarios["scenario_idx"].to_numpy()):
        for m in range(bundle.num_model):
            if (m, s) in observed.index:
                out[m, j] = float(observed.loc[(m, s)])
            else:
                missing += 1
    if missing:
        logger.warning("%d (model, scenario) pairs never observed; post-cutoff indicator set to 0 for them", missing)
    return out


def _rows(
    bundle: PosteriorBundle, scenarios: pd.DataFrame, *, n_ctx: int,
    side: str, models: Optional[pd.DataFrame] = None, respondents: Optional[pd.DataFrame] = None,
    cutoff: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Feature arrays for the (scenario, context, entity) grid, entity fastest."""
    S = len(scenarios)
    ent = models if side == "llm" else respondents
    K = len(ent)
    n = S * n_ctx * K
    s_rep = np.repeat(np.arange(S), n_ctx * K)
    e_rep = np.tile(np.repeat(np.arange(n_ctx), K), S)
    k_rep = np.tile(np.arange(K), S * n_ctx)

    feats: Dict[str, np.ndarray] = {
        "task_idx": np.full(n, _hb_task_idx(bundle), dtype=np.int64),
        "instruction_idx": np.zeros(n, dtype=np.int64),
        "context_idx": e_rep.astype(np.int64),
        "scenario_idx": scenarios["scenario_idx"].to_numpy()[s_rep].astype(np.int64),
        "case_idx": scenarios["case_idx"].to_numpy()[s_rep].astype(np.int64),
        "party_idx": scenarios["party_idx"].to_numpy()[s_rep].astype(np.int64),
        "is_concealment": scenarios["is_concealment"].to_numpy(dtype=float)[s_rep],
        "is_fraud_first": scenarios["is_fraud_first"].to_numpy(dtype=float)[s_rep],
        "high_temperature": np.zeros(n), "high_reasoning": np.zeros(n),
        "post_cutoff_case": np.zeros(n), "reasoning_on": np.zeros(n),
        "model_idx": np.zeros(n, dtype=np.int64), "participant_idx": np.zeros(n, dtype=np.int64),
        "age_idx": np.zeros(n, dtype=np.int64), "race_idx": np.zeros(n, dtype=np.int64),
        "gender_idx": np.zeros(n, dtype=np.int64), "ai_use_idx": np.zeros(n, dtype=np.int64),
    }
    if side == "llm":
        m = models["model_idx"].to_numpy()[k_rep]
        feats["model_idx"] = m.astype(np.int64)
        feats["reasoning_on"] = models["reasoning_on"].to_numpy(dtype=float)[k_rep]
        feats["post_cutoff_case"] = cutoff[m, s_rep]
        feats["is_ai"], feats["is_human"] = np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    else:
        for col in ["participant_idx", "age_idx", "race_idx", "gender_idx", "ai_use_idx"]:
            feats[col] = respondents[col].to_numpy()[k_rep].astype(np.int64)
        feats["is_ai"], feats["is_human"] = np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)
    return feats


# --------------------------------------------------------------------------
# Posterior computation
# --------------------------------------------------------------------------

def compute_theta(
    bundle: PosteriorBundle, feats: Dict[str, np.ndarray], *, S: int, n_ctx: int, K: int, chunk_size: int,
) -> np.ndarray:
    """``(n_draws, S, n_ctx)`` entity-averaged probabilities for one side."""
    draws = bundle.draws
    D = bundle.n_draws
    out = np.empty((D, S, n_ctx), dtype=np.float64)
    n_chunks = (D + chunk_size - 1) // chunk_size
    for c in range(n_chunks):
        lo, hi = c * chunk_size, min((c + 1) * chunk_size, D)
        chunk = {k: v[lo:hi] for k, v in draws.items()}
        eta = m1_eta(chunk, feats)                      # (d, S * n_ctx * K)
        out[lo:hi] = expit(eta).reshape(hi - lo, S, n_ctx, K).mean(axis=3)
        if (c + 1) % max(1, n_chunks // 10) == 0:
            logger.info("  chunk %d / %d", c + 1, n_chunks)
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p) - np.log1p(-p)


def summarize(samples: np.ndarray, hdi_prob: float) -> dict:
    s = np.asarray(samples).ravel()
    lo, hi = az.hdi(s, hdi_prob=hdi_prob)
    return {"mean": float(s.mean()), "median": float(np.median(s)), "hdi_low": float(lo), "hdi_high": float(hi)}


def build_table(theta_llm: np.ndarray, theta_hum: np.ndarray, ctx_levels: list[str], hdi_prob: float) -> tuple[pd.DataFrame, dict]:
    """Per-condition and overall summaries; also returns the per-condition / overall draws."""
    p_llm_e, p_hum_e = theta_llm.mean(axis=1), theta_hum.mean(axis=1)       # (D, E)
    p_llm_all, p_hum_all = p_llm_e.mean(axis=1), p_hum_e.mean(axis=1)        # (D,)
    rows = []
    for e, name in enumerate(ctx_levels):
        rows.append(_record(name, p_hum_e[:, e], p_llm_e[:, e], hdi_prob))
    rows.append(_record("overall", p_hum_all, p_llm_all, hdi_prob))
    draws = {
        "p_llm_by_context": p_llm_e, "p_human_by_context": p_hum_e,
        "p_llm_overall": p_llm_all, "p_human_overall": p_hum_all,
    }
    return pd.DataFrame(rows), draws


def _record(name: str, p_hum: np.ndarray, p_llm: np.ndarray, hdi_prob: float) -> dict:
    delta = p_hum - p_llm
    lor = _logit(p_hum) - _logit(p_llm)
    rec = {"scope": name, "n_draws": int(p_hum.shape[0])}
    for key, arr in [("p_human", p_hum), ("p_llm", p_llm), ("delta", delta), ("log_odds_ratio", lor)]:
        rec.update({f"{key}_{k}": v for k, v in summarize(arr, hdi_prob).items()})
    rec["pr_delta_gt_0"] = float(np.mean(delta > 0))
    return rec


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------

def _fmt(rec: dict, key: str, *, pct: bool, digits: int) -> str:
    scale = 100.0 if pct else 1.0
    return (f"{rec[f'{key}_mean'] * scale:.{digits}f} "
            f"[{rec[f'{key}_hdi_low'] * scale:.{digits}f}, {rec[f'{key}_hdi_high'] * scale:.{digits}f}]")


def latex_table(df: pd.DataFrame, hdi_prob: float, n_models: int, n_resp: int, n_scen: int) -> str:
    T1, T2 = "    ", "        "
    lines = [
        "\\begin{table}[!htbp]", f"{T1}\\centering", f"{T1}\\small",
        f"{T1}\\caption{{Posterior-predictive human-vs-LLM fraud rates on the common support of the "
        f"\\texttt{{human\\_baseline}} task ({n_scen} scenarios, control instruction), from M1. The LLM rate "
        f"averages over the {n_models} evaluated LLMs at their main-run settings and the human rate over the "
        f"{n_resp} respondents (participant and demographic effects included); both are then averaged uniformly "
        f"over scenarios and, for the overall row, over the five contextual conditions. Cells are posterior means "
        f"and {int(hdi_prob * 100)}\\% HDIs; rates and differences in percentage points, the log-odds ratio on the "
        "logit scale. Positive values mean humans are more likely to commit fraud than LLMs.}",
        f"{T1}\\label{{tab:reg_human_ai_gap}}",
        f"{T1}\\begin{{tabular}}{{lrrrr}}", f"{T2}\\toprule",
        f"{T2}\\textbf{{Context}} & \\textbf{{Human (\\%)}} & \\textbf{{LLM (\\%)}} & "
        "\\textbf{Human $-$ LLM (pp)} & \\textbf{Log-odds ratio} \\\\",
        f"{T2}\\midrule",
    ]
    for _, rec in df.iterrows():
        name = "Overall" if rec["scope"] == "overall" else CONTEXT_LABELS.get(rec["scope"], rec["scope"])
        if rec["scope"] == "overall":
            lines.append(f"{T2}\\midrule")
        lines.append(
            f"{T2}{name} & {_fmt(rec, 'p_human', pct=True, digits=1)} & {_fmt(rec, 'p_llm', pct=True, digits=1)} & "
            f"{_fmt(rec, 'delta', pct=True, digits=1)} & {_fmt(rec, 'log_odds_ratio', pct=False, digits=2)} \\\\"
        )
    lines += [f"{T2}\\bottomrule", f"{T1}\\end{{tabular}}", "\\end{table}"]
    return "\n".join(lines)


def make_figures(df: pd.DataFrame, figure_dir: Path, hdi_prob: float) -> None:
    per = df[df["scope"] != "overall"].reset_index(drop=True)
    x = np.arange(len(per))
    labels = [CONTEXT_LABELS.get(s, s) for s in per["scope"]]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    med = per["delta_mean"].to_numpy() * 100
    lo, hi = per["delta_hdi_low"].to_numpy() * 100, per["delta_hdi_high"].to_numpy() * 100
    ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.errorbar(x, med, yerr=np.vstack([med - lo, hi - med]), fmt="o", capsize=4, color="#444", ecolor="#888")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Human $-$ LLM fraud rate (pp)")
    ax.set_title(f"Posterior-predictive human-minus-LLM gap (mean and {int(hdi_prob * 100)}% HDI)")
    fig.tight_layout()
    save_figure(fig, figure_dir / "human_ai_gap_delta.pdf", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    w = 0.38
    for off, key, color, name in [(-w / 2, "p_human", "#3a7ca5", "Humans"), (w / 2, "p_llm", "#d18a4e", "LLMs")]:
        m = per[f"{key}_mean"].to_numpy() * 100
        l_, h_ = per[f"{key}_hdi_low"].to_numpy() * 100, per[f"{key}_hdi_high"].to_numpy() * 100
        ax.bar(x + off, m, w, yerr=np.vstack([m - l_, h_ - m]), color=color, capsize=3, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Posterior-predictive fraud rate (%)")
    ax.set_title("Human vs. LLM fraud rate on the human-baseline task")
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, figure_dir / "human_ai_gap_by_side.pdf", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(
    inference_data: Path = DEFAULT_INFERENCE_DATA,
    processed_data: Optional[Path] = None,
    features: Optional[Path] = None,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    table_dir: Path = DEFAULT_TABLE_DIR,
    chunk_size: int = 40,
    hdi_prob: float = 0.95,
    verbose: bool = False,
) -> None:
    """Compute the posterior-predictive human-vs-LLM gap from the M1 fit.

    Args:
        inference_data: Path to the M1 ``inference_data.nc``.
        processed_data: Override for ``processed_data.parquet`` (default: run layout).
        features: Override for ``features.pkl`` (default: next to the posterior).
        figure_dir: Where the figures, CSV / JSON summaries and draws are written.
        table_dir: Where the LaTeX table is written.
        chunk_size: Posterior draws processed per chunk (memory control; the
            human side evaluates scenarios x contexts x respondents rows per draw).
        hdi_prob: HDI mass.
        verbose: DEBUG logging.
    """
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout, force=True)
    figure_dir, table_dir = Path(figure_dir), Path(table_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(Path(inference_data), processed_data_path=processed_data, features_path=features)
    ctx_levels = bundle.levels("context_levels", CONTEXT_LEVELS)
    scenarios, models, respondents = build_design(bundle)
    cutoff = post_cutoff_matrix(bundle, scenarios)
    S, E, M, P = len(scenarios), len(ctx_levels), len(models), len(respondents)
    logger.info("Common support: %d scenarios x %d contexts; %d LLMs, %d respondents; %d draws", S, E, M, P, bundle.n_draws)

    llm_feats = _rows(bundle, scenarios, n_ctx=E, side="llm", models=models, cutoff=cutoff)
    hum_feats = _rows(bundle, scenarios, n_ctx=E, side="human", respondents=respondents)
    logger.info("LLM side ...")
    theta_llm = compute_theta(bundle, llm_feats, S=S, n_ctx=E, K=M, chunk_size=max(chunk_size * 10, 1))
    logger.info("Human side ...")
    theta_hum = compute_theta(bundle, hum_feats, S=S, n_ctx=E, K=P, chunk_size=chunk_size)

    df, draws = build_table(theta_llm, theta_hum, ctx_levels, hdi_prob)
    df.to_csv(figure_dir / "human_ai_gap.csv", index=False)
    (figure_dir / "human_ai_gap.json").write_text(json.dumps(df.to_dict(orient="records"), indent=2))
    np.savez_compressed(figure_dir / "human_ai_gap_draws.npz",
                        theta_llm=theta_llm.astype(np.float32), theta_human=theta_hum.astype(np.float32),
                        **{k: v.astype(np.float32) for k, v in draws.items()})
    (table_dir / "human_ai_gap.tex").write_text(latex_table(df, hdi_prob, M, P, S), encoding="utf-8")
    make_figures(df, figure_dir, hdi_prob)

    pd.set_option("display.width", 200)
    cols = ["scope", "p_human_mean", "p_human_hdi_low", "p_human_hdi_high", "p_llm_mean", "p_llm_hdi_low",
            "p_llm_hdi_high", "delta_mean", "delta_hdi_low", "delta_hdi_high", "log_odds_ratio_mean",
            "log_odds_ratio_hdi_low", "log_odds_ratio_hdi_high", "pr_delta_gt_0"]
    print(df[cols].to_string(index=False))
    logger.info("Wrote outputs to %s and %s", figure_dir, table_dir)


if __name__ == "__main__":
    jsonargparse.CLI(main, as_positional=False)
