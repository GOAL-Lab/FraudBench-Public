"""Synthetic evaluation frames for exercising the regression pipeline.

Rows are generated from the real metadata files (scenario ids, model ids
and human-baseline respondents) so that the ``merge_*`` processors join
exactly as they do on ``data/final/obs_data.parquet``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraudbench.analysis.regressions.process import CONTEXT_LEVELS, INSTRUCTION_LEVELS

INSTR = INSTRUCTION_LEVELS[1:]
CTX = CONTEXT_LEVELS[1:]
JUDGES = ["claude-haiku-4-5-20251001", "gemini-2.5-flash-lite", "gpt-5-nano-2025-08-07"]
_PREFIX = {"Claude": "anthropic/", "GPT": "openai/", "Gemini": "google/", "Grok": "grok/"}


def jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def make_frame(repo: Path, *, rows_per_model: int = 40, n_humans: int = 60, seed: int = 0) -> pd.DataFrame:
    """One ``obs_data``-like row per (model, draw) plus ``n_humans`` human rows."""
    rng = np.random.default_rng(seed)
    scen = [r["summary_id"] for r in jsonl(repo / "data/final/dataset_public.jsonl")]
    models = jsonl(repo / "data/final/models.jsonl")
    humans = jsonl(repo / "data/final/human_baseline/human_baseline_public.jsonl", limit=n_humans)

    rows, sid = [], 0
    for m in models:
        prefix = _PREFIX.get(m["family"], "hf-inference-providers/")
        for _ in range(rows_per_model):
            setting = rng.choice(["closed", "open", "agentic", "human_baseline"], p=[0.3, 0.3, 0.3, 0.1])
            instr = None if setting == "human_baseline" else rng.choice([None, *INSTR])
            ctx = rng.choice([None, *CTX])
            run = rng.choice(["main", "temperature_ablation", "reasoning_ablation"], p=[0.8, 0.1, 0.1])
            reasoning_on = bool(m["reasoning"]) or run == "reasoning_ablation"
            rows.append({
                "sample_id": f"s{sid}", "model": prefix + m["model"], "provider": "x",
                "setting": setting, "summary_id": rng.choice(scen),
                "human_respondent_id": "", "experiments": [c for c in [instr, ctx] if c],
                "run_category": run,
                "reasoning_budget": (m["max_reasoning_tokens"] if run == "reasoning_ablation" else 128.0)
                if m["max_reasoning_tokens"] else np.nan,
                "reasoning_effort": (m["max_reasoning_effort"] if run == "reasoning_ablation" else "minimal")
                if (m["max_reasoning_effort"] and not m["max_reasoning_tokens"]) else None,
                "tokens_reasoning": int(rng.integers(50, 5000)) if reasoning_on and rng.random() < 0.8 else 0,
                "score_fraud": int(rng.random() < 0.3),
                **{c: c == instr for c in INSTR}, **{c: c == ctx for c in CTX},
            })
            sid += 1
    for h in humans:
        exp = h["experiment"]
        rows.append({
            "sample_id": f"s{sid}", "model": "human", "provider": "",
            "setting": "human_baseline", "summary_id": h["summary_id"],
            "human_respondent_id": h["respondent_id"], "experiments": [] if exp == "control" else [exp],
            "run_category": "main", "reasoning_budget": np.nan, "reasoning_effort": None,
            "tokens_reasoning": 0, "score_fraud": int(rng.random() < 0.3),
            **{c: False for c in INSTR}, **{c: c == exp for c in CTX},
        })
        sid += 1
    df = pd.DataFrame(rows)
    df.loc[df.index[:3], "score_fraud"] = np.nan  # undefined majority votes
    return df


def make_per_score(frame: pd.DataFrame, *, seed: int = 1) -> pd.DataFrame:
    """Three judge votes per sample, ~1% unparseable."""
    rng = np.random.default_rng(seed)
    rows = [
        {"sample_id": sid, "scorer_model": j,
         "fraud_obs": None if rng.random() < 0.01 else bool(rng.random() < 0.3)}
        for sid in frame["sample_id"] for j in JUDGES
    ]
    return pd.DataFrame(rows)
