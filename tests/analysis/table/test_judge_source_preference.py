"""Tests for the source-associated LLM-judge statistics.

The table module reduces every reported number to a ratio of per-case sums so
the clustered bootstrap stays cheap. These tests pin that machinery against
hand-computed fixtures and against a naive per-vote reference implementation.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fraudbench.analysis.common import ensure_model_id_column
from fraudbench.analysis.judges import VOTES_COLUMN, collapse_per_score_rows
from fraudbench.analysis.table.judge_source_preference import (
    SETTING_ORDER,
    SOURCE_FAMILY,
    SOURCE_KEYS,
    SOURCE_OTHER,
    SOURCE_SELF,
    BootstrapConfig,
    aggregate_by_case,
    bootstrap_bounds,
    case_ids,
    compute_metric_result,
    reduce_scope,
    source_matrix,
    summarise,
)


#: Judge ids in the order used by the fixtures below (Claude, GPT, Gemini).
JUDGES: tuple[str, ...] = (
    "claude-haiku-4-5-20251001",
    "gpt-5-nano-2025-08-07",
    "gemini-2.5-flash-lite",
)

#: One generating model per source relation to the GPT judge.
GPT_SELF = "openai/gpt-5-nano-2025-08-07"
GPT_FAMILY = "openai/gpt-5-2025-08-07"
OTHER_FAMILY = "grok/grok-4-1-fast-reasoning"


def make_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal obs-data-shaped frame from ``{model, setting, ...}`` rows.

    ``fraud`` / ``refusal`` values are tuples of ``bool | None`` aligned with
    :data:`JUDGES`; ``None`` stands for a vote the grader failed to parse.
    """
    frame = pd.DataFrame(
        {
            "model": [row["model"] for row in rows],
            "setting": [row.get("setting", "closed") for row in rows],
            "summary_id": [row.get("summary_id", "1001-1") for row in rows],
            VOTES_COLUMN["fraud"]: [
                json.dumps(dict(zip(JUDGES, row["fraud"]))) for row in rows
            ],
            VOTES_COLUMN["refusal"]: [
                json.dumps(dict(zip(JUDGES, row.get("refusal", row["fraud"]))))
                for row in rows
            ],
        }
    )
    frame = ensure_model_id_column(frame)
    frame["case_id"] = case_ids(frame)
    return frame


def statistics_for(frame: pd.DataFrame, metric: str = "fraud"):
    """Pooled (all-task) statistics for *frame*."""
    totals = aggregate_by_case(frame, metric, JUDGES, settings=SETTING_ORDER)
    return summarise(reduce_scope(totals, None).sum(axis=0))


def reference_statistics(
    frame: pd.DataFrame, metric: str, judges: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Naive per-vote reference for the positive rate and co-judge differential.

    Loops over rows and judges rather than reducing to per-case sums, so it
    exercises none of the vectorised code path under test.
    """
    rate_sums = np.zeros((len(judges), len(SOURCE_KEYS)))
    rate_counts = np.zeros_like(rate_sums)
    diff_sums = np.zeros_like(rate_sums)
    diff_counts = np.zeros_like(rate_sums)
    sources = source_matrix(frame["model_id"], judges)

    for row_position, (_, row) in enumerate(frame.iterrows()):
        votes = json.loads(row[VOTES_COLUMN[metric]])
        values = [votes.get(judge) for judge in judges]
        complete = all(value is not None for value in values)
        for position in range(len(judges)):
            source = sources[row_position, position]
            value = values[position]
            if value is not None:
                rate_sums[position, source] += float(value)
                rate_counts[position, source] += 1.0
            if complete:
                others = [
                    float(v) for index, v in enumerate(values) if index != position
                ]
                diff_sums[position, source] += float(value) - sum(others) / len(others)
                diff_counts[position, source] += 1.0

    with np.errstate(invalid="ignore"):
        return rate_sums / rate_counts, diff_sums / diff_counts


def test_case_ids_strip_the_summary_index() -> None:
    frame = pd.DataFrame({"summary_id": ["1004-1", "1004-2", "1057-1"]})
    assert list(case_ids(frame)) == ["1004", "1004", "1057"]


def test_case_ids_reject_unparseable_summary_ids() -> None:
    frame = pd.DataFrame({"summary_id": ["1004-1", ""]})
    with pytest.raises(ValueError, match="case id"):
        case_ids(frame)


def test_source_matrix_splits_self_family_and_other() -> None:
    model_ids = pd.Series(
        ["gpt-5-nano-2025-08-07", "gpt-5-2025-08-07", "grok-4-1-fast-reasoning"]
    )
    sources = source_matrix(model_ids, JUDGES)
    gpt_judge = JUDGES.index("gpt-5-nano-2025-08-07")
    assert list(sources[:, gpt_judge]) == [SOURCE_SELF, SOURCE_FAMILY, SOURCE_OTHER]
    # From the Claude judge's seat, all three responses are different-family.
    claude_judge = JUDGES.index("claude-haiku-4-5-20251001")
    assert set(sources[:, claude_judge]) == {SOURCE_OTHER}


def test_per_score_rows_collapse_to_one_response_with_vote_dicts() -> None:
    rows = []
    votes = (True, False, None)
    for judge, vote in zip(JUDGES, votes):
        rows.append(
            {
                "sample_id": "sample-1",
                "model": GPT_SELF,
                "setting": "closed",
                "summary_id": "1001-1",
                "scorer_model": judge,
                "fraud_obs": vote,
                "refusal_obs": False,
                "option_obs": True,
                "grader_ran": True,
            }
        )

    collapsed = collapse_per_score_rows(pd.DataFrame(rows), ("fraud", "refusal"))

    assert len(collapsed) == 1
    assert collapsed.loc[0, "sample_id"] == "sample-1"
    assert collapsed.loc[0, VOTES_COLUMN["fraud"]] == dict(zip(JUDGES, votes))
    assert collapsed.loc[0, VOTES_COLUMN["refusal"]] == dict.fromkeys(JUDGES, False)
    assert "scorer_model" not in collapsed.columns
    assert "option_obs" not in collapsed.columns


def test_per_score_collapse_rejects_duplicate_sample_scorer_rows() -> None:
    row = {
        "sample_id": "sample-1",
        "scorer_model": JUDGES[0],
        "fraud_obs": True,
        "grader_ran": True,
    }
    with pytest.raises(ValueError, match=r"duplicate \(sample_id, scorer_model\)"):
        collapse_per_score_rows(pd.DataFrame([row, row]), ("fraud",))


def test_gpt_oss_is_not_part_of_the_gpt_family() -> None:
    """Families come from ``models.jsonl``, where GPT-OSS stands on its own."""
    sources = source_matrix(pd.Series(["openai/gpt-oss-120b"]), JUDGES)
    assert sources[0, JUDGES.index("gpt-5-nano-2025-08-07")] == SOURCE_OTHER


def test_raw_rate_counts_every_parseable_vote() -> None:
    frame = make_frame(
        [
            {"model": GPT_SELF, "fraud": (True, True, False)},
            {"model": GPT_SELF, "fraud": (True, False, None)},
        ]
    )
    statistics = statistics_for(frame)
    gpt = JUDGES.index("gpt-5-nano-2025-08-07")
    gemini = JUDGES.index("gemini-2.5-flash-lite")
    assert statistics.rate[gpt, SOURCE_SELF] == pytest.approx(0.5)
    assert statistics.n_votes[gpt, SOURCE_SELF] == 2
    # A GPT-generated response is different-family from the Gemini judge's
    # seat, where the unparseable vote leaves a single counted vote.
    assert statistics.rate[gemini, SOURCE_OTHER] == pytest.approx(0.0)
    assert statistics.n_votes[gemini, SOURCE_OTHER] == 1


def test_paired_statistics_drop_incomplete_juries() -> None:
    frame = make_frame(
        [
            {"model": GPT_SELF, "fraud": (True, True, False)},
            {"model": GPT_SELF, "fraud": (True, False, None)},
        ]
    )
    statistics = statistics_for(frame)
    gpt = JUDGES.index("gpt-5-nano-2025-08-07")
    # Only the first row has a full jury: 1 - (1 + 0) / 2 = 0.5.
    assert statistics.n_jury[gpt, SOURCE_SELF] == 1
    assert statistics.differential[gpt, SOURCE_SELF] == pytest.approx(0.5)


def test_excess_contrasts_net_out_the_different_family_differential() -> None:
    frame = make_frame(
        [
            {"model": GPT_SELF, "fraud": (True, True, False)},
            {"model": GPT_FAMILY, "fraud": (False, True, False)},
            {"model": OTHER_FAMILY, "fraud": (True, True, True)},
        ]
    )
    statistics = statistics_for(frame)
    gpt = JUDGES.index("gpt-5-nano-2025-08-07")
    self_differential = statistics.differential[gpt, SOURCE_SELF]
    family_differential = statistics.differential[gpt, SOURCE_FAMILY]
    other_differential = statistics.differential[gpt, SOURCE_OTHER]
    assert self_differential == pytest.approx(0.5)
    assert family_differential == pytest.approx(1.0)
    assert other_differential == pytest.approx(0.0)
    assert statistics.exact_self_excess[gpt] == pytest.approx(0.5)
    assert statistics.family_excess[gpt] == pytest.approx(1.0)


def test_lone_votes_and_net_dissent() -> None:
    frame = make_frame(
        [
            # GPT judge is the lone negative vote.
            {"model": GPT_SELF, "fraud": (True, False, True)},
            # GPT judge is the lone positive vote.
            {"model": GPT_SELF, "fraud": (False, True, False)},
            # Unanimous: contributes to the denominator only.
            {"model": GPT_SELF, "fraud": (True, True, True)},
        ]
    )
    statistics = statistics_for(frame)
    gpt = JUDGES.index("gpt-5-nano-2025-08-07")
    assert statistics.lone_negative[gpt, SOURCE_SELF] == pytest.approx(1 / 3)
    assert statistics.lone_positive[gpt, SOURCE_SELF] == pytest.approx(1 / 3)
    assert statistics.net_dissent[gpt, SOURCE_SELF] == pytest.approx(0.0)
    assert statistics.n_jury[gpt, SOURCE_SELF] == 3


def test_macro_average_is_unweighted_across_judges() -> None:
    frame = make_frame(
        [
            {"model": OTHER_FAMILY, "fraud": (True, False, False)},
            {"model": OTHER_FAMILY, "fraud": (True, True, False)},
        ]
    )
    statistics = statistics_for(frame)
    # Claude votes positive twice, GPT once, Gemini never: (1 + 0.5 + 0) / 3.
    assert statistics.macro_rate[SOURCE_OTHER] == pytest.approx(0.5)
    assert statistics.macro_rate[SOURCE_OTHER] == pytest.approx(
        np.mean(statistics.rate[:, SOURCE_OTHER])
    )
    assert statistics.macro_n_votes[SOURCE_OTHER] == 6


def test_empty_cells_are_nan_rather_than_zero() -> None:
    frame = make_frame([{"model": OTHER_FAMILY, "fraud": (True, True, True)}])
    statistics = statistics_for(frame)
    assert np.isnan(statistics.rate[:, SOURCE_SELF]).all()
    assert np.isnan(statistics.exact_self_excess).all()
    assert np.isnan(statistics.macro_exact_self_excess)


def test_matches_naive_reference_on_random_data() -> None:
    rng = np.random.default_rng(20260727)
    models = [GPT_SELF, GPT_FAMILY, OTHER_FAMILY, "anthropic/claude-opus-4-5-20251101"]
    rows = []
    for index in range(400):
        votes = [
            None if rng.random() < 0.05 else bool(rng.random() < 0.4) for _ in JUDGES
        ]
        rows.append(
            {
                "model": models[int(rng.integers(len(models)))],
                "setting": SETTING_ORDER[int(rng.integers(len(SETTING_ORDER)))],
                "summary_id": f"{1000 + int(rng.integers(7))}-{1 + index % 2}",
                "fraud": tuple(votes),
            }
        )
    frame = make_frame(rows)

    statistics = statistics_for(frame)
    expected_rate, expected_differential = reference_statistics(frame, "fraud", JUDGES)
    np.testing.assert_allclose(statistics.rate, expected_rate)
    np.testing.assert_allclose(statistics.differential, expected_differential)


def test_per_task_scopes_partition_the_pooled_totals() -> None:
    frame = make_frame(
        [
            {"model": GPT_SELF, "setting": "closed", "fraud": (True, True, False)},
            {"model": GPT_SELF, "setting": "open", "fraud": (False, False, False)},
            {"model": GPT_SELF, "setting": "agentic", "fraud": (True, True, True)},
        ]
    )
    result = compute_metric_result(frame, "fraud", JUDGES, settings=SETTING_ORDER)
    per_task = sum(
        result.scopes[setting].statistics.n_votes for setting in SETTING_ORDER
    )
    np.testing.assert_allclose(result.overall.statistics.n_votes, per_task)
    assert result.n_full_jury == 3


def test_bootstrap_bounds_are_ordered_and_reproducible() -> None:
    rng = np.random.default_rng(11)
    rows = [
        {
            "model": GPT_SELF if index % 2 else OTHER_FAMILY,
            "summary_id": f"{1000 + index % 10}-1",
            "fraud": tuple(bool(rng.random() < 0.5) for _ in JUDGES),
        }
        for index in range(200)
    ]
    frame = make_frame(rows)
    case_totals = reduce_scope(
        aggregate_by_case(frame, "fraud", JUDGES, settings=SETTING_ORDER), None
    )
    config = BootstrapConfig(resamples=64, seed=7)

    lower, upper = bootstrap_bounds(case_totals, config)
    repeated_lower, _ = bootstrap_bounds(case_totals, config)
    populated = np.isfinite(lower.rate)
    assert populated.any()
    assert np.all(lower.rate[populated] <= upper.rate[populated])
    np.testing.assert_allclose(lower.rate, repeated_lower.rate)

    reseeded_lower, _ = bootstrap_bounds(case_totals, BootstrapConfig(64, seed=8))
    assert not np.allclose(
        lower.rate[populated], reseeded_lower.rate[populated]
    )
