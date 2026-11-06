"""Tests for weighted human-validation agreement handling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudbench.analysis.judges import METRICS
from fraudbench.analysis.table.judge_irr import (
    HUMAN_SCORE_FIELD,
    MAJORITY_COL,
    ValidationDesign,
    _calibrated_weights,
    _human_consensus,
    _ratings_matrix,
    build_validation_design,
    compute_human_agreement,
    load_human_annotations,
    resolve_validation_sample_ids,
)


def _write_annotations(
    path: Path, rows: list[tuple[int | str, str, bool, bool, bool]]
) -> None:
    """Write ``(annotator_id, sample_id, fraud, refusal, options)`` rows as JSONL."""
    with path.open("w", encoding="utf-8") as handle:
        for annotator, sample_id, fraud, refusal, options in rows:
            record = {
                "annotator_id": annotator,
                "sample_id": sample_id,
                HUMAN_SCORE_FIELD["fraud"]: fraud,
                HUMAN_SCORE_FIELD["refusal"]: refusal,
                HUMAN_SCORE_FIELD["options"]: options,
            }
            handle.write(json.dumps(record) + "\n")


def test_build_validation_design_uses_population_and_full_sample_counts(
    tmp_path: Path,
) -> None:
    validated_path = tmp_path / "validated_samples.csv"
    pd.DataFrame({"Sample ID": ["s1", "s2", "s3"]}).to_csv(
        validated_path,
        index=False,
    )
    rows = [
        # Closed stratum: N_h=4, n_h=2.
        ("s1", "closed", "main", "model", "response", 0, 0),
        ("s2", "closed", "main", "model", "response", 0, 0),
        ("p1", "closed", "main", "model", "response", 0, 0),
        ("p2", "closed", "main", "model", "response", 0, 0),
        # Open stratum: N_h=3, n_h=1.
        ("s3", "open", "main", "model", "response", 1, 2),
        ("p3", "open", "main", "model", "response", 1, 2),
        ("p4", "open", "main", "model", "response", 1, 2),
        # Rows excluded by the original sampling-pool rules.
        ("ablation", "closed", "reasoning_ablation", "model", "response", 3, 3),
        ("ai-baseline", "human_baseline", "main", "model", "response", 3, 3),
        ("blank-agentic", "agentic", "main", "model", "", 3, 3),
    ]
    obs_path = tmp_path / "obs.csv"
    pd.DataFrame(
        rows,
        columns=[
            "sample_id",
            "setting",
            "run_category",
            "model",
            "completion_extended",
            "grader_counts_fraud",
            "grader_counts_refusal",
        ],
    ).to_csv(obs_path, index=False)

    design = build_validation_design(obs_path, validated_path)
    weights = design.items.set_index("sample_id")["item_weight"]

    assert weights.to_dict() == {"s1": 2.0, "s2": 2.0, "s3": 3.0}
    assert weights.sum() == pytest.approx(7.0)
    assert design.population_sizes == {
        ("closed", 0, 0): 4,
        ("open", 1, 2): 3,
    }
    assert design.sample_sizes == {
        ("closed", 0, 0): 2,
        ("open", 1, 2): 1,
    }


def test_build_validation_design_never_includes_human_baseline(
    tmp_path: Path,
) -> None:
    validated_path = tmp_path / "validated_samples.csv"
    pd.DataFrame({"Sample ID": ["s1"]}).to_csv(validated_path, index=False)
    obs_path = tmp_path / "obs.csv"
    pd.DataFrame(
        [
            ("s1", "closed", "main", "model", "response", 0, 0),
            ("p1", "closed", "main", "model", "response", 0, 0),
            # Human-baseline rows never enter the population, whoever answered.
            ("h1", "human_baseline", "main", "human", "response", 1, 1),
            ("hp1", "human_baseline", "main", "model", "response", 1, 1),
        ],
        columns=[
            "sample_id",
            "setting",
            "run_category",
            "model",
            "completion_extended",
            "grader_counts_fraud",
            "grader_counts_refusal",
        ],
    ).to_csv(obs_path, index=False)

    design = build_validation_design(obs_path, validated_path)

    assert design.items["sample_id"].tolist() == ["s1"]
    assert design.items["item_weight"].tolist() == [2.0]
    assert design.population_sizes == {("closed", 0, 0): 2}
    assert design.sample_sizes == {("closed", 0, 0): 1}


def test_metric_specific_weights_recalibrate_retained_strata() -> None:
    retained = pd.DataFrame(
        {
            "stratum": [
                ("closed", 0, 0),
                ("closed", 0, 0),
                ("open", 1, 2),
            ]
        }
    )

    weights = _calibrated_weights(
        retained,
        {
            ("closed", 0, 0): 100,
            ("open", 1, 2): 100,
        },
    )

    assert weights.tolist() == [50.0, 50.0, 100.0]
    assert weights.sum() == pytest.approx(200.0)


def test_annotation_loader_merges_ratings_per_item_across_annotators(
    tmp_path: Path,
) -> None:
    path = tmp_path / "human_annotations.jsonl"
    _write_annotations(
        path,
        [
            (1, "s1", True, False, False),
            (2, "s1", False, False, True),
            (2, "s2", False, True, False),
            (3, "s1", True, True, True),
        ],
    )

    human = load_human_annotations(path)

    assert set(human["annotator_id"]) == {"1", "2", "3"}
    assert list(human.columns) == [
        "sample_id",
        "annotator_id",
        "fraud_human",
        "refusal_human",
        "options_human",
    ]
    assert human.groupby("sample_id").size().to_dict() == {"s1": 3, "s2": 1}
    assert resolve_validation_sample_ids(human) == ["s1", "s2"]


def test_annotation_loader_parses_string_booleans(tmp_path: Path) -> None:
    path = tmp_path / "human_annotations.jsonl"
    path.write_text(
        json.dumps(
            {
                "annotator_id": 1,
                "sample_id": " s1 ",
                "fraud_score": "TRUE",
                "refusal_score": "False",
                "options_score": 0,
            }
        )
        + "\n"
    )

    human = load_human_annotations(path)

    assert human.loc[0, "sample_id"] == "s1"
    assert bool(human.loc[0, "fraud_human"]) is True
    assert bool(human.loc[0, "refusal_human"]) is False
    assert bool(human.loc[0, "options_human"]) is False


def test_annotation_loader_checks_against_explicit_validation_sample_list(
    tmp_path: Path,
) -> None:
    validated_path = tmp_path / "validated_samples.csv"
    pd.DataFrame({"Sample ID": ["s2", "s1"]}).to_csv(validated_path, index=False)
    path = tmp_path / "human_annotations.jsonl"
    _write_annotations(
        path,
        [(1, "s1", True, False, False), (2, "s2", False, False, True)],
    )

    human = load_human_annotations(path, validated_path)

    assert resolve_validation_sample_ids(human, validated_path) == ["s2", "s1"]

    pd.DataFrame({"Sample ID": ["s1", "s2", "s3"]}).to_csv(validated_path, index=False)
    with pytest.raises(ValueError, match="missing 1 validation sample ID"):
        load_human_annotations(path, validated_path)


def test_annotation_loader_rejects_missing_fields_and_blank_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "human_annotations.jsonl"
    path.write_text(
        json.dumps({"annotator_id": 1, "sample_id": "s1", "fraud_score": True})
        + "\n"
    )
    with pytest.raises(ValueError, match="missing required fields"):
        load_human_annotations(path)

    _write_annotations(path, [(1, "", True, False, False)])
    with pytest.raises(ValueError, match="blank 'sample_id'"):
        load_human_annotations(path)

    _write_annotations(path, [("", "s1", True, False, False)])
    with pytest.raises(ValueError, match="blank 'annotator_id'"):
        load_human_annotations(path)


def test_annotation_loader_rejects_invalid_score_values(tmp_path: Path) -> None:
    path = tmp_path / "human_annotations.jsonl"
    path.write_text(
        json.dumps(
            {
                "annotator_id": 1,
                "sample_id": "s1",
                "fraud_score": True,
                "refusal_score": "maybe",
                "options_score": False,
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="invalid refusal_score"):
        load_human_annotations(path)


def test_annotation_loader_rejects_ids_outside_validation_sample(
    tmp_path: Path,
) -> None:
    validated_path = tmp_path / "validated_samples.csv"
    pd.DataFrame({"Sample ID": ["s1"]}).to_csv(validated_path, index=False)
    path = tmp_path / "human_annotations.jsonl"
    _write_annotations(
        path,
        [(1, "s1", True, False, False), (2, "outside", False, False, True)],
    )

    with pytest.raises(ValueError, match="absent from"):
        load_human_annotations(path, validated_path)


def test_annotation_loader_rejects_repeated_item_for_one_annotator(
    tmp_path: Path,
) -> None:
    path = tmp_path / "human_annotations.jsonl"
    _write_annotations(
        path,
        [(1, "s1", True, False, False), (1, "s1", False, False, False)],
    )

    with pytest.raises(ValueError, match="more than once for one annotator"):
        load_human_annotations(path)


def test_ratings_matrix_lays_out_ratings_by_slot() -> None:
    human = pd.DataFrame(
        {
            "sample_id": ["s2", "s1", "s1", "s1"],
            "annotator_id": ["b", "c", "a", "b"],
            "fraud_human": [True, False, True, True],
        }
    )

    matrix = _ratings_matrix(human, "fraud_human")

    assert matrix.columns.tolist() == ["rating_1", "rating_2", "rating_3"]
    assert matrix.loc["s1"].tolist() == [True, True, False]
    assert bool(matrix.loc["s2", "rating_1"]) is True
    assert matrix.loc["s2"].isna().tolist() == [False, True, True]


def test_human_fleiss_kappa_is_invariant_to_rating_batch_labels() -> None:
    rng = np.random.default_rng(3)
    item_ids = [f"s{i}" for i in range(40)]
    rows: list[dict[str, object]] = []
    for sample_id in item_ids:
        n_ratings = int(rng.integers(2, 5))
        for slot in range(n_ratings):
            row: dict[str, object] = {"sample_id": sample_id, "annotator_id": f"r{slot}"}
            row.update({f"{metric}_human": bool(rng.integers(0, 2)) for metric in METRICS})
            rows.append(row)
    human = pd.DataFrame(rows)
    # Same ratings, but every rating comes from its own uniquely named file.
    relabelled = human.assign(annotator_id=[f"file_{i}" for i in range(len(human))])

    design = ValidationDesign(
        items=pd.DataFrame(
            {
                "sample_id": item_ids,
                "item_weight": [1.0] * len(item_ids),
                "stratum": [("closed", 0, 0)] * len(item_ids),
            }
        ),
        population_sizes={("closed", 0, 0): len(item_ids)},
        sample_sizes={("closed", 0, 0): len(item_ids)},
    )
    obs = pd.DataFrame(
        {
            "sample_id": item_ids,
            **{MAJORITY_COL[metric]: [True] * len(item_ids) for metric in METRICS},
        }
    )

    _, fleiss_a = compute_human_agreement(obs, human, design, seed=1)
    _, fleiss_b = compute_human_agreement(obs, relabelled, design, seed=1)

    for metric in METRICS:
        assert fleiss_a[metric].n == len(item_ids)
        assert fleiss_a[metric].kappa == pytest.approx(fleiss_b[metric].kappa)
        assert fleiss_a[metric].observed_agreement == pytest.approx(
            fleiss_b[metric].observed_agreement
        )


def test_seeded_consensus_keeps_singletons_and_weighted_fleiss_drops_them() -> None:
    rows: list[dict[str, object]] = []
    ratings = {
        "s1": [("A", True)],
        "s2": [("A", False), ("B", True)],
        "s3": [("A", False), ("B", False)],
        "s4": [("A", True), ("B", True)],
    }
    for sample_id, sample_ratings in ratings.items():
        for annotator, vote in sample_ratings:
            row: dict[str, object] = {
                "sample_id": sample_id,
                "annotator_id": annotator,
            }
            row.update({f"{metric}_human": vote for metric in METRICS})
            rows.append(row)
    human = pd.DataFrame(rows)

    first = _human_consensus(human, "fraud", np.random.default_rng(7))
    second = _human_consensus(human, "fraud", np.random.default_rng(7))
    pd.testing.assert_series_equal(first, second)
    assert bool(first.loc["s1"]) is True

    item_ids = ["s1", "s2", "s3", "s4"]
    design = ValidationDesign(
        items=pd.DataFrame(
            {
                "sample_id": item_ids,
                "item_weight": [1.0] * 4,
                "stratum": [("closed", 0, 0)] * 4,
            }
        ),
        population_sizes={("closed", 0, 0): 4},
        sample_sizes={("closed", 0, 0): 4},
    )
    obs = pd.DataFrame(
        {
            "sample_id": item_ids,
            **{
                MAJORITY_COL[metric]: [True, False, False, True]
                for metric in METRICS
            },
        }
    )

    cohen_a, fleiss_a = compute_human_agreement(obs, human, design, seed=7)
    cohen_b, fleiss_b = compute_human_agreement(obs, human, design, seed=7)

    for metric in METRICS:
        assert cohen_a[metric] == cohen_b[metric]
        assert fleiss_a[metric] == fleiss_b[metric]
        assert cohen_a[metric].n == 4
        assert fleiss_a[metric].n == 3
        assert fleiss_a[metric].kappa == pytest.approx(1.0 / 3.0)
