"""Lightweight merge of a raw agentic-trajectory JSONL onto a base seed JSONL.

A simpler alternative to ``format_trajectory.py`` that does no schema
validation: it just left-joins ``--merge-target`` rows onto agentic rows by
``summary_id`` and renames ``user_instruction`` to ``qa_agentic``.

Inputs:
    ``--trajectories``: JSONL of raw agentic outputs from
        ``simulate_trajectory.py`` (each row keyed on ``summary_id``).
    ``--merge-target``: optional JSONL of base seed rows (e.g. an
        intermediate produced by ``data_generation/qa/generate_vignettes.py``)
        whose columns are copied onto the agentic records.

There are no default paths: the public release ships finished
``agentic_trajectory`` / ``agentic_toolkits`` columns inside
``data/final/dataset_public.jsonl``, so this script is only relevant when
regenerating intermediates.

Usage:
    uv run data_generation/simulated_trajectories/agentic_merge.py \\
        --trajectories path/to/agentic_raw.jsonl \\
        --merge-target path/to/base_seeds.jsonl \\
        --output-path path/to/merged.jsonl
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

from jsonargparse import ArgumentParser


def load_jsonl_records(path: str, project_root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load records from a JSON or JSONL file with permissive parsing.

    Tries (1) parsing the whole file as a JSON value, then (2) falling back
    to one-record-per-line JSONL. Accepts both strict and ``strict=False``
    JSON (which tolerates unescaped control chars).

    Args:
        path: Input file path. May be relative if ``project_root`` is given.
        project_root: Optional base directory for resolving relative paths.

    Returns:
        List of decoded record dicts (empty if the file is empty).
    """
    if project_root and not os.path.isabs(path):
        path = os.path.join(project_root, path)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    if not content:
        return []
    
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        try:
            parsed = json.loads(content, strict=False)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass
    
    records: List[Dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append(json.loads(line, strict=False))
    return records


def merge_records_by_summary_id(
    agentic_records: List[Dict[str, Any]],
    base_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Left-join base records onto agentic records by ``summary_id``.

    Agentic fields override base fields on collision. After merging,
    ``user_instruction`` is moved to ``qa_agentic`` (only if ``qa_agentic``
    is not already present).

    Args:
        agentic_records: Driver list. Output preserves this order and only
            includes these summary_ids.
        base_records: Records to copy onto matching agentic rows.

    Returns:
        List of merged record dicts, one per agentic record.
    """
    merge_index = {
        str(item.get("summary_id")): item for item in base_records if item.get("summary_id") is not None
    }
    merged = []
    for record in agentic_records:
        key = str(record.get("summary_id"))
        base = merge_index.get(key, {}).copy()
        base.update(record)
        if "user_instruction" in base:
            if "qa_agentic" not in base:
                base["qa_agentic"] = base["user_instruction"]
            base.pop("user_instruction", None)
        merged.append(base)
    return merged


def load_and_merge_records(
    agentic_path: str,
    base_path: Optional[str] = None,
    project_root: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load agentic records and optionally merge in base records by summary_id.

    Args:
        agentic_path: Path to the agentic-trajectories JSON/JSONL file.
        base_path: Optional path to a base-records JSON/JSONL file.
        project_root: Optional base directory for resolving relative paths.

    Returns:
        Either the loaded agentic records (if ``base_path`` is None) or the
        merged records produced by :func:`merge_records_by_summary_id`.
    """
    agentic_records = load_jsonl_records(agentic_path, project_root=project_root)
    if not base_path:
        return agentic_records
    
    base_records = load_jsonl_records(base_path, project_root=project_root)
    return merge_records_by_summary_id(agentic_records, base_records)


def write_jsonl_records(
    records: List[Dict[str, Any]],
    output_path: Optional[str] = None,
) -> None:
    """Write records as JSONL to disk or stdout.

    Args:
        records: Iterable of dicts to serialize.
        output_path: Destination path. If None, writes to stdout.
    """
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    payload = "\n".join(lines) + ("\n" if lines else "")
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(payload)
    else:
        sys.stdout.write(payload)


def build_parser() -> ArgumentParser:
    """Construct the jsonargparse ``ArgumentParser`` for this script."""
    parser = ArgumentParser(
        description="Merge base records into agentic ones, keeping agentic rows only.",
    )
    parser.add_argument(
        "--trajectories",
        help="Path to agentic trajectories JSONL or JSON file.",
    )
    parser.add_argument(
        "--merge-target",
        "--merge_target",
        dest="merge_target",
        default=None,
        help="Path to existing/base JSONL or JSON file to merge onto.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output JSONL path. Defaults to stdout.",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Optional project root for relative paths.",
    )
    return parser


def main() -> None:
    """CLI entry point: parse args, load+merge inputs, write merged output."""
    parser = build_parser()
    args = parser.parse_args()
    records = load_and_merge_records(
        agentic_path=args.trajectories,
        base_path=args.merge_target,
        project_root=args.project_root,
    )
    write_jsonl_records(records, output_path=args.output_path)


if __name__ == "__main__":
    main()
