"""Strictly validate and merge agentic outputs onto a base scenarios dataset.

This script is an *intermediate* assembly step in the simulated-trajectory
pipeline. The official public release dataset
(``data/final/dataset_public.jsonl``) already ships ``agentic_trajectory``
and ``agentic_toolkits``, so this merger is only useful when regenerating
intermediates from a custom data pipeline.

This script expects valid JSON/JSONL inputs and emits clear parse errors
without attempting repair. If any input is invalid JSON, it exits non-zero.

Merge behavior (by summary_id):

- ``--input``: base scenarios JSONL (no default; supply the output of an
  upstream qa-vignettes / experiments generation step).
- ``--agentic_input`` (required): agentic-trajectory raw output:
    * agentic_trajectory
    * user_instruction -> agentic_user_instruction
    * model_sim_agent -> agentic_sim_agent
    * model_sim_simulator -> agentic_sim_simulator
    * model_sim_critiquer -> agentic_sim_critiquer
    * agentic_num_tool_calls recalculated from agentic_trajectory
- ``--toolkits_input`` (required): formatted-vignette JSONL (output of
  ``format_vignette_for_trajectory_simulation.py``):
    * toolkits
    * vignette_agentic
    * tool_responses
    * qa_agentic

Any missing rows or missing required fields are reported as errors and the
script exits non-zero without writing output, unless ``--drop_missing_rows`` is
set: then base rows with missing join rows and/or missing required merge fields
are skipped (warned) and the merge completes with the remaining rows.

Usage:
    uv run data_generation/simulated_trajectories/format_trajectory.py \\
        --input path/to/base_scenarios.jsonl \\
        --agentic_input path/to/agentic_raw.jsonl \\
        --toolkits_input path/to/format_vignette_final.jsonl \\
        --output_dir path/to/output_dir
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from jsonargparse import ArgumentParser


def _error_with_snippet(path: Path, content: str, exc: json.JSONDecodeError, context: str) -> str:
    lines = content.splitlines()
    line_text = ""
    if 1 <= exc.lineno <= len(lines):
        line_text = lines[exc.lineno - 1]
    pointer_col = max(1, exc.colno)
    pointer = " " * (pointer_col - 1) + "^"
    return (
        f"{context}\n"
        f"File: {path}\n"
        f"Line {exc.lineno}, Col {exc.colno}: {exc.msg}\n"
        f"Snippet: {line_text}\n"
        f"         {pointer}"
    )


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    content = path.read_text(encoding="utf-8")
    stripped = content.strip()
    if not stripped:
        raise ValueError(f"Input file is empty: {path}")

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(_error_with_snippet(path, content, exc, "Invalid JSON input.")) from exc

    if isinstance(parsed, dict):
        records: List[Any] = [parsed]
    elif isinstance(parsed, list):
        records = parsed
    else:
        raise ValueError(
            f"JSON input must decode to object or list of objects: {path} "
            f"(got {type(parsed).__name__})"
        )

    out: List[Dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(
                f"JSON record at index {idx} in {path} must be object, got {type(record).__name__}"
            )
        out.append(record)
    return out


def _load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError(f"Input file is empty: {path}")

    records: List[Dict[str, Any]] = []
    for line_no, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            pointer_col = max(1, exc.colno)
            pointer = " " * (pointer_col - 1) + "^"
            raise ValueError(
                "Invalid JSONL input.\n"
                f"File: {path}\n"
                f"Line {line_no}, Col {exc.colno}: {exc.msg}\n"
                f"Snippet: {line}\n"
                f"         {pointer}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                f"JSONL line {line_no} in {path} must decode to object, "
                f"got {type(parsed).__name__}"
            )
        records.append(parsed)

    if not records:
        raise ValueError(f"No JSONL records found in {path}")
    return records


def _load_records_strict(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl_records(path)
    if suffix == ".json":
        return _load_json_records(path)

    # Fallback for extension-less files: try JSON first, then JSONL.
    json_error: str | None = None
    try:
        return _load_json_records(path)
    except ValueError as exc:
        json_error = str(exc)
    try:
        return _load_jsonl_records(path)
    except ValueError as exc:
        raise ValueError(
            f"Unable to parse {path} as JSON or JSONL.\n"
            f"JSON parse error:\n{json_error}\n\n"
            f"JSONL parse error:\n{exc}"
        ) from exc


def _index_by_summary_id(records: List[Dict[str, Any]], *, source_name: str) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    for row_idx, record in enumerate(records, start=1):
        summary_id = record.get("summary_id")
        if not isinstance(summary_id, str) or not summary_id.strip():
            errors.append(f"[{source_name}] row {row_idx}: missing/invalid summary_id")
            continue
        if summary_id in index:
            errors.append(f"[{source_name}] duplicate summary_id '{summary_id}' at row {row_idx}")
            continue
        index[summary_id] = record

    if errors:
        raise ValueError("\n".join(errors))
    return index


def _require_field(
    row: Dict[str, Any],
    field: str,
    *,
    summary_id: str,
    source_name: str,
    errors: List[str],
) -> Any:
    if field not in row:
        errors.append(f"[{summary_id}] missing '{field}' in {source_name}")
        return None
    value = row[field]
    if value is None:
        errors.append(f"[{summary_id}] null '{field}' in {source_name}")
        return None
    return value


def _count_tool_messages(trajectory: Any, *, summary_id: str, errors: List[str]) -> int:
    if not isinstance(trajectory, list):
        errors.append(
            f"[{summary_id}] 'agentic_trajectory' must be list in agentic_raw "
            f"(got {type(trajectory).__name__})"
        )
        return 0

    count = 0
    for msg_idx, message in enumerate(trajectory):
        if not isinstance(message, dict):
            errors.append(
                f"[{summary_id}] agentic_trajectory[{msg_idx}] must be object "
                f"(got {type(message).__name__})"
            )
            continue
        if message.get("role") == "tool":
            count += 1
    return count


def _merge_records(
    base_records: List[Dict[str, Any]],
    agentic_records: List[Dict[str, Any]],
    toolkit_records: List[Dict[str, Any]],
    *,
    drop_missing_rows: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    base_by_id = _index_by_summary_id(base_records, source_name="base")
    agentic_by_id = _index_by_summary_id(agentic_records, source_name="agentic_raw")
    toolkit_by_id = _index_by_summary_id(toolkit_records, source_name="format_vignette_final")

    errors: List[str] = []
    warnings: List[str] = []
    merged: List[Dict[str, Any]] = []

    base_ids = set(base_by_id.keys())
    extra_agentic = sorted(set(agentic_by_id.keys()) - base_ids)
    extra_toolkits = sorted(set(toolkit_by_id.keys()) - base_ids)
    if extra_agentic:
        warnings.append(
            f"agentic_raw has {len(extra_agentic)} summary_id(s) not in base input "
            f"(first 10: {', '.join(extra_agentic[:10])})"
        )
    if extra_toolkits:
        warnings.append(
            f"format_vignette_final has {len(extra_toolkits)} summary_id(s) not in base input "
            f"(first 10: {', '.join(extra_toolkits[:10])})"
        )

    for base_record in base_records:
        summary_id = str(base_record["summary_id"])
        out_record = dict(base_record)
        row_errors: List[str] = []

        agentic_row = agentic_by_id.get(summary_id)
        toolkit_row = toolkit_by_id.get(summary_id)

        if agentic_row is None:
            row_errors.append(f"[{summary_id}] missing row in agentic_raw input")
        if toolkit_row is None:
            row_errors.append(f"[{summary_id}] missing row in format_vignette_final input")

        if agentic_row is not None:
            trajectory = _require_field(
                agentic_row,
                "agentic_trajectory",
                summary_id=summary_id,
                source_name="agentic_raw",
                errors=row_errors,
            )
            user_instruction = _require_field(
                agentic_row,
                "user_instruction",
                summary_id=summary_id,
                source_name="agentic_raw",
                errors=row_errors,
            )
            model_agent = _require_field(
                agentic_row,
                "model_sim_agent",
                summary_id=summary_id,
                source_name="agentic_raw",
                errors=row_errors,
            )
            model_simulator = _require_field(
                agentic_row,
                "model_sim_simulator",
                summary_id=summary_id,
                source_name="agentic_raw",
                errors=row_errors,
            )
            model_critiquer = _require_field(
                agentic_row,
                "model_sim_critiquer",
                summary_id=summary_id,
                source_name="agentic_raw",
                errors=row_errors,
            )

            if trajectory is not None:
                out_record["agentic_trajectory"] = trajectory
                out_record["agentic_num_tool_calls"] = _count_tool_messages(
                    trajectory, summary_id=summary_id, errors=row_errors
                )
            if user_instruction is not None:
                out_record["agentic_user_instruction"] = user_instruction
            if model_agent is not None:
                out_record["agentic_sim_agent"] = model_agent
            if model_simulator is not None:
                out_record["agentic_sim_simulator"] = model_simulator
            if model_critiquer is not None:
                out_record["agentic_sim_critiquer"] = model_critiquer

        if toolkit_row is not None:
            toolkits = _require_field(
                toolkit_row,
                "toolkits",
                summary_id=summary_id,
                source_name="format_vignette_final",
                errors=row_errors,
            )
            if toolkits is not None:
                out_record["agentic_toolkits"] = toolkits

            vignette_agentic = _require_field(
                toolkit_row,
                "vignette_agentic",
                summary_id=summary_id,
                source_name="format_vignette_final",
                errors=row_errors,
            )
            if vignette_agentic is not None:
                out_record["vignette_agentic"] = vignette_agentic

            tool_responses = _require_field(
                toolkit_row,
                "tool_responses",
                summary_id=summary_id,
                source_name="format_vignette_final",
                errors=row_errors,
            )
            if tool_responses is not None:
                out_record["tool_responses"] = tool_responses

            qa_agentic = _require_field(
                toolkit_row,
                "qa_agentic",
                summary_id=summary_id,
                source_name="format_vignette_final",
                errors=row_errors,
            )
            if qa_agentic is not None:
                out_record["qa_agentic"] = qa_agentic

        if row_errors:
            if drop_missing_rows:
                preview = "; ".join(row_errors[:3])
                if len(row_errors) > 3:
                    preview += f"; ... (+{len(row_errors) - 3} more)"
                warnings.append(f"[{summary_id}] skipped due to missing data: {preview}")
                continue
            errors.extend(row_errors)

        merged.append(out_record)

    return merged, errors, warnings


def _default_output_path(output_dir: Path, *, base_name: str = "agentic_trajectories_merged") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_dir / f"{base_name}_{timestamp}.jsonl"
    suffix = 1
    while candidate.exists():
        candidate = output_dir / f"{base_name}_{timestamp}_{suffix}.jsonl"
        suffix += 1
    return candidate


def _write_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    """CLI entry point: parse args, load inputs, run the strict merge, and write the merged JSONL."""
    parser = ArgumentParser(
        description=(
            "Strictly validate JSON inputs and merge agentic outputs onto base records."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Base scenarios JSONL (required, no default). Supply the output "
            "of an upstream qa-vignettes / experiments generation step."
        ),
    )
    parser.add_argument(
        "--agentic_input",
        type=str,
        required=True,
        help=(
            "Agentic input file (required). "
            "Recommended: output JSONL from fix_tool_responses.py."
        ),
    )
    parser.add_argument(
        "--toolkits_input",
        type=str,
        required=True,
        help=(
            "Formatted vignette JSONL (required, no default). Output of "
            "format_vignette_for_trajectory_simulation.py. Contributes "
            "toolkits, vignette_agentic, tool_responses, and qa_agentic."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSONL path (must not already exist)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=(
            "Directory for autogenerated output path when --output is omitted "
            "(no default; do NOT write to data/ in the public repo)."
        ),
    )
    parser.add_argument(
        "--drop_missing_rows",
        action="store_true",
        help=(
            "Skip base rows with missing data (missing join row and/or missing required "
            "fields from agentic_raw / format_vignette_final), warn, and continue."
        ),
    )
    args = parser.parse_args()

    base_path = Path(args.input).resolve()
    agentic_path = Path(args.agentic_input).resolve()
    toolkits_path = Path(args.toolkits_input).resolve()
    for input_path in (base_path, agentic_path, toolkits_path):
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.output:
        output_path = Path(args.output).resolve()
        if output_path.exists():
            raise FileExistsError(
                f"Output already exists: {output_path}. "
                "This script never overwrites existing files."
            )
    else:
        output_path = _default_output_path(Path(args.output_dir).resolve())

    try:
        base_records = _load_records_strict(base_path)
        agentic_records = _load_records_strict(agentic_path)
        toolkit_records = _load_records_strict(toolkits_path)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc

    print(f"Loaded base records: {len(base_records)} from {base_path}")
    print(f"Loaded agentic records: {len(agentic_records)} from {agentic_path}")
    print(f"Loaded toolkit records: {len(toolkit_records)} from {toolkits_path}")

    try:
        merged_records, errors, warnings = _merge_records(
            base_records=base_records,
            agentic_records=agentic_records,
            toolkit_records=toolkit_records,
            drop_missing_rows=args.drop_missing_rows,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc

    for warning in warnings:
        print(f"WARNING: {warning}")

    if args.drop_missing_rows:
        n_skip = len(base_records) - len(merged_records)
        if n_skip:
            print(f"Skipped {n_skip} base row(s) with missing agentic and/or toolkit joins.")

    if args.drop_missing_rows and not merged_records:
        print("ERROR: No output rows remained after --drop_missing_rows.")
        raise SystemExit(1)

    if errors:
        print(f"ERROR: found {len(errors)} merge issue(s):")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    _write_jsonl(merged_records, output_path)
    print(f"Wrote merged output: {output_path}")


if __name__ == "__main__":
    main()
