"""Read-only validator for formatted agentic trajectories.

Checks each record's ``messages``/``agentic_trajectory`` against the
registered ToolEmu toolkits:

1) ToolEmu compliance for tool calls and tool responses (function names,
   required parameters, parameter and return types, ``tool_call_id``
   formatting and pairing between assistant ``tool_calls`` and ``tool``
   role responses).
2) Per-trajectory datatype consistency for tool-call arguments and
   tool-response payload fields (scoped to a single ``summary_id``).

This script never rewrites files. It prints issues grouped by
``summary_id`` so records can be fixed manually upstream and exits
non-zero when any validation issue is found.

Usage:
    uv run data_generation/simulated_trajectories/validate_tools.py \\
        --input path/to/agentic_fixed.jsonl
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from jsonargparse import ArgumentParser

from fraudbench.utils.inspect_tools import TOOLKIT_CLASSES


@dataclass(frozen=True)
class ToolSpec:
    parameter_types: Dict[str, str]
    required_parameters: frozenset[str]
    return_types: Dict[str, str]


@dataclass(frozen=True)
class MalformedRowIssue:
    row: int
    summary_id: str
    row_start_line: int
    error_line: int
    error_col: int
    error: str
    snippet: str


SUMMARY_ID_LINE_PATTERN = re.compile(r'(?m)^\s*"summary_id"\s*:\s*"([^"]+)"')
SUMMARY_ID_OBJECT_START_PATTERN = re.compile(r'\{\s*"summary_id"\s*:\s*"')


def _summary_record_starts(content: str, summary_matches: List[re.Match[str]]) -> List[int]:
    starts = [m.start() for m in SUMMARY_ID_OBJECT_START_PATTERN.finditer(content)]
    if len(starts) == len(summary_matches):
        return starts

    aligned_starts: List[int] = []
    start_idx = 0
    for summary_match in summary_matches:
        while start_idx + 1 < len(starts) and starts[start_idx + 1] <= summary_match.start():
            start_idx += 1
        if starts and starts[start_idx] <= summary_match.start():
            aligned_starts.append(starts[start_idx])
            continue
        line_start = content.rfind("\n{", 0, summary_match.start())
        if line_start != -1:
            aligned_starts.append(line_start + 1)
            continue
        brace_start = content.rfind("{", 0, summary_match.start())
        aligned_starts.append(brace_start if brace_start >= 0 else summary_match.start())
    return aligned_starts


def _find_malformed_rows(content: str) -> List[MalformedRowIssue]:
    summary_matches = list(SUMMARY_ID_LINE_PATTERN.finditer(content))
    if not summary_matches:
        return []

    starts = _summary_record_starts(content, summary_matches)
    if not starts:
        return []

    malformed_rows: List[MalformedRowIssue] = []
    for row_idx, summary_match in enumerate(summary_matches):
        summary_id = summary_match.group(1)
        start = starts[row_idx] if row_idx < len(starts) else summary_match.start()
        end = starts[row_idx + 1] if row_idx + 1 < len(starts) else len(content)
        chunk = content[start:end]
        if row_idx + 1 < len(starts):
            last_close = chunk.rfind("}")
            if last_close != -1:
                chunk = chunk[: last_close + 1]
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            json.loads(chunk)
        except json.JSONDecodeError as exc:
            row_start_line = content.count("\n", 0, start) + 1
            error_line = row_start_line + exc.lineno - 1
            chunk_lines = chunk.splitlines()
            snippet = ""
            if 0 <= exc.lineno - 1 < len(chunk_lines):
                snippet = chunk_lines[exc.lineno - 1].strip()
            malformed_rows.append(
                MalformedRowIssue(
                    row=row_idx + 1,
                    summary_id=summary_id,
                    row_start_line=row_start_line,
                    error_line=error_line,
                    error_col=exc.colno,
                    error=exc.msg,
                    snippet=snippet,
                )
            )
    return malformed_rows


def _parse_json_by_brace(content: str) -> List[Any]:
    values: List[str] = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for idx, ch in enumerate(content):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            if depth == 0:
                start = idx
            depth += 1
        elif ch in "}]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    values.append(content[start : idx + 1])
                    start = None

    records: List[Any] = []
    for value in values:
        parsed: Any = None
        try:
            parsed = json.loads(value, strict=False)
        except json.JSONDecodeError:
            repaired = re.sub(r",\s*([}\]])", r"\1", value)
            try:
                parsed = json.loads(repaired, strict=False)
            except json.JSONDecodeError:
                parsed = None
        if parsed is None:
            continue
        if isinstance(parsed, list):
            records.extend(parsed)
        else:
            records.append(parsed)
    return records


def _offset_to_line_col(content: str, offset: int) -> Tuple[int, int]:
    line = content.count("\n", 0, offset) + 1
    last_newline = content.rfind("\n", 0, offset)
    col = offset - last_newline
    return line, col


def _parse_json_content_strict(content: str) -> Tuple[List[Any], List[str]]:
    content = content.strip()
    if not content:
        return [], []
    try:
        parsed = json.loads(content)
        return (parsed if isinstance(parsed, list) else [parsed]), []
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    idx = 0
    decoded_records: List[Any] = []
    issues: List[str] = []
    while idx < len(content):
        while idx < len(content) and content[idx].isspace():
            idx += 1
        if idx >= len(content):
            break
        try:
            obj, next_idx = decoder.raw_decode(content, idx)
        except json.JSONDecodeError as exc:
            line, col = _offset_to_line_col(content, exc.pos)
            issues.append(
                f"[json] strict parse error at line {line}, col {col}: {exc.msg}"
            )
            break
        decoded_records.extend(obj if isinstance(obj, list) else [obj])
        idx = next_idx
    if decoded_records and not issues:
        return decoded_records, []
    if issues:
        return [], issues

    # Fallback issue (strict whole-file and strict stream parsing both failed).
    return [], ["[json] strict parsing failed (invalid quotes/brackets/JSON tokens)."]


def _parse_json_content_lenient(content: str) -> List[Any]:
    content = content.strip()
    if not content:
        return []
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        pass
    try:
        parsed = json.loads(content, strict=False)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder(strict=False)
    idx = 0
    decoded_records: List[Any] = []
    while idx < len(content):
        while idx < len(content) and content[idx].isspace():
            idx += 1
        if idx >= len(content):
            break
        try:
            obj, next_idx = decoder.raw_decode(content, idx)
        except json.JSONDecodeError:
            decoded_records = []
            break
        decoded_records.extend(obj if isinstance(obj, list) else [obj])
        idx = next_idx
    if decoded_records:
        return decoded_records

    by_brace = _parse_json_by_brace(content)
    if by_brace:
        return by_brace

    records: List[Any] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            try:
                records.append(json.loads(line, strict=False))
            except json.JSONDecodeError:
                continue
    return records


def load_records(path: str) -> Tuple[List[Any], List[str], List[MalformedRowIssue]]:
    """Parse ``path`` permissively and surface row-level malformations.

    Tries strict whole-file/streamed JSON first, falling back to lenient
    parsing (``strict=False``, brace-walking, or per-line JSONL) if needed.

    Args:
        path: Input JSON or JSONL file.

    Returns:
        Tuple ``(records, json_issues, malformed_rows)`` where
        ``records`` is the decoded list, ``json_issues`` describes
        any whole-file parse failures, and ``malformed_rows`` lists
        per-record decode errors located via the ``summary_id`` key.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    malformed_rows = _find_malformed_rows(content)
    strict_records, json_issues = _parse_json_content_strict(content)
    if strict_records:
        return strict_records, [], malformed_rows
    lenient_records = _parse_json_content_lenient(content)
    if not lenient_records:
        json_issues = list(json_issues)
        json_issues.append("[json] unable to recover any records with lenient parsing.")
    return lenient_records, json_issues, malformed_rows


def load_tool_specs() -> Dict[str, ToolSpec]:
    """Reflect the registered ToolEmu toolkits into a ``{tool_name: ToolSpec}`` map.

    Each ``ToolSpec`` records parameter types, the set of required
    parameters, and return-field types so :func:`validate_records` can
    enforce schema conformance against actual trajectory messages.
    """
    specs: Dict[str, ToolSpec] = {}
    for toolkit_class in TOOLKIT_CLASSES:
        toolkit = toolkit_class()
        for tool_obj in toolkit.tools:
            param_types: Dict[str, str] = {}
            required = set()
            for param in tool_obj.parameters or []:
                name = param.get("name")
                if not isinstance(name, str) or not name:
                    continue
                param_types[name] = str(param.get("type", "string")).lower()
                if param.get("required", False):
                    required.add(name)
            return_types: Dict[str, str] = {}
            for ret in tool_obj.returns or []:
                name = ret.get("name")
                if not isinstance(name, str) or not name:
                    continue
                return_types[name] = str(ret.get("type", "string")).lower()
            specs[tool_obj.name] = ToolSpec(
                parameter_types=param_types,
                required_parameters=frozenset(required),
                return_types=return_types,
            )
    return specs


def _normalize_arguments(arguments: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if arguments is None:
        return {}, None
    if isinstance(arguments, dict):
        return arguments, None
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return None, "arguments is a string but not valid JSON"
        if isinstance(parsed, dict):
            return parsed, None
        return None, f"arguments JSON must decode to object, got {type(parsed).__name__}"
    return None, f"arguments must be object or JSON object string, got {type(arguments).__name__}"


def _parse_tool_content(content: Any) -> Tuple[Any, Optional[str]]:
    if not isinstance(content, str):
        return content, None
    stripped = content.strip()
    if not stripped:
        return content, None
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return content, None
    try:
        return json.loads(stripped), None
    except json.JSONDecodeError as exc:
        line, col = _offset_to_line_col(stripped, exc.pos)
        parse_err = f"JSON parse error at line {line}, col {col}: {exc.msg}"
        try:
            return json.loads(stripped, strict=False), parse_err
        except json.JSONDecodeError:
            return content, parse_err


def _value_kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _type_matches(value: Any, expected_type: str) -> bool:
    observed = _value_kind(value)
    if observed == "null":
        # ToolEmu does not encode requiredness for return fields.
        return True
    if expected_type == "number":
        return observed in ("number", "integer")
    return observed == expected_type


def _consistency_kind(value: Any) -> str:
    kind = _value_kind(value)
    if kind == "integer":
        return "number"
    return kind


def _collect_field_types(
    value: Any,
    *,
    tool_name: str,
    scope: str,
    path: str,
    field_types: Dict[Tuple[str, str, str], set[str]],
) -> None:
    kind = _consistency_kind(value)
    if kind == "null":
        return
    key = (tool_name, scope, path)
    field_types[key].add(kind)

    if isinstance(value, dict):
        for child_key, child_value in value.items():
            child_path = f"{path}.{child_key}" if path else str(child_key)
            _collect_field_types(
                child_value,
                tool_name=tool_name,
                scope=scope,
                path=child_path,
                field_types=field_types,
            )
    elif isinstance(value, list):
        child_path = f"{path}[]" if path else "[]"
        for child_value in value:
            _collect_field_types(
                child_value,
                tool_name=tool_name,
                scope=scope,
                path=child_path,
                field_types=field_types,
            )


def _validate_summary_record(
    *,
    summary_id: str,
    messages: List[Any],
    tool_specs: Dict[str, ToolSpec],
) -> List[str]:
    issues: List[str] = []
    seen_tool_call_ids: Dict[str, str] = {}
    field_types: Dict[Tuple[str, str, str], set[str]] = defaultdict(set)

    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            issues.append(f"[spec] message[{msg_idx}] must be object, got {type(msg).__name__}")
            continue
        role = msg.get("role")

        if role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls is None:
                tool_calls = []
            if not isinstance(tool_calls, list):
                issues.append(f"[spec] assistant message[{msg_idx}] tool_calls must be list")
                continue

            for tool_call in tool_calls:
                tool_call_id_label = "<missing>"
                if isinstance(tool_call, dict):
                    maybe_tool_call_id = tool_call.get("id")
                    if isinstance(maybe_tool_call_id, str) and maybe_tool_call_id:
                        tool_call_id_label = maybe_tool_call_id
                where = f"tool_call[{tool_call_id_label}]"
                if not isinstance(tool_call, dict):
                    issues.append(f"[spec] {where} must be object")
                    continue

                tool_name = tool_call.get("function")
                if not isinstance(tool_name, str) or not tool_name:
                    issues.append(f"[spec] {where} missing function")
                    continue
                tool_spec = tool_specs.get(tool_name)
                if tool_spec is None:
                    issues.append(f"[spec] {where} unknown tool '{tool_name}'")
                    continue

                args, arg_err = _normalize_arguments(tool_call.get("arguments"))
                if arg_err:
                    issues.append(f"[spec] {where} {arg_err}")
                    continue
                assert args is not None

                for required_key in sorted(tool_spec.required_parameters):
                    if required_key not in args:
                        issues.append(
                            f"[spec] {where} missing required '{required_key}' for {tool_name}"
                        )

                for arg_key, arg_value in args.items():
                    if arg_key not in tool_spec.parameter_types:
                        issues.append(
                            f"[spec] {where} unexpected argument '{arg_key}' for {tool_name}"
                        )
                        continue
                    expected_type = tool_spec.parameter_types[arg_key]
                    if not _type_matches(arg_value, expected_type):
                        observed_type = _value_kind(arg_value)
                        issues.append(
                            f"[spec] {where} arg '{arg_key}' expected {expected_type}, got {observed_type}"
                        )
                    _collect_field_types(
                        arg_value,
                        tool_name=tool_name,
                        scope="arguments",
                        path=arg_key,
                        field_types=field_types,
                    )

                tool_call_id = tool_call.get("id")
                if not isinstance(tool_call_id, str) or not tool_call_id:
                    issues.append(f"[spec] {where} missing id")
                elif not tool_call_id.startswith("tc_"):
                    issues.append(f"[spec] {where} id '{tool_call_id}' must start with tc_")
                elif tool_call_id in seen_tool_call_ids:
                    issues.append(f"[spec] {where} duplicate id '{tool_call_id}'")
                else:
                    seen_tool_call_ids[tool_call_id] = tool_name

        elif role == "tool":
            tool_call_id_label = "<missing>"
            maybe_tool_call_id = msg.get("tool_call_id")
            if isinstance(maybe_tool_call_id, str) and maybe_tool_call_id:
                tool_call_id_label = maybe_tool_call_id
            where = f"tool_response[{tool_call_id_label}]"
            tool_name = msg.get("function")
            if not isinstance(tool_name, str) or not tool_name:
                issues.append(f"[spec] {where} missing function")
                continue
            tool_spec = tool_specs.get(tool_name)
            if tool_spec is None:
                issues.append(f"[spec] {where} unknown tool '{tool_name}'")
                continue

            tool_call_id = msg.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                issues.append(f"[spec] {where} missing tool_call_id")
            elif not tool_call_id.startswith("tc_"):
                issues.append(f"[spec] {where} tool_call_id '{tool_call_id}' must start with tc_")
            elif tool_call_id not in seen_tool_call_ids:
                issues.append(f"[spec] {where} tool_call_id '{tool_call_id}' not found in prior tool_calls")
            elif seen_tool_call_ids[tool_call_id] != tool_name:
                expected = seen_tool_call_ids[tool_call_id]
                issues.append(
                    f"[spec] {where} function '{tool_name}' mismatches tool_call_id '{tool_call_id}' "
                    f"(expected {expected})"
                )

            parsed_content, content_json_error = _parse_tool_content(msg.get("content"))
            if content_json_error:
                issues.append(f"[json] {where} content string is invalid strict JSON: {content_json_error}")
            if isinstance(parsed_content, dict):
                for key, value in parsed_content.items():
                    if key not in tool_spec.return_types:
                        issues.append(
                            f"[spec] {where} unexpected response field '{key}' for {tool_name}"
                        )
                        continue
                    expected_type = tool_spec.return_types[key]
                    if not _type_matches(value, expected_type):
                        observed_type = _value_kind(value)
                        issues.append(
                            f"[spec] {where} field '{key}' expected {expected_type}, got {observed_type}"
                        )
                    _collect_field_types(
                        value,
                        tool_name=tool_name,
                        scope="response",
                        path=key,
                        field_types=field_types,
                    )
            else:
                observed_type = _value_kind(parsed_content)
                issues.append(
                    f"[spec] {where} content must be object for {tool_name} return schema, got {observed_type}"
                )
                _collect_field_types(
                    parsed_content,
                    tool_name=tool_name,
                    scope="response",
                    path="$",
                    field_types=field_types,
                )

    for (tool_name, scope, path), types in sorted(field_types.items()):
        if len(types) > 1:
            observed = ", ".join(sorted(types))
            issues.append(
                f"[consistency] {scope}.{tool_name}.{path} has mixed types within trajectory: {observed}"
            )

    return issues


def validate_records(records: List[Any]) -> Dict[str, List[str]]:
    """Run ToolEmu and per-trajectory consistency checks across all records.

    Args:
        records: List of trajectory record dicts (as produced by
            ``format_trajectory.py`` or ``fix_tool_responses.py``).

    Returns:
        Mapping from ``summary_id`` to a list of issue strings (empty
        when nothing is wrong). Records without a valid ``summary_id``
        are filed under ``"record[<idx>]"``.
    """
    tool_specs = load_tool_specs()
    issues_by_summary: Dict[str, List[str]] = defaultdict(list)

    for record_idx, record in enumerate(records):
        summary_id = f"record[{record_idx}]"
        if isinstance(record, dict):
            summary_id = str(record.get("summary_id", summary_id))
        if not isinstance(record, dict):
            issues_by_summary[summary_id].append(
                f"[spec] record must be object, got {type(record).__name__}"
            )
            continue
        messages = record.get("messages") or record.get("agentic_trajectory") or []
        if not isinstance(messages, list):
            issues_by_summary[summary_id].append("[spec] messages/agentic_trajectory must be list")
            continue
        summary_issues = _validate_summary_record(
            summary_id=summary_id,
            messages=messages,
            tool_specs=tool_specs,
        )
        if summary_issues:
            issues_by_summary[summary_id].extend(summary_issues)

    return dict(issues_by_summary)


def _print_issues(
    issues_by_summary: Dict[str, List[str]],
    *,
    max_issues: int,
    max_issues_per_summary: int,
) -> None:
    total_issues = sum(len(v) for v in issues_by_summary.values())
    print(
        f"Found {total_issues} validation issue(s) across "
        f"{len(issues_by_summary)} summary_id(s)."
    )

    printed = 0
    for summary_id in sorted(issues_by_summary.keys()):
        if printed >= max_issues:
            break
        print(f"\n[{summary_id}]")
        summary_issues = issues_by_summary[summary_id]
        to_show = summary_issues[:max_issues_per_summary]
        for issue in to_show:
            if printed >= max_issues:
                break
            print(f"- {issue}")
            printed += 1
        if len(summary_issues) > len(to_show):
            hidden = len(summary_issues) - len(to_show)
            print(f"- ... {hidden} more issue(s) for this summary_id")

    if total_issues > printed:
        print(f"\n... {total_issues - printed} additional issue(s) not shown.")


def _print_malformed_rows(malformed_rows: List[MalformedRowIssue]) -> None:
    if not malformed_rows:
        return
    print(f"Malformed JSON rows: {len(malformed_rows)}")
    for issue in malformed_rows:
        print(
            f"- row {issue.row} (summary_id={issue.summary_id}, row_start_line={issue.row_start_line}): "
            f"{issue.error} at line {issue.error_line}, col {issue.error_col}"
        )
        if issue.snippet:
            print(f"  snippet: {issue.snippet}")


def main() -> None:
    """CLI entry point: load the input file, validate, print issues, and exit non-zero on failures."""
    parser = ArgumentParser(
        description=(
            "Read-only validator for trajectory tool usage against ToolEmu specs "
            "plus per-summary datatype consistency checks."
        )
    )
    parser.add_argument("--input", type=str, required=True, help="Input JSON/JSONL trajectory file")
    parser.add_argument("--max_issues", type=int, default=500, help="Maximum total issues to print")
    parser.add_argument(
        "--max_issues_per_summary",
        type=int,
        default=80,
        help="Maximum issues to print per summary_id",
    )
    args = parser.parse_args()

    records, json_issues, malformed_rows = load_records(args.input)
    print(f"Loaded {len(records)} record(s) from {args.input}")
    _print_malformed_rows(malformed_rows)

    issues_by_summary = validate_records(records)
    if json_issues:
        issues_by_summary.setdefault("__json__", []).extend(json_issues)
    if issues_by_summary:
        _print_issues(
            issues_by_summary,
            max_issues=max(1, int(args.max_issues)),
            max_issues_per_summary=max(1, int(args.max_issues_per_summary)),
        )
        raise SystemExit(1)
    if malformed_rows:
        raise SystemExit(1)

    print("No ToolEmu compliance or within-trajectory type-consistency issues found.")


if __name__ == "__main__":
    main()
