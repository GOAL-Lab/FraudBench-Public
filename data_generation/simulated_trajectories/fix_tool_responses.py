"""Normalize malformed tool-response payloads in agentic trajectories.

The simulator LLM occasionally returns malformed tool-response payloads
(bare lists, prose-wrapped JSON, scrambled IDs). This script rewrites those
payloads in place so downstream validators (``validate_tools.py``) and
graders see ToolEmu-compliant shapes.

Fixes covered:

- Wrap bare list responses into schema-correct objects for:
  SlackSearchMessage, GmailSearchContacts, GmailSearchEmails,
  NotionManagerSearchContent, MessengerSearchInChat, FacebookManagerSearchPosts.
- Coerce GmailReadEmail responses into the object shape defined in
  ``core_virtual_tools.py``.
- Optionally rewrite ``id`` / ``*_id`` fields to fresh UUIDs (faker-seeded)
  so any privacy-sensitive identifiers leaked from upstream sources are
  scrubbed. ``summary_id`` and values starting with ``@`` are preserved.

Input supports JSON arrays, JSON stream (multiple objects concatenated),
and JSONL.

Usage:
    uv run data_generation/simulated_trajectories/fix_tool_responses.py \\
        --input path/to/agentic_raw.jsonl \\
        --output path/to/agentic_fixed.jsonl
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from faker import Faker
from jsonargparse import ArgumentParser

import fraudbench.utils.globals as fb_globals
from fraudbench.data_generation.toolemu.tools.core_virtual_tools import (
    FacebookManagerSearchFriends,
    FacebookManagerSearchPosts,
    GmailReadEmail,
    GmailSearchContacts,
    GmailSearchEmails,
    MessengerSearchInChat,
    NotionManagerSearchContent,
    SlackSearchChannelOrUser,
    SlackSearchMessage,
)


def _tool_field_default(tool_cls: Any, field_name: str, default: Any = None) -> Any:
    model_fields = getattr(tool_cls, "model_fields", None)
    if model_fields and field_name in model_fields:
        return model_fields[field_name].default
    return getattr(tool_cls, field_name, default)


def _tool_name(tool_cls: Any) -> str:
    name = _tool_field_default(tool_cls, "name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{tool_cls.__name__} has invalid tool name")
    return name


def _single_array_return_key(tool_cls: Any) -> str:
    returns = _tool_field_default(tool_cls, "returns", []) or []
    if len(returns) != 1:
        raise ValueError(f"{tool_cls.__name__} expected one return field, found {len(returns)}")
    ret = returns[0]
    name = ret.get("name")
    rtype = str(ret.get("type", "")).lower()
    if not isinstance(name, str) or not name:
        raise ValueError(f"{tool_cls.__name__} has invalid return field name")
    if rtype != "array":
        raise ValueError(
            f"{tool_cls.__name__} expected array return, found type '{ret.get('type')}'"
        )
    return name


def _return_field_types(tool_cls: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ret in (_tool_field_default(tool_cls, "returns", []) or []):
        name = ret.get("name")
        if isinstance(name, str) and name:
            out[name] = str(ret.get("type", "string")).lower()
    return out


WRAPPED_ARRAY_TOOL_KEYS: Dict[str, str] = {
    _tool_name(SlackSearchMessage): _single_array_return_key(SlackSearchMessage),
    _tool_name(SlackSearchChannelOrUser): _single_array_return_key(SlackSearchChannelOrUser),
    _tool_name(FacebookManagerSearchFriends): _single_array_return_key(FacebookManagerSearchFriends),
    _tool_name(GmailSearchContacts): _single_array_return_key(GmailSearchContacts),
    _tool_name(GmailSearchEmails): _single_array_return_key(GmailSearchEmails),
    _tool_name(NotionManagerSearchContent): _single_array_return_key(NotionManagerSearchContent),
    _tool_name(MessengerSearchInChat): _single_array_return_key(MessengerSearchInChat),
    _tool_name(FacebookManagerSearchPosts): _single_array_return_key(FacebookManagerSearchPosts),
}

GMAIL_READ_TOOL_NAME = _tool_name(GmailReadEmail)
GMAIL_READ_TYPES = _return_field_types(GmailReadEmail)
GMAIL_READ_STRING_FIELDS = tuple(
    field_name for field_name, field_type in GMAIL_READ_TYPES.items() if field_type == "string"
)
GMAIL_READ_ARRAY_FIELDS = tuple(
    field_name for field_name, field_type in GMAIL_READ_TYPES.items() if field_type == "array"
)
ID_KEY_PATTERN = re.compile(r"(?:^id$|_id$)", re.IGNORECASE)
FAKER_SEED = fb_globals.RANDOM_SEED
Faker.seed(FAKER_SEED)
FAKER = Faker()


@dataclass
class SampleIdMaps:
    tool_call_ids: Dict[str, str] = field(default_factory=dict)
    other_ids: Dict[str, str] = field(default_factory=dict)


@dataclass
class FixStats:
    records_seen: int = 0
    records_changed: int = 0
    tool_messages_seen: int = 0
    messages_changed: int = 0
    parsed_string_content: int = 0
    replaced_tool_call_ids: int = 0
    replaced_other_ids: int = 0
    fixes_by_tool: Dict[str, int] = field(default_factory=dict)

    def add_tool_fix(self, tool_name: str) -> None:
        self.fixes_by_tool[tool_name] = self.fixes_by_tool.get(tool_name, 0) + 1


def _coerce_json_like_value(value: Any) -> Tuple[Any, bool]:
    if not isinstance(value, str):
        return value, False

    stripped = value.strip()
    if not stripped:
        return value, False

    if stripped.startswith("```") and stripped.endswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 : -3].strip()
        else:
            stripped = stripped[3:-3].strip()

    if not stripped or not (stripped.startswith("{") or stripped.startswith("[")):
        return value, False

    for parse_fn in (
        lambda s: json.loads(s),
        lambda s: json.loads(s, strict=False),
        lambda s: ast.literal_eval(s),
    ):
        try:
            parsed = parse_fn(stripped)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue
        if isinstance(parsed, (dict, list)):
            return parsed, True

    return value, False


def _coerce_json_like_string(value: Any) -> Any:
    parsed, ok = _coerce_json_like_value(value)
    return parsed if ok else value


def _is_id_key(key: str) -> bool:
    if key == "summary_id":
        return False
    return bool(ID_KEY_PATTERN.search(key))


def _new_uuid4() -> str:
    return FAKER.uuid4()


def _new_tool_call_uuid4() -> str:
    return f"tc_{FAKER.uuid4()}"


def _replace_id_value(value: str, *, as_tool_call_id: bool, maps: SampleIdMaps) -> str:
    if value.startswith("@"):
        return value
    target_map = maps.tool_call_ids if as_tool_call_id else maps.other_ids
    if value not in target_map:
        target_map[value] = _new_tool_call_uuid4() if as_tool_call_id else _new_uuid4()
    return target_map[value]


def _is_tool_call_id_field(
    *,
    key: str,
    in_assistant_tool_call: bool,
    in_tool_content: bool,
    tool_content_depth: int,
) -> bool:
    if key == "tool_call_id":
        return True
    if key == "id" and in_assistant_tool_call:
        return True
    if key == "id" and in_tool_content and tool_content_depth == 0:
        return True
    return False


def _rewrite_ids(
    obj: Any,
    maps: SampleIdMaps,
    *,
    message_role: Optional[str] = None,
    active_tool_name: Optional[str] = None,
    in_assistant_tool_call: bool = False,
    in_tool_content: bool = False,
    tool_content_depth: int = 0,
) -> Any:
    if isinstance(obj, list):
        return [
            _rewrite_ids(
                item,
                maps,
                message_role=message_role,
                active_tool_name=active_tool_name,
                in_assistant_tool_call=in_assistant_tool_call,
                in_tool_content=in_tool_content,
                tool_content_depth=(tool_content_depth + 1 if in_tool_content else 0),
            )
            for item in obj
        ]

    if not isinstance(obj, dict):
        return obj

    current_role = message_role
    if isinstance(obj.get("role"), str):
        current_role = str(obj["role"])
    current_tool_name = active_tool_name
    if isinstance(obj.get("function"), str):
        current_tool_name = str(obj["function"])

    new_obj: Dict[str, Any] = {}
    for key, value in obj.items():
        if _is_id_key(key) and isinstance(value, str):
            as_tool_call_id = _is_tool_call_id_field(
                key=key,
                in_assistant_tool_call=in_assistant_tool_call,
                in_tool_content=in_tool_content,
                tool_content_depth=tool_content_depth,
            )
            new_obj[key] = _replace_id_value(value, as_tool_call_id=as_tool_call_id, maps=maps)
            continue

        if key == "tool_calls" and current_role == "assistant" and isinstance(value, list):
            rewritten_tool_calls: List[Any] = []
            for item in value:
                item_tool_name = current_tool_name
                if isinstance(item, dict) and isinstance(item.get("function"), str):
                    item_tool_name = str(item["function"])
                rewritten_tool_calls.append(
                    _rewrite_ids(
                        item,
                        maps,
                        message_role=current_role,
                        active_tool_name=item_tool_name,
                        in_assistant_tool_call=True,
                        in_tool_content=False,
                        tool_content_depth=0,
                    )
                )
            new_obj[key] = rewritten_tool_calls
            continue

        if key == "content" and current_role == "tool":
            parsed_content = _coerce_json_like_string(value)
            new_obj[key] = _rewrite_ids(
                parsed_content,
                maps,
                message_role=current_role,
                active_tool_name=current_tool_name,
                in_assistant_tool_call=False,
                in_tool_content=True,
                tool_content_depth=0,
            )
            continue

        new_obj[key] = _rewrite_ids(
            value,
            maps,
            message_role=current_role,
            active_tool_name=current_tool_name,
            in_assistant_tool_call=False,
            in_tool_content=in_tool_content,
            tool_content_depth=(tool_content_depth + 1 if in_tool_content else 0),
        )

    return new_obj


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _fix_wrapped_array_response(content: Any, expected_key: str) -> Tuple[Dict[str, Any], bool]:
    if isinstance(content, dict):
        if expected_key in content:
            out = dict(content)
            normalized = _to_list(out[expected_key])
            changed = normalized != out[expected_key]
            out[expected_key] = normalized
            return out, changed

        list_values = [v for v in content.values() if isinstance(v, list)]
        if len(list_values) == 1:
            return {expected_key: list_values[0]}, True

        return {expected_key: [content]}, True

    if isinstance(content, list):
        return {expected_key: content}, True

    if content is None:
        return {expected_key: []}, True

    return {expected_key: [content]}, True


def _coerce_gmail_read_from_list(items: List[Any]) -> Dict[str, Any]:
    if not items:
        return {}

    if all(isinstance(item, dict) for item in items):
        merged: Dict[str, Any] = {}
        for item in items:
            for key, value in item.items():
                if key not in merged or merged[key] in (None, "", [], {}):
                    merged[key] = value
        return merged

    non_empty_items = [item for item in items if item not in (None, "")]
    if len(non_empty_items) == 1 and isinstance(non_empty_items[0], dict):
        return dict(non_empty_items[0])

    if len(non_empty_items) == 1 and isinstance(non_empty_items[0], str):
        return {"body": non_empty_items[0]}

    return {"body": "\n".join(str(item) for item in non_empty_items)}


def _fix_gmail_read_response(content: Any) -> Tuple[Dict[str, Any], bool]:
    changed = False
    if isinstance(content, dict):
        out: Dict[str, Any] = dict(content)
    elif isinstance(content, list):
        out = _coerce_gmail_read_from_list(content)
        changed = True
    elif content is None:
        out = {}
        changed = True
    else:
        out = {"body": str(content)}
        changed = True

    for field_name in GMAIL_READ_STRING_FIELDS:
        default = "inbox" if field_name == "folder" else ""
        value = out.get(field_name, default)
        normalized: str
        if value is None:
            normalized = default
        elif isinstance(value, list):
            normalized = ", ".join(str(v) for v in value if v is not None)
        elif isinstance(value, str):
            normalized = value
        else:
            normalized = str(value)
        if out.get(field_name) != normalized:
            changed = True
        out[field_name] = normalized

    for field_name in GMAIL_READ_ARRAY_FIELDS:
        normalized_list = _to_list(out.get(field_name, []))
        if out.get(field_name) != normalized_list:
            changed = True
        out[field_name] = normalized_list

    if not out.get("folder"):
        out["folder"] = "inbox"
        changed = True

    return out, changed


def _fix_tool_message(message: Dict[str, Any], stats: FixStats) -> bool:
    if message.get("role") != "tool":
        return False

    tool_name = message.get("function")
    if not isinstance(tool_name, str):
        return False

    if tool_name not in WRAPPED_ARRAY_TOOL_KEYS and tool_name != GMAIL_READ_TOOL_NAME:
        return False

    stats.tool_messages_seen += 1
    changed = False

    content = message.get("content")
    parsed_content, parsed = _coerce_json_like_value(content)
    if parsed:
        message["content"] = parsed_content
        stats.parsed_string_content += 1
        content = parsed_content
        changed = True

    if tool_name in WRAPPED_ARRAY_TOOL_KEYS:
        expected_key = WRAPPED_ARRAY_TOOL_KEYS[tool_name]
        fixed_content, fixed = _fix_wrapped_array_response(content, expected_key)
        if fixed:
            message["content"] = fixed_content
            changed = True
    elif tool_name == GMAIL_READ_TOOL_NAME:
        fixed_content, fixed = _fix_gmail_read_response(content)
        if fixed:
            message["content"] = fixed_content
            changed = True

    if changed:
        stats.messages_changed += 1
        stats.add_tool_fix(tool_name)

    return changed


def _extract_messages(record: Any) -> Optional[List[Any]]:
    if not isinstance(record, dict):
        return None
    messages = record.get("agentic_trajectory")
    if isinstance(messages, list):
        return messages
    messages = record.get("messages")
    if isinstance(messages, list):
        return messages
    return None


def _parse_json_stream(content: str, *, strict: bool) -> List[Any]:
    decoder = json.JSONDecoder(strict=strict)
    idx = 0
    records: List[Any] = []
    length = len(content)

    while idx < length:
        while idx < length and content[idx].isspace():
            idx += 1
        if idx >= length:
            break
        obj, next_idx = decoder.raw_decode(content, idx)
        if isinstance(obj, list):
            records.extend(obj)
        else:
            records.append(obj)
        idx = next_idx
    return records


def _parse_json_by_brace(content: str) -> List[Any]:
    chunks: List[str] = []
    depth = 0
    in_string = False
    escape = False
    start: Optional[int] = None

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

        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    chunks.append(content[start : idx + 1])
                    start = None

    parsed_records: List[Any] = []
    for chunk in chunks:
        try:
            parsed_records.append(json.loads(chunk, strict=False))
        except json.JSONDecodeError:
            continue
    return parsed_records


def _load_records(input_path: Path) -> Tuple[List[Any], str, Optional[str]]:
    content = input_path.read_text(encoding="utf-8")
    stripped = content.strip()
    if not stripped:
        return [], "jsonl", None

    warning: Optional[str] = None
    records: List[Any] = []
    source_format = "jsonl"

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            records = parsed
            source_format = "json"
        else:
            records = [parsed]
            source_format = "jsonl"
        return records, source_format, None
    except json.JSONDecodeError:
        pass

    for strict in (True, False):
        try:
            records = _parse_json_stream(stripped, strict=strict)
            if records:
                source_format = "jsonl"
                break
        except json.JSONDecodeError:
            records = []

    if not records:
        records = _parse_json_by_brace(stripped)
        source_format = "jsonl"

    if not records:
        raise ValueError(f"Unable to parse records from {input_path}")

    expected_summary_ids = stripped.count('"summary_id"')
    if expected_summary_ids and len(records) != expected_summary_ids:
        warning = (
            f"WARNING: parsed {len(records)} records but found {expected_summary_ids} "
            "summary_id markers. Malformed records may still be present."
        )

    return records, source_format, warning


def _default_output_path(input_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = input_path.suffix if input_path.suffix else ".jsonl"
    base = f"{input_path.stem}_tool_response_fixed_{timestamp}"
    output_path = output_dir / f"{base}{suffix}"
    idx = 1
    while output_path.exists():
        output_path = output_dir / f"{base}_{idx}{suffix}"
        idx += 1
    return output_path


def _write_records(records: List[Any], output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _fix_records(records: List[Any], *, replace_ids: bool) -> Tuple[List[Any], FixStats]:
    stats = FixStats()
    fixed_records: List[Any] = []

    for record in records:
        stats.records_seen += 1
        changed = False
        working_record = record

        if replace_ids:
            maps = SampleIdMaps()
            rewritten_record = _rewrite_ids(working_record, maps)
            if maps.tool_call_ids or maps.other_ids:
                changed = True
            stats.replaced_tool_call_ids += len(maps.tool_call_ids)
            stats.replaced_other_ids += len(maps.other_ids)
            working_record = rewritten_record

        messages = _extract_messages(working_record)
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and _fix_tool_message(message, stats):
                    changed = True

        if changed:
            stats.records_changed += 1
        fixed_records.append(working_record)

    return fixed_records, stats


def main() -> None:
    """CLI entry point: load the input file, fix tool responses, and write back to disk."""
    parser = ArgumentParser(description="Fix malformed tool-response payload shapes.")
    parser.add_argument("--input", type=str, required=True, help="Input JSON/JSONL file.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path. Must not exist unless --inplace=true.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for autogenerated output path when --output is omitted.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="auto",
        choices=["auto", "json", "jsonl"],
        help="Output format. 'auto' preserves parsed input format.",
    )
    parser.add_argument(
        "--inplace",
        type=bool,
        default=False,
        help="If true, overwrite the input file.",
    )
    parser.add_argument(
        "--replace_ids",
        type=bool,
        default=True,
        help=(
            "If true, replace ID fields recursively (id/*_id), keep summary_id, "
            "preserve values starting with '@', and normalize tool-call IDs to tc_<uuid4>."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records, source_format, parse_warning = _load_records(input_path)
    fixed_records, stats = _fix_records(records, replace_ids=bool(args.replace_ids))

    if args.output_format == "auto":
        suffix = input_path.suffix.lower()
        if suffix == ".json":
            resolved_format = "json"
        elif suffix == ".jsonl":
            resolved_format = "jsonl"
        else:
            resolved_format = source_format
    else:
        resolved_format = args.output_format

    if args.inplace:
        if args.output:
            raise ValueError("Use either --inplace=true or --output, not both.")
        output_path = input_path
    elif args.output:
        output_path = Path(args.output).resolve()
        if output_path.exists():
            raise FileExistsError(f"Output already exists: {output_path}")
    else:
        output_dir = Path(args.output_dir).resolve() if args.output_dir else input_path.parent
        output_path = _default_output_path(input_path, output_dir)

    _write_records(fixed_records, output_path, resolved_format)

    print(f"Loaded records: {len(records)}")
    if parse_warning:
        print(parse_warning)
    print(f"Records changed: {stats.records_changed}")
    print(f"Tool messages checked: {stats.tool_messages_seen}")
    print(f"Tool messages changed: {stats.messages_changed}")
    print(f"String content parsed to JSON: {stats.parsed_string_content}")
    print(f"Replaced tool_call IDs: {stats.replaced_tool_call_ids}")
    print(f"Replaced other IDs: {stats.replaced_other_ids}")
    for tool_name in sorted(stats.fixes_by_tool.keys()):
        print(f"- {tool_name}: {stats.fixes_by_tool[tool_name]} fixes")
    print(f"Wrote output: {output_path}")


if __name__ == "__main__":
    main()
