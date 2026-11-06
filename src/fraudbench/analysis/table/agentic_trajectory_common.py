"""Shared helpers for parsing agentic trajectories from ``dataset_public.jsonl``.

Used by :mod:`fraudbench.analysis.table.agentic_tool_calls_histogram` and
:mod:`fraudbench.analysis.table.agentic_toolkit_frequency`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from fraudbench.utils import flatten_public_dataset


def load_agentic_trajectories(input_path: str) -> pd.DataFrame:
    """Load rows from ``dataset_public.jsonl`` (or a compatible JSONL).

    The returned DataFrame is flattened via
    :func:`fraudbench.utils.flatten_public_dataset` so that ``metadata`` /
    ``seed`` / ``human_baseline`` keys appear as flat columns. The top-level
    public columns ``agentic_trajectory`` and ``agentic_toolkits`` are
    required.
    """
    path = Path(input_path)
    if path.suffix != ".jsonl":
        raise ValueError(f"Expected .jsonl input, got {path.suffix}")
    df = pd.read_json(input_path, lines=True)
    df = flatten_public_dataset(df)
    print(f"Loaded {len(df)} rows from {input_path}")
    for required in ("agentic_trajectory", "agentic_toolkits"):
        if required not in df.columns:
            raise ValueError(
                f"Input must contain an {required!r} column "
                f"(expected schema: data/final/dataset_public.jsonl)."
            )
    return df


def count_tool_calls_in_trajectory(trajectory: Any) -> int:
    """Count OpenAI-style ``tool_calls`` on assistant messages."""
    if trajectory is None or not isinstance(trajectory, list):
        return 0
    n = 0
    for msg in trajectory:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        n += len(tool_calls)
    return n


def iter_tool_functions_in_trajectory(trajectory: Any) -> Iterator[str]:
    """Yield each ``function`` name from assistant ``tool_calls``."""
    if trajectory is None or not isinstance(trajectory, list):
        return
    for msg in trajectory:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                fn = tc.get("function")
                if fn:
                    yield str(fn)


@lru_cache(maxsize=1)
def _toolkit_prefixes_longest_first() -> tuple[str, ...]:
    """Registered FunctionToolkit class names, longest match first."""
    from fraudbench.data_generation.toolemu.tools import get_toolkit_dict

    names = tuple(sorted(get_toolkit_dict().keys(), key=len, reverse=True))
    return names


def function_name_to_toolkit(function_name: str) -> str:
    """Map e.g. ``GmailSearchContacts`` → toolkit ``Gmail`` using registry prefixes."""
    for prefix in _toolkit_prefixes_longest_first():
        if function_name.startswith(prefix):
            return prefix
    return "Unknown"
