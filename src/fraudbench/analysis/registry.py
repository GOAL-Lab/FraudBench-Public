"""Single source of truth for model and setting display metadata.

This module replaces the model/setting display constants that previously
lived in :mod:`fraudbench.utils.globals`. It loads two JSONL files at
import time:

* ``data/final/models.jsonl`` — one record per logical model, keyed by a
  bare (provider-agnostic) model id. Used to populate
  :data:`MODEL_DISPLAY_NAMES`, :data:`MODEL_FAMILIES`, :data:`MODEL_ORDER`,
  :data:`REASONING_MODELS`, :data:`REASONING_SYMBOL`, and
  :data:`REASONING_SUBSCRIPT_CAPTION`.
* ``data/final/settings.jsonl`` — one record per evaluation setting. Used
  to populate :data:`SETTINGS` and :data:`SETTING_DISPLAY_NAMES`.

The data files live under the repo (not the installed package) because the
analysis modules are already designed to run from a repo checkout. Override
the data directory by setting ``FRAUDBENCH_DATA_DIR`` if you need to run the
registry against a different location.

Bare vs LiteLLM ids
-------------------
The registry is keyed by **bare** model ids (e.g. ``"claude-opus-4-5-20251101"``,
``"zai-org/GLM-4.7"``) — i.e. provider-agnostic. The ``model`` column in
``obs_data.parquet`` carries LiteLLM-style ids that include a provider
prefix and ``:provider`` suffix (e.g. ``"anthropic/claude-opus-4-5-20251101"``,
``"hf-inference-providers/zai-org/GLM-4.7:cerebras"``). Use
:func:`normalize_model_id` (or the ``model_id`` column added by
:func:`fraudbench.analysis.common.ensure_model_id_column`) to bridge between
the two.

This separation means **provider drift is invisible at the registry level**:
both ``...:cerebras`` and ``...:novita`` rows for GLM-4.7 normalize to
``"zai-org/GLM-4.7"``, which has a single record in ``models.jsonl``.
LiteLLM ids and per-call configs live exclusively in
:mod:`fraudbench.utils.globals` (``MODEL_CONFIGS``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


REASONING_SYMBOL = r"$_{\textsc{r}}$"
"""Subscript marker appended after model aliases when reasoning was enabled for that table's protocol."""

# Inspect-AI provider prefixes that we strip from LiteLLM-style ids. Anything
# not in this set (e.g. ``deepseek-ai/``, ``Qwen/``, ``zai-org/``) is treated
# as part of the bare model id rather than a provider route.
_PROVIDER_PREFIXES: frozenset[str] = frozenset({
    "openai", "anthropic", "google", "grok", "gemini", "xai",
})

#: Bare model ids known to ``models.jsonl``. Populated below once the JSONL
#: has been loaded; consulted by :func:`normalize_model_id` to keep the
#: function idempotent on ids that already match the registry's keying.
_KNOWN_BARE_IDS: set[str] = set()


def normalize_model_id(model: str) -> str:
    """Strip Inspect-AI provider prefixes and provider suffixes from a model id.

    Examples:
        ``openai/gpt-5-2025-08-07`` -> ``gpt-5-2025-08-07``
        ``anthropic/claude-opus-4-5-20251101`` -> ``claude-opus-4-5-20251101``
        ``hf-inference-providers/Qwen/Qwen3-4B-Instruct-2507:nscale``
            -> ``Qwen/Qwen3-4B-Instruct-2507``
        ``hf-inference-providers/zai-org/GLM-4.7:cerebras`` -> ``zai-org/GLM-4.7``
        ``deepseek-ai/DeepSeek-V3.2`` -> ``deepseek-ai/DeepSeek-V3.2``  (org prefix preserved)
        ``openai/gpt-oss-120b`` -> ``openai/gpt-oss-120b``  (already a bare id)
        ``human`` -> ``human``

    The function is idempotent on values returned by previous calls, and on
    bare ids listed in ``data/final/models.jsonl``.
    """
    if not isinstance(model, str) or not model:
        return ""

    # Idempotence: ids that already match a bare id in the registry are left
    # untouched. This disambiguates ``openai/gpt-oss-120b`` (a bare id stored
    # in ``models.jsonl``) from ``openai/gpt-5-...`` (an Inspect-AI route).
    if model in _KNOWN_BARE_IDS:
        return model

    if model.startswith("hf-inference-providers/"):
        # Drop the routing prefix and the ``:provider`` suffix; keep the
        # ``org/model`` segment as the bare id.
        name = model[len("hf-inference-providers/"):]
        if ":" in name:
            name = name.split(":", 1)[0]
        return name

    if "/" in model:
        prefix, rest = model.split("/", 1)
        if prefix in _PROVIDER_PREFIXES:
            return rest
        return model

    return model


def derive_provider(model: str) -> str:
    """Extract the inference provider from a LiteLLM-style model id.

    Companion to :func:`normalize_model_id`: the latter strips routing info
    to recover the bare model id, while this returns whatever routing info
    was attached. Together they cover every transformation we apply to the
    ``model`` column on the way into ``obs_data.parquet``.

    Rules (parsed from the same format described in :func:`normalize_model_id`):

    1. **Bare id present in** ``models.jsonl`` (e.g. ``openai/gpt-oss-120b``,
       ``deepseek-ai/DeepSeek-V3.2``) — no embedded provider info, returns
       ``""``. Preserves :func:`normalize_model_id`'s idempotence convention.
    2. **HuggingFace route** ``hf-inference-providers/<org>/<model>:<provider>``
       — returns the trailing ``<provider>`` (e.g. ``"novita"``,
       ``"cerebras"``, ``"nscale"``). Without the ``:provider`` suffix the
       routing target is the HuggingFace router itself; we return ``""`` so
       callers can distinguish that case from a real provider name.
    3. **Provider-prefixed route** ``<prefix>/<rest>`` where ``<prefix>`` is
       in :data:`_PROVIDER_PREFIXES` (``openai``, ``anthropic``, ``google``,
       ``grok``, ``gemini``, ``xai``) — returns ``<prefix>``.
    4. **Anything else** (``"human"``, plain bare ids, unrecognized routes) —
       returns ``""``.

    Examples:
        ``openai/gpt-5-2025-08-07``                                 -> ``"openai"``
        ``anthropic/claude-opus-4-5-20251101``                      -> ``"anthropic"``
        ``google/gemini-2.5-pro``                                   -> ``"google"``
        ``grok/grok-4-1-fast-reasoning``                            -> ``"grok"``
        ``hf-inference-providers/zai-org/GLM-4.7:cerebras``         -> ``"cerebras"``
        ``hf-inference-providers/openai/gpt-oss-120b:novita``       -> ``"novita"``
        ``deepseek-ai/DeepSeek-V3.2``                               -> ``""``
        ``openai/gpt-oss-120b``                                     -> ``""``
        ``human``                                                   -> ``""``
        ``""``                                                      -> ``""``
    """
    if not isinstance(model, str) or not model:
        return ""

    # Bare ids stored in models.jsonl carry no provider info on their own;
    # inference routing is always layered on top (see normalize_model_id).
    if model in _KNOWN_BARE_IDS:
        return ""

    if model.startswith("hf-inference-providers/"):
        if ":" in model:
            return model.rsplit(":", 1)[1]
        return ""

    if "/" in model:
        prefix = model.split("/", 1)[0]
        if prefix in _PROVIDER_PREFIXES:
            return prefix

    return ""


def _resolve_data_dir() -> Path:
    """Resolve the repo's ``data/final`` directory.

    Order of resolution:

    1. ``FRAUDBENCH_DATA_DIR`` environment variable (must point at the
       directory that contains ``models.jsonl`` and ``settings.jsonl``).
    2. Walk up from this file looking for a ``data/final`` sibling.
    3. Raise ``FileNotFoundError`` with a helpful message.
    """
    env = os.environ.get("FRAUDBENCH_DATA_DIR")
    if env:
        path = Path(env)
        if not path.is_dir():
            raise FileNotFoundError(
                f"FRAUDBENCH_DATA_DIR is set to '{env}' but that directory does not exist."
            )
        return path

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data" / "final"
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not locate 'data/final' relative to "
        f"{here}. Set FRAUDBENCH_DATA_DIR to override."
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {lineno} of {path}: {e}"
                ) from e
    return records


_DATA_DIR = _resolve_data_dir()
_MODELS_JSONL = _DATA_DIR / "models.jsonl"
_SETTINGS_JSONL = _DATA_DIR / "settings.jsonl"

_MODEL_RECORDS: list[dict[str, Any]] = _load_jsonl(_MODELS_JSONL)
_SETTING_RECORDS: list[dict[str, Any]] = _load_jsonl(_SETTINGS_JSONL)

# Make `normalize_model_id` aware of the bare ids declared in models.jsonl so
# org-prefixed bare ids (e.g. `deepseek-ai/DeepSeek-V3.2`,
# `openai/gpt-oss-120b`) are preserved verbatim.
_KNOWN_BARE_IDS.update(
    str(r["model"]) for r in _MODEL_RECORDS if r.get("model")
)


# ---------------------------------------------------------------------------
# Model registry (keyed by bare model id)
# ---------------------------------------------------------------------------

def _require(record: dict[str, Any], key: str, source: Path) -> Any:
    if key not in record:
        raise KeyError(
            f"Missing required key '{key}' for record {record!r} in {source}"
        )
    return record[key]


# Sort models by (family_order, model_order) for stable cross-process output.
_SORTED_MODELS = sorted(
    _MODEL_RECORDS,
    key=lambda r: (
        _require(r, "family_order", _MODELS_JSONL),
        _require(r, "model_order", _MODELS_JSONL),
    ),
)


MODEL_DISPLAY_NAMES: dict[str, str] = {
    _require(r, "model", _MODELS_JSONL): _require(r, "alias", _MODELS_JSONL)
    for r in _SORTED_MODELS
}
"""Mapping from bare model id to display name (the ``alias`` from ``models.jsonl``)."""


def _build_model_families() -> list[tuple[str, list[str]]]:
    """Group models by ``family``, ordered by ``family_order`` then ``model_order``."""
    families: dict[str, list[tuple[int, str]]] = {}
    family_order_for: dict[str, int] = {}
    for r in _MODEL_RECORDS:
        family = _require(r, "family", _MODELS_JSONL)
        bare_id = _require(r, "model", _MODELS_JSONL)
        model_order = _require(r, "model_order", _MODELS_JSONL)
        family_order = _require(r, "family_order", _MODELS_JSONL)
        families.setdefault(family, []).append((model_order, bare_id))
        prior = family_order_for.setdefault(family, family_order)
        if prior != family_order:
            raise ValueError(
                f"family '{family}' has inconsistent family_order: "
                f"{prior} vs {family_order}"
            )

    return [
        (family, [m for _, m in sorted(members)])
        for family, members in sorted(families.items(), key=lambda kv: family_order_for[kv[0]])
    ]


MODEL_FAMILIES: list[tuple[str, list[str]]] = _build_model_families()
"""Ordered list of ``(family_name, [bare_model_id, ...])`` tuples used for table grouping."""


MODEL_ORDER: list[str] = [m for _, members in MODEL_FAMILIES for m in members]
"""Flat list of bare model ids in canonical display order."""


REASONING_SETTINGS: dict[str, tuple[str, ...]] = {
    _require(r, "model", _MODELS_JSONL): tuple(r["reasoning_settings"])
    for r in _MODEL_RECORDS
    if _require(r, "reasoning", _MODELS_JSONL) and r.get("reasoning_settings")
}
"""Bare model ids whose reasoning was enabled in only **some** evaluation
settings, mapped to the settings in which it was on.

Populated from the optional ``reasoning_settings`` list in ``models.jsonl``.
The Claude 4.5 family is the current case: the binary (``closed``), open and
human-baseline QA runs were made without an extended-thinking budget
(``reasoning_budget`` is empty on every such row of ``obs_data.parquet``),
while the agentic runs used ``reasoning_tokens: 1024``.
"""

REASONING_MODELS: set[str] = {
    _require(r, "model", _MODELS_JSONL)
    for r in _MODEL_RECORDS
    if _require(r, "reasoning", _MODELS_JSONL)
}
"""Bare model ids evaluated with reasoning enabled in the main protocol.

Rows are taken from the ``reasoning`` field in ``models.jsonl``, which is kept
aligned with the production configs in ``MODEL_CONFIGS`` (e.g. Gemini 2.5
Flash and Flash Lite use ``reasoning_tokens: 0`` in main runs, so they are
*not* members of this set). Use :func:`reasoning_suffix` / :func:`reasoning_label`
to render model labels and :data:`REASONING_SUBSCRIPT_CAPTION` in table
captions; for reasoning-ablation-only tables, the paper tables mark every
model with :data:`REASONING_SYMBOL` because reasoning was turned on for all
systems in that study.
"""


def reasoning_suffix(bare_id: str) -> str:
    """Return the label suffix (including a leading space) for *bare_id*:
    :data:`REASONING_SYMBOL` for models in :data:`REASONING_MODELS`, else ``""``."""
    if bare_id in REASONING_MODELS:
        return " " + REASONING_SYMBOL
    return ""


def reasoning_label(name: str, bare_id: str) -> str:
    """Append the reasoning marker (if any) to a display *name*."""
    return name + reasoning_suffix(bare_id)


REASONING_SUBSCRIPT_CAPTION = (
    rf"{REASONING_SYMBOL} indicates reasoning models that had reasoning turned "
    r"on during evaluation."
)
"""Standard LaTeX caption clause for tables whose model rows use :data:`REASONING_SYMBOL`."""


# ---------------------------------------------------------------------------
# Grader reasoning state
# ---------------------------------------------------------------------------
# This is intentionally separate from :data:`REASONING_MODELS` because the
# grader configurations in :data:`fraudbench.utils.globals.GRADER_MODEL_CONFIGS`
# do not match the main-protocol settings. Membership was verified against
# the scorer ``ModelEvent`` configs in the evaluation logs and against the
# per-grader ``scorer_tokens`` column of ``obs_data.parquet`` (2026-11-05):
#
# * ``gpt-5-nano-2025-08-07`` — grader sets ``reasoning_effort="medium"``;
#   reasoning tokens > 0 on every graded sample (\\textsc{r} on).
# * ``claude-haiku-4-5-20251001`` — grader sets no thinking budget
#   (``temperature=0.1`` only); ``reasoning_tokens`` is 0 on every graded
#   sample, i.e. no extended thinking (\\textsc{r} off).
# * ``gemini-2.5-flash-lite`` — grader sets ``temperature=0.1`` and no
#   ``reasoning_tokens``; Flash Lite's thinking is off by default and the
#   logs record 0 reasoning tokens on every graded sample (\\textsc{r} off).
GRADER_REASONING_MODELS: set[str] = {
    "gpt-5-nano-2025-08-07",
}
"""Bare ids of LLM-as-a-judge graders that ran with reasoning enabled.

Use this (not :data:`REASONING_MODELS`) when rendering grader / judge labels:
the grader configs in :mod:`fraudbench.utils.globals` are a separate
deployment from the main-protocol model configs.
"""


# ---------------------------------------------------------------------------
# Setting registry
# ---------------------------------------------------------------------------

_SORTED_SETTINGS = sorted(
    _SETTING_RECORDS,
    key=lambda r: _require(r, "order", _SETTINGS_JSONL),
)


SETTINGS: list[str] = [
    _require(r, "key", _SETTINGS_JSONL)
    for r in _SORTED_SETTINGS
    if not _require(r, "is_human", _SETTINGS_JSONL)
]
"""Evaluation setting keys excluding the human baseline (``["closed", "open", "agentic"]``)."""


SETTING_DISPLAY_NAMES: dict[str, str] = {
    _require(r, "key", _SETTINGS_JSONL): _require(r, "display_name", _SETTINGS_JSONL)
    for r in _SORTED_SETTINGS
}
"""Mapping from setting key (including ``human_baseline``) to display name."""


__all__ = [
    "GRADER_REASONING_MODELS",
    "REASONING_SETTINGS",
    "reasoning_label",
    "reasoning_suffix",
    "MODEL_DISPLAY_NAMES",
    "MODEL_FAMILIES",
    "MODEL_ORDER",
    "REASONING_MODELS",
    "REASONING_SUBSCRIPT_CAPTION",
    "REASONING_SYMBOL",
    "SETTINGS",
    "SETTING_DISPLAY_NAMES",
    "derive_provider",
    "normalize_model_id",
]
