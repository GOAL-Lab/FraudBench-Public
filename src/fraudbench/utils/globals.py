"""Global configuration values shared across FraudBench scripts.

Defines the random seed (``RANDOM_SEED`` / ``RANDOM_SEED_OFFSET``) used by
data-generation and evaluation entry points, the lists of evaluated and
ablation models, per-model LiteLLM / Inspect AI generation kwargs (e.g.
``GPT_5_CONFIG``, ``CLAUDE_4_5_CONFIG``, ``GEMINI_2_5_CONFIG``,
``HF_CONFIG``, etc.), default connection / retry / token budgets, and the
separate ``GRADER_MODEL_CONFIGS`` used by :mod:`fraudbench.utils.evaluate`.

Set ``RANDOM_SEED_OFFSET`` in the environment to reproduce a previous run
(otherwise a fresh time-based offset is used per process).
"""
import os
import time
from inspect_ai.model import CachePolicy

# Base random seed - combined with offset for reproducible but varied randomness
_BASE_SEED = 1487

# Offset can be set via environment variable for explicit reproducibility,
# otherwise defaults to current time (seconds since epoch, mod 10^6 for reasonable range)
# To reproduce a run, set RANDOM_SEED_OFFSET env var to the same value
_offset_env = os.environ.get("RANDOM_SEED_OFFSET")
if _offset_env is not None:
    _SEED_OFFSET = int(_offset_env)
else:
    # Use current time as offset - each script invocation gets a unique seed
    _SEED_OFFSET = int(time.time()) % 1_000_000

RANDOM_SEED = _BASE_SEED + _SEED_OFFSET
RANDOM_SEED_OFFSET = _SEED_OFFSET  # Expose for logging; to reproduce, run with RANDOM_SEED_OFFSET=<this value>

# Note: use gemini/ as provider for LiteLLM but google/ for Inspect
MODELS = ["openai/gpt-5-mini-2025-08-07", "anthropic/claude-sonnet-4-5-20250929", "gemini/gemini-2.5-flash"]
MODELS_ADVANCED = ["openai/gpt-5-2025-08-07", "gemini/gemini-2.5-pro", "anthropic/claude-opus-4-5-20251101"]
GRADER_MODELS = ["openai/gpt-5-nano-2025-08-07", "anthropic/claude-haiku-4-5-20251001", "google/gemini-2.5-flash-lite"]
NUM_RETRIES = 5
GRADER_NUM_RETRIES = 5
BATCH_SIZE = 50
TIMEOUT = 300
MAX_CONNECTIONS = 50
HF_MAX_CONNECTIONS = 50
GRADER_MAX_CONNECTIONS = 100
MAX_TOKENS = 25000
# Should be 25000 for QA setting

CACHE_POLICY = None # CachePolicy(expiry="6M", per_epoch=False)
GRADER_CACHE_POLICY = CachePolicy(expiry="6M", per_epoch=False)
TEMPERATURE = 0.00
REASONING_EFFORT = "minimal"

# Note: GPT 5.1 reasoning.effort supports: none (default), low, medium, and high. Might be worth trying on (none, low)

GPT_5_CONFIG = {
    "temperature": None, # Not supported for GPT 5
    "reasoning_effort": REASONING_EFFORT,
    "reasoning_summary": "detailed",
    "reasoning_history": "all",
    "cache": CACHE_POLICY,
    "max_connections": MAX_CONNECTIONS,
    "do_sample": False,
    "max_tokens": MAX_TOKENS,
}

CLAUDE_4_5_CONFIG = {
    "temperature": TEMPERATURE,
    "reasoning_summary": "detailed",
    "reasoning_history": "all",
    "reasoning_tokens": 1024, # Min 1024
    "cache": CACHE_POLICY,
    "max_connections": MAX_CONNECTIONS,
    "do_sample": False,
    "max_tokens": MAX_TOKENS,
}

GEMINI_2_5_CONFIG = {
    "temperature": TEMPERATURE,
    "reasoning_summary": "detailed",
    "reasoning_history": "all",
    # "reasoning_effort": REASONING_EFFORT, # Not supported for Gemini 2.5
    "reasoning_tokens": 0, # 0 to turn off reasoning for Flash and Flash Lite only. When reasoning is on: Min 1, Max 24576 for Flash; Min 512, Max 24576 for Flash Lite
    "cache": CACHE_POLICY,
    "max_connections": MAX_CONNECTIONS,
    "do_sample": False,
    "max_tokens": MAX_TOKENS,
}

GEMINI_2_5_PRO_CONFIG = {
    "temperature": TEMPERATURE,
    "reasoning_summary": "detailed",
    "reasoning_history": "all",
    # "reasoning_effort": REASONING_EFFORT, # Not supported for Gemini 2.5
    "reasoning_tokens": 128, # Min 128, Max 32768
    "cache": CACHE_POLICY,
    "max_connections": MAX_CONNECTIONS,
    "do_sample": False,
    "max_tokens": MAX_TOKENS,
}

CLAUDE_OPUS_4_5_CONFIG = {
    "temperature": TEMPERATURE,
    "effort": "low",
    "cache": CACHE_POLICY,
    "max_connections": MAX_CONNECTIONS,
    "reasoning_tokens": 1024, # Min 1024
    "do_sample": False,
    "max_tokens": MAX_TOKENS,
}

GROK_CONFIG = {
    "temperature": TEMPERATURE,
    "reasoning_summary": "detailed",
    "reasoning_history": "all",
    "cache": CACHE_POLICY,
    "max_connections": MAX_CONNECTIONS,
    "do_sample": False,
    "max_tokens": MAX_TOKENS,
}

HF_CONFIG = {
    "temperature": TEMPERATURE,
    "reasoning_summary": "detailed",
    "reasoning_history": "all",
    "cache": CACHE_POLICY,
    "max_connections": HF_MAX_CONNECTIONS,
    "do_sample": False,
    "max_tokens": MAX_TOKENS,
}

DEEPSEEK_R1_CONFIG = {
    "temperature": TEMPERATURE,
    "reasoning_summary": "detailed",
    "reasoning_history": "all",
    "cache": CACHE_POLICY,
    "max_connections": HF_MAX_CONNECTIONS,
    "do_sample": False,
    "max_tokens": 16000, # Custom tokens limit for DeepSeek R1:novita
}

QWEN3_235B_INSTRUCT_CONFIG = {
    "temperature": TEMPERATURE,
    "reasoning_summary": "detailed",
    "reasoning_history": "all",
    "cache": CACHE_POLICY,
    "max_connections": HF_MAX_CONNECTIONS,
    "do_sample": False,
    "max_tokens": 16384, # Custom tokens limit for Qwen 235B Instruct:novita
}

MODEL_CONFIGS = {
    "openai/gpt-5-nano-2025-08-07": GPT_5_CONFIG,
    "openai/gpt-5-2025-08-07": GPT_5_CONFIG,
    "openai/gpt-5-mini-2025-08-07": GPT_5_CONFIG,
    "google/gemini-2.5-pro": GEMINI_2_5_PRO_CONFIG,
    "google/gemini-2.5-flash": GEMINI_2_5_CONFIG,
    "google/gemini-2.5-flash-lite": GEMINI_2_5_CONFIG,
    "anthropic/claude-opus-4-5-20251101": CLAUDE_OPUS_4_5_CONFIG,
    "anthropic/claude-haiku-4-5-20251001": CLAUDE_4_5_CONFIG,
    "anthropic/claude-sonnet-4-5-20250929": CLAUDE_4_5_CONFIG,
    "grok/grok-4-1-fast-reasoning": GROK_CONFIG,
    "grok/grok-4-1-fast-non-reasoning": GROK_CONFIG,
    # "hf-inference-providers/zai-org/GLM-4.7-Flash:novita": HF_CONFIG,
    # "hf-inference-providers/zai-org/GLM-4.7-Flash:zai-org": HF_CONFIG,
    # "hf-inference-providers/zai-org/GLM-4.7:cerebras": HF_CONFIG,
    "hf-inference-providers/zai-org/GLM-4.7:novita": HF_CONFIG,
    # "hf-inference-providers/zai-org/GLM-4.7:zai-org": HF_CONFIG,
    "hf-inference-providers/moonshotai/Kimi-K2-Instruct-0905:novita": HF_CONFIG,
    # "hf-inference-providers/Qwen/Qwen3-235B-A22B-Thinking-2507:fireworks-ai": HF_CONFIG,
    "hf-inference-providers/Qwen/Qwen3-235B-A22B-Thinking-2507:novita": HF_CONFIG,
    "hf-inference-providers/Qwen/Qwen3-4B-Thinking-2507:nscale": HF_CONFIG,
    "hf-inference-providers/Qwen/Qwen3-235B-A22B-Instruct-2507:novita": QWEN3_235B_INSTRUCT_CONFIG,
    "hf-inference-providers/Qwen/Qwen3-4B-Instruct-2507:nscale": HF_CONFIG,
    "hf-inference-providers/deepseek-ai/DeepSeek-V3.2:novita": HF_CONFIG,
    "hf-inference-providers/deepseek-ai/DeepSeek-R1:novita": DEEPSEEK_R1_CONFIG,
    "hf-inference-providers/openai/gpt-oss-120b:novita": HF_CONFIG,
}

EVALUATED_MODELS = list(MODEL_CONFIGS.keys())
TEMPERATURE_ABLATION_MODELS = [
    "hf-inference-providers/Qwen/Qwen3-4B-Thinking-2507:nscale",
    "hf-inference-providers/Qwen/Qwen3-4B-Instruct-2507:nscale",
    "hf-inference-providers/deepseek-ai/DeepSeek-V3.2:novita",
    "grok/grok-4-1-fast-reasoning",
    "grok/grok-4-1-fast-non-reasoning",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash-lite",
    "anthropic/claude-sonnet-4-5-20250929",
]

REASONING_ABLATION_MODELS = [
    "openai/gpt-5-nano-2025-08-07",
    "openai/gpt-5-2025-08-07",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash-lite",
]

########## GRADER MODEL CONFIGS ##########

GRADER_GPT_5_CONFIG = {
    "temperature": None,
    "reasoning_effort": "medium",
    "reasoning_summary": "detailed",
    "reasoning_history": "all",
    "cache": GRADER_CACHE_POLICY,
    "max_connections": GRADER_MAX_CONNECTIONS,
    "max_retries": GRADER_NUM_RETRIES,
}

GRADER_CLAUDE_HAIKU_4_5_CONFIG = {
    "temperature": 0.1,
    "reasoning_summary": "detailed",
    "reasoning_history": "all",
    "cache": GRADER_CACHE_POLICY,
    "max_connections": GRADER_MAX_CONNECTIONS,
    "max_retries": GRADER_NUM_RETRIES,
}

GRADER_GEMINI_2_5_FLASH_LITE_CONFIG = {
    "temperature": 0.1,
    "reasoning_summary": "detailed",
    "reasoning_history": "all",
    # "reasoning_effort": "medium", # Not supported for Gemini 2.5 Flash Lite
    "cache": GRADER_CACHE_POLICY,
    "max_connections": GRADER_MAX_CONNECTIONS,
    "max_retries": GRADER_NUM_RETRIES,
}

GRADER_MODEL_CONFIGS = {
    "openai/gpt-5-nano-2025-08-07": GRADER_GPT_5_CONFIG,
    "anthropic/claude-haiku-4-5-20251001": GRADER_CLAUDE_HAIKU_4_5_CONFIG,
    "google/gemini-2.5-flash-lite": GRADER_GEMINI_2_5_FLASH_LITE_CONFIG,
}

# Plot styling lives in fraudbench.analysis.style (and paper/styles/fraudbench.mplstyle).
# Do not add COLORS / rcParams / setup_plot_style here.

# Display metadata (model display names, families, settings, reasoning flags)
# now lives in fraudbench.analysis.registry, sourced from data/final/{models,settings}.jsonl.
