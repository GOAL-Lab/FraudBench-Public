"""Shared utilities for the FraudBench codebase.

Re-exports helpers for prompt construction, LiteLLM batch completion,
JSON extraction, fraud-tuple formatting, public-dataset loading, and the
"surgery kit" output-refinement framework. ``UTILS_DIR`` is exposed so
sibling code can locate the prompt-text files that live next to this
module.
"""

import os

UTILS_DIR = os.path.dirname(__file__)

from .utils import (
    setup_logger,
    extract_tuple,
    format_with_facts,
    format_tuple_for_prompt,
    extract_json_object,
    extract_json_objects,
    make_renamed_json_extractor,
    apply_prompt_to_df,
    litellm_chat_completion_with_retry,
    randomly_generate_email,
    randomly_generate_username,
    read_prompt_file,
    read_tuple_def,
    majority_vote,
    apply_placeholders_to_df,
)

from .surgery_kits import (
    SurgeryKitUnitTest,
    SurgeryKitModule,
)

from .public_dataset import (
    flatten_public_dataset,
    load_public_dataset,
)
