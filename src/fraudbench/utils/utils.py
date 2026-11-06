"""General-purpose helpers used across the FraudBench codebase.

Groups several loosely-related families of utilities:

- Logging setup (:func:`setup_logger`).
- Username / email synthesis from real-name strings
  (:func:`randomly_generate_username`, :func:`randomly_generate_email`).
- Fraud-tuple extraction and prompt formatting
  (:func:`extract_tuple`, :func:`format_with_facts`,
  :func:`apply_placeholders_to_df`, :func:`format_tuple_for_prompt`,
  :func:`read_tuple_def`).
- Plurality voting across grader judges (:func:`majority_vote`).
- JSON extraction from LLM output (:func:`extract_json_object`,
  :func:`extract_json_objects`, :func:`make_renamed_json_extractor`).
- LiteLLM wrappers with retry / batching support
  (:func:`litellm_chat_completion_with_retry`,
  :func:`litellm_batch_completion_with_retry`).
- The high-level :func:`apply_prompt_to_df` driver that stitches the
  above together to apply an LLM prompt to every row of a DataFrame and
  optionally refine outputs via a surgery kit.
"""

import ast
import json
import logging
import pickle
import re
import string
from collections import Counter
from datetime import datetime
from typing import Callable, List, Tuple, Dict, Union, Optional, Any

import os
import random
from pathlib import Path

import litellm
import numpy as np
import pandas as pd
from . import globals as globals
from .surgery_kits import SurgeryKitUnitTest, SurgeryKitModule
from tqdm import tqdm


def setup_logger(log_file: str = None, logger_name: str = None) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Also configures the root logger and common module loggers (e.g., 'toolemu')
    to use the same handlers, so all logging across the codebase goes to the
    same log file.
    
    Args:
        log_file: Path to log file. If None, creates one with timestamp.
        logger_name: Name for the logger. If None, uses '__main__'.
    
    Returns:
        Configured logger
    """
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{timestamp}.log"
    
    if logger_name is None:
        logger_name = '__main__'
    
    # Ensure the log directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Create file handler
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Configure root logger so all loggers inherit these handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    # Remove any existing handlers from root
    root_logger.handlers = []
    root_logger.addHandler(fh)
    root_logger.addHandler(ch)
    
    # Create the named logger (it will inherit from root)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # Don't add handlers to named logger - it inherits from root
    logger.handlers = []
    
    return logger

# Lowercase the first letter of roles (they follow a comma, so shouldn't be capitalized)
def _lowercase_first(s: str) -> str:
    s = str(s)
    if s and s[0].isalpha():
        return s[0].lower() + s[1:]
    return s

def _add_period(s: str) -> str:
    s = str(s)
    if s and s[-1] not in string.punctuation:
        return s + '.'
    return s

def _parse_text_to_dict(text) -> dict:
    """
    Parse text into a dictionary. Handles multiple formats:
    1. Already a dict → return as-is
    2. String representation of dict (e.g., "{'key': 'value'}") → parse with ast.literal_eval/json.loads
    3. Text format "key1: value1\nkey2: value2\n..." → parse line by line
    
    Args:
        text: Dict, string representation of dict, or "key: value" text format
        
    Returns:
        Dictionary of parsed key-value pairs
    """
    # If already a dict, return as-is
    if isinstance(text, dict):
        return text
    
    # If None or empty, return empty dict
    if not text:
        return {}
    
    text_str = str(text).strip()
    
    # Try to parse as dict string representation (e.g., "{'key': 'value'}" or '{"key": "value"}')
    if text_str.startswith('{') and text_str.endswith('}'):
        # Try ast.literal_eval first (handles Python dict syntax with single quotes)
        try:
            result = ast.literal_eval(text_str)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError):
            pass
        
        # Fallback to json.loads (handles JSON with double quotes)
        try:
            result = json.loads(text_str)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    
    # Parse "key: value" text format
    return {
        key.strip(): value.strip() 
        for line in text_str.split('\n')
        if ':' in line 
        for key, value in [line.split(':', 1)]
    }

# ============================================================================
# Username and Email Generation Helpers
# ============================================================================

# Common name prefixes to remove (case-insensitive)
_NAME_PREFIXES = {'mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'professor', 'sir', 'madam', 'mx'}

# Roman numeral pattern for suffix removal
_ROMAN_NUMERAL_PATTERN = re.compile(r'^(I{1,3}|IV|V|VI{0,3}|IX|X{1,3}|XI{0,3}|XII{0,3})$', re.IGNORECASE)

def _clean_name_for_username(full_name: str) -> Tuple[str, str]:
    """
    Clean a full name by removing prefixes and suffixes.
    
    Removes:
    - Prefixes: Mr., Mrs., Ms., Miss, Dr., Prof., Professor, Sir, Madam, Mx.
    - Suffixes: Anything after a comma, Roman numerals (I, II, III, IV, etc.)
    
    Args:
        full_name: Full name string to clean
        
    Returns:
        Tuple of (first_name, last_name), both lowercase
    """
    if not full_name or not full_name.strip():
        return ('', '')
    
    # Remove anything after a comma (suffixes like Jr., PhD, etc.)
    name = full_name.split(',')[0].strip()
    
    # Split into parts
    parts = name.split()
    
    if not parts:
        return ('', '')
    
    # Remove prefix if present (check first part)
    if parts and parts[0].rstrip('.').lower() in _NAME_PREFIXES:
        parts = parts[1:]
    
    if not parts:
        return ('', '')
    
    # Remove Roman numeral suffix if present (check last part)
    if parts and _ROMAN_NUMERAL_PATTERN.match(parts[-1].rstrip('.')):
        parts = parts[:-1]
    
    if not parts:
        return ('', '')
    
    # Extract first and last name
    first_name = parts[0].lower()
    last_name = parts[-1].lower() if len(parts) > 1 else ''
    
    # Remove any non-alphanumeric characters
    first_name = re.sub(r'[^a-z0-9]', '', first_name)
    last_name = re.sub(r'[^a-z0-9]', '', last_name)
    
    return (first_name, last_name)


def randomly_generate_username(full_name: str, seed: int) -> str:
    """
    Randomly generate a username from a full name.
    
    Cleans the name by removing prefixes (Mr., Mrs., Dr., Prof., etc.) and 
    suffixes (anything after comma, Roman numerals). Then randomly selects 
    one of four username formats:
    - firstname.lastname
    - firstnamelastname
    - lastname + first initial
    - firstname + random 4-digit string
    
    Args:
        full_name: Full name string (e.g., "Dr. John Smith, III")
        seed: Random seed for reproducible username generation
        
    Returns:
        Username preceded by "@" (e.g., "@john.smith")
    """
    random.seed(seed)
    
    first_name, last_name = _clean_name_for_username(full_name)
    
    if not first_name:
        # Fallback if name couldn't be parsed
        return f"@user{random.randint(1000, 9999)}"
    
    # Define username generation methods
    methods = []
    
    # Method 1: firstname.lastname (only if last name exists)
    if last_name:
        methods.append(f"{first_name}.{last_name}")
    
    # Method 2: firstnamelastname (only if last name exists)
    if last_name:
        methods.append(f"{first_name}{last_name}")
    
    # Method 3: first initial + last name (only if last name exists)
    if last_name:
        methods.append(f"{first_name[0]}{last_name}")
    
    # Method 4: firstname + random 4-digit string (always available)
    random_digits = f"{random.randint(0, 9999):04d}"
    methods.append(f"{first_name}{random_digits}")
    
    # Randomly select a method
    username = random.choice(methods)
    
    return f"{username}"

def randomly_generate_email(full_name: str, 
                            seed: int,
                            company_name: Optional[str] = None,
                            ) -> str:
    """
    Randomly generate an email address from a full name.
    
    Args:
        full_name: Full name string (e.g., "Dr. John Smith, III")
        company_name: Optional company name. If provided, creates domain by
                      concatenating all words, lowercasing, and adding .com.
                      If not provided, uses gmail.com as the domain.
        seed: Optional random seed for reproducible generation. If not provided,
              uses current time-based randomness.
        
    Returns:
        Full email address (e.g., "john.smith@acmecorp.com")
    """  
    # Generate username (without the @ prefix)
    username = randomly_generate_username(full_name, seed)
    
    # Generate domain
    if company_name and company_name.strip():
        # Take first 2 words if longer than 2 words, concatenate, lowercase, remove non-alphanumeric, add .com
        words = company_name.split()
        if len(words) > 2:
            words = words[:2]
        domain_base = ''.join(words).lower()
        domain_base = re.sub(r'[^a-z0-9]', '', domain_base)
        domain = f"{domain_base}.com"
    else:
        domain = "gmail.com"
    
    return f"{username}@{domain}"


# ============================================================================
# Fraud Tuple Extraction and CSV Processing Helpers
# ============================================================================

def extract_tuple(row: pd.Series, use_ind_name: bool = False) -> Dict[str, str]:
    """
    Helper function to extract fraud tuple fields from CSV
    
    Args:
        row: A pandas Series representing a row from the DataFrame (expects lowercase column names)
        use_ind_name: If True and defendant_is_org or plaintiff_is_org is True,
                      use the individual name and modify the role to indicate
                      they are an employee of the organization.
    """
    # ``data/final/dataset_public.jsonl`` (after flattening ``metadata``)
    # exposes ``is_concealment`` directly. We also derive it from a
    # ``fraud_type`` string when present so this helper still works on
    # intermediate frames produced upstream of the public flattening step.
    is_concealment = row.get('is_concealment', None)
    if is_concealment is None:
        is_concealment = row.get('fraud_type', '') == 'Fraudulent Concealment'

    duty_to_disclose = ''
    duty_to_disclose_details = ''

    if not is_concealment:
        # Fraudulent Misrepresentation
        actual_facts = str(row.get('misrep_actual_facts', ''))
        fraudulent_facts = str(row.get('misrep_misrepresentation', ''))
        
    else:
        # Fraudulent Concealment
        actual_facts = str(row.get('conceal_concealed_facts', ''))
        fraudulent_facts = str(row.get('conceal_concealment', ''))
        # Extract duty to disclose from conceal_other or construct from category-specific fields
        duty_to_disclose = str(row.get('conceal_duty_category', '') or '')

        if 'prior statement' in duty_to_disclose.lower():
            duty_to_disclose += ' (the fraud perpetrator has made a prior statement and knows that it will likely mislead another if not amended, even if it was not misleading when made)'
            ps_statement = row.get('conceal_ps_statement', '')
            ps_intent = row.get('conceal_ps_intent', '')
            
            if ps_statement:
                duty_to_disclose_details += f"The prior statement made by the fraud perpetrator was: {_add_period(ps_statement)}"
            if ps_intent:
                duty_to_disclose_details += f"The fraud perpetrator knew that the prior statement was likely to mislead because: {_add_period(ps_intent)}."
                
        elif 'relationship' in duty_to_disclose.lower():
            duty_to_disclose += ' (the fraud perpetrator is in a fiduciary or confidential relationship with the fraud victim that obliges the perpetrator to make disclosures)'
            r_relationship = row.get('conceal_r_relationship', '')
            r_duties = row.get('conceal_r_duties', '')

            if r_relationship: 
                duty_to_disclose_details += f"The fraud perpetrator and the fraud victim had a confidential or fiduciary relationship: {_add_period(r_relationship)}"
            if r_duties:
                duty_to_disclose_details += f"The fraud perpetrator is obliged to disclose because: {_add_period(r_duties)}"
            
        elif 'assumption' in duty_to_disclose.lower():
            duty_to_disclose += ' (the fraud perpetrator knows that the fraud victim in a transaction is mistaken about a basic assumption behind it, and that the victim, because of the relationship between them, the customs of the trade, or other circumstances such as the perpetrator having superior knowledge, would reasonably expect disclosure of what the fraud perpetrator knows)'
            m_assumption = row.get('conceal_m_assumption', '')
            m_reason = row.get('conceal_m_disclosure_reason', '')
            
            if m_assumption:
                duty_to_disclose_details += f"The fraud perpetrator knew that the fraud victim was mistaken about a basic assumption behind the transaction: {_add_period(m_assumption)}"
            if m_reason:
                duty_to_disclose_details += f"The fraud perpetrator knew that the fraud victim would reasonably expect disclosure of what the fraud perpetrator knows because: {_add_period(m_reason)}"
    
    # Extract defendant name and role
    defendant_name = str(row.get('defendant_name', ''))
    defendant_role = str(row.get('defendant_role', ''))
    
    # Extract plaintiff name and role
    plaintiff_name = str(row.get('plaintiff_name', ''))
    plaintiff_role = str(row.get('plaintiff_role', ''))
    
    # Handle use_ind_name: if a party is an organization, use individual name and modify role
    if use_ind_name:
        if row.get('defendant_is_org'):
            defendant_name = str(row.get('defendant_ind_name', defendant_name))
            defendant_role = f"An employee of {row.get('defendant_name', '')} which is a {_lowercase_first(row.get('defendant_role', ''))}"
        
        if row.get('plaintiff_is_org'):
            plaintiff_name = str(row.get('plaintiff_ind_name', plaintiff_name))
            plaintiff_role = f"An employee of {row.get('plaintiff_name', '')} which is a {_lowercase_first(row.get('plaintiff_role', ''))}"
    
    result = {
        'summary_id': row.get('summary_id', None),
        'is_concealment': bool(is_concealment),
        'defendant_name': defendant_name,
        'defendant_role': defendant_role,
        'plaintiff_name': plaintiff_name,
        'plaintiff_role': plaintiff_role,
        'actual_facts': actual_facts,
        'fraudulent_facts': fraudulent_facts,
        'intent': str(row.get('intent', '')),
        'materiality': str(row.get('materiality', '')),
        'justifiable_reliance': str(row.get('justifiable_reliance', '')),
        'duty_to_disclose': duty_to_disclose,
        'duty_to_disclose_details': duty_to_disclose_details,
    }
    
    # Include placeholders if present (for placeholder formatting)
    if 'placeholders' in row and row.get('placeholders'):
        result['placeholders'] = _parse_text_to_dict(row.get('placeholders'))
    
    return result

def format_with_facts(data: Dict[str, Any], 
                      placeholders_key: str = 'placeholders'
                    ) -> Dict[str, Any]:
    """
    Apply placeholder formatting to all string values in a dictionary.
    
    Args:
        data: Dictionary containing string values with placeholders like {placeholder_name}
        placeholders_key: Key in the dictionary containing the placeholders (either as a 
                          text string "key1: value1\nkey2: value2" or as a dict)
        
    Returns:
        Dictionary with placeholders replaced in all string values
    """
    data_formatted = data.copy()
    
    # Get placeholders - handle both string and dict formats
    placeholders_raw = data.get(placeholders_key)
    if placeholders_raw is None:
        logging.warning(f"No placeholders found at key '{placeholders_key}', returning data unchanged")
        return data_formatted
    
    if isinstance(placeholders_raw, dict):
        placeholders_dict = placeholders_raw
    elif isinstance(placeholders_raw, str):
        placeholders_dict = _parse_text_to_dict(placeholders_raw)
    else:
        logging.warning(f"Placeholders at key '{placeholders_key}' is neither a dict nor string, returning data unchanged")
        return data_formatted
    
    # Apply .format() to each string value in the dictionary
    for key, value in data_formatted.items():
        if key != placeholders_key and isinstance(value, str):
            try:
                data_formatted[key] = value.format(**placeholders_dict)
            except (KeyError, ValueError) as e:
                raise ValueError(f"Failed to format key '{key}': {e}")
    
    return data_formatted


def apply_placeholders_to_df(df: pd.DataFrame, placeholders_column: str = 'placeholders') -> pd.DataFrame:
    """
    Apply placeholder substitution to all rows in a dataframe.
    
    Args:
        df: DataFrame with a placeholders column containing substitution values
        placeholders_column: Name of column containing placeholder dict (default: 'placeholders')
        
    Returns:
        DataFrame with placeholders substituted in all string fields
    """
    if placeholders_column not in df.columns:
        return df
    
    formatted_rows = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        try:
            formatted_row = format_with_facts(row_dict, placeholders_key=placeholders_column)
            formatted_rows.append(formatted_row)
        except Exception:
            # If formatting fails, keep original row
            formatted_rows.append(row_dict)
    
    return pd.DataFrame(formatted_rows)


def majority_vote(votes: List[List[Optional[Any]]], random_seed: Optional[int] = None) -> List[Optional[Any]]:
    """Take majority/plurality vote across multiple judge results.
    
    Logic:
    1. If a majority exists (> 50% of valid votes), use that
    2. Otherwise, take the plurality (most common vote)
    3. If plurality is tied, randomly select from tied options using random_seed
    
    Args:
        votes: List of result lists, one per judge. Each result list has one value per row.
        random_seed: Seed for random selection when there's a tie (default: None)
        
    Returns:
        List of vote results (None only if all votes are None)
    """
    rng = np.random.default_rng(random_seed)
    
    num_rows = len(votes[0])
    results = []
    
    for row_idx in range(num_rows):
        row_votes = [v[row_idx] for v in votes if v[row_idx] is not None]
        if not row_votes:
            results.append(None)
        else:
            vote_counts = Counter(row_votes)
            most_common = vote_counts.most_common()
            top_count = most_common[0][1]
            
            # Check for majority (> 50%)
            if top_count > len(row_votes) / 2:
                results.append(most_common[0][0])
            else:
                # No majority - take plurality or break tie randomly
                tied_options = [opt for opt, count in most_common if count == top_count]
                if len(tied_options) == 1:
                    # Clear plurality winner
                    results.append(tied_options[0])
                else:
                    # Tie - randomly select from tied options
                    results.append(rng.choice(tied_options))
    
    return results


def read_prompt_file(path: str, strip_triple_quotes: bool = True) -> str:
    """Read a prompt text file from disk, optionally stripping ``\"\"\"`` wrappers.

    Args:
        path: Path to the prompt file (UTF-8).
        strip_triple_quotes: If ``True`` (default), remove a leading and
            trailing ``\"\"\"`` from the file so prompts can be authored as
            triple-quoted blocks for editor friendliness.

    Returns:
        The stripped file contents.
    """
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if strip_triple_quotes and content.startswith('"""') and content.endswith('"""'):
        content = content[3:-3]
    return content


def read_tuple_def(is_concealment: bool, utils_dir: Optional[str] = None) -> str:
    """Load the fraud-tuple field definition prompt for the given fraud type.

    Args:
        is_concealment: If ``True``, load the Fraudulent Concealment
            definition (``PROMPT_TUPLE_DEF_FC.txt``); otherwise load the
            Fraudulent Misrepresentation definition
            (``PROMPT_TUPLE_DEF_FM.txt``).
        utils_dir: Directory containing the prompt files. Defaults to the
            directory this module lives in.

    Returns:
        The raw prompt text (triple-quote wrappers preserved).
    """
    if utils_dir is None:
        utils_dir = str(Path(__file__).resolve().parent)
    tuple_def_file = 'PROMPT_TUPLE_DEF_FC.txt' if is_concealment else 'PROMPT_TUPLE_DEF_FM.txt'
    return read_prompt_file(os.path.join(utils_dir, tuple_def_file), strip_triple_quotes=False)


def _replace_quotes_with_single(s: str) -> str:
    """Replace double quotes and curly quotes with straight single quotes for cleaner JSON output."""
    if not isinstance(s, str):
        return s
    # Replace straight double quotes, curly double quotes, and curly single quotes with straight single quotes
    return (s
            .replace('"', "'")   # straight double quote
            .replace('“', "'")   # left curly double quote
            .replace('”', "'")   # right curly double quote
            .replace('‘', "'")   # left curly single quote
            .replace('’', "'"))  # right curly single quote

def _format_tuple_for_prompt_fm(tuple_dict: Dict[str, str]) -> str:
    """
    Format a fraud tuple for Fraudulent Misrepresentation cases as a JSON object string.
    
    Returns a JSON object with keys:
    fraud_perpetrator, fraud_perpetrator_role, fraud_victim, fraud_victim_role,
    actual_facts, fraudulent_misrepresentation, intent, materiality, justifiable_reliance
    """
    formatted_dict = {
        "fraud_perpetrator": _replace_quotes_with_single(tuple_dict['defendant_name']),
        "fraud_perpetrator_role": _replace_quotes_with_single(_lowercase_first(tuple_dict['defendant_role'])),
        "fraud_victim": _replace_quotes_with_single(tuple_dict['plaintiff_name']),
        "fraud_victim_role": _replace_quotes_with_single(_lowercase_first(tuple_dict['plaintiff_role'])),
        "actual_facts": _replace_quotes_with_single(tuple_dict['actual_facts']),
        "fraudulent_misrepresentation": _replace_quotes_with_single(tuple_dict['fraudulent_facts']),
        "intent": _replace_quotes_with_single(tuple_dict['intent']),
        "materiality": _replace_quotes_with_single(tuple_dict['materiality']),
        "justifiable_reliance": _replace_quotes_with_single(tuple_dict['justifiable_reliance'])
    }
    return json.dumps(formatted_dict, indent=4)

def _format_tuple_for_prompt_fc(tuple_dict: Dict[str, str]) -> str:
    """
    Format a fraud tuple for Fraudulent Concealment cases as a JSON object string.
    
    Returns a JSON object with keys:
    fraud_perpetrator, fraud_perpetrator_role, fraud_victim, fraud_victim_role,
    concealed_facts, fraudulent_representation, intent, materiality, justifiable_reliance,
    duty_to_disclose, duty_to_disclose_details
    """
    formatted_dict = {
        "fraud_perpetrator": _replace_quotes_with_single(tuple_dict['defendant_name']),
        "fraud_perpetrator_role": _replace_quotes_with_single(_lowercase_first(tuple_dict['defendant_role'])),
        "fraud_victim": _replace_quotes_with_single(tuple_dict['plaintiff_name']),
        "fraud_victim_role": _replace_quotes_with_single(_lowercase_first(tuple_dict['plaintiff_role'])),
        "concealed_facts": _replace_quotes_with_single(tuple_dict['actual_facts']),
        "fraudulent_representation": _replace_quotes_with_single(tuple_dict['fraudulent_facts']),
        "intent": _replace_quotes_with_single(tuple_dict['intent']),
        "materiality": _replace_quotes_with_single(tuple_dict['materiality']),
        "justifiable_reliance": _replace_quotes_with_single(tuple_dict['justifiable_reliance']),
        "duty_to_disclose": _replace_quotes_with_single(tuple_dict.get('duty_to_disclose', '')),
        "duty_to_disclose_details": _replace_quotes_with_single(tuple_dict.get('duty_to_disclose_details', ''))
    }
    return json.dumps(formatted_dict, indent=4)

def format_tuple_for_prompt(
    tuple_dict: Dict[str, str],
    apply_placeholder_formatting: bool = True,
    placeholders_key: str = 'placeholders',
    is_concealment: bool = None,
) -> str:
    """
    Format a fraud tuple dictionary into a string suitable for LLM prompts.

    Uses different formatting based on fraud type:
    - Fraudulent Misrepresentation: (fraud perpetrator, fraud victim, actual facts,
      fraudulent misrepresentation, intent, materiality, justifiable reliance)
    - Fraudulent Concealment: (fraud perpetrator, fraud victim, concealed facts,
      fraudulent representation, intent, materiality, justifiable reliance, duty to disclose)

    Args:
        tuple_dict: Dictionary from extract_fraud_tuple_from_row()
        apply_placeholder_formatting: If True, replace placeholders like {name} with
                                      values from the placeholders key in tuple_dict
        placeholders_key: Key in tuple_dict containing the placeholders (default: 'placeholders')
        is_concealment: True for Fraudulent Concealment, False for Fraudulent Misrepresentation.
                        If None, uses the 'is_concealment' key from tuple_dict.

    Returns:
        Formatted string representation of the fraud tuple
    """
    if apply_placeholder_formatting:
        tuple_dict = format_with_facts(tuple_dict, placeholders_key=placeholders_key)

    if is_concealment is None:
        is_concealment = tuple_dict.get('is_concealment', False)

    if is_concealment:
        return _format_tuple_for_prompt_fc(tuple_dict)
    return _format_tuple_for_prompt_fm(tuple_dict)

# ============================================================================
# Evaluation Item Creation Helpers
# ============================================================================

def _assign_model(model: Union[str, List[str]], n_items: int, random_seed: int) -> List[str]:
    if isinstance(model, str):
        return [model] * n_items
    else:
        random.seed(random_seed)
        return [random.choice(model) for _ in range(n_items)]

def _build_batch_messages(df: pd.DataFrame, 
                         prompt_builder: Callable[[Any], str]
                         ) -> List[List[Dict[str, str]]]:
    all_prompts = [prompt_builder(row) for _, row in df.iterrows()]
    all_batch_messages = [
        [{"role": "user", "content": prompt}]
        for prompt in all_prompts
    ]

    return all_batch_messages

def make_renamed_json_extractor(
    rename_map: Dict[str, str],
    drop_keys: Optional[List[str]] = None,
) -> Callable[[str], Optional[Dict[str, Any]]]:
    """
    Build a JSON extractor that renames keys after parsing.

    Use this when an LLM prompt template emits one set of display names
    (e.g. ``qa_open`` -- the names from the original paper) but the rest
    of the codebase / public dataset uses different internal column names
    (e.g. ``query_open``). The extractor first parses the response with
    :func:`extract_json_object`, then applies ``rename_map`` to the
    top-level keys and drops any keys listed in ``drop_keys``.

    Args:
        rename_map: ``{prompt_display_name: internal_column_name}``.
        drop_keys: Optional list of prompt-display names to drop from
            the parsed response (e.g. legacy fields no longer in the
            public schema).

    Returns:
        A callable suitable for ``apply_prompt_to_df(response_extractor=...)``.
    """
    drop_set = set(drop_keys or [])

    def _extract(text: str) -> Optional[Dict[str, Any]]:
        parsed = extract_json_object(text)
        if not isinstance(parsed, dict):
            return parsed
        renamed: Dict[str, Any] = {}
        for k, v in parsed.items():
            if k in drop_set:
                continue
            renamed[rename_map.get(k, k)] = v
        return renamed

    return _extract


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a single JSON object from a string.
    
    Handles common cases like:
    - Pure JSON string
    - JSON wrapped in markdown code blocks (```json ... ```)
    - JSON embedded in other text
    
    Args:
        text: String that may contain a JSON object
        
    Returns:
        Parsed JSON object as a dict, or None if no valid JSON found
    """
    if not text:
        return None
    
    # Try parsing the entire string as JSON first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to extract from markdown code block
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object pattern { ... }
    # Use a simple brace-matching approach
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    brace_count = 0
    for i, char in enumerate(text[start_idx:], start=start_idx):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                try:
                    return json.loads(text[start_idx:i+1])
                except json.JSONDecodeError:
                    # Continue searching for another valid JSON
                    pass
    
    return None

def extract_json_objects(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Extract multiple JSON objects from a string.
    
    Handles common cases like:
    - JSON array: [{ ... }, { ... }]
    - Multiple separate JSON objects: { ... } { ... }
    - JSON wrapped in markdown code blocks
    - JSON embedded in other text
    
    Args:
        text: String that may contain one or more JSON objects
        
    Returns:
        List of parsed JSON objects as dicts, or None if no valid JSON found
    """
    if not text:
        return None
    
    # Try parsing the entire string as a JSON array first
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, list):
            return parsed
        else:
            return [parsed]
    except json.JSONDecodeError:
        pass
    
    # Try to extract from markdown code block
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_block_match:
        try:
            parsed = json.loads(code_block_match.group(1).strip())
            if isinstance(parsed, list):
                return parsed
            else:
                return [parsed]
        except json.JSONDecodeError:
            pass
    
    # Find all JSON objects using brace matching
    objects = []
    i = 0
    while i < len(text):
        start_idx = text.find('{', i)
        if start_idx == -1:
            break
        
        brace_count = 0
        for j, char in enumerate(text[start_idx:], start=start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    try:
                        obj = json.loads(text[start_idx:j+1])
                        objects.append(obj)
                        i = j + 1
                        break
                    except json.JSONDecodeError:
                        i = start_idx + 1
                        break
        else:
            # Unbalanced braces, move past this starting brace
            i = start_idx + 1
    
    return objects if objects else None

def _process_responses_and_extract_results(
    responses: Dict[str, Dict[str, Any]],
    row_ids: List[str],
    response_extractor: Callable[[str], Optional[Any]],
    logger: logging.Logger,
    id_column: str = 'summary_id'
) -> List[Optional[Any]]:
    """
    Process LLM responses and extract results.
    
    Args:
        responses: Dictionary mapping row_ids to response data. Each entry should have
                        'original_completion' and optionally 'refined_output' (from surgery kit).
                        Extracts from 'refined_output' if present, otherwise from 'original_completion'.
        row_ids: List of row IDs in order (determines result list ordering)
        response_extractor: Function that extracts the result from response text
        logger: Logger instance for logging
        id_column: Column name for row IDs for logging
    
    Returns:
        List of extracted results (or None for failed rows), in same order as row_ids
    """
    results = [None] * len(row_ids)
    
    for idx, row_id in enumerate(row_ids):
        if row_id not in responses:
            logger.warning(f"No response for {id_column}={row_id}")
            continue
        
        response_data = responses[row_id]
        
        try:
            # Use refined output if available, otherwise use original completion
            if 'refined_output' in response_data:
                answer = response_data['refined_output'].strip()
                logger.debug(f"Using refined output for {id_column}={row_id}")
            elif response_data.get('original_completion') is not None:
                answer = response_data['original_completion']['choices'][0]['message']['content'].strip()
            else:
                logger.warning(f"No valid response for {id_column}={row_id}")
                continue
            
            result = response_extractor(answer)
            
            if result is None:
                logger.warning(f"Could not extract result from response for {id_column}={row_id}: {answer}")
            else:
                logger.info(f"Row {idx} ({id_column}: {row_id}): {result}")
                results[idx] = result
                
        except Exception as e:
            logger.error(f"Error processing response for {id_column}={row_id}: {e}")
    
    return results

def apply_prompt_to_df(
    df: pd.DataFrame,
    prompt_builder: Callable[[Any], str],
    response_extractor: Callable[[str], Optional[Any]],
    model: Optional[Union[str, List[str]]],
    logger: logging.Logger,
    random_seed: Optional[int],
    num_retries: int = globals.NUM_RETRIES,
    id_column: str = 'summary_id',
    model_column: str = 'model',
    show_progress: bool = True,
    out: Optional[str] = 'logs/',
    out_suffix: Optional[str] = None,
    model_col_suffix: Optional[str] = None,
    surgery_kit: Optional[SurgeryKitModule] = None,
    surgery_kit_tests: Optional[List[SurgeryKitUnitTest]] = None,
    surgery_kit_kwargs: Optional[Dict[str, Any]] = None,
    return_verbose: bool = False,
    timestamp: Optional[str] = None,
    **kwargs
) -> Union[List[Optional[Any]], Dict[str, Any]]:
    """
    Iterate through a DataFrame and apply an LLM prompt to each row using batch_completion.
    
    Args:
        df: pandas DataFrame to process (columns will be lowercased, placeholders converted to dicts)
        prompt_builder: Function that takes a row and returns a prompt string
        response_extractor: Function that takes the LLM response and extracts the result
        model: LiteLLM model string, list of models (randomly assigns to rows), or None
               (if None, uses model_column from df)
        logger: Logger instance for logging
        num_retries: Maximum retry attempts for failed items (default: 3)
        id_column: Column name for row IDs for logging (default: 'summary_id')
        model_column: Column name for model when model=None (default: 'model')
        show_progress: Whether to show a progress bar (default: True)
        random_seed: Seed for random model assignment when model is a list
        out: Optional directory path to export JSON responses and results CSV
        out_suffix: Optional suffix for output file name
        model_col_suffix: Optional suffix for model column name (e.g., 'generation' -> 'model_generation')
        return_verbose: If True, return a dict with results, models, messages, and df (default: False)
        timestamp: Optional timestamp string for output filenames (if None, generates one)
        **kwargs: Additional arguments passed to litellm (max_tokens, temperature, etc.)
        
    Returns:
        Default: List of extracted results (or None for failed rows)
        If return_verbose=True: Dict with keys:
            - 'results': List of extracted results
            - 'models': List of model used per row
            - 'messages': List of full conversation history per row (input + assistant response)
            - 'df': Updated DataFrame with results and model columns merged
    """
    # Data should already be preprocessed via data/preprocess.py
    df = df.copy()
    if surgery_kit_kwargs is None:
        surgery_kit_kwargs = {}
    
    if id_column not in df.columns:
        raise ValueError(f"Missing required id column '{id_column}' in dataframe")
    if df[id_column].duplicated().any():
        duplicate_ids = df.loc[df[id_column].duplicated(), id_column].unique().tolist()
        raise ValueError(f"Duplicate values found in '{id_column}': {duplicate_ids[:10]}")
    
    # Build all prompts and messages upfront
    rows_list = [row for _, row in df.iterrows()]
    row_ids = [row.get(id_column, f"row_{i}") for i, row in enumerate(rows_list)]
    all_batch_messages = _build_batch_messages(df, prompt_builder)
    
    # Assign models to each row
    if model is None:
        # Use model_column from dataframe
        if model_column not in df.columns:
            raise ValueError(f"model=None requires a '{model_column}' column in the dataframe")
        models = df[model_column].tolist()
    else:
        if random_seed is None:
            raise ValueError("random_seed is required when model is not None")
        models = _assign_model(model, len(df), random_seed)
    
    # Group indices by model
    model_to_indices: Dict[str, List[int]] = {}
    for idx, m in enumerate(models):
        if m not in model_to_indices:
            model_to_indices[m] = []
        model_to_indices[m].append(idx)
    
    # Track all responses (indexed by original row index)
    all_responses = [None] * len(rows_list)
    
    # Process each model group separately - collect responses only
    pbar = tqdm(total=len(df), desc="Processing items") if show_progress else None
    
    for model_name, indices in model_to_indices.items():
        logger.info(f"Processing {len(indices)} items with model: {model_name}")
        
        # Get messages for this model group
        model_messages = [all_batch_messages[i] for i in indices]
        
        # Call batch completion with retry for this model
        responses = litellm_batch_completion_with_retry(
            model=model_name,
            messages=model_messages,
            num_retries=num_retries,
            logger=logger,
            **kwargs
        )
        
        # Store responses at their original indices
        for local_idx, response in enumerate(responses):
            original_idx = indices[local_idx]
            all_responses[original_idx] = response
        
        # Update progress bar by number of items processed
        if pbar:
            pbar.update(len(indices))
    
    if pbar:
        pbar.close()

    # Store response objects in a dictionary by row_ids (include original model, prompt for surgery kit and logging)
    responses_dict = {
        row_ids[idx]: {
            'original_completion': response, 
            'model': models[idx],
            'messages': all_batch_messages[idx]
        } 
        for idx, response in enumerate(all_responses)
    }

    if surgery_kit and surgery_kit_tests:
        logger.info(f"Running surgery kit on {len(responses_dict)} items...")
        for row_id, response in responses_dict.items():
            # Skip if original_completion is None (failed batch item)
            if response['original_completion'] is None:
                logger.warning(f"Skipping surgery kit for {id_column}={row_id}: no original completion")
                continue
            
            # Build original_messages: the messages sent + the assistant's response
            original_messages = list(response['messages'])  # Make a copy
            original_output = response['original_completion']['choices'][0]['message']['content']
            original_messages.append({'role': 'assistant', 'content': original_output})
            
            refined_output, refine_round, refinement_completions = surgery_kit.run(
                instruction=prompt_builder(df.loc[df[id_column] == row_id].iloc[0]),
                original_output=original_output,
                unit_tests=surgery_kit_tests,
                refine_engine=response['model'],
                refine_engine_kwargs=surgery_kit_kwargs,
                logger=logger,
                original_messages=original_messages
            )   
            responses_dict[row_id]['refined_output'] = refined_output
            responses_dict[row_id]['refine_round'] = refine_round
            responses_dict[row_id]['refinement_completions'] = refinement_completions
    
    # Use provided timestamp or generate one for output files
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export all responses as a single pickle file if out directory is specified
    # Using pickle to preserve complete response objects including hidden attributes
    if out:
        os.makedirs(out, exist_ok=True)
        
        pickle_path = os.path.join(out, f'{timestamp}.pkl')
        try:
            with open(pickle_path, 'wb') as f:
                pickle.dump(responses_dict, f)
            logger.info(f"Exported all responses to {pickle_path}")
        except Exception as e:
            logger.error(f"Failed to export responses pickle: {e}")
    
    # Process all responses and extract results (after all models are done)
    # responses_dict contains original completions and refined outputs (if surgery kit was used)
    results = _process_responses_and_extract_results(
        responses=responses_dict,
        row_ids=row_ids,
        response_extractor=response_extractor,
        logger=logger,
        id_column=id_column
    )
    
    # Build results DataFrame with model column
    results_df = pd.DataFrame(results)
    if model:
        # Determine base column name
        base_col_name = f'model_{model_col_suffix}' if model_col_suffix else 'model'
        
        # Check if column already exists in df, if so increment suffix
        model_col_name = base_col_name
        suffix_num = 2
        while model_col_name in df.columns:
            model_col_name = f'{base_col_name}_{suffix_num}'
            suffix_num += 1
        
        results_df[model_col_name] = models
    results_merged = pd.concat([df.reset_index(drop=True), results_df], axis=1)
    
    # Export results JSONL if out directory is specified
    if out:
        if out_suffix:
            outpath = os.path.join(out, f'{timestamp}_{out_suffix}.jsonl')
        else:
            outpath = os.path.join(out, f'{timestamp}.jsonl')
        results_merged.to_json(outpath, orient='records', lines=True, force_ascii=False)
        logger.info(f"Exported results JSONL to {outpath}")
    
    # Return verbose dict or just results
    if return_verbose:
        # Build messages list (in same order as row_ids)
        messages_list = []
        for row_id in row_ids:
            if row_id in responses_dict:
                response_data = responses_dict[row_id]
                # Build complete conversation: input messages + assistant response
                msgs = list(response_data['messages'])  # Copy original messages
                assistant_content = response_data['original_completion']['choices'][0]['message']['content']
                msgs.append({'role': 'assistant', 'content': assistant_content})
                messages_list.append(msgs)
            else:
                messages_list.append(None)
        
        return {
            'results': results,
            'models': models,
            'messages': messages_list,
            'df': results_merged
        }
    
    return results


# ============================================================================
# LiteLLM Helpers
# ============================================================================

def litellm_chat_completion_with_retry(model,
                                       messages,
                                       num_retries: int = globals.NUM_RETRIES,
                                       logger: Optional[logging.Logger] = None,
                                       log_dict: Optional[Dict[str, Any]] = None,
                                       log_id: Optional[str] = None,
                                       **kwargs):
    """A wrapper function to call litellm.ChatCompletion.create with retry.
    Args:
        model: The model to use for the completion.
        messages: [{'role': 'system'/'user'/'assistant', 'content': '...'}, ...]
        logger: Optional logger for logging. If None, uses module-level logger.
        log_dict: Optional dictionary to accumulate log entries. If provided along with log_id,
                  the call details and response will be logged to this dictionary.
        log_id: Optional ID for this log entry (used as key in log_dict).
        **kwargs: max_tokens, temperature, top_p, ... (see LiteLLM API documentation for details)
    
    Returns:
        LiteLLM response object
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.debug(f"Calling litellm.completion with model={model}")
    
    try:
        response = litellm.completion(model=model, 
                                      messages=messages, 
                                      num_retries=num_retries,
                                      **kwargs)
        logger.debug(f"Received response from {model}: {response.usage if hasattr(response, 'usage') else 'no usage info'}")
        
        # Log the call if log_dict and log_id are provided
        if log_dict is not None and log_id is not None:
            log_dict[log_id] = {
                'model': model,
                'messages': messages,
                'original_completion': response,
                'kwargs': kwargs
            }
            logger.debug(f"Logged LLM call with id={log_id}")
        
        return response
    except Exception as e:
        logger.error(f"litellm.completion failed for model={model}: {e}")
        raise


def litellm_batch_completion_with_retry(model,
                                        messages: List[List[Dict[str, str]]],
                                        num_retries: int = globals.NUM_RETRIES,
                                        max_connections: int = globals.MAX_CONNECTIONS,
                                        logger: Optional[logging.Logger] = None,
                                        **kwargs):
    """
    A wrapper function for litellm.batch_completion with retry support.
    
    First attempts batch_completion for all messages. For any failures,
    retries those individually using litellm.completion with its built-in num_retries.
    
    Args:
        model: The model to use for completion.
        messages: List of message lists, where each inner list is 
                  [{'role': 'system'/'user'/'assistant', 'content': '...'}, ...]
        num_retries: Maximum number of retry attempts for failed items (default: 5)
        logger: Optional logger for logging retry attempts
        **kwargs: Additional arguments passed to litellm (max_tokens, temperature, etc.)
    
    Returns:
        List of responses in the same order as input messages.
        Failed items after all retries will be None.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    n_messages = len(messages)
    results = [None] * n_messages
    
    # Track indices that still need processing
    pending_indices = list(range(n_messages))
    
    # First attempt: try batch_completion for all messages
    try:
        pending_messages = [messages[i] for i in pending_indices]
        responses = litellm.batch_completion(
            model=model,
            messages=pending_messages,
            max_workers=max_connections,
            **kwargs
        )
        
        # Process batch responses - identify successes and failures
        failed_indices = []
        for i, response in enumerate(responses):
            original_idx = pending_indices[i]
            
            # Check if response is an exception or error
            if isinstance(response, Exception):
                logger.warning(f"Batch item {original_idx} failed with exception: {response}")
                failed_indices.append(original_idx)
            elif isinstance(response, dict) and 'error' in response:
                logger.warning(f"Batch item {original_idx} failed with error: {response.get('error')}")
                failed_indices.append(original_idx)
            else:
                # Success - store the result
                results[original_idx] = response
        
        pending_indices = failed_indices
        
    except Exception as e:
        # Entire batch failed - all items need individual retry
        logger.error(f"Batch completion failed entirely: {e}")
        # pending_indices remains as all indices
    
    # Retry failed items individually using litellm.completion with built-in retries
    if pending_indices:
        logger.info(f"Retrying {len(pending_indices)} failed items individually...")
        
        for original_idx in pending_indices:
            msg = messages[original_idx]
            try:
                response = litellm.completion(
                    model=model,
                    messages=msg,
                    num_retries=num_retries,
                    **kwargs
                )
                results[original_idx] = response
                logger.debug(f"Item {original_idx} succeeded")
            except Exception as e:
                logger.error(f"Item {original_idx} failed after {num_retries} retries: {e}")
                # results[original_idx] remains None
    
    return results