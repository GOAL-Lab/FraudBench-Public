import logging
import threading

from langchain_litellm import ChatLiteLLM
from langchain_core.language_models import BaseLanguageModel
from langchain_community.adapters.openai import convert_dict_to_message
from langchain_core.outputs import ChatGeneration, ChatResult

from .my_typing import *

def llm_register_args(parser, prefix=None, shortprefix=None, defaults={}):
    model_name = defaults.get("model_name", "gpt-4")
    temperature = defaults.get("temperature", 0.0)
    max_tokens = defaults.get("max_tokens", None)
    default_retries = 8 if model_name.startswith("claude") else 12
    max_retries = defaults.get("max_retries", default_retries)
    request_timeout = defaults.get("request_timeout", 300)
    if prefix is None:
        prefix = ""
        shortprefix = ""
    else:
        prefix += "-"
        shortprefix = shortprefix or prefix[0]
    parser.add_argument(
        f"--{prefix}model-name",
        f"-{shortprefix}m",
        type=str,
        default=model_name,
        help=f"Model name.",
    )
    parser.add_argument(
        f"--{prefix}temperature",
        f"-{shortprefix}t",
        type=float,
        default=temperature,
        help="Temperature for sampling",
    )
    parser.add_argument(
        f"--{prefix}max-tokens",
        f"-{shortprefix}mt",
        type=int,
        default=max_tokens,
        help="Max tokens for sampling",
    )
    parser.add_argument(
        f"--{prefix}max-retries",
        f"-{shortprefix}mr",
        type=int,
        default=max_retries,
        help="Max retries for each request",
    )
    parser.add_argument(
        f"--{prefix}request-timeout",
        f"-{shortprefix}rt",
        type=int,
        default=request_timeout,
        help="Timeout for each request",
    )
    reasoning_effort = defaults.get("reasoning_effort", "medium")
    parser.add_argument(
        f"--{prefix}reasoning-effort",
        f"-{shortprefix}re",
        type=str,
        default=reasoning_effort,
        choices=[None, 'low', 'medium', 'high'],
        help="Reasoning effort for GPT-5/Gemini models (low/medium/high). Claude does not support this.",
    )


def get_model_name(name, fixed_version=True, version=None):
    """
    Name resolver for model names. Simply returns the name as-is.
    The fixed_version and version params are kept for backward compatibility but ignored.
    """
    return name


class APIUsageTracker(object):
    """A singleton class to track API usage."""

    _instance = None
    _token_usage_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(APIUsageTracker, cls).__new__(cls)
            cls._instance.token_usage = {}
        return cls._instance

    def get_token_usage(self):
        return self.token_usage

    def reset_token_usage(self):
        self.token_usage = {}

    def increment_token_usage(self, model, prompt_tokens, completion_tokens, total_tokens: int = None):
        if not total_tokens:
            total_tokens = prompt_tokens + completion_tokens

        with self._token_usage_lock:
            if model in self.token_usage:
                self.token_usage[model]['prompt_tokens'] += prompt_tokens
                self.token_usage[model]['completion_tokens'] += completion_tokens
                self.token_usage[model]['total_tokens'] += total_tokens
            else:
                self.token_usage[model] = {'prompt_tokens': prompt_tokens, 
                                           'completion_tokens': completion_tokens,
                                           'total_tokens': total_tokens}


api_usage_tracker = APIUsageTracker()

logger = logging.getLogger(__name__)

def print_api_usage(func):
    """A decorator to log API usage even if an exception is raised."""

    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.error(f'{func.__name__} raised an exception: {e}')
            raise
        finally:
            logger.info("API Usage Information:")
            logger.info(api_usage_tracker.get_token_usage())

    return wrapper

def parse_llm_response(
        res: ChatResult, index: int = 0, one_generation_only: bool = True
) -> str:
    res = res.generations[0]
    if one_generation_only:
        assert len(res) == 1, res
    res = res[index]
    # if the decoding stops due to length, the response text is likely truncated so we discard it
    if res.generation_info.get("finish_reason", None) == "length":
        raise ValueError(f"Discard a response due to length: {res.text}")
    return res.text.strip()

class ChatLiteLLMWrapper(ChatLiteLLM):
    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        # Format list of messages
        generations = []
        
        # Check for empty or missing choices - this would cause IndexError downstream
        choices = response.get("choices", [])
        if not choices:
            error_msg = f"LLM API returned empty choices. Response: {response}"
            logger.error(error_msg)
            # Check if there's an error in the response
            if "error" in response:
                error_msg = f"LLM API error: {response['error']}"
            raise ValueError(error_msg)
        
        for res in choices:
            message = convert_dict_to_message(res["message"])
            # Use construct() to bypass Pydantic validation issues with langchain 0.0.277
            # Should still work with Langchain 0.3.27
            gen = ChatGeneration.construct(
                message=message,
                text=message.content,
                generation_info=dict(
                    finish_reason=res.get("finish_reason"),
                ),
            )
            generations.append(gen)

        # Final safety check - ensure we never return empty generations
        # This would cause IndexError: list index out of range in langchain_core
        if not generations:
            error_msg = f"No generations created from {len(choices)} choices. Response: {response}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Track token usage
        usage = response.get("usage", {})
        llm_output = {"token_usage": usage, "model_name": self.model}
        api_usage_tracker.increment_token_usage(
            model=llm_output['model_name'],
            prompt_tokens=usage.get('prompt_tokens', 0),
            completion_tokens=usage.get('completion_tokens', 0),
            total_tokens=usage.get('total_tokens', 0)
        )
        return ChatResult(generations=generations, llm_output=llm_output)


def load_litellm_with_args(args, prefix=None, model='gpt-4o', fixed_version=True, callbacks=None, **kwargs):
    """Load a LiteLLM chat model with arguments from argparse or dict.
    
    Args:
        args: Either an argparse.Namespace or dict containing model configuration
        prefix: Optional prefix for argument names (e.g., 'agent', 'simulator')
        model: Default model name if not specified in args
        fixed_version: Whether to use fixed model versions
        callbacks: Optional list of callback handlers (e.g., LLMResponseLogger) to attach to the LLM
        **kwargs: Additional keyword arguments to pass to ChatLiteLLMWrapper
        
    Returns:
        ChatLiteLLMWrapper instance configured with the specified parameters
    """
    # Convert argparse Namespace to dict if needed
    if hasattr(args, '__dict__'):
        args_dict = vars(args)
    else:
        args_dict = args
    
    # Build the key names with prefix
    if prefix:
        model_key = f'{prefix}_model_name'
        temp_key = f'{prefix}_temperature'
        max_tokens_key = f'{prefix}_max_tokens'
        max_retries_key = f'{prefix}_max_retries'
        timeout_key = f'{prefix}_request_timeout'
        reasoning_effort_key = f'{prefix}_reasoning_effort'
    else:
        model_key = 'model_name'
        temp_key = 'temperature'
        max_tokens_key = 'max_tokens'
        max_retries_key = 'max_retries'
        timeout_key = 'request_timeout'
        reasoning_effort_key = 'reasoning_effort'
    
    # Extract parameters from args
    model_name = args_dict.get(model_key, model)
    temperature = args_dict.get(temp_key, 0.0)
    max_tokens = args_dict.get(max_tokens_key, None)
    max_retries = args_dict.get(max_retries_key, 12)
    request_timeout = args_dict.get(timeout_key, 300)
    reasoning_effort = args_dict.get(reasoning_effort_key, 'medium')
    
    # Resolve model name
    resolved_model_name = get_model_name(model_name, fixed_version=fixed_version)
    is_gpt5 = 'gpt-5' in resolved_model_name.lower()
    is_gemini = 'gemini' in resolved_model_name.lower()
    supports_reasoning_effort = is_gpt5 or is_gemini
    
    # Build kwargs for ChatLiteLLMWrapper
    llm_kwargs = {
        'max_retries': max_retries,
        'request_timeout': request_timeout,
        'drop_params': True,  # Automatically drop unsupported params for different models
    }
    
    # GPT-5 models don't support temperature=0.0, only temperature=1.0
    # For deterministic behavior, omit temperature and set reasoning_effort=None
    if is_gpt5:
        if temperature == 1.0:
            llm_kwargs['temperature'] = temperature
    else:
        llm_kwargs['temperature'] = temperature
    
    # Set reasoning_effort for GPT-5 and Gemini models (Claude does not support this)
    if supports_reasoning_effort:
        llm_kwargs['reasoning_effort'] = reasoning_effort  # None, 'low', 'medium', or 'high'
    if max_tokens is not None:
        llm_kwargs['max_tokens'] = max_tokens
    
    # Attach callbacks if provided
    if callbacks is not None:
        llm_kwargs['callbacks'] = callbacks
    
    llm_kwargs.update(kwargs)
    
    return ChatLiteLLMWrapper(
        model=resolved_model_name,
        **llm_kwargs,
    )

def get_model_category(llm: BaseLanguageModel):
    """Extract the model family from a model name.
    
    Handles formats like:
    - openai/gpt-5-2025-08-07 -> gpt
    - gemini/gemini-2.5-pro -> gemini  
    - anthropic/claude-opus-4-5 -> claude
    - gpt-4 -> gpt (legacy format)
    """
    model = llm.model.lower()
    
    # Check for known model families in the model name
    families = ["gpt", "claude", "gemini", "llama", "mistral", "deepseek"]
    for family in families:
        if family in model:
            return family
    
    raise ValueError(f"Unknown model family for model: {llm.model}")
