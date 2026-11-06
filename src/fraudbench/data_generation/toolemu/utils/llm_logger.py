"""LLM Response Logger - Captures all LLM calls during trajectory simulation.

This module provides callback handlers that log all LLM requests and responses,
similar to how apply_prompt_to_df() exports logs as ChatResponse objects.
"""

import json
import pickle
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


@dataclass
class LLMCallLog:
    """Single LLM call log entry."""
    timestamp: str
    role: str  # 'agent', 'simulator', 'critiquer'
    model: str
    input_messages: Any  # Can be list of messages or list of prompt strings
    output_content: str
    token_usage: Optional[Dict[str, int]] = None
    case_idx: Optional[int] = None
    step_idx: Optional[int] = None
    finish_reason: Optional[str] = None


class LLMResponseLogger(BaseCallbackHandler):
    """Callback handler that logs all LLM requests and responses.
    
    Attach this to an LLM to capture all calls made through it.
    
    Example:
        logger = LLMResponseLogger(role="agent")
        llm = ChatLiteLLMWrapper(model="gpt-4o", callbacks=[logger])
        # ... use llm ...
        logs = logger.get_logs()
    """
    
    def __init__(self, role: str = "unknown"):
        super().__init__()
        self.role = role
        self._logs: List[LLMCallLog] = []
        self._lock = threading.Lock()
        self._current_input: Any = None
        self._case_idx: Optional[int] = None
        self._step_idx: int = 0
    
    def set_context(self, case_idx: int, step_idx: int = 0):
        """Set current case/step context for logging.
        
        Args:
            case_idx: Index of the current case being processed
            step_idx: Current step within the trajectory (default 0)
        """
        self._case_idx = case_idx
        self._step_idx = step_idx
    
    def increment_step(self):
        """Increment the step counter."""
        self._step_idx += 1
    
    def reset_step(self):
        """Reset step counter to 0."""
        self._step_idx = 0
    
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> None:
        """Capture input prompts for text-based LLMs."""
        self._current_input = prompts
    
    def on_chat_model_start(
        self, 
        serialized: Dict[str, Any], 
        messages: List[List[BaseMessage]], 
        **kwargs: Any
    ) -> None:
        """Capture input messages for chat models."""
        # Convert BaseMessage objects to serializable dicts
        self._current_input = [
            [{"role": m.type, "content": m.content} for m in msg_list]
            for msg_list in messages
        ]
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Capture output and create log entry."""
        with self._lock:
            # Extract model name from llm_output if available
            model_name = "unknown"
            token_usage = None
            if response.llm_output:
                model_name = response.llm_output.get("model_name", "unknown")
                raw_token_usage = response.llm_output.get("token_usage")
                # Convert token_usage to a serializable dict
                if raw_token_usage is not None:
                    if isinstance(raw_token_usage, dict):
                        token_usage = raw_token_usage
                    elif hasattr(raw_token_usage, '__dict__'):
                        # Handle Usage objects from LiteLLM/OpenAI
                        token_usage = dict(raw_token_usage.__dict__)
                    elif hasattr(raw_token_usage, 'model_dump'):
                        # Handle Pydantic models
                        token_usage = raw_token_usage.model_dump()
                    else:
                        # Fallback: try to extract common attributes
                        token_usage = {
                            "prompt_tokens": getattr(raw_token_usage, 'prompt_tokens', 0),
                            "completion_tokens": getattr(raw_token_usage, 'completion_tokens', 0),
                            "total_tokens": getattr(raw_token_usage, 'total_tokens', 0),
                        }
            
            # Log each generation
            for gen_list in response.generations:
                for gen in gen_list:
                    # Extract finish reason if available
                    finish_reason = None
                    if hasattr(gen, 'generation_info') and gen.generation_info:
                        finish_reason = gen.generation_info.get("finish_reason")
                    
                    # Get output content
                    if hasattr(gen, 'text'):
                        output_content = gen.text
                    elif hasattr(gen, 'message') and hasattr(gen.message, 'content'):
                        output_content = gen.message.content
                    else:
                        output_content = str(gen)
                    
                    log = LLMCallLog(
                        timestamp=datetime.now().isoformat(),
                        role=self.role,
                        model=model_name,
                        input_messages=self._current_input,
                        output_content=output_content,
                        token_usage=token_usage,
                        case_idx=self._case_idx,
                        step_idx=self._step_idx,
                        finish_reason=finish_reason,
                    )
                    self._logs.append(log)
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Log errors as well."""
        with self._lock:
            log = LLMCallLog(
                timestamp=datetime.now().isoformat(),
                role=self.role,
                model="error",
                input_messages=self._current_input,
                output_content=f"ERROR: {str(error)}",
                token_usage=None,
                case_idx=self._case_idx,
                step_idx=self._step_idx,
                finish_reason="error",
            )
            self._logs.append(log)
    
    def get_logs(self) -> List[LLMCallLog]:
        """Get all logged calls.
        
        Returns:
            Copy of the logs list
        """
        with self._lock:
            return self._logs.copy()
    
    def clear(self):
        """Clear all logs."""
        with self._lock:
            self._logs.clear()
            self._step_idx = 0


class TrajectoryLLMLogger:
    """Container for all role-specific loggers in trajectory simulation.
    
    This class manages separate loggers for agent, simulator, and critiquer,
    and provides methods to export combined logs.
    
    Example:
        trajectory_logger = TrajectoryLLMLogger()
        
        # Attach loggers to LLMs
        agent_llm = load_litellm_with_args(args, callbacks=[trajectory_logger.agent])
        simulator_llm = load_litellm_with_args(args, callbacks=[trajectory_logger.simulator])
        
        # After simulation
        trajectory_logger.export_pickle("logs/llm_calls.pkl")
    """
    
    def __init__(self):
        self.agent = LLMResponseLogger(role="agent")
        self.simulator = LLMResponseLogger(role="simulator")
        self.critiquer = LLMResponseLogger(role="critiquer")
    
    def set_context(self, case_idx: int, step_idx: int = 0):
        """Set context for all loggers.
        
        Args:
            case_idx: Index of the current case being processed
            step_idx: Current step within the trajectory
        """
        self.agent.set_context(case_idx, step_idx)
        self.simulator.set_context(case_idx, step_idx)
        self.critiquer.set_context(case_idx, step_idx)
    
    def increment_step(self):
        """Increment step counter for all loggers."""
        self.agent.increment_step()
        self.simulator.increment_step()
        self.critiquer.increment_step()
    
    def reset_step(self):
        """Reset step counter for all loggers."""
        self.agent.reset_step()
        self.simulator.reset_step()
        self.critiquer.reset_step()
    
    def get_all_logs(self) -> List[LLMCallLog]:
        """Get combined logs from all roles, sorted by timestamp.
        
        Returns:
            List of all LLMCallLog entries sorted chronologically
        """
        all_logs = (
            self.agent.get_logs() +
            self.simulator.get_logs() +
            self.critiquer.get_logs()
        )
        return sorted(all_logs, key=lambda x: x.timestamp)
    
    def get_logs_for_case(self, case_idx: int) -> List[LLMCallLog]:
        """Get logs for a specific case.
        
        Args:
            case_idx: The case index to filter by
            
        Returns:
            List of LLMCallLog entries for the specified case
        """
        return [log for log in self.get_all_logs() if log.case_idx == case_idx]
    
    def export_pickle(self, path: str) -> None:
        """Export all logs to pickle file.
        
        Args:
            path: File path to save the pickle file
        """
        logs = self.get_all_logs()
        with open(path, 'wb') as f:
            pickle.dump([asdict(log) for log in logs], f)
    
    def export_jsonl(self, path: str) -> None:
        """Export all logs to JSONL file.
        
        Args:
            path: File path to save the JSONL file
        """
        def json_serializer(obj):
            """Custom JSON serializer for non-serializable objects."""
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            elif hasattr(obj, 'model_dump'):
                return obj.model_dump()
            else:
                return str(obj)
        
        logs = self.get_all_logs()
        with open(path, 'w', encoding='utf-8') as f:
            for log in logs:
                f.write(json.dumps(asdict(log), ensure_ascii=False, default=json_serializer) + '\n')
    
    def clear_all(self):
        """Clear all loggers."""
        self.agent.clear()
        self.simulator.clear()
        self.critiquer.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of logged calls.
        
        Returns:
            Dictionary with counts and token usage by role
        """
        summary = {
            "total_calls": 0,
            "by_role": {},
            "total_tokens": {"prompt": 0, "completion": 0, "total": 0}
        }
        
        for role, logger in [("agent", self.agent), ("simulator", self.simulator), ("critiquer", self.critiquer)]:
            logs = logger.get_logs()
            role_tokens = {"prompt": 0, "completion": 0, "total": 0}
            
            for log in logs:
                if log.token_usage:
                    role_tokens["prompt"] += log.token_usage.get("prompt_tokens", 0)
                    role_tokens["completion"] += log.token_usage.get("completion_tokens", 0)
                    role_tokens["total"] += log.token_usage.get("total_tokens", 0)
            
            summary["by_role"][role] = {
                "call_count": len(logs),
                "tokens": role_tokens
            }
            summary["total_calls"] += len(logs)
            summary["total_tokens"]["prompt"] += role_tokens["prompt"]
            summary["total_tokens"]["completion"] += role_tokens["completion"]
            summary["total_tokens"]["total"] += role_tokens["total"]
        
        return summary


# Global singleton instance for convenience
trajectory_llm_logger = TrajectoryLLMLogger()

