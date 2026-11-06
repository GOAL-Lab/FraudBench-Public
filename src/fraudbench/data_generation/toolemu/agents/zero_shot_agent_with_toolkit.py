"""Zero-shot agent with toolkit."""
import logging
import re

from langchain_classic.agents.agent import Agent

logger = logging.getLogger(__name__)
from langchain_classic.agents.mrkl.base import ZeroShotAgent
from langchain_classic.chains import LLMChain
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.agents import (
    AgentAction,
    AgentFinish,
)
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool

from procoder.functional import (
    add_refnames,
    format_multiple_prompts,
)
from fraudbench.data_generation.toolemu.prompts.agent import *
from fraudbench.data_generation.toolemu.tools.tool_interface import BaseToolkit
from fraudbench.data_generation.toolemu.utils import get_first_json_object_str
from fraudbench.data_generation.toolemu.utils.my_typing import *

FINAL_ANSWER_ACTION = "Final Answer:"
AGENT_TYPES = ["fraud_enhanced", "naive"]


class ZeroShotAgentWithToolkit(ZeroShotAgent):
    @staticmethod
    def get_all_tools(toolkits: Sequence[BaseToolkit]) -> List[BaseTool]:
        """Return all tools available to the agent."""
        all_tools = []
        for toolkit in toolkits:
            all_tools += toolkit.tools
        return all_tools

    @classmethod
    def create_prompt(
            cls,
            toolkits: Sequence[BaseToolkit],
            prompt_type: Optional[str] = "fraud_enhanced",
            input_variables: Optional[List[str]] = None,
            use_chat_format: Optional[bool] = False,
    ) -> PromptTemplate:
        """Create prompt in the style of the zero shot agent."""
        # [Prepare agent prompt]
        toolkit_strings = "\n".join(
            [toolkit.create_description("medium") for toolkit in toolkits]
        )
        tool_names = ", ".join([tool.name for tool in cls.get_all_tools(toolkits)])

        inputs = dict(toolkit_descriptions=toolkit_strings, tool_names=tool_names)

        add_refnames(AGENT_DUMMY_VARS, inputs, include_brackets=False)

        system_info = eval(f"AGENT_{prompt_type.upper()}_SYSTEM_INFO")
        prompt_instruction = eval(f"AGENT_{prompt_type.upper()}_PROMPT")

        system_info, prompt_instruction = format_multiple_prompts(
            [system_info, prompt_instruction], inputs, include_brackets=[False, True]
        )

        if use_chat_format:
            agent_system_message = SystemMessage(content=system_info)
            agent_instruction_message = HumanMessagePromptTemplate.from_template(
                template=prompt_instruction
            )

            messages = [
                agent_system_message,
                agent_instruction_message,
            ]
            return ChatPromptTemplate.from_messages(messages=messages)
        else:
            template = "\n\n".join([system_info, prompt_instruction])

            if input_variables is None:
                input_variables = ["input", "agent_scratchpad"]
            return PromptTemplate(template=template, input_variables=input_variables)

    @classmethod
    def from_llm_and_toolkits(
            cls,
            llm: BaseLanguageModel,
            toolkits: Sequence[BaseToolkit],
            agent_type: Optional[str] = "fraud_enhanced",
            callback_manager: Optional[BaseCallbackManager] = None,
            use_chat_format: Optional[bool] = False,
            input_variables: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        tools = cls.get_all_tools(toolkits)
        cls._validate_tools(tools)

        assert agent_type in AGENT_TYPES, f"agent_type must be one of {AGENT_TYPES}"

        # Use the standard prompt for all model types
        # Modern Langchain handles message formatting consistently across providers
        prompt_type = agent_type

        prompt = cls.create_prompt(
            toolkits,
            prompt_type=prompt_type,
            input_variables=input_variables,
            use_chat_format=use_chat_format,
        )
        
        # Create LLMChain for langchain-classic compatibility
        # ZeroShotAgent in langchain-classic requires an llm_chain, not separate llm and prompt
        llm_chain = LLMChain(llm=llm, prompt=prompt, callback_manager=callback_manager)
        
        tool_names = [tool.name for tool in tools]
        return cls(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)

    def _fix_text(self, text: str):
        text = text.lstrip()
        if text.startswith(self.llm_prefix):
            text = text[len(self.llm_prefix):]

        return text.rstrip() + "\n"

    def _extract_tool_and_input(self, text: str) -> Optional[Tuple[str, str]]:
        try:
            result = self._get_action_and_input(text)
        except ValueError:
            # result = "None", "{}"
            result = None  # return None, so that the agent can continue its generation
        return result

    ############# Adapted from Langchain (0.0.141) for compatibility #############

    @property
    def finish_tool_name(self) -> str:
        """Return the name of the finish tool."""
        return "Final Answer"

    def _get_action_and_input(self, llm_output: str) -> Tuple[str, str]:
        """Parse out the action and input from the LLM output.

        Note: if you're specifying a custom prompt for the ZeroShotAgent,
        you will need to ensure that it meets the following Regex requirements.
        The string starting with "Action:" and the following string starting
        with "Action Input:" should be separated by a newline.
        """
        if FINAL_ANSWER_ACTION in llm_output:
            return (
                self.finish_tool_name,
                llm_output.split(FINAL_ANSWER_ACTION)[-1].strip(),
            )
        # \s matches against tab/newline/whitespace
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # disable checking to avoid exception, let the tool call validate the inputs
        action_input = get_first_json_object_str(
            action_input, enable_check=False, strict=False
        )
        # set strict=False to allow single quote, comments, escape characters, etc
        return action, action_input.strip(" ").strip('"')

    def _get_next_action(self, full_inputs: Dict[str, str]) -> AgentAction:
        # Use direct invocation with llm_chain (langchain-classic compatibility)
        # Access prompt and llm through self.llm_chain
        model_name = getattr(self.llm_chain.llm, 'model', 'unknown')
        logger.info(f"[AGENT] Invoking LLM ({model_name}) to decide next action...")
        
        prompt_value = self.llm_chain.prompt.format_prompt(**full_inputs)
        if hasattr(prompt_value, 'to_messages'):
            # Chat model - use messages
            messages = prompt_value.to_messages()
            if not messages:
                logger.error(f"[AGENT] ERROR: Empty messages list! Prompt type: {type(self.llm_chain.prompt)}")
                logger.error(f"[AGENT] Full inputs keys: {list(full_inputs.keys())}")
                raise ValueError("Cannot invoke LLM with empty messages list")
            logger.debug(f"[AGENT] Invoking with {len(messages)} messages, types: {[type(m).__name__ for m in messages]}")
            response = self.llm_chain.llm.invoke(messages)
            full_output = response.content if hasattr(response, 'content') else str(response)
        else:
            # Text model - use string
            prompt_text = prompt_value.to_string()
            response = self.llm_chain.llm.invoke(prompt_text)
            full_output = response.content if hasattr(response, 'content') else str(response)
        
        logger.info(f"[AGENT] LLM response received (length: {len(full_output)} chars)")
        
        full_output = self._fix_text(full_output)
        parsed_output = self._extract_tool_and_input(full_output)
        retry_count = 0
        while parsed_output is None:
            retry_count += 1
            logger.info(f"[AGENT] Could not parse output, retrying (attempt {retry_count})...")
            full_output = self._fix_text(full_output)
            full_inputs["agent_scratchpad"] += full_output
            
            # Invoke again with updated inputs
            prompt_value = self.llm_chain.prompt.format_prompt(**full_inputs)
            if hasattr(prompt_value, 'to_messages'):
                messages = prompt_value.to_messages()
                response = self.llm_chain.llm.invoke(messages)
                output = response.content if hasattr(response, 'content') else str(response)
            else:
                prompt_text = prompt_value.to_string()
                response = self.llm_chain.llm.invoke(prompt_text)
                output = response.content if hasattr(response, 'content') else str(response)
            
            full_output += output
            parsed_output = self._extract_tool_and_input(full_output)
        
        logger.info(f"[AGENT] Decided action: {parsed_output[0]}")
        return AgentAction(
            tool=parsed_output[0], tool_input=parsed_output[1], log=full_output
        )

    async def _aget_next_action(self, full_inputs: Dict[str, str]) -> AgentAction:
        # Use direct invocation with llm_chain (langchain-classic compatibility)
        # Access prompt and llm through self.llm_chain
        prompt_value = self.llm_chain.prompt.format_prompt(**full_inputs)
        if hasattr(prompt_value, 'to_messages'):
            # Chat model - use messages
            messages = prompt_value.to_messages()
            response = await self.llm_chain.llm.ainvoke(messages)
            full_output = response.content if hasattr(response, 'content') else str(response)
        else:
            # Text model - use string
            prompt_text = prompt_value.to_string()
            response = await self.llm_chain.llm.ainvoke(prompt_text)
            full_output = response.content if hasattr(response, 'content') else str(response)
        
        parsed_output = self._extract_tool_and_input(full_output)
        while parsed_output is None:
            full_output = self._fix_text(full_output)
            full_inputs["agent_scratchpad"] += full_output
            
            # Invoke again with updated inputs
            prompt_value = self.llm_chain.prompt.format_prompt(**full_inputs)
            if hasattr(prompt_value, 'to_messages'):
                messages = prompt_value.to_messages()
                response = await self.llm_chain.llm.ainvoke(messages)
                output = response.content if hasattr(response, 'content') else str(response)
            else:
                prompt_text = prompt_value.to_string()
                response = await self.llm_chain.llm.ainvoke(prompt_text)
                output = response.content if hasattr(response, 'content') else str(response)
            
            full_output += output
            parsed_output = self._extract_tool_and_input(full_output)
        return AgentAction(
            tool=parsed_output[0], tool_input=parsed_output[1], log=full_output
        )

    def plan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        action = self._get_next_action(full_inputs)
        if action.tool == self.finish_tool_name:
            return AgentFinish({"output": action.tool_input}, action.log)
        return action

    async def aplan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        action = await self._aget_next_action(full_inputs)
        if action.tool == self.finish_tool_name:
            return AgentFinish({"output": action.tool_input}, action.log)
        return action
