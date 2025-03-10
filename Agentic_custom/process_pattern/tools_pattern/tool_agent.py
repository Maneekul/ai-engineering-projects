import json
from colorama import Fore
import json
import sys
import os


from process_pattern.tools_pattern.tool import Tool
from process_pattern.tools_pattern.tool import validate_arguments
from process_pattern.utils.completions import ChatHistory
from process_pattern.utils.completions import completions_create
from process_pattern.utils.completions import update_chat_history
from process_pattern.utils.extraction import extract_tag_content
from process_pattern.reflection_pattern.reflection_agent import ai_llm

from process_pattern.utils.completions import build_prompt_structure


TOOL_SYSTEM_PROMPT = """
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug
into functions. Pay special attention to the properties 'types'. You should use those types as in a Python dict.
For each function call return a json object with function name and arguments within <tool_call></tool_call>
XML tags as follows:

<tool_call>
{"name": <function-name>,"arguments": <args-dict>,  "id": <monotonically-increasing-id>}
</tool_call>

Here are the available tools:

<tools>
%s
</tools>
"""

class ToolAgent:
    def __init__(self, tools: Tool | list[Tool]):
        self.tools = tools if isinstance(tools, list) else [tools]
        self.tools_dict = {tool.name: tool for tool in self.tools}

    def add_tool_signatures(self) -> str:
        return "".join([tool.fn_signature for tool in self.tools])

    def process_tool_calls(self, tool_calls_content: list) -> dict:
        observations = {}
        for tool_call_str in tool_calls_content:
            tool_call = json.loads(tool_call_str)
            tool_name = tool_call["name"]
            tool = self.tools_dict[tool_name]

            print(Fore.GREEN + f"\nUsing Tool: {tool_name}")
            validated_tool_call = validate_arguments(tool_call, json.loads(tool.fn_signature))
            print(Fore.GREEN + f"\nTool call dict: \n{validated_tool_call}")

            result = tool.run(**validated_tool_call["arguments"])
            print(Fore.GREEN + f"\nTool result: \n{result}")
            observations[validated_tool_call["id"]] = result

        return observations

    def run(self, user_msg: str) -> str:
        user_prompt = build_prompt_structure(prompt=user_msg, role="user")
        tool_chat_history = ChatHistory([
            build_prompt_structure(
                prompt=TOOL_SYSTEM_PROMPT % self.add_tool_signatures(), role="system"
            ),
            user_prompt,
        ])
        agent_chat_history = ChatHistory([user_prompt])

        tool_call_response = ai_llm(str(tool_chat_history))
        tool_calls = extract_tag_content(tool_call_response, "tool_call")

        if tool_calls.found:
            observations = self.process_tool_calls(tool_calls.content)
            update_chat_history(agent_chat_history, f'"Observation: {observations}"', "user")

        return ai_llm(str(agent_chat_history))
