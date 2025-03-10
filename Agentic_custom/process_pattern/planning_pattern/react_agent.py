import json

from colorama import Fore


from process_pattern.tools_pattern.tool import Tool
from process_pattern.tools_pattern.tool import validate_arguments
from process_pattern.utils.completions import ChatHistory
# from process_pattern.utils.completions import completions_create
from process_pattern.utils.completions import update_chat_history
from process_pattern.utils.extraction import extract_tag_content
from process_pattern.reflection_pattern.reflection_agent import ai_llm
from process_pattern.utils.completions import build_prompt_structure

BASE_SYSTEM_PROMPT = ""

REACT_SYSTEM_PROMPT = """
You operate by running a loop with the following steps: Thought, Action, Observation.
You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug
into functions. Pay special attention to the properties 'types'. You should use those types as in a Python dict.

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:

<tool_call>
{"name": <function-name>,"arguments": <args-dict>, "id": <monotonically-increasing-id>}
</tool_call>

Here are the available tools / actions:

<tools>
%s
</tools>

Example session:

<question>What's the current temperature in Madrid?</question>
<thought>I need to get the current weather in Madrid</thought>
<tool_call>{"name": "get_current_weather","arguments": {"location": "Madrid", "unit": "celsius"}, "id": 0}</tool_call>

You will be called again with this:

<observation>{0: {"temperature": 25, "unit": "celsius"}}</observation>

You then output:

<response>The current temperature in Madrid is 25 degrees Celsius</response>

Additional constraints:

- If the user asks you something unrelated to any of the tools above, answer freely enclosing your answer with <response></response> tags.
"""




class ReactAgent:
    def __init__(self, tools: Tool | list[Tool], model: str = "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4", system_prompt: str = BASE_SYSTEM_PROMPT) -> None:
        self.model = model
        self.system_prompt = system_prompt
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

    def run(self, user_msg: str, max_rounds: int = 10) -> str:
        user_prompt = build_prompt_structure(prompt=user_msg, role="user", tag="question")
        if self.tools:
            self.system_prompt += "\n" + REACT_SYSTEM_PROMPT % self.add_tool_signatures()

        chat_history = ChatHistory([
            build_prompt_structure(prompt=self.system_prompt, role="system"),
            user_prompt,
        ])

        for _ in range(max_rounds):
            completion = ai_llm(str(chat_history))
            response = extract_tag_content(completion, "response")
            if response.found:
                return response.content[0]

            thought = extract_tag_content(completion, "thought")
            tool_calls = extract_tag_content(completion, "tool_call")

            update_chat_history(chat_history, completion, "assistant")
            print(Fore.MAGENTA + f"\nThought: {thought.content[0]}")

            if tool_calls.found:
                observations = self.process_tool_calls(tool_calls.content)
                print(Fore.BLUE + f"\nObservations: {observations}")
                update_chat_history(chat_history, f"{observations}", "user")

        return ai_llm(str(chat_history))